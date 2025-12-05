#!/usr/bin/env python3
import rclpy
import numpy as np
from numpy.linalg import norm
from rclpy.node import Node
from mavros_msgs.msg import AttitudeTarget
from geometry_msgs.msg import PoseStamped, TwistStamped
from scipy.spatial.transform import Rotation as R
from rclpy.qos import QoSProfile, ReliabilityPolicy
from interfaces.msg import PhiEst
from .traj import TargetTraj
from rclpy.clock import Clock
import csv
import os
from pathlib import Path
from datetime import datetime

class TrajController(Node):
    def __init__(self, name):
        # 初始化
        super().__init__(name)
        self.get_logger().info("Node running: %s" % name)

        # 控制频率
        self.control_rate = 50.0
        self.traj = TargetTraj(FLAG=1)

        # 初始化自适应参数估计器
        self.current_hat = CurrentHat()

        # 初始化日志记录
        self.setup_flight_log()

        # 初始化时钟
        self.clock = Clock()

        # 初始化状态变量
        self.current_pa = None
        self.current_velo = None
        self.current_phi = np.zeros((3, 12))

        # 轨迹计时器
        self.traj_t = -1.0
        self.t_0 = self.clock.now()

        # 输出计数器（用于定期输出F_sp组成）
        self.output_counter = 0
        self.output_interval = 50  # 每50次控制循环输出一次（即每秒1次）

        # 订阅和发布
        qos_best_effort = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        qos_reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self.pa_sub_ = self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self.pa_cb, qos_best_effort)
        self.velo_sub_ = self.create_subscription(
            TwistStamped, '/mavros/local_position/velocity_local', self.velo_cb, qos_best_effort)
        self.phi_sub_ = self.create_subscription(
            PhiEst, '/control/phiest', self.phi_cb, qos_best_effort)

        self.controller_pub_ = self.create_publisher(
            AttitudeTarget, '/mavros/setpoint_raw/attitude', qos_reliable)

        self.controller_timer_ = self.create_timer(1 / self.control_rate, self.controller_cb)

        # 初始化参数
        self.declare_parameter('sliding_gain', [0.3, 0.3, 0.5])  # 滑模跟踪增益
        self.declare_parameter('tracking_gain', [3.0, 3.0, 5.0])  # 跟踪增益
        self.declare_parameter('traj_mode', False)  # 轨迹模式开关

        # 系统常量
        self.mass = 2.0  # 无人机质量(kg)。
        self.gravity = 9.8015  # 重力加速度(m/s^2)

    def setup_flight_log(self):
        """初始化飞行日志记录"""
        # 使用环境变量或当前工作目录来定位源代码目录
        # 获取ROS工作空间的src目录
        if 'ROS_WORKSPACE' in os.environ:
            workspace = Path(os.environ['ROS_WORKSPACE'])
            package_dir = workspace / 'src' / 'nl_controller'
        else:
            # 尝试从当前工作目录推断（假设在 px4_ws 目录下）
            cwd = Path.cwd()
            if 'px4_ws' in str(cwd):
                # 找到 px4_ws 路径
                parts = cwd.parts
                px4_ws_index = parts.index('px4_ws')
                workspace = Path(*parts[:px4_ws_index+1])
                package_dir = workspace / 'src' / 'nl_controller'
            else:
                # 默认使用当前工作目录
                package_dir = Path.cwd()

        # 创建log目录（相对路径）
        log_dir = package_dir / 'log'
        log_dir.mkdir(exist_ok=True)

        # 生成带时间戳的文件名（精确到分钟）
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
        log_filename = f'{timestamp}.csv'
        self.log_file_path = log_dir / log_filename

        # 创建CSV文件并写入表头
        self.log_file = open(self.log_file_path, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)

        # 写入表头
        header = [
            'timestamp',
            'x', 'y', 'z',
            'vx', 'vy', 'vz',
            'roll', 'pitch', 'yaw',
            'x_des', 'y_des', 'z_des',
            'position_error', 'velocity_error'
        ]
        self.csv_writer.writerow(header)
        self.log_file.flush()  # 立即写入磁盘

        self.get_logger().info(f"Flight log initialized: {self.log_file_path}")

    def pa_cb(self, msg):
        self.current_pa = msg

    def velo_cb(self, msg):
        self.current_velo = msg

    def phi_cb(self, msg):
        phi_array = np.vstack([np.array(msg.row1), np.array(msg.row2), np.array(msg.row3)])
        self.current_phi = phi_array

    def update_trajectory_time(self, traj_mode):
        prev_mode = self.traj_t != -1.0

        if traj_mode and not prev_mode:
            # 从非轨迹模式切换到轨迹模式
            self.t_0 = self.clock.now()
            self.traj_t = 0.0
            self.get_logger().info("Starting trajectory tracking")
            # 重置自适应参数以避免累积误差
            self.current_hat = CurrentHat()
        elif traj_mode:
            self.traj_t = (self.clock.now() - self.t_0).nanoseconds * 1e-9
        elif not traj_mode and prev_mode:
            # 从轨迹模式切换到非轨迹模式
            self.traj_t = -1.0
            self.t_0 = self.clock.now()
            self.get_logger().info("Stopping trajectory tracking")

    def get_current_state(self):
        """获取当前状态"""
        # 当前位置
        pose = np.array([
            [self.current_pa.pose.position.x],
            [self.current_pa.pose.position.y],
            [self.current_pa.pose.position.z]
        ])

        # 当前速度
        velo = np.array([
            [self.current_velo.twist.linear.x],
            [self.current_velo.twist.linear.y],
            [self.current_velo.twist.linear.z]
        ])

        # 当前姿态四元数
        quaternion = [
            self.current_pa.pose.orientation.x,
            self.current_pa.pose.orientation.y,
            self.current_pa.pose.orientation.z,
            self.current_pa.pose.orientation.w
        ]

        # 计算旋转矩阵（body坐标系到world坐标系）
        r = R.from_quat(quaternion)
        rotation_matrix = r.as_matrix()

        # body系z轴方向在world坐标系中的表示
        body_z = np.dot(rotation_matrix, np.array([[0], [0], [1]]))

        return pose, velo, rotation_matrix, body_z

    def calculate_desired_force(self, pose, velo, traj_p, traj_v, traj_a, sliding_gain, tracking_gain):
        """计算期望力"""
        # 计算复合误差（滑模面）
        s = (velo - traj_v + sliding_gain * (pose - traj_p))

        # 前馈参考加速度
        a_r = traj_a - sliding_gain * (velo - traj_v) + np.array([[0], [0], [self.gravity]])

        # 计算自适应补偿力
        adaptive_force = self.current_phi @ self.current_hat.a  # (3*12) * (12*1) = 3*1

        # NL控制部分
        nl_control = a_r - tracking_gain * s

        # 自适应补偿部分
        adaptive_compensation = adaptive_force / self.mass

        # 计算期望力（包含重力补偿、跟踪控制和自适应补偿）
        F_sp = nl_control - adaptive_compensation

        # 定期输出F_sp组成
        self.output_counter += 1
        if self.output_counter >= self.output_interval:
            self.output_counter = 0
            self.get_logger().info(
                f"\n===== F_sp 组成 =====\n"
                f"NL控制 (a_r - K*s): [{nl_control[0,0]:.4f}, {nl_control[1,0]:.4f}, {nl_control[2,0]:.4f}]\n"
                f"自适应补偿 (Φa/m): [{adaptive_compensation[0,0]:.4f}, {adaptive_compensation[1,0]:.4f}, {adaptive_compensation[2,0]:.4f}]\n"
                f"F_sp 总计:         [{F_sp[0,0]:.4f}, {F_sp[1,0]:.4f}, {F_sp[2,0]:.4f}]"
            )

        # 限制输出范围
        F_sp = np.clip(F_sp, np.array([[-5.0], [-5.0], [0.0]]), np.array([[5.0], [5.0], [19.6]]))

        return F_sp, s, adaptive_force


    def calculate_attitude_from_force(self, F_sp, body_z, yaw_sp):
        # 计算所需推力大小（点乘）
        thrust = float(np.dot(F_sp.T, body_z))

        # 创建控制消息
        attitude_target = AttitudeTarget()

        # 计算归一化推力值（考虑效率）
        normalized_thrust = -0.0015 * thrust * thrust + 0.0764 * thrust + 0.1237
        attitude_target.thrust = np.clip(normalized_thrust, 0.0, 1.0)

        actual_thrust_vector = body_z * thrust

        # 基于期望力方向计算期望姿态
        body_z_sp = F_sp / norm(F_sp)  # 期望z轴方向

        # 使用期望偏航角创建x参考方向
        x_C = np.array([[np.cos(yaw_sp)], [np.sin(yaw_sp)], [0]])

        # 计算期望y轴
        body_y_sp = np.cross(body_z_sp.flatten(), x_C.flatten()).reshape(3, 1)
        body_y_sp = body_y_sp / norm(body_y_sp)

        # 计算期望x轴
        body_x_sp = np.cross(body_y_sp.flatten(), body_z_sp.flatten()).reshape(3, 1)

        # 构建期望旋转矩阵
        RM_sp = np.hstack([body_x_sp, body_y_sp, body_z_sp])

        # 转换为四元数
        r_sp = R.from_matrix(RM_sp)
        quaternion_sp = r_sp.as_quat()

        # 设置期望姿态
        attitude_target.orientation.x = quaternion_sp[0]
        attitude_target.orientation.y = quaternion_sp[1]
        attitude_target.orientation.z = quaternion_sp[2]
        attitude_target.orientation.w = quaternion_sp[3]

        # 设置控制掩码
        attitude_target.type_mask = int(7)
        # print("F_sp:", F_sp.T)
        # print("body_z:", body_z.T)
        # print("Thrust:", thrust)

        return attitude_target, actual_thrust_vector

    def update_adaptive_parameters(self, s, delta_tk=None):
        """更新自适应参数估计
        
        使用σ-修正自适应律: ȧ = -λa + γΦᵀs
        离散化: a_{k+1} = (1 - λΔt)a_k + γΔt·Φᵀs
        """
        if delta_tk is None:
            delta_tk = 1 / self.control_rate

        # 自适应参数
        lambda_factor = 0.01  # 遗忘因子
        gamma = 0.5  # 自适应增益

        # σ-修正自适应律
        # ȧ = -λa + γΦᵀs
        self.current_hat.a = (1 - lambda_factor * delta_tk) * self.current_hat.a + \
                             gamma * delta_tk * self.current_phi.T @ s

    def log_flight_data(self, pose, velo, rotation_matrix, traj_p, s):
        """记录飞行数据到CSV文件"""
        # 从旋转矩阵提取欧拉角（roll, pitch, yaw）
        r = R.from_matrix(rotation_matrix)
        euler_angles = r.as_euler('xyz', degrees=True)  # 返回角度
        roll, pitch, yaw = euler_angles

        # 获取当前时间戳（秒）
        current_time = self.get_clock().now().nanoseconds * 1e-9

        # 计算误差
        position_error = norm(pose - traj_p)
        velocity_error = norm(s)  # 使用滑模面作为速度误差的度量

        # 准备数据行
        data_row = [
            current_time,
            pose[0, 0], pose[1, 0], pose[2, 0],  # x, y, z
            velo[0, 0], velo[1, 0], velo[2, 0],  # vx, vy, vz
            roll, pitch, yaw,  # roll, pitch, yaw
            traj_p[0, 0], traj_p[1, 0], traj_p[2, 0],  # x_des, y_des, z_des
            position_error, velocity_error  # 误差
        ]

        # 写入CSV
        self.csv_writer.writerow(data_row)
        self.log_file.flush()  # 确保数据立即写入磁盘

    def controller_cb(self):
        """控制器主回调函数"""
        if self.current_pa is None or self.current_velo is None or self.current_phi is None:
            self.get_logger().warn("Waiting for all required data...")
            return

        # 获取参数
        sliding_gain = np.array(self.get_parameter('sliding_gain').value).reshape(3, 1)
        tracking_gain = np.array(self.get_parameter('tracking_gain').value).reshape(3, 1)
        traj_mode = self.get_parameter('traj_mode').value

        # 更新轨迹时间
        self.update_trajectory_time(traj_mode)

        # 获取目标轨迹
        traj_p = self.traj.pose(self.traj_t)
        traj_v = self.traj.velo(self.traj_t)
        traj_a = self.traj.acce(self.traj_t)
        traj_yaw = self.traj.yaw(self.traj_t)

        # 获取当前状态
        pose, velo, rotation_matrix, body_z = self.get_current_state()

        # 计算期望力
        F_sp, s, adaptive_force = self.calculate_desired_force(
            pose, velo, traj_p, traj_v, traj_a, sliding_gain, tracking_gain)

        # 计算期望姿态和推力
        attitude_target, actual_thrust_vector = self.calculate_attitude_from_force(F_sp, body_z, traj_yaw)

        # 发布控制指令
        self.controller_pub_.publish(attitude_target)

        # 更新自适应参数
        self.update_adaptive_parameters(s)

        # 记录飞行数据到CSV
        self.log_flight_data(pose, velo, rotation_matrix, traj_p, s)

        # 记录调试信息
        if self.get_logger().get_effective_level() <= rclpy.logging.LoggingSeverity.DEBUG:
            self.get_logger().debug(f"Traj time: {self.traj_t:.2f}, Thrust: {attitude_target.thrust:.3f}")
            position_error = norm(pose - traj_p)
            self.get_logger().debug(f"Position error: {position_error:.3f}m")

    def destroy_node(self):
        """节点销毁时关闭日志文件"""
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()
            self.get_logger().info(f"Flight log saved to: {self.log_file_path}")
        super().destroy_node()


class CurrentHat:
    """自适应参数估计器"""

    def __init__(self):
        self.a = np.zeros((12, 1))  # 参数估计值


def main(args=None):
    node = None
    try:
        rclpy.init(args=args)
        node = TrajController("traj_controller_NF3")
        rclpy.spin(node)
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()