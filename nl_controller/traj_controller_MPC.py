#!/usr/bin/env python3
import rclpy
import numpy as np
from numpy.linalg import norm
from rclpy.node import Node
from mavros_msgs.msg import AttitudeTarget
from geometry_msgs.msg import PoseStamped, TwistStamped
from scipy.spatial.transform import Rotation as R
from rclpy.qos import QoSProfile, ReliabilityPolicy
from .traj import TargetTraj
from rclpy.clock import Clock
import csv
import os
import cvxpy as cp


class MPCTrajController(Node):
    def __init__(self, name):
        # 初始化
        super().__init__(name)
        self.get_logger().info("Node running: %s" % name)

        # 控制频率
        self.control_rate = 50.0  # 50Hz
        self.dt = 1.0 / self.control_rate
        self.traj = TargetTraj(FLAG=1)

        # Log
        # 使用当前时间创建唯一的文件名
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"mpc_trajectory_log_{current_time}.csv"

        # 确保日志目录存在
        log_dir = "/home/swarm/wz/Sim/Log"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 初始化数据记录
        self.log_file_path = os.path.join(log_dir, log_filename)
        self.log_file = open(self.log_file_path, mode='w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(["Time", "Pos_x", "Pos_y", "Pos_z",
                                  "Vel_x", "Vel_y", "Vel_z",
                                  "Des_x", "Des_y", "Des_z",
                                  "Position_Error", "Velocity_Error",
                                  "MPC_Force_x", "MPC_Force_y", "MPC_Force_z"])  # 更新标题行

        self.get_logger().info(f"Logging data to {self.log_file_path}")

        # 初始化时钟
        self.clock = Clock()

        # 系统常量
        self.gravity = 9.8
        self.thrust_efficiency = 0.72
        self.mass = 1.0  # 飞行器质量，kg

        # 初始化状态变量
        self.current_pa = None
        self.current_velo = None

        # 轨迹计时器
        self.traj_t = -1.0
        self.t_0 = self.clock.now()

        # 订阅和发布
        qos_best_effort = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        qos_reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self.pa_sub_ = self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self.pa_cb, qos_best_effort)
        self.velo_sub_ = self.create_subscription(
            TwistStamped, '/mavros/local_position/velocity_local', self.velo_cb, qos_best_effort)

        self.controller_pub_ = self.create_publisher(
            AttitudeTarget, '/control/attitude', qos_reliable)

        self.controller_timer_ = self.create_timer(1 / self.control_rate, self.controller_cb)

        # 初始化MPC参数
        self.declare_parameter('horizon', 10)  # 预测步长
        self.declare_parameter('traj_mode', False)  # 轨迹模式开关

        # MPC权重
        self.declare_parameter('position_weight', [10.0, 10.0, 15.0])  # 位置误差权重
        self.declare_parameter('velocity_weight', [5.0, 5.0, 7.0])  # 速度误差权重
        self.declare_parameter('control_weight', [1.0, 1.0, 1.0])  # 控制输入权重
        self.declare_parameter('control_rate_weight', [0.5, 0.5, 0.5])  # 控制变化率权重

        # 控制约束
        self.declare_parameter('max_force', [5.0, 5.0, 19.6])  # 最大力
        self.declare_parameter('min_force', [-5.0, -5.0, 0.0])  # 最小力

    def pa_cb(self, msg):
        self.current_pa = msg

    def velo_cb(self, msg):
        self.current_velo = msg

    def update_trajectory_time(self, traj_mode):
        if traj_mode and self.traj_t == -1.0:
            self.t_0 = self.clock.now()
            self.traj_t = 0.0
            self.get_logger().info("Starting trajectory tracking with MPC")
        elif traj_mode:
            self.traj_t = (self.clock.now() - self.t_0).nanoseconds * 1e-9
        else:
            self.traj_t = -1.0
            self.t_0 = self.clock.now()

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

    def get_reference_trajectory(self, t_current, horizon):
        """获取未来参考轨迹"""
        ref_pos = []
        ref_vel = []
        ref_acc = []
        ref_yaw = []

        dt = self.dt
        for i in range(horizon):
            t = t_current + i * dt
            ref_pos.append(self.traj.pose(t))
            ref_vel.append(self.traj.velo(t))
            ref_acc.append(self.traj.acce(t))
            ref_yaw.append(self.traj.yaw(t))

        return ref_pos, ref_vel, ref_acc, ref_yaw

    def mpc_controller(self, current_state, reference_trajectory, weights, constraints):
        """MPC控制器"""
        # 获取参数
        horizon = self.get_parameter('horizon').value
        dt = self.dt

        # 解包当前状态
        x, v = current_state
        position = x.flatten()
        velocity = v.flatten()

        # 解包参考轨迹
        ref_pos, ref_vel, ref_acc = reference_trajectory

        # 解包权重和约束
        pos_weight, vel_weight, control_weight, control_rate_weight = weights
        max_force, min_force = constraints

        # 定义状态空间模型（线性时不变系统）
        # 状态 x = [px, py, pz, vx, vy, vz]
        # 输入 u = [Fx, Fy, Fz]
        # dx/dt = Ax + Bu + g

        # 离散化状态空间模型
        A = np.zeros((6, 6))
        A[0:3, 3:6] = np.eye(3)  # 位置导数等于速度
        A[3:6, 3:6] = np.zeros((3, 3))  # 速度导数由力决定

        B = np.zeros((6, 3))
        B[3:6, 0:3] = np.eye(3) / self.mass  # F = ma -> a = F/m

        # 重力影响
        g = np.zeros((6, 1))
        g[5, 0] = -self.gravity  # z方向重力加速度

        # 离散化
        Ad = np.eye(6) + A * dt
        Bd = B * dt
        gd = g * dt

        # 初始状态
        x0 = np.zeros((6, 1))
        x0[0:3, 0] = position
        x0[3:6, 0] = velocity

        # 定义MPC优化问题
        x = cp.Variable((6, horizon + 1))
        u = cp.Variable((3, horizon))

        # 目标函数
        cost = 0

        for i in range(horizon):
            # 状态误差成本
            pos_error = x[0:3, i + 1] - ref_pos[i].flatten()
            vel_error = x[3:6, i + 1] - ref_vel[i].flatten()

            cost += cp.sum(cp.multiply(pos_weight, cp.square(pos_error)))
            cost += cp.sum(cp.multiply(vel_weight, cp.square(vel_error)))

            # 控制输入成本
            cost += cp.sum(cp.multiply(control_weight, cp.square(u[:, i])))

            # 控制变化率成本（平滑控制）
            if i > 0:
                cost += cp.sum(cp.multiply(control_rate_weight, cp.square(u[:, i] - u[:, i - 1])))

        # 约束条件
        constraints = []

        # 初始状态约束
        constraints += [x[:, 0] == x0.flatten()]

        # 系统动态约束
        for i in range(horizon):
            constraints += [x[:, i + 1] == Ad @ x[:, i] + Bd @ u[:, i] + gd.flatten()]

        # 控制输入约束
        for i in range(horizon):
            constraints += [u[:, i] <= max_force]
            constraints += [u[:, i] >= min_force]

        # 解决MPC问题
        problem = cp.Problem(cp.Minimize(cost), constraints)
        try:
            problem.solve(solver=cp.OSQP, verbose=False)

            if problem.status != cp.OPTIMAL:
                self.get_logger().warn(f"MPC optimization not optimal. Status: {problem.status}")
                # 如果优化失败，返回一个安全的默认值
                return np.array([[0], [0], [self.mass * self.gravity]])

            # 返回第一个控制输入
            control_input = u[:, 0].value.reshape(3, 1)
            return control_input

        except Exception as e:
            self.get_logger().error(f"MPC optimization error: {e}")
            # 返回一个安全的默认值
            return np.array([[0], [0], [self.mass * self.gravity]])

    def calculate_attitude_from_force(self, F_sp, body_z, yaw_sp):
        # 计算所需推力大小（点乘）
        thrust = float(np.dot(F_sp.T, body_z))

        # 创建控制消息
        attitude_target = AttitudeTarget()

        # 计算归一化推力值（考虑效率）
        normalized_thrust = np.sqrt(thrust / self.gravity * self.thrust_efficiency * self.thrust_efficiency)
        attitude_target.thrust = np.clip(normalized_thrust, 0.0, 1.0)

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

        # 设置控制掩码（仅控制姿态和推力）
        attitude_target.type_mask = int(7)  # 忽略角速度控制

        return attitude_target

    def controller_cb(self):
        """控制器主回调函数"""
        if self.current_pa is None or self.current_velo is None:
            self.get_logger().warn("Waiting for pose and velocity data...")
            return

        # 获取参数
        traj_mode = self.get_parameter('traj_mode').value
        horizon = self.get_parameter('horizon').value

        # 获取权重参数
        position_weight = np.array(self.get_parameter('position_weight').value)
        velocity_weight = np.array(self.get_parameter('velocity_weight').value)
        control_weight = np.array(self.get_parameter('control_weight').value)
        control_rate_weight = np.array(self.get_parameter('control_rate_weight').value)

        # 获取约束参数
        max_force = np.array(self.get_parameter('max_force').value)
        min_force = np.array(self.get_parameter('min_force').value)

        # 更新轨迹时间
        self.update_trajectory_time(traj_mode)

        # 获取当前轨迹点
        traj_p = self.traj.pose(self.traj_t)
        traj_v = self.traj.velo(self.traj_t)
        traj_a = self.traj.acce(self.traj_t)
        traj_yaw = self.traj.yaw(self.traj_t)

        # 获取当前状态
        pose, velo, rotation_matrix, body_z = self.get_current_state()

        # 获取未来参考轨迹
        ref_pos, ref_vel, ref_acc, ref_yaw = self.get_reference_trajectory(self.traj_t, horizon)

        # 记录数据
        position_error = norm(pose - traj_p)
        velocity_error = norm(velo - traj_v)

        # 执行MPC控制
        current_state = (pose, velo)
        reference_trajectory = (ref_pos, ref_vel, ref_acc)
        weights = (position_weight, velocity_weight, control_weight, control_rate_weight)
        constraints = (max_force, min_force)

        # 计算MPC控制力
        F_sp = self.mpc_controller(current_state, reference_trajectory, weights, constraints)

        # 记录数据到日志
        timestamp = self.clock.now().nanoseconds * 1e-9
        self.csv_writer.writerow([timestamp,
                                  pose[0, 0], pose[1, 0], pose[2, 0],
                                  velo[0, 0], velo[1, 0], velo[2, 0],
                                  traj_p[0, 0], traj_p[1, 0], traj_p[2, 0],
                                  position_error, velocity_error,
                                  F_sp[0, 0], F_sp[1, 0], F_sp[2, 0]])

        # 确保数据实时写入文件
        self.log_file.flush()

        # 计算期望姿态和推力
        attitude_target = self.calculate_attitude_from_force(F_sp, body_z, traj_yaw)

        # 发布控制指令
        self.controller_pub_.publish(attitude_target)

        # 记录调试信息
        if self.get_logger().get_effective_level() <= rclpy.logging.LoggingSeverity.DEBUG:
            self.get_logger().debug(f"Traj time: {self.traj_t:.2f}, Thrust: {attitude_target.thrust:.3f}")
            self.get_logger().debug(f"Position error: {position_error:.3f}m")
            self.get_logger().debug(f"MPC Force: [{F_sp[0, 0]:.3f}, {F_sp[1, 0]:.3f}, {F_sp[2, 0]:.3f}]")


def main(args=None):
    try:
        rclpy.init(args=args)
        node = MPCTrajController("mpc_traj_controller")
        rclpy.spin(node)
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        # 关闭日志文件
        if hasattr(node, 'log_file') and node.log_file is not None:
            node.log_file.close()
        rclpy.shutdown()


if __name__ == '__main__':
    main()