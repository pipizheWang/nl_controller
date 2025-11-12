#!/usr/bin/env python3
import rclpy
import numpy as np
import math
from numpy.linalg import norm
from rclpy.node import Node
from mavros_msgs.msg import AttitudeTarget
from geometry_msgs.msg import PoseStamped, TwistStamped
from scipy.spatial.transform import Rotation as R
from rclpy.qos import QoSProfile, ReliabilityPolicy
from .traj import TargetTraj
from rclpy.clock import Clock
import threading
import time


class BacksteppingController(Node):
    def __init__(self, name):
        super().__init__(name)
        self.get_logger().info("Node running: %s" % name)

        # 控制频率
        self.control_rate = 50.0
        self.traj = TargetTraj(FLAG=1)

        # 初始化时钟
        self.clock = Clock()

        # 初始化状态变量
        self.current_pa = None
        self.current_velo = None

        # 轨迹计时器
        self.traj_t = -1.0
        self.t_0 = self.clock.now()

        # Backstepping 控制参数
        self.k_1 = 1.0
        self.k_2 = 18.0
        self.k_3 = 2.0

        # 自适应控制参数
        self.gama = 0.002
        # X通道自适应参数
        self.a_x_hat = 1e-7
        self.rho_x_hat = 1e-7
        self.z_3 = 0.0
        self.alpha_3 = 0.0
        # Y通道自适应参数
        self.a_y_hat = 1e-7
        self.rho_y_hat = 1e-7
        self.e_3 = 0.0
        self.tao_3 = 0.0

        # 系统常量
        self.gravity = 9.8
        self.thrust_efficiency = 0.74
        self.mass = 0.05  # 无人机质量 (kg)

        # 订阅和发布
        qos_best_effort = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        qos_reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self.pa_sub_ = self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self.pa_cb, qos_best_effort)
        self.velo_sub_ = self.create_subscription(
            TwistStamped, '/mavros/local_position/velocity_local', self.velo_cb, qos_best_effort)

        self.controller_pub_ = self.create_publisher(
            AttitudeTarget, '/mavros/setpoint_raw/attitude', qos_reliable)

        self.controller_timer_ = self.create_timer(1 / self.control_rate, self.controller_cb)

        # 初始化参数
        self.declare_parameter('traj_mode', False)  # 轨迹模式开关
        self.declare_parameter('k1', 1.0)           # Backstepping参数1
        self.declare_parameter('k2', 18.0)          # Backstepping参数2
        self.declare_parameter('k3', 2.0)           # Backstepping参数3
        self.declare_parameter('gama', 0.002)       # 自适应学习率

        # 启动自适应参数更新线程
        self.adaptive_thread = threading.Thread(target=self.adaptive_control_update)
        self.adaptive_thread.daemon = True
        self.adaptive_thread.start()

    def pa_cb(self, msg):
        self.current_pa = msg

    def velo_cb(self, msg):
        self.current_velo = msg

    def update_trajectory_time(self, traj_mode):
        if traj_mode and self.traj_t == -1.0:
            self.t_0 = self.clock.now()
            self.traj_t = 0.0
            self.get_logger().info("Starting trajectory tracking")
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

        # 计算旋转矩阵和欧拉角
        r = R.from_quat(quaternion)
        rotation_matrix = r.as_matrix()
        euler_angles = r.as_euler('xyz')  # roll, pitch, yaw

        # body系z轴方向在world坐标系中的表示
        body_z = np.dot(rotation_matrix, np.array([[0], [0], [1]]))

        return pose, velo, rotation_matrix, body_z, euler_angles

    def x_backstepping(self, x_1d, x_1d_dot, x_1d_ddot, x_1d_dddot, x_1, x_2, theta):
        """X方向的Backstepping控制计算俯仰角命令"""
        g = self.gravity

        z_1 = x_1 - x_1d
        alpha_1 = x_1d_dot - self.k_1 * z_1
        alpha_1_dot = x_1d_ddot - self.k_1 * (x_2 - x_1d_dot)

        z_2 = x_2 - alpha_1
        alpha_2 = 1/g * (alpha_1_dot - self.k_2*z_2 - z_1)
        alpha_1_ddot = x_1d_dddot - self.k_1 * (g*math.tan(theta) - x_1d_ddot)

        self.z_3 = math.tan(theta) - alpha_2
        alpha_2_dot = 1/g * (alpha_1_ddot - self.k_2*(g*math.tan(theta) - alpha_1_dot) - (x_2 - x_1d_dot))

        self.alpha_3 = -self.a_x_hat * theta + (math.cos(theta))**2 * (alpha_2_dot - g*z_2 - self.k_3*self.z_3)
        theta_c = self.rho_x_hat * self.alpha_3

        return theta_c

    def y_backstepping(self, y_1d, y_1d_dot, y_1d_ddot, y_1d_dddot, y_1, y_2, phi):
        """Y方向的Backstepping控制计算滚转角命令"""
        g = self.gravity

        e_1 = y_1 - y_1d
        tao_1 = y_1d_dot - self.k_1 * e_1
        tao_1_dot = y_1d_ddot - self.k_1 * (y_2 - y_1d_dot)

        e_2 = y_2 - tao_1
        tao_2 = -1/g * (tao_1_dot - self.k_2*e_2 - e_1)
        tao_1_ddot = y_1d_dddot - self.k_1 * (-g*math.tan(phi) - y_1d_ddot)

        self.e_3 = math.tan(phi) - tao_2
        tao_2_dot = -1/g * (tao_1_ddot - self.k_2*(-g*math.tan(phi) - tao_1_dot) - (y_2 - y_1d_dot))

        self.tao_3 = -self.a_y_hat*phi + (math.cos(phi))**2 * (tao_2_dot + g*e_2 - self.k_3 * self.e_3)
        phi_c = self.rho_y_hat * self.tao_3

        return phi_c

    def adaptive_control_update(self):
        """自适应参数更新线程"""
        rate = 50  # Hz
        while rclpy.ok():
            if self.current_pa is not None:
                # 获取当前姿态角
                quaternion = [
                    self.current_pa.pose.orientation.x,
                    self.current_pa.pose.orientation.y,
                    self.current_pa.pose.orientation.z,
                    self.current_pa.pose.orientation.w
                ]
                r = R.from_quat(quaternion)
                euler_angles = r.as_euler('xyz')
                phi, theta, _ = euler_angles

                # 自适应参数更新律
                dt = 1.0 / rate
                
                # X通道参数更新
                if abs(math.cos(theta)) > 1e-6:
                    a_x_hat_dot = 1/(math.cos(theta))**2 * theta * self.z_3
                    rho_x_hat_dot = -self.gama * self.z_3 / (math.cos(theta))**2 * self.alpha_3
                    self.a_x_hat += a_x_hat_dot * dt
                    self.rho_x_hat += rho_x_hat_dot * dt

                # Y通道参数更新
                if abs(math.cos(phi)) > 1e-6:
                    a_y_hat_dot = 1/(math.cos(phi))**2 * phi * self.e_3
                    rho_y_hat_dot = -self.gama * self.e_3 / (math.cos(phi))**2 * self.tao_3
                    self.a_y_hat += a_y_hat_dot * dt
                    self.rho_y_hat += rho_y_hat_dot * dt

            time.sleep(1.0 / rate)

    def saturate_angle(self, angle, max_angle):
        """角度饱和限制"""
        return np.clip(angle, -max_angle, max_angle)

    def saturate_thrust(self, thrust):
        """推力饱和限制"""
        return np.clip(thrust, 0.0, 1.0)  # 根据您原代码的限制

    def calculate_lqr_thrust_yaw(self, pose, velo, traj_p, traj_v, euler_angles):
        """计算LQR推力和偏航控制（简化版）"""
        # 位置误差
        pos_error = pose - traj_p
        vel_error = velo - traj_v
        
        # 简化的推力控制（基于Z方向）
        z_error = pos_error[2, 0]
        vz_error = vel_error[2, 0]
        
        # 推力控制律
        balance_thrust = 0.73  # 平衡推力
        thrust_command = balance_thrust + 0.02 * z_error + 0.02 * vz_error
        thrust_command = self.saturate_thrust(thrust_command)
        
        # 偏航控制（简化）
        yaw_command = 0.0  # 保持偏航为0
        
        return thrust_command, yaw_command

    def calculate_backstepping_attitude(self, pose, velo, traj_p, traj_v, traj_a, euler_angles):
        """计算Backstepping姿态控制"""
        # 获取期望轨迹的导数
        traj_p_dot = traj_v
        traj_p_ddot = traj_a
        # 简化：假设三阶导数为0
        traj_p_dddot = np.zeros_like(traj_a)

        # 当前状态
        x_1, y_1, z_1 = pose[0, 0], pose[1, 0], pose[2, 0]
        x_2, y_2, z_2 = velo[0, 0], velo[1, 0], velo[2, 0]
        
        # 期望轨迹
        x_1d, y_1d, z_1d = traj_p[0, 0], traj_p[1, 0], traj_p[2, 0]
        x_1d_dot, y_1d_dot, z_1d_dot = traj_p_dot[0, 0], traj_p_dot[1, 0], traj_p_dot[2, 0]
        x_1d_ddot, y_1d_ddot, z_1d_ddot = traj_p_ddot[0, 0], traj_p_ddot[1, 0], traj_p_ddot[2, 0]
        x_1d_dddot, y_1d_dddot, z_1d_dddot = traj_p_dddot[0, 0], traj_p_dddot[1, 0], traj_p_dddot[2, 0]

        # 当前姿态角
        roll, pitch, yaw = euler_angles

        # Backstepping控制计算
        pitch_command = self.x_backstepping(x_1d, x_1d_dot, x_1d_ddot, x_1d_dddot, x_1, x_2, pitch)
        roll_command = self.y_backstepping(y_1d, y_1d_dot, y_1d_ddot, y_1d_dddot, y_1, y_2, roll)

        # 角度饱和限制（转换为度）
        roll_command_deg = self.saturate_angle(math.degrees(roll_command), 20.0)
        pitch_command_deg = self.saturate_angle(math.degrees(pitch_command), 20.0)

        return roll_command_deg, pitch_command_deg

    def create_attitude_target(self, roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd):
        """创建姿态控制消息"""
        attitude_target = AttitudeTarget()
        
        # 将角度命令转换为四元数
        roll_rad = math.radians(roll_cmd)
        pitch_rad = math.radians(pitch_cmd)
        yaw_rad = math.radians(yaw_cmd)
        
        r_sp = R.from_euler('xyz', [roll_rad, pitch_rad, yaw_rad])
        quaternion_sp = r_sp.as_quat()
        
        attitude_target.orientation.x = quaternion_sp[0]
        attitude_target.orientation.y = quaternion_sp[1]
        attitude_target.orientation.z = quaternion_sp[2]
        attitude_target.orientation.w = quaternion_sp[3]
        
        # 推力转换（从牛顿到归一化值）
        normalized_thrust = np.sqrt(thrust_cmd / self.gravity * self.thrust_efficiency)
        attitude_target.thrust = np.clip(normalized_thrust, 0.0, 1.0)
        
        # 设置控制掩码
        attitude_target.type_mask = int(7)  # 忽略角速度控制
        
        return attitude_target

    def controller_cb(self):
        """控制器主回调函数"""
        if self.current_pa is None or self.current_velo is None:
            self.get_logger().warn("Waiting for pose and velocity data...")
            return

        # 获取参数
        traj_mode = self.get_parameter('traj_mode').value
        self.k_1 = self.get_parameter('k1').value
        self.k_2 = self.get_parameter('k2').value
        self.k_3 = self.get_parameter('k3').value
        self.gama = self.get_parameter('gama').value

        # 更新轨迹时间
        self.update_trajectory_time(traj_mode)

        # 获取目标轨迹
        traj_p = self.traj.pose(self.traj_t)
        traj_v = self.traj.velo(self.traj_t)
        traj_a = self.traj.acce(self.traj_t)
        traj_yaw = self.traj.yaw(self.traj_t)

        # 获取当前状态
        pose, velo, rotation_matrix, body_z, euler_angles = self.get_current_state()

        # 计算LQR推力和偏航控制
        thrust_cmd, yaw_cmd = self.calculate_lqr_thrust_yaw(pose, velo, traj_p, traj_v, euler_angles)

        # 计算Backstepping姿态控制
        roll_cmd, pitch_cmd = self.calculate_backstepping_attitude(pose, velo, traj_p, traj_v, traj_a, euler_angles)

        # 创建控制指令
        attitude_target = self.create_attitude_target(roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd)

        # 发布控制指令
        self.controller_pub_.publish(attitude_target)

        # 调试信息
        if self.get_logger().get_effective_level() <= rclpy.logging.LoggingSeverity.DEBUG:
            self.get_logger().debug(f"Thrust: {thrust_cmd:.3f}, Roll: {roll_cmd:.1f}°, Pitch: {pitch_cmd:.1f}°")
            self.get_logger().debug(f"Adaptive params - a_x: {self.a_x_hat:.6f}, rho_x: {self.rho_x_hat:.6f}")


def main(args=None):
    try:
        rclpy.init(args=args)
        node = BacksteppingController("backstepping_controller")
        rclpy.spin(node)
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()