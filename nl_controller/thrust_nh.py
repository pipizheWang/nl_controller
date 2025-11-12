#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from mavros_msgs.msg import AttitudeTarget
from geometry_msgs.msg import TwistStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.clock import Clock
import csv
from pathlib import Path
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class ThrustCalibration(Node):
    def __init__(self, name):
        super().__init__(name)
        self.get_logger().info("Node running: %s" % name)

        # 控制频率
        self.control_rate = 50.0
        self.clock = Clock()

        # 物理常数
        self.gravity = 9.81  # 重力加速度 m/s²

        # 状态变量
        self.current_velo = None
        self.prev_velo = None
        self.prev_time = None

        # 标定参数
        # 从高推力开始递减，确保无人机已起飞，避免地面效应
        self.thrust_levels = np.arange(0.85, 0.695, -0.005)  # 从0.85下降到0.70，间隔0.005（共31个点）
        self.current_thrust_idx = 0
        self.is_calibrating = False
        self.steady_state_duration = 3.0  # 每个推力保持3秒
        self.data_collection_start = None

        # 数据收集
        self.accel_buffer = deque(maxlen=int(self.control_rate * self.steady_state_duration))
        self.calibration_data = []  # [(thrust, acceleration), ...]

        # QoS配置
        qos_best_effort = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        qos_reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        # 订阅和发布
        self.velo_sub_ = self.create_subscription(
            TwistStamped, '/mavros/local_position/velocity_local', 
            self.velo_cb, qos_best_effort)

        self.controller_pub_ = self.create_publisher(
            AttitudeTarget, '/mavros/setpoint_raw/attitude', qos_reliable)

        self.controller_timer_ = self.create_timer(1 / self.control_rate, self.controller_cb)

        # 初始化日志
        self.setup_calibration_log()

        self.get_logger().info("Thrust calibration node initialized")
        self.get_logger().info(f"Gravity compensation: g = {self.gravity} m/s²")
        self.get_logger().info(f"Testing {len(self.thrust_levels)} thrust levels: {self.thrust_levels}")

    def setup_calibration_log(self):
        """初始化标定日志"""
        package_dir = Path.cwd()
        log_dir = package_dir / 'log'
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
        self.log_file_path = log_dir / f'thrust_calibration_{timestamp}.csv'

        self.log_file = open(self.log_file_path, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)

        # 写入表头
        header = ['timestamp', 'thrust', 'vz', 'az_measured', 'az_thrust_compensated']
        self.csv_writer.writerow(header)
        self.log_file.flush()

        self.get_logger().info(f"Calibration log: {self.log_file_path}")

    def velo_cb(self, msg):
        """速度回调函数"""
        self.current_velo = msg

    def calculate_acceleration(self):
        """计算当前加速度（z轴）"""
        if self.current_velo is None or self.prev_velo is None or self.prev_time is None:
            return None

        current_time = self.get_clock().now()
        dt = (current_time - self.prev_time).nanoseconds * 1e-9

        if dt <= 0:
            return None

        # 计算z轴加速度
        current_vz = self.current_velo.twist.linear.z
        prev_vz = self.prev_velo.twist.linear.z
        az = (current_vz - prev_vz) / dt

        return az

    def start_calibration(self):
        """开始标定流程"""
        self.is_calibrating = True
        self.current_thrust_idx = 0
        self.data_collection_start = self.get_clock().now()
        self.accel_buffer.clear()
        self.get_logger().info("Starting calibration...")

    def controller_cb(self):
        """控制器主回调"""
        if self.current_velo is None:
            self.get_logger().warn("Waiting for velocity data...")
            return

        # 手动开始标定（可通过参数触发）
        if not self.is_calibrating:
            # 等待用户命令开始标定
            return

        # 计算加速度
        az = self.calculate_acceleration()
        
        # 更新历史数据
        self.prev_velo = self.current_velo
        self.prev_time = self.get_clock().now()

        if az is None:
            return

        # 获取当前推力值
        current_thrust = self.thrust_levels[self.current_thrust_idx]

        # 发布推力指令（保持水平，只改变推力）
        attitude_target = AttitudeTarget()
        attitude_target.thrust = float(current_thrust)
        # 保持当前姿态（设置为单位四元数）
        attitude_target.orientation.w = 1.0
        attitude_target.orientation.x = 0.0
        attitude_target.orientation.y = 0.0
        attitude_target.orientation.z = 0.0
        attitude_target.type_mask = int(7)  # 控制姿态和推力

        self.controller_pub_.publish(attitude_target)

        # 收集数据
        elapsed_time = (self.get_clock().now() - self.data_collection_start).nanoseconds * 1e-9
        
        # 忽略前0.5秒的瞬态响应
        if elapsed_time > 0.5:
            self.accel_buffer.append(az)

        # 记录原始数据
        current_time = self.get_clock().now().nanoseconds * 1e-9
        vz = self.current_velo.twist.linear.z
        az_thrust_compensated = az + self.gravity  # 补偿重力
        self.csv_writer.writerow([current_time, current_thrust, vz, az, az_thrust_compensated])
        self.log_file.flush()

        # 检查是否达到稳态持续时间
        if elapsed_time >= self.steady_state_duration:
            # 计算该推力下的平均加速度
            if len(self.accel_buffer) > 0:
                avg_acceleration_measured = np.mean(list(self.accel_buffer))
                std_acceleration = np.std(list(self.accel_buffer))
                
                # 补偿重力：测量的是 az_measured = az_thrust - g
                # 所以油门产生的加速度是 az_thrust = az_measured + g
                avg_acceleration_thrust = avg_acceleration_measured + self.gravity
                
                self.calibration_data.append((current_thrust, avg_acceleration_thrust))
                self.get_logger().info(
                    f"Thrust: {current_thrust:.3f}, Az_measured: {avg_acceleration_measured:.3f} ± {std_acceleration:.3f} m/s², "
                    f"Az_thrust: {avg_acceleration_thrust:.3f} m/s² (with gravity compensation)")

            # 进入下一个推力级别
            self.current_thrust_idx += 1
            self.accel_buffer.clear()
            self.data_collection_start = self.get_clock().now()

            # 检查是否完成所有测试
            if self.current_thrust_idx >= len(self.thrust_levels):
                self.finish_calibration()

    def finish_calibration(self):
        """完成标定并进行拟合"""
        self.is_calibrating = False
        self.get_logger().info("Calibration complete!")

        if len(self.calibration_data) < 3:
            self.get_logger().error("Not enough data for fitting")
            return

        # 跳过第一组数据（误差较大的初始数据）
        calibration_data_filtered = self.calibration_data[1:]
        self.get_logger().info(f"Skipping first data point, using {len(calibration_data_filtered)} points for fitting")

        # 提取数据
        thrusts = np.array([d[0] for d in calibration_data_filtered])
        accelerations = np.array([d[1] for d in calibration_data_filtered])

        # 反向拟合：从加速度到推力 thrust = f(az)
        # 线性模型: thrust = k * az + b
        def linear_model_inverse(az, k, b):
            return k * az + b

        # 二次模型: thrust = a * az^2 + b * az + c
        def quadratic_model_inverse(az, a, b, c):
            return a * az**2 + b * az + c

        try:
            # 线性拟合 (az -> thrust)
            popt_linear, _ = curve_fit(linear_model_inverse, accelerations, thrusts)
            k, b = popt_linear
            self.get_logger().info(f"Linear fit: thrust = {k:.4f} * az + {b:.4f}")

            # 二次拟合 (az -> thrust)
            popt_quad, _ = curve_fit(quadratic_model_inverse, accelerations, thrusts)
            a, b_q, c = popt_quad
            self.get_logger().info(f"Quadratic fit: thrust = {a:.4f} * az² + {b_q:.4f} * az + {c:.4f}")

            # 绘图
            self.plot_results(thrusts, accelerations, popt_linear, popt_quad)

        except Exception as e:
            self.get_logger().error(f"Fitting error: {e}")

    def plot_results(self, thrusts, accelerations, popt_linear, popt_quad):
        """绘制拟合结果 - 加速度到推力的映射"""
        plt.figure(figsize=(12, 5))
        
        # 子图1: 加速度 -> 推力 (实际应用方向)
        plt.subplot(1, 2, 1)
        plt.scatter(accelerations, thrusts, color='blue', label='Measured data', s=50)

        # 拟合曲线
        az_range = np.linspace(accelerations.min(), accelerations.max(), 100)
        
        # 线性拟合
        k, b = popt_linear
        linear_fit = k * az_range + b
        plt.plot(az_range, linear_fit, 'r-', label=f'Linear: T = {k:.4f}*az + {b:.4f}', linewidth=2)

        # 二次拟合
        a, b_q, c = popt_quad
        quad_fit = a * az_range**2 + b_q * az_range + c
        plt.plot(az_range, quad_fit, 'g--', label=f'Quadratic: T = {a:.4f}*az² + {b_q:.4f}*az + {c:.4f}', linewidth=2)

        plt.xlabel('Thrust-generated Acceleration (m/s²)', fontsize=12)
        plt.ylabel('Normalized Thrust', fontsize=12)
        plt.title('Acceleration to Thrust Mapping (Gravity Compensated)', fontsize=14)
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)

        # 子图2: 推力 -> 加速度 (反向查看)
        plt.subplot(1, 2, 2)
        plt.scatter(thrusts, accelerations, color='orange', label='Measured data', s=50)
        
        thrust_range = np.linspace(thrusts.min(), thrusts.max(), 100)
        
        # 反向计算加速度 (用于可视化)
        # 从 thrust = k*az + b 推导 az = (thrust - b) / k
        az_from_linear = (thrust_range - b) / k
        plt.plot(thrust_range, az_from_linear, 'r-', label='Linear (inverse)', linewidth=2)
        
        # 对于二次拟合的反向关系（更复杂，这里简化处理）
        # 对数据按推力排序后绘制趋势线
        sorted_indices = np.argsort(thrusts)
        thrusts_sorted = thrusts[sorted_indices]
        accelerations_sorted = accelerations[sorted_indices]
        plt.plot(thrusts_sorted, accelerations_sorted, 'g--', label='Data trend', linewidth=2, alpha=0.5)

        plt.xlabel('Normalized Thrust', fontsize=12)
        plt.ylabel('Acceleration (m/s²)', fontsize=12)
        plt.title('Thrust to Acceleration (Verification)', fontsize=14)
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图像
        plot_path = self.log_file_path.parent / f'thrust_calibration_{datetime.now().strftime("%Y-%m-%d_%H%M")}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.get_logger().info(f"Plot saved to: {plot_path}")
        
        plt.show()

    def destroy_node(self):
        """销毁节点"""
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()
            self.get_logger().info(f"Calibration log saved: {self.log_file_path}")
        super().destroy_node()


def main(args=None):
    node = None
    try:
        rclpy.init(args=args)
        node = ThrustCalibration("thrust_nh")
        
        # 启动标定流程
        node.start_calibration()
        
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Calibration interrupted by user")
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()