#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from mavros_msgs.msg import AttitudeTarget, RCOut
from geometry_msgs.msg import PoseStamped, TwistStamped
from scipy.spatial.transform import Rotation as R
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu
from interfaces.msg import PhiEst
from .neural_network import load_model
import torch


class PhiEstimate(Node):
    def __init__(self, name):
        super().__init__(name)
        self.get_logger().info(f"Node running: {name}")

        self.Rate = 50.0
        self.current_pa = PoseStamped()
        self.current_velo = TwistStamped()
        self.current_pwm = RCOut()
        self.nf_hover_pwm = 910.0
        self.hover_pwm = 770.0
        self.hover_pwm_ratio = self.nf_hover_pwm / self.hover_pwm

        # 加载神经网络模型
        import os
        model_folder = os.path.dirname(__file__)
        model_name = 'neural-fly_dim-a-4_v-q-pwm-epoch-950'
        self.neural_network = load_model(modelname=model_name, modelfolder=model_folder)

        # 创建订阅者
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pa_cb, qos_profile)
        self.create_subscription(TwistStamped, '/mavros/local_position/velocity_local', self.velo_cb, qos_profile)
        self.create_subscription(RCOut, '/mavros/rc/out', self.pwm_cb, qos_profile)

        # 创建发布者
        self.phi_pub_ = self.create_publisher(PhiEst, '/control/phiest', qos_profile)

        # 定时器
        self.create_timer(1 / self.Rate, self.phi_cb)

    def pa_cb(self, msg):
        self.current_pa = msg

    def velo_cb(self, msg):
        self.current_velo = msg

    def pwm_cb(self, msg):
        self.current_pwm = msg

    def phi_cb(self):
        try:
            velo = np.array(
                [self.current_velo.twist.linear.x, self.current_velo.twist.linear.y, self.current_velo.twist.linear.z])
            quaternion = [self.current_pa.pose.orientation.x, self.current_pa.pose.orientation.y,
                          self.current_pa.pose.orientation.z, self.current_pa.pose.orientation.w]

            # PWM处理
            pwm_len = len(self.current_pwm.channels)
            PWM = np.zeros(4)
            for i in range(min(4, pwm_len)):
                PWM[i] = self.current_pwm.channels[i] / 1000 * self.hover_pwm_ratio

            # 神经网络输入
            inputs = np.concatenate([velo, quaternion, PWM])
            inputs = torch.tensor(inputs, dtype=torch.float)
            Phi_output_ori = self.neural_network.phi(inputs).detach().numpy()

            # Phi_output_true = Phi_output_ori[:4]
            Phi = np.zeros((3, 12))
            Phi[0, 0:4] = Phi_output_ori
            Phi[1, 4:8] = Phi_output_ori
            Phi[2, 8:12] = Phi_output_ori

            # print(Phi)

            Phi_msg = PhiEst()
            Phi_msg.row1 = Phi[0, :].tolist()
            Phi_msg.row2 = Phi[1, :].tolist()
            Phi_msg.row3 = Phi[2, :].tolist()

            self.phi_pub_.publish(Phi_msg)
            self.get_logger().info("PhiEst Published Successfully")
        except Exception as e:
            self.get_logger().error(f"Error in phi_cb: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = PhiEstimate("Phi_estimate")
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
