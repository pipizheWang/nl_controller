#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from interfaces.msg import PhiEst
from rclpy.qos import QoSProfile, ReliabilityPolicy
import numpy as np

class DemoNode(Node):
    def __init__(self):
        super().__init__('demo_node')
        self.get_logger().info('Hello, test! Node initialized.')

        self.current_phi = None
        self.control_rate = 50.0
        self.a = np.ones((12, 1))
        self.b = np.zeros((3, 1))

        qos_best_effort = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # 订阅 PhiEst 消息
        self.phi_sub_ = self.create_subscription(
            PhiEst, '/control/phiest', self.phi_cb, qos_best_effort)

        self.print_timer_ = self.create_timer(1 / self.control_rate, self.print_b)

    def phi_cb(self, msg):
        # 将 PhiEst 消息转换为 numpy 数组，形状为 (3, 12)
        phi_array = np.vstack([np.array(msg.row1), np.array(msg.row2), np.array(msg.row3)])
        self.current_phi = phi_array

    def print_b(self):
        if self.current_phi is not None:
            # 执行矩阵乘法：(3,12) @ (12,1) -> (3,1)
            print(self.current_phi)
            self.b = self.current_phi @ self.a
            print(self.b)
        else:
            self.get_logger().warn('当前还未接收到 PhiEst 消息')

def main(args=None):
    rclpy.init(args=args)
    node = DemoNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
