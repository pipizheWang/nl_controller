
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

class TestNode(Node):
    def __init__(self):
        super().__init__('test_node')
        self.get_logger().info('Test node is running!')

def main(args=None):
    rclpy.init(args=args)
    node = TestNode()
    rclpy.spin_once(node)
    node.destroy_node()
    rclpy.shutdown()
    return 0

if __name__ == '__main__':
    main()
