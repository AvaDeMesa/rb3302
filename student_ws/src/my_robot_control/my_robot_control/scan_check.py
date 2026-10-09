#!/usr/bin/env python3
"""
scan_check.py  -  helper to find which lidar angle is the FRONT of the robot.

Run it, then hold your hand (or a box) about 30 cm directly in front of the robot.
It prints the angle of the closest object. That angle is your FRONT_ANGLE_DEG.

    ros2 run my_robot_control scan_check
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class ScanCheck(Node):
    def __init__(self):
        super().__init__("scan_check")
        self.scan = None
        self.create_subscription(LaserScan, "scan", self.scan_callback, qos_profile_sensor_data)
        self.create_timer(1.0, self.print_info)  # print once per second

    def scan_callback(self, msg):
        self.scan = msg

    def print_info(self):
        if self.scan is None:
            self.get_logger().info("No scan received yet...")
            return

        s = self.scan
        closest_range = float("inf")
        closest_angle = None
        for i, r in enumerate(s.ranges):
            if math.isfinite(r) and s.range_min <= r <= s.range_max and r < closest_range:
                closest_range = r
                closest_angle = math.degrees(s.angle_min + i * s.angle_increment)

        self.get_logger().info(
            f"readings: {len(s.ranges)} | "
            f"angle range: {math.degrees(s.angle_min):.0f} to {math.degrees(s.angle_max):.0f} deg"
        )
        if closest_angle is None:
            self.get_logger().info("Nothing detected")
        else:
            self.get_logger().info(
                f"CLOSEST object: {closest_range:.2f} m at angle {closest_angle:.0f} deg"
            )


def main(args=None):
    rclpy.init(args=args)
    node = ScanCheck()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
