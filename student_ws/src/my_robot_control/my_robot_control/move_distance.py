#!/usr/bin/env python3
"""
move_distance.py  -  BASE CODE

Moves the robot forward by the distance you give it (in metres).
It stops early if the lidar sees something in front of the robot.

Run it like this:
    ros2 run my_robot_control move_distance 1.5
"""

import math
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

# ------------------------------------------------------------------
# SETTINGS - change these to suit your robot
# ------------------------------------------------------------------
FORWARD_SPEED = 0.2        # m/s. Teleop uses 0.4 by default, 0.2 is gentler
STOP_DISTANCE = 0.30       # m. Stop if something is closer than this in front
FRONT_ANGLE_DEG = 180      # which lidar angle is "front"? Find out with scan_check
FRONT_HALF_WIDTH_DEG = 15  # look this many degrees to each side of "front"
TURN_SPEED = 0.5           # rad/s, maximum turning speed


ANGLE_TOLERANCE_DEG = 2.0  # stop turning when within this many degrees of the target

ODOM_TOPIC = "odom/raw"    # published by waverover_driver.py
SCAN_TOPIC = "scan"        # published by rplidar_ros


def quaternion_to_yaw(q):
    """Turn the odometry orientation (a quaternion) into a heading in radians.
    0 = facing the way the robot started, +ve = turned anticlockwise (left)."""
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )

def normalize_angle(angle):
    """Keep an angle between -pi and +pi (so 350 deg becomes -10 deg)."""
    return math.atan2(math.sin(angle), math.cos(angle))


class MoveDistance(Node):
    def __init__(self, target_distance, target_angle_deg):
        super().__init__("move_distance")
        self.target_distance = target_distance
        self.target_angle = math.radians(target_angle_deg)   # NEW
        self.start_yaw = None                                # NEW
        self.state = "ROTATE"                                # NEW: "ROTATE" then "MOVE"


        # ---------------- Variables that hold the robot's data ----------------
        # Position and heading (filled in by odom_callback)
        self.x = None
        self.y = None
        self.yaw = None
        # Where the robot was when this program began
        self.start_x = None
        self.start_y = None
        # The latest lidar scan (filled in by scan_callback)
        self.scan = None
        # Becomes True when the job is finished
        self.done = False

        # ---------------- ROS connections ----------------
        # Publisher: how we SEND movement commands (same topic as keyboard teleop)
        self.cmd_pub = self.create_publisher(Twist, "cmd_vel", 10)

        # Subscribers: how we RECEIVE data
        self.create_subscription(Odometry, ODOM_TOPIC, self.odom_callback, 10)
        # Lidars normally need "sensor data" QoS, otherwise messages may not arrive
        self.create_subscription(
            LaserScan, SCAN_TOPIC, self.scan_callback, qos_profile_sensor_data
        )

        # Timer: runs control_loop() 20 times per second
        self.create_timer(0.05, self.control_loop)

        self.get_logger().info(f"Target distance: {self.target_distance} m")

    # ==================================================================
    # CALLBACKS - run automatically whenever new data arrives
    # ==================================================================
    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = quaternion_to_yaw(msg.pose.pose.orientation)

        # Remember where we started (only the first time)
        if self.start_x is None:
            self.start_x = self.x
            self.start_y = self.y
            self.start_yaw = self.yaw      # NEW

    def scan_callback(self, msg):
        # Keep the whole scan message. Useful fields:
        #   msg.ranges          -> list of distances in metres
        #   msg.angle_min       -> angle of the first reading (radians)
        #   msg.angle_increment -> angle between readings (radians)
        self.scan = msg

    # ==================================================================
    # SENSOR FUNCTIONS - read information (add your own here)
    # ==================================================================
    def get_distance_travelled(self):
        """Straight-line distance from the starting point, in metres."""
        return math.hypot(self.x - self.start_x, self.y - self.start_y)

    def get_front_distance(self):
        """Closest object in front of the robot, in metres.
        Returns infinity if nothing is seen."""
        closest = float("inf")
        for i, r in enumerate(self.scan.ranges):
            # Ignore bad readings (inf, nan, too close, too far)
            if not math.isfinite(r):
                continue
            if r < self.scan.range_min or r > self.scan.range_max:
                continue

            # Work out which direction this reading points in
            angle_deg = math.degrees(self.scan.angle_min + i * self.scan.angle_increment)
            # Difference from "front", wrapped to the range -180..180
            diff = (angle_deg - FRONT_ANGLE_DEG + 180.0) % 360.0 - 180.0

            if abs(diff) <= FRONT_HALF_WIDTH_DEG and r < closest:
                closest = r
        return closest
    def get_angle_turned(self):
        """How far the robot has turned since the start, in radians (+ve = left)."""
        return normalize_angle(self.yaw - self.start_yaw)

    # ==================================================================
    # MOVEMENT FUNCTIONS - make the robot move (add your own here)
    # ==================================================================
    def move(self, linear=0.0, angular=0.0):
        """linear  = forward speed in m/s  (negative = backwards)
        angular = turning speed in rad/s (positive = left/anticlockwise)"""
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.cmd_pub.publish(msg)

    def stop(self):
        self.move(0.0, 0.0)

    # ==================================================================
    # MAIN LOGIC - runs 20 times per second
    # ==================================================================
    def control_loop(self):
        if self.done:
            return

        # Wait until we have received both odometry and a lidar scan
        if self.x is None or self.scan is None:
            self.get_logger().info("Waiting for odom and scan...", throttle_duration_sec=2.0)
            return

        # ---- STEP 1: ROTATE ----
        if self.state == "ROTATE":
            error = normalize_angle(self.target_angle - self.get_angle_turned())

            if abs(error) <= math.radians(ANGLE_TOLERANCE_DEG):
                self.stop()
                # Reset the start point so distance is measured from here
                self.start_x = self.x
                self.start_y = self.y
                self.get_logger().info("Rotation done, now moving forward")
                self.state = "MOVE"
            else:
                # Turn left if error is positive, right if negative.
                # Slow down as we get close so we don't overshoot.
                speed = max(0.3, min(TURN_SPEED, 1.5 * abs(error)))
                direction = 1.0 if error > 0 else -1.0
                self.move(0.0, direction * speed)
            return

# ---- STEP 2: MOVE (your existing code continues below) ----

        travelled = self.get_distance_travelled()
        front = self.get_front_distance()

        # 1) Safety first: obstacle in front?
        if front < STOP_DISTANCE:
            self.stop()
            self.get_logger().warn(
                f"Obstacle {front:.2f} m ahead. Stopped after {travelled:.2f} m."
            )
            self.done = True

        # 2) Have we gone far enough?
        elif travelled >= self.target_distance:
            self.stop()
            self.get_logger().info(f"Reached target. Travelled {travelled:.2f} m.")
            self.done = True

        # 3) Otherwise keep driving forward
        else:
            self.move(FORWARD_SPEED, 0.0)


def main(args=None):
    # Read the distance typed after the command
    if len(sys.argv) < 2:
        print("Usage: ros2 run my_robot_control move_distance <distance_in_metres>")
        return
    # replace the "distance = float(sys.argv[1])" part with:
    try:
        distance = float(sys.argv[1])
        angle_deg = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    except ValueError:
        print("Distance and angle must be numbers, e.g. 1.5 90")
        return
    if distance <= 0:
        print("Distance must be greater than 0")
        return

    rclpy.init(args=args)
    node = MoveDistance(distance, angle_deg)

    try:
        # Keep running until the job is done
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.stop()  # always leave the robot stopped
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
