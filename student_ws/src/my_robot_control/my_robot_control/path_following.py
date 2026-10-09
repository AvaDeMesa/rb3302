#!/usr/bin/env python3
"""
path_following.py

Based on the structure of path_planning.py, connected to the Wave Rover
using the working connection code from move_distance.py.

Right now it has ONE working function (move_distance) to test the connection:
it rotates by an angle, then drives forward a distance, and stops early if
the lidar sees something in front.

Run it like this:
    ros2 run my_robot_control path_following <distance_m> [angle_deg]
Example (turn 90 deg left, then drive 1.5 m):
    ros2 run my_robot_control path_following 1.5 90
"""

import math
import sys
import heapq

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.logging import set_logger_level, LoggingSeverity
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

# Pillow is only needed to draw the A* grid picture. Skip it if not installed.
try:
    from PIL import Image
except ImportError:
    Image = None


np.set_printoptions(
    2, suppress=True, threshold=np.inf
)  # Print numpy arrays to 2 d.p., no scientific notation, do not truncate

set_logger_level("path_following", level=LoggingSeverity.DEBUG)  # or LoggingSeverity.INFO

# ------------------------------------------------------------------
# SETTINGS - change these to suit your robot
# ------------------------------------------------------------------
is_simulation = False   # Gazebo simulation is kept below but NOT active. Leave False for the real robot.

# ---- Speed settings (easy to change) ----
MAX_TRANSLATE_VELOCITY = 0.3   # m/s. Hard cap on forward/back speed (from path_planning.py)
MAX_TURN_VELOCITY = 0.6        # rad/s. Hard cap on turning speed (2 x the translate cap, as in path_planning.py)
FORWARD_SPEED = 0.2            # m/s. Speed used by move_distance (must be <= MAX_TRANSLATE_VELOCITY)
TURN_SPEED = 0.5               # rad/s. Max turning speed used by move_distance (must be <= MAX_TURN_VELOCITY)

# ---- Safety and lidar settings ----
STOP_DISTANCE = 0.30           # m. Stop if something is closer than this in front
FRONT_ANGLE_DEG = 180          # which lidar angle is "front" on this robot
FRONT_HALF_WIDTH_DEG = 15      # look this many degrees to each side of "front"
ANGLE_TOLERANCE_DEG = 2.0      # stop turning when within this many degrees of the target
SIDE_HALF_WIDTH_DEG = 20       # same idea for the left, right and back readings

# ---- Control loop and PD gains (tune on the real robot: Kp first, then Kd) ----
CONTROL_DT = 0.05              # s. Timer period (20 Hz). Used by the timer AND the PD maths
ANGLE_UNIT_NOTE = "all PD angle errors are in RADIANS"

TURN_KP = 1.5                  # (rad/s) per rad of heading error
TURN_KD = 0.0                  # damping, raise if the turn overshoots
TURN_MIN_SPEED = 0.3           # rad/s. Below this the motors stall

DRIVE_KP = 1.0                 # (m/s) per metre of distance remaining
DRIVE_KD = 0.0                 # usually not needed for driving
DRIVE_MIN_SPEED = 0.08         # m/s. Below this the motors stall
DISTANCE_TOLERANCE = 0.10      # m. Stop when this close to the target distance

HEADING_KP = 1.0               # (rad/s) per rad. Keeps the robot straight while driving
HEADING_KD = 0.0

WAYPOINT_TOLERANCE = 0.05      # m. Waypoint counts as reached within this distance



# ---- Topics ----
if is_simulation:
    ODOM_TOPIC = "odom"        # Gazebo odometry (not active)
else:
    ODOM_TOPIC = "odom/raw"    # published by waverover_driver.py
SCAN_TOPIC = "scan"            # published by rplidar_ros

# ---- A* map settings (for later) ----
occupancy_grid_resolution = 0.2

sim_goal_list = [(3.4, -3.6), (3.2, 0.2), (2.4, -3.6), (-0.4, -3.8)]
sim_grid_start = (-1.0, -5.0)

irl_goal_list = [(2.1, -1.7), (2.3, -0.3), (1.5, -1.7), (0.3, -1.7)]
irl_grid_start = (0.1, -1.9)


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


class WaypointNode(Node):
    # Define parameters (kept from path_planning.py, used later for A* path following)
    goal_reached = True      # True when the robot has reached its goal (default True for 1st goal)
    prev_wp = (0, 0)         # position of previous waypoint, updated at each waypoint
    target_direction = 'up'  # direction of waypoint relative to the start, updated as we go
    prev_ET = 0              # E(t-1) for PD linear velocity
    prev_SA = 0              # E(t-1) for PD angular velocity
    direction_dict = {'up': 0, 'left': 90, 'right': -90, 'down': 180}  # robot target angle (degrees) for each direction
    correct_direction = False  # True once the robot is facing the target direction
    waypoints = []           # to be edited after each A*
    map_array = None

    # Starting position for the 1st goal A* (used for future calculations)
    if is_simulation:
        startingx = sim_grid_start[0]
        startingy = sim_grid_start[1]
    else:
        startingx = irl_grid_start[0]
        startingy = irl_grid_start[1]

    prev_goal = (int(-startingx / 0.2), int(-startingy / 0.2))  # for use in A*

    '''Node to move the Wave Rover. Pose comes from wheel odometry, lidar from the RPLidar.'''
    def __init__(self, target_distance=None, target_angle_deg=0.0,
                 map_array=None, goal_list=None, is_simulation=False, auto_mode=False):
        super().__init__('path_following')

        self.auto_mode = auto_mode        # True = run auto_move_step() instead of move_distance_step()
        self.auto_state = "START"         # your own state machine variable for auto mode

        self.get_logger().info("Starting WaypointNode")

        self.is_simulation = is_simulation

        # ---------------- Variables for the move_distance test ----------------
        self.target_distance = target_distance
        self.target_angle = math.radians(target_angle_deg)
        self.state = "ROTATE"        # "ROTATE" first, then "MOVE"
        self.done = False            # becomes True when the job is finished

        # ---------------- Variables that hold the robot's data ----------------
        # Position and heading (filled in by odom_callback)
        self.x = None
        self.y = None
        self.yaw = None
        # Where the robot was when this program began
        self.start_x = None
        self.start_y = None
        self.start_yaw = None
        # The latest lidar scan (filled in by scan_callback)
        self.scan = None

        # ---------------- For A* later ----------------
        self.goal_list = goal_list if goal_list is not None else []
        self.map_array = map_array

        # ---------------- ROS connections ----------------
        # Publisher: how we SEND movement commands (same topic as keyboard teleop)
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)

        # Subscribers: how we RECEIVE data
        self.create_subscription(Odometry, ODOM_TOPIC, self.odom_callback, 10)
        # Lidars need "sensor data" QoS, otherwise messages may not arrive
        self.create_subscription(
            LaserScan, SCAN_TOPIC, self.scan_callback, qos_profile_sensor_data
        )

        # Timer: runs timer_callback() 20 times per second
        self.timer = self.create_timer(CONTROL_DT, self.timer_callback)

        if self.target_distance is not None:
            self.get_logger().info(
                f"Target: rotate {target_angle_deg} deg, then move {self.target_distance} m"
            )

    # ==================================================================
    # CALLBACKS - run automatically whenever new data arrives
    # ==================================================================
    def odom_callback(self, msg):
        '''Store the robot's position (x, y) and heading (yaw, radians) from odometry.'''
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = quaternion_to_yaw(msg.pose.pose.orientation)

        # Remember where we started (only the first time)
        if self.start_x is None:
            self.start_x = self.x
            self.start_y = self.y
            self.start_yaw = self.yaw

    def scan_callback(self, msg):
        '''Keep the whole scan message. Useful fields:
            msg.ranges          -> list of distances in metres
            msg.angle_min       -> angle of the first reading (radians)
            msg.angle_increment -> angle between readings (radians)'''
        self.scan = msg

    # ==================================================================
    # SENSOR FUNCTIONS - read information (add your own here)
    # ==================================================================
    def get_pose(self):
        '''Pose as (x, y, heading in DEGREES), the same format path_planning.py used.'''
        return np.array((self.x, self.y, math.degrees(self.yaw)))

    def get_distance_travelled(self):
        '''Straight-line distance from the starting point, in metres.'''
        return math.hypot(self.x - self.start_x, self.y - self.start_y)

    def get_angle_turned(self):
        '''How far the robot has turned since the start, in radians (+ve = left).'''
        return normalize_angle(self.yaw - self.start_yaw)

    def get_region_distance(self, center_deg=0.0, half_width_deg=15.0):
        '''Closest object in a slice of the lidar scan, in metres.
        center_deg is relative to the robot's front (0 = front, +90 = left,
        -90 = right, 180 = back). Returns inf if nothing is seen.'''
        closest = float("inf")
        for i, r in enumerate(self.scan.ranges):
            if not math.isfinite(r):
                continue
            if r < self.scan.range_min or r > self.scan.range_max:
                continue
            lidar_deg = math.degrees(self.scan.angle_min + i * self.scan.angle_increment)
            rel_deg = (lidar_deg - FRONT_ANGLE_DEG + 180.0) % 360.0 - 180.0   # relative to front
            diff = (rel_deg - center_deg + 180.0) % 360.0 - 180.0
            if abs(diff) <= half_width_deg and r < closest:
                closest = r
        return closest

    def get_left_distance(self):
        return self.get_region_distance(+90.0, SIDE_HALF_WIDTH_DEG)

    def get_right_distance(self):
        return self.get_region_distance(-90.0, SIDE_HALF_WIDTH_DEG)

    def get_back_distance(self):
        return self.get_region_distance(180.0, SIDE_HALF_WIDTH_DEG)

    def get_front_distance(self):
        '''Closest object in front of the robot, in metres.
        Returns infinity if nothing is seen.'''
        return self.get_region_distance(0.0, FRONT_HALF_WIDTH_DEG)

    def obstacle_ahead(self, threshold=STOP_DISTANCE):    #This for detecting obstacles at the front, change according to need
        return self.get_front_distance() < threshold

    # ==================================================================
    # MOVEMENT FUNCTIONS - make the robot move (add your own here)
    # ==================================================================
    def move(self, linear=0.0, angular=0.0):
        '''Publishes a Twist message. The Wave Rover can only drive forward/back and turn.
        linear  = forward speed in m/s  (negative = backwards)
        angular = turning speed in rad/s (positive = left/anticlockwise)
        Both are capped by the speed settings at the top of the file.'''
        twist_msg = Twist()
        linear = max(-MAX_TRANSLATE_VELOCITY, min(MAX_TRANSLATE_VELOCITY, linear))
        angular = max(-MAX_TURN_VELOCITY, min(MAX_TURN_VELOCITY, angular))
        twist_msg.linear.x = float(linear)
        twist_msg.angular.z = float(angular)
        self.publisher_.publish(twist_msg)

    def stop(self):
        self.move(0.0, 0.0)


#---------------------------

    def turn_by_step(self, angle_deg):
        '''Turn by angle_deg relative to the heading when reset_reference() was called.
        Call every tick. Returns True when finished.'''
        target = math.radians(angle_deg)
        error = normalize_angle(target - self.get_angle_turned())
        if abs(error) <= math.radians(ANGLE_TOLERANCE_DEG):
            self.stop()
            return True
        speed = max(0.3, min(TURN_SPEED, 1.5 * abs(error)))
        self.move(0.0, (1.0 if error > 0 else -1.0) * speed)
        return False

    def drive_step(self, distance_m, speed=FORWARD_SPEED, check_obstacle=True):
        '''Drive forward distance_m from where reset_reference() was called.
        Call every tick. Returns True when finished (or blocked by an obstacle).'''
        if check_obstacle and self.obstacle_ahead():
            self.stop()
            self.get_logger().warn("Obstacle ahead, stopping")
            return True
        if self.get_distance_travelled() >= distance_m:
            self.stop()
            return True
        self.move(speed, 0.0)
        return False

#---------------------------

    def reset_reference(self):
        '''Make "distance travelled" and "angle turned" start from zero HERE.
        Call this whenever you start a new turn or drive.'''
        self.start_x = self.x
        self.start_y = self.y
        self.start_yaw = self.yaw

    def auto_move_step(self):
        '''==== YOUR AUTOMATED MOVEMENT CODE GOES HERE ====
        Called 20 times per second from timer_callback when auto_mode is True.
        Do NOT use while/sleep loops. Use self.auto_state to remember where you are.

        Available:
        Sensors : self.get_pose()                  -> (x, y, heading_deg)
                    self.get_front_distance()        -> metres
                    self.get_left_distance(), get_right_distance(), get_back_distance()
                    self.get_region_distance(center_deg, half_width_deg)
                    self.get_all_distances()         -> dict
                    self.obstacle_ahead()            -> True/False
        Movement: self.move(linear_m_s, angular_rad_s)   (+angular = left)
                    self.stop()
                    self.turn_by_step(deg)  /  self.drive_step(metres)
                    self.reset_reference()
        Finish  : self.done = True
        '''
        # Example state machine skeleton (delete or replace):
        if self.auto_state == "START":
            self.reset_reference()
            self.auto_state = "DRIVE"

        elif self.auto_state == "DRIVE":
            if self.drive_step(1.0):
                self.reset_reference()
                self.auto_state = "TURN"

        elif self.auto_state == "TURN":
            if self.turn_by_step(90):
                self.reset_reference()
                self.auto_state = "FINISHED"

        elif self.auto_state == "FINISHED":
            self.stop()
            self.done = True

    def move_distance_step(self):
        '''ONE working function: rotate by an angle, then drive a distance.
        Called 20 times per second by timer_callback (no blocking loops).'''

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

        # ---- STEP 2: MOVE ----
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

    def set_waypoints(self, waypoints: list):
        '''Set new waypoints when a goal has been reached (for A* later)'''
        self.goal_reached = False
        self.waypoints = waypoints
        self.current_waypoint_idx = 0

    # ==================================================================
    # MAIN LOGIC - runs 20 times per second
    # ==================================================================
    def timer_callback(self):
        """Controller loop. Insert path planning and PID control logic here"""
        if self.done:
            return

        # Wait until we have received both odometry and a lidar scan
        if self.x is None or self.scan is None:
            self.get_logger().info("Waiting for odom and scan...", throttle_duration_sec=2.0)
            return

        self.get_logger().debug(f"Pose: {self.get_pose()}")

        # Test function: rotate then move.
        # LATER: replace this with self.plan_next_goal() / self.follow_waypoints()
        if self.auto_mode:
            self.auto_move_step()
        elif self.target_distance is not None:
            self.move_distance_step()
        

    # ==================================================================
    # A* PATH FOLLOWING (kept from path_planning.py, NOT used yet)
    # To use it later, call these from timer_callback instead of move_distance_step.
    # ==================================================================
    def plan_next_goal(self):
        '''Plan an A* path to the next goal in goal_list. Needs self.map_array.'''
        if self.map_array is None:
            self.get_logger().warn("No map loaded, cannot plan")
            return
        if len(self.goal_list) < 1:  # when no more goals in goal list
            print("all goal reached!")
            return
        goal = self.goal_list[0]  # get next goal
        grid_goal = (int((goal[0] - self.startingx) / 0.2), int((goal[1] - self.startingy) / 0.2))  # goal in grid coords
        print(f"grid goal is {grid_goal}")

        grid = Grid(self.map_array, (0, 0), (-1, -1))  # initialise grid
        a_star_path = grid.astar(self.prev_goal, grid_goal)  # get A* path to goal
        waypoints = grid.waypoint(a_star_path)  # extract waypoints only
        print("Waypoints:", waypoints)

        self.set_waypoints(waypoints)  # set waypoints into list
        grid.draw_grid_map(waypoints, a_star_path)  # show the grid

        self.prev_goal = grid_goal  # update the prev goal
        self.goal_list.pop(0)       # remove goal once A* is planned

    def follow_waypoints(self):
        '''PD controller: turn to face the next waypoint, then drive to it.
        Uses only forward speed and turning (Wave Rover cannot move sideways).'''
        if self.goal_reached:
            return
        if len(self.waypoints) == 0:  # if no more waypoints
            print('Goal Reached')
            self.goal_reached = True
            return

        pose = self.get_pose()  # (x, y, heading in degrees)

        # Global position of target waypoint from grid values
        waypoint_x = 0.2 * self.waypoints[0][0] + self.startingx
        waypoint_y = 0.2 * self.waypoints[0][1] + self.startingy
        target_waypoint = (waypoint_x, waypoint_y)

        # Difference in x & y from waypoint
        dx = waypoint_x - pose[0]
        dy = waypoint_y - pose[1]

        # Define target direction
        if abs(waypoint_x - self.prev_wp[0]) > 0.5:
            self.target_direction = 'up' if dx > 0 else 'down'
        elif abs(waypoint_y - self.prev_wp[1]) > 0.5:
            self.target_direction = 'left' if dy > 0 else 'right'

        # Get error E(t) based on target direction
        if self.target_direction == 'up':
            E_t = dx
        elif self.target_direction == 'down':
            E_t = -dx
        elif self.target_direction == 'left':
            E_t = dy
        else:
            E_t = -dy

        if abs(E_t) < 0.15:  # 0.15 m is where the robot accepts it has reached the waypoint
            self.stop()
            print('waypoint reached!')
            self.waypoints.pop(0)  # remove waypoint
            self.correct_direction = False
            self.prev_ET = 0
            self.prev_wp = target_waypoint  # update prev waypoint
            return

        if self.correct_direction:  # facing the right way: PD for forward speed
            Kp_v = 2.0
            Kd_v = 1.0
            dt = 0.05

            v_error = (E_t - self.prev_ET) / dt
            Vx = Kp_v * E_t + Kd_v * v_error

            self.move(Vx, 0.0)  # move() caps the speed
            self.prev_ET = E_t
            print(f'moving towards {self.target_direction} at {Vx:.2f}')
        else:
            target_angle = self.direction_dict[self.target_direction]
            shortest_angle = ((target_angle - pose[2] + 180) % 360) - 180  # shortest turn, degrees

            if abs(shortest_angle) < 0.9:  # within 0.9 degrees of target
                self.stop()
                self.correct_direction = True
                self.prev_SA = 0
                print("direction reached")
            else:
                # PD control for turning
                Kp_w = 1.5
                Kd_w = 2.0
                dt = 0.05

                w_error = (shortest_angle - self.prev_SA) / dt
                Wz = Kp_w * shortest_angle + Kd_w * w_error

                self.move(0.0, Wz)  # move() caps the turn speed
                self.prev_SA = shortest_angle
                print(f'turning towards {self.target_direction} at {Wz:.2f}')


class CellDetail():
    '''Cell details for A* algorithm'''
    def __init__(self):
        self.parent_i = 0  # Parent cell's row index
        self.parent_j = 0  # Parent cell's column index
        self.f = float('inf')  # Total cost function
        self.g = float('inf')  # Cost from start to current cell
        self.h = 0  # Heuristic cost from current cell to end


class Grid():
    '''
    Grid class to use with occupancy grid. Contains the following functions:
        check_grid_validity : Uses flood fill to check if there's a valid path from start to goal position
        draw_grid_map : Creates a colour image of the grid, as well as waypoints and full solution path if given.

    __init__ input Args:
        grid_array : 2D numpy array representing the occupancy grid
        starting_position : tuple of starting indices within the numpy array
        goal_position : tuple of goal indices within the numpy array. Works with negative indices as well
    '''
    def __init__(self, grid_array: np.array = np.array([]), starting_position: tuple = (0, 0), goal_position: tuple = (-1, -1)):
        self.grid = grid_array
        self.shape = self.grid.shape
        self.starting_position = starting_position

        # If goal position given with negative index, need to convert to +ve
        if goal_position[0] < 0:
            goal_x = self.shape[0] + goal_position[0]
        else:
            goal_x = goal_position[0]
        if goal_position[1] < 0:
            goal_y = self.shape[1] + goal_position[1]
        else:
            goal_y = goal_position[1]

        self.goal_position = (goal_x, goal_y)

    def check_grid_validity(self):
        '''Use flood fill to check if there's a viable path between start and goal positions'''
        grid = self.grid.copy()
        flood_stack = [(self.goal_position[0], self.goal_position[1])]
        while flood_stack:
            tile = flood_stack[0]
            del flood_stack[0]
            try:
                next_tile = (tile[0] + 1, tile[1])
                if grid[next_tile] == 0:
                    grid[next_tile] = 1
                    flood_stack.append(next_tile)
            except:pass
            try:
                next_tile = (tile[0] - 1, tile[1])
                if grid[next_tile] == 0:
                    grid[next_tile] = 1
                    flood_stack.append(next_tile)
            except:pass
            try:
                next_tile = (tile[0], tile[1] + 1)
                if grid[next_tile] == 0:
                    grid[next_tile] = 1
                    flood_stack.append(next_tile)
            except:pass
            try:
                next_tile = (tile[0], tile[1] - 1)
                if grid[next_tile] == 0:
                    grid[next_tile] = 1
                    flood_stack.append(next_tile)
            except:pass
        if grid[self.starting_position] == 1:  # flood reached the start from the goal
            return True
        else:
            return False

    def draw_grid_map(self, waypoints: list = (), path: list = (), obstacle_threshold: float = 50):
        '''Creates an image of the maze and path taken. Maze walls in blue, empty space in white, path taken in green and waypoints in red

        Args:
            waypoints : list of tuple coordinates (grid indices) for the waypoints. Shown in red, takes precedence over path
            path : list of tuple coordinates (grid indices) forming the solution path. Shown in green
            obstacle_threshold : threshold for whether a grid cell is considered occupied
        '''
        if Image is None:
            print("Pillow (PIL) is not installed, cannot draw the grid map")
            return

        image_grid = np.ones((self.grid.shape[0], self.grid.shape[1], 3), dtype=np.uint8)
        image_grid[self.grid <= obstacle_threshold] = (255, 255, 255)
        image_grid[self.grid > obstacle_threshold] = (0, 0, 255)

        for x, y in path:
            image_grid[x][y] = (0, 255, 0)

        for point in waypoints:
            image_grid[point] = (255, 0, 0)

        image_grid = np.flip(image_grid, axis=1)[::-1]
        img = Image.fromarray(image_grid, 'RGB')

        # Resize image
        base_width = 500
        wpercent = (base_width / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((base_width, hsize), Image.Resampling.NEAREST)

        img.show()
        img.save('grid_map.png')
        print("Grid map saved as 'grid_map.png'")

    def waypoint(self, Apath):
        '''Keep only the cells where the path changes direction (plus start and end)'''
        waypoints = []
        waypoints.append(Apath[0])
        for i in range(1, len(Apath) - 1):
            current_dir = (Apath[i][0] - Apath[i - 1][0], Apath[i][1] - Apath[i - 1][1])
            next_dir = (Apath[i + 1][0] - Apath[i][0], Apath[i + 1][1] - Apath[i][1])
            if current_dir != next_dir:
                waypoints.append(Apath[i])
        waypoints.append(Apath[-1])
        return waypoints

    def astar(self, start, end):
        grid = self.grid

        # Check if cell is within the grid
        def is_within(row, col):
            return (row >= 0) and (row < grid.shape[0]) and (col >= 0) and (col < grid.shape[1])

        # Check if a cell is unblocked
        def is_unblocked(row, col):
            if grid[row][col] < 50:
                return True
            else:
                return False

        # Check if a cell is the goal
        def is_goal(row, col, goal):
            return row == goal[0] and col == goal[1]

        # Calculate heuristic (Manhattan distance)
        def calculate_heuristic(row, col, goal):
            return abs(row - goal[0]) + abs(col - goal[1])

        # Trace the path from start to goal
        def trace_path(cell_details, goal):
            path = []
            row = goal[0]
            col = goal[1]

            # Trace the path from destination to source using parent cell
            while not (cell_details[row][col].parent_i == row and cell_details[row][col].parent_j == col):
                path.append((row, col))
                temp_row = cell_details[row][col].parent_i
                temp_col = cell_details[row][col].parent_j
                row = temp_row
                col = temp_col

            # Add the source cell to the path
            path.append((row, col))
            # Reverse the path to get the path from source to destination
            path.reverse()

            for i in path:
                print("->", i, end=" ")
            print()
            return path

        # A* algorithm

        # Check if the src and goal are valid
        if not is_within(start[0], start[1]) or not is_within(end[0], end[1]):
            print("Source or the goal is invalid")
            return

        # Check if the src and goal are unblocked
        if not is_unblocked(start[0], start[1]) or not is_unblocked(end[0], end[1]):
            print("Source or the goal is blocked")
            return

        # Check if we are already at the destination
        if is_goal(start[0], start[1], end):
            print("We are already at the destination")
            return

        num_rows = self.grid.shape[0]
        num_cols = self.grid.shape[1]
        # Create a closed list to keep track of visited cells
        closed_list = []
        for row in range(num_rows):
            row_list = []
            for col in range(num_cols):
                row_list.append(False)
            closed_list.append(row_list)

        # Create a cell details list to store details of each cell
        cell_details = []
        for row in range(num_rows):
            row_details = []
            for col in range(num_cols):
                row_details.append(CellDetail())
            cell_details.append(row_details)

        # Initialize the parameters of the starting node
        i, j = start[0], start[1]
        cell_details[i][j].f = 0.0
        cell_details[i][j].g = 0.0
        cell_details[i][j].h = 0.0
        cell_details[i][j].parent_i = i
        cell_details[i][j].parent_j = j

        # Create an open list of neighbouring cells with the start cell
        open_list = []
        heapq.heappush(open_list, (0.0, i, j))

        # Initialize if end is found
        found_goal = False

        # Main loop of A* algorithm
        while len(open_list) > 0:

            # Choose the cell with the lowest f value from open list
            chosen_cell = heapq.heappop(open_list)

            # Mark the cell as visited in closed list
            i = chosen_cell[1]
            j = chosen_cell[2]
            closed_list[i][j] = True

            # For each direction, check possible neighbouring cells
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
            for dir in directions:
                new_i = i + dir[0]
                new_j = j + dir[1]

                # If the successor is valid, unblocked, and not visited
                if is_within(new_i, new_j) and is_unblocked(new_i, new_j) and not closed_list[new_i][new_j]:
                    # If the successor is the goal, we are done
                    if is_goal(new_i, new_j, end):

                        # Set the parent of the goal cell
                        cell_details[new_i][new_j].parent_i = i
                        cell_details[new_i][new_j].parent_j = j
                        print("Goal found!")

                        # Trace and print the path from source to destination
                        path = trace_path(cell_details, end)
                        found_goal = True
                        print(path)
                        return path
                    else:
                        # Calculate g, h, f values
                        g_new = cell_details[i][j].g + 1.0
                        h_new = calculate_heuristic(new_i, new_j, end)
                        f_new = g_new + h_new

                        # If the cell is not in the open list or has a lower f value, update it and add to open list
                        if cell_details[new_i][new_j].f == float('inf') or cell_details[new_i][new_j].f > f_new:

                            heapq.heappush(open_list, (f_new, new_i, new_j))

                            cell_details[new_i][new_j].f = f_new
                            cell_details[new_i][new_j].g = g_new
                            cell_details[new_i][new_j].h = h_new
                            cell_details[new_i][new_j].parent_i = i
                            cell_details[new_i][new_j].parent_j = j

        if not found_goal:
            print("Failed to find the Goal Cell")
            return None


def main(args=None):
    print("Starting path following")

    # Read the distance (and optional angle) typed after the command
    if len(sys.argv) < 2:
        print("Usage: ros2 run my_robot_control path_following <distance_m> [angle_deg]")
        print("   or: ros2 run my_robot_control path_following auto")
        return

    auto_mode = sys.argv[1].lower() == "auto"
    distance, angle_deg = None, 0.0
    if not auto_mode:
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

    # LATER (for A*): load your map and pass it in, for example:
    #   import os
    #   filepath = os.path.dirname(os.path.realpath(__file__))
    #   map_array = np.load(filepath + '/ca2_irl_map.npy', allow_pickle=True)
    #   waypoint = WaypointNode(distance, angle_deg, map_array, irl_goal_list, is_simulation)
    # For Gazebo, use ca2_sim_map.npy with sim_goal_list and set is_simulation = True at the top.
    waypoint = WaypointNode(distance, angle_deg, is_simulation=is_simulation,
                            auto_mode=auto_mode)

    try:
        # Keep running until the job is done
        while rclpy.ok() and not waypoint.done:
            rclpy.spin_once(waypoint, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            waypoint.stop()  # always leave the robot stopped
        except Exception:
            pass
        waypoint.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    print("Shutting down")


if __name__ == '__main__':
    main()