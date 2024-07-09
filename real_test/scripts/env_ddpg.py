#!/usr/bin/env python
# coding=utf-8

import rospy
import numpy as np
import math
import time
import random
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry


class Env():
    def __init__(self):
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=1)  # Publish linear velocity and angular velocity
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)  # Subscribe odometry information 
        self.sub_scan = rospy.Subscriber('scan', LaserScan, self.getScans)  # Subscribe laser data 
        #define parameters of APF
        self.k_att_1 = 10.0
        self.k_att_2 = 100.0
        self.k_rep = 0.5
        self.m_inner = 0.8
        self.m_outer = 2
        self.n = 2
        self.d_rep = 1.0 
        self.ds = 0.1  # Goal threshold
        self.dcll = -0.12  # Obstacle threshold
        self.state_dim = 14  # 10 potential points, angle2goal, d_goal, yaw, angular_velocity
        self.action_dim = 1  # angular velocity 
        self.state = None  # Initialize state 
        self.goal = [3.5, 0.0]  # set goal position
        self.position = Point()  # Initialize agent position 
        self.angle = 0.0  # Yaw of agent
        self.angle2goal = 10.0  # Angle between agent and goal 
        self.distance2goal = 10.0  # Distance between agent and goal 
        self.linear_velocity = 0.0  
        self.angular_velocity = 0.0
        self.angular_velocity_low = -1.5  # Threshold of angular velocity
        self.angular_velocity_high = 1.5 
        self.poential = 0.0  # Potential of agent's position
        self.scans = []  # Re-oriented laser data 
        self.start_time = 0.0  # Start time of this episode 
        self.rate = rospy.Rate(10)  # Publish rate during training 
        self.i_ep = 0  # Current episode 
        self.end = ''  # Mark the end type of the episode 

    def getOdometry(self, odom):
        """ Decode the odometry information 
        """
        self.position = odom.pose.pose.position  # Get odometry position 
        orientation = odom.pose.pose.orientation
        [x, y, z, w] = [orientation.x, orientation.y, orientation.z, orientation.w]
        # _, _, yaw = euler_from_quaternion([x, y, z, w])   # yaw [-pi, pi]
        yaw = math.atan2(2*(w*z+x*y),1-2*(z*z+y*y))  # Get the yaw
        self.angle = yaw
        self.angle2goal = math.atan2(self.goal[1] - self.position.y, self.goal[0] - self.position.x) 
        self.distance2goal = ((self.goal[0] - self.position.x) ** 2 + (self.goal[1] - self.position.y) ** 2) ** 0.5
        if self.distance2goal < self.ds and self.end != 'end':  # Test whether reach the goal 
            self.end = 'Reach the goal!'

    def getScans(self, scan_msgs):
        """ Decode laser data to global coordinate 
        """
        ranges = list(scan_msgs.ranges)
        theta_min = scan_msgs.angle_min
        theta_max = scan_msgs.angle_max
        theta_inc = scan_msgs.angle_increment
        x_P = self.position.x
        y_P = self.position.y
        theta = self.angle
        cos_value = math.cos(theta)
        sin_value = math.sin(theta)
        self.scans = []
        for i in range(len(ranges)):
            if ranges[i] == 0.0:
                ranges[i] = 3.5
            if np.isnan(ranges[i]):
                ranges[i] = 0.001
            if ranges[i] < self.dcll and self.end != 'end':  # Test whether a collision happened 
                self.end = 'Collide with obstacles!'
                break
            distance = ranges[i]
            theta = theta_min + theta_inc * i
            x = distance * math.cos(theta)
            y = distance * math.sin(theta)
            remap_x = cos_value * x - sin_value * y + x_P
            remap_y = sin_value * x + cos_value * y + y_P
            self.scans.append([remap_x, remap_y])

    def cal_U(self, loc):
        d_goal = ((loc[0] - self.goal[0]) ** 2 + (loc[1] - self.goal[1]) ** 2) ** 0.5  # Distance to goal
        U_att = (0.5 * self.k_att_2 * d_goal ** self.m_inner + 0.5 * self.k_att_1 * d_goal ** self.m_outer)  # Attractive field
        U_rep = 0  # Repulsive field 
        for scan in self.scans:
            d_obs = ((loc[0] - scan[0]) ** 2 + (loc[1] - scan[1]) ** 2) ** 0.5  # Distance to obstacles 
            # Add all repulsive field
            if (d_obs > self.d_rep): 
                U_rep += 0
            else:
                U_rep += 0.5 * self.k_rep * ((1.0 / d_obs - 1.0 / self.d_rep) ** 2)
        U_total = U_att + U_rep  # Total potential field 
        return U_total

    def cal_U_array(self):
        """calculate the poential
        """
        U_0 = self.cal_U([self.position.x, self.position.y])  # potential field of agent's position 
        U_array = []  # Surrounding potential field 
        poetntial_number = 10  # Number of surrounding feature 
        delta_theta = 2 * math.pi / poetntial_number  # Angle difference between two surrounding features 
        delta_len = 0.1  # Distance between surrounding features and agent
        for i in range(poetntial_number):
            theta = delta_theta * i
            delta_x = delta_len * math.cos(theta + self.angle)
            delta_y = delta_len * math.sin(theta + self.angle)
            loc = [self.position.x + delta_x, self.position.y + delta_y]  # Position of surrounding features in global coordinate 
            U_total = self.cal_U(loc)
            U_array.append(U_total)  
        U_min = min(U_array)
        U_max = max(U_array)
        U_array = [(val - U_min) / (U_max - U_min) for val in U_array]  # Scale potential to [0, 1]
        self.potential = (U_0) / (U_max - U_min)
        return U_array

    def reset(self): 
        """ Reset for a new episode 
        """
        # Reset gazebo simulation 
        self.end = ''
        self.distance2goal = 10.0
        self.scans = []

        U_array = self.cal_U_array()  # Get surrounding potential fields 
        U_array.extend([self.angle2goal, self.distance2goal, self.angle, self.angular_velocity])  # Form state describer 
        self.state = U_array 
        self.start_time = rospy.Time.now().to_sec()  # Start clock 
        return np.array(self.state, dtype=np.float32) 

    def step(self, action): 
        """ Step based on state and action 
        """
        done = 0
        self.angular_velocity = action
        self.linear_velocity = (2.5 - abs(self.angular_velocity)) / 13.0  # Linear velocity is a function of angular velocity 
        vel_cmd = Twist()
        vel_cmd.linear.x =self.linear_velocity
        vel_cmd.angular.z = self.angular_velocity
        self.pub_cmd_vel.publish(vel_cmd)  # Publish velocity message 
        self.rate.sleep()  # Contrain publish frequency for training 

        U_array = self.cal_U_array()
        U_array.extend([self.angle2goal, self.distance2goal, self.angle, self.angular_velocity])  # Form state describer 
        self.state = U_array 
        reward = self.potential * (-1.0)  # + self.angular_velocity * (-1.0)  # Reward is the negative of the scaled potential 
        info = ''
        if self.end == 'Reach the goal!':
            reward += 3000  # Add a huge reward 
            done = 1 
            info = self.end 
            self.pub_cmd_vel.publish(Twist())  # Stop the agent 
        elif self.end == 'Collide with obstacles!': 
            reward -= 6000  # Add a huge penalty 
            done = 1
            info = self.end 
            self.pub_cmd_vel.publish(Twist())  # Stop the agent 
        if ((rospy.Time.now().to_sec() - self.start_time) > 100):  # Running too long 
            done = 1
            info = "Exceed 100 seconds!"
            self.pub_cmd_vel.publish(Twist())

        if done:
            self.end = 'end'
            self.goal = [0, 0]

        return np.array(self.state, dtype=np.float32), reward, done, info

    def close(self): 
        """ Close the environment
        """
        rospy.sleep(1)
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        






        

    
    






