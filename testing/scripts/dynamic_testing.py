#!/usr/bin/env python
# coding=utf-8

import rospy
import math
import random
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState 
from geometry_msgs.msg import Point

class MovingTesting():
    def __init__(self, model_name, current_position=None, velocity=None, moving_type=None):
        self.model = ModelState() 
        self.model.model_name = model_name
        self.model.pose.position.x = current_position[0]
        self.model.pose.position.y = current_position[1]
        self.velocity = velocity
        self.pub_goal_position = rospy.Publisher('goal_position', Point, queue_size=1)  
        if moving_type is not None:
            self.moving_type = moving_type 

    def moving_obstacle(self):
        """ Move an obstacle in horizontal or verticle 
        """ 
        start_time = rospy.Time.now().to_sec()  # Start time clock 
        # velocity = 0.04 * random.choice([-1, 1])  # Linear velocity 
        running = rospy.get_param('running')  # Help to break the while loop when the environment is going to close 
        position = [self.model.pose.position.x, self.model.pose.position.y]  # Mark down the current position 
        while not rospy.is_shutdown() and running: 
            # Move the obstacle according to the time passed by instead of iteration 
            if self.moving_type == 'vertical': 
                self.model.pose.position.y = position[1] + (rospy.Time.now().to_sec() - start_time) * self.velocity
            elif self.moving_type == 'horizontal': 
                self.model.pose.position.x = position[0] + (rospy.Time.now().to_sec() - start_time) * self.velocity
            # Check the position and send the moving command 
            if self.moving_type == 'vertical' and (self.model.pose.position.y <= -2.0 or self.model.pose.position.y >= 2.0):  
                start_time = rospy.Time.now().to_sec() 
                position = [self.model.pose.position.x, self.model.pose.position.y]
                self.velocity *= -1 
                rospy.sleep(0.1)
            elif self.moving_type == 'horizontal' and (self.model.pose.position.x <= -4.0 or self.model.pose.position.x >= 4.0):
                start_time = rospy.Time.now().to_sec() 
                position = [self.model.pose.position.x, self.model.pose.position.y]
                self.velocity *= -1 
                rospy.sleep(0.1)
            else: 
                try:
                    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)   
                    set_state(self.model)
                    rospy.sleep(0.08)
                except:
                    pass
            running = rospy.get_param('running')  # Check whether keep running 
        rospy.set_param('model_is_moving', 0)   # Tell the environment the thread is end 

    def moving_goal(self):
        """ Move a goal in circular mannar
        """
        goal_list = [[-4.0, 0.0], [-3.0, 1.0], [-2.0, 0.0], [-1.0, -1.0], [0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, -1.0], [4.0, 0.0]]
        # velocity_list = [0.04, -0.04, 0.04, -0.04, -0.04, -0.04, 0.04, 0.04, -0.04]
        # goal_list = [[-4.0, random.uniform(-2, 2)], [-3.0, random.uniform(-2, 2)], [-2.0, random.uniform(-2, 2)], [-1.0, random.uniform(-2, 2)], [0.0, random.uniform(-2, 2)], [1.0, random.uniform(-2, 2)], [2.0, random.uniform(-2, 2)], [3.0, random.uniform(-2, 2)], [4.0, random.uniform(-2, 2)]]
        velocity_list = [0.04 * random.choice([-1, 1]), 0.04 * random.choice([-1, 1]), 0.04 * random.choice([-1, 1]), 0.04 * random.choice([-1, 1]), 0.04 * random.choice([-1, 1]), 0.04 * random.choice([-1, 1]), 0.04 * random.choice([-1, 1]), 0.04 * random.choice([-1, 1]), 0.04 * random.choice([-1, 1])]
        i = 0
        self.model.pose.position.x = goal_list[i][0]
        self.model.pose.position.y = goal_list[i][1]
        velocity = velocity_list[i]
        rospy.set_param('model_is_moving', 1) 
        start_time = rospy.Time.now().to_sec() 
        # velocity = 0.04 * random.choice([-1, 1])  # Linear velocity 
        running = rospy.get_param('running') 
        position = [self.model.pose.position.x, self.model.pose.position.y]  # Mark down the current position 
        while not rospy.is_shutdown() and running:
            next = 0
            next = rospy.get_param('next')
            if next and i != len(goal_list):
                i += 1
                self.model.pose.position.x = goal_list[i][0]
                self.model.pose.position.y = goal_list[i][1]
                velocity = velocity_list[i]
                goal = Point()
                goal.x = self.model.pose.position.x
                goal.y = self.model.pose.position.y
                self.pub_goal_position.publish(goal)
                rospy.set_param('next', 0)
                next = 0
                position = [self.model.pose.position.x, self.model.pose.position.y]
                start_time = rospy.Time.now().to_sec() 
            self.model.pose.position.y = position[1] + (rospy.Time.now().to_sec() - start_time) * velocity
            if (self.model.pose.position.y <= (goal_list[i][1] - 0.8) or self.model.pose.position.y >= (goal_list[i][1] + 0.8)):  
                start_time = rospy.Time.now().to_sec() 
                position = [self.model.pose.position.x, self.model.pose.position.y]
                velocity *= -1 
                rospy.sleep(0.1)
            else:
                try:
                    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)   
                    set_state(self.model)
                    goal = Point()
                    goal.x = self.model.pose.position.x
                    goal.y = self.model.pose.position.y
                    self.pub_goal_position.publish(goal)
                except:
                    pass
            running = rospy.get_param('running')  # Check whether keep running 
        rospy.set_param('model_is_moving', 0)   # Tell the environment the thread is end 


    

