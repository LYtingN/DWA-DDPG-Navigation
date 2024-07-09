#!/usr/bin/env python
# coding=utf-8

import rospy
import math
import random
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState 
from geometry_msgs.msg import Point

class MovingTraining(): 
    """ Move the goal and obstacles 
    """
    def __init__(self, model_name, current_position=None, moving_type=None):
        self.model = ModelState() 
        self.model.model_name = model_name
        self.model.pose.position.x = current_position[0]
        self.model.pose.position.y = current_position[1]
        self.pub_goal_position = rospy.Publisher('goal_position', Point, queue_size=1)  
        if moving_type is not None:
            self.moving_type = moving_type

    def moving_obstacle(self): 
        """ Move an obstacle in horizontal or verticle 
        """
        rospy.set_param('model_is_moving', 1)  # Tell the environment an obstacle is moving
        start_time = rospy.Time.now().to_sec()  # Start time clock 
        velocity = 0.03 * random.choice([-1, 1])  # Linear velocity 
        running = rospy.get_param('running')  # Help to break the while loop when the environment is going to close 
        position = [self.model.pose.position.x, self.model.pose.position.y]  # Mark down the current position 
        while not rospy.is_shutdown() and running: 
            # Move the obstacle according to the time passed by instead of iteration 
            if self.moving_type == 'vertical': 
                self.model.pose.position.y = position[1] + (rospy.Time.now().to_sec() - start_time) * velocity
            elif self.moving_type == 'horizontal': 
                self.model.pose.position.x = position[0] + (rospy.Time.now().to_sec() - start_time) * velocity
            # Check the position and send the moving command 
            if self.moving_type == 'vertical' and (self.model.pose.position.y <= -1.4 or self.model.pose.position.y >= 1.4):  
                start_time = rospy.Time.now().to_sec() 
                position = [self.model.pose.position.x, self.model.pose.position.y]
                velocity *= -1 
                rospy.sleep(0.1)
            elif self.moving_type == 'horizontal' and (self.model.pose.position.x <= -1.4 or self.model.pose.position.x >= 1.4):
                start_time = rospy.Time.now().to_sec() 
                position = [self.model.pose.position.x, self.model.pose.position.y]
                velocity *= -1 
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
        rospy.set_param('model_is_moving', 1) 
        start_time = rospy.Time.now().to_sec() 
        init_angle = math.atan2(self.model.pose.position.y, self.model.pose.position.x)  # Mark down the initial angle 
        angular_velocity = 0.03 * random.choice([-1, 1])  # Angular velocity 
        running = rospy.get_param('running') 
        while not rospy.is_shutdown() and running:
            angle = angular_velocity * (rospy.Time.now().to_sec() - start_time) + init_angle
            self.model.pose.position.x = 2.0 * math.cos(angle)
            self.model.pose.position.y = 2.0 * math.sin(angle)
            try:
                set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)   
                set_state(self.model)
                goal = Point()
                goal.x = self.model.pose.position.x
                goal.y = self.model.pose.position.y
                self.pub_goal_position.publish(goal)
            except:
                pass
            running = rospy.get_param('running')
        rospy.set_param('model_is_moving', 0) 




    

