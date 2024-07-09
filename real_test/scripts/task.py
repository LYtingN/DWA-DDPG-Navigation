#!/usr/bin/env python
# coding=utf-8
import sys,os

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add to system path
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rospy
import time 
import datetime
import argparse
from env_ddpg import Env
from ddpg import DDPG

def get_args():
    """ Hyperparameters
    """
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Obtain current time
    parser = argparse.ArgumentParser(description="hyperparameters")      
    parser.add_argument('--algo_name',default='DDPG',type=str,help="name of algorithm")
    parser.add_argument('--env_name',default='DDPG_Env',type=str,help="name of environment")
    parser.add_argument('--train_eps',default=1100,type=int,help="episodes of training")
    parser.add_argument('--test_eps',default=100,type=int,help="episodes of testing")
    parser.add_argument('--gamma',default=0.99,type=float,help="discounted factor")
    parser.add_argument('--critic_lr',default=1e-4,type=float,help="learning rate of critic")
    parser.add_argument('--actor_lr',default=1e-5,type=float,help="learning rate of actor")
    parser.add_argument('--memory_capacity',default=100000,type=int,help="memory capacity")
    parser.add_argument('--batch_size',default=128,type=int)
    parser.add_argument('--target_update',default=2,type=int)
    parser.add_argument('--soft_tau',default=1e-2,type=float)
    parser.add_argument('--hidden_dim',default=256,type=int)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available==True else 'cpu',type=str,help="cpu or cuda") 
    parser.add_argument('--result_path',default=curr_path + "/outputs/" + parser.parse_args().env_name + '/' + curr_time + '/results/')
    parser.add_argument('--model_path',default=curr_path + "/outputs/" + parser.parse_args().env_name + '/' + curr_time + '/models/') 
    parser.add_argument('--save_fig',default=True,type=bool,help="if save figure or not")   
    args = parser.parse_args()                           
    return args

def env_agent_config(cfg):
    """ Create environment and agent
    """
    env = Env()  # Create environment
    n_states = env.state_dim  # State dimension
    n_actions = env.action_dim  # Action dimension
    print("n_states: " + str(n_states) + ", n_actions: " + str(n_actions))
    agent = DDPG(n_states, n_actions, cfg)  # Create agent
    return env, agent

def test(cfg, env, agent):
    """ Testing
    """
    rospy.loginfo("Start testing!")
    rospy.loginfo("Env: %s; Algorithm: %s; Device: %s" %(cfg.env_name, cfg.algo_name, cfg.device))
    ep_reward = 0  # Episode reward
    ep_step = 0  # Iteration times in a episode
    state = env.reset()  # Reset environment to initial state
    done = 0  # # Set the episode to undone
    while not done:
        ep_step += 1  # Count one iteration times
        action = agent.choose_action(state)  # Get action from actor net, action <-- [-1, 1]
        next_state, reward, done, info = env.step(action)  # Update environment and return transition
        state = next_state  # Update state
        ep_reward += reward  # Accumulate step reward         

    rospy.loginfo("Reward: %.2f; Info: %s; Step: %d" %(ep_reward, info, ep_step)) 
    rospy.loginfo("Finish testing!")
    env.close()  # Close the environment   
    # plt.figure()
    # plt.plot(actions)
    # plt.show()
    return 0


if __name__ == "__main__":
    rospy.init_node('turtlebot3_ddpg')
    cfg = get_args()

    # # Training
    # env, agent = env_agent_config(cfg)
    # make_dir(cfg.result_path, cfg.model_path)  
    # res_dic = train(cfg, env, agent)
    # save_args(cfg) 
    # agent.save(path=cfg.model_path) 
    # save_results(res_dic, tag='train', path=cfg.result_path)  
    # plot_rewards(res_dic['rewards'], res_dic['ma_rewards'], cfg, tag="train") 

    # Testing
    env, agent = env_agent_config(cfg)
    # agent.load(path=cfg.model_path) 
    succ_lst = []
    coll_lst = []
    time_lst = []
    for i in range(1000, 1001, 20):
        agent.load(path=curr_path + "/outputs/" + 'DDPG_Env' + '/' + '20221216-051603' + '/models/' + str(i)) 
        test(cfg, env, agent)
    # # save_results(res_dic, tag='test', path=cfg.result_path) 
    # # plot_rewards(res_dic['rewards'], res_dic['ma_rewards'],cfg, tag="test") 
