#!/usr/bin/env python

import click
import numpy as np
import gym
import sys
import argparse
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
GAMES=1
def include_bias(ob):
    return [ob[0],ob[1],1]


def checkArgs(args=None):
    parser = argparse.ArgumentParser(description='Policy Gradient')
    parser.add_argument('--alpha',help = 'Learning Rate(alpha)', default = 0.1,type=float)
    parser.add_argument('--gamma',help = 'Discount rate', default=0.01,type=float)
    parser.add_argument('--episodes',help = 'Number of episodes',default=10000,type=int)
    parser.add_argument('--env',help = 'chakra/visham', default='chakra')
    parser.add_argument('--render',help = 'Render Environment', action='store_true',default=False)
    parser.add_argument('--batchSize',help = 'Batch Size',default=10,type=int)

    args = parser.parse_args(args)
    return args



m=MLPRegressor()
def fit_v(states,rewards):
    global m
    m.partial_fit(states,rewards)



def v(state):
    value=m.predict([state])
    return value[0]
def chakra_get_action(theta, ob, rng=np.random):
    ob_1 = include_bias(ob)
    mean = theta.dot(ob_1)#/np.linalg.norm(theta.dot(ob_1)) 
    return rng.normal(loc=mean, scale=1.)#*0.01

def get_tragectory(env,theta,rng,renderer):
    rewards = []
    actions = []
    states = []
    ob=env.reset()
    
    done=False
    
    while not done:
        
        action = chakra_get_action(theta, ob, rng=rng)
        next_ob, rew, done,_ = env._step(action)#env.action_space.sample())
        
        states.append(ob)
        actions.append(action)
        rewards.append(rew)
        
        ob = next_ob
        if renderer and RENDER:
            env._render()
    return states,actions,rewards



def get_grad(states,actions,rewards,theta):
    
    for i in range(len(rewards)-2,-1,-1):
        rewards[i]=rewards[i]+LAMBDA*rewards[i+1]
    fit_v(states,rewards)
    grad=np.zeros(theta.shape)
    lam=LAMBDA
    for i in range(len(states)):
        s=states[i]
        ob_1 = include_bias(s)
        mean = theta.dot(ob_1)

        dmean=np.array([[[s[0],s[1],1],[0,0,0]],[[0,0,0],[s[0],s[1],1]]]) #derivative of mean wrt theta
        dpi=actions[i]-mean #derivative of log pi wrt mean
        grad += np.dot(dpi,dmean)*(rewards[i]-(v([0.5,0.5])+v([0.5,-0.5])+v([-0.5,0.5])+v([-0.5,-0.5]))/4.0)
        #print v(states[i])
    return grad/len(states)


def main(env_id):
    allRew=[]
    # Register the environment
    rng = np.random.RandomState(42)
    global ALPHA
    if env_id == 'chakra':
        from rlpa2 import chakra
        env = gym.make('chakra-v0')
        get_action = chakra_get_action
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
    elif env_id =='visham':
        from rlpa2 import visham
        env = gym.make('visham-v0')
        get_action = chakra_get_action
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
    else:
        raise ValueError(
            "Unsupported environment: must be 'chakra' ")

    env.seed(42)
    renderer=False
    # Initialize parameters
    theta = rng.normal(scale=0.01, size=(action_dim, obs_dim+1))
    grad=np.zeros((theta.shape))
    for _ in range(GAMES):
        
        for e in range(EPISODES):
            if e>EPISODES-EPISODES/10:
                renderer=True
            #get tragectory
            states,actions,rewards=get_tragectory(env,theta,rng,renderer)
            
            #calculate gradiants and make update
            grad += get_grad(states,actions,rewards,theta)
            if (e+1)%BATCH_SIZE == 0:
                grad = grad/(np.linalg.norm(grad)+1e-8) 
                theta += ALPHA*grad
                grad=np.zeros(theta.shape)
            print("Episode %d reward: %.2f" % (e,np.sum(rewards)))
            allRew.append(np.sum(rewards))
            
               
        
        


if __name__ == "__main__":
    args=checkArgs(sys.argv[1:])
    print "Starting With : ",args
    env_id=args.env
    EPISODES=args.episodes
    BATCH_SIZE=args.batchSize
    ALPHA=args.alpha
    LAMBDA=args.gamma #gamma in the actual context
    RENDER=args.render
    main(env_id)
