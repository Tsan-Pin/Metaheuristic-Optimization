#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 17:39:16 2021

@author: BenHong
"""
import time
import random as rd
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

d = 30
f_opt = -418.9828 * d

def constraint(x): # constraint in the homework 
    if x > 500:
        return 500 
    elif x < -500:
        return -500
    else :
        return x

def f(x):  # optimaize target 
    sum = 0 
    for i in range(0,d-1):
        sum -= x[i]*math.sin(math.sqrt(abs(x[i])))
    return sum 

def error(f_x): # relative error 
    return abs(f_x-f_opt)/abs(f_opt)

# w:weight , n:number of Particle Swarm ,T : max iteration 
def Particle_Swarm_Optimization(n,T):
    start = time.time()
    ParticleSwarm = np.random.uniform(-500,500,size=(n,d)) #generate Particle Swarm
    fitness = [f(ParticleSwarm[i]) for i in range(n)]
    Local_optimal = ParticleSwarm.copy()
    optimal_value = [f(Local_optimal[i]) for i in range(n)]
    Global_optimal = np.zeros(d+1)
    Global_optimal[:-1] = Local_optimal[np.argsort(optimal_value)[0]]
    Global_optimal[-1] = min(optimal_value )
    velocity = np.zeros((n,d))
    e = []
    speed = []
    e.append(error(Global_optimal[-1])) 
    print('Relative error:',e[-1],"\n position:",Global_optimal[:-1])
    t = 0
    while(time.time()-start < 60*10 and t<T):
        w = 0.9 - 0.5/T*t    # weight linear decay
        c = 2.5 - 1.5/T*t
        r_1 = rd.random()
        r_2 = rd.random()
        velocity = [w*velocity[i] + r_1*(Local_optimal[i] - ParticleSwarm[i]) + r_2*(Global_optimal[:-1]- ParticleSwarm[i]) for i in range (n)]
        ParticleSwarm = ParticleSwarm + velocity
        for i in range(n):                    ##bound Particle Swarm
            for j in range (d):
                ParticleSwarm[i][j]=constraint(ParticleSwarm[i][j])
                
        fitness = [f(ParticleSwarm[i]) for i in range(n)]
        
        for i in range (n):
            if optimal_value[i] > fitness[i] :
                Local_optimal[i] = ParticleSwarm[i]
        optimal_value = [f(Local_optimal[i]) for i in range(n)]
        Global_optimal_old = Global_optimal.copy()
        Global_optimal[:-1] = Local_optimal[np.argsort(optimal_value)[0]]
        Global_optimal[-1] = min(optimal_value)
        e.append(error(Global_optimal[-1])) 
        speed.append(LA.norm((Global_optimal[:-1]-Global_optimal_old[:-1]),2))   
        t+=1
        print('Relative error:',e[-1],"\n position:",Global_optimal[:-1])
    plt.title("Relative error") # title
    plt.plot(e)
    plt.show()
    plt.title("velocity of the GB particle") # title
    plt.plot(speed)
    plt.show()  