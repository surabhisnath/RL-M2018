
# coding: utf-8

# In[242]:


import numpy as np
import os
import math
import random


# In[243]:


# Gridworld
rows = 4
cols = 4
num_states = rows * cols
action = ["Up", "Down", "Right", "Left"]
num_actions = 4
lim = 0.00001


# In[244]:


#Perform policy evaluation
def policy_evaluation(V, policy):
    cnt = 0
    while(1):
        delta = 0
        for s in range(1, num_states-1):
            oldV = V[s]
            newV = 0
            news = 1000000
            for a in range(num_actions):
                r = s // cols
                c = s % cols
                if a == 0 and r - 1 >= 0:
                        news = (r - 1) * cols + c
                elif a == 1 and r + 1 < rows:
                        news = (r + 1) * cols + c
                elif a == 2 and c + 1 < cols:
                        news = r*cols + c + 1
                elif a == 3 and c - 1 >= 0:
                        news = r*cols + c - 1
                else:
                    news = s

                newV = newV + policy[s,a] * (-1 + V[news])
            V[s] = newV
            delta = max(delta, np.abs(oldV - newV))
        
        cnt += 1
        if delta < lim:
            return V


# In[245]:


#Perform policy improvement
def policy_improvement(V, policy):
    policy_stable = True
    for s in range(1, num_states-1):
        currentpol_s = policy[s].copy()
        
        holder = np.zeros(num_actions)
        for a in range(num_actions):
            r = s // cols
            c = s % cols
            if a == 0 and r - 1 >= 0:
                    news = (r - 1) * cols + c
            elif a == 1 and r + 1 < rows:
                    news = (r + 1) * cols + c
            elif a == 2 and c + 1 < cols:
                    news = r*cols + c + 1
            elif a == 3 and c - 1 >= 0:
                    news = r*cols + c - 1
            else:
                news = s
            holder[a] = (-1 + V[news]) #Undiscounted

        policy[s] = np.zeros(num_actions)
        holder = np.round(holder,2)
        maxval = max(holder)
        freq = holder.tolist().count(maxval)

        for l in range(num_actions):
            if holder[l] == maxval:
                policy[s, l] = 1/freq
                
                if currentpol_s[l] != policy[s, l]:
                    policy_stable = False
                    

    return policy_stable, policy


# In[246]:


#Policy iteration successively calls policy evaluation and policy improvement till no change in updated policy
def policy_iteration():
    V = np.zeros(num_states)
    policy = np.full((num_states, num_actions), 1/num_actions, dtype=float)
    res = True
    while(1):
        V = policy_evaluation(V, policy)
        res, policy = policy_improvement(V, policy)
        if res == True:
            break
    return V, policy


# In[247]:


V, policy = policy_iteration()


# In[249]:


print(V)
print(policy)


# ## Value Iteration

# In[239]:


#Perform value iteration
def value_iteration(V, policy):
    V = np.zeros(num_states)
    lim = 0.000001
    
    while(1):
        delta = 0
        for s in range(1, num_states-1):
            oldV = V[s]
            newV = -np.inf
            for a in range(num_actions):
                r = s // cols
                c = s % cols
                if a == 0 and r - 1 >= 0:
                        news = (r - 1) * cols + c
                elif a == 1 and r + 1 < rows:
                        news = (r + 1) * cols + c
                elif a == 2 and c + 1 < cols:
                        news = r*cols + c + 1
                elif a == 3 and c - 1 >= 0:
                        news = r*cols + c - 1
                else:
                    news = s
                newV = max(newV, (-1 + V[news]))
            V[s] = newV
            delta = max(delta, abs(oldV - newV))
        
        if delta < lim:
            break
    
    policy = np.full((num_states, num_actions), 1/num_actions, dtype=float)
    for s in range(1, num_states-1):
        currentpol_s = policy[s].copy()
        policy[s] = np.zeros(num_actions)
        holder = np.zeros(num_actions)

        for a in range(num_actions):
            r = s // cols
            c = s % cols
            if a == 0 and r - 1 >= 0:
                    news = (r - 1) * cols + c
            elif a == 1 and r + 1 < rows:
                    news = (r + 1) * cols + c
            elif a == 2 and c + 1 < cols:
                    news = r*cols + c + 1
            elif a == 3 and c - 1 >= 0:
                    news = r*cols + c - 1
            else:
                news = s
            holder[a] =  (-1 + V[news])

        maxval = max(holder)
        freq = holder.tolist().count(maxval)
        for l in range(num_actions):
            if holder[l] == maxval:
                policy[s, l] = 1/freq
    
    return V, policy


# In[240]:


V, policy = value_iteration(V, policy)


# In[241]:


# On comparing Value iteration and Policy iteration, both result in same V and policy
print(V)
print(policy)

