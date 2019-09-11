
# coding: utf-8

# In[2]:


import numpy as np
import os
import math
import random
from scipy.optimize import linprog


# In[3]:


# Define the matrix and other constants
rows = 5
cols = 5
num_states = rows * cols
num_actions = 4
A = np.zeros(shape=(num_actions*num_states, num_states))
b = np.zeros(num_actions*num_states)
discount = 0.9


# In[119]:


# Takes in current position/state and action number to return new position/state and corresponding reward
def take_action(row, col, action):
    if row == 0 and col == 1:
        return 4, 1, 10
    elif row == 0 and col == 3:
        return 2, 3, 5
    elif action == 0 and row - 1 >= 0: #Up
        return row - 1, col, 0
    elif action == 1 and row + 1 < rows: #Down
        return row + 1, col, 0
    elif action == 2 and col + 1 < cols:  #Right
        return row, col + 1, 0
    elif action == 3 and col - 1 >= 0: #Left
        return row, col - 1, 0
    else:
        return row, col, -1


# In[120]:


# Here since to find V*, we need to take the action which maximizes the expression 
# Since max is a non linear function, we write it as 4 different equations
# In this case we will need to optimize Ax >= b
global_cnt = 0
for r in range(rows):
    for c in range(cols):
        for a in range(num_actions):
            # Get neighbour 
            newr, newc, reward = take_action(r, c, a)
            if newr == r and newc == c:
                A[global_cnt, cols * newr + newc] = - 1 + discount
            else:
                A[global_cnt, cols * r + c] = -1
                A[global_cnt, cols * newr + newc] = discount
            b[global_cnt] = reward
            global_cnt += 1


# In[121]:


# Get solution using the linprog optimizer
res = linprog(np.array([1]*num_states), A_ub=A, b_ub=-b)


# In[127]:


V = res.x


# In[128]:


# Print optimal policy
d = {0:"Up", 1:"Down", 2:"Right", 3:"Left"}


# In[138]:


optim_policy = [""]*num_states
cnt = 0
for i in range(rows):
    for j in range(cols):
        arr = []
        for a in range(num_actions):
            newr, newc, reward = take_action(i, j, a)
            newstate = cols * newr + newc
            val = V[newstate]
            arr.append(val)
        arr = np.round(arr,1)
        maxval = max(arr)
        directions = ""
        for k in range(num_actions):
            if arr[k] == maxval:
                directions = directions + d[k] + " "
        optim_policy[cnt] = directions
        cnt += 1


# In[139]:


optim_policy = np.array(optim_policy)
print(optim_policy.reshape(rows, cols))

