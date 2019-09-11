
# coding: utf-8

# In[35]:


import numpy as np
import os
import math
import random


# In[36]:


# Define the matrix and other constants 
rows = 5
cols = 5
num_states = rows * cols
action = {"North", "South", "East", "West"}
num_actions = 4
A = np.zeros(shape=(num_states, num_states))
b = []
discount = 0.9
bforbetween = 0
bforedge = 0.25
bforcorner = 0.5 # (0.25 + 0.25)
sp1 = 21
sp2 = 13
cornercoeff = discount * 0.5 - 1
edgecoeff = discount * 0.25 - 1
betweencoeff = -1


# In[37]:


# Every expression for V(s) can be written as a linear equation in terms of other V(s') and constant bias terms
# Here, since there are 25 states, we will have 25 equations
# Then the equation will be Ax = b where A is a 25 x 25 matrix and x, b are 25 x 1 matrices
# We can observe a trend in equations of the corner cells vs the edge cells vs the middle cells

for r in range(rows):
    for c in range(cols):
        indexin1d = c + r * cols
        if r == 0 and c == 1:
            b.append(-1 * 10)
            A[indexin1d][indexin1d] = betweencoeff
            A[indexin1d][sp1] = discount
            
        elif r == 0 and c == 3:
            b.append(-1 * 5)
            A[indexin1d, indexin1d] = betweencoeff
            A[indexin1d, sp2] = discount
            
        elif r > 0 and r < rows - 1 and c > 0 and c < cols - 1: # between
            b.append(bforbetween)
            A[indexin1d, indexin1d] = betweencoeff
            A[indexin1d, (c + 1) + r * cols] = discount * 0.25
            A[indexin1d, c + (r + 1) * cols] = discount * 0.25
            A[indexin1d, (c - 1) + r * cols] = discount * 0.25
            A[indexin1d, c + (r - 1) * cols] = discount * 0.25
            
        elif r == 0 and c == 0: # corner tl
            b.append(bforcorner)
            A[indexin1d, indexin1d] = cornercoeff
            A[indexin1d, (c + 1) + r * cols] = discount * 0.25
            A[indexin1d, c + (r + 1) * cols] = discount * 0.25
            
        elif r == 0 and c == cols - 1: # corner rt 
            b.append(bforcorner)
            A[indexin1d, indexin1d] = cornercoeff
            A[indexin1d, (c - 1) + r * cols] = discount * 0.25
            A[indexin1d, c + (r + 1) * cols] = discount * 0.25

        elif r == rows - 1 and c == 0: # corner bl 
            b.append(bforcorner)
            A[indexin1d, indexin1d] = cornercoeff
            A[indexin1d, (c + 1) + r * cols] = discount * 0.25
            A[indexin1d, c + (r - 1) * cols] = discount * 0.25
        
        elif r == rows - 1 and c == cols - 1: # corner bl 
            b.append(bforcorner)
            A[indexin1d, indexin1d] = cornercoeff
            A[indexin1d, (c - 1) + r * cols] = discount * 0.25
            A[indexin1d, c + (r - 1) * cols] = discount * 0.25

        elif r == 0:
            b.append(bforedge)
            A[indexin1d, indexin1d] = edgecoeff
            A[indexin1d, (c + 1) + r * cols] = discount * 0.25
            A[indexin1d, c + (r + 1) * cols] = discount * 0.25
            A[indexin1d, (c - 1) + r * cols] = discount * 0.25

        elif r == rows - 1:
            b.append(bforedge)
            A[indexin1d, indexin1d] = edgecoeff
            A[indexin1d, (c + 1) + r * cols] = discount * 0.25
            A[indexin1d, (c - 1) + r * cols] = discount * 0.25
            A[indexin1d, c + (r - 1) * cols] = discount * 0.25

        elif c == 0:
            b.append(bforedge)
            A[indexin1d, indexin1d] = edgecoeff
            A[indexin1d, (c + 1) + r * cols] = discount * 0.25
            A[indexin1d, c + (r + 1) * cols] = discount * 0.25
            A[indexin1d, c + (r - 1) * cols] = discount * 0.25

        elif c == cols - 1:
            b.append(bforedge)
            A[indexin1d, indexin1d] = edgecoeff
            A[indexin1d, c + (r + 1) * cols] = discount * 0.25
            A[indexin1d, (c - 1) + r * cols] = discount * 0.25
            A[indexin1d, c + (r - 1) * cols] = discount * 0.25


# In[38]:


# Solve the system of linear equations using solver and print V(s) values 
result = np.linalg.solve(A, b)
result = np.round(result, 1)
print(result.reshape(rows,cols))

