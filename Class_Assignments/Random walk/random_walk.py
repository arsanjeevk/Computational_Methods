import numpy as np
import matplotlib . pyplot as plt

#parameters
N = 1000
l = 1

#single random walk
position = [0]
for i in range ( N ) :
  if np . random . random () < 0.5:
    new_pos = position [ -1] + l
  else :
    new_pos = position [ -1] - l
  position . append ( new_pos )

#plotting
fig , ax = plt . subplots (1 , 2 , figsize =(12 , 5) )
ax [0]. plot ( position )
ax [0]. set_xlabel (" Step Number ")
ax [0]. set_ylabel (" Position ")
ax [0]. set_title (" Random Walk Trajectory ")
ax [1]. hist ( position , bins =40)
ax [1]. set_xlabel (" Position ")
ax [1]. set_ylabel (" Frequency ")
ax [1]. set_title (" Position Distribution ")
plt . tight_layout ()
plt . show ()

#now simulating many walkers
walkers = 1000
final_positions = []
for i in range ( walkers ) :
    pos = 0
    for j in range ( N ) :
        if np . random . random () < 0.5:
          pos += l
        else :
          pos -= l
    final_positions . append ( pos )
plt . hist ( final_positions , bins =50)
plt . xlabel (" Final Position (m)")
plt . ylabel (" values ")
plt . title (" Distribution of Final Positions (1000 Walkers )")
plt . show ()
