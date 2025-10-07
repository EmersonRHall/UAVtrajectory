This is week 4 of the research.   
This code implements a baseline trajectory planner for an unmanned aerial vehicle using the A* algorithm in MATLAB. 
The environment is modeled as a bounded area with static obstacles, which are inflated by a safety margin to ensure clearance.
An occupancy grid with 0.2 meter resolution is used to represent free and occupied space.
The program generates a collision-free path from the start position at [−8,−8] to the goal at [10,4]. 
The resulting trajectory avoids the obstacles while maintaining clearance, producing a path length of about 23.7 meters with roughly 2,247 nodes expanded.
This baseline provides a reliable benchmark for evaluating more advanced trajectory optimization and machine learning methods in future work.
