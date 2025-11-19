This is Week 11 of the research.  

Google Drive Link that includes videos and charts: https://drive.google.com/drive/folders/12Fcdur5CxTAVu31tRVvBr4MUsXnlgcEP?usp=share_link

This MATLAB script performs final validation of a three-UAV planning and control system across multiple terrains and weather conditions. It compares a classical A* Baseline mode with a Machine Learning mode using identical setups for fairness. Each terrain—urban, natural, and manmade—is generated with fixed layouts, and three UAVs are assigned collision-free start and goal positions with terrain-specific routing so their paths remain distinct.

During each run, the system simulates calm, breeze, gusty, and storm conditions using a wind and gust model. Global planning is handled by A* on an inflated grid, and the resulting paths are smoothed before being tracked by a local controller that manages repulsion, obstacle sliding, and wind effects. The script then records success, time, path length, clearance, speed, and replanning frequency.

For every terrain, weather, and mode, the script automatically produces a video, a metrics CSV file, and summary figures that compare Baseline and Machine Learning performance. Running the script only requires MATLAB with the Image Processing Toolbox, and all outputs are generated after selecting the desired terrain at the prompt.
