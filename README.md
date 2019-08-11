This project uses AI Gym's Breakout-v4-deterministic environment.

![](https://github.com/Caffa/Breakout-v4-deterministic/blob/master/Breakout%20Gifs/4FramesPerformance.gif)

We trained 2 main models:
- 16 thread model: looks at past 3 frames and considers 5 N-steps. Batch Size used is 20, Experience Queue size is 256.
- 4 thread model: looks at past 8 frames and considers 10 N-steps. Batch Size used is 64, Experience Queue size is 64. Auto-Fire at start implemented.
The 16 thread model ran for 48 hours at which point it begin de-learning as the model learnt to never fire. The 4 thread model ran for 24 hours and has auto-fire at the start of the scene enables (if the model doesn't decide to fire on its own).

Breakout Gifs folder contain the deployment of the 2 trained models Submissions folder contain the app and the final code. Important Results contain the graphs, saved rewards (in csv format), reporting outputs (for each episode) in a text file and the saved weights at every 100000 frames.

# Results of Running A3C Models
Can be found in  Folder

## 4 Threads A3C
### Training
The "Results/Important Results/Run with 4 Threads - Best Model" folder contains the results for training the 4 Threads A3C Model with the parameters as shown below

![4 Thread A3C Model Parameters](https://github.com/Caffa/Breakout-v4-deterministic/blob/master/Parameters/4%20Thread%20Model.png)

### Testing
The "Results/Important Results/Deployment (Evaluate All Trained Weights)" contains the code and results for testing the 16 Threads A3C Model to find which trained weights were the best.

We then ran the top 7 best trained weights for a 100 episodes each. The code and results of that are in "Results/Important Results/BenchMark/4 Thread" along with the end state images for each episode.

## 16 Threads A3C
### Training
The "Results/Important Results/Run with 16 Threads" folder contains the results for training the 16 Threads A3C Model with the parameters as shown below

![16 Thread A3C Model Parameter](https://github.com/Caffa/Breakout-v4-deterministic/blob/master/Parameters/16%20threads.png)

### Testing
The "Results/Important Results/Deployment" contains the code and results for testing the 16 Threads A3C Model to find which trained weights were the best.

We then ran the top 20 best trained weights for a 100 episodes each. The code and results of that are in "Results/Important Results/BenchMark/16 Thread" along with the end state images for each episode.

# Application (GUI)
![App UI](https://github.com/Caffa/Breakout-v4-deterministic/blob/master/Misc/appUI.png)
Our application can be run by
## Instructions
In order to run the application,

1. cd into the folder "Submission/Application - GUI" 
2. If necessary, install the requirements according to the respective requirements.txt
3. Run with python3 app.py

# Source Code for A3C Models
Can be found in the Submissions/Code Folder
## Instructions
In order to run the code,

1. cd into the folder containing that .py file
2. If necessary, install the requirements according to the respective requirements.txt
3. Run with python3 <filename>

## Details on Code
### Testing
Deploy4Threads.py
This runs the 4 Threads A3C Model in a testing mode, using the weights in the trained_weights folder. Use this to test.

### Training
train4Threads_8LookBacks.py
This runs the 4 Threads A3C Model in a training mode with randomly initialised weights. This model runs by looking at the past 8 frames. Use this to train.

### Other models

train16Threads_3LookBacks.py
This runs the 16 Threads A3C Model in a training mode with randomly initialised weights. This model runs by looking at the past 3 frames.

# Source Code for Other Models
Can be found in "Results/Previous Tests" Folder
