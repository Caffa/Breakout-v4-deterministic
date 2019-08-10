This project uses AI Gym's Breakout-v4-deterministic environment.

I trained 2 models:

16 thread model: looks at past 3 frames and considers 5 N-steps. Batch Size used is 20, Experience Queue size is 256.
4 thread model: looks at past 8 frames and considers 10 N-steps. Batch Size used is 64, Experience Queue size is 64. Auto-Fire at start implemented.
The 16 thread model ran for 48 hours at which point it begin de-learning as the model learnt to never fire. The 4 thread model ran for 24 hours and has auto-fire at the start of the scene enables (if the model doesn't decide to fire on its own).

Breakout Gifs folder contain the deployment of the 2 trained models Submissions folder contain the app and the final code. Important Results contain the graphs, saved rewards (in csv format), reporting outputs (for each episode) in a text file and the saved weights at every 100000 frames.

To run using the app, click on the built app (built with pyinstaller and on guizero)

In order to run the code, cd into the folder containing that .py file and run with python3

App uses icon from freepik, https://www.flaticon.com/authors/freepik : Creative Commons BY 3.0
