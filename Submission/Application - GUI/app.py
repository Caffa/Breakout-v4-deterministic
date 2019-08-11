import os
# dir_path = os.path.dirname(os.path.realpath("app.py"))
# print(dir_path)
# print(os.getcwd())
#
# os.chdir(dir_path)
from guizero import App,Window, PushButton, Slider, Text, TextBox, Picture, Box, CheckBox

def train16Threads():
    # target = "Train16Threads_3LookBacks.py"
    import Train16Threads_3LookBacks
    Train16Threads_3LookBacks.main()
    # executeTarget(target)

def train4Threads():
    import Train4Threads_8LookBacks
    Train4Threads_8LookBacks.main()
    # target = "Train4Threads_8LookBacks.py"
    # executeTarget(target)

def testShow4Threads():
    import Test4Threads
    Test4Threads.main()
    # target = "Test4Threads.py"
    # executeTarget(target)

def testShow16Threads():
    import Test16Threads
    Test16Threads.main()
    # target = "Test16Threads.py"
    # executeTarget(target)


# def basicPathSetup():
#     #get current path of file and make sure we are in this directory
#     dir_path = os.path.dirname(os.path.realpath(__file__))
#     os.chdir(dir_path)
#     return dir_path
#
# def executeTarget(targetFile):
#     dir_path = os.path.dirname(os.path.realpath(__file__))
#     os.chdir(dir_path)
#     try:
#         pathOfTarget = os.path.join(os.getcwd(), targetFile)
#         if os.path.isfile(pathOfTarget):
#             try:
#                 execfile(targetFile)
#             except:
#                 print("Error executing python file with execfile, please run change to this directory then run" + str(pathOfTarget) +" in bash")
#         else:
#             myCmd = 'cd ' + dir_path + os.pathsep
#             os.system(myCmd)
#             print("Changed directory to " + dir_path)
#     except:
#         print("Problems changing directory")
#     # dir_path = basicPathSetup()
#     else:
#         try:
#             pathOfTarget = os.path.join(dir_path, targetFile)
#             myCmd = 'python3 ' + pathOfTarget
#             os.system(myCmd)
#         except:
#             print("Error executing python file, please run " + str(pathOfTarget) +" in bash")
#         else:
#             try:
#                 execfile(targetFile)
#             except:
#                 print("Error executing python file, please run change to this directory then run" + str(pathOfTarget) +" in bash")
#             else:
#                 execfile(pathOfTarget)

# def train16Threads():
#     target = "Train16Threads_3LookBacks.py"
#     executeTarget(target)
#
# def train4Threads():
#     target = "Train4Threads_8LookBacks.py"
#     executeTarget(target)
#
# def testShow4Threads():
#     target = "Test4Threads.py"
#     executeTarget(target)
#
# def testShow16Threads():
#     target = "Test16Threads.py"
#     executeTarget(target)


app = App(title="Breakout", layout="grid")

col4 = Box(app, layout="grid", grid=[0,1], align="top")

col1 = Box(app, layout="grid", grid=[0,0], align="top")
TrainLabel = Text(col1, text="Best Model:", grid=[0,0])
buttonTrainFour = PushButton(col1, text= "Train with A3C: 4 Threads, Look back 8 Frames", grid=[1,0], command=train4Threads) #starts training with
BTrainLabel = Text(col1, text="Second Best Model:", grid=[0,1])
buttonTrain = PushButton(col1, text= "Train with A3C: 16 Threads, Look back 3 Frames", grid=[1,1], command=train16Threads) #starts training with

BTestLabel = Text(col4, text="Best Model:", grid=[0,1])
buttonTestFour = PushButton(col4, text= "Test Pre-trained 4 Threads, Look back 8 Frames", grid=[1,1], command=testShow4Threads) #test for
B2TestLabel = Text(col4, text="Second Best Model:", grid=[0,2])
buttonTestSixTeen = PushButton(col4, text= "Test Pre-trained 16 Threads Model", grid=[1,2], command=testShow16Threads) #test for
# framesToTestLabel = Text(col4, text="Test for _ Episodes:", grid=[0,3])
# framesToTestInput = TextBox(col4, text="100", grid=[1,3])

app.display()
