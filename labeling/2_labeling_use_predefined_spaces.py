# Initialization
import numpy as np
# Test
from matplotlib import pyplot as plt
from IPython.display import clear_output
from subprocess import Popen
import sys

file_names = ["/9-smoke", "/10-smoke", "/11-smoke", "/12-smoke", "/13-smoke", "/14-smoke",
              "/15-smoke", "/16-smoke", "/17-smoke", "/18-smoke", "/19-smoke"]
#file_names = ["/8-dust"]
path = "/media/juli/98F29C83F29C67721/SemesterProject/1_data/1_Upload/new/"

def edit_frames(frames, load_file):
    print load_file
    print frames.shape
    for j, frame in enumerate(frames):
        print j
        for i, point in enumerate(frame):
            x = point[0]
            y = point[1]
            z = point[2]
            # Definition of spaces for each of the datasets
            logic_func = True
            if "1-smoke-ETH" in load_file:  # Needs editing!!!
                logic_func = x + y > 0 and x + y < 0.7 and y - x < 0.6 and y - x > -1  # ETH_data/1.npy
            elif "2-smoke-ETH" in load_file:  # Needs editing!!!
                logic_func = x + y > 0 and x < 1.6 and y - x > -1 and z - 0.1 * y > -0.08 and y - x < 1.5  # ETH_data/2.npy
            elif "1-dust" in load_file:
                logic_func = (x < 1 or x > 3 or y < 2 or y > 4 or z < -1.5 or z > -0.5) and ((x**2+y**2)**.5 < 11)
            elif "2-dust" in load_file:
                # To filter out human
                if j < 180:
                    logic_func = (x < 1 or x > 3 or y < 2 or y > 4 or z < -1.8 or z > 0.5)
                elif j > 180 and j < 192:
                    logic_func = ((x-2.5)**2 + (y-4)**2)**.5 > 0.5
                elif j < 230:
                    logic_func = (x < 1 or x > 3 or y < 4 or y > 6 or z < -1.8 or z > 0.5)
                elif j < 369:
                    logic_func = (x < 1 or x > 3.5 or y < 5 or y > 7 or z < -1.8 or z > 0.5)
                elif j < 445:
                    logic_func = (x < 1 or x > 3.5 or y < 6 or y > 8 or z < -1.8 or z > 0.5)
                elif j < 500:
                    logic_func = (x < 1.5 or x > 4 or y < 4.2 or y > 7 or z < -1.8 or z > 0.5)
                elif j < 525:
                    logic_func = (x < 1 or x > 4 or y < 6 or y > 10 or z < -1.8 or z > 0.5)
                elif j < 550:
                    logic_func = y < 7
                elif j < 750:
                    logic_func = y < 10
                elif j < 775:
                    logic_func = y < 8
                elif j < 790:
                    logic_func = y < 7 and x < 2
                elif j >= 790:
                    logic_func = False
                # To Filter out disturbances
                logic_func = logic_func and (x < 2 or x > 4.4 or y < 2.5 or y > 4.7 or z < -1.7 or z > - 1.3) and \
                (x**2+y**2)**.5 < 12 and (x > -3 or y < 5.5)
            elif "3-dust" in load_file:  # QUT_data/1.npy
                logic_func = y - 1.3 * x > 0
                if j > 500 and j < 525 and x > 2 and x < 3 and y > 3 and y < 4:
                    logic_func = False
                elif (x**2 + y**2)**.5 > 10:
                    logic_func = False
            elif "4-dust" in load_file:
                if j < 96 or j > 700:
                    logic_func = False
                elif j < 125:
                    logic_func = y > x - 1.3
                elif j < 200:
                    logic_func = x < 2 or y > 2
                elif j > 340 and j < 410:
                    logic_func = y > 1
                #elif j > 380 and j < 410 and y < x - 2:
                #    logic_func = False
                elif j > 450 and j < 480:
                    logic_func = x > 0 and x < 4
                elif j > 570 and j < 630:
                    logic_func = x < 3
                elif j > 690:
                    logic_func = y > 0
                #if y < 0:
                    #logic_func = np.sqrt(x*x+y*y+z*z) < 3.6
                else:
                    logic_func = x < 3 #and np.sqrt(x*x+y*y+z*z) < 5
                logic_func = logic_func and (x < 3.5 and x > -2) and (x**2+(y-3)**2)**.5 < 5
            elif "5-dust" in load_file:
                logic_func = np.sqrt(x*x+y*y+z*z) < 3.6 and x < 3 or (x > 2 and x < 4 and y > 2 and y < 4)
            elif "6-dust" in load_file:
                logic_func = (x < 2 or x > 4 or y < 2 or y > 4) and (x < 4.5 or x > 5.5 or y < 4 or y > 5)
                if j > 340 and j < 375:
                    logic_func = logic_func and (x < 2 or x > 4 or y < 1 or y > 4)
                elif j > 395 and j < 420:
                    logic_func = logic_func and (x < 2 or x > 4 or y < 2 or y > 5)
                logic_func = logic_func and x < 5 and y < 9
            elif "7-dust" in load_file:
                if j < 212 or j > 900:
                    logic_func = False
                elif j > 570:
                    logic_func = y < 5.6
                else:
                    logic_func = y < 4.7 and x < 3
            elif "8-dust" in load_file:
                if j > 200 and j < 230:
                    logic_func = x < 2 or x > 3 or y < 2 or y > 3
                elif j > 380:
                    logic_func = y > -1
                logic_func = logic_func and x < 3
                if y < 0:
                    logic_func = logic_func and np.sqrt(x*x+y*y+z*z) < 3.5
                else:
                    logic_func = logic_func and (x**2 + y**2)**.5 < 9
            elif "9-smoke" in load_file:
                logic_func = (x < 1.5 or y < -0.7 or y > 0.6) and y > -2 and ((x**2 + y**2)**.5 < 10)
            elif "10-smoke" in load_file:
                logic_func = (x < 1.5 or y < -0.7 or y > 0.6) and ((x**2 + y**2)**.5 < 10)
            elif "11-smoke" in load_file:
                logic_func = (x < 1.5 or x > 2.4 or y < -0.7 or y > 0.6) and x > -1 and ((x**2 + y**2)**.5 < 10)
            elif "12-smoke" in load_file:
                logic_func = (x < 2 or x > 3 or y < -1 or y > 0.2) and ((x**2 + y**2)**.5 < 10)
            elif "13-smoke" in load_file:
                logic_func = (x < 2 or x > 3 or y < -1 or y > 0.2) and ((x**2 + y**2)**.5 < 10)
            elif "14-smoke" in load_file:
                logic_func = y > 4.2 and ((x**2 + y**2)**.5 < 10)
            elif "15-smoke" in load_file:
                logic_func = (x < 2 or y > 4.4) and ((x**2 + y**2)**.5 < 10)
            elif "16-smoke" in load_file:
                logic_func = x < 2.1 and ((x**2 + y**2)**.5 < 10)
            elif "17-smoke" in load_file:
                logic_func = x < 2.1 and ((x**2 + y**2)**.5 < 10)
            elif "18-smoke" in load_file:
                if j > 440:
                    logic_func = y > 0
                logic_func = logic_func and x < 2.1 and y < 7 and y > -3 and ((x**2 + y**2)**.5 < 10)
            # No idea why I need to start if loop again here
            if "19-smoke" in load_file:
                logic_func = x < 2.1 and x > -3 and ((x**2 + y**2)**.5 < 10)
            elif "20-smoke" in load_file:
                logic_func = True
            elif "21-smoke" in load_file:
                logic_func = True
            elif "22-smoke" in load_file:
                logic_func = True
            elif "23-smoke" in load_file:
                logic_func = True
            elif "24-smoke" in load_file:
                logic_func = True
            elif "25-smoke" in load_file:
                logic_func = True
            if not logic_func:
                frames[j, i, len(frames[0,0])-3] = 1
                frames[j, i, len(frames[0,0])-2] = 0
                frames[j, i, len(frames[0,0])-1] = 0
    return frames.astype(dtype=np.float32)

for name in file_names:
    print(name)
    frames = np.load(path + name + "_labeled.npy")

    # Accumulation and Spaces (to cut out person)

    final_frames = edit_frames(frames, name)
    # print final_frames.shape
    np.save(path + name + "_labeled_spaces.npy", final_frames)