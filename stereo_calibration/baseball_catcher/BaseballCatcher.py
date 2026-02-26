import threading
import time
import numpy as np
import cv2 as cv
from tkinter import *
from tkinter import font
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
from datetime import datetime
import os
from glob import glob
from pathlib import Path

# Use $ ls /dev/tty* to find Keyspan port name
PORT = '/dev/ttyUSB0'                   # Linux
#PORT = '/dev/tty.serial-10009FB44'      # Mac

'''
Set WEBCAM to 1 to use your webcam or 0 to use the Flea2 cameras on the lab machine
Set CATCHER to 1 to use the catcher connected to the lab machine or 0 to use your own computer
'''
WEBCAM = 0
CATCHER = 1
if WEBCAM:
    camera = cv.VideoCapture(0)
    width = int(camera.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))
else:
    from Flea2Camera2 import FleaCam
    camera = FleaCam()
    height, width, channel = camera.frame_shape

if CATCHER:
    from roboteq_wrapper import RoboteqWrapper                        
    from roboteq_constants import *

# Button Definitions
ON = 1
OFF = 0
SINGLE = 0
SEQUENCE = 1
LEFT = 0
RIGHT = 1
STEREO = 2
REPLAY_SPEED = 1.0
MAX_FRAMES = 50

def cvMat2tkImg(arr):  # Convert OpenCV image Mat to image for display
    rgb = cv.cvtColor(arr, cv.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    return ImageTk.PhotoImage(img)

class App(Frame):
    def __init__(self, winname='Baseball Catcher'):  # GUI Design
        self.root = Tk()
        self.processFlag = True
        self.acquireFlag = False
        self.firstSingle = True
        self.catchBallFlag = False
        self.frameIndex = 0
        self.imageSeq = []

        helv18 = font.Font(family='Helvetica', size=12, weight='bold')
        # print("Width",windowWidth,"Height",windowHeight)
        self.root.wm_title(winname)
        positionRight = int(self.root.winfo_screenwidth() / 2 - width / 2)
        positionDown = int(self.root.winfo_screenheight() / 2 - height / 2)
        # Positions the window in the center of the page.
        self.root.geometry("+{}+{}".format(positionRight, positionDown))
        self.root.wm_protocol("WM_DELETE_WINDOW", self.exitApp)
        # Frame.__init__(self, self.root)
        display1 = Frame(self.root)
        display2 = Frame(self.root)
        display3 = Frame(self.root)
        # capture and display the first frame
        if WEBCAM:                        
            ret0, frame = camera.read()
            half = cv.resize(frame, (int(width / 2), int(height / 2)))
            both = cv.hconcat([half, half])
        else:
            lframe, rframe = camera.getFrame()
            both = cv.hconcat([lframe, rframe])
        image = cvMat2tkImg(both)
        self.panel = Label(display1, image=image)
        self.panel.image = image
        self.panel.pack(side="top")
        display1.pack()

        # camera control
        # load and replay
        Button(display2, text="Load Sequence", width=12, height=1, command=self.loadSeq, font=helv18).grid(column=0, row=0)
        Button(display2, text="Replay Sequence", width=12, height=1, command=self.replaySeq, font=helv18).grid(column=0, row=1)
        self.maxFrames = StringVar()
        Label(display2, text='Max Frames:', width=12, font=helv18).grid(column=1, row=0, sticky=E)
        Entry(display2, textvariable=self.maxFrames, width=5, font=helv18).grid(column=2, row=0, sticky=W)
        self.maxFrames.set(str(MAX_FRAMES))
        Label(display2, text='Replay Speed:', width=12, font=helv18).grid(column=1, row=1, sticky=E)
        self.replaySpeed = DoubleVar()
        Scale(display2, from_=0.1, to=1.0, resolution=0.1, showvalue=False, width=20, length=80, variable=self.replaySpeed,
              font=helv18, orient='horizontal').grid(column=2, row=1, sticky=W)
        self.replaySpeed.set(1.0)
        # acquisition control
        Button(display2, text="Capture Frames", width=12, height=1, command=self.acquireSeq, font=helv18).grid(column=3, row=0)
        Button(display2, text="Save Sequence", width=12, height=1, command=self.saveSeq, font=helv18).grid(column=3, row=1)
        self.currentFrames = StringVar()
        Label(display2, width=3, textvariable=self.currentFrames, font=helv18).grid(column=4, row=0, sticky=W)
        self.currentFrames.set(0)
        self.acqMode = IntVar()
        Radiobutton(display2, text="Single", variable=self.acqMode, value=SINGLE, width=7, font=helv18).grid(column=5, row=0, sticky=W)
        Radiobutton(display2, text="Sequence", variable=self.acqMode, value=SEQUENCE, width=10, font=helv18).grid(
            column=6, row=0, columnspan=2, sticky=W)
        self.acqMode.set(SEQUENCE)
        self.saveMode = IntVar()
        Radiobutton(display2, text="Stereo", variable=self.saveMode, value=STEREO, width=7, font=helv18).grid(column=5, row=1, sticky=W)
        Radiobutton(display2, text="Left", variable=self.saveMode, value=LEFT, width=5, font=helv18).grid(column=6, row=1, sticky=W)
        Radiobutton(display2, text="Right", variable=self.saveMode, value=RIGHT, width=7, font=helv18).grid(column=7, row=1, sticky=W)
        self.saveMode.set(STEREO)
        # display checkbox                        
        self.disMode = IntVar()
        Checkbutton(display2, text='Display', variable=self.disMode, onvalue=ON, offvalue=OFF, width=8,
                    font=helv18).grid(column=8, row=0, sticky=W)
        self.disMode.set(ON)
        self.camMode = IntVar()
        Checkbutton(display2, text='Use Camera', variable=self.camMode, onvalue=ON, offvalue=OFF, width=12,
                    font=helv18).grid(column=8, row=1, columnspan=2, sticky=E)
        self.camMode.set(ON)  # set to use camera as default
        display2.pack(side='left', padx=20)

        # catcher control buttons
        Button(display3, text="Reset Catcher", width=10, height=2, command=self.resetCatcher, font=helv18).grid(column=0, row=0, rowspan=2)
        Button(display3, text="Move Home", width=10, height=1, command=self.moveHome, font=helv18).grid(column=1, row=0)
        Button(display3, text="Set Home", width=10, height=1, command=self.setHome, font=helv18).grid(column=1, row=1)
        # sliders
        self.sliderH = DoubleVar()
        self.sliderV = DoubleVar()
        if CATCHER:
            Scale(display3, from_=-CATCHER_W/2, to=CATCHER_W/2, resolution=0.1, length=255, width=10, variable=self.sliderH, font=helv18, orient='horizontal').grid(column=2, row=0)
            Scale(display3, from_=-CATCHER_H/2, to=CATCHER_H/2, resolution=0.1, length=255, width=10, variable=self.sliderV, font=helv18, orient='horizontal').grid(column=2, row=1)
        else:
            Scale(display3, from_=-10, to=10, resolution=0.1, length=255, width=10, variable=self.sliderH, font=helv18, orient='horizontal').grid(column=2, row=0)
            Scale(display3, from_=-10, to=10, resolution=0.1, length=255, width=10, variable=self.sliderV, font=helv18, orient='horizontal').grid(column=2, row=1)
        self.sliderH.set(0)
        self.sliderV.set(0)
        self.currentX = StringVar()
        Label(display3, width=5, textvariable=self.currentX, font=helv18).grid(column=3, row=0, sticky=W)
        self.currentX.set(0)
        self.currentY = StringVar()
        Label(display3, width=5, textvariable=self.currentY, font=helv18).grid(column=3, row=1, sticky=W)
        self.currentY.set(0)
        Button(display3, text="Move Catcher", width=10, height=1, command=self.moveCatcher, font=helv18).grid(column=4, row=0)
        Button(display3, text="Stop Catcher", width=10, height=1, command=self.stopCatcher, font=helv18).grid(column=4, row=1)
        Button(display3, text="Catch Ball", width=8, height=2, command=self.catchBall, font=helv18).grid(column=5, row=0)
        Button(display3, text="Stop Catch", width=8, height=2, command=self.stopCatch, font=helv18).grid(column=5, row=1)
        Button(display3, text="Quit", width=4, height=2, command=self.quitProgram, font=helv18).grid(column=6, row=0, rowspan=2)
        display3.pack(side='right', ipadx=10)
        if CATCHER:
            self.Catcher = RoboteqWrapper(PORT)
            if self.Catcher.Device.is_roboteq_connected:
                self.currentX.set(self.Catcher.GetEncoderCount(X_MOTOR))
                self.currentY.set(self.Catcher.GetEncoderCount(Y_MOTOR))

        # threading
        self.stopevent = threading.Event()
        self.thread = threading.Thread(target=self.process, args=())
        self.thread.start()                        

    def acquireSeq(self):
        if self.acqMode.get() == SINGLE:
            if self.firstSingle:
                self.imageSeq.clear()  # clear list to be safe
                self.frameIndex = 0
                self.firstSingle = False
            if self.frameIndex >= int(self.maxFrames.get()):
                if messagebox.askyesno('All frames acquired', 'Keep this image sequence?') == NO:
                    self.frameIndex = 0
                    self.currentFrames.set(str(self.frameIndex))
                    self.imageSeq.clear()
                return
        else:
            self.firstSingle = True
            self.imageSeq.clear()  # clear list to be safe
            self.frameIndex = 0
        self.camMode.set(1)         # Use camera input
        self.processFlag = True
        self.acquireFlag = True

    def saveSeq(self):
        if len(self.imageSeq) > 0:
            self.processFlag = False        # saving frames takes time, suspend process
            folder_selected = filedialog.askdirectory(initialdir=".")
            now = datetime.now()  # get date time
            path = os.path.join(folder_selected,
                                f"{now.year}{now.month:02d}{now.day:02d}{now.hour:02d}{now.minute:02d}{now.second:02d}")
            # path = os.path.join(folder_selected, "l")
            os.mkdir(path)  # make the main directory
            lpath = os.path.join(path, "L")  # make the L subdirectory
            os.mkdir(lpath)
            rpath = os.path.join(path, "R")  # make the R subdirectory
            os.mkdir(rpath)
            count = 0
            for both in self.imageSeq:
                l, r = both
                if self.saveMode.get() == LEFT or self.saveMode.get() == STEREO:  # save the left frame
                    l_img_name = '{}/{}.png'.format(lpath, count)
                    cv.imwrite(l_img_name, l)
                if self.saveMode.get() == RIGHT or self.saveMode.get() == STEREO:  # save the right frame
                    r_img_name = '{}/{}.png'.format(rpath, count)
                    cv.imwrite(r_img_name, r)
                count += 1
            self.processFlag = True         # resume process

    def loadSeq(self):
        path = filedialog.askdirectory(initialdir=".")      # select direrctory
        lpath = os.path.join(path, str("L"))  # get L folder
        rpath = os.path.join(path, str("R"))  # get R folder
        lfilelist = [y for x in os.walk(lpath) for y in glob(os.path.join(x[0], '*.png'))]  # read all png files in the directory
        lfilelist.sort(key=lambda x: int(Path(x).stem))  # extract and sort by the filename as integer
        rfilelist = [y for x in os.walk(rpath) for y in glob(os.path.join(x[0], '*.png'))]  # read all png files in the directory
        rfilelist.sort(key=lambda x: int(Path(x).stem))  # extract and sort by the filename as integer
        self.imageSeq.clear()  # clear the list to be safe
        if len(lfilelist) > 0 or len(rfilelist) > 0:     # one of them must have some images
            nFrames = len(lfilelist) if len(lfilelist) else len(rfilelist)
            for i in range(0, nFrames):  # there should be the same number of frames in both directories
                if len(lfilelist) > 0:
                    lframe = cv.imread(lfilelist[i])
                    if len(rfilelist) == 0:
                        rframe = lframe  # use left frame for right
                if len(rfilelist) > 0:
                    rframe = cv.imread(rfilelist[i])
                    if len(lfilelist) == 0:
                        lframe = rframe # use right frame for left
                both = lframe, rframe  # store both at the same time
                self.imageSeq.append(both)

    def replaySeq(self):
        if len(self.imageSeq) > 0:
            self.processFlag = False    # replay sequence outside the process
            self.frameIndex = 0
            self.timer = threading.Timer(1.0 - float(self.replaySpeed.get()), self.replay)  # start a timer (non-blocking) to give main thread time to stop
            self.timer.start()

    def replay(self):
        left, right = self.imageSeq[self.frameIndex]
        both = cv.hconcat([left, right])
        image = cvMat2tkImg(both)
        self.panel.configure(image=image)
        self.panel.image = image
        self.frameIndex += 1
        self.currentFrames.set(str(self.frameIndex))
        if self.frameIndex < MAX_FRAMES and self.frameIndex < len(self.imageSeq):
            self.timer = threading.Timer(1.0 - float(self.replaySpeed.get()), self.replay)  # start a timer (non-blocking) to give main thread time to stop
            self.timer.start()
        else:
            self.timer.cancel()
            self.processFlag = True     # resume process after replay

    def resetCatcher(self):
        if CATCHER:
            self.Catcher.SetToDefault()                        

    def moveHome(self):
        if CATCHER:
            self.Catcher.MoveAtSpeed(X_HOME_SPEED, Y_HOME_SPEED)

    def setHome(self):
        if CATCHER:
            self.Catcher.setHome()

    def moveCatcher(self):
        if CATCHER:
            self.Catcher.MoveToXY(self.sliderH.get(), self.sliderV.get())

    def stopCatcher(self):
        if CATCHER:
            self.Catcher.MoveAtSpeed(0, 0)  # set back to open loop speed mode and set speed to 0 to release the motors
            self.currentX.set(self.Catcher.GetEncoderCount(X_MOTOR))
            self.currentY.set(self.Catcher.GetEncoderCount(Y_MOTOR))

    def catchBall(self):
        self.processFlag = True     # Capture image
        self.camMode.set(ON)        # Get camera input
        self.disMode.set(OFF)       # Turn off display to go faster
        self.acquireFlag = False    # Don't acquire images into buffer
        self.catchBallFlag = True       # Set flag to start catching ball
        self.Catcher.MoveToXY(0, 0) # move the cathcer to the center or your desired starting location

    def stopCatch(self):
        self.disMode.set(ON)        # Turn on display
        self.catchBallFlag = False      # Set flag to stop catching ball
        if CATCHER:
            self.Catcher.MoveAtSpeed(0, 0)  # set back to open loop speed mode and set speed to 0 to release the motors
            self.currentX.set(self.Catcher.GetEncoderCount(X_MOTOR))
            self.currentY.set(self.Catcher.GetEncoderCount(Y_MOTOR))

    def run(self):  # run main loop
        self.root.mainloop()

    def quitProgram(self):  # click on the Quit button
        self.processFlag = False  # suspend process first before closing the program
        t = threading.Timer(0.5, self.quit)  # start a timer (non-blocking) to give main thread time to stop process
        t.start()

    def exitApp(self):  # click on the red cross (quit) button
        self.processFlag = False  # suspend process first before closing the program
        t = threading.Timer(0.5, self.quit)  # start a timer (non-blocking) to give main thread time to stop process
        t.start()

    def quit(self):
        self.stopevent.set()
        if CATCHER:
            del self.Catcher
        self.root.quit()

    def process(self):
        while not self.stopevent.is_set():
            if self.processFlag:
                if self.camMode.get():
                    if WEBCAM:
                        ret0, frame = camera.read()
                        lframe = cv.resize(frame, (int(width / 2), int(height / 2)))
                        rframe = cv.resize(frame, (int(width / 2), int(height / 2)))
                    else:
                        lframe, rframe = camera.getFrame()
                    if self.disMode.get():
                        both = cv.hconcat([lframe, rframe])
                        image = cvMat2tkImg(both)
                        self.panel.configure(image=image)
                        self.panel.image = image
                    if self.acquireFlag:
                        frames = lframe, rframe
                        self.imageSeq.append(frames)
                        self.frameIndex += 1
                        self.currentFrames.set(str(self.frameIndex))
                        if self.acqMode.get() == SINGLE or self.frameIndex == int(self.maxFrames.get()):
                            self.acquireFlag = False
                if self.catchBallFlag:
                    if CATCHER:
                        # add your code here to detect the ball

                        # add your code here to track the ball.

                        # add your code here to estimate the ball landing location after sufficient frames are acquired

                        # add your code to move the catcher to the estimated location

                        # wait for the move to be done


                        # Set to open loop and speed to 0 in case the catcher hits the safety stop
                        self.Catcher.MoveAtSpeed(0, 0)
                        # Move the cathcer back to the center or your desired starting location
                        self.Catcher.MoveToXY(0, 0)   # Center is (0,0) Positive X moves to left, Positive Y moves up.
            else:
                time.sleep(1)

app = App()
app.run()
# release the camera
if WEBCAM:
    camera.release()
else:
    del camera
