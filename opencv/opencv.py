import threading
import numpy as np
import cv2 as cv
from PIL import Image, ImageTk
from tkinter import Tk, Frame, Button, BOTH, Label, Scale, Radiobutton  # Graphical User Inetrface Stuff
from tkinter import font as tkFont
import tkinter as tk

camera = cv.VideoCapture(0)
width = int(camera.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))
videoout = cv.VideoWriter('./Video.avi', cv.VideoWriter_fourcc(*'XVID'), 25, (width, height))  # Video format

# Button Definitions
ORIGINAL = 0
BINARY = 1
EDGE = 2
LINE = 3
ABSDIFF = 4
RGB = 5
HSV = 6
CORNERS = 7
CONTOURS = 8

abs_dif_count = 0

def cvMat2tkImg(arr):  # Convert OpenCV image Mat to image for display
    rgb = cv.cvtColor(arr, cv.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    return ImageTk.PhotoImage(img)

def binarization(image, l, h):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    binary = cv.inRange(gray, l, h)
    frame = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
    return frame

def edges(image, l, h):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, l, h)
    frame = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    return frame

def corner(image, l, h):
    img = image.copy()
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    
    # find Harris corners
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray,2,3,0.04)
    dst = cv.dilate(dst,None)
    ret, dst = cv.threshold(dst,l,h,0)
    dst = np.uint8(dst)
    
    # find centroids
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
    
    # define the criteria to stop and refine the corners
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    centroids = centroids[1:] # remove background label
    corners = corners[1:]
    # Now draw them
    res = np.hstack((centroids,corners)).astype(int)
    for x0, y0, x1, y1 in res:
        cv.circle(img, (x0, y0), 4, (0, 0, 255), -1)   # original corner (red)
        cv.circle(img, (x1, y1), 4, (0, 255, 0), -1)   # sub-pixel refined (green)
    return img


def lines(image, l, h):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, l, h, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, np.pi/180,30,50,10)

    if lines is None:
        return image
    
    output = image.copy()
    for x1, y1, x2, y2 in lines[:, 0]:
        cv.line(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return output

prev_gray = None
def differencing(image):
    global prev_gray

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    if prev_gray is None:
        prev_gray = gray.copy()
        return prev_gray
    
    diff = cv.absdiff(gray, prev_gray)
    diff = cv.cvtColor(diff, cv.COLOR_GRAY2BGR)
    prev_gray = gray
    return diff

def contours(image, l, h):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # gray = cv.GaussianBlur(gray, (3,3), 0)
    _, thresh = cv.threshold(gray, l, h, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    output = image.copy()
    for i in contours:
        area = cv.contourArea(i)
        if area >= 300:
            x,y,w,h = cv.boundingRect(i)
            cv.rectangle(output, (x,y), (x+w, y+h), (100,100,0), 2)
    return output



class App(Frame):
    def __init__(self, winname='OpenCV'):  # GUI Design

        self.root = Tk()
        self.stopflag = True
        self.buffer = np.zeros((height, width, 3), dtype=np.uint8)

        global helv18
        helv18 = tkFont.Font(family='Helvetica', size=18, weight='bold')
        # print("Width",windowWidth,"Height",windowHeight)
        self.root.wm_title(winname)
        positionRight = int(self.root.winfo_screenwidth() / 2 - width / 2)
        positionDown = int(self.root.winfo_screenheight() / 2 - height / 2)
        # Positions the window in the center of the page.
        self.root.geometry("+{}+{}".format(positionRight, positionDown))
        self.root.wm_protocol("WM_DELETE_WINDOW", self.exitApp)
        Frame.__init__(self, self.root)
        self.pack(fill=BOTH, expand=1)
        # capture and display the first frame
        ret0, frame = camera.read()
        image = cvMat2tkImg(frame)
        self.panel = Label(image=image)
        self.panel.image = image
        self.panel.pack(side="top")
        # buttons
        global btnQuit
        btnQuit = Button(text="Quit", command=self.quit)
        btnQuit['font'] = helv18
        btnQuit.pack(side='right', pady=2)
        global btnStart
        btnStart = Button(text="Start", command=self.startstop)
        btnStart['font'] = helv18
        btnStart.pack(side='right', pady=2)
        # sliders
        global Slider1, Slider2
        Slider2 = Scale(self.root, from_=0, to=255, length=255, orient='horizontal')
        Slider2.pack(side='right')
        Slider2.set(192)
        Slider1 = Scale(self.root, from_=0, to=255, length=255, orient='horizontal')
        Slider1.pack(side='right')
        Slider1.set(64)
        # radio buttons
        global mode
        mode = tk.IntVar()
        mode.set(ORIGINAL)
        Radiobutton(self.root, text="Original", variable=mode, value=ORIGINAL).pack(side='left', pady=4)
        Radiobutton(self.root, text="Binary", variable=mode, value=BINARY).pack(side='left', pady=4)
        Radiobutton(self.root, text="Edge", variable=mode, value=EDGE).pack(side='left', pady=4)
        Radiobutton(self.root, text="Line", variable=mode, value=LINE).pack(side='left', pady=4)
        Radiobutton(self.root, text="Corners", variable=mode, value=CORNERS).pack(side='left', pady=4)
        Radiobutton(self.root, text="Abs Diff", variable=mode, value=ABSDIFF).pack(side='left', pady=4)
        Radiobutton(self.root, text="Contours", variable=mode, value=CONTOURS).pack(side='left', pady=4)
        # threading
        self.stopevent = threading.Event()
        self.thread = threading.Thread(target=self.capture, args=())
        self.thread.start()

    def capture(self):
        while not self.stopevent.is_set():
            if not self.stopflag:
                ret0, frame = camera.read()
                if mode.get() == BINARY:
                    lThreshold = Slider1.get()
                    hThreshold = Slider2.get()
                    # Add your code here
                    frame = binarization(frame, lThreshold, hThreshold)

                elif mode.get() == EDGE:
                    lThreshold = Slider1.get()
                    hThreshold = Slider2.get()
                    # Add your code here
                    frame = edges(frame, lThreshold, hThreshold)

                elif mode.get() == LINE:
                    lThreshold = Slider1.get()
                    hThreshold = Slider2.get()
                    # Add your code here
                    frame = lines(frame, lThreshold, hThreshold)

                elif mode.get() == CORNERS:
                    lThreshold = Slider1.get()
                    hThreshold = Slider2.get()
                    # Add your code here
                    frame = corner(frame, lThreshold, hThreshold)

                elif mode.get() == ABSDIFF:
                    lThreshold = Slider1.get()
                    hThreshold = Slider2.get()
                    # Add your code here
                    frame = differencing(frame)

                elif mode.get() == CONTOURS:
                    lThreshold = Slider1.get()
                    hThreshold = Slider2.get()
                    # Add your code here
                    frame = contours(frame, lThreshold, hThreshold)

                image = cvMat2tkImg(frame)
                self.panel.configure(image=image)
                self.panel.image = image
                videoout.write(frame)

    def startstop(self):  # toggle flag to start and stop
        if btnStart.config('text')[-1] == 'Start':
            btnStart.config(text='Stop')
        else:
            btnStart.config(text='Start')
        self.stopflag = not self.stopflag

    def run(self):  # run main loop
        self.root.mainloop()

    def quit(self):  # exit loop
        self.stopflag = True
        t = threading.Timer(1.0, self.stop)  # start a timer (non-blocking) to give main thread time to stop
        t.start()

    def exitApp(self):  # exit loop
        self.stopflag = True
        t = threading.Timer(1.0, self.stop)  # start a timer (non-blocking) to give main thread time to stop
        t.start()

    def stop(self):
        self.stopevent.set()
        self.root.quit()


app = App()
app.run()
# release the camera
camera.release()
