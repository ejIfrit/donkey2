#!/usr/bin/env python3
"""
Scripts to drive a donkey 2 car and train a model for it.

Usage:
    test.py (playback) [--tub=<tub1,tub2,..tubn>]  [--model=<model>] [--no_cache]
    test.py (edgeplayback) [--tub=<tub1,tub2,..tubn>]  [--model=<model>] [--no_cache]
    test.py (bothplayback) [--tub=<tub1,tub2,..tubn>]  [--model=<model>] [--no_cache]
    test.py (car) [--tub=<tub1,tub2,..tubn>]  [--model=<model>] [--no_cache]
    test.py (singleShot) [--tub=<tub1,tub2,..tubn>] [--tubInd=<tubInd>]

Options:
    -h --help        Show this screen.
    --tub TUBPATHS   List of paths to tubs. Comma separated. Use quotes to use wildcards. ie "~/tubs/*"
    --js             Use physical joystick.
"""

import os
import sys
from docopt import docopt
import cv2
import donkeycar as dk
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import numpy as np
import random
import glob
#import parts

from donkeycar.parts.datastore import Tub, TubHandler, TubGroup
from donkeycar.parts.controller import LocalWebController, JoystickController

print('imports done')


class playBackClass(object):
    def __init__(self):
        pass
    def run(self,args, parser):

        '''
        Load the images from a tub and create a movie from them.
        Movie
        '''

        if args.tub is None:
            print("ERR>> --tub argument missing.")
            parser.print_help()
            return

        conf = os.path.expanduser(args.config)
        if not os.path.exists(conf):
            print("No config file at location: %s. Add --config to specify\
                 location or run from dir containing config.py." % conf)
            return

        self.cfg = dk.load_config(conf)

        if args.type is None and args.model is not None:
            args.type = self.cfg.DEFAULT_MODEL_TYPE
            print("Model type not provided. Using default model type from config file")

        if args.salient:
            if args.model is None:
                print("ERR>> salient visualization requires a model. Pass with the --model arg.")
                parser.print_help()

            if args.type not in ['linear', 'categorical']:
                print("Model type {} is not supported. Only linear or categorical is supported for salient visualization".format(args.type))
                parser.print_help()
                return

        self.model_type = args.type
        self.tub = Tub(args.tub)

        start = args.start
        self.end_index = args.end if args.end != -1 else len(self.tub)
        num_frames = self.end_index - start

        # Move to the correct offset
        self.current = 0
        self.iterator = self.tub.__iter__()
        while self.current < start:
            self.iterator.next()
            self.current += 1


        userAngles = []
        nFrames = len(self.tub.get_index(shuffled=False))
        pilotAngles = np.ones(nFrames)*100
        for iRec in self.tub.get_index(shuffled=False):
            record = self.tub.get_record(iRec)
            userAngle = float(record["user/angle"])
            userAngle = (userAngle-0.5)*2*45*np.pi/180.
            userAngles.append(userAngle)



        self.index = self.tub.get_index(shuffled=False)
        fig = plt.figure()
        ax1 = plt.subplot2grid((3,2),(0,0))
        ax2 = plt.subplot2grid((3,2),(0,1))
        ax3 = plt.subplot2grid((3,2),(1,0),colspan=2)
        ax3.plot(userAngles)
        ax3.set_ylim([-90*np.pi/180,90*np.pi/180])
        pilotAnglePlot, = ax3.plot(pilotAngles,linestyle = '' ,marker = '.')
        currentPlot, =ax3.plot(0,userAngles[0],marker = 'o')
        # TODO: put slider in its own separate class
        axSliderFrame = plt.axes([0.2, 0.08, 0.4, 0.03], facecolor='gray')
        sFrame = Slider(axSliderFrame, 'Frame no.', 0,nFrames, valinit=0,valfmt='%d')
        frameNo = [0]
        frameCurrent = [0]
        def updateFrameSlider(val):
            frameNo[0] = int(sFrame.val)
            frameCurrent[0] = 0
        sFrame.on_changed(updateFrameSlider)

        record = self.tub.get_record(1)
        img = record["cam/image_array"]
        imPlot = ax1.imshow(img,animated=True)

        # get actual steering angle from the saved data
        steerLineTub, = ax1.plot([80,80],[0,80],color='orange')
        # get output steering angle from the driver
        steerLinePilot, = ax1.plot([80,80],[0,80],color='blue')
        imPlotProcessed = ax2.imshow(self.imProc(img),animated=True)
        # set up steering lines

        def animate(i):
            # sort out the slider
            actualFrame = (frameCurrent[0]+frameNo[0])%nFrames+1
            sFrame.vline.set_xdata([actualFrame,actualFrame])
            record = self.tub.get_record(actualFrame)
            img = record["cam/image_array"]
            userAngle = float(record["user/angle"])
            # turn user angle into an actual angle
            userAngle = (userAngle-0.5)*2*45*np.pi/180.
            #TODO: make a unified controller that takes in an image to get rid of this mess

            #TODO: sort this as it's showing actual angle (as it should) for pilot but not for user angle
            #if np.isnan(pilotAngle): pilotAngle = None
            if angle is not None: steerLinePilot.set_data([80,80+40*np.sin(angle)],[80,80-40*np.cos(angle)])
            if angle is not None:
                pilotAngles[actualFrame] = angle
                pilotAnglePlot.set_ydata(pilotAngles)



            steerLineTub.set_data([80,80+40*np.sin(userAngle)],[80,80-40*np.cos(userAngle)])
            imPlot.set_array(img)
            imPlotProcessed.set_array(self.imProc(img))
            currentPlot.set_data(actualFrame%nFrames, userAngle)
            frameCurrent[0]+=1
            return imPlot,
        ani = animation.FuncAnimation(fig, animate, np.arange(1, self.tub.get_num_records()),
                                  interval=100, blit=False)

        plt.show()






if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config()
    print(args)
    tub = args['--tub']
    model = args['--model']
    cache = not args['--no_cache']
    pb = playBackClass()
    pb.run(cfg, tub, model)
    
