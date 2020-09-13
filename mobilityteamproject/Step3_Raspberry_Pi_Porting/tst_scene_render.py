#!/usr/bin/env python


# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

from numpy import pi, sin, cos


defaultSize = 512

class TestSceneRender():

    def __init__(self, bgImg = None, fgImg = None,
        deformation = False, speed = 0.25, **params):
        self.time = 0.0
        self.timeStep = 1.0 / 30.0
        self.foreground = fgImg
        self.deformation = deformation
        self.speed = speed

        if bgImg is not None:
            self.sceneBg = bgImg.copy()
        else:
            self.sceneBg = np.zeros(defaultSize, defaultSize, np.uint8)

        self.w = self.sceneBg.shape[0]
        self.h = self.sceneBg.shape[1]

        if fgImg is not None:
            self.foreground = fgImg.copy()
            self.center = self.currentCenter = (int(self.w/2 - fgImg.shape[0]/2), int(self.h/2 - fgImg.shape[1]/2))

            self.xAmpl = self.sceneBg.shape[0] - (self.center[0] + fgImg.shape[0])
            self.yAmpl = self.sceneBg.shape[1] - (self.center[1] + fgImg.shape[1])

        self.initialRect = np.array([ (self.h/2, self.w/2), (self.h/2, self.w/2 + self.w/10),
         (self.h/2 + self.h/10, self.w/2 + self.w/10), (self.h/2 + self.h/10, self.w/2)]).astype(int)
        self.currentRect = self.initialRect

    def getXOffset(self, time):
        return int( self.xAmpl*cos(time*self.speed))


    def getYOffset(self, time):
        return int(self.yAmpl*sin(time*self.speed))

    def setInitialRect(self, rect):
        self.initialRect = rect

    def getRectInTime(self, time):

        if self.foreground is not None:
            tmp = np.array(self.center) + np.array((self.getXOffset(time), self.getYOffset(time)))
            x0, y0 = tmp
            x1, y1 = tmp + self.foreground.shape[0:2]
            return np.array([y0, x0, y1, x1])
        else:
            x0, y0 = self.initialRect[0] + np.array((self.getXOffset(time), self.getYOffset(time)))
            x1, y1 = self.initialRect[2] + np.array((self.getXOffset(time), self.getYOffset(time)))
            return np.array([y0, x0, y1, x1])

    def getCurrentRect(self):

        if self.foreground is not None:

            x0 = self.currentCenter[0]
            y0 = self.currentCenter[1]
            x1 = self.currentCenter[0] + self.foreground.shape[0]
            y1 = self.currentCenter[1] + self.foreground.shape[1]
            return np.array([y0, x0, y1, x1])
        else:
            x0, y0 = self.currentRect[0]
            x1, y1 = self.currentRect[2]
            return np.array([x0, y0, x1, y1])

    def getNextFrame(self):
        img = self.sceneBg.copy()

        if self.foreground is not None:
            self.currentCenter = (self.center[0] + self.getXOffset(self.time), self.center[1] + self.getYOffset(self.time))
            img[self.currentCenter[0]:self.currentCenter[0]+self.foreground.shape[0],
             self.currentCenter[1]:self.currentCenter[1]+self.foreground.shape[1]] = self.foreground
        else:
            self.currentRect = self.initialRect + np.int( 30*cos(self.time*self.speed) + 50*sin(self.time*self.speed))
            if self.deformation:
                self.currentRect[1:3] += int(self.h/20*cos(self.time))
            cv.fillConvexPoly(img, self.currentRect, (0, 0, 255))

        self.time += self.timeStep
        return img

    def resetTime(self):
        self.time = 0.0


def main():
    backGr = cv.imread(cv.samples.findFile('graf1.png'))
    fgr = cv.imread(cv.samples.findFile('box.png'))

    render = TestSceneRender(backGr, fgr)

    while True:

        img = render.getNextFrame()
        cv.imshow('img', img)

        ch = cv.waitKey(3)
        if  ch == 27:
            break

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
