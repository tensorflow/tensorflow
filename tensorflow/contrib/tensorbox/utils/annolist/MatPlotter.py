import os
import sys
import string
import matplotlib
matplotlib.use('Agg')
from pylab import *
import numpy as np

class MatPlotter:
    fontsize=15
    color=0
    colors=["r-", "b-", "k-", "c-", "m-", "y-"]
    colors+=[x + "-" for x in colors]
    colors+=["g-", "g--"]
    curFigure=[]
    legendNames=[]
    fontsizeLegend=14
    legendPlace='lower right'
    legendborderpad = None
    legendlabelsep = None


    def __init__(self, fontsize=15):
        # self.newFigure()
        self.fontsize=fontsize
        self.fontsizeLegend=fontsize - 1
        pass

    def formatLegend(self, newFontSize = 14, newPlace = 'lower right', borderpad = None, labelsep = None):
        self.fontsizeLegend=newFontSize
        self.legendPlace=newPlace
        self.legendborderpad = borderpad
        self.legendlabelsep = labelsep

    def newFigure(self, plotTitle="", fsize=rcParams['figure.figsize']):
        return self.newRPCFigure(plotTitle, fsize)

    def newRPCFigure(self, plotTitle="", fsize=rcParams['figure.figsize']):
        curFigure = figure(figsize=fsize)
        self.title = title(plotTitle, fontsize=self.fontsize)
        #subplots_adjust(left=0.085, right=0.975, top=0.975, bottom=0.085)
        subplots_adjust(right=0.975, top=0.975)

        #axis('equal')
        axis([0, 1, 0, 1])
        xticklocs, xticklabels = xticks(arange(0, 1.01, 0.1))
        setp(xticklabels, size=self.fontsize)
        yticklocs, yticklabels = yticks(arange(0, 1.01, 0.1))
        setp(yticklabels, size=self.fontsize)
        self.xlabel = xlabel("1-precision")
        self.xlabel.set_size(self.fontsize+2)
        self.ylabel = ylabel("recall")
        self.ylabel.set_size(self.fontsize+4)
        grid()
        hold(True)

    def newFPPIFigure(self, plotTitle="", fsize=rcParams['figure.figsize']):
        curFigure = figure(figsize=fsize)
        self.title = title(plotTitle, fontsize=self.fontsize)
        subplots_adjust(left=0.085, right=0.975, top=0.975, bottom=0.085)

        #axis('equal')
        axis([0, 100, 0, 1])
        xticklocs, xticklabels = xticks(arange(0, 100.01, 0.5))
        setp(xticklabels, size=self.fontsize)
        yticklocs, yticklabels = yticks(arange(0, 1.01, 0.1))
        setp(yticklabels, size=self.fontsize)
        self.xlabel = xlabel("false positives per image")
        self.xlabel.set_size(self.fontsize+2)
        self.ylabel = ylabel("recall")
        self.ylabel.set_size(self.fontsize+4)
        grid()
        hold(True)


    def newFreqFigure(self, plotTitle="", maxX = 10, maxY = 10,fsize=rcParams['figure.figsize']):
        curFigure = figure(figsize=fsize)
        self.title = title(plotTitle, fontsize=self.fontsize)
        subplots_adjust(left=0.085, right=0.975, top=0.975, bottom=0.1)
        #axis('equal')

        axis([0, maxX, 0, maxY])
        xticklocs, xticklabels = xticks(arange(0, maxX + 0.01, maxX * 1.0/ 10))
        setp(xticklabels, size=self.fontsize)
        yticklocs, yticklabels = yticks(arange(0, maxY + 0.01, maxY * 1.0/ 10))
        setp(yticklabels, size=self.fontsize)
        self.xlabel = xlabel("False positive / ground truth rect")
        self.xlabel.set_size(self.fontsize+2)
        self.ylabel = ylabel("True positives / ground truth rect")
        self.ylabel.set_size(self.fontsize+4)
        grid()
        hold(True)

    def newFPPWFigure(self, plotTitle="", fsize=rcParams['figure.figsize']):
        curFigure = figure(figsize=fsize)
        self.title = title(plotTitle, fontsize=self.fontsize)
        subplots_adjust(left=0.085, right=0.975, top=0.975, bottom=0.085)

        self.xlabel = xlabel("false positive per windows (FPPW)")
        self.xlabel.set_size(self.fontsize+2)
        self.ylabel = ylabel("miss rate")
        self.ylabel.set_size(self.fontsize+4)

        grid()
        hold(True)

    def newLogFPPIFigure(self, plotTitle="", fsize=rcParams['figure.figsize']):
        curFigure = figure(figsize=fsize)
        self.title = title(plotTitle, fontsize=self.fontsize)
        subplots_adjust(left=0.085, right=0.975, top=0.975, bottom=0.1)

        #axis('equal')

        self.xlabel = xlabel("false positives per image")
        self.xlabel.set_size(self.fontsize+2)
        self.ylabel = ylabel("miss rate")
        self.ylabel.set_size(self.fontsize+4)
        grid()
        hold(True)

    def loadRPCData(self, fname):
        self.filename = fname
        self.prec=[]
        self.rec=[]
        self.score=[]
        self.fppi=[]
        file = open(fname)

        precScores = []
        for i in range(1,10,1):
            precScores.append(100 - i * 10)

        fppiScores=[]
        for i in range(0, 500, 5):
            fppiScores.append(i * 1.0 / 100.0)



        precinfo = []
        fppiinfo = []
        eerinfo = []
        logAvInfo = []

        logAvMR= []
        self.lamr = 0;
        self.eer = None;
        firstLine = True
        leadingZeroCount = 0

        for line in file.readlines():
            vals = line.split()
            #vals=line.split(" ")
            #for val in vals:
            #       if val=="":
            #               vals.remove(val)
            self.prec.append(1-float(vals[0]))
            self.rec.append(float(vals[1]))
            self.score.append(float(vals[2]))

            if(len(vals)>3):
                self.fppi.append(float(vals[3]))
                if firstLine and not float(vals[3]) == 0:
                    firstLine = False

                    lamrcount = 1
                    self.lamr = 1 - float(vals[1])

                    lowest_fppi = math.ceil( math.log(float(vals[3]))/ math.log(10) * 10 )
                    print "lowest_fppi: ",lowest_fppi;

                    # MA: temporarily commented out
                    # for i in range(lowest_fppi, 1, 1):
                    #       logAvMR.append(10** (i * 1.0 / 10))

            #self.score.append(float(vals[2][:-1]))
            #print 1-self.prec[-1], self.rec[-1], self.score[-1]
            if (len(self.prec)>1):
                diff = (1-self.prec[-1]-self.rec[-1]) * (1-self.prec[-2]-self.rec[-2])
                if ( diff <0):
                    eerinfo.append( "EER between: %.03f and %.03f\tScore:%f" % (self.rec[-1], self.rec[-2], self.score[-1]))
                    self.eer = (self.rec[-1]+self.rec[-2]) * 0.5
                if ( diff == 0 and 1-self.prec[-1]-self.rec[-1]==0):
                    eerinfo.append( "EER: %.03f\tScore:%f" % (self.rec[-1], self.score[-1]))
                    self.eer = self.rec[-1]

            #Remove already passed precision
            if (len(precScores) > 0 and (float(vals[0])) < precScores[0] / 100.0):
                precinfo.append("%d percent precision score: %f, recall: %.03f" % (precScores[0], float(vals[2]), float(vals[1])))
                while(len(precScores) > 0 and precScores[0]/100.0 > float(vals[0])):
                    precScores.pop(0)

            #Remove already passed precision
            if(len(vals) > 3):
                if (len(fppiScores) > 0 and (float(vals[3])) > fppiScores[0]):
                    fppiinfo.append("%f fppi score: %f, recall: %.03f" % (fppiScores[0], float(vals[2]), float(vals[1])))
                    while(len(fppiScores) > 0 and fppiScores[0] < float(vals[3])):
                        fppiScores.pop(0)

                if (len(logAvMR) > 0 and (float(vals[3])) > logAvMR[0]):
                    while(len(logAvMR) > 0 and logAvMR[0] < float(vals[3])):
                        logAvInfo.append("%f fppi, miss rate: %.03f, score: %f" % (logAvMR[0], 1-float(vals[1]), float(vals[2])) )
                        self.lamr += 1-float(vals[1])
                        lamrcount += 1
                        logAvMR.pop(0)

                lastMR = 1-float(vals[1])


        if(len(vals)>3):
            for i in logAvMR:
                logAvInfo.append("%f fppi, miss rate: %.03f, extended" % (i, lastMR) )
                self.lamr += lastMR
                lamrcount += 1

        for i in precinfo:
            print i;
        print;
        for i in fppiinfo:
            print i;
        print
        for i in eerinfo:
            print i;
        print
        print "Recall at first false positive: %.03f" % self.rec[0]
        if(len(vals)>3):
            print
            for i in logAvInfo:
                print i;
            self.lamr =  self.lamr * 1.0 / lamrcount
            print "Log average miss rate in [10^%.01f, 10^0]: %.03f" % (lowest_fppi / 10.0, self.lamr )



        print; print
        file.close()

    def loadFreqData(self, fname):
        self.filename = fname
        self.prec=[]
        self.rec=[]
        self.score=[]
        file = open(fname)

        for line in file.readlines():
            vals = line.split()

            self.prec.append(float(vals[0]))
            self.rec.append(float(vals[1]))
            self.score.append(float(vals[2]))

        file.close()

    def loadFPPWData(self, fname):
        self.loadFreqData(fname)

    def finishPlot(self, axlimits = [0,1.0,0,1.0]):
        # MA:
        #self.legend = legend(self.legendNames, self.legendPlace, pad = self.legendborderpad, labelsep = self.legendlabelsep)
        self.legend = legend(self.legendNames, self.legendPlace)

        lstrings = self.legend.get_texts()
        setp(lstrings, fontsize=self.fontsizeLegend)
        #line= plot( [1 - axlimits[0], 0], [axlimits[3], 1 - axlimits[3] ] , 'k')
        line= plot( [1, 0], [0, 1] , 'k')

    def finishFreqPlot(self):
        self.legend = legend(self.legendNames, self.legendPlace, pad = self.legendborderpad, labelsep = self.legendlabelsep)
        lstrings = self.legend.get_texts()
        setp(lstrings, fontsize=self.fontsizeLegend)


    def show(self, plotEER = True, axlimits = [0,1.0,0,1.0]):
        if (plotEER):
            self.finishPlot(axlimits)
            axis(axlimits)
        else:
            self.finishFreqPlot()

        show()

    def saveCurrentFigure(self, plotEER, filename, axlimits = [0,1.0,0,1.0]):
        if (plotEER):
            self.finishPlot(axlimits)
            axis(axlimits)
        else:
            self.finishFreqPlot()

        print "Saving: " + filename
        savefig(filename)

    def plotRFP(self, numImages, fname, line="r-"):
        print 'NOT YET IMPLEMENTED'

    def plotRPC(self, fname, descr="line", style="-1", axlimits = [0,1.0,0,1.0], linewidth = 2, dashstyle = [], addEER = False ):
        self.loadRPCData(fname)

        #axis(axlimits);
        if (style=="-1"):
            if dashstyle != []:
                line = plot(self.prec, self.rec, self.colors[self.color], dashes = dashstyle)
            else:
                line = plot(self.prec, self.rec, self.colors[self.color])
            self.color=self.color+1
            self.color=self.color % len(self.colors)
        else:
            if dashstyle != []:
                line = plot(self.prec, self.rec, style, dashes = dashstyle)
            else:
                line = plot(self.prec, self.rec, style)

        axis(axlimits)

        if addEER and self.eer != None:
            descr += " (%.01f%%)" % (self.eer * 100)

        setp(line, 'linewidth', linewidth)
        self.legendNames= self.legendNames+[descr]

    def plotFPPI(self, fname, descr="line", style="-1", axlimits = [0,2,0,1], linewidth = 2, dashstyle = []):
        self.loadRPCData(fname)

        if (style=="-1"):
            if dashstyle != []:
                line = plot(self.fppi, self.rec, self.colors[self.color], dashes = dashstyle)
            else:
                line = plot(self.fppi, self.rec, self.colors[self.color])
            self.color=self.color+1
            self.color=self.color % len(self.colors)
        else:
            if dashstyle != []:
                line = plot(self.fppi, self.rec, style, dashes = dashstyle)
            else:
                line = plot(self.fppi, self.rec, style)

        axis(axlimits);

        setp(line, 'linewidth', linewidth)
        self.legendNames= self.legendNames+[descr]


    def plotFreq(self, fname, descr="line", style="-1", linewidth = 2, dashstyle = []):
        self.loadFreqData(fname)
        if (style=="-1"):
            if dashstyle != []:
                line = plot(self.prec, self.rec, self.colors[self.color], dashes = dashstyle)
            else:
                line = plot(self.prec, self.rec, self.colors[self.color])
            self.color=self.color+1
            self.color=self.color % len(self.colors)
        else:
            if dashstyle != []:
                line = plot(self.prec, self.rec, style, dashes = dashstyle)
            else:
                line = plot(self.prec, self.rec, style)


        setp(line, 'linewidth', linewidth)
        self.legendNames= self.legendNames+[descr]

    def plotFPPW(self, fname, descr="line", style="-1", axlimits = [5e-6, 1e0, 1e-2, 0.5], linewidth = 2, dashstyle = []):
        self.loadFPPWData(fname)
        if (style=="-1"):
            if dashstyle != []:
                line = loglog(self.prec, self.rec, self.colors[self.color], dashes = dashstyle)
            else:
                line = loglog(self.prec, self.rec, self.colors[self.color])
            self.color=self.color+1
            self.color=self.color % len(self.colors)
        else:
            if dashstyle != []:
                line = loglog(self.prec, self.rec, style, dashes = dashstyle)
            else:
                line = loglog(self.prec, self.rec, style)

        xticklocs, xticklabels = xticks([1e-5, 1e-4,1e-3, 1e-2, 1e-1, 1e0])
        setp(xticklabels, size=self.fontsize)
        yticklocs, yticklabels = yticks(array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]),
                                                                        ("0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08","0.09", "0.1", "0.2", "0.3", "0.4", "0.5"))
        setp(yticklabels, size=self.fontsize)

        axis(axlimits)

        gca().yaxis.grid(True, 'minor')
        setp(line, 'linewidth', linewidth)

        self.legendNames= self.legendNames+[descr]

    def plotLogFPPI(self, fname, descr="line", style="-1", axlimits = [5e-3, 1e1, 1e-1, 1], linewidth = 2, dashstyle = [], addlamr = False):
        self.loadRPCData(fname)
        if (style=="-1"):
            if dashstyle != []:
                line = loglog(self.fppi, [1 - x for x in self.rec], self.colors[self.color], dashes = dashstyle)
            else:
                line = loglog(self.fppi, [1 - x for x in self.rec], self.colors[self.color])

            self.color=(self.color+1) % len(self.colors)
        else:
            if dashstyle != []:
                line = loglog(self.fppi, [1 - x for x in self.rec], style, dashes = dashstyle)
            else:
                line = loglog(self.fppi, [1 - x for x in self.rec], style)

        gca().yaxis.grid(True, 'minor')

        m = min(self.fppi)
        lax = axlimits[0]
        for i in self.fppi:
            if(i != m):
                lax = math.floor(log(i)/math.log(10))
                leftlabel = math.pow(10, lax)
                break

        m = max(self.fppi)
        rightlabel = math.pow(10, math.ceil(log(m)/math.log(10))) + 0.01

        k = leftlabel
        ticks = [k]
        while k < rightlabel:
            k = k * 10
            ticks.append(k)

        xticklocs, xticklabels = xticks(ticks)
        setp(xticklabels, size=self.fontsize)
        yticklocs, yticklabels = yticks(arange(0.1, 1.01, 0.1), ("0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"))
        setp(yticklabels, size=self.fontsize)

        axlimits[0] = lax
        axis(axlimits)

        setp(line, 'linewidth', linewidth)

        if addlamr:
            descr += " (%.01f%%)" % (self.lamr * 100)

        self.legendNames= self.legendNames+[descr]
