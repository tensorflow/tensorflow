#!/usr/bin/env python

import sys
import os
import random
import re
from AnnotationLib import *
from MatPlotter import *
from optparse import OptionParser
from copy import deepcopy
from math import sqrt


def main(argv):
	parser = OptionParser(usage="usage: %prog [options] <datafile> [...]")
	parser.add_option("-o", "--output-file",        action="store",
			dest="output", type="str", help="outfile. mandatory")
	parser.add_option("--fppw", action="store_true", dest="fppw", help="False Positives Per Window")
	parser.add_option("--colors", action="store", dest="colors", help="colors")
	parser.add_option("--fppi", action="store_true", dest="fppi", help="False Positives Per Image")
	parser.add_option("--lfppi", action="store_true", dest="lfppi", help="False Positives Per Image(log)")
	parser.add_option("-c", "--components", action="store", dest="ncomponents", type="int", help="show n trailing components of the part", default=3)
	parser.add_option("--cut-trailing", action="store", dest="cutcomponents", type="int", help="cut n trailing components of the part (applied after --components)", default=-1)
	parser.add_option("-t", "--title", action="store", dest="title", type="str", default="")
	parser.add_option("-f", "--fontsize", action="store", dest="fontsize", type="int", default=12)
	parser.add_option("-l", "--legend'", action="store", dest="legend", type="string", default="lr")
	(options, args) = parser.parse_args()
	plotter = MatPlotter(options.fontsize)
	
	position = "lower right"
	if(options.legend == "ur"):
		position = "upper right"
	if(options.legend == "ul"):
		position = "upper left"
	if(options.legend == "ll"):
		position = "lower left"	
	plotter.formatLegend(options.fontsize, newPlace = position)
	
	title = options.title
	colors = None
	if (options.colors):
		colors = options.colors.split()
	if (options.fppw):
		plotter.newFPPWFigure(title)
	elif (options.lfppi):
		plotter.newLogFPPIFigure(title)
	elif (options.fppi):
		plotter.newFPPIFigure(title)
	else:
		plotter.newFigure(title)		
		
	for i, filename in enumerate(args):
		if (os.path.isdir(filename)):
			filename = os.path.join(filename, "rpc", "result-minh-48")
		displayname = filename
		if (options.ncomponents > 0):
			suffix = None
			for idx in xrange(options.ncomponents):
				displayname, last = os.path.split(displayname)
				if (suffix):
					suffix = os.path.join(last, suffix)
				else:
					suffix = last
			displayname = suffix
		if (options.cutcomponents > 0):
			for idx in xrange(options.cutcomponents):
				displayname, last = os.path.split(displayname)
#		plusidx = displayname.index("+")
#		displayname = displayname[plusidx:]
		print "Plotting: "+displayname
		if (options.fppw):
			plotter.plotFPPW(filename, displayname)
		elif (options.lfppi):
			if colors:
				plotter.plotLogFPPI(filename, displayname, colors[i])
			else:
				plotter.plotLogFPPI(filename, displayname)
		elif (options.fppi):
			plotter.plotFPPI(filename, displayname)
		else:		
			plotter.plotRPC(filename, displayname)
	
	plotLine = not (options.fppw or options.lfppi or options.fppi);		
	
	if (options.output is None):
		plotter.show(plotLine)
	else:
		plotter.saveCurrentFigure(plotLine, options.output)
	return 0

if __name__ == "__main__":
	sys.exit(main(sys.argv))
