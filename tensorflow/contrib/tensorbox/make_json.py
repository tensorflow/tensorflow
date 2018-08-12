#A tool for hand labelling images
#Generates an IDL file
#pass in the directory where you store your images and a filename, then select the points on the images
#every time you hit next a line is generated
#the clear button removes are selected points on the current image
#when all files in the directory are processed, the idl file is written out

#ex: python make_idl.py train640x480 train.idl

#altered to output json
#ex: python make_json.py train640x480 train.json
#added button to skip an image
#enforce convention that rects are in top left, bottom right order
#correct name of image path object

import sys
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
import matplotlib.patches as patches
mpl.rcParams['toolbar'] = 'None'
from matplotlib.widgets import Button
from os import listdir
from os.path import isfile, join

json_images = []

top_corners = []
bottom_corners = []
patchCache = [] #the rectangles that get drawn on the image, stored so they can be removed in an orderly fashion

def removeAllPatches():
    for patch in patchCache:
        patch.remove()
    patchCache[:] = []

def skip(event):  #called when the skip button is hit
    global filename
    if len(onlyfiles) == 0:
        outfile.write(json.dumps(json_images, indent = 1))
        plt.close()
    else:
        filename = path + "/" + onlyfiles.pop()
        image = mpimg.imread(filename)
        imshow_obj.set_data(image)
        top_corners[:] = []
        bottom_corners[:] = []
        removeAllPatches()

def next(event):  #called when the next button is hit
    global filename
    global json_images
    
    rects = []

    one_decimal = "{0:0.1f}"
    for i in range(len(top_corners)):
        x1 = float(one_decimal.format(top_corners[i][0]))
        x2 = float(one_decimal.format(bottom_corners[i][0]))
        y1 = float(one_decimal.format(top_corners[i][1]))
        y2 = float(one_decimal.format(bottom_corners[i][1]))

        #enforce x1,y1 = top left, x2,y2 = bottom right

        tlx = min(x1,x2)
        tly = min(y1,y2)
        brx = max(x1,x2)
        bry = max(y1,y2)

        bbox = dict([("x1",tlx),("y1",tly),("x2",brx),("y2",bry)])
        rects.append(bbox)

    json_image = dict([("image_path",filename),("rects",rects)])

    json_images.append(json_image)

    progress_outfile.write(json.dumps(json_image, indent = 1))

    if len(onlyfiles) == 0:
        outfile.write(json.dumps(json_images, indent = 1))
        plt.close()
    else:
        filename = path + "/" + onlyfiles.pop()
        image = mpimg.imread(filename)
        imshow_obj.set_data(image)
        top_corners[:] = []
        bottom_corners[:] = []
        removeAllPatches()
    

def clear(event): #called when the clear button is hit
    top_corners[:] = []
    bottom_corners[:] = []
    removeAllPatches()

def onclick(event):  #called when anywhere inside the window is clicked
    if event.xdata > 1 and event.ydata > 1:
        if (len(top_corners) > len(bottom_corners)):
            bottom_corners.append([event.xdata,event.ydata])
            patchCache.append(patches.Rectangle((top_corners[-1][0], top_corners[-1][1])
                                           ,bottom_corners[-1][0] - top_corners[-1][0], bottom_corners[-1][1] - top_corners[-1][1],
                                           hatch='/',fill=False))
            ax.add_patch(patchCache[-1])
            plt.draw()
        else:
            top_corners.append([event.xdata,event.ydata])

def undo(event):  #called when the undo button is hit
    # Only act when a path was drawn
    if (len(top_corners) ==  len(bottom_corners)):
        bottom_corners.pop()
        top_corners.pop()
        to_remove = patchCache.pop()
        to_remove.remove()

ax = plt.gca()

#get our files for processing
if len(sys.argv) < 3:
    print "Too few params, try something like:  python make_json.py train640x480 train.json"
    exit()
path = sys.argv[1]
outfile_name = sys.argv[2]
outfile = open(outfile_name, 'w')
progress_outfile = open(outfile_name + "_work", 'w')



onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

#
filename = path + "/" + onlyfiles.pop()
image = mpimg.imread(filename)
imshow_obj = ax.imshow(image)

plt.axis("off")
fig = plt.gcf()
fig.canvas.mpl_connect('button_press_event', onclick)

#add the buttons to the bottom of the window
axundo = plt.axes([0.59, 0.01, 0.1, 0.075])
axnext = plt.axes([0.7, 0.01, 0.1, 0.075])
axclear = plt.axes([0.81, 0.01, 0.1, 0.075])
axskip = plt.axes([0.92, 0.01, 0.1, 0.075])
bundo = Button(axundo, 'Undo')
bundo.on_clicked(undo)
bnext = Button(axnext, 'Next')
bnext.on_clicked(next)
bclear = Button(axclear, 'Clear')
bclear.on_clicked(clear)
bskip = Button(axskip, 'Skip')
bskip.on_clicked(skip)
plt.show()

outfile.close()
progress_outfile.close()


print "finished"
