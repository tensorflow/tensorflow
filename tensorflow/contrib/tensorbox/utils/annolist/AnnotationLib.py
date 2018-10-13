import os

from math import sqrt

import gzip
import json
import bz2
import sys
import numpy as np;

from collections import MutableSequence

#import AnnoList_pb2
import PalLib;

import xml.dom.minidom
from xml.dom.minidom import Node
xml_dom_ext_available=False
try:
    import xml.dom.ext
    xml_dom_ext_available=True
except ImportError:
    pass


################################################
#
#  TODO: check distance function
#
################################################


def cmpAnnoRectsByScore(r1, r2):
    return cmp(r1.score, r2.score)

def cmpAnnoRectsByScoreDescending(r1, r2):
    return (-1)*cmp(r1.score, r2.score)

def cmpDetAnnoRectsByScore(r1, r2):
    return cmp(r1.rect.score, r2.rect.score);


def suffixMatch(fn1, fn2):
    l1 = len(fn1);
    l2 = len(fn2);
            
    if fn1[-l2:] == fn2:
        return True

    if fn2[-l1:] == fn1:
        return True

    return False            

class AnnoList(MutableSequence):
    """Define a list format, which I can customize"""
    TYPE_INT32 = 5;
    TYPE_FLOAT = 2;
    TYPE_STRING = 9;

    def __init__(self, data=None):
        super(AnnoList, self).__init__()

        self.attribute_desc = {};
        self.attribute_val_to_str = {};

        if not (data is None):
            self._list = list(data)
        else:
            self._list = list()     

    def add_attribute(self, name, dtype):
        _adesc = AnnoList_pb2.AttributeDesc();
        _adesc.name = name;
        if self.attribute_desc:
            _adesc.id = max((self.attribute_desc[d].id for d in self.attribute_desc)) + 1;
        else:
            _adesc.id = 0;

        if dtype == int:
            _adesc.dtype = AnnoList.TYPE_INT32;
        elif dtype == float or dtype == np.float32:
            _adesc.dtype = AnnoList.TYPE_FLOAT;
        elif dtype == str:
            _adesc.dtype = AnnoList.TYPE_STRING;
        else:
            print "unknown attribute type: ", dtype
            assert(False);
    
        #print "adding attribute: {}, id: {}, type: {}".format(_adesc.name, _adesc.id, _adesc.dtype);
        self.attribute_desc[name] = _adesc;

    def add_attribute_val(self, aname, vname, val):
        # add attribute before adding string corresponding to integer value
        assert(aname in self.attribute_desc);

        # check and add if new 
        if all((val_desc.id != val for val_desc in self.attribute_desc[aname].val_to_str)):
            val_desc = self.attribute_desc[aname].val_to_str.add()
            val_desc.id = val;
            val_desc.s = vname;

        # also add to map for quick access
        if not aname in self.attribute_val_to_str:
            self.attribute_val_to_str[aname] = {};

        assert(not val in self.attribute_val_to_str[aname]);
        self.attribute_val_to_str[aname][val] = vname;


    def attribute_get_value_str(self, aname, val):
        if aname in self.attribute_val_to_str and val in self.attribute_val_to_str[aname]:
            return self.attribute_val_to_str[aname][val];
        else:
            return str(val);

    def save(self, fname):
        save(fname, self);

    #MA: list interface   
    def __len__(self):
        return len(self._list)

    def __getitem__(self, ii):
        if isinstance(ii, slice):
            res = AnnoList();
            res.attribute_desc = self.attribute_desc;
            res._list = self._list[ii]
            return res;
        else:
            return self._list[ii]

    def __delitem__(self, ii):
        del self._list[ii]

    def __setitem__(self, ii, val):
        self._list[ii] = val
        return self._list[ii]

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return """<AnnoList %s>""" % self._list

    def insert(self, ii, val):
        self._list.insert(ii, val)

    def append(self, val):
        list_idx = len(self._list)
        self.insert(list_idx, val)


def is_compatible_attr_type(protobuf_type, attr_type):
    if protobuf_type == AnnoList.TYPE_INT32:
        return (attr_type == int);
    elif protobuf_type == AnnoList.TYPE_FLOAT:
        return (attr_type == float or attr_type == np.float32);
    elif protobuf_type == AnnoList.TYPE_STRING:
        return (attr_type == str);
    else:
        assert(false);


def protobuf_type_to_python(protobuf_type):
    if protobuf_type == AnnoList.TYPE_INT32:
        return int;
    elif protobuf_type == AnnoList.TYPE_FLOAT:
        return float;
    elif protobuf_type == AnnoList.TYPE_STRING:
        return str;
    else:
        assert(false);


class AnnoPoint(object):
    def __init__(self, x=None, y=None, id=None):
        self.x = x;
        self.y = y;
        self.id = id;

class AnnoRect(object):
    def __init__(self, x1=-1, y1=-1, x2=-1, y2=-1):

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        self.score = -1.0
        self.scale = -1.0
        self.articulations =[]
        self.viewpoints =[]
        self.d3 = []

        self.silhouetteID = -1
        self.classID = -1
        self.track_id = -1

        self.point = [];
        self.at = {};

    def width(self):
        return abs(self.x2-self.x1)

    def height(self):
        return abs(self.y2-self.y1)

    def centerX(self):
        return (self.x1+self.x2)/2.0

    def centerY(self):
        return (self.y1+self.y2)/2.0

    def left(self):
        return min(self.x1, self.x2)

    def right(self):
        return max(self.x1, self.x2)

    def top(self):
        return min(self.y1, self.y2)

    def bottom(self):
        return max(self.y1, self.y2)

    def forceAspectRatio(self, ratio, KeepHeight = False, KeepWidth = False):
        """force the Aspect ratio"""
        if KeepWidth or ((not KeepHeight) and self.width() * 1.0 / self.height() > ratio):
            # extend height
            newHeight = self.width() * 1.0 / ratio
            self.y1 = (self.centerY() - newHeight / 2.0)
            self.y2 = (self.y1 + newHeight)
        else:
            # extend width
            newWidth = self.height() * ratio
            self.x1 = (self.centerX() - newWidth / 2.0)
            self.x2 = (self.x1 + newWidth)
            
    def clipToImage(self, min_x, max_x, min_y, max_y):
        self.x1 = max(min_x, self.x1)
        self.x2 = max(min_x, self.x2)
        self.y1 = max(min_y, self.y1)
        self.y2 = max(min_y, self.y2)
        self.x1 = min(max_x, self.x1)
        self.x2 = min(max_x, self.x2)
        self.y1 = min(max_y, self.y1)
        self.y2 = min(max_y, self.y2)

    def printContent(self):
        print "Coords: ", self.x1, self.y1, self.x2, self.y2
        print "Score: ", self.score
        print "Articulations: ", self.articulations
        print "Viewpoints: ", self.viewpoints
        print "Silhouette: ", self.silhouetteID

    def ascii(self):
        r = "("+str(self.x1)+", "+str(self.y1)+", "+str(self.x2)+", "+str(self.y2)+")"
        if (self.score!=-1):
            r = r + ":"+str(self.score)
        if (self.silhouetteID !=-1):
            ri = r + "/"+str(self.silhouetteID)
        return r

    def writeIDL(self, file):
        file.write(" ("+str(self.x1)+", "+str(self.y1)+", "+str(self.x2)+", "+str(self.y2)+")")
        if (self.score!=-1):
            file.write(":"+str(self.score))
        if (self.silhouetteID !=-1):
            file.write("/"+str(self.silhouetteID))

    def writeJSON(self):
        jdoc = {"x1": self.x1, "x2": self.x2, "y1": self.y1, "y2": self.y2}
        
        if (self.score != -1):
            jdoc["score"] = self.score
        return jdoc

    def sortCoords(self):
        if (self.x1>self.x2):
            self.x1, self.x2 = self.x2, self.x1
        if (self.y1>self.y2):
            self.y1, self.y2 = self.y2, self.y1

    def rescale(self, factor):
        self.x1=(self.x1*float(factor))
        self.y1=(self.y1*float(factor))
        self.x2=(self.x2*float(factor))
        self.y2=(self.y2*float(factor))

    def resize(self, factor, factor_y = None):
        w = self.width()
        h = self.height()
        if factor_y is None:
            factor_y = factor
        centerX = float(self.x1+self.x2)/2.0
        centerY = float(self.y1+self.y2)/2.0
        self.x1 = (centerX - (w/2.0)*factor)
        self.y1 = (centerY - (h/2.0)*factor_y)
        self.x2 = (centerX + (w/2.0)*factor)
        self.y2 = (centerY + (h/2.0)*factor_y)
        

    def intersection(self, other):
        self.sortCoords()
        other.sortCoords()
        
        if(self.x1 >= other.x2):
            return (0, 0)           
        if(self.x2 <= other.x1):
            return (0, 0)
        if(self.y1 >= other.y2):
            return (0, 0)   
        if(self.y2 <= other.y1):
            return (0, 0)
    
        l = max(self.x1, other.x1);
        t = max(self.y1, other.y1);
        r = min(self.x2, other.x2);
        b = min(self.y2, other.y2);
        return (r - l, b - t)
        
        #Alternate implementation
        #nWidth  = self.x2 - self.x1
        #nHeight = self.y2 - self.y1
        #iWidth  = max(0,min(max(0,other.x2-self.x1),nWidth )-max(0,other.x1-self.x1))
        #iHeight = max(0,min(max(0,other.y2-self.y1),nHeight)-max(0,other.y1-self.y1))
        #return (iWidth, iHeight)

    def cover(self, other):
        nWidth = self.width()
        nHeight = self.height()
        iWidth, iHeight = self.intersection(other)              
        return float(iWidth * iHeight) / float(nWidth * nHeight)

    def overlap_pascal(self, other):
        self.sortCoords()
        other.sortCoords()

        nWidth  = self.x2 - self.x1
        nHeight = self.y2 - self.y1
        iWidth, iHeight = self.intersection(other)
        interSection = iWidth * iHeight
                
        union = self.width() * self.height() + other.width() * other.height() - interSection
                
        overlap = interSection * 1.0 / union
        return overlap

    def isMatchingPascal(self, other, minOverlap):
        overlap = self.overlap_pascal(other)
        if (overlap >= minOverlap and (self.classID == -1 or other.classID == -1 or self.classID == other.classID)):
            return 1
        else:
            return 0

    def distance(self, other, aspectRatio=-1, fixWH='fixheight'):
        if (aspectRatio!=-1):
            if (fixWH=='fixwidth'):
                dWidth  = float(self.x2 - self.x1)
                dHeight = dWidth / aspectRatio
            elif (fixWH=='fixheight'):
                dHeight = float(self.y2 - self.y1)
                dWidth  = dHeight * aspectRatio
        else:
            dWidth  = float(self.x2 - self.x1)
            dHeight = float(self.y2 - self.y1)

        xdist   = (self.x1 + self.x2 - other.x1 - other.x2) / dWidth
        ydist   = (self.y1 + self.y2 - other.y1 - other.y2) / dHeight

        return sqrt(xdist*xdist + ydist*ydist)

    def isMatchingStd(self, other, coverThresh, overlapThresh, distThresh, aspectRatio=-1, fixWH=-1):
        cover = other.cover(self)
        overlap = self.cover(other)
        dist = self.distance(other, aspectRatio, fixWH)

        #if(self.width() == 24 ):
        #print cover, " ", overlap, " ", dist
        #print coverThresh, overlapThresh, distThresh
        #print (cover>=coverThresh and overlap>=overlapThresh and dist<=distThresh)
        
        if (cover>=coverThresh and overlap>=overlapThresh and dist<=distThresh and self.classID == other.classID):
            return 1
        else:
            return 0

    def isMatching(self, other, style, coverThresh, overlapThresh, distThresh, minOverlap, aspectRatio=-1, fixWH=-1):
        #choose matching style
        if (style == 0):
            return self.isMatchingStd(other, coverThresh, overlapThresh, distThresh, aspectRatio=-1, fixWH=-1)

        if (style == 1):
            return self.isMatchingPascal(other, minOverlap)

    def addToXML(self, node, doc): # no Silhouette yet
        rect_el = doc.createElement("annorect")
        for item in "x1 y1 x2 y2 score scale track_id".split():
            coord_el = doc.createElement(item)
            coord_val = doc.createTextNode(str(self.__getattribute__(item)))
            coord_el.appendChild(coord_val)
            rect_el.appendChild(coord_el)
            
        articulation_el = doc.createElement("articulation")
        for articulation in self.articulations:
            id_el = doc.createElement("id")
            id_val = doc.createTextNode(str(articulation))
            id_el.appendChild(id_val)
            articulation_el.appendChild(id_el)
        if(len(self.articulations) > 0):
            rect_el.appendChild(articulation_el)
            
        viewpoint_el    = doc.createElement("viewpoint")
        for viewpoint in self.viewpoints:
            id_el = doc.createElement("id")
            id_val = doc.createTextNode(str(viewpoint))
            id_el.appendChild(id_val)
            viewpoint_el.appendChild(id_el)
        if(len(self.viewpoints) > 0):
            rect_el.appendChild(viewpoint_el)
    
        d3_el    = doc.createElement("D3")                                      
        for d in self.d3:
            id_el = doc.createElement("id")
            id_val = doc.createTextNode(str(d))
            id_el.appendChild(id_val)
            d3_el.appendChild(id_el)
        if(len(self.d3) > 0):
            rect_el.appendChild(d3_el)
                            
        if self.silhouetteID != -1:
            silhouette_el    = doc.createElement("silhouette")
            id_el = doc.createElement("id")
            id_val = doc.createTextNode(str(self.silhouetteID))
            id_el.appendChild(id_val)
            silhouette_el.appendChild(id_el)
            rect_el.appendChild(silhouette_el)

        if self.classID != -1:
            class_el    = doc.createElement("classID")
            class_val = doc.createTextNode(str(self.classID))
            class_el.appendChild(class_val)
            rect_el.appendChild(class_el)

        if len(self.point) > 0:
            annopoints_el = doc.createElement("annopoints")

            for p in self.point:
                point_el = doc.createElement("point");
                
                point_id_el = doc.createElement("id");
                point_id_val = doc.createTextNode(str(p.id));
                point_id_el.appendChild(point_id_val);
                point_el.appendChild(point_id_el);

                point_x_el = doc.createElement("x");
                point_x_val = doc.createTextNode(str(p.x));
                point_x_el.appendChild(point_x_val);
                point_el.appendChild(point_x_el);

                point_y_el = doc.createElement("y");
                point_y_val = doc.createTextNode(str(p.y));
                point_y_el.appendChild(point_y_val);
                point_el.appendChild(point_y_el);

                annopoints_el.appendChild(point_el);
        
            rect_el.appendChild(annopoints_el);
            
        node.appendChild(rect_el)



class Annotation(object):

    def __init__(self):
        self.imageName = ""
        self.imagePath = ""
        self.rects =[]
        self.frameNr = -1

    def clone_empty(self):
        new = Annotation()
        new.imageName = self.imageName
        new.imagePath = self.imagePath
        new.frameNr   = self.frameNr
        new.rects     = []
        return new

    def filename(self):
        return os.path.join(self.imagePath, self.imageName)

    def printContent(self):
        print "Name: ", self.imageName
        for rect in self.rects:
            rect.printContent()

    def writeIDL(self, file):
        if (self.frameNr == -1):
            file.write("\""+os.path.join(self.imagePath, self.imageName)+"\"")
        else:
            file.write("\""+os.path.join(self.imagePath, self.imageName)+"@%d\"" % self.frameNr)

        if (len(self.rects)>0):
            file.write(":")
        i=0
        for rect in self.rects:
            rect.writeIDL(file)
            if (i+1<len(self.rects)):
                file.write(",")
            i+=1

    def writeJSON(self):
        jdoc = {}
        jdoc['image_path'] = os.path.join(self.imagePath, self.imageName)
        jdoc['rects'] = []
        for rect in self.rects:
            jdoc['rects'].append(rect.writeJSON())
        return jdoc

    def addToXML(self, node, doc): # no frame# yet
        annotation_el = doc.createElement("annotation")
        img_el = doc.createElement("image")
        name_el = doc.createElement("name")             
        name_val = doc.createTextNode(os.path.join(self.imagePath, self.imageName))
        name_el.appendChild(name_val)
        img_el.appendChild(name_el)
        
        if(self.frameNr != -1):
            frame_el = doc.createElement("frameNr")
            frame_val = doc.createTextNode(str(self.frameNr))
            frame_el.appendChild(frame_val)
            img_el.appendChild(frame_el)
    
        annotation_el.appendChild(img_el)
        for rect in self.rects:
            rect.addToXML(annotation_el, doc)
        node.appendChild(annotation_el)


    def sortByScore(self, dir="ascending"):
        if (dir=="descending"):
            self.rects.sort(cmpAnnoRectsByScoreDescending)
        else:
            self.rects.sort(cmpAnnoRectsByScore)

    def __getitem__(self, index):
        return self.rects[index]

class detAnnoRect:
    def __init(self):
        self.imageName = ""
        self.frameNr = -1
        self.rect = AnnoRect()
        self.imageIndex = -1
        self.boxIndex = -1

#####################################################################
### Parsing

def parseTii(filename):

    # MA: this must be some really old code
    assert(False);
    annotations = []

    #--- parse xml ---#
    doc = xml.dom.minidom.parse(filename)

    #--- get tags ---#
    for file in doc.getElementsByTagName("file"):

        anno = Annotation()

        for filename in file.getElementsByTagName("filename"):
            aNode = filename.getAttributeNode("Src")
            anno.imageName = aNode.firstChild.data[:-4]+".png"

        for objects in file.getElementsByTagName("objects"):

            for vehicle in objects.getElementsByTagName("vehicle"):

                aNode = vehicle.getAttributeNode("Type")
                type = aNode.firstChild.data

                if (type=="pedestrian"):

                    rect = AnnoRect()
                    aNode = vehicle.getAttributeNode("FR")
                    frontrear = aNode.firstChild.data
                    aNode = vehicle.getAttributeNode("SD")
                    side = aNode.firstChild.data
                    if (frontrear == "1"):
                        orientation="FR"
                    elif (side == "1"):
                        orientation="SD"
                    aNode = vehicle.getAttributeNode( orientation+"_TopLeft_X")
                    rect.x1 = float(aNode.firstChild.data)
                    aNode = vehicle.getAttributeNode( orientation+"_TopLeft_Y")
                    rect.y1 = float(aNode.firstChild.data)
                    aNode = vehicle.getAttributeNode( orientation+"_BottomRight_X")
                    rect.x2 = float(aNode.firstChild.data)
                    aNode = vehicle.getAttributeNode( orientation+"_BottomRight_Y")
                    rect.y2 = float(aNode.firstChild.data)
                    print "pedestrian:", anno.imageName, rect.x1, rect.y1, rect.x2, rect.y2
                    anno.rects.append(rect)

        annotations.append(anno)

    return annotations

def parseXML(filename):
    filename = os.path.realpath(filename)

    name, ext = os.path.splitext(filename)

    annotations = AnnoList([])

    if(ext == ".al"):
        file = open(filename,'r')
        lines = file.read()
        file.close()

    if(ext == ".gz"):
        zfile = gzip.GzipFile(filename)
        lines = zfile.read()
        zfile.close()

    if(ext == ".bz2"):
        bfile = bz2.BZ2File(filename)
        lines = bfile.read()
        bfile.close()

    #--- parse xml ---#
    doc = xml.dom.minidom.parseString(lines)

    #--- get tags ---#
    for annotation in doc.getElementsByTagName("annotation"):
        anno = Annotation()
        for image in annotation.getElementsByTagName("image"):
            for name in image.getElementsByTagName("name"):
                anno.imageName = name.firstChild.data

            for fn in image.getElementsByTagName("frameNr"):
                anno.frameNr = int(fn.firstChild.data)

        rects = []
        for annoRect in annotation.getElementsByTagName("annorect"):
            rect = AnnoRect()

            for x1 in annoRect.getElementsByTagName("x1"):
                rect.x1 = float(x1.firstChild.data)

            for y1 in annoRect.getElementsByTagName("y1"):
                rect.y1 = float(y1.firstChild.data)

            for x2 in annoRect.getElementsByTagName("x2"):
                rect.x2 = float(x2.firstChild.data)

            for y2 in annoRect.getElementsByTagName("y2"):
                rect.y2 = float(y2.firstChild.data)

            for scale in annoRect.getElementsByTagName("scale"):
                rect.scale = float(scale.firstChild.data)

            for score in annoRect.getElementsByTagName("score"):
                rect.score = float(score.firstChild.data)

            for classID in annoRect.getElementsByTagName("classID"):
                rect.classID = int(classID.firstChild.data)

            for track_id in annoRect.getElementsByTagName("track_id"):
                rect.track_id = int(track_id.firstChild.data)

            for articulation in annoRect.getElementsByTagName("articulation"):
                for id in articulation.getElementsByTagName("id"):
                    rect.articulations.append(int(id.firstChild.data))
                #print "Articulations: ", rect.articulations

            for viewpoint in annoRect.getElementsByTagName("viewpoint"):
                for id in viewpoint.getElementsByTagName("id"):
                    rect.viewpoints.append(int(id.firstChild.data))
                    #print "Viewpoints: ", rect.viewpoints
                    
            for d in annoRect.getElementsByTagName("D3"):
                for id in d.getElementsByTagName("id"):
                    rect.d3.append(float(id.firstChild.data))

            for silhouette in annoRect.getElementsByTagName("silhouette"):
                for id in silhouette.getElementsByTagName("id"):
                    rect.silhouetteID = int(id.firstChild.data)
                #print "SilhouetteID: ", rect.silhouetteID

            for annoPoints in annoRect.getElementsByTagName("annopoints"):                          
                for annoPoint in annoPoints.getElementsByTagName("point"):

                    p = AnnoPoint();
                    for annoPointX in annoPoint.getElementsByTagName("x"):
                        p.x = int(float(annoPointX.firstChild.data));

                    for annoPointY in annoPoint.getElementsByTagName("y"):
                        p.y = int(float(annoPointY.firstChild.data));
                        
                    for annoPointId in annoPoint.getElementsByTagName("id"):
                        p.id = int(annoPointId.firstChild.data);

                    assert(p.x != None and p.y != None and p.id != None);
                    rect.point.append(p);                                   

            rects.append(rect)

        anno.rects = rects
        annotations.append(anno)

    return annotations

def parseJSON(filename):
    filename = os.path.realpath(filename)
    name, ext = os.path.splitext(filename)
    assert ext == '.json'

    annotations = AnnoList([])
    with open(filename, 'r') as f:
        jdoc = json.load(f)

    for annotation in jdoc:
        anno = Annotation()
        anno.imageName = annotation["image_path"]

        rects = []
        for annoRect in annotation["rects"]:
            rect = AnnoRect()

            rect.x1 = annoRect["x1"]
            rect.x2 = annoRect["x2"]
            rect.y1 = annoRect["y1"]
            rect.y2 = annoRect["y2"]
            if "score" in annoRect:
                rect.score = annoRect["score"]

            rects.append(rect)

        anno.rects = rects
        annotations.append(anno)

    return annotations
    
def parse(filename, abs_path=False):
    #print "Parsing: ", filename
    name, ext = os.path.splitext(filename)
    
    if (ext == ".gz" or ext == ".bz2"):
        name, ext = os.path.splitext(name)

    if(ext == ".idl"):
        annolist = parseIDL(filename)
    elif(ext == ".al"):
        annolist = parseXML(filename)
    elif(ext == ".pal"):
        annolist = PalLib.pal2al(PalLib.loadPal(filename));
    elif(ext == ".json"):
        annolist = parseJSON(filename)
    else:
        annolist = AnnoList([]);

    if abs_path:
        basedir = os.path.dirname(os.path.abspath(filename))
        for a in annolist:
            a.imageName = basedir + "/" + os.path.basename(a.imageName)

    return annolist


def parseIDL(filename):
    filename = os.path.realpath(filename)

    name, ext = os.path.splitext(filename)

    lines = []
    if(ext == ".idl"):
        file = open(filename,'r')
        lines = file.readlines()
        file.close()

    if(ext == ".gz"):
        zfile = gzip.GzipFile(filename)
        lines = zfile.readlines()
        zfile.close()

    if(ext == ".bz2"):
        bfile = bz2.BZ2File(filename)
        lines = bfile.readlines()
        bfile.close()

    annotations = AnnoList([])

    for line in lines:
        anno = Annotation()

        ### remove line break
        if (line[-1]=='\n'):
            line = line[:-1]; # remove '\n'
        lineLen = len(line)
        #print line

        ### get image name
        posImageEnd = line.find('\":')
        if (posImageEnd==-1):
            posImageEnd = line.rfind("\"")
        anno.imageName = line[1:posImageEnd]
        #print anno.imageName

        pos = anno.imageName.rfind("@")
        if (pos >= 0):
            anno.frameNr = int(anno.imageName[pos+1:])
            anno.imageName = anno.imageName[:pos]
            if anno.imageName[-1] == "/":
                anno.imageName = anno.imageName[:-1]
        else:
            anno.frameNr = -1

        ### get rect list
        # we split by ','. there are 3 commas for each rect and 1 comma seperating the rects
        rectSegs=[]
        if (posImageEnd!=-1 and posImageEnd+4<lineLen):

            line = line[posImageEnd+3:-1]; # remove ; or .

            segments = line.split(',')
            if (len(segments)%4!=0):
                print "Parse Errror"
            else:
                for i in range(0,len(segments),4):
                    rectSeg = segments[i]+","+segments[i+1]+","+segments[i+2]+","+segments[i+3]
                    rectSegs.append(rectSeg)
                    #print rectSegs

            ## parse rect segments
            for rectSeg in rectSegs:
                #print "RectSeg: ", rectSeg
                rect = AnnoRect()
                posBracket1 = rectSeg.find('(')
                posBracket2 = rectSeg.find(')')
                coordinates = rectSeg[posBracket1+1:posBracket2].split(',')
                #print coordinates
                #print "Coordinates: ",coordinates                              
                rect.x1 = float(round(float(coordinates[0].strip())))
                rect.y1 = float(round(float(coordinates[1].strip())))
                rect.x2 = float(round(float(coordinates[2].strip())))
                rect.y2 = float(round(float(coordinates[3].strip())))
                posColon = rectSeg.find(':')
                posSlash = rectSeg.find('/')
                if (posSlash!=-1):
                    rect.silhouetteID = int(rectSeg[posSlash+1:])
                else:
                    rectSeg+="\n"
                if (posColon!=-1):
                    #print rectSeg[posColon+1:posSlash]
                    rect.score = float(rectSeg[posColon+1:posSlash])
                anno.rects.append(rect)

        annotations.append(anno)

    return annotations



    

#####################################################################
### Saving

def save(filename, annotations):
    print "saving: ", filename;

    name, ext = os.path.splitext(filename)

    if (ext == ".gz" or ext == ".bz2"):
        name, ext = os.path.splitext(name)

    if(ext == ".idl"):
        return saveIDL(filename, annotations)           

    elif(ext == '.json'):
        return saveJSON(filename, annotations)

    elif(ext == ".al"):
        return saveXML(filename, annotations)

    elif(ext == ".pal"):
        return PalLib.savePal(filename, PalLib.al2pal(annotations));


    else:
        assert(False);
        return False;

def saveIDL(filename, annotations):
    [name, ext] = os.path.splitext(filename)

    if(ext == ".idl"):
        file = open(filename,'w')

    if(ext == ".gz"):
        file = gzip.GzipFile(filename, 'w')

    if(ext == ".bz2"):
        file = bz2.BZ2File(filename, 'w')

    i=0
    for annotation in annotations:
        annotation.writeIDL(file)
        if (i+1<len(annotations)):
            file.write(";\n")
        else:
            file.write(".\n")
        i+=1

    file.close()

def saveJSON(filename, annotations):
    [name, ext] = os.path.splitext(filename)

    jdoc = []
    for annotation in annotations:
        jdoc.append(annotation.writeJSON())

    with open(filename, 'w') as f:
        f.write(json.dumps(jdoc, indent=2, sort_keys=True))


def idlBase(filename):
    if (filename.rfind(".pal") == len(filename) - 4):
        return (filename[:-4], ".pal")

    if (filename.rfind(".json") == len(filename) - 5):
        return (filename[:-5], ".json")

    if (filename.rfind(".idl") == len(filename) - 4):
        return (filename[:-4], ".idl")

    if (filename.rfind(".al") == len(filename) - 3):
        return (filename[:-3], ".al")

    if (filename.rfind(".idl.gz") == len(filename) - 7):
        return (filename[:-7], ".idl.gz")

    if (filename.rfind(".idl.bz2") == len(filename) - 8):
        return (filename[:-8], ".idl.bz2")

    if (filename.rfind(".al.gz") == len(filename) - 6):
        return (filename[:-6], ".al.gz")

    if (filename.rfind(".al.bz2") == len(filename) - 7):
        return (filename[:-7], ".al.bz2")

def saveXML(filename, annotations):
    document = xml.dom.minidom.Document()
    rootnode = document.createElement("annotationlist")
    for anno in annotations:
        anno.addToXML(rootnode, document)
    document.appendChild(rootnode)
    [name, ext] = os.path.splitext(filename)
    if(ext == ".al"):
        writer = open(filename,'w')
    elif(ext == ".gz"):
        writer = gzip.GzipFile(filename, 'w')
    elif(ext == ".bz2"):
        writer = bz2.BZ2File(filename, 'w')
    else:
        print "invalid filename - .al(.gz|.bz2) is accepted"
        return


    if xml_dom_ext_available:
        xml.dom.ext.PrettyPrint(document, writer)
    else:
        # MA: skip header (currently Matlab's loadannotations can't deal with the header)
        document.documentElement.writexml(writer);

        #document.writexml(writer)

    document.unlink()





#####################################################################
### Statistics

def getStats(annotations):
    no = 0
    noTiny =0
    noSmall =0
    heights = []
    widths =[]

    ###--- get all rects ---###
    for anno in annotations:
        no = no + len(anno.rects)
        for rect in anno.rects:
            if (rect.height()<36):
                noTiny=noTiny+1
            if (rect.height()<128):
                noSmall=noSmall+1
            heights.append(rect.height())
            if (rect.width()==0):
                print "Warning: width=0 in image ", anno.imageName
                widths.append(1)
            else:
                widths.append(rect.width())
                if (float(rect.height())/float(rect.width())<1.5):
                    print "Degenerated pedestrian annotation: ", anno.imageName

    ###--- compute average height and variance ---###
    avgHeight = 0
    varHeight = 0


    minHeight = 0
    maxHeight = 0
    if len(heights) > 0:
        minHeight = heights[0]
        maxHeight = heights[0]

    for height in heights:
        avgHeight = avgHeight+height
        if (height > maxHeight):
            maxHeight = height
        if (height < minHeight):
            minHeight = height

    if (no>0):
        avgHeight = avgHeight/no
    for height in heights:
        varHeight += (height-avgHeight)*(height-avgHeight)
    if (no>1):
        varHeight=float(varHeight)/float(no-1)

    ###--- compute average width and variance ---###
    avgWidth = 0
    varWidth = 0
    for width in widths:
        avgWidth = avgWidth+width
    if (no>0):
        avgWidth = avgWidth/no
    for width in widths:
        varWidth += (width-avgWidth)*(width-avgWidth)

    if (no>1):
        varWidth=float(varWidth)/float(no-1)

    ###--- write statistics ---###
    print "  Total # rects:", no
    print "     avg. Width:", avgWidth, " (", sqrt(varWidth), "standard deviation )"
    print "    avg. Height:", avgHeight, " (", sqrt(varHeight), "standard deviation )"
    print "     tiny rects:", noTiny, " (< 36 pixels)"
    print "    small rects:", noSmall, " (< 128 pixels)"
    print "    minimum height:", minHeight
    print "    maximum height:", maxHeight

    ###--- return ---###
    return [widths, heights]

############################################################
##
##  IDL merging
##

def mergeIDL(detIDL, det2IDL, detectionFuse= True, minOverlap = 0.5):
    mergedIDL = []

    for i,anno in enumerate(detIDL):
        mergedAnno = Annotation()
        mergedAnno.imageName = anno.imageName
        mergedAnno.frameNr = anno.frameNr
        mergedAnno.rects = anno.rects

        imageFound = False
        filterIndex = -1
        for i,filterAnno in enumerate(det2IDL):
            if (suffixMatch(anno.imageName, filterAnno.imageName) and anno.frameNr == filterAnno.frameNr):
                filterIndex = i
                imageFound = True
                break

        if(not imageFound):
            mergedIDL.append(mergedAnno)
            continue

        for rect in det2IDL[filterIndex].rects:
            matches = False

            for frect in anno.rects:
                if rect.overlap_pascal(frect) > minOverlap:
                    matches = True
                    break

            if (not matches or detectionFuse == False):
                mergedAnno.rects.append(rect)

        mergedIDL.append(mergedAnno)

    return mergedIDL


############################################################################33
#
# Function to force the aspect ratio of annotations to ratio = width / height
#
#
def forceAspectRatio(annotations, ratio, KeepHeight = False, KeepWidth = False):
    for anno in annotations:
        for rect in anno.rects:
            rect.forceAspectRatio(ratio, KeepHeight, KeepWidth)
            #Determine which side needs to be extended
#                       if (rect.width() * 1.0 / rect.height() > ratio):
#
#                               #Too wide -> extend height
#                               newHeight = rect.width() * 1.0 / ratio
#                               rect.y1 = int(rect.centerY() - newHeight / 2.0)
#                               rect.y2 = int(rect.y1 + newHeight)
#
#                       else:
#                               #Too short -> extend width
#                               newWidth = rect.height() * ratio
#                               rect.x1 = int(rect.centerX() - newWidth / 2.0)
#                               rect.x2 = int(rect.x1 + newWidth)


###################################################################
# Function to greedyly remove subset detIDL from gtIDL
#
# returns two sets
#
# [filteredIDL, missingRecallIDL]
#
# filteredIDL == Rects that were present in both sets
# missingRecallIDL == Rects that were only present in set gtIDL
#
###################################################################
def extractSubSet(gtIDL, detIDL):
    filteredIDL = []
    missingRecallIDL = []

    for i,gtAnno in enumerate(gtIDL):
        filteredAnno = Annotation()
        filteredAnno.imageName = gtAnno.imageName
        filteredAnno.frameNr = gtAnno.frameNr

        missingRecallAnno = Annotation()
        missingRecallAnno.imageName = gtAnno.imageName
        missingRecallAnno.frameNr = gtAnno.frameNr

        imageFound = False
        filterIndex = -1
        for i,anno in enumerate(detIDL):
            if (suffixMatch(anno.imageName, gtAnno.imageName) and anno.frameNr == gtAnno.frameNr):
                filterIndex = i
                imageFound = True
                break

        if(not imageFound):
            print "Image not found " + gtAnno.imageName + " !"
            missingRecallIDL.append(gtAnno)
            filteredIDL.append(filteredAnno)
            continue

        matched = [-1] * len(detIDL[filterIndex].rects)
        for j, rect in enumerate(gtAnno.rects):
            matches = False

            matchingID = -1
            minCenterPointDist = -1
            for k,frect in enumerate(detIDL[filterIndex].rects):
                minCover = 0.5
                minOverlap = 0.5
                maxDist = 0.5

                if rect.isMatchingStd(frect, minCover,minOverlap, maxDist):
                    if (matchingID == -1 or rect.distance(frect) < minCenterPointDist):
                        matchingID = k
                        minCenterPointDist = rect.distance(frect)
                        matches = True

            if (matches):
                #Already matched once check if you are the better match
                if(matched[matchingID] >= 0):
                    #Take the match with the smaller center point distance
                    if(gtAnno.rects[matched[matchingID]].distance(frect) > rect.distance(frect)):
                        missingRecallAnno.rects.append(gtAnno.rects[matched[matchingID]])
                        filteredAnno.rects.remove(gtAnno.rects[matched[matchingID]])
                        filteredAnno.rects.append(rect)
                        matched[matchingID] = j
                    else:
                        missingRecallAnno.rects.append(rect)
                else:
                    #Not matched before.. go on and add the match
                    filteredAnno.rects.append(rect)
                    matched[matchingID] = j
            else:
                missingRecallAnno.rects.append(rect)

        filteredIDL.append(filteredAnno)
        missingRecallIDL.append(missingRecallAnno)

    return (filteredIDL     , missingRecallIDL)

###########################################################
#
#  Function to remove all detections with a too low score
#
#
def filterMinScore(detections, minScore):
    newDetections = []
    for anno in detections:
        newAnno = Annotation()
        newAnno.frameNr = anno.frameNr
        newAnno.imageName = anno.imageName
        newAnno.imagePath = anno.imagePath
        newAnno.rects = []

        for rect in anno.rects:
            if(rect.score >= minScore):
                newAnno.rects.append(rect)

        newDetections.append(newAnno)
    return newDetections

# foo.idl -> foo-suffix.idl, foo.idl.gz -> foo-suffix.idl.gz etc
def suffixIdlFileName(filename, suffix):
    exts = [".idl", ".idl.gz", ".idl.bz2"]
    for ext in exts:
        if filename.endswith(ext):
            return filename[0:-len(ext)] + "-" + suffix + ext
    raise ValueError("this does not seem to be a valid filename for an idl-file")

if __name__ == "__main__":
# test output
    idl = parseIDL("/tmp/asdf.idl")
    idl[0].rects[0].articulations = [4,2]
    idl[0].rects[0].viewpoints = [2,3]
    saveXML("", idl)


def annoAnalyze(detIDL):
    allRects = []
    
    for i,anno in enumerate(detIDL):
        for j in anno.rects:
            newRect = detAnnoRect()
            newRect.imageName = anno.imageName
            newRect.frameNr = anno.frameNr
            newRect.rect = j
            allRects.append(newRect)

    allRects.sort(cmpDetAnnoRectsByScore)
    
    filteredIDL = AnnoList([])
    for i in allRects:
        a = Annotation()
        a.imageName = i.imageName
        a.frameNr = i.frameNr
        a.rects = []
        a.rects.append(i.rect)
        filteredIDL.append(a)
        
    return filteredIDL


