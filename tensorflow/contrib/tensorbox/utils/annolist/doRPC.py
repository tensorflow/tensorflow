#!/usr/bin/env python

import os, sys
from AnnotationLib import *
from optparse import OptionParser
import copy
import math

# BASED ON WIKIPEDIA VERSION
# n - number of nodes
# C - capacity matrix
# F - flow matrix
# s - source
# t - sink
# sumC - sum over rows of C (too speed up computation)

def edmonds_karp(n, C, s, t, sumC):

    # Residual capacity from u to v is C[u][v] - F[u][v]
    F = [[0] * n for i in xrange(n)]
    while True:
        P = [-1] * n # Parent table
        P[s] = s
        M = [0] * n  # Capacity of path to node
        M[s] = float('infinity')
        Q = [s]      # BFS queue
        while Q:
            u = Q.pop(0)
            for v in xrange(n):
                # There is available capacity,
        # and v is not seen before in search
                if C[u][v] - F[u][v] > 0 and P[v] == -1:
                    P[v] = u
                    M[v] = min(M[u], C[u][v] - F[u][v])
                    if v != t:
                        if(sumC[u] > 0):
                            Q.append(v)
                    else:
                        # Backtrack search, and write flow
                        while P[v] != v:
                            u = P[v]
                            F[u][v] += M[t]
                            F[v][u] -= M[t]
                            v = u
                        Q = None
                        break
        if P[t] == -1: # We did not find a path to t
            return (F)

class AnnoGraph:

    def __init__(self, anno, det, ignore, style, minCover, minOverlap, maxDistance, ignoreOverlap):

        # setting rects
        #print anno.imageName
        self.anno = anno
        self.det = det
        self.det.sortByScore("descending")

        # generate initial graph
        self.n = len(det.rects)
        self.m = len(anno.rects)

        # Number of nodes = number of detections + number of GT + source + sink
        self.a = self.n + self.m + 2

        # Flow matrix
        self.F = [[0] * self.a for i in xrange(self.a)]
        # Capacity matrix
        self.C = [[0] * self.a for i in xrange(self.a)]

        # Connect source to all detections
        for i in range(1, self.n + 1):
            self.C[0][i] = 1
            self.C[i][0] = 1

        # Connect sink to all GT
        for i in range(self.n + 1, self.a - 1):
            self.C[i][self.a - 1] = 1
            self.C[self.a - 1][i] = 1

        # Overall flow
        self.full_flow = 0
        self.ignore_flow = 0

        # match rects / Adjacency matrix
        self.M = [[] for i in xrange(self.n)]
        self.match(style, minCover, minOverlap, maxDistance)
        self.nextN = 0

        # Deactivate All Non Matching detections
        # Save row sums for capacity matrix
        self.sumC = []
        self.sumC.append(self.n)
        for q in [len(self.M[j]) for j in xrange(len(self.M))]:
            self.sumC.append(q)
        for q in [1] * self.m:
            self.sumC.append(q)

        # Initially no links are active
        self.sumC_active = []
        self.sumC_active.append(self.n)
        for q in [len(self.M[j]) for j in xrange(len(self.M))]:
            self.sumC_active.append(0)
        for q in [1] * self.m:
            self.sumC_active.append(q)

        #
        self.ignore = [ 0 ] * self.m
        for ig in ignore.rects:
            for i, r in enumerate(anno.rects):
                if(ig.overlap_pascal(r) > ignoreOverlap):
                    self.ignore[i] = 1


    def match(self, style, minCover, minOverlap, maxDistance):
        for i in xrange(self.n):
            detRect = self.det.rects[i]
            for j in xrange(self.m):
                annoRect = self.anno.rects[j]

                # Bastian Leibe's matching style
                if(style == 0):
                    assert False;
                    if detRect.isMatchingStd(annoRect, minCover, minOverlap, maxDistance):
                        self.M[i].append(self.n + 1 + j)

                # Pascal Matching style
                if(style == 1):
                    if (detRect.isMatchingPascal(annoRect, minOverlap)):
                        self.M[i].append(self.n + 1 + j)

    def decreaseScore(self, score):
        capacity_change = False
        for i in xrange(self.nextN, self.n):
            if (self.det.rects[i].score >= score):
                capacity_change = self.insertIntoC(i + 1) or capacity_change
                self.nextN += 1
            else:
                break

        if capacity_change:
            self.F = edmonds_karp(self.a, self.C, 0, self.a - 1, self.sumC_active)
            self.full_flow = sum([self.F[0][i] for i in xrange(self.a)])
            self.ignore_flow = sum([self.F[i][self.a - 1] * self.ignore[i - 1 - self.n] for i in range(1 + self.n, 1 + self.n + self.m )])

        return capacity_change

    def addBB(self, rect):
        self.nextN += 1

        capacity_change = self.insertIntoC(rect.boxIndex + 1)

        if capacity_change:
            self.F = edmonds_karp(self.a, self.C, 0, self.a - 1, self.sumC_active)
            self.full_flow = sum([self.F[0][i] for i in xrange(self.a)])
            self.ignore_flow = sum([self.F[i][self.a - 1] * self.ignore[i - 1 - self.n] for i in range(1 + self.n, 1 + self.n + self.m )])

        return capacity_change

    def     insertIntoC(self, i):
        #print "Inserting node", i, self.det.rects[i-1].score, "of image", self.anno.imageName

        for match in self.M[i - 1]:
            #print "  match: ", match
            self.C[i][match] = 1
            self.C[match][i] = 1

        self.sumC_active[i] = self.sumC[i]

        return self.sumC[i] > 0

    def maxflow(self):
        return self.full_flow - self.ignore_flow

    def consideredDets(self):
        return self.nextN - self.ignore_flow

    def ignoredFlow(self):
        return self.ignore_flow

    def getTruePositives(self):
        ret = copy.copy(self.anno)
        ret.rects = []
        #iterate over GT
        for i in xrange(self.n + 1, self.a - 1):
            #Flow to sink > 0
            if(self.F[i][self.a - 1] > 0 and self.ignore[i - self.n - 1] == 0):
                #Find associated det
                for j in xrange(1, self.n + 1):
                    if(self.F[j][i] > 0):
                        ret.rects.append(self.det[j - 1])
                        break
        return ret

    def getIgnoredTruePositives(self):
        ret = copy.copy(self.anno)
        ret.rects = []
        #iterate over GT
        for i in xrange(self.n + 1, self.a - 1):
            #Flow to sink > 0
            if(self.F[i][self.a - 1] > 0 and self.ignore[i - self.n - 1] == 1):
                #Find associated det
                for j in xrange(1, self.n + 1):
                    if(self.F[j][i] > 0):
                        ret.rects.append(self.det[j - 1])
                        break
        return ret

    def getMissingRecall(self):
        ret = copy.copy(self.anno)
        ret.rects = []
        for i in xrange(self.n + 1, self.a - 1):
            if(self.F[i][self.a - 1] == 0 and self.ignore[i - self.n - 1] == 0):
                ret.rects.append(self.anno.rects[i - self.n - 1])
        return ret

    def getFalsePositives(self):
        ret = copy.copy(self.det)
        ret.rects = []
        for i in xrange(1, self.n + 1):
            if(self.F[0][i] == 0):
                ret.rects.append(self.det[i - 1])
        return ret

def asort(idlGT, idlDet, minWidth, minHeight, style, minCover, minOverlap, maxDistance, maxWidth=float('inf'), maxHeight=float('inf')):
    #Asort too small object in ground truth

    for x,anno in enumerate(idlGT):

        imageFound = False
        filterIndex = -1
        for i,filterAnno in enumerate(idlDet):
            if (suffixMatch(anno.imageName, filterAnno.imageName) and anno.frameNr == filterAnno.frameNr):
                filterIndex = i
                imageFound = True
                break

        if(not imageFound):
            continue

        validGTRects = []
        for j in anno.rects:
            if (j.width() >= minWidth) and (j.height() >= minHeight) and (j.width() <= maxWidth) and (j.height() <= maxHeight):
                validGTRects.append(j)
            else:
                # Sort out detections that would have matched
                matchingIndexes = []

                for m,frect in enumerate(idlDet[filterIndex].rects):
                    if(style == 0):
                        if (j.isMatchingStd(frect, minCover,minOverlap, maxDistance)):
                            overlap = j.overlap_pascal(frect)
                            matchingIndexes.append((m,overlap))

                    if(style == 1):
                        if(j.isMatchingPascal(frect, minOverlap)):
                            overlap = j.overlap_pascal(frect)
                            matchingIndexes.append((m, overlap))

                for m in xrange(len(matchingIndexes) - 1, -1, -1):
                    matching_rect = idlDet[filterIndex].rects[matchingIndexes[m][0]]
                    matching_overlap = matchingIndexes[m][1]
                    better_overlap_found = False
                    for l in anno.rects:
                        if l.overlap_pascal(matching_rect) > matching_overlap:
                            better_overlap_found = True

                    if better_overlap_found:
                        continue

                    del idlDet[filterIndex].rects[matchingIndexes[m][0]]

        idlGT[x].rects = validGTRects

    #Sort out too small false positives
    for x,anno in enumerate(idlDet):

        imageFound = False
        filterIndex = -1
        for i,filterAnno in enumerate(idlGT):
            if (suffixMatch(anno.imageName, filterAnno.imageName) and anno.frameNr == filterAnno.frameNr):
                filterIndex = i
                imageFound = True
                break

        if(not imageFound):
            continue

        validDetRects = []
        for j in anno.rects:
            if (j.width() >= minWidth) and (j.height() >= minHeight) and (j.width() <= maxWidth) and (j.height() <= maxHeight):
                validDetRects.append(j)
            else:
                for frect in idlGT[filterIndex].rects:

                    if(style == 0):
                        if j.isMatchingStd(frect, minCover,minOverlap, maxDistance):
                            validDetRects.append(j)

                    if(style == 1):
                        if(j.isMatchingPascal(frect, minOverlap)):
                            validDetRects.append(j)


        idlDet[x].rects = validDetRects


# MA: simplified version that does Pascal style matching with one parameter controlling "intersection-over-union" matching threshold
def comp_prec_recall(annoIDL, detIDL, minOverlap):
    ignoreIDL = copy.deepcopy(annoIDL)
    for anno in ignoreIDL:
        anno.rects = []

    precs, recalls, scores, fppi, graphs = comp_prec_recall_all_params(annoIDL, detIDL, ignoreIDL, minOverlap=minOverlap);
    return precs, recalls, scores, fppi

def comp_prec_recall_graphs(annoIDL, detIDL, minOverlap):
    ignoreIDL = copy.deepcopy(annoIDL)
    for anno in ignoreIDL:
        anno.rects = []

    precs, recalls, scores, fppi, graphs = comp_prec_recall_all_params(annoIDL, detIDL, ignoreIDL, minOverlap=minOverlap);
    return graphs


# MA: full version
def comp_prec_recall_all_params(annoIDL, detIDL, ignoreIDL, minWidth=0, minHeight=0, maxWidth=float('inf'), maxHeight=float('inf'),
                                matchingStyle=1, minCover=0.5, minOverlap=0.5, maxDistance=0.5, ignoreOverlap=0.9, verbose=False):

    # Asort detections which are too small/too big
    if verbose:
        print "Asorting too large/ too small detections"
        print "minWidth:", minWidth
        print "minHeight:", minHeight
        print "maxWidth: ", maxWidth
        print "maxHeight: ", maxHeight

    asort(annoIDL, detIDL, minWidth, minHeight, matchingStyle, minCover, minOverlap, maxDistance, maxWidth, maxHeight)

    #Debugging asort
    #saveIDL("testGT.idl", annoIDL)
    #saveIDL("testDET.idl", detIDL)



    noAnnotations = 0
    for anno in annoIDL:
        for j,detAnno in enumerate(detIDL):
            if (suffixMatch(anno.imageName, detIDL[j].imageName) and anno.frameNr == detIDL[j].frameNr):
                noAnnotations = noAnnotations + len(anno.rects)
                break

    if verbose:
        print "#Annotations:", noAnnotations

        ###--- set up graphs ---###
        print "Setting up graphs ..."

    graphs = []
    allRects = []
    missingFrames = 0
    for i in xrange(len(annoIDL)):

        imageFound = False
        filterIndex = -1

        for j, detAnno in enumerate(detIDL):
            if (suffixMatch(annoIDL[i].imageName, detIDL[j].imageName) and annoIDL[i].frameNr == detIDL[j].frameNr):
                filterIndex = j
                imageFound = True
                break

        if(not imageFound):
            print "No annotation/detection pair found for: " + annoIDL[i].imageName + " frame: " + str(annoIDL[i].frameNr)
            missingFrames += 1
            continue;

        graphs.append(AnnoGraph(annoIDL[i], detIDL[filterIndex], ignoreIDL[i], matchingStyle, minCover, minOverlap, maxDistance, ignoreOverlap))

        for j,rect in enumerate(detIDL[filterIndex]):
            newRect = detAnnoRect()
            newRect.imageName = anno.imageName
            newRect.frameNr = anno.frameNr
            newRect.rect = rect
            newRect.imageIndex = i - missingFrames
            newRect.boxIndex = j
            allRects.append(newRect)

    if verbose:
        print "missingFrames: ", missingFrames
        print "Number of detections on annotated frames: " , len(allRects)

        ###--- get scores from all rects ---###
        print "Sorting scores ..."

    allRects.sort(cmpDetAnnoRectsByScore)
    allRects.reverse()

    ###--- gradually decrease score ---###
    if verbose:
        print "Gradually decrease score ..."

    lastScore = float('infinity')

    precs = [1.0]
    recalls = [0.0]
    #fppi = [ 10**(math.floor(math.log(1.0 / float(len(annoIDL)))/math.log(10) * 10.0) / 10.0) ]
    fppi = [ 1.0 / float(len(annoIDL)) ]
    scores = [lastScore]

    numDet = len(allRects)
    sf = lastsf = 0
    cd = lastcd = 0
    iflow = lastiflow = 0

    changed = False
    firstFP = True
    for i,nextrect in enumerate(allRects):
        score = nextrect.rect.score;

        # updating true and false positive counts
        sf = sf - graphs[nextrect.imageIndex].maxflow()
        cd = cd - graphs[nextrect.imageIndex].consideredDets()
        iflow = iflow - graphs[nextrect.imageIndex].ignoredFlow()

        #changed = changed or graphs[nextrect.imageIndex].decreaseScore(score)
        changed = graphs[nextrect.imageIndex].addBB(nextrect) or changed
        sf = sf + graphs[nextrect.imageIndex].maxflow()
        cd = cd + graphs[nextrect.imageIndex].consideredDets()
        iflow = iflow + graphs[nextrect.imageIndex].ignoredFlow()

        if(firstFP and cd - sf != 0):
            firstFP = False
            changed = True

        if (i == numDet - 1 or score != allRects[i + 1].rect.score or firstFP or i == len(allRects)):
            if(changed or i == numDet - 1 or i == len(allRects)):

                if(lastcd > 0):
                    scores.append(lastScore)
                    recalls.append(float(lastsf) / float(noAnnotations - lastiflow))
                    precs.append(float(lastsf) / float(lastcd))
                    fppi.append(float(lastcd - lastsf) / float(len(annoIDL)))

                if (cd > 0):
                    scores.append(score)
                    recalls.append(float(sf) / float(noAnnotations - iflow))
                    precs.append(float(sf) / float(cd))
                    fppi.append(float(cd - sf) / float(len(annoIDL)))


            changed = False

        lastScore = score
        lastsf = sf
        lastcd = cd
        lastiflow = iflow

    return precs, recalls, scores, fppi, graphs

def main():

    parser = OptionParser(usage="usage: %prog [options] <groundTruthIdl> <detectionIdl>")

    parser.add_option("-o", "--outFile",
              action="store", type="string", dest="outFile")
    parser.add_option("-a", "--analysisFiles",
              action="store", type="string", dest="analysisFile")

    parser.add_option("-s", "--minScore",
              action="store", type="float", dest="minScore")

    parser.add_option("-w", "--minWidth",
                            action="store", type="int", dest="minWidth", default=0)

    parser.add_option("-u", "--minHeight",
                            action="store", type="int", dest="minHeight",default=0)
    parser.add_option("--maxWidth", action="store", type="float", dest="maxWidth", default=float('inf'))
    parser.add_option("--maxHeight", action="store", type="float", dest="maxHeight", default=float('inf'))

    parser.add_option("-r", "--fixAspectRatio",
                    action="store", type="float", dest="aspectRatio")

    parser.add_option("-p", "--Pascal-Style", action="store_true", dest="pascalStyle")
    parser.add_option("-l", "--Leibe-Seemann-Matching-Style", action="store_true", dest="leibeStyle")

    parser.add_option("--minCover", action="store", type="float", dest="minCover", default=0.5)
    parser.add_option("--maxDistance", action="store", type="float", dest="maxDistance", default=0.5)
    parser.add_option("--minOverlap", action="store", type="float", dest="minOverlap", default=0.5)


    parser.add_option("--clipToImageWidth", action="store", type="float", dest="clipWidth", default= None)
    parser.add_option("--clipToImageHeight", action="store", type="float", dest="clipHeight", default= None)

    parser.add_option("-d", "--dropFirst", action="store_true", dest="dropFirst")

    #parser.add_option("-c", "--class", action="store", type="int", dest="classID", default=-1)
    parser.add_option("-c", "--class", action="store", type="int", dest="classID", default = None)

    parser.add_option("-i", "--ignore", action="store", type="string", dest="ignoreFile")
    parser.add_option("--ignoreOverlap", action="store", type="float", dest="ignoreOverlap", default = 0.9)

    (options, args) = parser.parse_args()

    if (len(args) < 2):
        print "Please specify annotation and detection as arguments!"
        parser.print_help()
        sys.exit(1)

    annoFile = args[0]

    # First figure out the minimum height and width we are dealing with
    minWidth =  options.minWidth
    minHeight = options.minHeight
    maxWidth =  options.maxWidth
    maxHeight = options.maxHeight

    print "Minimum width: %d height: %d" % (minWidth, minHeight)

    # Load files
    annoIDL = parse(annoFile)
    detIDL = []
    for dets in args[1:]:
        detIDL += parse(dets)


    if options.ignoreFile != None:
        ignoreIDL = parse(options.ignoreFile)
    else:
        ignoreIDL = copy.deepcopy(annoIDL)
        for anno in ignoreIDL:
            anno.rects = []

    if(options.classID is not None):
        for anno in annoIDL:
            anno.rects = [rect for rect in anno.rects if (rect.classID == options.classID  or rect.classID == -1)]
        for anno in detIDL:
            anno.rects = [rect for rect in anno.rects if (rect.classID == options.classID or rect.classID == -1)]
        for anno in ignoreIDL:
            anno.rects = [rect for rect in anno.rects if (rect.classID == options.classID or rect.classID == -1)]

    # prevent division by zero when fixing aspect ratio
    for anno in annoIDL:
        anno.rects = [rect for rect in anno.rects if rect.width() > 0 and rect.height() > 0]
    for anno in detIDL:
        anno.rects = [rect for rect in anno.rects if rect.width() > 0 and rect.height() > 0]
    for anno in ignoreIDL:
        anno.rects = [rect for rect in anno.rects if rect.width() > 0 and rect.height() > 0]


    # Fix aspect ratio
    if (not options.aspectRatio == None):
        forceAspectRatio(annoIDL, options.aspectRatio)
        forceAspectRatio(detIDL, options.aspectRatio)
        forceAspectRatio(ignoreIDL, options.aspectRatio)

    # Deselect detections with too low score
    if (not options.minScore == None):
        for i,anno in enumerate(detIDL):
            validRects = []
            for rect in anno.rects:
                if (rect.score >= options.minScore):
                    validRects.append(rect)
            anno.rects = validRects

    # Clip detections to the image dimensions
    if(options.clipWidth != None or options.clipHeight != None):
        min_x = -float('inf')
        min_y = -float('inf')
        max_x = float('inf')
        max_y = float('inf')

        if(options.clipWidth != None):
            min_x = 0
            max_x = options.clipWidth
        if(options.clipHeight != None):
            min_y = 0
            max_y = options.clipHeight

        print "Clipping width: (%.02f-%.02f); clipping height: (%.02f-%.02f)" % (min_x, max_x, min_y, max_y)
        for anno in annoIDL:
            for rect in anno:
                rect.clipToImage(min_x, max_x, min_y, max_y)
        for anno in detIDL:
            for rect in anno:
                rect.clipToImage(min_x, max_x, min_y, max_y)

    # Setup matching style; standard is Pascal
    # style
    matchingStyle = 1

    # Pascal style
    if (options.pascalStyle == True):
        matchingStyle = 1

    if (options.leibeStyle == True):
        matchingStyle = 0

    if (options.pascalStyle and options.leibeStyle):
        print "Conflicting matching styles!"
        sys.exit(1)

    if (options.dropFirst == True):
        print "Drop first frame of each sequence..."
        newIDL = []
        for i, anno in enumerate(detIDL):
            if (i > 1 and detIDL[i].frameNr == detIDL[i-1].frameNr + 1 and detIDL[i].frameNr == detIDL[i-2].frameNr + 2 and  detIDL[i].frameNr == detIDL[i-3].frameNr + 3  and detIDL[i].frameNr == detIDL[i-4].frameNr + 4):
                newIDL.append(anno)
        detIDL = newIDL

    verbose = True;
    precs, recalls, scores, fppi, graphs = comp_prec_recall_all_params(annoIDL, detIDL, ignoreIDL,
                                                               minWidth=options.minWidth, minHeight=options.minHeight,
                                                               maxWidth=options.maxWidth, maxHeight=options.maxHeight,
                                                               matchingStyle=matchingStyle,
                                                               minCover=options.minCover, minOverlap=options.minOverlap,
                                                               maxDistance=options.maxDistance, ignoreOverlap=options.ignoreOverlap,
                                                               verbose=verbose);

    ###--- output to file ---###
    outfilename = options.outFile
    if outfilename is None:
        outputDir = os.path.dirname(os.path.abspath(args[1]))
        outputFile = os.path.basename(os.path.abspath(args[1]))
        [base, ext] = idlBase(outputFile)
        #outfilename = outputDir + "/rpc-" + base + ".txt"

        outfilename = outputDir + "/rpc-" + base + "_overlap" + str(options.minOverlap) + ".txt"

    print "saving:\n" + outfilename;

    file = open(outfilename, 'w')
    for i in xrange(len(precs)):
        file.write(str(precs[i])+" "+str(recalls[i])+" "+str(scores[i])+ " " + str(fppi[i])+ "\n")
    file.close()

    # Extracting failure cases
    if(options.analysisFile != None):

        anaPrefix = options.analysisFile

        falsePositives = AnnoList([])
        truePositives = AnnoList([])
        missingRecall = AnnoList([])
        ignoredTruePositives = AnnoList([])

        for i in xrange(len(graphs)):
            falsePositives.append(graphs[i].getFalsePositives())
            truePositives.append(graphs[i].getTruePositives())
            truePositives[-1].imageName = falsePositives[-1].imageName
            truePositives[-1].imagePath = falsePositives[-1].imagePath
            missingRecall.append(graphs[i].getMissingRecall())
            missingRecall[-1].imageName = falsePositives[-1].imageName
            missingRecall[-1].imagePath = falsePositives[-1].imagePath
            if options.ignoreFile != None:
                ignoredTruePositives.append(graphs[i].getIgnoredTruePositives())

        #saveIDL(anaPrefix + "-falsePositives.idl.gz", falsePositives);
        falsePositives.save(anaPrefix + "-falsePositives.pal")

        sortedFP = annoAnalyze(falsePositives);
        #saveIDL(anaPrefix + "-falsePositives-sortedByScore.idl.gz", sortedFP);
        #saveIDL(anaPrefix + "-truePositives.idl.gz", truePositives);

        # saveIDL(anaPrefix + "-falsePositives-sortedByScore.idl", sortedFP);
        # saveIDL(anaPrefix + "-truePositives.idl", truePositives);

        sortedFP.save(anaPrefix + "-falsePositives-sortedByScore.pal")
        truePositives.save(anaPrefix + "-truePositives.pal")

        sortedFP = annoAnalyze(truePositives);
        #saveIDL(anaPrefix + "-truePositives-sortedByScore.idl.gz", sortedFP);
        #saveIDL(anaPrefix + "-truePositives-sortedByScore.idl", sortedFP);
        sortedFP.save(anaPrefix + "-truePositives-sortedByScore.pal")

        if options.ignoreFile != None:
            #saveIDL(anaPrefix + "-ignoredTruePositives.idl.gz", ignoredTruePositives)
            #saveIDL(anaPrefix + "-ignoredTruePositives.idl", ignoredTruePositives)
            ignoredTruePositives.save(anaPrefix + "-ignoredTruePositives.pal")

        #saveIDL(anaPrefix + "-missingRecall.idl.gz", missingRecall);
        #saveIDL(anaPrefix + "-missingRecall.idl", missingRecall);
        missingRecall.save(anaPrefix + "-missingRecall.pal")


if __name__ == "__main__":
    main()
