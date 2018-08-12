from libcpp.vector cimport vector
from libcpp.set cimport set

class PyRect(object):
    def __init__(self, cx, cy, width, height, confidence):
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.confidence = confidence
        self.true_confidence = confidence
    def overlaps(self, other):
        if abs(self.cx - other.cx) > (self.width + other.width) / 1.5:
            return False
        elif abs(self.cy - other.cy) > (self.height + other.height) / 2.0:
            return False
        else:
            return True
    def distance(self, other):
        return sum(map(abs, [self.cx - other.cx, self.cy - other.cy,
                       self.width - other.width, self.height - other.height]))
    def intersection(self, other):
        left = max(self.cx - self.width/2., other.cx - other.width/2.)
        right = min(self.cx + self.width/2., other.cx + other.width/2.)
        width = max(right - left, 0)
        top = max(self.cy - self.height/2., other.cy - other.height/2.)
        bottom = min(self.cy + self.height/2., other.cy + other.height/2.)
        height = max(bottom - top, 0)
        return width * height
    def area(self):
        return self.height * self.width
    def union(self, other):
        return self.area() + other.area() - self.intersection(other)
    def iou(self, other):
        return self.intersection(other) / self.union(other)
    def __eq__(self, other):
        return (self.cx == other.cx and 
            self.cy == other.cy and
            self.width == other.width and
            self.height == other.height and
            self.confidence == other.confidence)

cdef extern from "stitch_rects.hpp":
    cdef cppclass Rect:
        Rect(int cx, int cy, int width, int height, float confidence)
        int cx_
        int cy_
        int width_
        int height_
        float confidence_
        float true_confidence_

    cdef void filter_rects(vector[vector[vector[Rect] ] ]& all_rects,
                      vector[Rect]* stitched_rects,
                      float threshold,
                      float max_threshold,
                      float tau,
                      float conf_alpha);

def stitch_rects(all_rects, tau=0.25):
    """
    Implements the stitching procedure discussed in the paper. 
    Complicated, but we find that it does better than simpler versions
    and generalizes well across widely varying box sizes.

    Input:
        all_rects : 2d grid with each cell containing a vector of PyRects
    """
    for row in all_rects:
        assert len(row) == len(all_rects[0])
    
    cdef vector[vector[vector[Rect]]] c_rects
    cdef vector[vector[Rect]] c_row
    cdef vector[Rect] c_column
    for i, row in enumerate(all_rects):
        c_rects.push_back(c_row)
        for j, column in enumerate(row):
            c_rects[i].push_back(c_column)
            for py_rect in column:
                c_rects[i][j].push_back(
                    Rect(
                        py_rect.cx,
                        py_rect.cy,
                        py_rect.width,
                        py_rect.height,
                        py_rect.confidence)
                    )

    cdef vector[Rect] acc_rects;

    thresholds = [(.80, 1.0),
                  (.70, 0.9),
                  (.60, 0.8),
                  (.50, 0.7),
                  (.40, 0.6),
                  (.30, 0.5),
                  (.20, 0.4),
                  (.10, 0.3),
                  (.05, 0.2),
                  (.02, 0.1),
                  (.005, 0.04),
                  (.001, 0.01),
                  ]
    t_conf_alphas = [(tau, 1.0),
                     #(1 - (1 - tau) * 0.75, 0.5),
                     #(1 - (1 - tau) * 0.5, 0.1),
                     #(1 - (1 - tau) * 0.25, 0.005),
                     ]
    for t, conf_alpha in t_conf_alphas:
        for lower_t, upper_t in thresholds:
            if lower_t * conf_alpha > 0.0001:
                filter_rects(c_rects, &acc_rects, lower_t * conf_alpha,
                             upper_t * conf_alpha, t, conf_alpha)

    py_acc_rects = []
    for i in range(acc_rects.size()):
        acc_rect = PyRect(
            acc_rects[i].cx_,
            acc_rects[i].cy_,
            acc_rects[i].width_,
            acc_rects[i].height_,
            acc_rects[i].confidence_)
        acc_rect.true_confidence = acc_rects[i].true_confidence_
        py_acc_rects.append(acc_rect)
    return py_acc_rects
