#ifndef STITCH_RECTS_HPP
#define STITCH_RECTS_HPP

#include <vector>
#include <math.h>
#include <stdlib.h>

#include "./hungarian/hungarian.hpp"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

using std::vector;

class Rect {
 public:
  int cx_;
  int cy_;
  int width_;
  int height_;
  float confidence_;
  float true_confidence_;

  explicit Rect(int cx, int cy, int width, int height, float confidence) {
    cx_ = cx;
    cy_ = cy;
    width_ = width;
    height_ = height;
    confidence_ = confidence;
    true_confidence_ = confidence;
  }

  Rect(const Rect& other) {
    cx_ = other.cx_;
    cy_ = other.cy_;
    width_ = other.width_;
    height_ = other.height_;
    confidence_ = other.confidence_;
    true_confidence_ = other.true_confidence_;
  }

  bool overlaps(const Rect& other, float tau) const {
    if (fabs(cx_ - other.cx_) > (width_ + other.width_) / 1.5) {
      return false;
    } else if (fabs(cy_ - other.cy_) > (height_ + other.height_) / 2.0) {
      return false;
    } else {
      return iou(other) > tau;
    }
  }

  int distance(const Rect& other) const {
    return (fabs(cx_ - other.cx_) + fabs(cy_ - other.cy_) +
            fabs(width_ - other.width_) + fabs(height_ - other.height_));
  }

  float intersection(const Rect& other) const {
    int left = MAX(cx_ - width_ / 2., other.cx_ - other.width_ / 2.);
    int right = MIN(cx_ + width_ / 2., other.cx_ + other.width_ / 2.);
    int width = MAX(right - left, 0);

    int top = MAX(cy_ - height_ / 2., other.cy_ - other.height_ / 2.);
    int bottom = MIN(cy_ + height_ / 2., other.cy_ + other.height_ / 2.);
    int height = MAX(bottom - top, 0);
    return width * height;
  }

  int area() const {
    return height_ * width_;
  }

  float union_area(const Rect& other) const {
    return this->area() + other.area() - this->intersection(other);
  }

  float iou(const Rect& other) const {
    return this->intersection(other) / this->union_area(other);
  }

  bool operator==(const Rect& other) const {
    return (cx_ == other.cx_ && 
      cy_ == other.cy_ &&
      width_ == other.width_ &&
      height_ == other.height_ &&
      confidence_ == other.confidence_);
  }
};

void filter_rects(const vector<vector<vector<Rect> > >& all_rects,
                  vector<Rect>* stitched_rects,
                  float threshold,
                  float max_threshold,
                  float tau,
                  float conf_alpha);

#endif // STITCH_RECTS_HPP
