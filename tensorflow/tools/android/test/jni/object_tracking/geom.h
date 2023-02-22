/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_TOOLS_ANDROID_TEST_JNI_OBJECT_TRACKING_GEOM_H_
#define TENSORFLOW_TOOLS_ANDROID_TEST_JNI_OBJECT_TRACKING_GEOM_H_

#include "tensorflow/tools/android/test/jni/object_tracking/logging.h"
#include "tensorflow/tools/android/test/jni/object_tracking/utils.h"

namespace tf_tracking {

struct Size {
  Size(const int width, const int height) : width(width), height(height) {}

  int width;
  int height;
};


class Point2f {
 public:
  Point2f() : x(0.0f), y(0.0f) {}
  Point2f(const float x, const float y) : x(x), y(y) {}

  inline Point2f operator- (const Point2f& that) const {
    return Point2f(this->x - that.x, this->y - that.y);
  }

  inline Point2f operator+ (const Point2f& that) const {
    return Point2f(this->x + that.x, this->y + that.y);
  }

  inline Point2f& operator+= (const Point2f& that) {
    this->x += that.x;
    this->y += that.y;
    return *this;
  }

  inline Point2f& operator-= (const Point2f& that) {
    this->x -= that.x;
    this->y -= that.y;
    return *this;
  }

  inline Point2f operator- (const Point2f& that) {
    return Point2f(this->x - that.x, this->y - that.y);
  }

  inline float LengthSquared() {
    return Square(this->x) + Square(this->y);
  }

  inline float Length() {
    return sqrtf(LengthSquared());
  }

  inline float DistanceSquared(const Point2f& that) {
    return Square(this->x - that.x) + Square(this->y - that.y);
  }

  inline float Distance(const Point2f& that) {
    return sqrtf(DistanceSquared(that));
  }

  float x;
  float y;
};

inline std::ostream& operator<<(std::ostream& stream, const Point2f& point) {
  stream << point.x << "," << point.y;
  return stream;
}

class BoundingBox {
 public:
  BoundingBox()
      : left_(0),
        top_(0),
        right_(0),
        bottom_(0) {}

  BoundingBox(const BoundingBox& bounding_box)
      : left_(bounding_box.left_),
        top_(bounding_box.top_),
        right_(bounding_box.right_),
        bottom_(bounding_box.bottom_) {
    SCHECK(left_ < right_, "Bounds out of whack! %.2f vs %.2f!", left_, right_);
    SCHECK(top_ < bottom_, "Bounds out of whack! %.2f vs %.2f!", top_, bottom_);
  }

  BoundingBox(const float left,
              const float top,
              const float right,
              const float bottom)
      : left_(left),
        top_(top),
        right_(right),
        bottom_(bottom) {
    SCHECK(left_ < right_, "Bounds out of whack! %.2f vs %.2f!", left_, right_);
    SCHECK(top_ < bottom_, "Bounds out of whack! %.2f vs %.2f!", top_, bottom_);
  }

  BoundingBox(const Point2f& point1, const Point2f& point2)
      : left_(MIN(point1.x, point2.x)),
        top_(MIN(point1.y, point2.y)),
        right_(MAX(point1.x, point2.x)),
        bottom_(MAX(point1.y, point2.y)) {}

  inline void CopyToArray(float* const bounds_array) const {
    bounds_array[0] = left_;
    bounds_array[1] = top_;
    bounds_array[2] = right_;
    bounds_array[3] = bottom_;
  }

  inline float GetWidth() const {
    return right_ - left_;
  }

  inline float GetHeight() const {
    return bottom_ - top_;
  }

  inline float GetArea() const {
    const float width = GetWidth();
    const float height = GetHeight();
    if (width <= 0 || height <= 0) {
      return 0.0f;
    }

    return width * height;
  }

  inline Point2f GetCenter() const {
    return Point2f((left_ + right_) / 2.0f,
                   (top_ + bottom_) / 2.0f);
  }

  inline bool ValidBox() const {
    return GetArea() > 0.0f;
  }

  // Returns a bounding box created from the overlapping area of these two.
  inline BoundingBox Intersect(const BoundingBox& that) const {
    const float new_left = MAX(this->left_, that.left_);
    const float new_right = MIN(this->right_, that.right_);

    if (new_left >= new_right) {
      return BoundingBox();
    }

    const float new_top = MAX(this->top_, that.top_);
    const float new_bottom = MIN(this->bottom_, that.bottom_);

    if (new_top >= new_bottom) {
      return BoundingBox();
    }

    return BoundingBox(new_left, new_top,  new_right, new_bottom);
  }

  // Returns a bounding box that can contain both boxes.
  inline BoundingBox Union(const BoundingBox& that) const {
    return BoundingBox(MIN(this->left_, that.left_),
                       MIN(this->top_, that.top_),
                       MAX(this->right_, that.right_),
                       MAX(this->bottom_, that.bottom_));
  }

  inline float PascalScore(const BoundingBox& that) const {
    SCHECK(GetArea() > 0.0f, "Empty bounding box!");
    SCHECK(that.GetArea() > 0.0f, "Empty bounding box!");

    const float intersect_area = this->Intersect(that).GetArea();

    if (intersect_area <= 0) {
      return 0;
    }

    const float score =
        intersect_area / (GetArea() + that.GetArea() - intersect_area);
    SCHECK(InRange(score, 0.0f, 1.0f), "Invalid score! %.2f", score);
    return score;
  }

  inline bool Intersects(const BoundingBox& that) const {
    return InRange(that.left_, left_, right_)
        || InRange(that.right_, left_, right_)
        || InRange(that.top_, top_, bottom_)
        || InRange(that.bottom_, top_, bottom_);
  }

  // Returns whether another bounding box is completely inside of this bounding
  // box. Sharing edges is ok.
  inline bool Contains(const BoundingBox& that) const {
    return that.left_ >= left_ &&
        that.right_ <= right_ &&
        that.top_ >= top_ &&
        that.bottom_ <= bottom_;
  }

  inline bool Contains(const Point2f& point) const {
    return InRange(point.x, left_, right_) && InRange(point.y, top_, bottom_);
  }

  inline void Shift(const Point2f shift_amount) {
    left_ += shift_amount.x;
    top_ += shift_amount.y;
    right_ += shift_amount.x;
    bottom_ += shift_amount.y;
  }

  inline void ScaleOrigin(const float scale_x, const float scale_y) {
    left_ *= scale_x;
    right_ *= scale_x;
    top_ *= scale_y;
    bottom_ *= scale_y;
  }

  inline void Scale(const float scale_x, const float scale_y) {
    const Point2f center = GetCenter();
    const float half_width = GetWidth() / 2.0f;
    const float half_height = GetHeight() / 2.0f;

    left_ = center.x - half_width * scale_x;
    right_ = center.x + half_width * scale_x;

    top_ = center.y - half_height * scale_y;
    bottom_ = center.y + half_height * scale_y;
  }

  float left_;
  float top_;
  float right_;
  float bottom_;
};
inline std::ostream& operator<<(std::ostream& stream, const BoundingBox& box) {
  stream << "[" << box.left_ << " - " << box.right_
         << ", " << box.top_ << " - " << box.bottom_
         << ",  w:" << box.GetWidth() << " h:" << box.GetHeight() << "]";
  return stream;
}


class BoundingSquare {
 public:
  BoundingSquare(const float x, const float y, const float size)
      : x_(x), y_(y), size_(size) {}

  explicit BoundingSquare(const BoundingBox& box)
      : x_(box.left_), y_(box.top_), size_(box.GetWidth()) {
#ifdef SANITY_CHECKS
    if (std::abs(box.GetWidth() - box.GetHeight()) > 0.1f) {
      LOG(WARNING) << "This is not a square: " << box << std::endl;
    }
#endif
  }

  inline BoundingBox ToBoundingBox() const {
    return BoundingBox(x_, y_, x_ + size_, y_ + size_);
  }

  inline bool ValidBox() {
    return size_ > 0.0f;
  }

  inline void Shift(const Point2f shift_amount) {
    x_ += shift_amount.x;
    y_ += shift_amount.y;
  }

  inline void Scale(const float scale) {
    const float new_size = size_ * scale;
    const float position_diff = (new_size - size_) / 2.0f;
    x_ -= position_diff;
    y_ -= position_diff;
    size_ = new_size;
  }

  float x_;
  float y_;
  float size_;
};
inline std::ostream& operator<<(std::ostream& stream,
                                const BoundingSquare& square) {
  stream << "[" << square.x_ << "," << square.y_ << " " << square.size_ << "]";
  return stream;
}


inline BoundingSquare GetCenteredSquare(const BoundingBox& original_box,
                                        const float size) {
  const float width_diff = (original_box.GetWidth() - size) / 2.0f;
  const float height_diff = (original_box.GetHeight() - size) / 2.0f;
  return BoundingSquare(original_box.left_ + width_diff,
                        original_box.top_ + height_diff,
                        size);
}

inline BoundingSquare GetCenteredSquare(const BoundingBox& original_box) {
  return GetCenteredSquare(
      original_box, MIN(original_box.GetWidth(), original_box.GetHeight()));
}

}  // namespace tf_tracking

#endif  // TENSORFLOW_TOOLS_ANDROID_TEST_JNI_OBJECT_TRACKING_GEOM_H_
