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

#ifndef THIRD_PARTY_TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_FRAME_PAIR_H_
#define THIRD_PARTY_TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_FRAME_PAIR_H_

#include "tensorflow/examples/android/jni/object_tracking/keypoint.h"

namespace tf_tracking {

// A class that records keypoint correspondences from pairs of
// consecutive frames.
class FramePair {
 public:
  FramePair()
      : start_time_(0),
        end_time_(0),
        number_of_keypoints_(0) {}

  // Cleans up the FramePair so that they can be reused.
  void Init(const int64 start_time, const int64 end_time);

  void AdjustBox(const BoundingBox box,
                 float* const translation_x,
                 float* const translation_y,
                 float* const scale_x,
                 float* const scale_y) const;

 private:
  // Returns the weighted median of the given deltas, computed independently on
  // x and y. Returns 0,0 in case of failure. The assumption is that a
  // translation of 0.0 in the degenerate case is the best that can be done, and
  // should not be considered an error.
  //
  // In the case of scale,  a slight exception is made just to be safe and
  // there is a check for 0.0 explicitly, but that shouldn't ever be possible to
  // happen naturally because of the non-zero + parity checks in FillScales.
  Point2f GetWeightedMedian(const float* const weights,
                            const Point2f* const deltas) const;

  float GetWeightedMedianScale(const float* const weights,
                               const Point2f* const deltas) const;

  // Weights points based on the query_point and cutoff_dist.
  int FillWeights(const BoundingBox& box,
                  float* const weights) const;

  // Fills in the array of deltas with the translations of the points
  // between frames.
  void FillTranslations(Point2f* const translations) const;

  // Fills in the array of deltas with the relative scale factor of points
  // relative to a given center. Has the ability to override the weight to 0 if
  // a degenerate scale is detected.
  // Translation is the amount the center of the box has moved from one frame to
  // the next.
  int FillScales(const Point2f& old_center,
                 const Point2f& translation,
                 float* const weights,
                 Point2f* const scales) const;

  // TODO(andrewharp): Make these private.
 public:
  // The time at frame1.
  int64 start_time_;

  // The time at frame2.
  int64 end_time_;

  // This array will contain the keypoints found in frame 1.
  Keypoint frame1_keypoints_[kMaxKeypoints];

  // Contain the locations of the keypoints from frame 1 in frame 2.
  Keypoint frame2_keypoints_[kMaxKeypoints];

  // The number of keypoints in frame 1.
  int number_of_keypoints_;

  // Keeps track of which keypoint correspondences were actually found from one
  // frame to another.
  // The i-th element of this array will be non-zero if and only if the i-th
  // keypoint of frame 1 was found in frame 2.
  bool optical_flow_found_keypoint_[kMaxKeypoints];

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(FramePair);
};

}  // namespace tf_tracking

#endif  // THIRD_PARTY_TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_FRAME_PAIR_H_
