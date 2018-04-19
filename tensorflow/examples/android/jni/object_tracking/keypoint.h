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

#ifndef TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_KEYPOINT_H_
#define TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_KEYPOINT_H_

#include "tensorflow/examples/android/jni/object_tracking/geom.h"
#include "tensorflow/examples/android/jni/object_tracking/image-inl.h"
#include "tensorflow/examples/android/jni/object_tracking/image.h"
#include "tensorflow/examples/android/jni/object_tracking/logging.h"
#include "tensorflow/examples/android/jni/object_tracking/time_log.h"
#include "tensorflow/examples/android/jni/object_tracking/utils.h"

#include "tensorflow/examples/android/jni/object_tracking/config.h"

namespace tf_tracking {

// For keeping track of keypoints.
struct Keypoint {
  Keypoint() : pos_(0.0f, 0.0f), score_(0.0f), type_(0) {}
  Keypoint(const float x, const float y)
      : pos_(x, y), score_(0.0f), type_(0) {}

  Point2f pos_;
  float score_;
  uint8_t type_;
};

inline std::ostream& operator<<(std::ostream& stream, const Keypoint keypoint) {
  return stream << "[" << keypoint.pos_ << ", "
      << keypoint.score_ << ", " << keypoint.type_ << "]";
}

}  // namespace tf_tracking

#endif  // TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_KEYPOINT_H_
