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

#ifndef TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_TRACKED_OBJECT_H_
#define TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_TRACKED_OBJECT_H_

#ifdef __RENDER_OPENGL__
#include "tensorflow/tools/android/test/jni/object_tracking/gl_utils.h"
#endif
#include "tensorflow/tools/android/test/jni/object_tracking/object_detector.h"

namespace tf_tracking {

// A TrackedObject is a specific instance of an ObjectModel, with a known
// position in the world.
// It provides the last known position and number of recent detection failures,
// in addition to the more general appearance data associated with the object
// class (which is in ObjectModel).
// TODO(andrewharp): Make getters/setters follow styleguide.
class TrackedObject {
 public:
  TrackedObject(const std::string& id, const Image<uint8_t>& image,
                const BoundingBox& bounding_box, ObjectModelBase* const model);

  ~TrackedObject();

  void UpdatePosition(const BoundingBox& new_position, const int64_t timestamp,
                      const ImageData& image_data, const bool authoritative);

  // This method is called when the tracked object is detected at a
  // given position, and allows the associated Model to grow and/or prune
  // itself based on where the detection occurred.
  void OnDetection(ObjectModelBase* const model,
                   const BoundingBox& detection_position,
                   const MatchScore match_score, const int64_t timestamp,
                   const ImageData& image_data);

  // Called when there's no detection of the tracked object. This will cause
  // a tracking failure after enough consecutive failures if the area under
  // the current bounding box also doesn't meet a minimum correlation threshold
  // with the model.
  void OnDetectionFailure() {}

  inline bool IsVisible() const {
    return tracked_correlation_ >= kMinimumCorrelationForTracking ||
        num_consecutive_frames_below_threshold_ < kMaxNumDetectionFailures;
  }

  inline float GetCorrelation() {
    return tracked_correlation_;
  }

  inline MatchScore GetMatchScore() {
    return tracked_match_score_;
  }

  inline BoundingBox GetPosition() const {
    return last_known_position_;
  }

  inline BoundingBox GetLastDetectionPosition() const {
    return last_detection_position_;
  }

  inline const ObjectModelBase* GetModel() const {
    return object_model_;
  }

  inline const std::string& GetName() const {
    return id_;
  }

  inline void Draw() const {
#ifdef __RENDER_OPENGL__
    if (tracked_correlation_ < kMinimumCorrelationForTracking) {
      glColor4f(MAX(0.0f, -tracked_correlation_),
                MAX(0.0f, tracked_correlation_),
                0.0f,
                1.0f);
    } else {
      glColor4f(MAX(0.0f, -tracked_correlation_),
                MAX(0.0f, tracked_correlation_),
                1.0f,
                1.0f);
    }

    // Render the box itself.
    BoundingBox temp_box(last_known_position_);
    DrawBox(temp_box);

    // Render a box inside this one (in case the actual box is hidden).
    const float kBufferSize = 1.0f;
    temp_box.left_ -= kBufferSize;
    temp_box.top_ -= kBufferSize;
    temp_box.right_ += kBufferSize;
    temp_box.bottom_ += kBufferSize;
    DrawBox(temp_box);

    // Render one outside as well.
    temp_box.left_ -= -2.0f * kBufferSize;
    temp_box.top_ -= -2.0f * kBufferSize;
    temp_box.right_ += -2.0f * kBufferSize;
    temp_box.bottom_ += -2.0f * kBufferSize;
    DrawBox(temp_box);
#endif
  }

  // Get current object's num_consecutive_frames_below_threshold_.
  inline int64_t GetNumConsecutiveFramesBelowThreshold() {
    return num_consecutive_frames_below_threshold_;
  }

  // Reset num_consecutive_frames_below_threshold_ to 0.
  inline void resetNumConsecutiveFramesBelowThreshold() {
    num_consecutive_frames_below_threshold_ = 0;
  }

  inline float GetAllowableDistanceSquared() const {
    return allowable_detection_distance_;
  }

 private:
  // The unique id used throughout the system to identify this
  // tracked object.
  const std::string id_;

  // The last known position of the object.
  BoundingBox last_known_position_;

  // The last known position of the object.
  BoundingBox last_detection_position_;

  // When the position was last computed.
  int64_t position_last_computed_time_;

  // The object model this tracked object is representative of.
  ObjectModelBase* object_model_;

  Image<float> last_detection_thumbnail_;

  Image<float> last_frame_thumbnail_;

  // The correlation of the object model with the preview frame at its last
  // tracked position.
  float tracked_correlation_;

  MatchScore tracked_match_score_;

  // The number of consecutive frames that the tracked position for this object
  // has been under the correlation threshold.
  int num_consecutive_frames_below_threshold_;

  float allowable_detection_distance_;

  friend std::ostream& operator<<(std::ostream& stream,
                                  const TrackedObject& tracked_object);

  TF_DISALLOW_COPY_AND_ASSIGN(TrackedObject);
};

inline std::ostream& operator<<(std::ostream& stream,
                                const TrackedObject& tracked_object) {
  stream << tracked_object.id_
      << " " << tracked_object.last_known_position_
      << " " << tracked_object.position_last_computed_time_
      << " " << tracked_object.num_consecutive_frames_below_threshold_
      << " " << tracked_object.object_model_
      << " " << tracked_object.tracked_correlation_;
  return stream;
}

}  // namespace tf_tracking

#endif  // TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_TRACKED_OBJECT_H_
