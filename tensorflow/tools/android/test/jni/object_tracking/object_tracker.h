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

#ifndef TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_OBJECT_TRACKER_H_
#define TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_OBJECT_TRACKER_H_

#include <map>
#include <string>

#include "tensorflow/tools/android/test/jni/object_tracking/config.h"
#include "tensorflow/tools/android/test/jni/object_tracking/flow_cache.h"
#include "tensorflow/tools/android/test/jni/object_tracking/geom.h"
#include "tensorflow/tools/android/test/jni/object_tracking/integral_image.h"
#include "tensorflow/tools/android/test/jni/object_tracking/keypoint_detector.h"
#include "tensorflow/tools/android/test/jni/object_tracking/logging.h"
#include "tensorflow/tools/android/test/jni/object_tracking/object_model.h"
#include "tensorflow/tools/android/test/jni/object_tracking/optical_flow.h"
#include "tensorflow/tools/android/test/jni/object_tracking/time_log.h"
#include "tensorflow/tools/android/test/jni/object_tracking/tracked_object.h"
#include "tensorflow/tools/android/test/jni/object_tracking/utils.h"

namespace tf_tracking {

typedef std::map<const std::string, TrackedObject*> TrackedObjectMap;

inline std::ostream& operator<<(std::ostream& stream,
                                const TrackedObjectMap& map) {
  for (TrackedObjectMap::const_iterator iter = map.begin();
      iter != map.end(); ++iter) {
    const TrackedObject& tracked_object = *iter->second;
    const std::string& key = iter->first;
    stream << key << ": " << tracked_object;
  }
  return stream;
}


// ObjectTracker is the highest-level class in the tracking/detection framework.
// It handles basic image processing, keypoint detection, keypoint tracking,
// object tracking, and object detection/relocalization.
class ObjectTracker {
 public:
  ObjectTracker(const TrackerConfig* const config,
                ObjectDetectorBase* const detector);
  virtual ~ObjectTracker();

  virtual void NextFrame(const uint8_t* const new_frame,
                         const int64_t timestamp,
                         const float* const alignment_matrix_2x3) {
    NextFrame(new_frame, NULL, timestamp, alignment_matrix_2x3);
  }

  // Called upon the arrival of a new frame of raw data.
  // Does all image processing, keypoint detection, and object
  // tracking/detection for registered objects.
  // Argument alignment_matrix_2x3 is a 2x3 matrix (stored row-wise) that
  // represents the main transformation that has happened between the last
  // and the current frame.
  // Argument align_level is the pyramid level (where 0 == finest) that
  // the matrix is valid for.
  virtual void NextFrame(const uint8_t* const new_frame,
                         const uint8_t* const uv_frame, const int64_t timestamp,
                         const float* const alignment_matrix_2x3);

  virtual void RegisterNewObjectWithAppearance(const std::string& id,
                                               const uint8_t* const new_frame,
                                               const BoundingBox& bounding_box);

  // Updates the position of a tracked object, given that it was known to be at
  // a certain position at some point in the past.
  virtual void SetPreviousPositionOfObject(const std::string& id,
                                           const BoundingBox& bounding_box,
                                           const int64_t timestamp);

  // Sets the current position of the object in the most recent frame provided.
  virtual void SetCurrentPositionOfObject(const std::string& id,
                                          const BoundingBox& bounding_box);

  // Tells the ObjectTracker to stop tracking a target.
  void ForgetTarget(const std::string& id);

  // Fills the given out_data buffer with the latest detected keypoint
  // correspondences, first scaled by scale_factor (to adjust for downsampling
  // that may have occurred elsewhere), then packed in a fixed-point format.
  int GetKeypointsPacked(uint16_t* const out_data,
                         const float scale_factor) const;

  // Copy the keypoint arrays after computeFlow is called.
  // out_data should be at least kMaxKeypoints * kKeypointStep long.
  // Currently, its format is [x1 y1 found x2 y2 score] repeated N times,
  // where N is the number of keypoints tracked.  N is returned as the result.
  int GetKeypoints(const bool only_found, float* const out_data) const;

  // Returns the current position of a box, given that it was at a certain
  // position at the given time.
  BoundingBox TrackBox(const BoundingBox& region,
                       const int64_t timestamp) const;

  // Returns the number of frames that have been passed to NextFrame().
  inline int GetNumFrames() const {
    return num_frames_;
  }

  inline bool HaveObject(const std::string& id) const {
    return objects_.find(id) != objects_.end();
  }

  // Returns the TrackedObject associated with the given id.
  inline const TrackedObject* GetObject(const std::string& id) const {
    TrackedObjectMap::const_iterator iter = objects_.find(id);
    CHECK_ALWAYS(iter != objects_.end(),
                 "Unknown object key! \"%s\"", id.c_str());
    TrackedObject* const object = iter->second;
    return object;
  }

  // Returns the TrackedObject associated with the given id.
  inline TrackedObject* GetObject(const std::string& id) {
    TrackedObjectMap::iterator iter = objects_.find(id);
    CHECK_ALWAYS(iter != objects_.end(),
                 "Unknown object key! \"%s\"", id.c_str());
    TrackedObject* const object = iter->second;
    return object;
  }

  bool IsObjectVisible(const std::string& id) const {
    SCHECK(HaveObject(id), "Don't have this object.");

    const TrackedObject* object = GetObject(id);
    return object->IsVisible();
  }

  virtual void Draw(const int canvas_width, const int canvas_height,
                    const float* const frame_to_canvas) const;

 protected:
  // Creates a new tracked object at the given position.
  // If an object model is provided, then that model will be associated with the
  // object. If not, a new model may be created from the appearance at the
  // initial position and registered with the object detector.
  virtual TrackedObject* MaybeAddObject(const std::string& id,
                                        const Image<uint8_t>& image,
                                        const BoundingBox& bounding_box,
                                        const ObjectModelBase* object_model);

  // Find the keypoints in the frame before the current frame.
  // If only one frame exists, keypoints will be found in that frame.
  void ComputeKeypoints(const bool cached_ok = false);

  // Finds the correspondences for all the points in the current pair of frames.
  // Stores the results in the given FramePair.
  void FindCorrespondences(FramePair* const curr_change) const;

  inline int GetNthIndexFromEnd(const int offset) const {
    return GetNthIndexFromStart(curr_num_frame_pairs_ - 1 - offset);
  }

  BoundingBox TrackBox(const BoundingBox& region,
                       const FramePair& frame_pair) const;

  inline void IncrementFrameIndex() {
    // Move the current framechange index up.
    ++num_frames_;
    ++curr_num_frame_pairs_;

    // If we've got too many, push up the start of the queue.
    if (curr_num_frame_pairs_ > kNumFrames) {
      first_frame_index_ = GetNthIndexFromStart(1);
      --curr_num_frame_pairs_;
    }
  }

  inline int GetNthIndexFromStart(const int offset) const {
    SCHECK(offset >= 0 && offset < curr_num_frame_pairs_,
          "Offset out of range!  %d out of %d.", offset, curr_num_frame_pairs_);
    return (first_frame_index_ + offset) % kNumFrames;
  }

  void TrackObjects();

  const std::unique_ptr<const TrackerConfig> config_;

  const int frame_width_;
  const int frame_height_;

  int64_t curr_time_;

  int num_frames_;

  TrackedObjectMap objects_;

  FlowCache flow_cache_;

  KeypointDetector keypoint_detector_;

  int curr_num_frame_pairs_;
  int first_frame_index_;

  std::unique_ptr<ImageData> frame1_;
  std::unique_ptr<ImageData> frame2_;

  FramePair frame_pairs_[kNumFrames];

  std::unique_ptr<ObjectDetectorBase> detector_;

  int num_detected_;

 private:
  void TrackTarget(TrackedObject* const object);

  bool GetBestObjectForDetection(
      const Detection& detection, TrackedObject** match) const;

  void ProcessDetections(std::vector<Detection>* const detections);

  void DetectTargets();

  // Temp object used in ObjectTracker::CreateNewExample.
  mutable std::vector<BoundingSquare> squares;

  friend std::ostream& operator<<(std::ostream& stream,
                                  const ObjectTracker& tracker);

  TF_DISALLOW_COPY_AND_ASSIGN(ObjectTracker);
};

inline std::ostream& operator<<(std::ostream& stream,
                                const ObjectTracker& tracker) {
  stream << "Frame size: " << tracker.frame_width_ << "x"
         << tracker.frame_height_ << std::endl;

  stream << "Num frames: " << tracker.num_frames_ << std::endl;

  stream << "Curr time: " << tracker.curr_time_ << std::endl;

  const int first_frame_index = tracker.GetNthIndexFromStart(0);
  const FramePair& first_frame_pair = tracker.frame_pairs_[first_frame_index];

  const int last_frame_index = tracker.GetNthIndexFromEnd(0);
  const FramePair& last_frame_pair = tracker.frame_pairs_[last_frame_index];

  stream << "first frame: " << first_frame_index << ","
         << first_frame_pair.end_time_ << "    "
         << "last frame: " << last_frame_index << ","
         << last_frame_pair.end_time_ << "   diff: "
         << last_frame_pair.end_time_ - first_frame_pair.end_time_ << "ms"
         << std::endl;

  stream << "Tracked targets:";
  stream << tracker.objects_;

  return stream;
}

}  // namespace tf_tracking

#endif  // TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_OBJECT_TRACKER_H_
