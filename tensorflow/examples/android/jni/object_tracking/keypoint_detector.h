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

#ifndef THIRD_PARTY_TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_KEYPOINT_DETECTOR_H_
#define THIRD_PARTY_TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_KEYPOINT_DETECTOR_H_

#include <vector>

#include "tensorflow/core/platform/types.h"

#include "tensorflow/examples/android/jni/object_tracking/image-inl.h"
#include "tensorflow/examples/android/jni/object_tracking/image.h"
#include "tensorflow/examples/android/jni/object_tracking/image_data.h"
#include "tensorflow/examples/android/jni/object_tracking/optical_flow.h"

using namespace tensorflow;

namespace tf_tracking {

struct Keypoint;

class KeypointDetector {
 public:
  explicit KeypointDetector(const KeypointDetectorConfig* const config)
      : config_(config),
        keypoint_scratch_(new Image<uint8>(config_->image_size)),
        interest_map_(new Image<bool>(config_->image_size)),
        fast_quadrant_(0) {
    interest_map_->Clear(false);
  }

  ~KeypointDetector() {}

  // Finds a new set of keypoints for the current frame, picked from the current
  // set of keypoints and also from a set discovered via a keypoint detector.
  // Special attention is applied to make sure that keypoints are distributed
  // within the supplied ROIs.
  void FindKeypoints(const ImageData& image_data,
                     const std::vector<BoundingBox>& rois,
                     const FramePair& prev_change,
                     FramePair* const curr_change);

 private:
  // Compute the corneriness of a point in the image.
  float HarrisFilter(const Image<int32>& I_x, const Image<int32>& I_y,
                     const float x, const float y) const;

  // Adds a grid of candidate keypoints to the given box, up to
  // max_num_keypoints or kNumToAddAsCandidates^2, whichever is lower.
  int AddExtraCandidatesForBoxes(
      const std::vector<BoundingBox>& boxes,
      const int max_num_keypoints,
      Keypoint* const keypoints) const;

  // Scan the frame for potential keypoints using the FAST keypoint detector.
  // Quadrant is an argument 0-3 which refers to the quadrant of the image in
  // which to detect keypoints.
  int FindFastKeypoints(const Image<uint8>& frame,
                        const int quadrant,
                        const int downsample_factor,
                        const int max_num_keypoints,
                        Keypoint* const keypoints);

  int FindFastKeypoints(const ImageData& image_data,
                        const int max_num_keypoints,
                        Keypoint* const keypoints);

  // Score a bunch of candidate keypoints.  Assigns the scores to the input
  // candidate_keypoints array entries.
  void ScoreKeypoints(const ImageData& image_data,
                      const int num_candidates,
                      Keypoint* const candidate_keypoints);

  void SortKeypoints(const int num_candidates,
                    Keypoint* const candidate_keypoints) const;

  // Selects a set of keypoints falling within the supplied box such that the
  // most highly rated keypoints are picked first, and so that none of them are
  // too close together.
  int SelectKeypointsInBox(
      const BoundingBox& box,
      const Keypoint* const candidate_keypoints,
      const int num_candidates,
      const int max_keypoints,
      const int num_existing_keypoints,
      const Keypoint* const existing_keypoints,
      Keypoint* const final_keypoints) const;

  // Selects from the supplied sorted keypoint pool a set of keypoints that will
  // best cover the given set of boxes, such that each box is covered at a
  // resolution proportional to its size.
  void SelectKeypoints(
      const std::vector<BoundingBox>& boxes,
      const Keypoint* const candidate_keypoints,
      const int num_candidates,
      FramePair* const frame_change) const;

  // Copies and compacts the found keypoints in the second frame of prev_change
  // into the array at new_keypoints.
  static int CopyKeypoints(const FramePair& prev_change,
                          Keypoint* const new_keypoints);

  const KeypointDetectorConfig* const config_;

  // Scratch memory for keypoint candidacy detection and non-max suppression.
  std::unique_ptr<Image<uint8> > keypoint_scratch_;

  // Regions of the image to pay special attention to.
  std::unique_ptr<Image<bool> > interest_map_;

  // The current quadrant of the image to detect FAST keypoints in.
  // Keypoint detection is staggered for performance reasons. Every four frames
  // a full scan of the frame will have been performed.
  int fast_quadrant_;

  Keypoint tmp_keypoints_[kMaxTempKeypoints];
};

}  // namespace tf_tracking

#endif  // THIRD_PARTY_TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_KEYPOINT_DETECTOR_H_
