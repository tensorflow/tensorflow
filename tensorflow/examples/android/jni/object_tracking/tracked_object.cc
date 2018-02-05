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

#include "tensorflow/examples/android/jni/object_tracking/tracked_object.h"

namespace tf_tracking {

static const float kInitialDistance = 20.0f;

static void InitNormalized(const Image<uint8_t>& src_image,
                           const BoundingBox& position,
                           Image<float>* const dst_image) {
  BoundingBox scaled_box(position);
  CopyArea(src_image, scaled_box, dst_image);
  NormalizeImage(dst_image);
}

TrackedObject::TrackedObject(const std::string& id, const Image<uint8_t>& image,
                             const BoundingBox& bounding_box,
                             ObjectModelBase* const model)
    : id_(id),
      last_known_position_(bounding_box),
      last_detection_position_(bounding_box),
      position_last_computed_time_(-1),
      object_model_(model),
      last_detection_thumbnail_(kNormalizedThumbnailSize,
                                kNormalizedThumbnailSize),
      last_frame_thumbnail_(kNormalizedThumbnailSize, kNormalizedThumbnailSize),
      tracked_correlation_(0.0f),
      tracked_match_score_(0.0),
      num_consecutive_frames_below_threshold_(0),
      allowable_detection_distance_(Square(kInitialDistance)) {
  InitNormalized(image, bounding_box, &last_detection_thumbnail_);
}

TrackedObject::~TrackedObject() {}

void TrackedObject::UpdatePosition(const BoundingBox& new_position,
                                   const int64_t timestamp,
                                   const ImageData& image_data,
                                   const bool authoritative) {
  last_known_position_ = new_position;
  position_last_computed_time_ = timestamp;

  InitNormalized(*image_data.GetImage(), new_position, &last_frame_thumbnail_);

  const float last_localization_correlation = ComputeCrossCorrelation(
      last_detection_thumbnail_.data(),
      last_frame_thumbnail_.data(),
      last_frame_thumbnail_.data_size_);
  LOGV("Tracked correlation to last localization:   %.6f",
       last_localization_correlation);

  // Correlation to object model, if it exists.
  if (object_model_ != NULL) {
    tracked_correlation_ =
        object_model_->GetMaxCorrelation(last_frame_thumbnail_);
    LOGV("Tracked correlation to model:               %.6f",
         tracked_correlation_);

    tracked_match_score_ =
        object_model_->GetMatchScore(new_position, image_data);
    LOGV("Tracked match score with model:             %.6f",
         tracked_match_score_.value);
  } else {
    // If there's no model to check against, set the tracked correlation to
    // simply be the correlation to the last set position.
    tracked_correlation_ = last_localization_correlation;
    tracked_match_score_ = MatchScore(0.0f);
  }

  // Determine if it's still being tracked.
  if (tracked_correlation_ >= kMinimumCorrelationForTracking &&
      tracked_match_score_ >= kMinimumMatchScore) {
    num_consecutive_frames_below_threshold_ = 0;

    if (object_model_ != NULL) {
      object_model_->TrackStep(last_known_position_, *image_data.GetImage(),
                               *image_data.GetIntegralImage(), authoritative);
    }
  } else if (tracked_match_score_ < kMatchScoreForImmediateTermination) {
    if (num_consecutive_frames_below_threshold_ < 1000) {
      LOGD("Tracked match score is way too low (%.6f), aborting track.",
           tracked_match_score_.value);
    }

    // Add an absurd amount of missed frames so that all heuristics will
    // consider it a lost track.
    num_consecutive_frames_below_threshold_ += 1000;

    if (object_model_ != NULL) {
      object_model_->TrackLost();
    }
  } else {
    ++num_consecutive_frames_below_threshold_;
    allowable_detection_distance_ *= 1.1f;
  }
}

void TrackedObject::OnDetection(ObjectModelBase* const model,
                                const BoundingBox& detection_position,
                                const MatchScore match_score,
                                const int64_t timestamp,
                                const ImageData& image_data) {
  const float overlap = detection_position.PascalScore(last_known_position_);
  if (overlap > kPositionOverlapThreshold) {
    // If the position agreement with the current tracked position is good
    // enough, lock all the current unlocked examples.
    object_model_->TrackConfirmed();
    num_consecutive_frames_below_threshold_ = 0;
  }

  // Before relocalizing, make sure the new proposed position is better than
  // the existing position by a small amount to prevent thrashing.
  if (match_score <= tracked_match_score_ + kMatchScoreBuffer) {
    LOGI("Not relocalizing since new match is worse: %.6f < %.6f + %.6f",
         match_score.value, tracked_match_score_.value,
         kMatchScoreBuffer.value);
    return;
  }

  LOGI("Relocalizing! From (%.1f, %.1f)[%.1fx%.1f] to "
       "(%.1f, %.1f)[%.1fx%.1f]:   %.6f > %.6f",
       last_known_position_.left_, last_known_position_.top_,
       last_known_position_.GetWidth(), last_known_position_.GetHeight(),
       detection_position.left_, detection_position.top_,
       detection_position.GetWidth(), detection_position.GetHeight(),
       match_score.value, tracked_match_score_.value);

  if (overlap < kPositionOverlapThreshold) {
    // The path might be good, it might be bad, but it's no longer a path
    // since we're moving the box to a new position, so just nuke it from
    // orbit to be safe.
    object_model_->TrackLost();
  }

  object_model_ = model;

  // Reset the last detected appearance.
  InitNormalized(
      *image_data.GetImage(), detection_position, &last_detection_thumbnail_);

  num_consecutive_frames_below_threshold_ = 0;
  last_detection_position_ = detection_position;

  UpdatePosition(detection_position, timestamp, image_data, false);
  allowable_detection_distance_ = Square(kInitialDistance);
}

}  // namespace tf_tracking
