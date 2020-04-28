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

#include <float.h>

#include "tensorflow/examples/android/jni/object_tracking/config.h"
#include "tensorflow/examples/android/jni/object_tracking/frame_pair.h"

namespace tf_tracking {

void FramePair::Init(const int64_t start_time, const int64_t end_time) {
  start_time_ = start_time;
  end_time_ = end_time;
  memset(optical_flow_found_keypoint_, false,
         sizeof(*optical_flow_found_keypoint_) * kMaxKeypoints);
  number_of_keypoints_ = 0;
}

void FramePair::AdjustBox(const BoundingBox box,
                          float* const translation_x,
                          float* const translation_y,
                          float* const scale_x,
                          float* const scale_y) const {
  static float weights[kMaxKeypoints];
  static Point2f deltas[kMaxKeypoints];
  memset(weights, 0.0f, sizeof(*weights) * kMaxKeypoints);

  BoundingBox resized_box(box);
  resized_box.Scale(0.4f, 0.4f);
  FillWeights(resized_box, weights);
  FillTranslations(deltas);

  const Point2f translation = GetWeightedMedian(weights, deltas);

  *translation_x = translation.x;
  *translation_y = translation.y;

  const Point2f old_center = box.GetCenter();
  const int good_scale_points =
      FillScales(old_center, translation, weights, deltas);

  // Default scale factor is 1 for x and y.
  *scale_x = 1.0f;
  *scale_y = 1.0f;

  // The assumption is that all deltas that make it to this stage with a
  // corresponding optical_flow_found_keypoint_[i] == true are not in
  // themselves degenerate.
  //
  // The degeneracy with scale arose because if the points are too close to the
  // center of the objects, the scale ratio determination might be incalculable.
  //
  // The check for kMinNumInRange is not a degeneracy check, but merely an
  // attempt to ensure some sort of stability. The actual degeneracy check is in
  // the comparison to EPSILON in FillScales (which I've updated to return the
  // number good remaining as well).
  static const int kMinNumInRange = 5;
  if (good_scale_points >= kMinNumInRange) {
    const float scale_factor = GetWeightedMedianScale(weights, deltas);

    if (scale_factor > 0.0f) {
      *scale_x = scale_factor;
      *scale_y = scale_factor;
    }
  }
}

int FramePair::FillWeights(const BoundingBox& box,
                           float* const weights) const {
  // Compute the max score.
  float max_score = -FLT_MAX;
  float min_score = FLT_MAX;
  for (int i = 0; i < kMaxKeypoints; ++i) {
    if (optical_flow_found_keypoint_[i]) {
      max_score = MAX(max_score, frame1_keypoints_[i].score_);
      min_score = MIN(min_score, frame1_keypoints_[i].score_);
    }
  }

  int num_in_range = 0;
  for (int i = 0; i < kMaxKeypoints; ++i) {
    if (!optical_flow_found_keypoint_[i]) {
      weights[i] = 0.0f;
      continue;
    }

    const bool in_box = box.Contains(frame1_keypoints_[i].pos_);
    if (in_box) {
      ++num_in_range;
    }

    // The weighting based off distance.  Anything within the bounding box
    // has a weight of 1, and everything outside of that is within the range
    // [0, kOutOfBoxMultiplier), falling off with the squared distance ratio.
    float distance_score = 1.0f;
    if (!in_box) {
      const Point2f initial = box.GetCenter();
      const float sq_x_dist =
          Square(initial.x - frame1_keypoints_[i].pos_.x);
      const float sq_y_dist =
          Square(initial.y - frame1_keypoints_[i].pos_.y);
      const float squared_half_width = Square(box.GetWidth() / 2.0f);
      const float squared_half_height = Square(box.GetHeight() / 2.0f);

      static const float kOutOfBoxMultiplier = 0.5f;
      distance_score = kOutOfBoxMultiplier *
          MIN(squared_half_height / sq_y_dist, squared_half_width / sq_x_dist);
    }

    // The weighting based on relative score strength. kBaseScore - 1.0f.
    float intrinsic_score =  1.0f;
    if (max_score > min_score) {
      static const float kBaseScore = 0.5f;
      intrinsic_score = ((frame1_keypoints_[i].score_ - min_score) /
         (max_score - min_score)) * (1.0f - kBaseScore) + kBaseScore;
    }

    // The final score will be in the range [0, 1].
    weights[i] = distance_score * intrinsic_score;
  }

  return num_in_range;
}

void FramePair::FillTranslations(Point2f* const translations) const {
  for (int i = 0; i < kMaxKeypoints; ++i) {
    if (!optical_flow_found_keypoint_[i]) {
      continue;
    }
    translations[i].x =
        frame2_keypoints_[i].pos_.x - frame1_keypoints_[i].pos_.x;
    translations[i].y =
        frame2_keypoints_[i].pos_.y - frame1_keypoints_[i].pos_.y;
  }
}

int FramePair::FillScales(const Point2f& old_center,
                          const Point2f& translation,
                          float* const weights,
                          Point2f* const scales) const {
  int num_good = 0;
  for (int i = 0; i < kMaxKeypoints; ++i) {
    if (!optical_flow_found_keypoint_[i]) {
      continue;
    }

    const Keypoint keypoint1 = frame1_keypoints_[i];
    const Keypoint keypoint2 = frame2_keypoints_[i];

    const float dist1_x = keypoint1.pos_.x - old_center.x;
    const float dist1_y = keypoint1.pos_.y - old_center.y;

    const float dist2_x = (keypoint2.pos_.x - translation.x) - old_center.x;
    const float dist2_y = (keypoint2.pos_.y - translation.y) - old_center.y;

    // Make sure that the scale makes sense; points too close to the center
    // will result in either NaNs or infinite results for scale due to
    // limited tracking and floating point resolution.
    // Also check that the parity of the points is the same with respect to
    // x and y, as we can't really make sense of data that has flipped.
    if (((dist2_x > EPSILON && dist1_x > EPSILON) ||
         (dist2_x < -EPSILON && dist1_x < -EPSILON)) &&
         ((dist2_y > EPSILON && dist1_y > EPSILON) ||
          (dist2_y < -EPSILON && dist1_y < -EPSILON))) {
      scales[i].x = dist2_x / dist1_x;
      scales[i].y = dist2_y / dist1_y;
      ++num_good;
    } else {
      weights[i] = 0.0f;
      scales[i].x = 1.0f;
      scales[i].y = 1.0f;
    }
  }
  return num_good;
}

struct WeightedDelta {
  float weight;
  float delta;
};

// Sort by delta, not by weight.
inline int WeightedDeltaCompare(const void* const a, const void* const b) {
  return (reinterpret_cast<const WeightedDelta*>(a)->delta -
          reinterpret_cast<const WeightedDelta*>(b)->delta) <= 0 ? 1 : -1;
}

// Returns the median delta from a sorted set of weighted deltas.
static float GetMedian(const int num_items,
                       const WeightedDelta* const weighted_deltas,
                       const float sum) {
  if (num_items == 0 || sum < EPSILON) {
    return 0.0f;
  }

  float current_weight = 0.0f;
  const float target_weight = sum / 2.0f;
  for (int i = 0; i < num_items; ++i) {
    if (weighted_deltas[i].weight > 0.0f) {
      current_weight += weighted_deltas[i].weight;
      if (current_weight >= target_weight) {
        return weighted_deltas[i].delta;
      }
    }
  }
  LOGW("Median not found! %d points, sum of %.2f", num_items, sum);
  return 0.0f;
}

Point2f FramePair::GetWeightedMedian(
    const float* const weights, const Point2f* const deltas) const {
  Point2f median_delta;

  // TODO(andrewharp): only sort deltas that could possibly have an effect.
  static WeightedDelta weighted_deltas[kMaxKeypoints];

  // Compute median X value.
  {
    float total_weight = 0.0f;

    // Compute weighted mean and deltas.
    for (int i = 0; i < kMaxKeypoints; ++i) {
      weighted_deltas[i].delta = deltas[i].x;
      const float weight = weights[i];
      weighted_deltas[i].weight = weight;
      if (weight > 0.0f) {
        total_weight += weight;
      }
    }
    qsort(weighted_deltas, kMaxKeypoints, sizeof(WeightedDelta),
          WeightedDeltaCompare);
    median_delta.x = GetMedian(kMaxKeypoints, weighted_deltas, total_weight);
  }

  // Compute median Y value.
  {
    float total_weight = 0.0f;

    // Compute weighted mean and deltas.
    for (int i = 0; i < kMaxKeypoints; ++i) {
      const float weight = weights[i];
      weighted_deltas[i].weight = weight;
      weighted_deltas[i].delta = deltas[i].y;
      if (weight > 0.0f) {
        total_weight += weight;
      }
    }
    qsort(weighted_deltas, kMaxKeypoints, sizeof(WeightedDelta),
          WeightedDeltaCompare);
    median_delta.y = GetMedian(kMaxKeypoints, weighted_deltas, total_weight);
  }

  return median_delta;
}

float FramePair::GetWeightedMedianScale(
    const float* const weights, const Point2f* const deltas) const {
  float median_delta;

  // TODO(andrewharp): only sort deltas that could possibly have an effect.
  static WeightedDelta weighted_deltas[kMaxKeypoints * 2];

  // Compute median scale value across x and y.
  {
    float total_weight = 0.0f;

    // Add X values.
    for (int i = 0; i < kMaxKeypoints; ++i) {
      weighted_deltas[i].delta = deltas[i].x;
      const float weight = weights[i];
      weighted_deltas[i].weight = weight;
      if (weight > 0.0f) {
        total_weight += weight;
      }
    }

    // Add Y values.
    for (int i = 0; i < kMaxKeypoints; ++i) {
      weighted_deltas[i + kMaxKeypoints].delta = deltas[i].y;
      const float weight = weights[i];
      weighted_deltas[i + kMaxKeypoints].weight = weight;
      if (weight > 0.0f) {
        total_weight += weight;
      }
    }

    qsort(weighted_deltas, kMaxKeypoints * 2, sizeof(WeightedDelta),
          WeightedDeltaCompare);

    median_delta = GetMedian(kMaxKeypoints * 2, weighted_deltas, total_weight);
  }

  return median_delta;
}

}  // namespace tf_tracking
