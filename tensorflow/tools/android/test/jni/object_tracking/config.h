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

#ifndef TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_CONFIG_H_
#define TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_CONFIG_H_

#include <math.h>

#include "tensorflow/tools/android/test/jni/object_tracking/geom.h"

namespace tf_tracking {

// Arbitrary keypoint type ids for labeling the origin of tracked keypoints.
enum KeypointType {
  KEYPOINT_TYPE_DEFAULT = 0,
  KEYPOINT_TYPE_FAST = 1,
  KEYPOINT_TYPE_INTEREST = 2
};

// Struct that can be used to more richly store the results of a detection
// than a single number, while still maintaining comparability.
struct MatchScore {
  explicit MatchScore(double val) : value(val) {}
  MatchScore() { value = 0.0; }

  double value;

  MatchScore& operator+(const MatchScore& rhs) {
    value += rhs.value;
    return *this;
  }

  friend std::ostream& operator<<(std::ostream& stream,
                                  const MatchScore& detection) {
    stream << detection.value;
    return stream;
  }
};
inline bool operator< (const MatchScore& cC1, const MatchScore& cC2) {
    return cC1.value < cC2.value;
}
inline bool operator> (const MatchScore& cC1, const MatchScore& cC2) {
    return cC1.value > cC2.value;
}
inline bool operator>= (const MatchScore& cC1, const MatchScore& cC2) {
    return cC1.value >= cC2.value;
}
inline bool operator<= (const MatchScore& cC1, const MatchScore& cC2) {
    return cC1.value <= cC2.value;
}

// Fixed seed used for all random number generators.
static const int kRandomNumberSeed = 11111;

// TODO(andrewharp): Move as many of these settings as possible into a settings
// object which can be passed in from Java at runtime.

// Whether or not to use ESM instead of LK flow.
static const bool kUseEsm = false;

// This constant gets added to the diagonal of the Hessian
// before solving for translation in 2dof ESM.
// It ensures better behavior especially in the absence of
// strong texture.
static const int kEsmRegularizer = 20;

// Do we want to brightness-normalize each keypoint patch when we compute
// its flow using ESM?
static const bool kDoBrightnessNormalize = true;

// Whether or not to use fixed-point interpolated pixel lookups in optical flow.
#define USE_FIXED_POINT_FLOW 1

// Whether to normalize keypoint windows for intensity in LK optical flow.
// This is a define for now because it helps keep the code streamlined.
#define NORMALIZE 1

// Number of keypoints to store per frame.
static const int kMaxKeypoints = 76;

// Keypoint detection.
static const int kMaxTempKeypoints = 1024;

// Number of floats each keypoint takes up when exporting to an array.
static const int kKeypointStep = 7;

// Number of frame deltas to keep around in the circular queue.
static const int kNumFrames = 512;

// Number of iterations to do tracking on each keypoint at each pyramid level.
static const int kNumIterations = 3;

// The number of bins (on a side) to divide each bin from the previous
// cache level into.  Higher numbers will decrease performance by increasing
// cache misses, but mean that cache hits are more locally relevant.
static const int kCacheBranchFactor = 2;

// Number of levels to put in the cache.
// Each level of the cache is a square grid of bins, length:
// branch_factor^(level - 1) on each side.
//
// This may be greater than kNumPyramidLevels. Setting it to 0 means no
// caching is enabled.
static const int kNumCacheLevels = 3;

// The level at which the cache pyramid gets cut off and replaced by a matrix
// transform if such a matrix has been provided to the cache.
static const int kCacheCutoff = 1;

static const int kNumPyramidLevels = 4;

// The minimum number of keypoints needed in an object's area.
static const int kMaxKeypointsForObject = 16;

// Minimum number of pyramid levels to use after getting cached value.
// This allows fine-scale adjustment from the cached value, which is taken
// from the center of the corresponding top cache level box.
// Can be [0, kNumPyramidLevels).
static const int kMinNumPyramidLevelsToUseForAdjustment = 1;

// Window size to integrate over to find local image derivative.
static const int kFlowIntegrationWindowSize = 3;

// Total area of integration windows.
static const int kFlowArraySize =
    (2 * kFlowIntegrationWindowSize + 1) * (2 * kFlowIntegrationWindowSize + 1);

// Error that's considered good enough to early abort tracking.
static const float kTrackingAbortThreshold = 0.03f;

// Maximum number of deviations a keypoint-correspondence delta can be from the
// weighted average before being thrown out for region-based queries.
static const float kNumDeviations = 2.0f;

// The length of the allowed delta between the forward and the backward
// flow deltas in terms of the length of the forward flow vector.
static const float kMaxForwardBackwardErrorAllowed = 0.5f;

// Threshold for pixels to be considered different.
static const int kFastDiffAmount = 10;

// How far from edge of frame to stop looking for FAST keypoints.
static const int kFastBorderBuffer = 10;

// Determines if non-detected arbitrary keypoints should be added to regions.
// This will help if no keypoints have been detected in the region yet.
static const bool kAddArbitraryKeypoints = true;

// How many arbitrary keypoints to add along each axis as candidates for each
// region?
static const int kNumToAddAsCandidates = 1;

// In terms of region dimensions, how closely can we place keypoints
// next to each other?
static const float kClosestPercent = 0.6f;

// How many FAST qualifying pixels must be connected to a pixel for it to be
// considered a candidate keypoint for Harris filtering.
static const int kMinNumConnectedForFastKeypoint = 8;

// Size of the window to integrate over for Harris filtering.
// Compare to kFlowIntegrationWindowSize.
static const int kHarrisWindowSize = 2;


// DETECTOR PARAMETERS

// Before relocalizing, make sure the new proposed position is better than
// the existing position by a small amount to prevent thrashing.
static const MatchScore kMatchScoreBuffer(0.01f);

// Minimum score a tracked object can have and still be considered a match.
// TODO(andrewharp): Make this a per detector thing.
static const MatchScore kMinimumMatchScore(0.5f);

static const float kMinimumCorrelationForTracking = 0.4f;

static const MatchScore kMatchScoreForImmediateTermination(0.0f);

// Run the detector every N frames.
static const int kDetectEveryNFrames = 4;

// How many features does each feature_set contain?
static const int kFeaturesPerFeatureSet = 10;

// The number of FeatureSets managed by the object detector.
// More FeatureSets can increase recall at the cost of performance.
static const int kNumFeatureSets = 7;

// How many FeatureSets must respond affirmatively for a candidate descriptor
// and position to be given more thorough attention?
static const int kNumFeatureSetsForCandidate = 2;

// How large the thumbnails used for correlation validation are.  Used for both
// width and height.
static const int kNormalizedThumbnailSize = 11;

// The area of intersection divided by union for the bounding boxes that tells
// if this tracking has slipped enough to invalidate all unlocked examples.
static const float kPositionOverlapThreshold = 0.6f;

// The number of detection failures allowed before an object goes invisible.
// Tracking will still occur, so if it is actually still being tracked and
// comes back into a detectable position, it's likely to be found.
static const int kMaxNumDetectionFailures = 4;


// Minimum square size to scan with sliding window.
static const float kScanMinSquareSize = 16.0f;

// Minimum square size to scan with sliding window.
static const float kScanMaxSquareSize = 64.0f;

// Scale difference for consecutive scans of the sliding window.
static const float kScanScaleFactor = sqrtf(2.0f);

// Step size for sliding window.
static const int kScanStepSize = 10;


// How tightly to pack the descriptor boxes for confirmed exemplars.
static const float kLockedScaleFactor = 1 / sqrtf(2.0f);

// How tightly to pack the descriptor boxes for unconfirmed exemplars.
static const float kUnlockedScaleFactor = 1 / 2.0f;

// How tightly the boxes to scan centered at the last known position will be
// packed.
static const float kLastKnownPositionScaleFactor = 1.0f / sqrtf(2.0f);

// The bounds on how close a new object example must be to existing object
// examples for detection to be valid.
static const float kMinCorrelationForNewExample = 0.75f;
static const float kMaxCorrelationForNewExample = 0.99f;


// The number of safe tries an exemplar has after being created before
// missed detections count against it.
static const int kFreeTries = 5;

// A false positive is worth this many missed detections.
static const int kFalsePositivePenalty = 5;

struct ObjectDetectorConfig {
  const Size image_size;

  explicit ObjectDetectorConfig(const Size& image_size)
      : image_size(image_size) {}
  virtual ~ObjectDetectorConfig() = default;
};

struct KeypointDetectorConfig {
  const Size image_size;

  bool detect_skin;

  explicit KeypointDetectorConfig(const Size& image_size)
      : image_size(image_size),
        detect_skin(false) {}
};


struct OpticalFlowConfig {
  const Size image_size;

  explicit OpticalFlowConfig(const Size& image_size)
      : image_size(image_size) {}
};

struct TrackerConfig {
  const Size image_size;
  KeypointDetectorConfig keypoint_detector_config;
  OpticalFlowConfig flow_config;
  bool always_track;

  float object_box_scale_factor_for_features;

  explicit TrackerConfig(const Size& image_size)
      : image_size(image_size),
        keypoint_detector_config(image_size),
        flow_config(image_size),
        always_track(false),
        object_box_scale_factor_for_features(1.0f) {}
};

}  // namespace tf_tracking

#endif  // TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_CONFIG_H_
