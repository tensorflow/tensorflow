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

#ifndef TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_FLOW_CACHE_H_
#define TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_FLOW_CACHE_H_

#include "tensorflow/examples/android/jni/object_tracking/geom.h"
#include "tensorflow/examples/android/jni/object_tracking/utils.h"

#include "tensorflow/examples/android/jni/object_tracking/config.h"
#include "tensorflow/examples/android/jni/object_tracking/optical_flow.h"

namespace tf_tracking {

// Class that helps OpticalFlow to speed up flow computation
// by caching coarse-grained flow.
class FlowCache {
 public:
  explicit FlowCache(const OpticalFlowConfig* const config)
      : config_(config),
        image_size_(config->image_size),
        optical_flow_(config),
        fullframe_matrix_(NULL) {
    for (int i = 0; i < kNumCacheLevels; ++i) {
      const int curr_dims = BlockDimForCacheLevel(i);
      has_cache_[i] = new Image<bool>(curr_dims, curr_dims);
      displacements_[i] = new Image<Point2f>(curr_dims, curr_dims);
    }
  }

  ~FlowCache() {
    for (int i = 0; i < kNumCacheLevels; ++i) {
      SAFE_DELETE(has_cache_[i]);
      SAFE_DELETE(displacements_[i]);
    }
    delete[](fullframe_matrix_);
    fullframe_matrix_ = NULL;
  }

  void NextFrame(ImageData* const new_frame,
                 const float* const align_matrix23) {
    ClearCache();
    SetFullframeAlignmentMatrix(align_matrix23);
    optical_flow_.NextFrame(new_frame);
  }

  void ClearCache() {
    for (int i = 0; i < kNumCacheLevels; ++i) {
      has_cache_[i]->Clear(false);
    }
    delete[](fullframe_matrix_);
    fullframe_matrix_ = NULL;
  }

  // Finds the flow at a point, using the cache for performance.
  bool FindFlowAtPoint(const float u_x, const float u_y,
                       float* const flow_x, float* const flow_y) const {
    // Get the best guess from the cache.
    const Point2f guess_from_cache = LookupGuess(u_x, u_y);

    *flow_x = guess_from_cache.x;
    *flow_y = guess_from_cache.y;

    // Now refine the guess using the image pyramid.
    for (int pyramid_level = kMinNumPyramidLevelsToUseForAdjustment - 1;
        pyramid_level >= 0; --pyramid_level) {
      if (!optical_flow_.FindFlowAtPointSingleLevel(
          pyramid_level, u_x, u_y, false, flow_x, flow_y)) {
        return false;
      }
    }

    return true;
  }

  // Determines the displacement of a point, and uses that to calculate a new
  // position.
  // Returns true iff the displacement determination worked and the new position
  // is in the image.
  bool FindNewPositionOfPoint(const float u_x, const float u_y,
                              float* final_x, float* final_y) const {
    float flow_x;
    float flow_y;
    if (!FindFlowAtPoint(u_x, u_y, &flow_x, &flow_y)) {
      return false;
    }

    // Add in the displacement to get the final position.
    *final_x = u_x + flow_x;
    *final_y = u_y + flow_y;

    // Assign the best guess, if we're still in the image.
    if (InRange(*final_x, 0.0f, static_cast<float>(image_size_.width) - 1) &&
        InRange(*final_y, 0.0f, static_cast<float>(image_size_.height) - 1)) {
      return true;
    } else {
      return false;
    }
  }

  // Comparison function for qsort.
  static int Compare(const void* a, const void* b) {
    return *reinterpret_cast<const float*>(a) -
           *reinterpret_cast<const float*>(b);
  }

  // Returns the median flow within the given bounding box as determined
  // by a grid_width x grid_height grid.
  Point2f GetMedianFlow(const BoundingBox& bounding_box,
                        const bool filter_by_fb_error,
                        const int grid_width,
                        const int grid_height) const {
    const int kMaxPoints = 100;
    SCHECK(grid_width * grid_height <= kMaxPoints,
          "Too many points for Median flow!");

    const BoundingBox valid_box = bounding_box.Intersect(
        BoundingBox(0, 0, image_size_.width - 1, image_size_.height - 1));

    if (valid_box.GetArea() <= 0.0f) {
      return Point2f(0, 0);
    }

    float x_deltas[kMaxPoints];
    float y_deltas[kMaxPoints];

    int curr_offset = 0;
    for (int i = 0; i < grid_width; ++i) {
      for (int j = 0; j < grid_height; ++j) {
        const float x_in = valid_box.left_ +
            (valid_box.GetWidth() * i) / (grid_width - 1);

        const float y_in = valid_box.top_ +
            (valid_box.GetHeight() * j) / (grid_height - 1);

        float curr_flow_x;
        float curr_flow_y;
        const bool success = FindNewPositionOfPoint(x_in, y_in,
                                                    &curr_flow_x, &curr_flow_y);

        if (success) {
          x_deltas[curr_offset] = curr_flow_x;
          y_deltas[curr_offset] = curr_flow_y;
          ++curr_offset;
        } else {
          LOGW("Tracking failure!");
        }
      }
    }

    if (curr_offset > 0) {
      qsort(x_deltas, curr_offset, sizeof(*x_deltas), Compare);
      qsort(y_deltas, curr_offset, sizeof(*y_deltas), Compare);

      return Point2f(x_deltas[curr_offset / 2], y_deltas[curr_offset / 2]);
    }

    LOGW("No points were valid!");
    return Point2f(0, 0);
  }

  void SetFullframeAlignmentMatrix(const float* const align_matrix23) {
    if (align_matrix23 != NULL) {
      if (fullframe_matrix_ == NULL) {
        fullframe_matrix_ = new float[6];
      }

      memcpy(fullframe_matrix_, align_matrix23,
             6 * sizeof(fullframe_matrix_[0]));
    }
  }

 private:
  Point2f LookupGuessFromLevel(
      const int cache_level, const float x, const float y) const {
    // LOGE("Looking up guess at %5.2f %5.2f for level %d.", x, y, cache_level);

    // Cutoff at the target level and use the matrix transform instead.
    if (fullframe_matrix_ != NULL && cache_level == kCacheCutoff) {
      const float xnew = x * fullframe_matrix_[0] +
                         y * fullframe_matrix_[1] +
                             fullframe_matrix_[2];
      const float ynew = x * fullframe_matrix_[3] +
                         y * fullframe_matrix_[4] +
                             fullframe_matrix_[5];

      return Point2f(xnew - x, ynew - y);
    }

    const int level_dim = BlockDimForCacheLevel(cache_level);
    const int pixels_per_cache_block_x =
        (image_size_.width + level_dim - 1) / level_dim;
    const int pixels_per_cache_block_y =
        (image_size_.height + level_dim - 1) / level_dim;
    const int index_x = x / pixels_per_cache_block_x;
    const int index_y = y / pixels_per_cache_block_y;

    Point2f displacement;
    if (!(*has_cache_[cache_level])[index_y][index_x]) {
      (*has_cache_[cache_level])[index_y][index_x] = true;

      // Get the lower cache level's best guess, if it exists.
      displacement = cache_level >= kNumCacheLevels - 1 ?
          Point2f(0, 0) : LookupGuessFromLevel(cache_level + 1, x, y);
      // LOGI("Best guess at cache level %d is %5.2f, %5.2f.", cache_level,
      //      best_guess.x, best_guess.y);

      // Find the center of the block.
      const float center_x = (index_x + 0.5f) * pixels_per_cache_block_x;
      const float center_y = (index_y + 0.5f) * pixels_per_cache_block_y;
      const int pyramid_level = PyramidLevelForCacheLevel(cache_level);

      // LOGI("cache level %d: [%d, %d (%5.2f / %d, %5.2f / %d)] "
      //      "Querying %5.2f, %5.2f at pyramid level %d, ",
      //      cache_level, index_x, index_y,
      //      x, pixels_per_cache_block_x, y, pixels_per_cache_block_y,
      //      center_x, center_y, pyramid_level);

      // TODO(andrewharp): Turn on FB error filtering.
      const bool success = optical_flow_.FindFlowAtPointSingleLevel(
          pyramid_level, center_x, center_y, false,
          &displacement.x, &displacement.y);

      if (!success) {
        LOGV("Computation of cached value failed for level %d!", cache_level);
      }

      // Store the value for later use.
      (*displacements_[cache_level])[index_y][index_x] = displacement;
    } else {
      displacement = (*displacements_[cache_level])[index_y][index_x];
    }

    // LOGI("Returning %5.2f, %5.2f for level %d",
    //      displacement.x, displacement.y, cache_level);
    return displacement;
  }

  Point2f LookupGuess(const float x, const float y) const {
    if (x < 0 || x >= image_size_.width || y < 0 || y >= image_size_.height) {
      return Point2f(0, 0);
    }

    // LOGI("Looking up guess at %5.2f %5.2f.", x, y);
    if (kNumCacheLevels > 0) {
      return LookupGuessFromLevel(0, x, y);
    } else {
      return Point2f(0, 0);
    }
  }

  // Returns the number of cache bins in each dimension for a given level
  // of the cache.
  int BlockDimForCacheLevel(const int cache_level) const {
    // The highest (coarsest) cache level has a block dim of kCacheBranchFactor,
    // thus if there are 4 cache levels, requesting level 3 (0-based) should
    // return kCacheBranchFactor, level 2 should return kCacheBranchFactor^2,
    // and so on.
    int block_dim = kNumCacheLevels;
    for (int curr_level = kNumCacheLevels - 1; curr_level > cache_level;
        --curr_level) {
      block_dim *= kCacheBranchFactor;
    }
    return block_dim;
  }

  // Returns the level of the image pyramid that a given cache level maps to.
  int PyramidLevelForCacheLevel(const int cache_level) const {
    // Higher cache and pyramid levels have smaller dimensions. The highest
    // cache level should refer to the highest image pyramid level. The
    // lower, finer image pyramid levels are uncached (assuming
    // kNumCacheLevels < kNumPyramidLevels).
    return cache_level + (kNumPyramidLevels - kNumCacheLevels);
  }

  const OpticalFlowConfig* const config_;

  const Size image_size_;
  OpticalFlow optical_flow_;

  float* fullframe_matrix_;

  // Whether this value is currently present in the cache.
  Image<bool>* has_cache_[kNumCacheLevels];

  // The cached displacement values.
  Image<Point2f>* displacements_[kNumCacheLevels];

  TF_DISALLOW_COPY_AND_ASSIGN(FlowCache);
};

}  // namespace tf_tracking

#endif  // TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_FLOW_CACHE_H_
