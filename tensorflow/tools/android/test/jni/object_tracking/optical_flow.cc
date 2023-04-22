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

#include "tensorflow/tools/android/test/jni/object_tracking/optical_flow.h"

#include <math.h>

#include "tensorflow/tools/android/test/jni/object_tracking/config.h"
#include "tensorflow/tools/android/test/jni/object_tracking/flow_cache.h"
#include "tensorflow/tools/android/test/jni/object_tracking/frame_pair.h"
#include "tensorflow/tools/android/test/jni/object_tracking/geom.h"
#include "tensorflow/tools/android/test/jni/object_tracking/image-inl.h"
#include "tensorflow/tools/android/test/jni/object_tracking/image.h"
#include "tensorflow/tools/android/test/jni/object_tracking/image_data.h"
#include "tensorflow/tools/android/test/jni/object_tracking/keypoint.h"
#include "tensorflow/tools/android/test/jni/object_tracking/keypoint_detector.h"
#include "tensorflow/tools/android/test/jni/object_tracking/time_log.h"
#include "tensorflow/tools/android/test/jni/object_tracking/utils.h"

namespace tf_tracking {

OpticalFlow::OpticalFlow(const OpticalFlowConfig* const config)
    : config_(config),
      frame1_(NULL),
      frame2_(NULL),
      working_size_(config->image_size) {}


void OpticalFlow::NextFrame(const ImageData* const image_data) {
  // Special case for the first frame: make sure the image ends up in
  // frame1_ so that keypoint detection can be done on it if desired.
  frame1_ = (frame1_ == NULL) ? image_data : frame2_;
  frame2_ = image_data;
}


// Static heart of the optical flow computation.
// Lucas Kanade algorithm.
bool OpticalFlow::FindFlowAtPoint_LK(const Image<uint8_t>& img_I,
                                     const Image<uint8_t>& img_J,
                                     const Image<int32_t>& I_x,
                                     const Image<int32_t>& I_y, const float p_x,
                                     const float p_y, float* out_g_x,
                                     float* out_g_y) {
  float g_x = *out_g_x;
  float g_y = *out_g_y;
  // Get values for frame 1.  They remain constant through the inner
  // iteration loop.
  float vals_I[kFlowArraySize];
  float vals_I_x[kFlowArraySize];
  float vals_I_y[kFlowArraySize];

  const int kPatchSize = 2 * kFlowIntegrationWindowSize + 1;
  const float kWindowSizeFloat = static_cast<float>(kFlowIntegrationWindowSize);

#if USE_FIXED_POINT_FLOW
  const int fixed_x_max = RealToFixed1616(img_I.width_less_one_) - 1;
  const int fixed_y_max = RealToFixed1616(img_I.height_less_one_) - 1;
#else
  const float real_x_max = I_x.width_less_one_ - EPSILON;
  const float real_y_max = I_x.height_less_one_ - EPSILON;
#endif

  // Get the window around the original point.
  const float src_left_real = p_x - kWindowSizeFloat;
  const float src_top_real = p_y - kWindowSizeFloat;
  float* vals_I_ptr = vals_I;
  float* vals_I_x_ptr = vals_I_x;
  float* vals_I_y_ptr = vals_I_y;
#if USE_FIXED_POINT_FLOW
  // Source integer coordinates.
  const int src_left_fixed = RealToFixed1616(src_left_real);
  const int src_top_fixed = RealToFixed1616(src_top_real);

  for (int y = 0; y < kPatchSize; ++y) {
    const int fp_y = Clip(src_top_fixed + (y << 16), 0, fixed_y_max);

    for (int x = 0; x < kPatchSize; ++x) {
      const int fp_x = Clip(src_left_fixed + (x << 16), 0, fixed_x_max);

      *vals_I_ptr++ = img_I.GetPixelInterpFixed1616(fp_x, fp_y);
      *vals_I_x_ptr++ = I_x.GetPixelInterpFixed1616(fp_x, fp_y);
      *vals_I_y_ptr++ = I_y.GetPixelInterpFixed1616(fp_x, fp_y);
    }
  }
#else
  for (int y = 0; y < kPatchSize; ++y) {
    const float y_pos = Clip(src_top_real + y, 0.0f, real_y_max);

    for (int x = 0; x < kPatchSize; ++x) {
      const float x_pos = Clip(src_left_real + x, 0.0f, real_x_max);

      *vals_I_ptr++ = img_I.GetPixelInterp(x_pos, y_pos);
      *vals_I_x_ptr++ = I_x.GetPixelInterp(x_pos, y_pos);
      *vals_I_y_ptr++ = I_y.GetPixelInterp(x_pos, y_pos);
    }
  }
#endif

  // Compute the spatial gradient matrix about point p.
  float G[] = { 0, 0, 0, 0 };
  CalculateG(vals_I_x, vals_I_y, kFlowArraySize, G);

  // Find the inverse of G.
  float G_inv[4];
  if (!Invert2x2(G, G_inv)) {
    return false;
  }

#if NORMALIZE
  const float mean_I = ComputeMean(vals_I, kFlowArraySize);
  const float std_dev_I = ComputeStdDev(vals_I, kFlowArraySize, mean_I);
#endif

  // Iterate kNumIterations times or until we converge.
  for (int iteration = 0; iteration < kNumIterations; ++iteration) {
    // Get values for frame 2.
    float vals_J[kFlowArraySize];

    // Get the window around the destination point.
    const float left_real = p_x + g_x - kWindowSizeFloat;
    const float top_real  = p_y + g_y - kWindowSizeFloat;
    float* vals_J_ptr = vals_J;
#if USE_FIXED_POINT_FLOW
    // The top-left sub-pixel is set for the current iteration (in 16:16
    // fixed). This is constant over one iteration.
    const int left_fixed = RealToFixed1616(left_real);
    const int top_fixed  = RealToFixed1616(top_real);

    for (int win_y = 0; win_y < kPatchSize; ++win_y) {
      const int fp_y = Clip(top_fixed + (win_y << 16), 0, fixed_y_max);
      for (int win_x = 0; win_x < kPatchSize; ++win_x) {
        const int fp_x = Clip(left_fixed + (win_x << 16), 0, fixed_x_max);
        *vals_J_ptr++ = img_J.GetPixelInterpFixed1616(fp_x, fp_y);
      }
    }
#else
    for (int win_y = 0; win_y < kPatchSize; ++win_y) {
      const float y_pos = Clip(top_real + win_y, 0.0f, real_y_max);
      for (int win_x = 0; win_x < kPatchSize; ++win_x) {
        const float x_pos = Clip(left_real + win_x, 0.0f, real_x_max);
        *vals_J_ptr++ = img_J.GetPixelInterp(x_pos, y_pos);
      }
    }
#endif

#if NORMALIZE
    const float mean_J = ComputeMean(vals_J, kFlowArraySize);
    const float std_dev_J = ComputeStdDev(vals_J, kFlowArraySize, mean_J);

    // TODO(andrewharp): Probably better to completely detect and handle the
    // "corner case" where the patch is fully outside the image diagonally.
    const float std_dev_ratio = std_dev_J > 0.0f ? std_dev_I / std_dev_J : 1.0f;
#endif

    // Compute image mismatch vector.
    float b_x = 0.0f;
    float b_y = 0.0f;

    vals_I_ptr = vals_I;
    vals_J_ptr = vals_J;
    vals_I_x_ptr = vals_I_x;
    vals_I_y_ptr = vals_I_y;

    for (int win_y = 0; win_y < kPatchSize; ++win_y) {
      for (int win_x = 0; win_x < kPatchSize; ++win_x) {
#if NORMALIZE
        // Normalized Image difference.
        const float dI =
            (*vals_I_ptr++ - mean_I) - (*vals_J_ptr++ - mean_J) * std_dev_ratio;
#else
        const float dI = *vals_I_ptr++ - *vals_J_ptr++;
#endif
        b_x += dI * *vals_I_x_ptr++;
        b_y += dI * *vals_I_y_ptr++;
      }
    }

    // Optical flow... solve n = G^-1 * b
    const float n_x = (G_inv[0] * b_x) + (G_inv[1] * b_y);
    const float n_y = (G_inv[2] * b_x) + (G_inv[3] * b_y);

    // Update best guess with residual displacement from this level and
    // iteration.
    g_x += n_x;
    g_y += n_y;

    // LOGV("Iteration %d: delta (%.3f, %.3f)", iteration, n_x, n_y);

    // Abort early if we're already below the threshold.
    if (Square(n_x) + Square(n_y) < Square(kTrackingAbortThreshold)) {
      break;
    }
  }  // Iteration.

  // Copy value back into output.
  *out_g_x = g_x;
  *out_g_y = g_y;
  return true;
}


// Pointwise flow using translational 2dof ESM.
bool OpticalFlow::FindFlowAtPoint_ESM(
    const Image<uint8_t>& img_I, const Image<uint8_t>& img_J,
    const Image<int32_t>& I_x, const Image<int32_t>& I_y,
    const Image<int32_t>& J_x, const Image<int32_t>& J_y, const float p_x,
    const float p_y, float* out_g_x, float* out_g_y) {
  float g_x = *out_g_x;
  float g_y = *out_g_y;
  const float area_inv = 1.0f / static_cast<float>(kFlowArraySize);

  // Get values for frame 1. They remain constant through the inner
  // iteration loop.
  uint8_t vals_I[kFlowArraySize];
  uint8_t vals_J[kFlowArraySize];
  int16_t src_gradient_x[kFlowArraySize];
  int16_t src_gradient_y[kFlowArraySize];

  // TODO(rspring): try out the IntegerPatchAlign() method once
  // the code for that is in ../common.
  const float wsize_float = static_cast<float>(kFlowIntegrationWindowSize);
  const int src_left_fixed = RealToFixed1616(p_x - wsize_float);
  const int src_top_fixed = RealToFixed1616(p_y - wsize_float);
  const int patch_size = 2 * kFlowIntegrationWindowSize + 1;

  // Create the keypoint template patch from a subpixel location.
  if (!img_I.ExtractPatchAtSubpixelFixed1616(src_left_fixed, src_top_fixed,
                                             patch_size, patch_size, vals_I) ||
      !I_x.ExtractPatchAtSubpixelFixed1616(src_left_fixed, src_top_fixed,
                                           patch_size, patch_size,
                                           src_gradient_x) ||
      !I_y.ExtractPatchAtSubpixelFixed1616(src_left_fixed, src_top_fixed,
                                           patch_size, patch_size,
                                           src_gradient_y)) {
    return false;
  }

  int bright_offset = 0;
  int sum_diff = 0;

  // The top-left sub-pixel is set for the current iteration (in 16:16 fixed).
  // This is constant over one iteration.
  int left_fixed = RealToFixed1616(p_x + g_x - wsize_float);
  int top_fixed  = RealToFixed1616(p_y + g_y - wsize_float);

  // The truncated version gives the most top-left pixel that is used.
  int left_trunc = left_fixed >> 16;
  int top_trunc = top_fixed >> 16;

  // Compute an initial brightness offset.
  if (kDoBrightnessNormalize &&
      left_trunc >= 0 && top_trunc >= 0 &&
      (left_trunc + patch_size) < img_J.width_less_one_ &&
      (top_trunc + patch_size) < img_J.height_less_one_) {
    int templ_index = 0;
    const uint8_t* j_row = img_J[top_trunc] + left_trunc;

    const int j_stride = img_J.stride();

    for (int y = 0; y < patch_size; ++y, j_row += j_stride) {
      for (int x = 0; x < patch_size; ++x) {
        sum_diff += static_cast<int>(j_row[x]) - vals_I[templ_index++];
      }
    }

    bright_offset = static_cast<int>(static_cast<float>(sum_diff) * area_inv);
  }

  // Iterate kNumIterations times or until we go out of image.
  for (int iteration = 0; iteration < kNumIterations; ++iteration) {
    int jtj[3] = { 0, 0, 0 };
    int jtr[2] = { 0, 0 };
    sum_diff = 0;

    // Extract the target image values.
    // Extract the gradient from the target image patch and accumulate to
    // the gradient of the source image patch.
    if (!img_J.ExtractPatchAtSubpixelFixed1616(left_fixed, top_fixed,
                                               patch_size, patch_size,
                                               vals_J)) {
      break;
    }

    const uint8_t* templ_row = vals_I;
    const uint8_t* extract_row = vals_J;
    const int16_t* src_dx_row = src_gradient_x;
    const int16_t* src_dy_row = src_gradient_y;

    for (int y = 0; y < patch_size; ++y, templ_row += patch_size,
         src_dx_row += patch_size, src_dy_row += patch_size,
         extract_row += patch_size) {
      const int fp_y = top_fixed + (y << 16);
      for (int x = 0; x < patch_size; ++x) {
        const int fp_x = left_fixed + (x << 16);
        int32_t target_dx = J_x.GetPixelInterpFixed1616(fp_x, fp_y);
        int32_t target_dy = J_y.GetPixelInterpFixed1616(fp_x, fp_y);

        // Combine the two Jacobians.
        // Right-shift by one to account for the fact that we add
        // two Jacobians.
        int32_t dx = (src_dx_row[x] + target_dx) >> 1;
        int32_t dy = (src_dy_row[x] + target_dy) >> 1;

        // The current residual b - h(q) == extracted - (template + offset)
        int32_t diff = static_cast<int32_t>(extract_row[x]) -
                       static_cast<int32_t>(templ_row[x]) - bright_offset;

        jtj[0] += dx * dx;
        jtj[1] += dx * dy;
        jtj[2] += dy * dy;

        jtr[0] += dx * diff;
        jtr[1] += dy * diff;

        sum_diff += diff;
      }
    }

    const float jtr1_float = static_cast<float>(jtr[0]);
    const float jtr2_float = static_cast<float>(jtr[1]);

    // Add some baseline stability to the system.
    jtj[0] += kEsmRegularizer;
    jtj[2] += kEsmRegularizer;

    const int64_t prod1 = static_cast<int64_t>(jtj[0]) * jtj[2];
    const int64_t prod2 = static_cast<int64_t>(jtj[1]) * jtj[1];

    // One ESM step.
    const float jtj_1[4] = { static_cast<float>(jtj[2]),
                             static_cast<float>(-jtj[1]),
                             static_cast<float>(-jtj[1]),
                             static_cast<float>(jtj[0]) };
    const double det_inv = 1.0 / static_cast<double>(prod1 - prod2);

    g_x -= det_inv * (jtj_1[0] * jtr1_float + jtj_1[1] * jtr2_float);
    g_y -= det_inv * (jtj_1[2] * jtr1_float + jtj_1[3] * jtr2_float);

    if (kDoBrightnessNormalize) {
      bright_offset +=
          static_cast<int>(area_inv * static_cast<float>(sum_diff) + 0.5f);
    }

    // Update top left position.
    left_fixed = RealToFixed1616(p_x + g_x - wsize_float);
    top_fixed  = RealToFixed1616(p_y + g_y - wsize_float);

    left_trunc = left_fixed >> 16;
    top_trunc = top_fixed >> 16;

    // Abort iterations if we go out of borders.
    if (left_trunc < 0 || top_trunc < 0 ||
        (left_trunc + patch_size) >= J_x.width_less_one_ ||
        (top_trunc + patch_size) >= J_y.height_less_one_) {
      break;
    }
  }  // Iteration.

  // Copy value back into output.
  *out_g_x = g_x;
  *out_g_y = g_y;
  return true;
}


bool OpticalFlow::FindFlowAtPointReversible(
    const int level, const float u_x, const float u_y,
    const bool reverse_flow,
    float* flow_x, float* flow_y) const {
  const ImageData& frame_a = reverse_flow ? *frame2_ : *frame1_;
  const ImageData& frame_b = reverse_flow ? *frame1_ : *frame2_;

  // Images I (prev) and J (next).
  const Image<uint8_t>& img_I = *frame_a.GetPyramidSqrt2Level(level * 2);
  const Image<uint8_t>& img_J = *frame_b.GetPyramidSqrt2Level(level * 2);

  // Computed gradients.
  const Image<int32_t>& I_x = *frame_a.GetSpatialX(level);
  const Image<int32_t>& I_y = *frame_a.GetSpatialY(level);
  const Image<int32_t>& J_x = *frame_b.GetSpatialX(level);
  const Image<int32_t>& J_y = *frame_b.GetSpatialY(level);

  // Shrink factor from original.
  const float shrink_factor = (1 << level);

  // Image position vector (p := u^l), scaled for this level.
  const float scaled_p_x = u_x / shrink_factor;
  const float scaled_p_y = u_y / shrink_factor;

  float scaled_flow_x = *flow_x / shrink_factor;
  float scaled_flow_y = *flow_y / shrink_factor;

  // LOGE("FindFlowAtPoint level %d: %5.2f, %5.2f (%5.2f, %5.2f)", level,
  //     scaled_p_x, scaled_p_y, &scaled_flow_x, &scaled_flow_y);

  const bool success = kUseEsm ?
    FindFlowAtPoint_ESM(img_I, img_J, I_x, I_y, J_x, J_y,
                        scaled_p_x, scaled_p_y,
                        &scaled_flow_x, &scaled_flow_y) :
    FindFlowAtPoint_LK(img_I, img_J, I_x, I_y,
                       scaled_p_x, scaled_p_y,
                       &scaled_flow_x, &scaled_flow_y);

  *flow_x = scaled_flow_x * shrink_factor;
  *flow_y = scaled_flow_y * shrink_factor;

  return success;
}


bool OpticalFlow::FindFlowAtPointSingleLevel(
    const int level,
    const float u_x, const float u_y,
    const bool filter_by_fb_error,
    float* flow_x, float* flow_y) const {
  if (!FindFlowAtPointReversible(level, u_x, u_y, false, flow_x, flow_y)) {
    return false;
  }

  if (filter_by_fb_error) {
    const float new_position_x = u_x + *flow_x;
    const float new_position_y = u_y + *flow_y;

    float reverse_flow_x = 0.0f;
    float reverse_flow_y = 0.0f;

    // Now find the backwards flow and confirm it lines up with the original
    // starting point.
    if (!FindFlowAtPointReversible(level, new_position_x, new_position_y,
                                   true,
                                   &reverse_flow_x, &reverse_flow_y)) {
      LOGE("Backward error!");
      return false;
    }

    const float discrepancy_length =
        sqrtf(Square(*flow_x + reverse_flow_x) +
              Square(*flow_y + reverse_flow_y));

    const float flow_length = sqrtf(Square(*flow_x) + Square(*flow_y));

    return discrepancy_length <
        (kMaxForwardBackwardErrorAllowed * flow_length);
  }

  return true;
}


// An implementation of the Pyramidal Lucas-Kanade Optical Flow algorithm.
// See http://robots.stanford.edu/cs223b04/algo_tracking.pdf for details.
bool OpticalFlow::FindFlowAtPointPyramidal(const float u_x, const float u_y,
                                           const bool filter_by_fb_error,
                                           float* flow_x, float* flow_y) const {
  const int max_level = MAX(kMinNumPyramidLevelsToUseForAdjustment,
                            kNumPyramidLevels - kNumCacheLevels);

  // For every level in the pyramid, update the coordinates of the best match.
  for (int l = max_level - 1; l >= 0; --l) {
    if (!FindFlowAtPointSingleLevel(l, u_x, u_y,
                                    filter_by_fb_error, flow_x, flow_y)) {
      return false;
    }
  }

  return true;
}

}  // namespace tf_tracking
