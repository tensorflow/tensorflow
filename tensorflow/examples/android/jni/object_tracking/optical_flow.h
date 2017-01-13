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

#ifndef THIRD_PARTY_TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_OPTICAL_FLOW_H_
#define THIRD_PARTY_TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_OPTICAL_FLOW_H_

#include "tensorflow/core/platform/types.h"

#include "tensorflow/examples/android/jni/object_tracking/geom.h"
#include "tensorflow/examples/android/jni/object_tracking/image-inl.h"
#include "tensorflow/examples/android/jni/object_tracking/image.h"
#include "tensorflow/examples/android/jni/object_tracking/utils.h"

#include "tensorflow/examples/android/jni/object_tracking/config.h"
#include "tensorflow/examples/android/jni/object_tracking/frame_pair.h"
#include "tensorflow/examples/android/jni/object_tracking/image_data.h"
#include "tensorflow/examples/android/jni/object_tracking/keypoint.h"

using namespace tensorflow;

namespace tf_tracking {

class FlowCache;

// Class encapsulating all the data and logic necessary for performing optical
// flow.
class OpticalFlow {
 public:
  explicit OpticalFlow(const OpticalFlowConfig* const config);

  // Add a new frame to the optical flow.  Will update all the non-keypoint
  // related member variables.
  //
  // new_frame should be a buffer of grayscale values, one byte per pixel,
  // at the original frame_width and frame_height used to initialize the
  // OpticalFlow object.  Downsampling will be handled internally.
  //
  // time_stamp should be a time in milliseconds that later calls to this and
  // other methods will be relative to.
  void NextFrame(const ImageData* const image_data);

  // An implementation of the Lucas-Kanade Optical Flow algorithm.
  static bool FindFlowAtPoint_LK(const Image<uint8>& img_I,
                                 const Image<uint8>& img_J,
                                 const Image<int32>& I_x,
                                 const Image<int32>& I_y,
                                 const float p_x,
                                 const float p_y,
                                 float* out_g_x,
                                 float* out_g_y);

  // Pointwise flow using translational 2dof ESM.
  static bool FindFlowAtPoint_ESM(const Image<uint8>& img_I,
                                  const Image<uint8>& img_J,
                                  const Image<int32>& I_x,
                                  const Image<int32>& I_y,
                                  const Image<int32>& J_x,
                                  const Image<int32>& J_y,
                                  const float p_x,
                                  const float p_y,
                                  float* out_g_x,
                                  float* out_g_y);

  // Finds the flow using a specific level, in either direction.
  // If reversed, the coordinates are in the context of the latest
  // frame, not the frame before it.
  // All coordinates used in parameters are global, not scaled.
  bool FindFlowAtPointReversible(
      const int level, const float u_x, const float u_y,
      const bool reverse_flow,
      float* final_x, float* final_y) const;

  // Finds the flow using a specific level, filterable by forward-backward
  // error. All coordinates used in parameters are global, not scaled.
  bool FindFlowAtPointSingleLevel(const int level,
                                  const float u_x, const float u_y,
                                  const bool filter_by_fb_error,
                                  float* flow_x, float* flow_y) const;

  // Pyramidal optical-flow using all levels.
  bool FindFlowAtPointPyramidal(const float u_x, const float u_y,
                                const bool filter_by_fb_error,
                                float* flow_x, float* flow_y) const;

 private:
  const OpticalFlowConfig* const config_;

  const ImageData* frame1_;
  const ImageData* frame2_;

  // Size of the internally allocated images (after original is downsampled).
  const Size working_size_;

  TF_DISALLOW_COPY_AND_ASSIGN(OpticalFlow);
};

}  // namespace tf_tracking

#endif  // THIRD_PARTY_TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_OPTICAL_FLOW_H_
