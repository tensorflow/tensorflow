/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_COMPATIBILITY_GPU_COMPATIBILITY_RECOMMENDER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_COMPATIBILITY_GPU_COMPATIBILITY_RECOMMENDER_H_

#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/experimental/acceleration/compatibility/android_info.h"
#include "tensorflow/lite/experimental/acceleration/compatibility/gpu_compatibility.h"

namespace tflite {
namespace acceleration {

// This class recommends best TfLiteGPU delegate options for Android devices.
//
// Example usage:
//   tflite::Interpreter* interpreter = ... ;
//   tflite::acceleration::AndroidInfo android_info;
//   tflite::gpu::GpuInfo gpu_info;
//   CHECK(tflite::acceleration::RequestAndroidInfo(&android_info));
//   CHECK(tflite::gpu::gl::EglEnvironment::NewEglEnvironment(&env));
//   CHECK(tflite::gpu::gl::RequestGpuInfo(&tflite_gpu_info));
//   tflite::acceleration::GPUCompatibilityRecommender recommender;
//   TfLiteDelegate* gpu_delegate = nullptr;
//   TfLiteGpuDelegateOptions gpu_options;
//   if (list.Includes(android_info, gpu_info)) {
//     gpu_options = recommender.BestOptionsFor(android_info, gpu_info);
//     gpu_delegate = TfLiteGpuDelegateCreate(&gpu_options);
//     CHECK_EQ(interpreter->ModifyGraphWithDelegate(gpu_delegate), TfLiteOk);
//   } else {
//     // Fallback path.
//   }

class GPUCompatibilityRecommender : public GPUCompatibilityList {
 public:
  GPUCompatibilityRecommender() {}
  GPUCompatibilityRecommender(const GPUCompatibilityRecommender&) = delete;
  GPUCompatibilityRecommender& operator=(const GPUCompatibilityRecommender&) =
      delete;

  // Returns the best TfLiteGpuDelegateOptionsV2 for the provided device specs
  // based on the database. The output can be modified as desired before passing
  // to delegate creation.
  TfLiteGpuDelegateOptionsV2 GetBestOptionsFor(
      const AndroidInfo& android_info,
      const ::tflite::gpu::GpuInfo& gpu_info) const;
};

}  // namespace acceleration
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_COMPATIBILITY_GPU_COMPATIBILITY_RECOMMENDER_H_
