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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_COMPATIBILITY_GPU_COMPATIBILITY_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_COMPATIBILITY_GPU_COMPATIBILITY_H_

#include <map>
#include <memory>
#include <string>

#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/experimental/acceleration/compatibility/android_info.h"
#include "tensorflow/lite/experimental/acceleration/compatibility/devicedb.h"

namespace tflite {
namespace acceleration {

// This class provides information on GPU delegate support.
//
// The GPU delegate is supported on a subset of Android devices, depending on
// Android version, OpenGL ES version, GPU chipset etc. The support is based on
// measure stability, correctness and peformance. For more detail see README.md.
//
// Example usage:
//   tflite::Interpreter* interpreter = ... ;
//   tflite::acceleration::AndroidInfo android_info;
//   tflite::gpu::GpuInfo gpu_info;
//   EXPECT_OK(tflite::acceleration::RequestAndroidInfo(&android_info));
//   EXPECT_OK(tflite::gpu::gl::EglEnvironment::NewEglEnvironment(&env));
//   EXPECT_OK(tflite::gpu::gl::RequestGpuInfo(&tflite_gpu_info));
//   tflite::acceleration::GPUCompatibilityList list;
//   TfLiteDelegate* gpu_delegate = nullptr;
//   TfLiteGpuDelegateOptions gpu_options;
//   if (list.Includes(android_info, gpu_info)) {
//     gpu_options = list.BestOptionsFor(android_info, gpu_info);
//     gpu_delegate = TfLiteGpuDelegateCreate(&gpu_options);
//     EXPECT_EQ(interpreter->ModifyGraphWithDelegate(gpu_delegate), TfLiteOk);
//   } else {
//     // Fallback path.
//   }
class GPUCompatibilityList {
 public:
  // Construct list from bundled data. Returns a unique_ptr to a nullptr if
  // creation fails.
  static std::unique_ptr<GPUCompatibilityList> Create();
  // Constructs list from the given flatbuffer data. Returns a unique_ptr to a
  // nullptr is the given flatbuffer is empty or invalid.
  static std::unique_ptr<GPUCompatibilityList> Create(
      const unsigned char* compatibility_list_flatbuffer, int length);
  // Returns true if the provided device specs are supported by the database.
  bool Includes(const AndroidInfo& android_info,
                const ::tflite::gpu::GpuInfo& gpu_info) const;

  // Returns the best TfLiteGpuDelegateOptionsV2 for the provided device specs
  // based on the database. The output can be modified as desired before passing
  // to delegate creation.
  TfLiteGpuDelegateOptionsV2 GetBestOptionsFor(
      const AndroidInfo& android_info,
      const ::tflite::gpu::GpuInfo& gpu_info) const;

  // Convert android_info and gpu_info into a set of variables used for querying
  // the list, and update variables from list data. See variables.h
  // and devicedb.h for more information.
  std::map<std::string, std::string> CalculateVariables(
      const AndroidInfo& android_info,
      const ::tflite::gpu::GpuInfo& gpu_info) const;

  GPUCompatibilityList(const GPUCompatibilityList&) = delete;
  GPUCompatibilityList& operator=(const GPUCompatibilityList&) = delete;

  // Checks if the provided byte array represents a valid compatibility list
  static bool IsValidFlatbuffer(const unsigned char* data, int len);

 protected:
  const DeviceDatabase* database_;

 private:
  explicit GPUCompatibilityList(
      const unsigned char* compatibility_list_flatbuffer);
};

}  // namespace acceleration
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_COMPATIBILITY_GPU_COMPATIBILITY_H_
