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
#include <string>

#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
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
// Reads from the flatbuffer.
// Example usage:
//   tflite::acceleration::GPUCompatibilityList list;
//   tflite::acceleration::AndroidInfo android_info;
//   tflite::gpu::GpuInfo gpu_info;
//   ...
//   if(list.Includes(android_info, gpu_info)){
//    // SUPPORTED.
//   } else{
//    // UNSUPPORTED.
//   }
class GPUCompatibilityList {
 public:
  // Construct list from bundled data.
  GPUCompatibilityList();
  // Constructs list from the given flatbuffer.
  explicit GPUCompatibilityList(
      const unsigned char* compatibility_list_flatbuffer);
  // Returns true if the provided device specs are supported by the database.
  bool Includes(const AndroidInfo& android_info,
                const ::tflite::gpu::GpuInfo& gpu_info) const;
  // Convert android_info and gpu_info into a set of variables used for querying
  // the list, and update variables from list data. See variables.h
  // and devicedb.h for more information.
  std::map<std::string, std::string> CalculateVariables(
      const AndroidInfo& android_info,
      const ::tflite::gpu::GpuInfo& gpu_info) const;
  GPUCompatibilityList(const GPUCompatibilityList&) = delete;
  GPUCompatibilityList& operator=(const GPUCompatibilityList&) = delete;
  // Indicates if the database is loaded.
  bool IsDatabaseLoaded() const;

 protected:
  const DeviceDatabase* database_;
};
}  // namespace acceleration
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_COMPATIBILITY_GPU_COMPATIBILITY_H_
