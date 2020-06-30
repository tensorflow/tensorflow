/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_KERNEL_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_KERNEL_H_

#include <string>

#include "tensorflow/lite/delegates/gpu/cl/cl_context.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_program.h"
#include "tensorflow/lite/delegates/gpu/cl/opencl_wrapper.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {

// Arguments binding to CLKernel can be manual or automatic
// In manual you specify binding index explicitly
// In automatic binding, index auto-incremented with every binding call
// Also, if you use automatic mode you must call ResetBindingCounter
//   before parameters binding
class CLKernel {
 public:
  CLKernel() {}

  // Move only
  CLKernel(CLKernel&& kernel);
  CLKernel& operator=(CLKernel&& kernel);
  CLKernel(const CLKernel&) = delete;
  CLKernel& operator=(const CLKernel&) = delete;

  ~CLKernel();

  cl_kernel kernel() const { return kernel_; }

  absl::Status CreateFromProgram(const CLProgram& program,
                                 const std::string& function_name);

  absl::Status SetMemory(int index, cl_mem memory);
  absl::Status SetMemoryAuto(cl_mem memory);
  template <typename T>
  absl::Status SetBytes(int index, const T& value) const {
    return SetBytes(index, static_cast<const void*>(&value), sizeof(T));
  }
  template <typename T>
  absl::Status SetBytesAuto(const T& value) {
    return SetBytesAuto(static_cast<const void*>(&value), sizeof(T));
  }

  int GetPrivateMemorySize() const { return private_memory_size_; }
  int GetMaxWorkGroupSize() const { return max_work_group_size_; }

  int GetBindingCounter() const { return binding_counter_; }
  void ResetBindingCounter() { binding_counter_ = 0; }

  // Do not use this function
  // workaround for Mali memory leak
  absl::Status ReInit() const;

 private:
  void Release();
  absl::Status SetBytes(int index, const void* ptr, int length) const;
  absl::Status SetBytesAuto(const void* ptr, int length);

  int private_memory_size_;
  int max_work_group_size_;
  int binding_counter_ = -1;

  std::string function_name_;
  // reference to program from which kernel was created
  cl_program program_ = nullptr;
  cl_kernel kernel_ = nullptr;
};

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_KERNEL_H_
