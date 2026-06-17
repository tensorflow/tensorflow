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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_MEMORY_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_MEMORY_H_

#include "tensorflow/lite/delegates/gpu/cl/opencl_wrapper.h"
#include "tensorflow/lite/delegates/gpu/common/access_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {

// RAII wrapper for OpenCL memory object.
//
// Image is moveable but not copyable.
class CLMemory {
 public:
  // Creates invalid object.
  CLMemory() : CLMemory(nullptr, false) {}

  CLMemory(cl_mem memory, bool has_ownership)
      : memory_(memory), has_ownership_(has_ownership) {}

  // Move-only
  CLMemory(const CLMemory&) = delete;
  CLMemory& operator=(const CLMemory&) = delete;
  CLMemory(CLMemory&& image)
      : memory_(image.memory_), has_ownership_(image.has_ownership_) {
    image.memory_ = nullptr;
  }

  ~CLMemory() { Invalidate(); }

  CLMemory& operator=(CLMemory&& image) {
    if (this != &image) {
      Invalidate();
      std::swap(memory_, image.memory_);
      has_ownership_ = image.has_ownership_;
    }
    return *this;
  }

  cl_mem memory() const { return memory_; }

  bool is_valid() const { return memory_ != nullptr; }

  // @return true if this object actually owns corresponding CL memory
  //         and manages it's lifetime.
  bool has_ownership() const { return has_ownership_; }

  cl_mem Release() {
    cl_mem to_return = memory_;
    memory_ = nullptr;
    return to_return;
  }

 private:
  void Invalidate() {
    if (memory_ && has_ownership_) {
      clReleaseMemObject(memory_);
    }
    memory_ = nullptr;
  }

  cl_mem memory_ = nullptr;
  bool has_ownership_ = false;
};

cl_mem_flags ToClMemFlags(AccessType access_type);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_MEMORY_H_
