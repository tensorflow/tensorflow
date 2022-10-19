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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_CONTEXT_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_CONTEXT_H_

#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"
#include "tensorflow/lite/delegates/gpu/cl/opencl_wrapper.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {

// A RAII wrapper around opencl context
class CLContext {
 public:
  CLContext() {}
  CLContext(cl_context context, bool has_ownership);
  CLContext(cl_context context, bool has_ownership, CLDevice& device);
  // Move only
  CLContext(CLContext&& context);
  CLContext& operator=(CLContext&& context);
  CLContext(const CLContext&) = delete;
  CLContext& operator=(const CLContext&) = delete;

  ~CLContext();

  cl_context context() const { return context_; }

  bool IsFloatTexture2DSupported(int num_channels, DataType data_type,
                                 cl_mem_flags flags = CL_MEM_READ_WRITE) const;

 private:
  void Release();

  cl_context context_ = nullptr;
  bool has_ownership_ = false;
};

absl::Status CreateCLContext(const CLDevice& device, CLContext* result);
absl::Status CreateCLGLContext(const CLDevice& device,
                               cl_context_properties egl_context,
                               cl_context_properties egl_display,
                               CLContext* result);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_CONTEXT_H_
