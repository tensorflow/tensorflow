/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_COMMAND_BUFFER_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_COMMAND_BUFFER_H_

#include "tensorflow/lite/delegates/gpu/cl/cl_command_queue.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_event.h"
#include "tensorflow/lite/delegates/gpu/cl/opencl_wrapper.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {

class CLCommandBuffer {
 public:
  CLCommandBuffer() = default;
  // Move only
  CLCommandBuffer(CLCommandBuffer&& cb);
  CLCommandBuffer& operator=(CLCommandBuffer&& cb);
  CLCommandBuffer(const CLCommandBuffer&) = delete;
  CLCommandBuffer& operator=(const CLCommandBuffer&) = delete;

  ~CLCommandBuffer() { Release(); }

  absl::Status Init(CLCommandQueue* queue, bool simultaneous_use = false);
  absl::Status Finalize();
  absl::Status Enqueue(CLCommandQueue* queue, CLEvent* event = nullptr);
  cl_command_buffer_khr GetCommandBuffer() const { return cb_; }

 private:
  void Release();
  cl_command_buffer_khr cb_ = nullptr;
};

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_COMMAND_BUFFER_H_
