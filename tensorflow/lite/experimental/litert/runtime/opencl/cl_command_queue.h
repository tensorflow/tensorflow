// Copyright 2024 The TensorFlow Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file is a copy of third_party/ml_drift/cl/cl_command_queue.h.
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_OPENCL_CL_COMMAND_QUEUE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_OPENCL_CL_COMMAND_QUEUE_H_

#include <cstddef>
#include <string>

#include "absl/status/status.h"
#include <CL/cl.h>
#include "tensorflow/lite/experimental/litert/runtime/opencl/cl_context.h"
#include "tensorflow/lite/experimental/litert/runtime/opencl/cl_device.h"

namespace litert {
namespace cl {

// A wrapper around opencl command queue
class ClCommandQueue {
 public:
  ClCommandQueue();
  ClCommandQueue(cl_command_queue queue, bool has_ownership);

  // Move only
  ClCommandQueue(ClCommandQueue&& queue);
  ClCommandQueue& operator=(ClCommandQueue&& queue);
  ClCommandQueue(const ClCommandQueue&) = delete;
  ClCommandQueue& operator=(const ClCommandQueue&) = delete;

  virtual ~ClCommandQueue();

  cl_command_queue queue() const { return queue_; }

  absl::Status EnqueueWriteBuffer(cl_mem memory, size_t size_in_bytes,
                                  const void* data, bool async = false);
  absl::Status EnqueueReadBuffer(cl_mem memory, size_t size_in_bytes,
                                 void* data, bool async = false);

  absl::Status WaitForCompletion();

 protected:
  void Release();

  cl_command_queue queue_ = nullptr;
  bool has_ownership_ = false;
};

class ProfilingCommandQueue : public ClCommandQueue {
 public:
  ProfilingCommandQueue();
  explicit ProfilingCommandQueue(cl_command_queue queue);

  // Move only
  ProfilingCommandQueue(ProfilingCommandQueue&& queue);
  ProfilingCommandQueue& operator=(ProfilingCommandQueue&& queue);
  ProfilingCommandQueue(const ProfilingCommandQueue&) = delete;
  ProfilingCommandQueue& operator=(const ProfilingCommandQueue&) = delete;

 private:
  std::string current_label_;
};

absl::Status CreateClCommandQueue(const ClDevice& device,
                                  const ClContext& context,
                                  ClCommandQueue* result);

}  // namespace cl
}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_OPENCL_CL_COMMAND_QUEUE_H_
