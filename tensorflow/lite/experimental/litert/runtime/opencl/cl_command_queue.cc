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

// This file is a copy of third_party/ml_drift/cl/cl_command_queue.cc.
#include "tensorflow/lite/experimental/litert/runtime/opencl/cl_command_queue.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include <CL/cl.h>
#include "tensorflow/lite/experimental/litert/runtime/opencl/cl_context.h"
#include "tensorflow/lite/experimental/litert/runtime/opencl/cl_device.h"
#include "tensorflow/lite/experimental/litert/runtime/opencl/opencl_wrapper.h"

namespace litert {
namespace cl {
namespace {

absl::StatusOr<cl_command_queue> CreateClCommandQueueWithProperties(
    const ClDevice& device, const ClContext& context,
    cl_command_queue_properties queue_properties) {
  int error_code;
  cl_command_queue queue;
  if (clCreateCommandQueueWithProperties) {
    std::vector<cl_queue_properties> props;
    if (queue_properties != 0) {
      props.push_back(CL_QUEUE_PROPERTIES);
      props.push_back(queue_properties);
    }
    props.push_back(0);

    queue = clCreateCommandQueueWithProperties(context.context(), device.id(),
                                               props.data(), &error_code);
  } else {
    // Backwards compatibility for OpenCL versions before 2.0.
    queue = clCreateCommandQueue(context.context(), device.id(),
                                 queue_properties, &error_code);
  }
  if (!queue) {
    return absl::UnknownError(absl::StrCat(
        "Failed to create a command queue - ", std::to_string(error_code)));
  }
  return queue;
}

}  // namespace

ClCommandQueue::ClCommandQueue() = default;

ClCommandQueue::ClCommandQueue(cl_command_queue queue, bool has_ownership)
    : queue_(queue), has_ownership_(has_ownership) {}

ClCommandQueue::ClCommandQueue(ClCommandQueue&& queue)
    : queue_(queue.queue_), has_ownership_(queue.has_ownership_) {
  queue.queue_ = nullptr;
}

ClCommandQueue& ClCommandQueue::operator=(ClCommandQueue&& queue) {
  if (this != &queue) {
    Release();
    std::swap(queue_, queue.queue_);
    has_ownership_ = queue.has_ownership_;
  }
  return *this;
}

ClCommandQueue::~ClCommandQueue() { Release(); }

void ClCommandQueue::Release() {
  if (has_ownership_ && queue_) {
    clReleaseCommandQueue(queue_);
    queue_ = nullptr;
  }
}

absl::Status ClCommandQueue::EnqueueWriteBuffer(cl_mem memory,
                                                size_t size_in_bytes,
                                                const void* data, bool async) {
  const cl_bool blocking = async ? CL_FALSE : CL_TRUE;
  auto error_code = clEnqueueWriteBuffer(
      queue_, memory, blocking, 0, size_in_bytes, data, 0, nullptr, nullptr);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrCat("Failed to upload data to GPU (clEnqueueWriteBuffer) - ",
                     std::to_string(error_code)));
  }
  return absl::OkStatus();
}

absl::Status ClCommandQueue::EnqueueReadBuffer(cl_mem memory,
                                               size_t size_in_bytes, void* data,
                                               bool async) {
  const cl_bool blocking = async ? CL_FALSE : CL_TRUE;
  auto error_code = clEnqueueReadBuffer(
      queue_, memory, blocking, 0, size_in_bytes, data, 0, nullptr, nullptr);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrCat("Failed to read data from GPU (clEnqueueReadBuffer) - ",
                     std::to_string(error_code)));
  }
  return absl::OkStatus();
}

absl::Status ClCommandQueue::WaitForCompletion() {
  auto error_code = clFinish(queue_);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrCat("Failed to clFinish - ", std::to_string(error_code)));
  }
  return absl::OkStatus();
}

absl::Status CreateClCommandQueue(const ClDevice& device,
                                  const ClContext& context,
                                  ClCommandQueue* result) {
  auto queue = CreateClCommandQueueWithProperties(device, context, 0);
  if (!queue.ok()) {
    return queue.status();
  }
  *result = ClCommandQueue(*queue, true);
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace litert
