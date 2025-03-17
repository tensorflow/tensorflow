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

#include "tensorflow/lite/experimental/litert/runtime/opencl/cl_context.h"

#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include <CL/cl.h>
#include "tensorflow/lite/experimental/litert/runtime/opencl/cl_device.h"
#include "tensorflow/lite/experimental/litert/runtime/opencl/opencl_wrapper.h"

namespace litert {
namespace cl {
namespace {

absl::Status CreateClContext(const ClDevice& device,
                             const std::vector<cl_context_properties>& props,
                             ClContext* result) {
  int error_code;
  cl_device_id device_id = device.id();
  std::vector<cl_context_properties> props_local = props;
  if (!props_local.empty()) {
    props_local.push_back(0);
  }
  cl_context_properties* properties_ptr =
      props_local.empty() ? nullptr : props_local.data();
  cl_context context = clCreateContext(properties_ptr, 1, &device_id, nullptr,
                                       nullptr, &error_code);
  if (!context) {
    return absl::UnknownError(
        absl::StrCat("Failed to create a compute context - ", error_code));
  }

  *result = ClContext(context, true);
  return absl::OkStatus();
}

}  // namespace

ClContext::ClContext() = default;

ClContext::ClContext(cl_context context, bool has_ownership)
    : context_(context), has_ownership_(has_ownership) {}

ClContext::ClContext(cl_context context, bool has_ownership, ClDevice& device)
    : context_(context), has_ownership_(has_ownership) {}

ClContext::ClContext(ClContext&& context)
    : context_(context.context_), has_ownership_(context.has_ownership_) {
  context.context_ = nullptr;
}

ClContext& ClContext::operator=(ClContext&& context) {
  if (this != &context) {
    Release();
    std::swap(context_, context.context_);
    has_ownership_ = context.has_ownership_;
  }
  return *this;
}

ClContext::~ClContext() { Release(); }

void ClContext::Release() {
  if (has_ownership_ && context_) {
    clReleaseContext(context_);
    context_ = nullptr;
  }
}

absl::Status CreateClContext(const ClDevice& device, ClContext* result) {
  std::vector<cl_context_properties> props;
  return CreateClContext(device, props, result);
}

absl::Status CreateClGlContext(const ClDevice& device,
                               cl_context_properties egl_context,
                               cl_context_properties egl_display,
                               ClContext* result) {
  cl_context_properties platform =
      reinterpret_cast<cl_context_properties>(device.platform());

  std::vector<cl_context_properties> props = {CL_GL_CONTEXT_KHR,   egl_context,
                                              CL_EGL_DISPLAY_KHR,  egl_display,
                                              CL_CONTEXT_PLATFORM, platform};

  return CreateClContext(device, props, result);
}

}  // namespace cl
}  // namespace litert
