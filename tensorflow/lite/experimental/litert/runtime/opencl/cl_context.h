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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_OPENCL_CL_CONTEXT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_OPENCL_CL_CONTEXT_H_

#include "absl/status/status.h"
#include <CL/cl.h>
#include "tensorflow/lite/experimental/litert/runtime/opencl/cl_device.h"

namespace litert {
namespace cl {

// A RAII wrapper around opencl context
class ClContext {
 public:
  ClContext();
  ClContext(cl_context context, bool has_ownership);
  ClContext(cl_context context, bool has_ownership, ClDevice& device);
  // Move only
  ClContext(ClContext&& context);
  ClContext& operator=(ClContext&& context);
  ClContext(const ClContext&) = delete;
  ClContext& operator=(const ClContext&) = delete;

  ~ClContext();

  cl_context context() const { return context_; }

 private:
  void Release();

  cl_context context_ = nullptr;
  bool has_ownership_ = false;
};

absl::Status CreateClContext(const ClDevice& device, ClContext* result);
absl::Status CreateClGlContext(const ClDevice& device,
                               cl_context_properties egl_context,
                               cl_context_properties egl_display,
                               ClContext* result);

}  // namespace cl
}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_OPENCL_CL_CONTEXT_H_
