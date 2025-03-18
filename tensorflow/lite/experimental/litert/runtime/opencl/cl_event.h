// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_OPENCL_CL_EVENT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_OPENCL_CL_EVENT_H_

#include <CL/cl.h>
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

namespace litert {
namespace cl {

Expected<void> WaitForEvents(int num_events, const cl_event* event_list);

Expected<void> SetUserEventStatus(cl_event event);

Expected<cl_event> CreateUserEvent(cl_context context);

}  // namespace cl
}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_OPENCL_CL_EVENT_H_
