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

#include "tensorflow/lite/experimental/litert/runtime/opencl/cl_event.h"

#include "absl/strings/str_format.h"
#include <CL/cl.h>
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/runtime/opencl/opencl_wrapper.h"

namespace litert {
namespace cl {

Expected<void> WaitForEvents(int num_events, const cl_event* event_list) {
  cl_int res = clWaitForEvents(num_events, event_list);
  if (res != CL_SUCCESS) {
    return Error(
        kLiteRtStatusErrorRuntimeFailure,
        absl::StrFormat("clWaitForEvents fails with error code %d", res));
  }
  return {};
}

Expected<void> SetUserEventStatus(cl_event event) {
  cl_int res = clSetUserEventStatus(event, CL_COMPLETE);
  if (res != CL_SUCCESS) {
    return Error(
        kLiteRtStatusErrorRuntimeFailure,
        absl::StrFormat("clSetUserEventStatus fails with error code %d", res));
  }
  return {};
}

Expected<cl_event> CreateUserEvent(cl_context context) {
  cl_int res;
  cl_event user_event = clCreateUserEvent(context, &res);
  if (res != CL_SUCCESS) {
    return Error(
        kLiteRtStatusErrorRuntimeFailure,
        absl::StrFormat("clCreateUserEvent fails with error code %d", res));
  }
  return user_event;
}

}  // namespace cl
}  // namespace litert
