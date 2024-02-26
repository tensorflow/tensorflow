/* Copyright 2021 The OpenXLA Authors.

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
#include "xla/service/cpu/runtime_custom_call_status.h"

#include "absl/base/dynamic_annotations.h"
#include "xla/service/custom_call_status_internal.h"

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY bool __xla_cpu_runtime_StatusIsSuccess(
    const void* status_ptr) {
  auto status = static_cast<const XlaCustomCallStatus*>(status_ptr);
  return !xla::CustomCallStatusGetMessage(status).has_value();
}
