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

#ifndef XLA_SERVICE_CPU_RUNTIME_CUSTOM_CALL_STATUS_H_
#define XLA_SERVICE_CPU_RUNTIME_CUSTOM_CALL_STATUS_H_

extern "C" {

// Returns true iff the given 'XlaCustomCallStatus' is in a success state, so
// that generated code can return early if a CustomCall fails.
extern bool __xla_cpu_runtime_StatusIsSuccess(
    const void* /* XlaCustomCallStatus* */ status_ptr);
}

#endif  // XLA_SERVICE_CPU_RUNTIME_CUSTOM_CALL_STATUS_H_
