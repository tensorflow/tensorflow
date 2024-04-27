/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/pjrt/c/pjrt_c_api_gpu.h"

#include "absl/base/call_once.h"
#include "absl/log/initialize.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_gpu_internal.h"
#include "tsl/platform/platform.h"

const PJRT_Api* GetPjrtApi() {
  // Initialize ABSL logging because code within XLA uses it.
#ifndef PLATFORM_GOOGLE
  static absl::once_flag once;
  absl::call_once(once, []() { absl::InitializeLog(); });
#endif  // PLATFORM_GOOGLE
  return pjrt::gpu_plugin::GetGpuPjrtApi();
}
