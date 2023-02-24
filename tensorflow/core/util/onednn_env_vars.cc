/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifdef INTEL_MKL

#include "tensorflow/core/util/onednn_env_vars.h"

#include "absl/base/call_once.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

bool AreWeightsFrozen() {
  static bool weights_const = false;
  static absl::once_flag once;
  absl::call_once(once, [&] {
    TF_CHECK_OK(ReadBoolFromEnvVar("TF_ONEDNN_ASSUME_FROZEN_WEIGHTS",
                                   /*default_value*/ false, &weights_const));
  });
  return weights_const;
}

bool UseSystemAlloc() {
  static bool use_sys_alloc = false;
  static absl::once_flag once;
  absl::call_once(once, [&] {
    TF_CHECK_OK(ReadBoolFromEnvVar("TF_ONEDNN_USE_SYSTEM_ALLOCATOR",
                                   /*default_value*/ false, &use_sys_alloc));
  });
  return use_sys_alloc;
}

bool ThreadPoolUseCallerThread() {
  static bool threadpool_use_caller_thread = false;
  static absl::once_flag once;
  absl::call_once(once, [&] {
    TF_CHECK_OK(ReadBoolFromEnvVar("TF_ONEDNN_THREADPOOL_USE_CALLER_THREAD",
                                   /*default_value*/ false,
                                   &threadpool_use_caller_thread));
  });
  return threadpool_use_caller_thread;
}

}  // namespace tensorflow
#endif  // INTEL_MKL
