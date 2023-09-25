/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_ZEN_UTIL_H_
#define TENSORFLOW_CORE_UTIL_ZEN_UTIL_H_

#ifdef AMD_ZENDNN

#include "absl/base/call_once.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

inline int64_t GetMempool() {
  static absl::once_flag once;
  static int64_t mempool = 1;
  absl::call_once(once, [&] {
    TF_CHECK_OK(
        ReadInt64FromEnvVar("ZENDNN_ENABLE_MEMPOOL", mempool, &mempool));
    return mempool;
  });
  return mempool;
}

inline bool IsBlockedFormatEnabled() {
  static absl::once_flag once;
  static bool blocked_format = false;
  static int64_t conv_algo = 1;
  absl::call_once(once, [&] {
    TF_CHECK_OK(ReadInt64FromEnvVar("ZENDNN_CONV_ALGO", conv_algo, &conv_algo));
    blocked_format = (conv_algo == 3);
    return blocked_format;
  });
  return blocked_format;
}

}  // namespace tensorflow

#endif  // AMD_ZENDNN
#endif  // TENSORFLOW_CORE_UTIL_ZEN_UTIL_H_
