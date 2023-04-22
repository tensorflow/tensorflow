/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_KEY_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_KEY_H_

#include <functional>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"

namespace tensorflow {
namespace tpu {

struct TpuCompilationCacheKey {
  // Prefix of the key.
  std::string prefix;

  // A boolean flag to specify if `guaranteed_const` is used. Guarantee const is
  // normally used in TPU inference to avoid re-copying unchanged variables onto
  // the TPU device. It promises the value is identical for every execution in
  // the same session even if the actual value changes in later executions.
  bool has_guaranteed_const = false;

  // Unique session identifier. It is set when `has_guaranteed_const` is true.
  std::string session_handle;

  // Fingerprint of `guaranteed_const` value. It is set when the value of the
  // `has_guaranteed_const` is true. Produce the value when necessary.
  std::function<std::string()> guaranteed_const_fingerprint;

  // A more verbose key for debugging purpose.
  std::string debug_string;

  // Constructs the TPU compilation cache key by concatenating the `prefix`,
  // `session_handle` and `guaranteed_const_fingerprint`.
  std::string ToString() const {
    if (!has_guaranteed_const) {
      return prefix;
    }
    return absl::StrCat(prefix, "|", session_handle, "|",
                        guaranteed_const_fingerprint());
  }

  explicit TpuCompilationCacheKey() {}
  explicit TpuCompilationCacheKey(const std::string& p) : prefix(p) {}
};

}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_KEY_H_
