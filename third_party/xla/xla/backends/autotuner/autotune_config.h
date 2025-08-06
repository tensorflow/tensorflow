/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_AUTOTUNER_AUTOTUNE_CONFIG_H_
#define XLA_BACKENDS_AUTOTUNER_AUTOTUNE_CONFIG_H_

#include <string>

namespace xla {

struct AutotuneConfig {
  // Whether to skip configs that failed to compile.
  bool skip_failing_configs = true;
  // Whether to check the correctness of the output buffers and OOM reads on
  // Input Buffers.
  bool check_buffers = false;
  // Relative tolerance for correctness check.
  float relative_tolerance = 1e-6;
  // Whether to crash the process on check failure.
  bool crash_on_check_failure = false;
  // Directory to store autotune results. If empty, caching is disabled.
  std::string autotune_cache_dir = "";
  // Mode for the autotune cache.
  // READ: Only load existing profiles, never run the autotuner.
  // WRITE: Only run the autotuner and write results, never load existing
  //        profiles.
  // READ_WRITE: Load existing profiles if available, otherwise run the
  //             autotuner and update the cache.
  enum class CacheMode {
    READ,
    WRITE,
    READ_WRITE,
  };
  CacheMode autotune_cache_mode = CacheMode::READ_WRITE;
};

}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_AUTOTUNE_CONFIG_H_
