/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// This file contains declarations relating to kernel cache configuration
// parameters recognized by the StreamExecutor.
#ifndef TENSORFLOW_STREAM_EXECUTOR_KERNEL_CACHE_CONFIG_H_
#define TENSORFLOW_STREAM_EXECUTOR_KERNEL_CACHE_CONFIG_H_

namespace stream_executor {

// This enum represents potential configurations of L1/shared memory when
// running a particular kernel. These values represent user preference, and
// the runtime is not required to respect these choices.
enum class KernelCacheConfig {
  // Indicates no preference for device L1/shared memory configuration.
  kNoPreference,

  // Indicates a preference for more shared memory than L1 cache.
  kPreferShared,

  // Indicates a preference for more L1 cache than shared memory.
  kPreferL1,

  // Indicates a preference for equal amounts of L1 cache and shared memory.
  kPreferEqual,
};

}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_KERNEL_CACHE_CONFIG_H_
