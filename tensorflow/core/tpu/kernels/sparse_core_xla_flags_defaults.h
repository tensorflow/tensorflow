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

#ifndef TENSORFLOW_CORE_TPU_KERNELS_SPARSE_CORE_XLA_FLAGS_DEFAULTS_H_
#define TENSORFLOW_CORE_TPU_KERNELS_SPARSE_CORE_XLA_FLAGS_DEFAULTS_H_

#include <stdint.h>

namespace tensorflow {

constexpr int kDefaultSparseCoreMinibatchMaxDivisionLevel = 6;
constexpr bool kDefaultDisableTableStacking = false;
constexpr int64_t kDefaultXlaSparseCoreStackingMemLimit = 2097152;
constexpr int64_t kDefaultXlaSparseCoreStackingTableShardLimit = 2147483648;

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_SPARSE_CORE_XLA_FLAGS_DEFAULTS_H_
