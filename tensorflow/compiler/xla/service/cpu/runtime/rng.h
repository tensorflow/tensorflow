// Copyright 2023 The TensorFlow Authors
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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_RNG_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_RNG_H_

#include "tensorflow/compiler/xla/runtime/custom_call_registry.h"

namespace xla {
namespace cpu {

// Populate custom call implementing XLA CPU RNGs.
void PopulateXlaCpuRngCall(runtime::DirectCustomCallRegistry& registry);

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_RNG_H_
