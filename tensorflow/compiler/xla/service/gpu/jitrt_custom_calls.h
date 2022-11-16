// Copyright 2022 The TensorFlow Authors
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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_JITRT_CUSTOM_CALLS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_JITRT_CUSTOM_CALLS_H_

#include <cstdint>
#include <memory>
#include <tuple>

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "tensorflow/compiler/xla/mlir/runtime/transforms/custom_call_encoding.h"
#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/custom_call_registry.h"
#include "tensorflow/compiler/xla/runtime/type_id.h"
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/conv.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/tracing.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"

namespace xla {
namespace gpu {
class JitRtKernelsCache;

// Populate custom calls implementing XLA GPU runtime API.
void PopulateXlaGpuCustomCalls(runtime::DirectCustomCallRegistry& registry);

// Populate mapping from XLA (SE) enums/structs type id to symbol names.
void PopulateXlaGpuTypeIdNames(runtime::TypeIDNameRegistry& registry);

// Populate encoding from LMHLO attributes to XLA(SE) enums and structs.
void PopulateLmhloToXlaAttrEncoding(
    runtime::CustomCallAttrEncodingSet& encoding);




}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_JITRT_CUSTOM_CALLS_H_
