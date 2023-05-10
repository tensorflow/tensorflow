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

#include "tensorflow/compiler/xla/service/gpu/runtime/concurrent_region.h"

#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/graph_launch.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/support.h"

namespace xla {
namespace gpu {

//===----------------------------------------------------------------------===//
// Define custom calls that mark the concurrent region in CUDA graphs.
//===----------------------------------------------------------------------===//

using xla::runtime::CustomCall;

static absl::Status RegionBegin(ConcurrentRegionStatus* region_status) {
  region_status->StartConcurrentRegion();
  return absl::OkStatus();
}

static absl::Status RegionEnd(ConcurrentRegionStatus* region_status) {
  region_status->EndConcurrentRegion();
  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    Begin, FunctionWrapper<RegionBegin>(), checks,
    CustomCall::Bind("xla.gpu.concurrent_region.begin")
        .UserData<ConcurrentRegionStatus*>());

XLA_RUNTIME_DEFINE_CUSTOM_CALL(End, FunctionWrapper<RegionEnd>(), checks,
                               CustomCall::Bind("xla.gpu.concurrent_region.end")
                                   .UserData<ConcurrentRegionStatus*>());

void RegisterConcurrentRegionCustomCalls(
    runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.gpu.concurrent_region.begin", Begin);
  registry.Register("xla.gpu.concurrent_region.end", End);
}

}  // namespace gpu
}  // namespace xla
