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

#include "tensorflow/compiler/xla/service/gpu/runtime/stream_synchronization.h"

#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/concurrent_region.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/support.h"

namespace xla {
namespace gpu {

static absl::Status AwaitImpl(ConcurrentRegionStatus* region_status,
                              int64_t from, absl::Span<const int64_t> to) {
  TF_ASSIGN_OR_RETURN(se::Stream * from_stream, region_status->GetStream(from));
  for (int64_t to_index : to) {
    TF_ASSIGN_OR_RETURN(se::Stream * to_stream,
                        region_status->GetStream(to_index));
    from_stream->ThenWaitFor(to_stream);
  }

  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//
// Define custom calls that mark the concurrent region in CUDA graphs.
//===----------------------------------------------------------------------===//

using xla::runtime::CustomCall;

XLA_RUNTIME_DEFINE_CUSTOM_CALL(Await, FunctionWrapper<AwaitImpl>(), checks,
                               CustomCall::Bind("xla.streams.await")
                                   .UserData<ConcurrentRegionStatus*>()
                                   .Attr<int64_t>("from")
                                   .Attr<absl::Span<const int64_t>>("to"));

void RegisterStreamSynchronizationCustomCalls(
    runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.streams.await", Await);
}

}  // namespace gpu
}  // namespace xla
