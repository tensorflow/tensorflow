/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/runtime/topk.h"

#include "absl/status/status.h"
#include "xla/runtime/custom_call.h"
#include "xla/runtime/custom_call_registry.h"
#include "xla/runtime/executable.h"
#include "xla/service/gpu/runtime/support.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
using ::xla::runtime::CustomCall;
using ::xla::runtime::StridedMemrefView;

static absl::Status TopkImpl(const ServiceExecutableRunOptions* run_options,
                             StridedMemrefView data,
                             StridedMemrefView top_elements,
                             StridedMemrefView indices) {
  return absl::UnimplementedError("TopkImpl is not implemented.");
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    Topk, FunctionWrapper<TopkImpl>(), checks,
    CustomCall::Bind("__gpu$TopK")
        .UserData<const ServiceExecutableRunOptions*>()
        .Arg<StridedMemrefView>()  // input
        .Arg<StridedMemrefView>()  // output (values)
        .Arg<StridedMemrefView>()  // output (indices)
);

void RegisterTopkCustomCall(runtime::DirectCustomCallRegistry& registry) {
  registry.Register("__gpu$TopK", Topk);
}

}  // namespace xla::gpu
