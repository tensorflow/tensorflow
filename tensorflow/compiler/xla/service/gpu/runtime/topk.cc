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

#include "tensorflow/compiler/xla/service/gpu/runtime/topk.h"

#include <stdint.h>

#include <cstddef>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/custom_call_registry.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/support.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/topk_kernel.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla::gpu {
using ::xla::runtime::CustomCall;
using ::xla::runtime::StridedMemrefView;

static absl::Status TopkImpl(const ServiceExecutableRunOptions* run_options,
                             StridedMemrefView data,
                             StridedMemrefView top_elements,
                             StridedMemrefView indices) {
  if (data.sizes.size() > 2)
    return absl::InvalidArgumentError("Invalid input shape");
  if (indices.dtype != PrimitiveType::S32)
    return absl::InvalidArgumentError("Indices should be S32");
  bool has_batch = data.sizes.size() == 2;
  size_t batch_size = has_batch ? data.sizes[0] : 1;
  size_t n = has_batch ? data.sizes[1] : data.sizes[0];
  size_t k = has_batch ? top_elements.sizes[1] : top_elements.sizes[0];
  return RunTopk(se::gpu::AsGpuStreamValue(run_options->stream()), data.dtype,
                 data.data, n, top_elements.data,
                 static_cast<uint32_t*>(indices.data), k, batch_size);
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
