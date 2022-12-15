/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/runtime/gemm.h"

#include <utility>

#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/support.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/xla.pb.h"

namespace xla {
namespace gpu {

using xla::runtime::CustomCall;
using xla::runtime::State;
using xla::runtime::StridedMemrefView;

static absl::Status GemmImpl(const ServiceExecutableRunOptions* run_options,
                             const DebugOptions* debug_options,
                             State<GemmConfig> state, StridedMemrefView lhs,
                             StridedMemrefView rhs, StridedMemrefView out,
                             int64_t algorithm, double alpha_real,
                             double alpha_imag, double beta,
                             DotDimensionNumbers dot_dims) {
  se::DeviceMemoryBase lhs_data = GetDeviceAddress(lhs);
  se::DeviceMemoryBase rhs_data = GetDeviceAddress(rhs);
  se::DeviceMemoryBase output_data = GetDeviceAddress(out);

  VLOG(3) << "Running GEMM";
  se::Stream* stream = run_options->stream();

  // Get the gemm config from the state.
  absl::StatusOr<GemmConfig*> config = state.GetOrCreate([&] {
    return ToAbsl(GetGemmConfig(lhs, rhs, out, algorithm, alpha_real,
                                alpha_imag, beta, dot_dims.lhs_batch,
                                dot_dims.lhs_contract, dot_dims.rhs_batch,
                                dot_dims.rhs_contract));
  });
  if (!config.ok()) return config.status();

  Status executed = RunGemm(**config, lhs_data, rhs_data, output_data, stream);

  if (!executed.ok()) return ToAbslStatus(executed);

  return absl::OkStatus();
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    Gemm, FunctionWrapper<GemmImpl>(), checks,
    CustomCall::Bind("xla.gpu.gemm")
        .UserData<const ServiceExecutableRunOptions*>()
        .UserData<const DebugOptions*>()
        .State<GemmConfig>("uid")
        .Arg<StridedMemrefView>()  // lhs
        .Arg<StridedMemrefView>()  // rhs
        .Arg<StridedMemrefView>()  // out
        .Attr<int64_t>("algorithm")
        .Attr<double>("alpha_real")
        .Attr<double>("alpha_imag")
        .Attr<double>("beta")
        .Attr<DotDimensionNumbers>("dot_dims"));

void RegisterGemmCustomCalls(runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.gpu.gemm", Gemm);
}

}  // namespace gpu
}  // namespace xla
