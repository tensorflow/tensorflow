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

#include <optional>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/support.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/stream_executor/blas.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory.h"
#include "tensorflow/compiler/xla/xla.pb.h"

namespace xla {
namespace gpu {

using xla::runtime::CustomCall;
using xla::runtime::State;
using xla::runtime::StridedMemrefView;

// Run autotuning to select the best algorithm and save the algorithm to
// GemmConfig.
//
// Difference between this function and GemmAlgorithmPicker:
// This function doesn't check correctness of the algorithms and doesn't
// reinitialize output buffers everytime gemm runs. The result is always
// nondeterministic.
// TODO(anlunx): Reuse GemmAlgorithmPicker::DoGemmAutotune.
void DoRuntimeAutotuning(se::Stream* stream, GemmConfig& config,
                         se::DeviceMemoryBase lhs, se::DeviceMemoryBase rhs,
                         se::DeviceMemoryBase out) {
  if (config.algorithm != stream_executor::blas::kRuntimeAutotuning) return;

  std::vector<se::blas::AlgorithmType> algorithms;
  stream->parent()->GetBlasGemmAlgorithms(stream, &algorithms);
  std::vector<se::blas::ProfileResult> profile_results;
  for (const se::blas::AlgorithmType algorithm : algorithms) {
    se::blas::ProfileResult profile_result;
    Status autotune_run =
        RunGemm(config, lhs, rhs, out, stream, algorithm, &profile_result);
    if (!autotune_run.ok()) continue;

    if (profile_result.is_valid()) {
      profile_results.push_back(std::move(profile_result));
    }
  }

  if (profile_results.empty()) {
    config.algorithm = stream_executor::blas::kDefaultAlgorithm;
    return;
  }

  auto selected_result = absl::c_min_element(
      profile_results, [](const se::blas::ProfileResult& lhs,
                          const se::blas::ProfileResult& rhs) {
        return lhs.elapsed_time_in_ms() < rhs.elapsed_time_in_ms();
      });

  config.algorithm = selected_result->algorithm();
}

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
    StatusOr<GemmConfig> gemm_config =
        GetGemmConfig(lhs, rhs, out, algorithm, alpha_real, alpha_imag, beta,
                      dot_dims.lhs_batch, dot_dims.lhs_contract,
                      dot_dims.rhs_batch, dot_dims.rhs_contract);
    if (!gemm_config.ok()) return ToAbsl(gemm_config);
    DoRuntimeAutotuning(stream, *gemm_config, lhs_data, rhs_data, output_data);
    return ToAbsl(gemm_config);
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
