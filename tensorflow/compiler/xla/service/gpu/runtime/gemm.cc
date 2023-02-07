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
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/stream_executor/blas.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory.h"
#include "tensorflow/compiler/xla/xla.pb.h"

#if GOOGLE_CUDA
#include "tensorflow/compiler/xla/service/gpu/gemm_algorithm_picker.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/redzone_allocator.h"
#endif

namespace xla {
namespace gpu {

using xla::runtime::CustomCall;
using xla::runtime::State;
using xla::runtime::StridedMemrefView;

#if GOOGLE_CUDA
// TODO(anlunx): Runtime autotuning should be protected by an exclusive lock to
// achieve precision. Right now it is protected by a reader lock acquired by
// GpuExecutable::ExecuteAsyncOnStreamImpl, so it may run cuncurrently with
// another runtime autotuning.
Status DoRuntimeAutotuning(se::Stream* stream, GemmConfig& config,
                           se::DeviceMemoryBase lhs, se::DeviceMemoryBase rhs,
                           se::DeviceMemoryBase out, const Shape& output_shape,
                           double beta, const DebugOptions* debug_options) {
  VLOG(3) << "Running GEMM runtime autotuning";
  std::vector<se::blas::AlgorithmType> algorithms;
  stream->parent()->GetBlasGemmAlgorithms(stream, &algorithms);

  // Set autotune_level to 3 to disable correctness checking, which avoids
  // memory allocation during runtime.
  AutotuneConfig autotune_config{
      /*autotune_level=*/3,
      /*should_crash_on_check_failure=*/true,
  };

  // RedzoneAllocator will have size 0 for this autotune_level.
  se::RedzoneAllocator buffer_allocator =
      CreateRedzoneAllocator(stream, stream->parent()->GetAllocator(),
                             *debug_options, autotune_config);

  TF_ASSIGN_OR_RETURN(
      auto best_algorithm_idx,
      GetBestBlasAlgorithm(
          stream, buffer_allocator, /*gemm_str=*/std::nullopt, autotune_config,
          lhs, rhs, out, algorithms, output_shape, HloModuleConfig(), beta,
          [&](const se::blas::AlgorithmType& algorithm)
              -> StatusOr<se::blas::ProfileResult> {
            se::blas::ProfileResult profile_result;
            // We expect GemmWithAlgorithm to fail sometimes -- in fact, it will
            // fail for all algorithms if we're targeting < sm_50.  But because
            // we pass a non-null ProfileResult, DoGemmWithAlgorithm should
            // always return true, and the actual success-ness is returned in
            // ProfileResult::is_valid.
            TF_RETURN_IF_ERROR(RunGemm(config, lhs, rhs, out, stream, algorithm,
                                       &profile_result));
            return std::move(profile_result);
          }));

  if (best_algorithm_idx.has_value()) {
    config.algorithm = algorithms[best_algorithm_idx.value()];
    return OkStatus();
  } else {
    return InternalError("Runtime autotuning failed to select an algorithm");
  }
}
#endif

static absl::Status GemmImpl(const ServiceExecutableRunOptions* run_options,
                             const DebugOptions* debug_options,
                             State<GemmConfig> state, StridedMemrefView lhs,
                             StridedMemrefView rhs, StridedMemrefView out,
                             int64_t algorithm, double alpha_real,
                             double alpha_imag, double beta,
                             DotDimensionNumbers dot_dims,
                             absl::Span<const int32_t> precision) {
  se::DeviceMemoryBase lhs_data = GetDeviceAddress(lhs);
  se::DeviceMemoryBase rhs_data = GetDeviceAddress(rhs);
  se::DeviceMemoryBase output_data = GetDeviceAddress(out);

  VLOG(3) << "Running GEMM";
  se::Stream* stream = run_options->stream();
  Shape output_shape = ToShape(out);

  // Get the gemm config from the state.
  absl::StatusOr<GemmConfig*> config = state.GetOrCreate([&] {
    StatusOr<GemmConfig> gemm_config =
        GetGemmConfig(lhs, rhs, out, algorithm, alpha_real, alpha_imag, beta,
                      dot_dims.lhs_batch, dot_dims.lhs_contract,
                      dot_dims.rhs_batch, dot_dims.rhs_contract,
                      precision.empty() ? se::blas::kDefaultComputePrecision
                                        : *absl::c_max_element(precision));
#if GOOGLE_CUDA
    if (!gemm_config.ok()) return ToAbsl(gemm_config);
    if (gemm_config->algorithm == stream_executor::blas::kRuntimeAutotuning) {
      auto status =
          DoRuntimeAutotuning(stream, *gemm_config, lhs_data, rhs_data,
                              output_data, output_shape, beta, debug_options);
      if (!status.ok())
        return absl::StatusOr<GemmConfig>(
            absl::InternalError(status.ToString()));
    }
#endif
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
        .Attr<DotDimensionNumbers>("dot_dims")
        .Attr<absl::Span<const int32_t>>("precision"));

void RegisterGemmCustomCalls(runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.gpu.gemm", Gemm);
}

}  // namespace gpu
}  // namespace xla
