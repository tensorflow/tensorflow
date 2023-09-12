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

#include "xla/service/gpu/runtime/gemm.h"

#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "xla/runtime/custom_call.h"
#include "xla/runtime/executable.h"
#include "xla/service/gpu/gpu_asm_opts_util.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/non_atomically_upgradeable_rw_lock.h"
#include "xla/service/gpu/runtime/support.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/status.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/xla.pb.h"

#if GOOGLE_CUDA
#include "xla/service/gpu/gemm_algorithm_picker.h"
#include "xla/stream_executor/gpu/redzone_allocator.h"
#endif

namespace xla {
namespace gpu {

using xla::runtime::CustomCall;
using xla::runtime::State;
using xla::runtime::StridedMemrefView;

#if GOOGLE_CUDA

Status DoRuntimeAutotuning(se::Stream* stream, GemmConfig& config,
                           se::DeviceMemoryBase lhs, se::DeviceMemoryBase rhs,
                           se::DeviceMemoryBase out, const Shape& output_shape,
                           double beta, const DebugOptions* debug_options,
                           NonAtomicallyUpgradeableRWLock* gpu_lock) {
  VLOG(3) << "Running GEMM runtime autotuning";
  std::vector<se::blas::AlgorithmType> algorithms;
  stream->parent()->GetBlasGemmAlgorithms(stream, &algorithms);
  const bool deterministic_ops = debug_options->xla_gpu_deterministic_ops();

  AutotuneConfig autotune_config{
      DeviceConfig{stream->parent(), stream->parent()->GetAllocator()},
      *debug_options};

  // TODO(jlebar): We should not use stream->parent()->GetAllocator() here;
  // that's the global CUDA allocator.  There may not be any free space in
  // there, because TF usually gobbles it all up for its own BFCAllocator.  We
  // should use the allocator the user passed when running the XLA program.
  se::RedzoneAllocator buffer_allocator(
      stream, stream->parent()->GetAllocator(),
      PtxOptsFromDebugOptions(*debug_options),
      /*memory_limit=*/std::numeric_limits<int64_t>::max(),
      /*redzone_size=*/autotune_config.should_check_correctness()
          ? debug_options->xla_gpu_redzone_padding_bytes()
          : 0);

  // Upgrade the reader lock for execution to a writer lock to protect runtime
  // autotuning.
  NonAtomicallyUpgradeableRWLock::WriterLock writer_lock =
      gpu_lock->UpgradeToWriterMutexLock();

  TF_ASSIGN_OR_RETURN(
      AutotuneResult best_algorithm,
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
            TF_RETURN_IF_ERROR(RunGemm(config, lhs, rhs, out, deterministic_ops,
                                       stream, algorithm, &profile_result));
            return std::move(profile_result);
          }));

  if (best_algorithm.has_gemm()) {
    config.algorithm = algorithms[best_algorithm.gemm().algorithm()];
    return OkStatus();
  } else {
    return InternalError("Runtime autotuning failed to select an algorithm");
  }
}
#endif

static absl::Status GemmImpl(const ServiceExecutableRunOptions* run_options,
                             const DebugOptions* debug_options,
                             NonAtomicallyUpgradeableRWLock* gpu_lock,
                             State<GemmConfig> state, StridedMemrefView lhs,
                             StridedMemrefView rhs, StridedMemrefView out,
                             int64_t algorithm, double alpha_real,
                             double alpha_imag, double beta,
                             DotDimensionNumbers dot_dims,
                             absl::Span<const int32_t> precision) {
  se::DeviceMemoryBase lhs_data = GetDeviceAddress(lhs);
  se::DeviceMemoryBase rhs_data = GetDeviceAddress(rhs);
  se::DeviceMemoryBase output_data = GetDeviceAddress(out);
  const bool deterministic_ops = debug_options->xla_gpu_deterministic_ops();

  VLOG(3) << "Running GEMM";
  se::Stream* stream = run_options->stream();
  Shape output_shape = ToShape(out);

  // Get the gemm config from the state.
  TF_ASSIGN_OR_RETURN(GemmConfig * gemm_config, state.GetOrCreate([&] {
    StatusOr<GemmConfig> gemm_config =
        GetGemmConfig(lhs, rhs, out, algorithm, alpha_real, alpha_imag, beta,
                      dot_dims.lhs_batch, dot_dims.lhs_contract,
                      dot_dims.rhs_batch, dot_dims.rhs_contract,
                      precision.empty() ? se::blas::kDefaultComputePrecision
                                        : *absl::c_max_element(precision));
    return ToAbsl(gemm_config);
  }));

  // Set the gemm algorithm by runtime autotuning. We do runtime autotuning
  // outside of state.GetOrCreate() because otherwise it would be a potential
  // deadlock.
  if (gemm_config->algorithm == stream_executor::blas::kRuntimeAutotuning) {
#if GOOGLE_CUDA
    auto status = DoRuntimeAutotuning(stream, *gemm_config, lhs_data, rhs_data,
                                      output_data, output_shape, beta,
                                      debug_options, gpu_lock);
    if (!status.ok()) {
      return absl::InternalError(status.ToString());
    }
#else
    return absl::InternalError(
        "Failed to run runtime autotuner because CUDA is not enabled");
#endif
  }

  return RunGemm(*gemm_config, lhs_data, rhs_data, output_data,
                 deterministic_ops, stream);
}

static absl::Status InitCuBLASImpl(
    const ServiceExecutableRunOptions* run_options) {
  // Initialize (with memoization) BlasSupport here because cublasCreate fails
  // during gpu graph capturing.
  se::StreamExecutor* executor = run_options->stream()->parent();
  if (!executor->AsBlas()) {
    return absl::InternalError("Failed to initialize BLAS support");
  }
  return absl::OkStatus();
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    Gemm, FunctionWrapper<GemmImpl>(), checks,
    CustomCall::Bind("xla.gpu.gemm")
        .UserData<const ServiceExecutableRunOptions*>()
        .UserData<const DebugOptions*>()
        .UserData<NonAtomicallyUpgradeableRWLock*>()
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

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    InitCuBLAS, FunctionWrapper<InitCuBLASImpl>(), checks,
    CustomCall::Bind("xla.gpu.init_cublas")
        .UserData<const ServiceExecutableRunOptions*>());

void RegisterGemmCustomCalls(runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.gpu.gemm", Gemm);
  registry.Register("xla.gpu.init_cublas", InitCuBLAS);
}

}  // namespace gpu
}  // namespace xla
