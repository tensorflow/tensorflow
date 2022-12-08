/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.1
==============================================================================*/

#include "tensorflow/compiler/xla/service/gpu/runtime/cublas_lt_matmul.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "tensorflow/compiler/xla/mlir/runtime/transforms/custom_call_encoding.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/runtime/logical_result.h"
#include "tensorflow/compiler/xla/runtime/state.h"
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/support.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/stream_executor/scratch_allocator.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/tsl/platform/status.h"

#if GOOGLE_CUDA
#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_blas_lt.h"
#endif  // GOOGLE_CUDA

namespace xla {
namespace gpu {

namespace se = ::stream_executor;

using llvm::ArrayRef;

using xla::runtime::CustomCall;
using xla::runtime::Executable;
using xla::runtime::State;

namespace lmhlo_gpu = ::mlir::lmhlo_gpu;

#if GOOGLE_CUDA

namespace {
struct CublasLtMatmul {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  absl::Status operator()(
      const ServiceExecutableRunOptions* run_options,
      const DebugOptions* debug_options, State<GemmConfig> gemm_config,
      State<cublas_lt::MatmulPlan> matmul_plan, runtime::StridedMemrefView a,
      runtime::StridedMemrefView b, runtime::StridedMemrefView c,
      runtime::StridedMemrefView d,
      std::optional<runtime::StridedMemrefView> bias,
      std::optional<runtime::StridedMemrefView> aux, int64_t algorithm,
      double alpha_real, double alpha_imag, double beta,
      DotDimensionNumbers dot_dims, se::cuda::BlasLt::Epilogue epilogue,
      ArrayRef<int32_t> precision) const;

  static CublasLtMatmul Handler() { return CublasLtMatmul(); }
};
}  // namespace

absl::Status CublasLtMatmul::operator()(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options, State<GemmConfig> gemm_config,
    State<cublas_lt::MatmulPlan> matmul_plan, runtime::StridedMemrefView a,
    runtime::StridedMemrefView b, runtime::StridedMemrefView c,
    runtime::StridedMemrefView d,
    std::optional<runtime::StridedMemrefView> bias,
    std::optional<runtime::StridedMemrefView> aux, int64_t algorithm,
    double alpha_real, double alpha_imag, double beta,
    DotDimensionNumbers dot_dims, se::cuda::BlasLt::Epilogue epilogue,
    ArrayRef<int32_t> precision) const {
  VLOG(3) << "Running CublasLtMatmul";
  se::Stream* stream = run_options->stream();

  // Find the gemm config for this instance of matmul.
  absl::StatusOr<GemmConfig*> config = gemm_config.GetOrCreate([&] {
    return ToAbsl(GetGemmConfig(a, b, c, algorithm, alpha_real, alpha_imag,
                                beta, dot_dims.lhs_batch, dot_dims.lhs_contract,
                                dot_dims.rhs_batch, dot_dims.rhs_contract));
  });
  if (!config.ok()) return config.status();

  // Get the matmul plan for this instance of matmul.
  absl::StatusOr<cublas_lt::MatmulPlan*> plan = matmul_plan.GetOrCreate(
      [&] { return ToAbsl(cublas_lt::MatmulPlan::From(**config, epilogue)); });
  if (!plan.ok()) return plan.status();

  auto algos = (*plan)->GetAlgorithms(stream);
  if (!algos.ok()) return ToAbslStatus(algos.status());

  se::DeviceMemoryBase a_data = GetDeviceAddress(a);
  se::DeviceMemoryBase b_data = GetDeviceAddress(b);
  se::DeviceMemoryBase c_data = GetDeviceAddress(c);
  se::DeviceMemoryBase d_data = GetDeviceAddress(d);
  se::DeviceMemoryBase bias_data;
  if (bias.has_value()) bias_data = GetDeviceAddress(*bias);
  se::DeviceMemoryBase aux_data;
  if (aux.has_value()) aux_data = GetDeviceAddress(*aux);

  se::OwningScratchAllocator<> scratch_allocator(
      stream->parent()->device_ordinal(), stream->parent()->GetAllocator());

  return ToAbslStatus((*plan)->ExecuteOnStream(
      stream, a_data, b_data, c_data, d_data, bias_data, aux_data,
      (*algos)[algorithm], scratch_allocator));
}

namespace {

// Adds custom call bindings for matmul operations.
template <typename... Ts>
auto BindMatmulAttributes(runtime::CustomCallBinding<Ts...> binding) {
  return std::move(binding)
      .template Attr<int64_t>("algorithm")
      .template Attr<double>("alpha_real")
      .template Attr<double>("alpha_imag")
      .template Attr<double>("beta")
      .template Attr<DotDimensionNumbers>("dot_dims")
      .template Attr<se::cuda::BlasLt::Epilogue>("epilogue")
      .template Attr<ArrayRef<int32_t>>("precision");
}

auto CublasLtMatmulCall(const char* name) {
  return CustomCall::Bind(name)
      .UserData<const ServiceExecutableRunOptions*>()
      .UserData<const DebugOptions*>()
      .State<GemmConfig>("uid")
      .State<cublas_lt::MatmulPlan>("uid")
      .Arg<runtime::StridedMemrefView>()   // a
      .Arg<runtime::StridedMemrefView>()   // b
      .Arg<runtime::StridedMemrefView>()   // c
      .Arg<runtime::StridedMemrefView>();  // d
}

bool CublasLtMatmul(runtime::ExecutionContext* ctx, void** args, void** attrs,
                    void** rets) {
  static auto* handler =
      BindMatmulAttributes(
          CublasLtMatmulCall("xla.gpu.cublas.lt.matmul")
              .Value(std::optional<runtime::StridedMemrefView>())  // bias
              .Value(std::optional<runtime::StridedMemrefView>())  // aux
          )
          .To<checks>(CublasLtMatmul::Handler())
          .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

bool CublasLtMatmulBias(runtime::ExecutionContext* ctx, void** args,
                        void** attrs, void** rets) {
  static auto* handler =
      BindMatmulAttributes(
          CublasLtMatmulCall("xla.gpu.cublas.lt.matmul.bias")
              .Arg<runtime::StridedMemrefView>()                   // bias
              .Value(std::optional<runtime::StridedMemrefView>())  // aux
          )
          .To<checks>(CublasLtMatmul::Handler())
          .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

bool CublasLtMatmulAux(runtime::ExecutionContext* ctx, void** args,
                       void** attrs, void** rets) {
  static auto* handler =
      BindMatmulAttributes(
          CublasLtMatmulCall("xla.gpu.cublas.lt.matmul.aux")
              .Value(std::optional<runtime::StridedMemrefView>())  // bias
              .Arg<runtime::StridedMemrefView>()                   // aux
          )
          .To<checks>(CublasLtMatmul::Handler())
          .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

bool CublasLtMatmulBiasAux(runtime::ExecutionContext* ctx, void** args,
                           void** attrs, void** rets) {
  static auto* handler =
      BindMatmulAttributes(
          CublasLtMatmulCall("xla.gpu.cublas.lt.matmul.bias.aux")
              .Arg<runtime::StridedMemrefView>()  // bias
              .Arg<runtime::StridedMemrefView>()  // aux
          )
          .To<checks>(CublasLtMatmul::Handler())
          .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

}  // namespace

void PopulateCublasLtMatmulAttrEncoding(
    runtime::CustomCallAttrEncodingSet& encoding) {
  encoding.Add<runtime::EnumAttrEncoding<lmhlo_gpu::CublasLtMatmulEpilogueAttr,
                                         lmhlo_gpu::CublasLtMatmulEpilogue,
                                         se::cuda::BlasLt::Epilogue>>(
      [](lmhlo_gpu::CublasLtMatmulEpilogue value)
          -> se::cuda::BlasLt::Epilogue {
        return cublas_lt::AsBlasLtEpilogue(value).value();
      });
}

void RegisterMatmulCustomCalls(runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.gpu.cublas.lt.matmul", &xla::gpu::CublasLtMatmul);
  registry.Register("xla.gpu.cublas.lt.matmul.bias", CublasLtMatmulBias);
  registry.Register("xla.gpu.cublas.lt.matmul.aux", CublasLtMatmulAux);
  registry.Register("xla.gpu.cublas.lt.matmul.bias.aux", CublasLtMatmulBiasAux);
}

#endif  // GOOGLE_CUDA

}  // namespace gpu
}  // namespace xla
