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

#include "tensorflow/compiler/xla/service/gpu/jitrt_custom_calls.h"

#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/runtime/arguments.h"
#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/custom_call_registry.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/runtime/jit_executable.h"
#include "tensorflow/compiler/xla/runtime/type_id.h"
#include "tensorflow/compiler/xla/runtime/types.h"
#include "tensorflow/compiler/xla/service/custom_call_status_internal.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/service/gpu/fft_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_asm_opts_util.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_conv_runner.h"
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_gather_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_to_all_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_collective_permute_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_collective_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/cholesky.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/collectives.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/conv.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/cublas_lt_matmul.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/fft.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/gemm.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/io_feed.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/kernel_launch.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/memcpy.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/memset.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/support.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/triangular_solve.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/tsl/platform/human_readable_json.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/compiler/xla/service/gpu/cholesky_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/triangular_solve_thunk.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_stream.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#if GOOGLE_CUDA
#include "tensorflow/compiler/xla/service/gpu/runtime/graph_launch.h"
#endif  // GOOGLE_CUDA

namespace xla {
namespace gpu {

using Eigen::bfloat16;
using Eigen::half;

using llvm::ArrayRef;

using mlir::failure;
using mlir::FailureOr;
using mlir::LogicalResult;
using mlir::StringRef;
using mlir::succeeded;

using ::xla::runtime::AggregateAttrDef;
using ::xla::runtime::AggregateAttrEncoding;
using ::xla::runtime::CustomCall;
using ::xla::runtime::CustomCallAttrEncodingSet;
using ::xla::runtime::Executable;
using ::xla::runtime::FlatMemrefView;
using ::xla::runtime::StridedMemrefView;
using ::xla::runtime::Tagged;
using ::xla::runtime::TypeIDNameRegistry;

namespace se = ::stream_executor;

// Disable all CustomCall checks in optimized build.
static constexpr CustomCall::RuntimeChecks RuntimeChecks() {
#if defined(NDEBUG)
  return CustomCall::RuntimeChecks::kNone;
#else
  return CustomCall::RuntimeChecks::kDefault;
#endif
}

// -------------------------------------------------------------------------- //

// Add custom call arguments and attributes encoding for custom HLO enums and
// structs, so that we can pass them to custom calls.
void PopulateLmhloToXlaAttrEncoding(CustomCallAttrEncodingSet& encoding) {
  PopulateConvAttrEncoding(encoding);
  PopulateFftAttrEncoding(encoding);
  PopulateDotDimsAttrEncoding(encoding);

#if GOOGLE_CUDA
  PopulateCublasLtMatmulAttrEncoding(encoding);
#endif  // GOOGLE_CUDA
}

// -------------------------------------------------------------------------- //

template <typename MemrefArg>
static se::DeviceMemoryBase GetDeviceAddress(MemrefArg& memref) {
  uint64_t size = primitive_util::ByteWidth(memref.dtype);
  for (auto dim : memref.sizes) size *= dim;
  return se::DeviceMemoryBase(memref.data, size);
}

static se::DeviceMemoryBase GetDeviceAddress(runtime::FlatMemrefView& memref) {
  return se::DeviceMemoryBase(memref.data, memref.size_in_bytes);
}


// -------------------------------------------------------------------------- //


// -------------------------------------------------------------------------- //
// Implements JitRt custom call that forward to the Xla Custom Call handler.
//
// Longer term all Xla custom calls probably should be directly implemented as
// JitRt custom calls. However for smooth migration from Thunks to JitRt we have
// to seamlessly support all current XLA users.

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
namespace {
struct XlaCustomCall {
  using Stream = se::gpu::GpuStreamHandle;

  absl::Status operator()(const ServiceExecutableRunOptions* run_options,
                          const DebugOptions* debug_options,
                          CustomCall::RemainingArgs args,
                          StringRef call_target_name, int32_t api_version,
                          StringRef backend_config) const;
  static XlaCustomCall Handler() { return XlaCustomCall(); }
};
}  // namespace

absl::Status XlaCustomCall::operator()(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options, CustomCall::RemainingArgs args,
    StringRef call_target_name, int32_t api_version,
    StringRef backend_config) const {
  // Pattern match custom call to a few special cases, otherwise find the custom
  // call handler regustered with the runtime.
  if (call_target_name == kTriangularSolveCallTarget)
    return TriangularSolve::run(run_options, debug_options, args,
                                backend_config);

  // Find the Xla custom call handler.
  auto& platform_name = run_options->stream()->parent()->platform()->Name();
  void* call_target = CustomCallTargetRegistry::Global()->Lookup(
      call_target_name.str(), platform_name);
  if (!call_target) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Cannot find the Xla custom call handler ", call_target_name.str()));
  }

  // Prepare pointers to buffers to pass to the Xla custom call handler.
  llvm::SmallVector<void*> buffers;
  for (unsigned i = 0; i < args.size(); ++i) {
    if (auto memref = args.get<FlatMemrefView>(i); succeeded(memref)) {
      buffers.push_back(memref->data);
      continue;
    }

    if (auto strided = args.get<StridedMemrefView>(i); succeeded(strided)) {
      int64_t size_in_bytes = primitive_util::ByteWidth(strided->dtype);
      for (int64_t size : strided->sizes) size_in_bytes *= size;
      buffers.push_back(strided->data);
      continue;
    }

    // TODO(ezhulenev): Add dialect and type to model Xla custom call holes,
    // today we rely on the fact that custom calls do not support scalar
    // arguments and we can disambiguate holes from real arguments.
    if (auto hole = args.get<int64_t>(i); succeeded(hole)) {
      buffers.push_back(nullptr);
      continue;
    }

    return absl::InvalidArgumentError(
        "Failed to get arguments as (strided) memref view");
  }

  // Original custom call API version that doesn't support returning status.
  if (api_version == CustomCallApiVersion::API_VERSION_ORIGINAL) {
    using XlaCustomCallType = void (*)(Stream, void**, const char*, size_t);
    auto xla_call_target = reinterpret_cast<XlaCustomCallType>(call_target);

    xla_call_target(se::gpu::AsGpuStreamValue(run_options->stream()),
                    buffers.data(), backend_config.data(),
                    backend_config.size());

    return absl::OkStatus();
  }

  // Xla Custom call API returning status.
  if (api_version == CustomCallApiVersion::API_VERSION_STATUS_RETURNING ||
      api_version ==
          CustomCallApiVersion::API_VERSION_STATUS_RETURNING_UNIFIED) {
    using XlaCustomCallType =
        void (*)(Stream, void**, const char*, size_t, XlaCustomCallStatus*);
    auto xla_call_target = reinterpret_cast<XlaCustomCallType>(call_target);

    XlaCustomCallStatus custom_call_status;
    xla_call_target(se::gpu::AsGpuStreamValue(run_options->stream()),
                    buffers.data(), backend_config.data(),
                    backend_config.size(), &custom_call_status);

    if (auto message = CustomCallStatusGetMessage(&custom_call_status)) {
      return absl::InternalError(message.value());
    } else {
      return absl::OkStatus();
    }
  }

  return absl::InvalidArgumentError("Incorrect custom call API version");
}

static bool CustomCall(runtime::ExecutionContext* ctx, void** args,
                       void** attrs, void** rets) {
  static auto* handler = CustomCall::Bind("xla.gpu.memcpy")
                             .UserData<const ServiceExecutableRunOptions*>()
                             .UserData<const DebugOptions*>()
                             .Arg<CustomCall::RemainingArgs>()  // args
                             .Attr<std::string_view>("call_target_name")
                             .Attr<int32_t>("api_version")
                             .Attr<std::string_view>("backend_config")
                             .To<RuntimeChecks()>(XlaCustomCall::Handler())
                             .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// -------------------------------------------------------------------------- //


// Populate mapping from XLA (SE) enums/structs type id to symbol names.
void PopulateXlaGpuTypeIdNames(TypeIDNameRegistry& registry) {
#if GOOGLE_CUDA
  registry.Register<Tagged<se::cuda::BlasLt::Epilogue>>(
      "__type_id_se_cublas_lt_epilogue");
#endif  // GOOGLE_CUDA

  registry.Register<Tagged<se::dnn::ActivationMode>>(
      "__type_id_se_dnn_activation");
  registry.Register<Tagged<se::fft::Type>>("__type_id_se_fft_type");

  registry.Register<Tagged<DotDimensionNumbers>>(
      "__type_id_dot_dimension_numbers");
  registry.Register<Tagged<ConvDimensionNumbers>>(
      "__type_id_conv_dimension_numbers");
  registry.Register<Tagged<ConvBackendConfig>>("__type_id_conv_backend_config");

  RegisterTracingTypeIdNames(registry);
}

void PopulateXlaGpuCustomCalls(runtime::DirectCustomCallRegistry& registry) {
  RegisterKernelLaunchCustomCalls(registry);
  RegisterTracingCustomCalls(registry);

#if GOOGLE_CUDA
  // Graph launch kernels depend on Cuda Graph API.
  RegisterGraphLaunchCustomCalls(registry);
#endif  // GOOGLE_CUDA

  RegisterFftCustomCalls(registry);
  RegisterCholeskyCustomCalls(registry);
  RegisterCollectiveCustomCalls(registry);
  RegisterGemmCustomCalls(registry);

#if GOOGLE_CUDA
  RegisterMatmulCustomCalls(registry);
#endif  // GOOGLE_CUDA

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  registry.Register("xla.gpu.custom_call", &xla::gpu::CustomCall);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

  RegisterConvCustomCalls(registry);
  RegisterMemcpyCustomCalls(registry);
  RegisterIoFeedCustomCalls(registry);
  RegisterMemsetCustomCalls(registry);
}

}  // namespace gpu
}  // namespace xla
