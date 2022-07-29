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
#include <utility>

#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/xla/service/custom_call_status_internal.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/service/gpu/fft_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_asm_opts_util.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_conv_runner.h"
#include "tensorflow/compiler/xla/service/gpu/infeed_manager.h"
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_gather_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_to_all_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_collective_permute_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_collective_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/outfeed_manager.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tfrt_utils.h"
#include "tensorflow/core/platform/human_readable_json.h"
#include "tensorflow/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/stream_executor/gpu/gpu_types.h"
#include "tfrt/jitrt/custom_call.h"  // from @tf_runtime
#include "tfrt/jitrt/jitrt.h"  // from @tf_runtime
#include "tfrt/dtype/dtype.h"  // from @tf_runtime

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/compiler/xla/service/gpu/cholesky_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/triangular_solve_thunk.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

TFRT_DEFINE_EXPLICIT_DENSE_TYPE_ID(tfrt::jitrt::CustomCall,
                                   xla::gpu::JitRtKernelsCache);
TFRT_DEFINE_EXPLICIT_DENSE_TYPE_ID(tfrt::jitrt::CustomCall,
                                   xla::gpu::JitRtGemmConfigCache);
TFRT_DEFINE_EXPLICIT_DENSE_TYPE_ID(tfrt::jitrt::CustomCall,
                                   xla::gpu::JitRtCollectiveSupport);
TFRT_DEFINE_EXPLICIT_DENSE_TYPE_ID(tfrt::jitrt::CustomCall,
                                   xla::gpu::JitRtAsyncCollectiveSupport);
TFRT_DEFINE_EXPLICIT_DENSE_TYPE_ID(tfrt::jitrt::CustomCall,
                                   const xla::ServiceExecutableRunOptions);
TFRT_DEFINE_EXPLICIT_DENSE_TYPE_ID(tfrt::jitrt::CustomCall,
                                   const xla::DebugOptions);

namespace xla {
namespace gpu {

using llvm::ArrayRef;
using llvm::Error;
using llvm::Optional;

using mlir::failure;
using mlir::FailureOr;
using mlir::LogicalResult;
using mlir::StringRef;
using mlir::succeeded;
using mlir::success;

using tfrt::MakeStringError;
using tfrt::jitrt::CustomCall;
using tfrt::jitrt::DirectCustomCallLibrary;
using tfrt::jitrt::Executable;

namespace se = ::stream_executor;
namespace jitrt = ::tfrt::jitrt;
namespace lmhlo_gpu = ::mlir::lmhlo_gpu;
namespace mhlo = ::mlir::mhlo;
namespace runtime = ::tfrt::jitrt::runtime;

// Disable all CustomCall checks in optimized build.
static constexpr CustomCall::RuntimeChecks RuntimeChecks() {
#if defined(NDEBUG)
  return CustomCall::RuntimeChecks::kNone;
#else
  return CustomCall::RuntimeChecks::kDefault;
#endif
}

// -------------------------------------------------------------------------- //

void PopulateLmhloToXlaAttrEncoding(
    jitrt::CustomCallAttrEncodingSet& encoding) {
  encoding.Add<
      jitrt::EnumAttrEncoding<lmhlo_gpu::ActivationAttr, lmhlo_gpu::Activation,
                              se::dnn::ActivationMode>>(
      [](lmhlo_gpu::Activation value) -> se::dnn::ActivationMode {
        return ConvertConvActivationMode(value).value();
      });

  encoding.Add<
      jitrt::EnumAttrEncoding<mhlo::FftTypeAttr, mhlo::FftType, se::fft::Type>>(
      [](mhlo::FftType value) -> se::fft::Type {
        switch (value) {
          case mhlo::FftType::FFT:
            return se::fft::Type::kC2CForward;
          case mhlo::FftType::IFFT:
            return se::fft::Type::kC2CInverse;
          case mhlo::FftType::RFFT:
            return se::fft::Type::kR2C;
          case mhlo::FftType::IRFFT:
            return se::fft::Type::kC2R;
          default:
            return se::fft::Type::kInvalid;
        }
      });

  using DotDimsAttr = mhlo::DotDimensionNumbersAttr;
  encoding.Add<jitrt::AggregateAttrEncoding<DotDimsAttr, DotDimensionNumbers>>(
      encoding,
      jitrt::AggregateAttrDef<DotDimsAttr>()
          .Add("lhs_batch", &DotDimsAttr::getLhsBatchingDimensions)
          .Add("lhs_contract", &DotDimsAttr::getLhsContractingDimensions)
          .Add("rhs_batch", &DotDimsAttr::getRhsBatchingDimensions)
          .Add("rhs_contract", &DotDimsAttr::getRhsContractingDimensions));

  using ConvDimsAttr = mhlo::ConvDimensionNumbersAttr;
  encoding.Add<
      jitrt::AggregateAttrEncoding<ConvDimsAttr, ConvDimensionNumbers>>(
      encoding,
      jitrt::AggregateAttrDef<ConvDimsAttr>()
          .Add("input_batch_dim", &ConvDimsAttr::getInputBatchDimension)
          .Add("input_feature_dim", &ConvDimsAttr::getInputFeatureDimension)
          .Add("input_spatial_dims", &ConvDimsAttr::getInputSpatialDimensions)
          .Add("kernel_in_feature_dim",
               &ConvDimsAttr::getKernelInputFeatureDimension)
          .Add("kernel_out_feature_dim",
               &ConvDimsAttr::getKernelOutputFeatureDimension)
          .Add("kernel_spatial_dims", &ConvDimsAttr::getKernelSpatialDimensions)
          .Add("output_batch_dim", &ConvDimsAttr::getOutputBatchDimension)
          .Add("output_feature_dim", &ConvDimsAttr::getOutputFeatureDimension)
          .Add("output_spatial_dims",
               &ConvDimsAttr::getOutputSpatialDimensions));

  using ConvConfigAttr = lmhlo_gpu::ConvolutionBackendConfigAttr;
  encoding.Add<jitrt::AggregateAttrEncoding<ConvConfigAttr, ConvBackendConfig>>(
      encoding,
      jitrt::AggregateAttrDef<ConvConfigAttr>()
          .Add("algorithm", &ConvConfigAttr::getAlgorithm)
          .Add("tensor_ops_enabled", &ConvConfigAttr::getTensorOpsEnabled)
          .Add("is_cudnn_frontend", &ConvConfigAttr::getIsCudnnFrontend)
          .Add("knob_ids", &ConvConfigAttr::getKnobIds)
          .Add("knob_values", &ConvConfigAttr::getKnobValues)
          .Add("operand_0_layout", &ConvConfigAttr::getOperand_0Layout)
          .Add("operand_1_layout", &ConvConfigAttr::getOperand_1Layout)
          .Add("result_layout", &ConvConfigAttr::getResultLayout)
          .Add("workspace_size", &ConvConfigAttr::getWorkspaceSize));
}

// -------------------------------------------------------------------------- //

se::KernelBase* JitRtKernelsCache::Get(se::StreamExecutor* executor,
                                       const char* data, StringRef name) {
  Key key(executor, data, name);

  absl::MutexLock lock(&mutex_);
  auto it = kernels_cache_.find(key);
  if (it != kernels_cache_.end()) return it->second.get();

  return nullptr;
}

se::KernelBase* JitRtKernelsCache::Set(se::StreamExecutor* executor,
                                       const char* data, StringRef name,
                                       std::unique_ptr<se::KernelBase> kernel) {
  Key key(executor, data, name);

  absl::MutexLock lock(&mutex_);
  auto it = kernels_cache_.find(key);
  if (it != kernels_cache_.end()) return it->second.get();

  auto emplaced = kernels_cache_.try_emplace(key, std::move(kernel));
  return emplaced.first->second.get();
}

template <typename MemrefArg>
static se::DeviceMemoryBase GetDeviceAddress(MemrefArg& memref) {
  uint64_t size = tfrt::GetHostSize(memref.dtype);
  for (auto dim : memref.sizes) size *= dim;
  return se::DeviceMemoryBase(memref.data, size);
}

static se::DeviceMemoryBase GetDeviceAddress(jitrt::FlatMemrefView& memref) {
  return se::DeviceMemoryBase(memref.data, memref.size_in_bytes);
}

// -------------------------------------------------------------------------- //

const GemmConfig* JitRtGemmConfigCache::Get(int64_t uid) {
  absl::MutexLock lock(&mutex_);
  auto it = configs_.find(uid);
  if (it != configs_.end()) return &it->second;
  return nullptr;
}

const GemmConfig* JitRtGemmConfigCache::Set(int64_t uid, GemmConfig config) {
  absl::MutexLock lock(&mutex_);
  auto it = configs_.find(uid);
  if (it != configs_.end()) return &it->second;

  auto emplaced = configs_.try_emplace(uid, std::move(config));
  return &emplaced.first->second;
}

// -------------------------------------------------------------------------- //

JitRtAsyncCollectiveSupport::JitRtAsyncCollectiveSupport(
    se::Stream* async_comm_stream)
    : async_comm_stream_(async_comm_stream) {}

Status JitRtCollectiveSupport::MaybeBlockAfterFirstRun(int32_t uid,
                                                       int32_t device_ordinal,
                                                       se::Stream* stream) {
  bool block = [&] {
    absl::MutexLock lock(&mutex_);
    return executed_.try_emplace(Key(uid, device_ordinal), true).second;
  }();
  return block ? stream->BlockHostUntilDone() : Status::OK();
}

FailureOr<se::Event> JitRtAsyncCollectiveSupport::PopEvent(
    int32_t uid, int32_t device_ordinal) {
  const int64_t key = EventKey(uid, device_ordinal);

  absl::MutexLock lock(&mutex_);
  auto it = done_events_.find(key);
  if (it == done_events_.end()) return failure();

  se::Event done_event = std::move(it->second);
  done_events_.erase(it);
  return done_event;
}

LogicalResult JitRtAsyncCollectiveSupport::PushEvent(int32_t uid,
                                                     int32_t device_ordinal,
                                                     se::Event done_event) {
  const int64_t key = EventKey(uid, device_ordinal);

  absl::MutexLock lock(&mutex_);
  auto result = done_events_.try_emplace(key, std::move(done_event));
  if (!result.second) return failure();  // done event has not been consumed

  return success();
}

// -------------------------------------------------------------------------- //

static Shape ToShape(const jitrt::StridedMemrefView& memref) {
  PrimitiveType type = TfrtToPrimitiveType(memref.dtype);

  // Recover `minor_to_major` dimensions permutation from strides.
  auto indexed_strides_range =
      llvm::map_range(llvm::enumerate(memref.strides), [](auto pair) {
        return std::pair<int64_t, size_t>{pair.value(), pair.index()};
      });

  auto indexed_strides = llvm::to_vector(indexed_strides_range);
  llvm::stable_sort(indexed_strides);

  llvm::SmallVector<int64_t> minor_to_major;
  minor_to_major.reserve(indexed_strides.size());
  for (auto& pair : indexed_strides) minor_to_major.push_back(pair.second);

  return ShapeUtil::MakeShapeWithLayout(type, memref.sizes, minor_to_major);
}

static StatusOr<GemmConfig> GetGemmConfig(
    const jitrt::StridedMemrefView& lhs, const jitrt::StridedMemrefView& rhs,
    const jitrt::StridedMemrefView& out, int64_t algorithm, double alpha_real,
    double alpha_imag, double beta, ArrayRef<int64_t> lhs_batch,
    ArrayRef<int64_t> lhs_contract, ArrayRef<int64_t> rhs_batch,
    ArrayRef<int64_t> rhs_contract) {
  return GemmConfig::For(ToShape(lhs), lhs_batch, lhs_contract, ToShape(rhs),
                         rhs_batch, rhs_contract, ToShape(out), alpha_real,
                         alpha_imag, beta, algorithm,
                         se::blas::kDefaultComputePrecision);
}

// -------------------------------------------------------------------------- //

#if XLA_ENABLE_XCCL
FailureOr<NcclComm::Lock> GetNcclComm(const NcclExecuteParams& params,
                                      int64_t group_mode, int64_t op_id,
                                      ArrayRef<int64_t> replica_group_offsets,
                                      ArrayRef<int64_t> replica_group_values) {
  // TODO(b/233930690): Pass the attribute below as a nested array.
  // Pass an array of arrays using two vectors; one specifying all the values
  // and another specifying the (ending) offsets of each array in the other
  // vector. Example: [ [10, 20, 30, 40], [50, 60], [70, 80, 90] ] turns into
  // offsets=[4, 6, 9] values=[10, 20, 30, 40, 50, 60, 70, 80, 90].
  std::vector<ReplicaGroup> replica_groups;
  int i = 0;
  for (int64_t replica_group_end : replica_group_offsets) {
    ReplicaGroup replica_group;
    while (i < replica_group_end)
      replica_group.add_replica_ids(replica_group_values[i++]);
    replica_groups.push_back(replica_group);
  }

  auto comm =
      LockNcclComm(params, replica_groups,
                   static_cast<CollectiveOpGroupMode>(group_mode), op_id);
  if (comm.ok()) return std::move(comm.value());
  return failure();
}
#endif  // XLA_ENABLE_XCCL

FailureOr<std::vector<DeviceBufferPair>> GetDeviceBufferPairs(
    CustomCall::RemainingArgs& args) {
  // Add MemRef arguments as buffer arguments.
  const int buffer_pairs = args.size() / 2;
  std::vector<DeviceBufferPair> device_buffers;
  device_buffers.reserve(buffer_pairs);
  for (int i = 0; i < buffer_pairs; ++i) {
    auto source = args.get<jitrt::StridedMemrefView>(i);
    auto destination = args.get<jitrt::StridedMemrefView>(i + buffer_pairs);
    if (failed(source) || failed(destination)) {
      // Unsupported argument type.
      return failure();
    }

    int element_count = 1;
    for (int size : source->sizes) element_count *= size;
    device_buffers.emplace_back(DeviceBufferPair{
        TfrtToPrimitiveType(source->dtype), element_count,
        GetDeviceAddress(*source), GetDeviceAddress(*destination)});
  }
  return device_buffers;
}

// -------------------------------------------------------------------------- //

Error AsError(Status s) { return MakeStringError(s.error_message()); }

template <typename T>
Error AsError(StatusOr<T>& s) {
  assert(!s.ok());
  return AsError(s.status());
}

// -------------------------------------------------------------------------- //

namespace {
struct LaunchFunc {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  Error operator()(const ServiceExecutableRunOptions* run_options,
                   JitRtKernelsCache* kernels_cache, int32_t grid_size_x,
                   int32_t grid_size_y, int32_t grid_size_z,
                   int32_t block_size_x, int32_t block_size_y,
                   int32_t block_size_z, CustomCall::RemainingArgs args,
                   StringRef ptx, StringRef name) const;

  static LaunchFunc Handler() { return LaunchFunc(); }
};
}  // namespace

Error LaunchFunc::operator()(const ServiceExecutableRunOptions* run_options,
                             JitRtKernelsCache* kernels_cache,
                             int32_t grid_size_x, int32_t grid_size_y,
                             int32_t grid_size_z, int32_t block_size_x,
                             int32_t block_size_y, int32_t block_size_z,
                             CustomCall::RemainingArgs args, StringRef ptx,
                             StringRef name) const {
  se::Stream* stream = run_options->stream();
  se::StreamExecutor* executor = stream->parent();

  LaunchDimensions launch_dimensions(
      {grid_size_x, grid_size_y, grid_size_z},
      {block_size_x, block_size_y, block_size_z});

  se::KernelBase* kernel = kernels_cache->Get(executor, ptx.data(), name);

  // If kernel does not exists create it from the ptx.
  if (kernel == nullptr) {
    auto created = CreateKernel(absl::string_view(name.data(), name.size()),
                                args.size(), ptx.data(), {}, executor);
    if (!created.ok()) return AsError(created);

    kernel =
        kernels_cache->Set(executor, ptx.data(), name, std::move(*created));
  }

  VLOG(3) << "Launching " << kernel->name();
  absl::InlinedVector<se::DeviceMemoryBase, 4> buffer_args;
  buffer_args.reserve(args.size());

  // Add MemRef arguments as buffer arguments.
  for (unsigned i = 0; i < args.size(); ++i) {
    // Simple row major memref passed as shapeless buffer.
    auto memref = args.get<jitrt::FlatMemrefView>(i);
    if (succeeded(memref)) {
      buffer_args.emplace_back(GetDeviceAddress(*memref));
      continue;
    }

    // Memref layout must be encoded in the compiled device kernel, so we don't
    // have to pass strides or minor to major dimensions order to the kernel.
    auto strided = args.get<jitrt::StridedMemrefView>(i);
    if (succeeded(strided)) {
      buffer_args.emplace_back(GetDeviceAddress(*strided));
      continue;
    }

    return MakeStringError("Unsupported argumeent type");
  }

  // Execute device kernel on a main stream.
  auto executed =
      ExecuteKernelOnStream(*kernel, buffer_args, launch_dimensions, stream);
  if (!executed.ok()) return AsError(executed);

  return Error::success();
}

static bool LaunchFunc(runtime::KernelContext* ctx, void** args, void** attrs) {
  static auto* handler = CustomCall::Bind("xla.gpu.func.launch")
                             .UserData<const ServiceExecutableRunOptions*>()
                             .UserData<JitRtKernelsCache*>()
                             .Arg<int32_t>()   // grid_size_x
                             .Arg<int32_t>()   // grid_size_y
                             .Arg<int32_t>()   // grid_size_z
                             .Arg<int32_t>()   // block_size_x
                             .Arg<int32_t>()   // block_size_y
                             .Arg<int32_t>()   // block_size_x
                             .RemainingArgs()  // args
                             .Attr<StringRef>("ptx")
                             .Attr<StringRef>("kernel")
                             .To<RuntimeChecks()>(LaunchFunc::Handler())
                             .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs));
}

// -------------------------------------------------------------------------- //

namespace {
struct Gemm {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  Error operator()(const ServiceExecutableRunOptions* run_options,
                   const DebugOptions* debug_options,
                   JitRtGemmConfigCache* configs, jitrt::StridedMemrefView lhs,
                   jitrt::StridedMemrefView rhs, jitrt::StridedMemrefView out,
                   int64_t algorithm, double alpha_real, double alpha_imag,
                   double beta, DotDimensionNumbers dot_dims,
                   int64_t uid) const;

  static Gemm Handler() { return Gemm(); }
};
}  // namespace

Error Gemm::operator()(const ServiceExecutableRunOptions* run_options,
                       const DebugOptions* debug_options,
                       JitRtGemmConfigCache* configs,
                       jitrt::StridedMemrefView lhs,
                       jitrt::StridedMemrefView rhs,
                       jitrt::StridedMemrefView out, int64_t algorithm,
                       double alpha_real, double alpha_imag, double beta,
                       DotDimensionNumbers dot_dims, int64_t uid) const {
  se::DeviceMemoryBase lhs_data = GetDeviceAddress(lhs);
  se::DeviceMemoryBase rhs_data = GetDeviceAddress(rhs);
  se::DeviceMemoryBase output_data = GetDeviceAddress(out);

  VLOG(3) << "Running GEMM";
  se::Stream* stream = run_options->stream();

  // Find the gemm config for this instance of operation based on uid.
  const GemmConfig* config = configs->Get(uid);
  if (config == nullptr) {
    auto cfg = GetGemmConfig(lhs, rhs, out, algorithm, alpha_real, alpha_imag,
                             beta, dot_dims.lhs_batch, dot_dims.lhs_contract,
                             dot_dims.rhs_batch, dot_dims.rhs_contract);
    if (!cfg.ok()) return AsError(cfg);
    config = configs->Set(uid, std::move(*cfg));
  }

  Status executed = [&]() -> Status {
    return RunGemm(*config, lhs_data, rhs_data, output_data, stream);
  }();

  if (!executed.ok()) return AsError(executed);

  return Error::success();
}

static bool Gemm(runtime::KernelContext* ctx, void** args, void** attrs) {
  static auto* handler = CustomCall::Bind("xla.gpu.gemm")
                             .UserData<const ServiceExecutableRunOptions*>()
                             .UserData<const DebugOptions*>()
                             .UserData<JitRtGemmConfigCache*>()
                             .Arg<jitrt::StridedMemrefView>()  // lhs
                             .Arg<jitrt::StridedMemrefView>()  // rhs
                             .Arg<jitrt::StridedMemrefView>()  // out
                             .Attr<int64_t>("algorithm")
                             .Attr<double>("alpha_real")
                             .Attr<double>("alpha_imag")
                             .Attr<double>("beta")
                             .Attr<DotDimensionNumbers>("dot_dims")
                             .Attr<int64_t>("uid")
                             .To<RuntimeChecks()>(Gemm::Handler())
                             .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs));
}

// -------------------------------------------------------------------------- //

// TODO(ezhulenev): We need to find a better way to pass structured attributes
// to JitRt custom calls.

// TODO(ezhulenev): Add caching layer for convolution configs and runners.

namespace {

struct Window {
  ArrayRef<int64_t> window_strides;
  ArrayRef<int64_t> padding;
  ArrayRef<int64_t> lhs_dilation;
  ArrayRef<int64_t> rhs_dilation;
  ArrayRef<int64_t> window_reversal;
};

struct ConvAttrs {
  int64_t feature_group_count;
  double result_scale;
};

struct FusedConvAttrs {
  se::dnn::ActivationMode activation_mode;
};

struct SideInputAttrs {
  double side_input_scale;
};

}  // namespace

static GpuConvDescriptor GetConvDescriptor(
    CudnnConvKind kind,
    // Arguments
    jitrt::StridedMemrefView operand0, jitrt::StridedMemrefView operand1,
    jitrt::StridedMemrefView output, jitrt::FlatMemrefView scratch,
    // Attributes
    ConvDimensionNumbers dims, Window w, ConvBackendConfig b, ConvAttrs attrs,
    // Conv-specific arguments and attributes
    Optional<FusedConvAttrs> fused = llvm::None,
    Optional<SideInputAttrs> side_input = llvm::None) {
  // Build a convolution descriptor from the attributes.
  GpuConvDescriptor descriptor;
  descriptor.kind = kind;

  // Apply backend config layout to the shape.
  auto apply_layout = [](jitrt::StridedMemrefView& memref,
                         ArrayRef<int64_t> minor_to_major) {
    Shape shape = ToShape(memref);
    return ShapeUtil::MakeShapeWithLayout(shape.element_type(),
                                          shape.dimensions(), minor_to_major);
  };

  descriptor.operand0_shape = apply_layout(operand0, b.operand_0_layout);
  descriptor.operand1_shape = apply_layout(operand1, b.operand_1_layout);
  descriptor.result_shape = apply_layout(output, b.result_layout);

  // Set up convolution dimensions numbers.
  ConvolutionDimensionNumbers dns;
  dns.set_input_batch_dimension(dims.input_batch_dim);
  dns.set_input_feature_dimension(dims.input_feature_dim);
  dns.set_kernel_input_feature_dimension(dims.kernel_in_feature_dim);
  dns.set_kernel_output_feature_dimension(dims.kernel_out_feature_dim);
  dns.set_output_batch_dimension(dims.output_batch_dim);
  dns.set_output_feature_dimension(dims.output_feature_dim);
  for (int64_t d : dims.input_spatial_dims) dns.add_input_spatial_dimensions(d);
  for (int64_t d : dims.kernel_spatial_dims)
    dns.add_kernel_spatial_dimensions(d);
  for (int64_t d : dims.output_spatial_dims)
    dns.add_output_spatial_dimensions(d);
  descriptor.dnums = std::move(dns);

  // Put together convolution window config.
  for (auto index : llvm::seq<int>(0, w.window_strides.size())) {
    WindowDimension* dim = descriptor.window.add_dimensions();
    // Window size for a convolution is the same as the kernel size.
    // Kernel size of the convolution is operand1_shape. We need to look at
    // the convolution dimension numbers kernel spatial dimensions to get
    // the window size.
    int kernel_dim = descriptor.dnums.kernel_spatial_dimensions(index);
    dim->set_size(descriptor.operand0_shape.dimensions(kernel_dim));
    dim->set_stride(w.window_strides[index]);
    dim->set_padding_low(w.padding[index]);
    dim->set_padding_high(w.padding[index]);
    dim->set_base_dilation(w.lhs_dilation[index]);
    dim->set_window_dilation(w.rhs_dilation[index]);
    dim->set_window_reversal(w.window_reversal[index]);
  }

  descriptor.scratch_size = scratch.size_in_bytes;
  descriptor.feature_group_count = attrs.feature_group_count;
  descriptor.backend_config.set_conv_result_scale(attrs.result_scale);

  // Set up convolution algorigthm.
  auto* algo = descriptor.backend_config.mutable_algorithm();
  algo->set_algo_id(b.algorithm);
  algo->set_math_type(b.tensor_ops_enabled
                          ? se::dnn::AlgorithmProto::TENSOR_OP_MATH
                          : se::dnn::AlgorithmProto::DEFAULT_MATH);
  algo->set_is_cudnn_frontend(b.is_cudnn_frontend);

  if (b.workspace_size >= 0)
    algo->mutable_workspace_size()->set_value(b.workspace_size);

  for (unsigned i = 0; i < b.knob_ids.size(); ++i) {
    algo->mutable_tuning_knobs()->insert({b.knob_ids[i], b.knob_values[i]});
  }

  // Set attributes specific for fused convolutions.
  if (fused.has_value())
    descriptor.backend_config.set_activation_mode(fused->activation_mode);

  // Set attributes specific for convolutions with side input.
  if (side_input.has_value())
    descriptor.backend_config.set_side_input_scale(
        side_input->side_input_scale);

  return descriptor;
}

namespace {
struct Conv {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  Error operator()(
      const ServiceExecutableRunOptions* run_options,
      const DebugOptions* debug_options, jitrt::StridedMemrefView operand0,
      jitrt::StridedMemrefView operand1, Optional<jitrt::FlatMemrefView> bias,
      Optional<jitrt::StridedMemrefView> side_input,
      jitrt::StridedMemrefView output, jitrt::FlatMemrefView scratch,
      ConvDimensionNumbers conv_dims,
      // Window config
      ArrayRef<int64_t> window_strides, ArrayRef<int64_t> padding,
      ArrayRef<int64_t> lhs_dilation, ArrayRef<int64_t> rhs_dilation,
      ArrayRef<int64_t> window_reversal,
      // Backend config attributes
      ConvBackendConfig backend_config,
      // Remaining attributes
      int64_t feature_group_count, double result_scale,
      // Optional attributes for fused convolutions.
      Optional<se::dnn::ActivationMode> activation_mode = llvm::None,
      Optional<double> side_input_scale = llvm::None) const {
    // Build config for optional attributes.
    Optional<FusedConvAttrs> fused_attrs = llvm::None;
    if (activation_mode.has_value()) fused_attrs = {*activation_mode};

    Optional<SideInputAttrs> side_input_attrs = llvm::None;
    if (side_input_scale.has_value()) side_input_attrs = {*side_input_scale};

    // Prepare a descriptor for the XLA convolution.
    GpuConvDescriptor descriptor = GetConvDescriptor(
        kind, operand0, operand1, output, scratch, conv_dims,
        {window_strides, padding, lhs_dilation, rhs_dilation, window_reversal},
        backend_config, {feature_group_count, result_scale}, fused_attrs,
        side_input_attrs);

    // Convert descriptor to the Conv config.
    StatusOr<GpuConvConfig> config = GetGpuConvConfig(descriptor, "");
    if (!config.ok()) return AsError(config);

    // Prepare buffer arguments.
    std::vector<se::DeviceMemoryBase> buffers = {GetDeviceAddress(operand0),
                                                 GetDeviceAddress(operand1)};
    if (bias.has_value()) buffers.push_back(GetDeviceAddress(*bias));
    if (side_input.has_value())
      buffers.push_back(GetDeviceAddress(*side_input));

    se::DeviceMemoryBase result_buffer = GetDeviceAddress(output);
    se::DeviceMemoryBase scratch_buffer = GetDeviceAddress(scratch);

    RunConvOptions opts;

    // Create a runner for the given config.
    MaybeFusedConvRunner runner(*config);
    opts.runner_cache = &runner;

    // Run the convolution.
    auto st = RunGpuConv(*config, buffers, result_buffer, scratch_buffer,
                         run_options->stream(), opts);
    if (!st.ok() || !run_options->stream()->ok()) {
      return AsError(st);
    }

    return Error::success();
  }

  static Conv Handler(CudnnConvKind kind) { return Conv{kind}; }

  CudnnConvKind kind;
};

}  // namespace

// Adds custom call bindings for convolution operations.
template <typename... Ts>
static auto BindConvAttributes(jitrt::CustomCallBinding<Ts...> binding) {
  return std::move(binding)
      // Convolution dimensions numbers
      .template Attr<ConvDimensionNumbers>("conv_dims")
      // Window config
      .template Attr<ArrayRef<int64_t>>("window_strides")
      .template Attr<ArrayRef<int64_t>>("padding")
      .template Attr<ArrayRef<int64_t>>("lhs_dilation")
      .template Attr<ArrayRef<int64_t>>("rhs_dilation")
      .template Attr<ArrayRef<int64_t>>("window_reversal")
      // Backend config attributes
      .template Attr<ConvBackendConfig>("backend_config")
      // Remaining attributes.
      .template Attr<int64_t>("feature_group_count")
      .template Attr<double>("result_scale");
}

template <CudnnConvKind kind>
static bool ConvFn(runtime::KernelContext* ctx, void** args, void** attrs) {
  static auto* handler =
      BindConvAttributes(CustomCall::Bind("xla.gpu.conv")
                             .UserData<const ServiceExecutableRunOptions*>()
                             .UserData<const DebugOptions*>()
                             .Arg<jitrt::StridedMemrefView>()  // operand0
                             .Arg<jitrt::StridedMemrefView>()  // operand1
                             .Value(CustomCall::None)          // bias
                             .Value(CustomCall::None)          // side_input
                             .Arg<jitrt::StridedMemrefView>()  // output
                             .Arg<jitrt::FlatMemrefView>()     // scratch
                         )
          .To(Conv::Handler(kind))
          .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs));
}

template <CudnnConvKind kind>
static bool ConvFusedFn(runtime::KernelContext* ctx, void** args,
                        void** attrs) {
  static auto* handler =
      BindConvAttributes(CustomCall::Bind("xla.gpu.conv.fused")
                             .UserData<const ServiceExecutableRunOptions*>()
                             .UserData<const DebugOptions*>()
                             .Arg<jitrt::StridedMemrefView>()  // operand0
                             .Arg<jitrt::StridedMemrefView>()  // operand1
                             .Arg<jitrt::FlatMemrefView>()     // bias
                             .Value(CustomCall::None)          // side_input
                             .Arg<jitrt::StridedMemrefView>()  // output
                             .Arg<jitrt::FlatMemrefView>()     // scratch
                         )
          .Attr<se::dnn::ActivationMode>("activation_mode")
          .To(Conv::Handler(kind))
          .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs));
}

template <CudnnConvKind kind>
static bool ConvFuseSideInputdFn(runtime::KernelContext* ctx, void** args,
                                 void** attrs) {
  static auto* handler =
      BindConvAttributes(CustomCall::Bind("xla.gpu.conv.fused.side_input")
                             .UserData<const ServiceExecutableRunOptions*>()
                             .UserData<const DebugOptions*>()
                             .Arg<jitrt::StridedMemrefView>()  // operand0
                             .Arg<jitrt::StridedMemrefView>()  // operand1
                             .Arg<jitrt::FlatMemrefView>()     // bias
                             .Arg<jitrt::StridedMemrefView>()  // side_input
                             .Arg<jitrt::StridedMemrefView>()  // output
                             .Arg<jitrt::FlatMemrefView>()     // scratch
                         )
          .Attr<se::dnn::ActivationMode>("activation_mode")
          .Attr<double>("side_input_scale")
          .To(Conv::Handler(kind))
          .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs));
}

// -------------------------------------------------------------------------- //

namespace {
struct Infeed {
  Error operator()(const ServiceExecutableRunOptions* run_options,
                   CustomCall::RemainingArgs args, StringRef config) const;
  static Infeed Handler() { return Infeed(); }
};
}  // namespace

Error Infeed::operator()(const ServiceExecutableRunOptions* run_options,
                         CustomCall::RemainingArgs args,
                         StringRef config) const {
  VLOG(3) << "Infeeding to GPU";

  se::Stream* stream = run_options->stream();
  ShapeTree<se::ScopedDeviceMemory<uint8_t>> source_buffers =
      GetOrCreateInfeedManager(stream->parent())->BlockingGetNextDestination();

  // Check that we have correct number of arguments.
  if (args.size() != source_buffers.leaf_count())
    return MakeStringError("Incorrect number of arguments");

  size_t index = 0;
  for (auto& source : source_buffers.leaves()) {
    // Get the destination buffer.
    auto dest = args.get<jitrt::StridedMemrefView>(index);
    if (failed(dest))
      return MakeStringError("Failed to get the destination buffer");

    // Get the source buffer shape.
    const Shape& source_shape =
        ShapeUtil::GetSubshape(source_buffers.shape(), source.first);

    // Check that destination shape matches the source shape.
    Shape dest_shape = ToShape(*dest);
    if (!ShapeUtil::Equal(dest_shape, source_shape)) {
      return MakeStringError(
          "The destination shape does not match the source shape");
    }

    se::DeviceMemoryBase dest_address = GetDeviceAddress(*dest);
    se::ScopedDeviceMemory<uint8_t>& buffer = source.second;
    stream->ThenMemcpy(&dest_address, *buffer.ptr(), buffer.ptr()->size());

    ++index;
  }

  // TODO(ezhulenev): Make this function async?
  Status block_status = stream->BlockHostUntilDone();
  if (!block_status.ok()) return AsError(block_status);

  VLOG(3) << "Infeeding to GPU complete";

  return Error::success();
}

static bool Infeed(runtime::KernelContext* ctx, void** args, void** attrs) {
  static auto* handler = CustomCall::Bind("xla.gpu.infeed")
                             .UserData<const ServiceExecutableRunOptions*>()
                             .Arg<CustomCall::RemainingArgs>()  // args
                             .Attr<StringRef>("config")
                             .To<RuntimeChecks()>(Infeed::Handler())
                             .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs));
}

// -------------------------------------------------------------------------- //

namespace {
struct Outfeed {
  Error operator()(const ServiceExecutableRunOptions* run_options,
                   CustomCall::RemainingArgs args, StringRef config) const;
  static Outfeed Handler() { return Outfeed(); }
};
}  // namespace

Error Outfeed::operator()(const ServiceExecutableRunOptions* run_options,
                          CustomCall::RemainingArgs args,
                          StringRef config) const {
  VLOG(3) << "Outfeeding from GPU";

  se::Stream* stream = run_options->stream();
  OutfeedManager* outfeed_manager = GetOrCreateOutfeedManager(stream->parent());
  ShapeTree<std::unique_ptr<OutfeedBuffer>>* dest_buffers =
      outfeed_manager->BlockingGetNextDestination();

  // Check that we have correct number of arguments.
  if (args.size() != dest_buffers->leaf_count())
    return MakeStringError("Incorrect number of arguments");

  size_t index = 0;
  for (auto& dest : dest_buffers->leaves()) {
    // Get the source buffer.
    auto source = args.get<jitrt::StridedMemrefView>(index);
    if (failed(source))
      return MakeStringError("Failed to get the source buffer");

    // Get the source buffer shape.
    const Shape& dest_shape =
        ShapeUtil::GetSubshape(dest_buffers->shape(), dest.first);

    // Check that destination shape matches the source shape.
    Shape source_shape = ToShape(*source);
    if (!ShapeUtil::Equal(dest_shape, source_shape)) {
      return MakeStringError(
          "The destination shape does not match the source shape");
    }

    se::DeviceMemoryBase source_address = GetDeviceAddress(*source);
    std::unique_ptr<OutfeedBuffer>& buffer = dest.second;

    // Schedule the memory transfer.
    auto* dest_address = buffer->destination()->untyped_data();
    stream->ThenMemcpy(dest_address, source_address, buffer->length())
        .ThenDoHostCallback([&buffer]() { buffer->Done(); });

    ++index;
  }

  Status block_status = stream->BlockHostUntilDone();
  if (!block_status.ok()) return AsError(block_status);

  VLOG(3) << "Outfeeding from GPU complete";

  return Error::success();
}

static bool Outfeed(runtime::KernelContext* ctx, void** args, void** attrs) {
  static auto* handler = CustomCall::Bind("xla.gpu.outfeed")
                             .UserData<const ServiceExecutableRunOptions*>()
                             .Arg<CustomCall::RemainingArgs>()  // args
                             .Attr<StringRef>("config")
                             .To<RuntimeChecks()>(Outfeed::Handler())
                             .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs));
}

// -------------------------------------------------------------------------- //

namespace {

enum class MemcpyDirection { kDeviceToDevice, kDeviceToHost, kHostToDevice };

template <MemcpyDirection direction>
struct Memcpy {
  Error operator()(const ServiceExecutableRunOptions* run_options,
                   jitrt::FlatMemrefView dst, jitrt::FlatMemrefView src) const;
  static Memcpy Handler() { return Memcpy(); }
};
}  // namespace

template <MemcpyDirection direction>
Error Memcpy<direction>::operator()(
    const ServiceExecutableRunOptions* run_options, jitrt::FlatMemrefView dst,
    jitrt::FlatMemrefView src) const {
  se::Stream* stream = run_options->stream();

  if (dst.size_in_bytes != src.size_in_bytes) {
    return MakeStringError(
        "Source memref size does not match destination memref size");
  }

  switch (direction) {
    case MemcpyDirection::kDeviceToDevice: {
      se::DeviceMemoryBase dst_data = GetDeviceAddress(dst);
      se::DeviceMemoryBase src_data = GetDeviceAddress(src);
      stream->ThenMemcpy(&dst_data, src_data, src.size_in_bytes);
    } break;
    case MemcpyDirection::kDeviceToHost: {
      se::DeviceMemoryBase src_data = GetDeviceAddress(src);
      stream->ThenMemcpy(dst.data, src_data, src.size_in_bytes);
    } break;
    case MemcpyDirection::kHostToDevice: {
      se::DeviceMemoryBase dst_data = GetDeviceAddress(dst);
      stream->ThenMemcpy(&dst_data, src.data, src.size_in_bytes);
    } break;
  }

  // TODO(ezhulenev): H2D and D2H memcpy instead of blocking the execution
  // thread should return an async token that will become available when
  // transfer is completed.
  if (direction != MemcpyDirection::kDeviceToDevice) {
    auto st = stream->BlockHostUntilDone();
    if (!st.ok()) return AsError(st);
  }

  return Error::success();
}

template <MemcpyDirection direction>
static bool MemcpyFn(runtime::KernelContext* ctx, void** args, void** attrs) {
  static auto* handler = CustomCall::Bind("xla.gpu.memcpy")
                             .UserData<const ServiceExecutableRunOptions*>()
                             .Arg<jitrt::FlatMemrefView>()  // dst
                             .Arg<jitrt::FlatMemrefView>()  // src
                             .To<RuntimeChecks()>(Memcpy<direction>::Handler())
                             .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs));
}

// -------------------------------------------------------------------------- //

namespace {

struct Memset {
  Error operator()(const ServiceExecutableRunOptions* run_options,
                   jitrt::FlatMemrefView dst,
                   CustomCall::VariantArg constant) const;
  static Memset Handler() { return Memset(); }
};

}  // namespace

Error Memset::operator()(const ServiceExecutableRunOptions* run_options,
                         jitrt::FlatMemrefView dst,
                         CustomCall::VariantArg constant) const {
  uint32_t pattern;
  if (constant.isa<int32_t>())
    pattern = *constant.get<int32_t>();
  else if (constant.isa<float>())
    pattern = reinterpret_cast<uint32_t&>(*constant.get<float>());
  else
    return MakeStringError("Unsupported memset value type");

  se::Stream* stream = run_options->stream();

  if (dst.size_in_bytes % 4 != 0)
    return MakeStringError("Memref size is not divisible by 4");

  se::DeviceMemoryBase dst_data = GetDeviceAddress(dst);
  stream->ThenMemset32(&dst_data, pattern, dst.size_in_bytes);

  return Error::success();
}

static bool MemsetFn(runtime::KernelContext* ctx, void** args, void** attrs) {
  static auto* handler = CustomCall::Bind("xla.gpu.memset")
                             .UserData<const ServiceExecutableRunOptions*>()
                             .Arg<jitrt::FlatMemrefView>()   // dst
                             .Arg<CustomCall::VariantArg>()  // constant
                             .To<RuntimeChecks()>(Memset::Handler())
                             .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs));
}

// -------------------------------------------------------------------------- //

namespace {
struct Fft {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  Error operator()(const ServiceExecutableRunOptions* run_options,
                   jitrt::StridedMemrefView input,
                   jitrt::StridedMemrefView output,
                   ArrayRef<int64_t> fft_length, se::fft::Type fft_type) const;
  static Fft Handler() { return Fft(); }
};
}  // namespace

Error Fft::operator()(const ServiceExecutableRunOptions* run_options,
                      jitrt::StridedMemrefView input,
                      jitrt::StridedMemrefView output,
                      ArrayRef<int64_t> fft_length,
                      se::fft::Type fft_type) const {
  // TODO(ezhulenev): Cache FFT plans in the GpuExecutable.
  FftPlanCache fft_plan_cache;

  se::Stream* stream = run_options->stream();
  se::StreamExecutor* executor = stream->parent();

  if (input.dtype == tfrt::DType::F64 ||
      input.dtype == tfrt::DType::Complex128) {
    // Adjust FFT type to reflect double precision.
    switch (fft_type) {
      case se::fft::Type::kC2CForward:
        fft_type = se::fft::Type::kZ2ZForward;
        break;
      case se::fft::Type::kC2CInverse:
        fft_type = se::fft::Type::kZ2ZInverse;
        break;
      case se::fft::Type::kR2C:
        fft_type = se::fft::Type::kD2Z;
        break;
      case se::fft::Type::kC2R:
        fft_type = se::fft::Type::kZ2D;
        break;
      default:
        return MakeStringError("Unsupported FFT type");
    }
  }

  auto st =
      RunFft(GetDeviceAddress(input), ToShape(input), GetDeviceAddress(output),
             ToShape(output), fft_type, fft_length, executor->device_ordinal(),
             &fft_plan_cache, stream, run_options->allocator());
  if (!st.ok()) return AsError(st);

  return Error::success();
}

static bool Fft(runtime::KernelContext* ctx, void** args, void** attrs) {
  static auto* handler = CustomCall::Bind("xla.gpu.fft")
                             .UserData<const ServiceExecutableRunOptions*>()
                             .Arg<jitrt::StridedMemrefView>()  // input
                             .Arg<jitrt::StridedMemrefView>()  // output
                             .Attr<ArrayRef<int64_t>>("fft_length")
                             .Attr<se::fft::Type>("fft_type")
                             .To<RuntimeChecks()>(Fft::Handler())
                             .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs));
}

// -------------------------------------------------------------------------- //

namespace {
struct Cholesky {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  Error operator()(const ServiceExecutableRunOptions* run_options,
                   const DebugOptions* debug_options, jitrt::MemrefView operand,
                   jitrt::MemrefView a, jitrt::MemrefView workspace,
                   jitrt::MemrefView info, int64_t batch_size, bool is_lower,
                   int64_t n) const;
  static Cholesky Handler() { return Cholesky(); }
};
}  // namespace

Error Cholesky::operator()(const ServiceExecutableRunOptions* run_options,
                           const DebugOptions* debug_options,
                           jitrt::MemrefView operand, jitrt::MemrefView a,
                           jitrt::MemrefView workspace, jitrt::MemrefView info,
                           int64_t batch_size, bool is_lower, int64_t n) const {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  se::DeviceMemoryBase operand_buffer = GetDeviceAddress(operand);
  se::DeviceMemoryBase a_buffer = GetDeviceAddress(a);
  se::DeviceMemoryBase workspace_buffer = GetDeviceAddress(workspace);
  se::DeviceMemoryBase info_buffer = GetDeviceAddress(info);

  VLOG(3) << "Running Cholesky";
  se::Stream* stream = run_options->stream();

  // Copy operand to the a buffer if they are different.
  if (a.data != operand.data)
    stream->ThenMemcpy(&a_buffer, operand_buffer, operand_buffer.size());

  using UpperLower = se::blas::UpperLower;
  UpperLower uplo = is_lower ? UpperLower::kLower : UpperLower::kUpper;

  CholeskyParams params{n,        batch_size,       uplo,
                        a_buffer, workspace_buffer, info_buffer};
  auto executed =
      RunCholesky(xla::gpu::PtxOptsFromDebugOptions(*debug_options),
                  TfrtToPrimitiveType(operand.dtype), &params, stream);
  if (!executed.ok()) return AsError(executed);

  return Error::success();
#else  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  return failure();
#endif
}

static bool Cholesky(runtime::KernelContext* ctx, void** args, void** attrs) {
  static auto* handler = CustomCall::Bind("xla.gpu.cholesky")
                             .UserData<const ServiceExecutableRunOptions*>()
                             .UserData<const DebugOptions*>()
                             .Arg<jitrt::MemrefView>()  // operand
                             .Arg<jitrt::MemrefView>()  // a
                             .Arg<jitrt::MemrefView>()  // workspace
                             .Arg<jitrt::MemrefView>()  // info
                             .Attr<int64_t>("batch_size")
                             .Attr<bool>("is_lower")
                             .Attr<int64_t>("n")
                             .To<RuntimeChecks()>(Cholesky::Handler())
                             .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs));
}

// -------------------------------------------------------------------------- //

namespace {

// TODO(ezhulenev): Today XLA represents TriangularSolve as a "classic" XLA
// custom call operation, and we provide a thin adaptor from Xla custom call
// to JitRt custom call. Once we are fully migrated to JitRt exectuion, XLA
// compiler should directly emit properly typed TriangularSolve JitRt custom
// call (no need to pass config via the serialized string).
struct TriangularSolve {
  // Adaptor from XlaCustomCall API to properly typed TriangularSolve handler.
  static Error run(const ServiceExecutableRunOptions* run_options,
                   const DebugOptions* debug_options,
                   CustomCall::RemainingArgs args, StringRef backend_config);

  Error operator()(const ServiceExecutableRunOptions* run_options,
                   const DebugOptions* debug_options,
                   jitrt::StridedMemrefView a, jitrt::StridedMemrefView b,
                   jitrt::StridedMemrefView result, jitrt::FlatMemrefView temp,
                   bool left_side, bool lower, bool unit_diagonal,
                   TriangularSolveOptions::Transpose transpose_a) const;
  static TriangularSolve Handler() { return TriangularSolve(); }
};

}  // namespace

Error TriangularSolve::run(const ServiceExecutableRunOptions* run_options,
                           const DebugOptions* debug_options,
                           CustomCall::RemainingArgs args,
                           StringRef backend_config) {
  TriangularSolve handler = TriangularSolve::Handler();

  if (args.size() != 4)
    return MakeStringError("Expected 4 arguments, got %n", args.size());

  // Check if all arguments have the correct type.
  auto a = args.get<jitrt::StridedMemrefView>(0);
  auto b = args.get<jitrt::StridedMemrefView>(1);
  auto result = args.get<jitrt::StridedMemrefView>(2);
  auto temp = args.get<jitrt::FlatMemrefView>(3);
  if (failed(a) || failed(b) || failed(result) || failed(temp))
    return MakeStringError("Incorrect argument types");

  // Parse backend config string.
  TriangularSolveOptions opts;
  auto st = tensorflow::HumanReadableJsonToProto(backend_config.str(), &opts);
  if (!st.ok()) return AsError(st);

  return handler(run_options, debug_options, *a, *b, *result, *temp,
                 opts.left_side(), opts.lower(), opts.unit_diagonal(),
                 opts.transpose_a());
}

Error TriangularSolve::operator()(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options, jitrt::StridedMemrefView a,
    jitrt::StridedMemrefView b, jitrt::StridedMemrefView result,
    jitrt::FlatMemrefView temp, bool left_side, bool lower, bool unit_diagonal,
    TriangularSolveOptions::Transpose transpose_a) const {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  se::Stream* stream = run_options->stream();

  se::DeviceMemoryBase a_data = GetDeviceAddress(a);
  se::DeviceMemoryBase b_data = GetDeviceAddress(b);
  se::DeviceMemoryBase result_data = GetDeviceAddress(result);
  se::DeviceMemoryBase temp_data = GetDeviceAddress(temp);

  // Triangular solve is in-place on 'b', so copy 'b' to the output if they
  // aren't the same buffer.
  if (b.data != result.data)
    stream->ThenMemcpy(&result_data, b_data, b_data.size());

  Shape b_shape = ToShape(b);
  int64_t m = b_shape.dimensions(b_shape.rank() - 2);
  int64_t n = b_shape.dimensions(b_shape.rank() - 1);
  int64_t batch_size = std::accumulate(
      b_shape.dimensions().begin(), b_shape.dimensions().end() - 2, int64_t{1},
      [](int64_t a, int64_t b) { return a * b; });

  PrimitiveType elem_type = TfrtToPrimitiveType(b.dtype);
  int64_t elem_size = ShapeUtil::ByteSizeOfPrimitiveType(elem_type);
  int64_t a_batch_stride = left_side ? m * m * elem_size : n * n * elem_size;
  int64_t b_batch_stride = m * n * elem_size;

  using Side = se::blas::Side;
  using Diagonal = se::blas::Diagonal;
  using Transpose = se::blas::Transpose;
  using UpperLower = se::blas::UpperLower;

  // Convert custom call attributes to se::blas enums.
  UpperLower uplo = lower ? UpperLower::kLower : UpperLower::kUpper;
  Side side = left_side ? Side::kLeft : Side::kRight;
  Diagonal diagonal = unit_diagonal ? Diagonal::kUnit : Diagonal::kNonUnit;

  auto transpose = [&]() -> mlir::FailureOr<Transpose> {
    switch (transpose_a) {
      case TriangularSolveOptions::NO_TRANSPOSE:
        return se::blas::Transpose::kNoTranspose;
      case TriangularSolveOptions::TRANSPOSE:
        return se::blas::Transpose::kTranspose;
      case TriangularSolveOptions::ADJOINT:
        return se::blas::Transpose::kConjugateTranspose;
      default:
        return failure();
    }
  }();

  if (failed(transpose))
    return MakeStringError("Failed to convert transpose type");

  auto st = RunTriangulatSolve(
      a_data, result_data, temp_data, PtxOptsFromDebugOptions(*debug_options),
      uplo, side, diagonal, *transpose, elem_type, batch_size, m, n,
      a_batch_stride, b_batch_stride, stream);
  if (!st.ok()) return AsError(st);

  return Error::success();
#else  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  return failure();
#endif
}

// -------------------------------------------------------------------------- //
// Implements JitRt custom call that forward to the Xla Custom Call handler.
//
// Longer term all Xla custom calls probably should be directly implemented as
// JitRt custom calls. However for smooth migration from Thunks to JitRt we have
// to seamlessly support all current XLA users.
namespace {
struct XlaCustomCall {
  using Stream = se::gpu::GpuStreamHandle;

  Error operator()(const ServiceExecutableRunOptions* run_options,
                   const DebugOptions* debug_options,
                   CustomCall::RemainingArgs args, StringRef call_target_name,
                   int32_t api_version, StringRef backend_config) const;
  static XlaCustomCall Handler() { return XlaCustomCall(); }
};
}  // namespace

Error XlaCustomCall::operator()(const ServiceExecutableRunOptions* run_options,
                                const DebugOptions* debug_options,
                                CustomCall::RemainingArgs args,
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
    return MakeStringError("Cannot find the Xla custom call handler %s",
                           call_target_name.str());
  }

  // Prepare pointers to buffers to pass to the Xla custom call handler.
  llvm::SmallVector<void*> buffers;
  for (unsigned i = 0; i < args.size(); ++i) {
    auto memref = args.get<jitrt::FlatMemrefView>(i);
    if (failed(memref))
      return MakeStringError("Failed to get arguments as memref view");

    // We use zero-sized memrefs to represent holes in custom calls with target
    // arguments mapping (see `CustomCallTargetArgMapping`).
    buffers.push_back(memref->size_in_bytes == 0 ? nullptr : memref->data);
  }

  // Original custom call API version that doesn't support returning status.
  if (api_version == CustomCallApiVersion::API_VERSION_ORIGINAL) {
    using XlaCustomCallType = void (*)(Stream, void**, const char*, size_t);
    auto xla_call_target = reinterpret_cast<XlaCustomCallType>(call_target);

    xla_call_target(se::gpu::AsGpuStreamValue(run_options->stream()),
                    buffers.data(), backend_config.data(),
                    backend_config.size());

    return Error::success();
  }

  // Xla Custom call API returning status.
  if (api_version == CustomCallApiVersion::API_VERSION_STATUS_RETURNING) {
    using XlaCustomCallType =
        void (*)(Stream, void**, const char*, size_t, XlaCustomCallStatus*);
    auto xla_call_target = reinterpret_cast<XlaCustomCallType>(call_target);

    XlaCustomCallStatus custom_call_status;
    xla_call_target(se::gpu::AsGpuStreamValue(run_options->stream()),
                    buffers.data(), backend_config.data(),
                    backend_config.size(), &custom_call_status);

    if (auto message = CustomCallStatusGetMessage(&custom_call_status)) {
      return MakeStringError(message.value());
    } else {
      return Error::success();
    }
  }

  return MakeStringError("Incorrect custom call API version");
}

static bool CustomCall(runtime::KernelContext* ctx, void** args, void** attrs) {
  static auto* handler = CustomCall::Bind("xla.gpu.memcpy")
                             .UserData<const ServiceExecutableRunOptions*>()
                             .UserData<const DebugOptions*>()
                             .Arg<jitrt::CustomCall::RemainingArgs>()  // args
                             .Attr<StringRef>("call_target_name")
                             .Attr<int32_t>("api_version")
                             .Attr<StringRef>("backend_config")
                             .To<RuntimeChecks()>(XlaCustomCall::Handler())
                             .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs));
}

// ------------------------------------------------------------------------- //

namespace {
struct AllReduce {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  LogicalResult operator()(const ServiceExecutableRunOptions* run_options,
                           JitRtCollectiveSupport* collectives,
                           CustomCall::RemainingArgs args, int32_t uid,
                           int64_t group_mode, int64_t op_id,
                           int64_t reduction_kind,
                           ArrayRef<int64_t> replica_group_offsets,
                           ArrayRef<int64_t> replica_group_values) const;
  static AllReduce Handler() { return AllReduce(); }
};
}  // namespace

LogicalResult AllReduce::operator()(
    const ServiceExecutableRunOptions* run_options,
    JitRtCollectiveSupport* collectives, CustomCall::RemainingArgs args,
    int32_t uid, int64_t group_mode, int64_t op_id, int64_t reduction_kind,
    ArrayRef<int64_t> replica_group_offsets,
    ArrayRef<int64_t> replica_group_values) const {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running AllReduce";
  se::Stream* stream = run_options->stream();
  NcclExecuteParams params(*run_options, stream);

  auto comm = GetNcclComm(params, group_mode, op_id, replica_group_offsets,
                          replica_group_values);
  if (failed(comm)) return comm;

  auto device_buffers = GetDeviceBufferPairs(args);
  if (failed(device_buffers)) return device_buffers;

  auto executed = RunAllReduce(static_cast<ReductionKind>(reduction_kind),
                               *device_buffers, *stream, **comm);
  if (!executed.ok()) return failure();

  int32_t device_ordinal = stream->parent()->device_ordinal();
  if (!collectives->MaybeBlockAfterFirstRun(uid, device_ordinal, stream).ok())
    return failure();

  return success();
#else   // XLA_ENABLE_XCCL
  // NCCL disabled.
  return failure();
#endif  // XLA_ENABLE_XCCL
}

static bool AllReduce(runtime::KernelContext* ctx, void** args, void** attrs) {
  static auto* handler =
      CustomCall::Bind("xla.gpu.all_reduce")
          .UserData<const ServiceExecutableRunOptions*>()
          .UserData<JitRtCollectiveSupport*>()
          .RemainingArgs()  // args
          .Attr<int32_t>("uid")
          .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
          .Attr<int64_t>("op_id")
          .Attr<int64_t>("reduction_kind")  // ReductionKind
          .Attr<ArrayRef<int64_t>>("replica_group_offsets")
          .Attr<ArrayRef<int64_t>>("replica_group_values")
          .To<RuntimeChecks()>(AllReduce::Handler())
          .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs));
}

// ------------------------------------------------------------------------- //

namespace {
struct AllReduceStart {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  LogicalResult operator()(const ServiceExecutableRunOptions* run_options,
                           JitRtAsyncCollectiveSupport* async_collectives,
                           CustomCall::RemainingArgs args, int64_t group_mode,
                           int64_t op_id, int64_t reduction_kind,
                           ArrayRef<int64_t> replica_group_offsets,
                           ArrayRef<int64_t> replica_group_values,
                           int32_t uid) const;
  static AllReduceStart Handler() { return AllReduceStart(); }
};
}  // namespace

LogicalResult AllReduceStart::operator()(
    const ServiceExecutableRunOptions* run_options,
    JitRtAsyncCollectiveSupport* async_collectives,
    CustomCall::RemainingArgs args, int64_t group_mode, int64_t op_id,
    int64_t reduction_kind, ArrayRef<int64_t> replica_group_offsets,
    ArrayRef<int64_t> replica_group_values, int32_t uid) const {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running AllReduceStart";
  se::Stream* stream = run_options->stream();
  NcclExecuteParams params(*run_options, stream);

  auto comm = GetNcclComm(params, group_mode, op_id, replica_group_offsets,
                          replica_group_values);
  if (failed(comm)) return comm;

  auto device_buffers = GetDeviceBufferPairs(args);
  if (failed(device_buffers)) return device_buffers;

  // Wait until compute inputs are ready.
  async_collectives->async_comm_stream()->ThenWaitFor(params.stream);

  auto executed =
      RunAllReduce(static_cast<ReductionKind>(reduction_kind), *device_buffers,
                   *async_collectives->async_comm_stream(), **comm);
  if (!executed.ok()) return failure();

  // Create an event on the async stream for the completion of the all-reduce.
  se::Event done_event(async_collectives->async_comm_stream()->parent());
  if (!done_event.Init()) return failure();
  async_collectives->async_comm_stream()->ThenRecordEvent(&done_event);

  if (failed(async_collectives->PushEvent(
          uid, stream->parent()->device_ordinal(), std::move(done_event))))
    return failure();

  return success();
#else   // XLA_ENABLE_XCCL
  return failure();  // NCCL disabled.
#endif  // XLA_ENABLE_XCCL
}

static bool AllReduceStart(runtime::KernelContext* ctx, void** args,
                           void** attrs) {
  static auto* handler =
      CustomCall::Bind("xla.gpu.all_reduce_start")
          .UserData<const ServiceExecutableRunOptions*>()
          .UserData<JitRtAsyncCollectiveSupport*>()
          .RemainingArgs()              // args
          .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
          .Attr<int64_t>("op_id")
          .Attr<int64_t>("reduction_kind")  // ReductionKind
          .Attr<ArrayRef<int64_t>>("replica_group_offsets")
          .Attr<ArrayRef<int64_t>>("replica_group_values")
          .Attr<int32_t>("uid")
          .To<RuntimeChecks()>(AllReduceStart::Handler())
          .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs));
}

// ------------------------------------------------------------------------- //

namespace {
struct AllReduceDone {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  LogicalResult operator()(const ServiceExecutableRunOptions* run_options,
                           JitRtCollectiveSupport* collectives,
                           JitRtAsyncCollectiveSupport* async_collectives,
                           CustomCall::RemainingArgs args, int32_t uid) const;
  static AllReduceDone Handler() { return AllReduceDone(); }
};
}  // namespace

LogicalResult AllReduceDone::operator()(
    const ServiceExecutableRunOptions* run_options,
    JitRtCollectiveSupport* collectives,
    JitRtAsyncCollectiveSupport* async_collectives,
    CustomCall::RemainingArgs args, int32_t uid) const {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running AllReduceDone";
  se::Stream* stream = run_options->stream();

  int32_t device_ordinal = stream->parent()->device_ordinal();
  auto event = async_collectives->PopEvent(uid, device_ordinal);
  if (failed(event)) return failure();

  stream->ThenWaitFor(&*event);

  if (!collectives->MaybeBlockAfterFirstRun(uid, device_ordinal, stream).ok())
    return failure();

  return success();
#else   // XLA_ENABLE_XCCL
  return failure();  // NCCL disabled.
#endif  // XLA_ENABLE_XCCL
}

static bool AllReduceDone(runtime::KernelContext* ctx, void** args,
                          void** attrs) {
  static auto* handler = CustomCall::Bind("xla.gpu.all_reduce_done")
                             .UserData<const ServiceExecutableRunOptions*>()
                             .UserData<JitRtCollectiveSupport*>()
                             .UserData<JitRtAsyncCollectiveSupport*>()
                             .RemainingArgs()  // args
                             .Attr<int32_t>("uid")
                             .To<RuntimeChecks()>(AllReduceDone::Handler())
                             .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs));
}

// -------------------------------------------------------------------------- //

namespace {
struct ReduceScatter {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  LogicalResult operator()(const ServiceExecutableRunOptions* run_options,
                           JitRtCollectiveSupport* collectives,
                           CustomCall::RemainingArgs args, int32_t uid,
                           int64_t group_mode, int64_t op_id,
                           int64_t reduction_kind,
                           ArrayRef<int64_t> replica_group_offsets,
                           ArrayRef<int64_t> replica_group_values) const;
  static ReduceScatter Handler() { return ReduceScatter(); }
};
}  // namespace

LogicalResult ReduceScatter::operator()(
    const ServiceExecutableRunOptions* run_options,
    JitRtCollectiveSupport* collectives, CustomCall::RemainingArgs args,
    int32_t uid, int64_t group_mode, int64_t op_id, int64_t reduction_kind,
    ArrayRef<int64_t> replica_group_offsets,
    ArrayRef<int64_t> replica_group_values) const {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running ReduceScatter";
  se::Stream* stream = run_options->stream();
  NcclExecuteParams params(*run_options, stream);

  auto comm = GetNcclComm(params, group_mode, op_id, replica_group_offsets,
                          replica_group_values);
  if (failed(comm)) return comm;

  auto device_buffers = GetDeviceBufferPairs(args);
  if (failed(device_buffers)) return device_buffers;

  auto executed = RunReduceScatter(static_cast<ReductionKind>(reduction_kind),
                                   *device_buffers, *stream, **comm);
  if (!executed.ok()) return failure();

  int32_t device_ordinal = stream->parent()->device_ordinal();
  if (!collectives->MaybeBlockAfterFirstRun(uid, device_ordinal, stream).ok())
    return failure();

  return success();
#else   // XLA_ENABLE_XCCL
  // NCCL disabled.
  return failure();
#endif  // XLA_ENABLE_XCCL
}

static bool ReduceScatter(runtime::KernelContext* ctx, void** args,
                          void** attrs) {
  static auto* handler =
      CustomCall::Bind("xla.gpu.reduce_scatter")
          .UserData<const ServiceExecutableRunOptions*>()
          .UserData<JitRtCollectiveSupport*>()
          .RemainingArgs()  // args
          .Attr<int32_t>("uid")
          .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
          .Attr<int64_t>("op_id")
          .Attr<int64_t>("reduction_kind")  // ReductionKind
          .Attr<ArrayRef<int64_t>>("replica_group_offsets")
          .Attr<ArrayRef<int64_t>>("replica_group_values")
          .To<RuntimeChecks()>(ReduceScatter::Handler())
          .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs));
}

// -------------------------------------------------------------------------- //

namespace {
struct AllGather {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  LogicalResult operator()(const ServiceExecutableRunOptions* run_options,
                           JitRtCollectiveSupport* collectives,
                           CustomCall::RemainingArgs args, int32_t uid,
                           int64_t group_mode, int64_t op_id,
                           ArrayRef<int64_t> replica_group_offsets,
                           ArrayRef<int64_t> replica_group_values) const;
  static AllGather Handler() { return AllGather(); }
};
}  // namespace

LogicalResult AllGather::operator()(
    const ServiceExecutableRunOptions* run_options,
    JitRtCollectiveSupport* collectives, CustomCall::RemainingArgs args,
    int32_t uid, int64_t group_mode, int64_t op_id,
    ArrayRef<int64_t> replica_group_offsets,
    ArrayRef<int64_t> replica_group_values) const {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running AllGather";
  se::Stream* stream = run_options->stream();
  NcclExecuteParams params(*run_options, stream);

  auto comm = GetNcclComm(params, group_mode, op_id, replica_group_offsets,
                          replica_group_values);
  if (failed(comm)) return comm;

  auto device_buffers = GetDeviceBufferPairs(args);
  if (failed(device_buffers)) return device_buffers;

  if (!RunAllGather(*device_buffers, *stream, **comm).ok()) return failure();

  int32_t device_ordinal = stream->parent()->device_ordinal();
  if (!collectives->MaybeBlockAfterFirstRun(uid, device_ordinal, stream).ok())
    return failure();

  return success();
#else   // XLA_ENABLE_XCCL
  // NCCL disabled.
  return failure();
#endif  // XLA_ENABLE_XCCL
}

static bool AllGather(runtime::KernelContext* ctx, void** args, void** attrs) {
  static auto* handler =
      CustomCall::Bind("xla.gpu.all_gather")
          .UserData<const ServiceExecutableRunOptions*>()
          .UserData<JitRtCollectiveSupport*>()
          .RemainingArgs()  // args
          .Attr<int32_t>("uid")
          .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
          .Attr<int64_t>("op_id")
          .Attr<ArrayRef<int64_t>>("replica_group_offsets")
          .Attr<ArrayRef<int64_t>>("replica_group_values")
          .To<RuntimeChecks()>(AllGather::Handler())
          .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs));
}

// -------------------------------------------------------------------------- //

namespace {
struct AllToAll {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  LogicalResult operator()(const ServiceExecutableRunOptions* run_options,
                           JitRtCollectiveSupport* collectives,
                           CustomCall::RemainingArgs args, int32_t uid,
                           int64_t group_mode, bool has_split_dimension,
                           int64_t op_id,
                           ArrayRef<int64_t> replica_group_offsets,
                           ArrayRef<int64_t> replica_group_values) const;
  static AllToAll Handler() { return AllToAll(); }
};
}  // namespace

LogicalResult AllToAll::operator()(
    const ServiceExecutableRunOptions* run_options,
    JitRtCollectiveSupport* collectives, CustomCall::RemainingArgs args,
    int32_t uid, int64_t group_mode, bool has_split_dimension, int64_t op_id,
    ArrayRef<int64_t> replica_group_offsets,
    ArrayRef<int64_t> replica_group_values) const {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running AllToAll";
  se::Stream* stream = run_options->stream();
  NcclExecuteParams params(*run_options, stream);

  auto comm = GetNcclComm(params, group_mode, op_id, replica_group_offsets,
                          replica_group_values);
  if (failed(comm)) return comm;

  auto device_buffers = GetDeviceBufferPairs(args);
  if (failed(device_buffers)) return device_buffers;

  if (!RunAllToAll(has_split_dimension, *device_buffers, *stream, **comm).ok())
    return failure();

  int32_t device_ordinal = stream->parent()->device_ordinal();
  if (!collectives->MaybeBlockAfterFirstRun(uid, device_ordinal, stream).ok())
    return failure();

  return success();
#else   // XLA_ENABLE_XCCL
  // NCCL disabled.
  return failure();
#endif  // XLA_ENABLE_XCCL
}

static bool AllToAll(runtime::KernelContext* ctx, void** args, void** attrs) {
  static auto* handler =
      CustomCall::Bind("xla.gpu.all_to_all")
          .UserData<const ServiceExecutableRunOptions*>()
          .UserData<JitRtCollectiveSupport*>()
          .RemainingArgs()  // args
          .Attr<int32_t>("uid")
          .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
          .Attr<bool>("has_split_dimension")
          .Attr<int64_t>("op_id")
          .Attr<ArrayRef<int64_t>>("replica_group_offsets")
          .Attr<ArrayRef<int64_t>>("replica_group_values")
          .To<RuntimeChecks()>(AllToAll::Handler())
          .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs));
}

// -------------------------------------------------------------------------- //

namespace {
struct CollectivePermute {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  LogicalResult operator()(const ServiceExecutableRunOptions* run_options,
                           JitRtCollectiveSupport* collectives,
                           CustomCall::RemainingArgs args, int32_t uid,
                           int64_t group_mode, int64_t op_id,
                           ArrayRef<int64_t> replica_group_offsets,
                           ArrayRef<int64_t> replica_group_values,
                           ArrayRef<int64_t> source_peers,
                           ArrayRef<int64_t> target_peers) const;
  static CollectivePermute Handler() { return CollectivePermute(); }
};
}  // namespace

LogicalResult CollectivePermute::operator()(
    const ServiceExecutableRunOptions* run_options,
    JitRtCollectiveSupport* collectives, CustomCall::RemainingArgs args,
    int32_t uid, int64_t group_mode, int64_t op_id,
    ArrayRef<int64_t> replica_group_offsets,
    ArrayRef<int64_t> replica_group_values, ArrayRef<int64_t> source_peers,
    ArrayRef<int64_t> target_peers) const {
#if XLA_ENABLE_XCCL
  VLOG(3) << "Running CollectivePermute";
  se::Stream* stream = run_options->stream();
  NcclExecuteParams params(*run_options, stream);

  auto comm = GetNcclComm(params, group_mode, op_id, replica_group_offsets,
                          replica_group_values);
  if (failed(comm)) return comm;

  auto device_buffers = GetDeviceBufferPairs(args);
  if (failed(device_buffers)) return device_buffers;
  if (device_buffers->size() != 1) return failure();

  StatusOr<GlobalDeviceId> global_device_id = params.GetGlobalDeviceId();
  if (!global_device_id.ok()) return failure();

  StatusOr<DeviceAssignment::LogicalID> current_logical_id =
      params.device_assn->LogicalIdForDevice(global_device_id.value());
  if (!current_logical_id.ok()) return failure();

  const int64_t current_id = static_cast<CollectiveOpGroupMode>(group_mode) ==
                                     CollectiveOpGroupMode::kCrossReplica
                                 ? current_logical_id.value().replica_id
                                 : current_logical_id.value().computation_id;
  std::string device_string = NcclCollectiveThunk::GetDeviceString(params);

  NcclCollectivePermuteConfig::IdToSourceTargetMap id_to_source_target;
  for (int i = 0; i < source_peers.size(); ++i) {
    id_to_source_target.insert({target_peers[i], {}}).first->second.source =
        source_peers[i];
    id_to_source_target.insert({source_peers[i], {}}).first->second.target =
        target_peers[i];
  }
  const NcclCollectivePermuteConfig::SourceTargetMapEntry source_target =
      NcclCollectivePermuteConfig::GetSourceTarget(id_to_source_target,
                                                   current_id);

  auto executed =
      RunCollectivePermute(source_target, (*device_buffers)[0], *stream, **comm,
                           device_string, current_id);
  if (!executed.ok()) return failure();

  int32_t device_ordinal = stream->parent()->device_ordinal();
  if (!collectives->MaybeBlockAfterFirstRun(uid, device_ordinal, stream).ok())
    return failure();

  return success();
#else   // XLA_ENABLE_XCCL
  // NCCL disabled.
  return failure();
#endif  // XLA_ENABLE_XCCL
}

static bool CollectivePermute(runtime::KernelContext* ctx, void** args,
                              void** attrs) {
  static auto* handler =
      CustomCall::Bind("xla.gpu.collective_permute")
          .UserData<const ServiceExecutableRunOptions*>()
          .UserData<JitRtCollectiveSupport*>()
          .RemainingArgs()  // args
          .Attr<int32_t>("uid")
          .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
          .Attr<int64_t>("op_id")
          .Attr<ArrayRef<int64_t>>("replica_group_offsets")
          .Attr<ArrayRef<int64_t>>("replica_group_values")
          .Attr<ArrayRef<int64_t>>("source_peers")
          .Attr<ArrayRef<int64_t>>("target_peers")
          .To<RuntimeChecks()>(CollectivePermute::Handler())
          .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs));
}

// -------------------------------------------------------------------------- //

namespace {
struct ReplicaId {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  LogicalResult operator()(const ServiceExecutableRunOptions* run_options,
                           jitrt::FlatMemrefView result) const;
  static ReplicaId Handler() { return ReplicaId(); }
};
}  // namespace

LogicalResult ReplicaId::operator()(
    const ServiceExecutableRunOptions* run_options,
    jitrt::FlatMemrefView result) const {
  VLOG(3) << "Running ReplicaId";
  se::Stream* stream = run_options->stream();
  NcclExecuteParams params(*run_options, stream);

  StatusOr<GlobalDeviceId> global_device_id = params.GetGlobalDeviceId();
  if (!global_device_id.ok()) return failure();

  StatusOr<DeviceAssignment::LogicalID> logical_id =
      params.device_assn->LogicalIdForDevice(global_device_id.value());
  if (!logical_id.ok()) return failure();

  se::DeviceMemoryBase result_data = GetDeviceAddress(result);
  params.stream->ThenMemset32(&result_data, logical_id.value().replica_id,
                              /*size=*/4);

  return success();
}

static bool ReplicaId(runtime::KernelContext* ctx, void** args, void** attrs) {
  static auto* handler = CustomCall::Bind("xla.gpu.replica_id")
                             .UserData<const ServiceExecutableRunOptions*>()
                             .Arg<jitrt::FlatMemrefView>()  // result
                             .To<RuntimeChecks()>(ReplicaId::Handler())
                             .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs));
}

// -------------------------------------------------------------------------- //

namespace {
struct PartitionId {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  LogicalResult operator()(const ServiceExecutableRunOptions* run_options,
                           jitrt::FlatMemrefView result) const;
  static PartitionId Handler() { return PartitionId(); }
};
}  // namespace

LogicalResult PartitionId::operator()(
    const ServiceExecutableRunOptions* run_options,
    jitrt::FlatMemrefView result) const {
  VLOG(3) << "Running PartitionId";
  se::Stream* stream = run_options->stream();
  NcclExecuteParams params(*run_options, stream);

  StatusOr<GlobalDeviceId> global_device_id = params.GetGlobalDeviceId();
  if (!global_device_id.ok()) return failure();

  StatusOr<DeviceAssignment::LogicalID> logical_id =
      params.device_assn->LogicalIdForDevice(global_device_id.value());
  if (!logical_id.ok()) return failure();

  se::DeviceMemoryBase result_data = GetDeviceAddress(result);
  params.stream->ThenMemset32(&result_data, logical_id.value().computation_id,
                              /*size=*/4);

  return success();
}

static bool PartitionId(runtime::KernelContext* ctx, void** args,
                        void** attrs) {
  static auto* handler = CustomCall::Bind("xla.gpu.partition_id")
                             .UserData<const ServiceExecutableRunOptions*>()
                             .Arg<jitrt::FlatMemrefView>()  // result
                             .To<RuntimeChecks()>(PartitionId::Handler())
                             .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs));
}

// -------------------------------------------------------------------------- //

DirectCustomCallLibrary JitRtGpuCustomCalls() {
  DirectCustomCallLibrary lib;

  lib.Insert("xla.gpu.fft", &xla::gpu::Fft);
  lib.Insert("xla.gpu.cholesky", &xla::gpu::Cholesky);
  lib.Insert("xla.gpu.collective_permute", &xla::gpu::CollectivePermute);
  lib.Insert("xla.gpu.func.launch", &xla::gpu::LaunchFunc);
  lib.Insert("xla.gpu.gemm", &xla::gpu::Gemm);

  auto conv = [](StringRef name) { return ("xla.gpu.conv." + name).str(); };
  lib.Insert(conv("forward"), &ConvFn<CudnnConvKind::kForward>);
  lib.Insert(conv("backward.input"), &ConvFn<CudnnConvKind::kBackwardInput>);
  lib.Insert(conv("backward.filter"), &ConvFn<CudnnConvKind::kBackwardFilter>);
  lib.Insert(conv("forward.fused"),
             &ConvFusedFn<CudnnConvKind::kForwardActivation>);
  lib.Insert(conv("forward.fused.side_input"),
             &ConvFuseSideInputdFn<CudnnConvKind::kForwardActivation>);

  lib.Insert("xla.gpu.memcpy.d2d", &MemcpyFn<MemcpyDirection::kDeviceToDevice>);
  lib.Insert("xla.gpu.memcpy.h2d", &MemcpyFn<MemcpyDirection::kHostToDevice>);
  lib.Insert("xla.gpu.memcpy.d2h", &MemcpyFn<MemcpyDirection::kDeviceToHost>);
  lib.Insert("xla.gpu.memset", &MemsetFn);
  lib.Insert("xla.gpu.infeed", &xla::gpu::Infeed);
  lib.Insert("xla.gpu.outfeed", &xla::gpu::Outfeed);
  lib.Insert("xla.gpu.custom_call", &xla::gpu::CustomCall);

  // Collective operations.
  lib.Insert("xla.gpu.all_gather", &xla::gpu::AllGather);
  lib.Insert("xla.gpu.all_reduce", &xla::gpu::AllReduce);
  lib.Insert("xla.gpu.all_reduce_done", &xla::gpu::AllReduceDone);
  lib.Insert("xla.gpu.all_reduce_start", &xla::gpu::AllReduceStart);
  lib.Insert("xla.gpu.all_to_all", &xla::gpu::AllToAll);
  lib.Insert("xla.gpu.reduce_scatter", &xla::gpu::ReduceScatter);
  lib.Insert("xla.gpu.partition_id", &xla::gpu::PartitionId);
  lib.Insert("xla.gpu.replica_id", &xla::gpu::ReplicaId);

  return lib;
}

}  // namespace gpu
}  // namespace xla
