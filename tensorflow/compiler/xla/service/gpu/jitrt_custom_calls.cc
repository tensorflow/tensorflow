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
#include <iterator>
#include <memory>
#include <utility>

#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/xla/service/custom_call_status_internal.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/service/gpu/cholesky_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_asm_opts_util.h"
#include "tensorflow/compiler/xla/service/gpu/infeed_manager.h"
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_gather_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_collective_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/outfeed_manager.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/stream_executor/gpu/gpu_types.h"
#include "tfrt/jitrt/custom_call.h"  // from @tf_runtime
#include "tfrt/jitrt/jitrt.h"  // from @tf_runtime
#include "tfrt/dtype/dtype.h"  // from @tf_runtime

TFRT_DEFINE_EXPLICIT_DENSE_TYPE_ID(tfrt::jitrt::CustomCall,
                                   xla::gpu::JitRtKernelsCache);
TFRT_DEFINE_EXPLICIT_DENSE_TYPE_ID(tfrt::jitrt::CustomCall,
                                   xla::gpu::JitRtGemmConfigCache);
TFRT_DEFINE_EXPLICIT_DENSE_TYPE_ID(tfrt::jitrt::CustomCall,
                                   const xla::ServiceExecutableRunOptions);
TFRT_DEFINE_EXPLICIT_DENSE_TYPE_ID(tfrt::jitrt::CustomCall,
                                   const xla::DebugOptions);

namespace xla {
namespace gpu {

using llvm::ArrayRef;
using llvm::orc::MangleAndInterner;
using llvm::orc::SymbolMap;

using mlir::failure;
using mlir::FailureOr;
using mlir::LogicalResult;
using mlir::StringRef;
using mlir::succeeded;
using mlir::success;

using tfrt::jitrt::CustomCall;
using tfrt::jitrt::Executable;

namespace se = ::stream_executor;
namespace jitrt = ::tfrt::jitrt;
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

se::KernelBase* JitRtKernelsCache::Get(se::StreamExecutor* executor,
                                       const char* data) {
  Key key = {executor, data};

  absl::MutexLock lock(&mutex_);
  auto it = kernels_cache_.find(key);
  if (it != kernels_cache_.end()) return it->second.get();

  return nullptr;
}

se::KernelBase* JitRtKernelsCache::Set(se::StreamExecutor* executor,
                                       const char* data,
                                       std::unique_ptr<se::KernelBase> kernel) {
  Key key = {executor, data};

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

static PrimitiveType ToPrimitiveType(tfrt::DType dtype) {
  switch (dtype) {
    // Unsigned integer types.
    case tfrt::DType::UI8:
      return PrimitiveType::U8;
    case tfrt::DType::UI16:
      return PrimitiveType::U16;
    case tfrt::DType::UI32:
      return PrimitiveType::U32;
    case tfrt::DType::UI64:
      return PrimitiveType::U64;

    // Signed integer types.
    case tfrt::DType::I1:
      return PrimitiveType::PRED;
    case tfrt::DType::I8:
      return PrimitiveType::S8;
    case tfrt::DType::I16:
      return PrimitiveType::S16;
    case tfrt::DType::I32:
      return PrimitiveType::S32;
    case tfrt::DType::I64:
      return PrimitiveType::S64;

    // Floating point types.
    case tfrt::DType::F16:
      return PrimitiveType::F16;
    case tfrt::DType::F32:
      return PrimitiveType::F32;
    case tfrt::DType::F64:
      return PrimitiveType::F64;
    case tfrt::DType::BF16:
      return PrimitiveType::BF16;

    // Complex types.
    case tfrt::DType::Complex64:
      return PrimitiveType::C64;
    case tfrt::DType::Complex128:
      return PrimitiveType::C128;

    default:
      LOG(FATAL) << "Unsupported data type: " << dtype;
  }
}

static Shape ToShape(const jitrt::StridedMemrefView& memref) {
  PrimitiveType type = ToPrimitiveType(memref.dtype);

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
    const DebugOptions* debug_options, const jitrt::StridedMemrefView& lhs,
    const jitrt::StridedMemrefView& rhs, const jitrt::StridedMemrefView& out,
    int64_t algorithm, double alpha_real, double alpha_imag,
    ArrayRef<int64_t> lhs_batch, ArrayRef<int64_t> lhs_contract,
    ArrayRef<int64_t> rhs_batch, ArrayRef<int64_t> rhs_contract,
    llvm::Optional<double> beta = llvm::None) {
  return GemmConfig::For(ToShape(lhs), lhs_batch, lhs_contract, ToShape(rhs),
                         rhs_batch, rhs_contract, ToShape(out), alpha_real,
                         alpha_imag, beta.getValueOr(0.0), algorithm,
                         debug_options->xla_gpu_enable_cublaslt());
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
        ToPrimitiveType(source->dtype), element_count,
        GetDeviceAddress(*source), GetDeviceAddress(*destination)});
  }
  return device_buffers;
}

// -------------------------------------------------------------------------- //

namespace {
struct LaunchFunc {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  LogicalResult operator()(const ServiceExecutableRunOptions* run_options,
                           JitRtKernelsCache* kernels_cache,
                           int32_t grid_size_x, int32_t grid_size_y,
                           int32_t grid_size_z, int32_t block_size_x,
                           int32_t block_size_y, int32_t block_size_z,
                           CustomCall::RemainingArgs args, StringRef ptx,
                           StringRef name) const;

  static LaunchFunc Handler() { return LaunchFunc(); }
};
}  // namespace

LogicalResult LaunchFunc::operator()(
    const ServiceExecutableRunOptions* run_options,
    JitRtKernelsCache* kernels_cache, int32_t grid_size_x, int32_t grid_size_y,
    int32_t grid_size_z, int32_t block_size_x, int32_t block_size_y,
    int32_t block_size_z, CustomCall::RemainingArgs args, StringRef ptx,
    StringRef name) const {
  se::Stream* stream = run_options->stream();
  se::StreamExecutor* executor = stream->parent();

  LaunchDimensions launch_dimensions(
      {grid_size_x, grid_size_y, grid_size_z},
      {block_size_x, block_size_y, block_size_z});

  se::KernelBase* kernel = kernels_cache->Get(executor, ptx.data());

  // If kernel does not exists create it from the ptx.
  if (kernel == nullptr) {
    auto created = CreateKernel(absl::string_view(name.data(), name.size()),
                                args.size(), ptx.data(), {}, executor);
    if (!created.ok()) return failure();

    kernel = kernels_cache->Set(executor, ptx.data(), std::move(*created));
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

    // Unsupported argument type.
    return failure();
  }

  // Execute device kernel on a main stream.
  auto executed =
      ExecuteKernelOnStream(*kernel, buffer_args, launch_dimensions, stream);
  if (!executed.ok()) return failure();

  return success();
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

  return succeeded(handler->call(args, attrs, Executable::GetUserData(ctx)));
}

// -------------------------------------------------------------------------- //

namespace {
struct Gemm {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  LogicalResult operator()(
      const ServiceExecutableRunOptions* run_options,
      const DebugOptions* debug_options, JitRtGemmConfigCache* configs,
      jitrt::StridedMemrefView lhs, jitrt::StridedMemrefView rhs,
      jitrt::StridedMemrefView out, int64_t algorithm, double alpha_real,
      double alpha_imag, ArrayRef<int64_t> lhs_batch,
      ArrayRef<int64_t> lhs_contract, ArrayRef<int64_t> rhs_batch,
      ArrayRef<int64_t> rhs_contract, int64_t uid) const;

  static Gemm Handler() { return Gemm(); }
};
}  // namespace

LogicalResult Gemm::operator()(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options, JitRtGemmConfigCache* configs,
    jitrt::StridedMemrefView lhs, jitrt::StridedMemrefView rhs,
    jitrt::StridedMemrefView out, int64_t algorithm, double alpha_real,
    double alpha_imag, ArrayRef<int64_t> lhs_batch,
    ArrayRef<int64_t> lhs_contract, ArrayRef<int64_t> rhs_batch,
    ArrayRef<int64_t> rhs_contract, int64_t uid) const {
  se::DeviceMemoryBase lhs_data = GetDeviceAddress(lhs);
  se::DeviceMemoryBase rhs_data = GetDeviceAddress(rhs);
  se::DeviceMemoryBase output_data = GetDeviceAddress(out);

  VLOG(3) << "Running GEMM";
  se::Stream* stream = run_options->stream();

  // Find the gemm config for this instance of operation based on uid.
  const GemmConfig* config = configs->Get(uid);
  if (config == nullptr) {
    auto cfg = GetGemmConfig(debug_options, lhs, rhs, out, algorithm,
                             alpha_real, alpha_imag, lhs_batch, lhs_contract,
                             rhs_batch, rhs_contract);
    if (!cfg.ok()) return failure();
    config = configs->Set(uid, std::move(*cfg));
  }

  Status executed;
  if (config->use_cublaslt && stream->parent()->SupportsBlasPlans()) {
    se::OwningScratchAllocator<> scratch_allocator(
        run_options->device_ordinal(), run_options->allocator());
    executed = RunBlasLtMatmul(*config, lhs_data, rhs_data, output_data, stream,
                               scratch_allocator);
  } else {
    executed = RunGemm(*config, lhs_data, rhs_data, output_data, stream);
  }

  if (!executed.ok()) return failure();

  return success();
}

static bool Gemm(runtime::KernelContext* ctx, void** args, void** attrs) {
  static auto* handler =
      CustomCall::Bind("xla.gpu.gemm")
          .UserData<const ServiceExecutableRunOptions*>()
          .UserData<const DebugOptions*>()
          .UserData<JitRtGemmConfigCache*>()
          .Arg<jitrt::StridedMemrefView>()  // lhs
          .Arg<jitrt::StridedMemrefView>()  // rhs
          .Arg<jitrt::StridedMemrefView>()  // out
          .Attr<int64_t>("algorithm")
          .Attr<double>("alpha_real")
          .Attr<double>("alpha_imag")
          .Attr<ArrayRef<int64_t>>("lhs_batching_dimensions")
          .Attr<ArrayRef<int64_t>>("lhs_contracting_dimensions")
          .Attr<ArrayRef<int64_t>>("rhs_batching_dimensions")
          .Attr<ArrayRef<int64_t>>("rhs_contracting_dimensions")
          .Attr<int64_t>("uid")
          .To<RuntimeChecks()>(Gemm::Handler())
          .release();

  return succeeded(handler->call(args, attrs, Executable::GetUserData(ctx)));
}

// -------------------------------------------------------------------------- //

namespace {
struct GemmBias {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  LogicalResult operator()(
      const ServiceExecutableRunOptions* run_options,
      const DebugOptions* debug_options, JitRtGemmConfigCache* configs,
      jitrt::StridedMemrefView lhs, jitrt::StridedMemrefView rhs,
      jitrt::StridedMemrefView bias, jitrt::StridedMemrefView out,
      int64_t algorithm, double alpha_real, double alpha_imag, double beta,
      ArrayRef<int64_t> lhs_batch, ArrayRef<int64_t> lhs_contract,
      ArrayRef<int64_t> rhs_batch, ArrayRef<int64_t> rhs_contract,
      int64_t uid) const;
  static GemmBias Handler() { return GemmBias(); }
};
}  // namespace

LogicalResult GemmBias::operator()(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options, JitRtGemmConfigCache* configs,
    jitrt::StridedMemrefView lhs, jitrt::StridedMemrefView rhs,
    jitrt::StridedMemrefView bias, jitrt::StridedMemrefView out,
    int64_t algorithm, double alpha_real, double alpha_imag, double beta,
    ArrayRef<int64_t> lhs_batch, ArrayRef<int64_t> lhs_contract,
    ArrayRef<int64_t> rhs_batch, ArrayRef<int64_t> rhs_contract,
    int64_t uid) const {
  se::DeviceMemoryBase lhs_data = GetDeviceAddress(lhs);
  se::DeviceMemoryBase rhs_data = GetDeviceAddress(rhs);
  se::DeviceMemoryBase bias_data = GetDeviceAddress(bias);
  se::DeviceMemoryBase output_data = GetDeviceAddress(out);

  VLOG(3) << "Running GEMM + Bias [beta=" << beta << "]";
  se::Stream* stream = run_options->stream();

  // Find the gemm config for this instance of operation based on uid.
  const GemmConfig* config = configs->Get(uid);
  if (config == nullptr) {
    auto cfg = GetGemmConfig(debug_options, lhs, rhs, out, algorithm,
                             alpha_real, alpha_imag, lhs_batch, lhs_contract,
                             rhs_batch, rhs_contract, beta);
    if (!cfg.ok()) return failure();
    config = configs->Set(uid, std::move(*cfg));
  }

  // Copy bias to the output buffer of they are different.
  if (out.data != bias.data)
    stream->ThenMemcpy(&output_data, bias_data, bias_data.size());

  Status executed;
  if (config->use_cublaslt && stream->parent()->SupportsBlasPlans()) {
    se::OwningScratchAllocator<> scratch_allocator(
        run_options->device_ordinal(), run_options->allocator());
    executed = RunBlasLtMatmul(*config, lhs_data, rhs_data, output_data, stream,
                               scratch_allocator);
  } else {
    executed = RunGemm(*config, lhs_data, rhs_data, output_data, stream);
  }

  if (!executed.ok()) return failure();

  return success();
}

static bool GemmBias(runtime::KernelContext* ctx, void** args, void** attrs) {
  static auto* handler =
      CustomCall::Bind("xla.gpu.gemm.bias")
          .UserData<const ServiceExecutableRunOptions*>()
          .UserData<const DebugOptions*>()
          .UserData<JitRtGemmConfigCache*>()
          .Arg<jitrt::StridedMemrefView>()  // lhs
          .Arg<jitrt::StridedMemrefView>()  // rhs
          .Arg<jitrt::StridedMemrefView>()  // bias
          .Arg<jitrt::StridedMemrefView>()  // out
          .Attr<int64_t>("algorithm")
          .Attr<double>("alpha_real")
          .Attr<double>("alpha_imag")
          .Attr<double>("beta")
          .Attr<ArrayRef<int64_t>>("lhs_batching_dimensions")
          .Attr<ArrayRef<int64_t>>("lhs_contracting_dimensions")
          .Attr<ArrayRef<int64_t>>("rhs_batching_dimensions")
          .Attr<ArrayRef<int64_t>>("rhs_contracting_dimensions")
          .Attr<int64_t>("uid")
          .To<RuntimeChecks()>(GemmBias::Handler())
          .release();

  return succeeded(handler->call(args, attrs, Executable::GetUserData(ctx)));
}

// -------------------------------------------------------------------------- //

namespace {
struct Infeed {
  LogicalResult operator()(const ServiceExecutableRunOptions* run_options,
                           CustomCall::RemainingArgs args,
                           StringRef config) const;
  static Infeed Handler() { return Infeed(); }
};
}  // namespace

LogicalResult Infeed::operator()(const ServiceExecutableRunOptions* run_options,
                                 CustomCall::RemainingArgs args,
                                 StringRef config) const {
  VLOG(3) << "Infeeding to GPU";

  se::Stream* stream = run_options->stream();
  ShapeTree<se::ScopedDeviceMemory<uint8_t>> source_buffers =
      GetOrCreateInfeedManager(stream->parent())->BlockingGetNextDestination();

  // Check that we have correct number of arguments.
  if (args.size() != source_buffers.leaf_count()) return failure();

  // TODO(ezhulenev): Report human-readable error messages through errors.
  size_t index = 0;
  for (auto& source : source_buffers.leaves()) {
    // Get the destination buffer.
    auto dest = args.get<jitrt::StridedMemrefView>(index);
    if (failed(dest)) return failure();

    // Get the source buffer shape.
    const Shape& source_shape =
        ShapeUtil::GetSubshape(source_buffers.shape(), source.first);

    // Check that destination shape matches the source shape.
    // TODO(ezhulenev): Report human-readable error similar to infeed_thunk.
    Shape dest_shape = ToShape(*dest);
    if (!ShapeUtil::Equal(dest_shape, source_shape)) return failure();

    se::DeviceMemoryBase dest_address = GetDeviceAddress(*dest);
    se::ScopedDeviceMemory<uint8_t>& buffer = source.second;
    stream->ThenMemcpy(&dest_address, *buffer.ptr(), buffer.ptr()->size());

    ++index;
  }

  // TODO(ezhulenev): Make this function async?
  Status block_status = stream->BlockHostUntilDone();
  if (!block_status.ok()) return failure();

  VLOG(3) << "Infeeding to GPU complete";

  return success();
}

static bool Infeed(runtime::KernelContext* ctx, void** args, void** attrs) {
  static auto* handler = CustomCall::Bind("xla.gpu.infeed")
                             .UserData<const ServiceExecutableRunOptions*>()
                             .Arg<CustomCall::RemainingArgs>()  // args
                             .Attr<StringRef>("config")
                             .To<RuntimeChecks()>(Infeed::Handler())
                             .release();

  return succeeded(handler->call(args, attrs, Executable::GetUserData(ctx)));
}

// -------------------------------------------------------------------------- //

namespace {
struct Outfeed {
  LogicalResult operator()(const ServiceExecutableRunOptions* run_options,
                           CustomCall::RemainingArgs args,
                           StringRef config) const;
  static Outfeed Handler() { return Outfeed(); }
};
}  // namespace

LogicalResult Outfeed::operator()(
    const ServiceExecutableRunOptions* run_options,
    CustomCall::RemainingArgs args, StringRef config) const {
  VLOG(3) << "Outfeeding from GPU";

  se::Stream* stream = run_options->stream();
  OutfeedManager* outfeed_manager = GetOrCreateOutfeedManager(stream->parent());
  ShapeTree<std::unique_ptr<OutfeedBuffer>>* dest_buffers =
      outfeed_manager->BlockingGetNextDestination();

  // Check that we have correct number of arguments.
  if (args.size() != dest_buffers->leaf_count()) return failure();

  size_t index = 0;
  for (auto& dest : dest_buffers->leaves()) {
    // Get the source buffer.
    auto source = args.get<jitrt::StridedMemrefView>(index);
    if (failed(source)) return failure();

    // Get the source buffer shape.
    const Shape& dest_shape =
        ShapeUtil::GetSubshape(dest_buffers->shape(), dest.first);

    // Check that destination shape matches the source shape.
    // TODO(ezhulenev): Report human-readable error similar to outfeed_thunk.
    Shape source_shape = ToShape(*source);
    if (!ShapeUtil::Equal(dest_shape, source_shape)) return failure();

    se::DeviceMemoryBase source_address = GetDeviceAddress(*source);
    std::unique_ptr<OutfeedBuffer>& buffer = dest.second;

    // Schedule the memory transfer.
    auto* dest_address = buffer->destination()->untyped_data();
    stream->ThenMemcpy(dest_address, source_address, buffer->length())
        .ThenDoHostCallback([&buffer]() { buffer->Done(); });

    ++index;
  }

  Status block_status = stream->BlockHostUntilDone();
  if (!block_status.ok()) return failure();

  VLOG(3) << "Outfeeding from GPU complete";

  return success();
}

static bool Outfeed(runtime::KernelContext* ctx, void** args, void** attrs) {
  static auto* handler = CustomCall::Bind("xla.gpu.outfeed")
                             .UserData<const ServiceExecutableRunOptions*>()
                             .Arg<CustomCall::RemainingArgs>()  // args
                             .Attr<StringRef>("config")
                             .To<RuntimeChecks()>(Outfeed::Handler())
                             .release();

  return succeeded(handler->call(args, attrs, Executable::GetUserData(ctx)));
}

// -------------------------------------------------------------------------- //

namespace {

enum class MemcpyDirection { kDeviceToDevice, kDeviceToHost, kHostToDevice };

template <MemcpyDirection direction>
struct Memcpy {
  LogicalResult operator()(const ServiceExecutableRunOptions* run_options,
                           jitrt::FlatMemrefView dst,
                           jitrt::FlatMemrefView src) const;
  static Memcpy Handler() { return Memcpy(); }
};
}  // namespace

template <MemcpyDirection direction>
LogicalResult Memcpy<direction>::operator()(
    const ServiceExecutableRunOptions* run_options, jitrt::FlatMemrefView dst,
    jitrt::FlatMemrefView src) const {
  se::Stream* stream = run_options->stream();

  if (dst.size_in_bytes != src.size_in_bytes) return failure();

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
    if (!st.ok()) return failure();
  }

  return success();
}

template <MemcpyDirection direction>
static bool MemcpyFn(runtime::KernelContext* ctx, void** args, void** attrs) {
  static auto* handler = CustomCall::Bind("xla.gpu.memcpy")
                             .UserData<const ServiceExecutableRunOptions*>()
                             .Arg<jitrt::FlatMemrefView>()  // dst
                             .Arg<jitrt::FlatMemrefView>()  // src
                             .To<RuntimeChecks()>(Memcpy<direction>::Handler())
                             .release();

  return succeeded(handler->call(args, attrs, Executable::GetUserData(ctx)));
}

// -------------------------------------------------------------------------- //

namespace {
struct Cholesky {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  LogicalResult operator()(const ServiceExecutableRunOptions* run_options,
                           const DebugOptions* debug_options,
                           jitrt::MemrefView operand, jitrt::MemrefView a,
                           jitrt::MemrefView workspace, jitrt::MemrefView info,
                           int64_t batch_size, int64_t n, int64_t uplo) const;
  static Cholesky Handler() { return Cholesky(); }
};
}  // namespace

LogicalResult Cholesky::operator()(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options, jitrt::MemrefView operand,
    jitrt::MemrefView a, jitrt::MemrefView workspace, jitrt::MemrefView info,
    int64_t batch_size, int64_t n, int64_t uplo) const {
  se::DeviceMemoryBase operand_buffer = GetDeviceAddress(operand);
  se::DeviceMemoryBase a_buffer = GetDeviceAddress(a);
  se::DeviceMemoryBase workspace_buffer = GetDeviceAddress(workspace);
  se::DeviceMemoryBase info_buffer = GetDeviceAddress(info);

  VLOG(3) << "Running Cholesky";
  se::Stream* stream = run_options->stream();

  // Copy operand to the a buffer if they are different.
  if (a.data != operand.data)
    stream->ThenMemcpy(&a_buffer, operand_buffer, operand_buffer.size());

  CholeskyParams params{
      n,        batch_size,       static_cast<se::blas::UpperLower>(uplo),
      a_buffer, workspace_buffer, info_buffer};
  auto executed = RunCholesky(xla::gpu::PtxOptsFromDebugOptions(*debug_options),
                              ToPrimitiveType(operand.dtype), &params, stream);
  if (!executed.ok()) return failure();

  return success();
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
                             .Attr<int64_t>("n")
                             .Attr<int64_t>("uplo")  // se::blas::UpperLower
                             .To<RuntimeChecks()>(Cholesky::Handler())
                             .release();

  return succeeded(handler->call(args, attrs, Executable::GetUserData(ctx)));
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

  LogicalResult operator()(const ServiceExecutableRunOptions* run_options,
                           CustomCall::RemainingArgs args,
                           StringRef call_target_name, int32_t api_version,
                           StringRef backend_config) const;
  static XlaCustomCall Handler() { return XlaCustomCall(); }
};
}  // namespace

LogicalResult XlaCustomCall::operator()(
    const ServiceExecutableRunOptions* run_options,
    CustomCall::RemainingArgs args, StringRef call_target_name,
    int32_t api_version, StringRef backend_config) const {
  // Find the Xla custom call handler.
  auto& platform_name = run_options->stream()->parent()->platform()->Name();
  void* call_target = CustomCallTargetRegistry::Global()->Lookup(
      call_target_name.str(), platform_name);
  if (!call_target) return failure();

  // Prepare pointers to buffers to pass to the Xla custom call handler.
  llvm::SmallVector<void*> buffers;
  for (unsigned i = 0; i < args.size(); ++i) {
    auto memref = args.get<jitrt::FlatMemrefView>(i);
    if (failed(memref)) return failure();

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

    return success();
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
      return failure();
    } else {
      return success();
    }
  }

  return failure();
}

static bool CustomCall(runtime::KernelContext* ctx, void** args, void** attrs) {
  static auto* handler = CustomCall::Bind("xla.gpu.memcpy")
                             .UserData<const ServiceExecutableRunOptions*>()
                             .Arg<jitrt::CustomCall::RemainingArgs>()  // args
                             .Attr<StringRef>("call_target_name")
                             .Attr<int32_t>("api_version")
                             .Attr<StringRef>("backend_config")
                             .To<RuntimeChecks()>(XlaCustomCall::Handler())
                             .release();

  return succeeded(handler->call(args, attrs, Executable::GetUserData(ctx)));
}

// ------------------------------------------------------------------------- //

namespace {
struct AllReduce {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  LogicalResult operator()(const ServiceExecutableRunOptions* run_options,
                           CustomCall::RemainingArgs args, int64_t group_mode,
                           int64_t op_id, int64_t reduction_kind,
                           ArrayRef<int64_t> replica_group_offsets,
                           ArrayRef<int64_t> replica_group_values) const;
  static AllReduce Handler() { return AllReduce(); }
};
}  // namespace

LogicalResult AllReduce::operator()(
    const ServiceExecutableRunOptions* run_options,
    CustomCall::RemainingArgs args, int64_t group_mode, int64_t op_id,
    int64_t reduction_kind, ArrayRef<int64_t> replica_group_offsets,
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
          .RemainingArgs()              // args
          .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
          .Attr<int64_t>("op_id")
          .Attr<int64_t>("reduction_kind")  // ReductionKind
          .Attr<ArrayRef<int64_t>>("replica_group_offsets")
          .Attr<ArrayRef<int64_t>>("replica_group_values")
          .To<RuntimeChecks()>(AllReduce::Handler())
          .release();

  return succeeded(handler->call(args, attrs, Executable::GetUserData(ctx)));
}

// -------------------------------------------------------------------------- //

namespace {
struct AllGather {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  LogicalResult operator()(const ServiceExecutableRunOptions* run_options,
                           CustomCall::RemainingArgs args, int64_t group_mode,
                           int64_t op_id,
                           ArrayRef<int64_t> replica_group_offsets,
                           ArrayRef<int64_t> replica_group_values) const;
  static AllGather Handler() { return AllGather(); }
};
}  // namespace

LogicalResult AllGather::operator()(
    const ServiceExecutableRunOptions* run_options,
    CustomCall::RemainingArgs args, int64_t group_mode, int64_t op_id,
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

  auto executed = RunAllGather(*device_buffers, *stream, **comm);
  if (!executed.ok()) return failure();

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
          .RemainingArgs()              // args
          .Attr<int64_t>("group_mode")  // CollectiveOpGroupMode
          .Attr<int64_t>("op_id")
          .Attr<ArrayRef<int64_t>>("replica_group_offsets")
          .Attr<ArrayRef<int64_t>>("replica_group_values")
          .To<RuntimeChecks()>(AllGather::Handler())
          .release();

  return succeeded(handler->call(args, attrs, Executable::GetUserData(ctx)));
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

  return succeeded(handler->call(args, attrs, Executable::GetUserData(ctx)));
}

// -------------------------------------------------------------------------- //

SymbolMap JitRtCustomCallsSymbolMap(MangleAndInterner mangle) {
  SymbolMap symbol_map;

  auto bind = [&](llvm::StringRef name, auto symbol_ptr) {
    symbol_map[mangle(name)] = llvm::JITEvaluatedSymbol(
        llvm::pointerToJITTargetAddress(symbol_ptr), llvm::JITSymbolFlags());
  };

  bind("xla.gpu.all_gather", &xla::gpu::AllGather);
  bind("xla.gpu.all_reduce", &xla::gpu::AllReduce);
  bind("xla.gpu.cholesky", &xla::gpu::Cholesky);
  bind("xla.gpu.func.launch", &xla::gpu::LaunchFunc);
  bind("xla.gpu.gemm", &xla::gpu::Gemm);
  bind("xla.gpu.gemm.bias", &xla::gpu::GemmBias);
  bind("xla.gpu.memcpy.d2d", &MemcpyFn<MemcpyDirection::kDeviceToDevice>);
  bind("xla.gpu.memcpy.h2d", &MemcpyFn<MemcpyDirection::kHostToDevice>);
  bind("xla.gpu.memcpy.d2h", &MemcpyFn<MemcpyDirection::kDeviceToHost>);
  bind("xla.gpu.infeed", &xla::gpu::Infeed);
  bind("xla.gpu.outfeed", &xla::gpu::Outfeed);
  bind("xla.gpu.replica_id", &xla::gpu::ReplicaId);
  bind("xla.gpu.custom_call", &xla::gpu::CustomCall);

  return symbol_map;
}

}  // namespace gpu
}  // namespace xla
