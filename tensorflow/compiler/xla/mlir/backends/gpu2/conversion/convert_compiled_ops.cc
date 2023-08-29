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

#include "tensorflow/compiler/xla/mlir/backends/gpu2/conversion/convert_compiled_ops.h"

#include <stddef.h>

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "iree-dialects/Dialect/Input/InputDialect.h"
#include "iree-dialects/Dialect/Input/InputOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/backends/gpu2/conversion/de_bufferization.h"
#include "tensorflow/compiler/xla/mlir/backends/gpu2/conversion/xla_gpu_api.h"
#include "tensorflow/compiler/xla/mlir/backends/gpu2/ir/xla_gpu_dialect.h"
#include "tensorflow/compiler/xla/mlir/backends/gpu2/ir/xla_gpu_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/xla/service/gpu/conditional_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/copy_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/kernel_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/gpu/while_thunk.h"

namespace xla::gpu {

namespace {
using namespace mlir;                 // NOLINT
using namespace mlir::iree_compiler;  // NOLINT

using arith::ConstantIndexOp;
using arith::ConstantIntOp;

//===----------------------------------------------------------------------===//
// Helper functions to work with ThunkSequence
//===----------------------------------------------------------------------===//

// A helper class to extract thunks compiled from the given operation. It is
// typically a combination of memory copy thunks plus device kernels. Memory
// copy operations initialize buffers, and always go before kernels.
template <typename OpTy>
struct CompiledOp {
  OpTy op;
  std::vector<std::unique_ptr<DeviceToDeviceCopyThunk>> memcpy;
  std::vector<std::unique_ptr<KernelThunk>> kernels;
};

// Extracts from a Thunk sequence thunks that are corresponding to the given
// operation. Some operations can be represented as multiple thunks.
ThunkSequence extractThunksForOp(ThunkSequence *from, Operation *op) {
  ThunkSequence thunks;

  for (std::unique_ptr<Thunk> &thunk : *from) {
    // If a thunk was already extracted earlier for some other operation.
    if (thunk == nullptr) continue;

    // Look for thunks in the while loop condition and body.
    if (thunk->kind() == Thunk::kWhile) {
      auto *while_thunk = static_cast<WhileThunk *>(thunk.get());

      for (auto &thunk : extractThunksForOp(
               &while_thunk->condition_thunk_sequence()->thunks(), op))
        thunks.push_back(std::move(thunk));

      for (auto &thunk : extractThunksForOp(
               &while_thunk->body_thunk_sequence()->thunks(), op))
        thunks.push_back(std::move(thunk));
    }

    // Look for thunks in the conditional thunk branches.
    if (thunk->kind() == Thunk::kConditional) {
      auto *cond_thunk = static_cast<ConditionalThunk *>(thunk.get());

      for (auto &branch : cond_thunk->branch_thunks()) {
        for (auto &thunk : extractThunksForOp(&branch->thunks(), op))
          thunks.push_back(std::move(thunk));
      }
    }

    if (thunk->op() == op) thunks.push_back(std::move(thunk));
  }

  return thunks;
}

// Extracts compiled operation from the ThunkSequence if it is available.
template <typename T>
FailureOr<CompiledOp<T>> extractCompiledOp(
    T op, ThunkSequence *thunk_sequence, ConversionPatternRewriter &rewriter) {
  CompiledOp<T> compiled_op;
  compiled_op.op = op;

  // If thunk sequence is not available we just pass nullptr and return fake
  // kernel launch parameters later.
  if (thunk_sequence == nullptr) return compiled_op;

  // Otherwise steal thunks implementing given fusion operations.
  ThunkSequence thunks = extractThunksForOp(thunk_sequence, op);

  for (std::unique_ptr<Thunk> &thunk : thunks) {
    if (thunk->kind() == Thunk::Kind::kCopy) {
      assert(compiled_op.kernels.empty() &&
             "copy after kernel is not suppported");
      compiled_op.memcpy.push_back(std::unique_ptr<DeviceToDeviceCopyThunk>(
          static_cast<DeviceToDeviceCopyThunk *>(thunk.release())));
      continue;
    }

    if (thunk->kind() == Thunk::Kind::kKernel) {
      compiled_op.kernels.push_back(std::unique_ptr<KernelThunk>(
          static_cast<KernelThunk *>(thunk.release())));
      continue;
    }

    return rewriter.notifyMatchFailure(op, "unsupported thunk kind");
  }

  assert(!compiled_op.memcpy.empty() || !compiled_op.kernels.empty());
  return compiled_op;
}

//===----------------------------------------------------------------------===//
// Helper function to infer IREE dispatch ABI from kernel thunk
//===----------------------------------------------------------------------===//

// A pair of original buffer arguments that we track for tying together
// inplace buffer updates, and tensor arguments passed to iree_input
// dispatch.
using DispatchArguments = std::pair<SmallVector<TypedValue<MemRefType>>,
                                    SmallVector<TypedValue<TensorType>>>;

using KernelLaunchParams = std::pair<std::string, LaunchDimensions>;

SmallVector<int64_t> getTiedOperands(lmhlo::FusionOp);
SmallVector<int64_t> getTiedOperands(lmhlo::SortOp);

IREE::Input::PipelineLayoutAttr getPipelineLayout(lmhlo::FusionOp);
IREE::Input::PipelineLayoutAttr getPipelineLayout(lmhlo::SortOp);

DispatchArguments getDispatchArguments(lmhlo::FusionOp, DeBufferization &);
DispatchArguments getDispatchArguments(lmhlo::SortOp, DeBufferization &);

// In tests when we do not have ThunkSequence we create unique exported kernels
// names by incrementing a global counter.
std::atomic<int64_t> unknown_kernel_counter{0};

KernelLaunchParams getKernelLaunchParams(KernelThunk *kernel) {
  // Return fake kernel launch parameters when we do not have thunk sequence. We
  // use it only for writing MLIR tests when we do not have thunks.
  if (kernel == nullptr) {
    return KernelLaunchParams(
        "unknown." + std::to_string(unknown_kernel_counter.fetch_add(1)),
        LaunchDimensions(1, 1));
  }

  return KernelLaunchParams(kernel->kernel_name(), kernel->launch_dimensions());
}

DispatchArguments getDispatchArguments(KernelThunk *kernel,
                                       DeBufferization &state) {
  DispatchArguments args;

  auto *block = kernel->op()->getBlock();

  for (auto arg : kernel->values()) {
    args.first.push_back(cast<TypedValue<MemRefType>>(arg));
    args.second.push_back(state.remapped[block][args.first.back()]);
    assert(args.second.back() && "missing memref to tensor mapping");
  }

  return args;
}

SmallVector<int64_t> getTiedOperands(KernelThunk *kernel) {
  SmallVector<int64_t> tied_operands;
  for (size_t i = 0; i < kernel->arguments().size(); ++i) {
    if (kernel->written()[i]) tied_operands.push_back(i);
  }
  return tied_operands;
}

// Returns compiled op pipeline layout (ABI) inferred from the kernel thunk.
IREE::Input::PipelineLayoutAttr getPipelineLayout(MLIRContext *ctx,
                                                  KernelThunk *kernel) {
  SmallVector<IREE::Input::DescriptorSetBindingAttr> bindings;

  for (size_t i = 0; i < kernel->arguments().size(); ++i) {
    std::optional<IREE::Input::DescriptorFlags> flags;
    if (!kernel->written()[i]) flags = IREE::Input::DescriptorFlags::ReadOnly;

    bindings.push_back(IREE::Input::DescriptorSetBindingAttr::get(
        ctx, /*ordinal=*/bindings.size(),
        IREE::Input::DescriptorType::StorageBuffer, flags));
  }

  return IREE::Input::PipelineLayoutAttr::get(
      ctx, /*pushConstants=*/0,
      IREE::Input::DescriptorSetLayoutAttr::get(ctx, /*ordinal=*/0, bindings));
}

// Returns pipeline layout with given number of arguments and results buffers.
IREE::Input::PipelineLayoutAttr getPipelineLayout(MLIRContext *ctx,
                                                  size_t n_args,
                                                  size_t n_rets) {
  SmallVector<IREE::Input::DescriptorSetBindingAttr> bindings;

  for (size_t i = 0; i < n_args; ++i) {
    bindings.push_back(IREE::Input::DescriptorSetBindingAttr::get(
        ctx, /*ordinal=*/bindings.size(),
        IREE::Input::DescriptorType::StorageBuffer,
        IREE::Input::DescriptorFlags::ReadOnly));
  }

  for (size_t i = 0; i < n_rets; ++i) {
    bindings.push_back(IREE::Input::DescriptorSetBindingAttr::get(
        ctx, /*ordinal=*/bindings.size(),
        IREE::Input::DescriptorType::StorageBuffer, std::nullopt));
  }

  return IREE::Input::PipelineLayoutAttr::get(
      ctx, /*pushConstants=*/0,
      IREE::Input::DescriptorSetLayoutAttr::get(ctx, /*ordinal=*/0, bindings));
}

template <typename OpTy>
DispatchArguments getDispatchArguments(OpTy op, KernelThunk *kernel,
                                       DeBufferization &state) {
  return kernel ? getDispatchArguments(kernel, state)
                : getDispatchArguments(op, state);
}

template <typename OpTy>
SmallVector<int64_t> getTiedOperands(OpTy op, KernelThunk *kernel) {
  return kernel ? getTiedOperands(kernel) : getTiedOperands(op);
}

template <typename OpTy>
IREE::Input::PipelineLayoutAttr getPipelineLayout(OpTy op,
                                                  KernelThunk *kernel) {
  return kernel ? getPipelineLayout(op.getContext(), kernel)
                : getPipelineLayout(op);
}

//===----------------------------------------------------------------------===//
// Converts compiled op to an iree_input.dispatch operation
//===----------------------------------------------------------------------===//

// Keep a shared cache of optimization barriers between all compiled operation
// lowerings to minimize the number of created operations.
class OptimizationBarriers {
 public:
  struct Barrier {
    SmallVector<Value> values;
    IREE::Input::OptimizationBarrierOp optimization_barrier;
  };

  // Materializes values as constants at the top of the parent function entry
  // block and wraps them into optimization barrier.
  Barrier &getOrCreate(ImplicitLocOpBuilder &b, ArrayRef<int64_t> values);

 private:
  using Key = std::pair<func::FuncOp, ArrayAttr>;
  llvm::DenseMap<Key, Barrier> barriers_;
};

OptimizationBarriers::Barrier &OptimizationBarriers::getOrCreate(
    ImplicitLocOpBuilder &b, ArrayRef<int64_t> values) {
  Operation *parent = b.getInsertionBlock()->getParentOp();

  auto func = dyn_cast<func::FuncOp>(parent);
  if (!func) func = parent->getParentOfType<func::FuncOp>();

  auto &barrier = barriers_[std::make_pair(func, b.getIndexArrayAttr(values))];
  if (barrier.optimization_barrier) return barrier;

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(&func.front());

  barrier.values = llvm::to_vector(llvm::map_range(values, [&](int64_t value) {
    return b.create<arith::ConstantIndexOp>(value).getResult();
  }));
  barrier.optimization_barrier =
      b.create<IREE::Input::OptimizationBarrierOp>(barrier.values);

  return barrier;
}

template <typename OpTy>
struct ConvertCompiledOpToHal : public OpConversionPattern<OpTy> {
  using OpAdaptor = typename OpConversionPattern<OpTy>::OpAdaptor;

  ConvertCompiledOpToHal(TypeConverter &converter, MLIRContext *ctx,
                         IREE::Input::ExecutableSourceOp executable_source,
                         ThunkSequence *thunk_sequence, DeBufferization &state,
                         std::shared_ptr<int64_t> ordinal,
                         std::shared_ptr<OptimizationBarriers> barriers)
      : OpConversionPattern<OpTy>(converter, ctx),
        executable_source(executable_source.getSymNameAttr()),
        executable_source_body(&executable_source.getBody().front()),
        thunk_sequence(thunk_sequence),
        state(state),
        ordinal(std::move(ordinal)),
        barriers(std::move(barriers)) {}

  LogicalResult matchAndRewrite(
      OpTy op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override;

  StringAttr executable_source;
  Block *executable_source_body;
  ThunkSequence *thunk_sequence;
  DeBufferization &state;
  std::shared_ptr<int64_t> ordinal;
  std::shared_ptr<OptimizationBarriers> barriers;

  // Keep a mapping from a kernel name to exported function declaration.
  mutable llvm::StringMap<IREE::Input::ExecutableExportOp> exported;
};

template <typename OpTy>
LogicalResult ConvertCompiledOpToHal<OpTy>::matchAndRewrite(
    OpTy op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);

  auto *block = op->getBlock();

  // Extract compiled operation from the thunk sequence.
  auto compiled_op = extractCompiledOp(op, thunk_sequence, rewriter);
  if (failed(compiled_op))
    return rewriter.notifyMatchFailure(
        op, "failed to extract device compilation result for an operation");

  // Handle copy operations first, before handling kernel launch.
  for (auto &copy : compiled_op->memcpy) {
    auto src_memref = cast<TypedValue<MemRefType>>(copy->source_value());
    auto dst_memref = cast<TypedValue<MemRefType>>(copy->destination_value());

    auto src = state.remapped[block][stripReinterpretCast(src_memref)];
    auto dst = state.remapped[block][stripReinterpretCast(dst_memref)];

    assert(src && "unknown mapping from `src` memref to a tensor");
    assert(dst && "unknown mapping from `dst` memref to a tensor");

    auto rank = dst.getType().getRank();

    // Reshape src tensor to a dynamic tensor to prevent folding at Flow level.
    // TODO(ezhulenev): Find a solution that does not rely on compiler tricks.
    auto dyn_tensor =
        RankedTensorType::get(SmallVector<int64_t>(rank, ShapedType::kDynamic),
                              dst.getType().getElementType());
    auto &[dims, dims_barrier] =
        barriers->getOrCreate(b, dst.getType().getShape());

    Value dyn_src = b.create<IREE::Input::TensorReshapeOp>(
        dyn_tensor, src, /*source_dims=*/ValueRange(),
        /*result_dims=*/dims_barrier.getResults());

    // Update dst tensor with src.
    SmallVector<Value> start_indices(rank, b.create<arith::ConstantIndexOp>(0));
    Value updated = b.create<IREE::Input::TensorUpdateOp>(
        dst, ValueRange(), start_indices, dyn_src, dims);

    state.remapped[block][dst_memref] = cast<TypedValue<TensorType>>(updated);
  }

  // Compiled operation was a plain copy.
  if (thunk_sequence && compiled_op->kernels.empty()) {
    rewriter.eraseOp(op);
    return success();
  }

  SmallVector<KernelThunk *> kernels = llvm::to_vector(llvm::map_range(
      compiled_op->kernels, [&](auto &kernel) { return kernel.get(); }));
  // Always add a fake kernel if we are running without thunk sequence.
  if (!thunk_sequence) kernels.push_back(nullptr);

  // Dispatch all kernels defined by thunks.
  for (KernelThunk *kernel : kernels) {
    // Get kernel launch parameters from a kernel thunk.
    auto [kernel_name, dims] = getKernelLaunchParams(kernel);

    SmallVector<int64_t> workgroup_size = {
        dims.thread_counts_per_block().x,
        dims.thread_counts_per_block().y,
        dims.thread_counts_per_block().z,
    };

    SmallVector<int64_t> workload_size = {
        dims.block_counts().x,
        dims.block_counts().y,
        dims.block_counts().z,
    };

    int64_t shmem = dims.SharedMemBytes();

    // Pop trailing ones from workload sizes to keep IR small.
    while (workload_size.size() > 1 && workload_size.back() == 1)
      workload_size.pop_back();

    // Create `iree_input.executable.export` operation to export device
    // function.
    b.setInsertionPoint(executable_source_body->getTerminator());
    auto &executable_export = exported[kernel_name];
    if (!executable_export) {
      executable_export = b.create<IREE::Input::ExecutableExportOp>(
          /*sym_name=*/b.getStringAttr(kernel_name),
          /*ordinal=*/b.getIndexAttr((*ordinal)++),
          /*layout=*/getPipelineLayout(op, kernel),
          /*workgroup_size=*/b.getIndexArrayAttr(workgroup_size),
          /*subgroup_size=*/nullptr,
          /*workgroup_local_memory=*/shmem ? b.getIndexAttr(shmem) : nullptr);
    }

    // Replace `lmhlo.fusion` with a `iree_input.dispatch` operation.
    b.setInsertionPoint(op);

    // Materialize workload size as constants in the IR.
    SmallVector<Value> workload = llvm::to_vector(
        llvm::map_range(workload_size, [&](int64_t size) -> Value {
          return b.create<arith::ConstantIndexOp>(size);
        }));

    auto dispatch_args = getDispatchArguments(op, kernel, state);
    auto &memrefs = dispatch_args.first;
    auto &tensors = dispatch_args.second;

    // Prepare tied operands and corresponding result types.
    SmallVector<int64_t> tied_operands = getTiedOperands(op, kernel);
    SmallVector<Type> results = llvm::to_vector(llvm::map_range(
        tied_operands,
        [&](int64_t idx) -> Type { return tensors[idx].getType(); }));

    SmallVector<Value> tensor_vs = llvm::to_vector(
        llvm::map_range(tensors, [&](auto tensor) -> Value { return tensor; }));

    auto dispatch = b.create<IREE::Input::DispatchOp>(
        executable_export, workload, results,
        /*resultDims=*/ValueRange(), tensor_vs,
        /*argumentDims=*/ValueRange(), b.getIndexArrayAttr(tied_operands));

    // Keep track of all tensors updated inplace.
    for (auto result : llvm::enumerate(dispatch.getResults())) {
      auto arg = memrefs[tied_operands[result.index()]];
      state.remapped[block][arg] = cast<TypedValue<TensorType>>(result.value());
    }
  }

  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Converts compiled op to an XLA:GPU kernel dispatch API call
//===----------------------------------------------------------------------===//

TypedValue<ExecutionContextType> getExecutionContext(Operation *op) {
  auto func = op->getParentOfType<func::FuncOp>();
  return func.getArguments().front().cast<TypedValue<ExecutionContextType>>();
}

template <typename OpTy>
struct ConvertCompiledOpToApiCall : public OpConversionPattern<OpTy> {
  using OpAdaptor = typename OpConversionPattern<OpTy>::OpAdaptor;

  ConvertCompiledOpToApiCall(TypeConverter &converter, MLIRContext *ctx,
                             ThunkSequence *thunk_sequence,
                             DeBufferization &state, XlaGpuApi &api,
                             XlaGpuGraphs &graphs)
      : OpConversionPattern<OpTy>(converter, ctx),
        thunk_sequence(thunk_sequence),
        state(state),
        api(api),
        graphs(graphs) {}

  LogicalResult matchAndRewrite(
      OpTy op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override;

  // TODO(ezhulenev): This is a very conservative dependency tracking that
  // adds edges between all operations that touch the same buffers. We need to
  // track reads and writes separately and allow concurrent reads.

  void updateGraphDependencies(TypedValue<GraphNodeType> node,
                               ArrayRef<TypedValue<TensorType>> args,
                               SmallVector<int64_t> tied_operands) const {
    Block *block = node.getDefiningOp()->getBlock();
    for (auto arg : args) {
      graphs.dependency[block][arg] = node;
    }
  }

  SmallVector<TypedValue<GraphNodeType>> getGraphDependencies(
      Block *block, ArrayRef<TypedValue<TensorType>> args,
      SmallVector<int64_t> tied_operands) const {
    SetVector<TypedValue<GraphNodeType>> deps;
    for (auto &tensor : args) {
      auto it = graphs.dependency[block].find(tensor);
      if (it != graphs.dependency[block].end()) deps.insert(it->second);
    }
    return deps.takeVector();
  }

  ThunkSequence *thunk_sequence;
  DeBufferization &state;
  XlaGpuApi &api;
  XlaGpuGraphs &graphs;
};

template <typename OpTy>
LogicalResult ConvertCompiledOpToApiCall<OpTy>::matchAndRewrite(
    OpTy op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);

  auto *block = op->getBlock();
  auto module = op->template getParentOfType<ModuleOp>();

  // Detect if we are inside a graph dispatch region and we are building a graph
  // construction function.
  auto dispatch = op->template getParentOfType<GraphDispatchOp>();
  TypedValue<GraphType> graph = dispatch ? dispatch.getGraph() : nullptr;

  // Extract compiled operation from the thunk sequence.
  auto compiled_op = extractCompiledOp(op, thunk_sequence, rewriter);
  if (failed(compiled_op))
    return rewriter.notifyMatchFailure(
        op, "failed to extract device compilation result for an operation");

  // Handle copy operations first, before handling kernel launch.
  for (auto &copy : compiled_op->memcpy) {
    auto src_memref = cast<TypedValue<MemRefType>>(copy->source_value());
    auto dst_memref = cast<TypedValue<MemRefType>>(copy->destination_value());

    auto src = state.remapped[block][stripReinterpretCast(src_memref)];
    auto dst = state.remapped[block][stripReinterpretCast(dst_memref)];

    assert(src && "unknown mapping from `src` memref to a tensor");
    assert(dst && "unknown mapping from `dst` memref to a tensor");

    auto src_view = api.getBufferView(b, src);
    auto dst_view = api.getBufferView(b, dst);
    SmallVector<Value> args = {getExecutionContext(op), dst_view, src_view};

    // If we are inside a graph dispatch region, we convert memory copy
    // operation to a memory copy node.
    if (graph) {
      // These are the nodes that previously updated dispatch arguments, we need
      // to add them to a set of dependencies to build a correct DAG.
      Value dependencies = api.getGraphNodeList(
          b, getGraphDependencies(block, {dst, src}, /*tied_operands=*/{0}));

      // Add additional arguments required by node building API.
      args.insert(args.begin() + 1, {graph, dependencies});

      func::FuncOp create_node = api.getCreateD2DMemcpyNode(b, module);
      Value result = b.create<func::CallOp>(create_node.getSymName(),
                                            create_node.getResultTypes(), args)
                         .getResult(0);

      // Update dependencies to track updated dst tensor.
      updateGraphDependencies(cast<TypedValue<GraphNodeType>>(result),
                              {dst, src}, /*tied_operands=*/{0});
    }

    // For regular regions we simply dispatch the kernel using API call.
    if (!graph) {
      func::FuncOp memcpy = api.getD2DMemcpy(b, module);
      // TODO(ezhulenev): Should we import buffer view back and update
      // remapping?
      b.create<func::CallOp>(memcpy.getSymName(), TypeRange(), args);
    }
  }

  // Compiled operation was a plain copy.
  if (thunk_sequence && compiled_op->kernels.empty()) {
    rewriter.eraseOp(op);
    return success();
  }

  SmallVector<KernelThunk *> kernels = llvm::to_vector(llvm::map_range(
      compiled_op->kernels, [&](auto &kernel) { return kernel.get(); }));
  // Always add a fake kernel if we are running without thunk sequence.
  if (!thunk_sequence) kernels.push_back(nullptr);

  // Dispatch all kernels defined by thunks.
  for (KernelThunk *kernel : kernels) {
    // Get kernel launch parameters from a compiled fusion.
    auto launch_params = getKernelLaunchParams(kernel);
    auto [kernel_name, dims] = launch_params;

    // Create XLA:GPU device kernel (it will own loaded PTX/CUBIN at run time).
    func::FuncOp create_kernel = api.getCreateKernel(b, module);
    auto kernel_type = b.getType<KernelType>();

    auto initializer = [&](ImplicitLocOpBuilder &initializer_builder) -> Value {
      auto [kernel_name, dims] = launch_params;  // capture requires C++20

      Value name = b.create<IREE::Input::ByteBufferConstantOp>(
          b.getType<IREE::Input::ByteBufferType>(),
          b.getStringAttr("kernel_name"), kernel_name,
          /*alignment=*/nullptr, /*mime_type=*/nullptr);
      Value shmem = b.create<ConstantIntOp>(dims.SharedMemBytes(), 32);

      return initializer_builder
          .create<func::CallOp>(api.getCreateKernel(b, module).getSymName(),
                                kernel_type, ValueRange({name, shmem}))
          .getResult(0);
    };

    // Keep loaded device kernel as a module global.
    IREE::Input::GlobalOp kernel_global = api.getOrCreateGlobal(
        "__xla_gpu_kernel." + kernel_name, kernel_type, module, b, initializer);
    auto loaded_kernel = api.loadGlobal<KernelType>(b, kernel_global);

    // Prepare arguments for kernel dispatch.
    SmallVector<Value> workgroup_size = {
        b.create<ConstantIntOp>(dims.thread_counts_per_block().x, 32),
        b.create<ConstantIntOp>(dims.thread_counts_per_block().y, 32),
        b.create<ConstantIntOp>(dims.thread_counts_per_block().z, 32),
    };

    SmallVector<Value> workload_size = {
        b.create<ConstantIntOp>(dims.block_counts().x, 32),
        b.create<ConstantIntOp>(dims.block_counts().y, 32),
        b.create<ConstantIntOp>(dims.block_counts().z, 32),
    };

    auto dispatch_args = getDispatchArguments(op, kernel, state);
    auto &tensors = dispatch_args.second;

    Value buffer_views = api.getBufferViewList(b, tensors);

    // Prepare arguments for the kernel dispatch/create API call.
    SmallVector<Value> args = {getExecutionContext(op), loaded_kernel,
                               buffer_views};
    args.append(workgroup_size.begin(), workgroup_size.end());
    args.append(workload_size.begin(), workload_size.end());

    // If we are inside a graph dispatch region, we convert compiled operation
    // to a kernel node with explicit dependencies.
    if (graph) {
      auto tied_operands = getTiedOperands(op, kernel);

      // These are the nodes that previously updated dispatch arguments, we need
      // to add them to a set of dependencies to build a correct DAG.
      Value dependencies = api.getGraphNodeList(
          b, getGraphDependencies(block, tensors, tied_operands));

      // Add additional arguments required by node building API.
      args.insert(args.begin() + 1, {graph, dependencies});

      func::FuncOp create_node = api.getCreateKernelNode(b, module);
      Value result = b.create<func::CallOp>(create_node.getSymName(),
                                            create_node.getResultTypes(), args)
                         .getResult(0);

      // Update dependencies to track all updated tensors.
      updateGraphDependencies(cast<TypedValue<GraphNodeType>>(result), tensors,
                              tied_operands);
    }

    // For regular regions we simply dispatch the kernel using API call.
    if (!graph) {
      func::FuncOp dispatch_kernel = api.getDispatchKernel(b, module);
      // TODO(ezhulenev): Should we import buffer view back and update
      // remapping?
      b.create<func::CallOp>(dispatch_kernel.getSymName(),
                             dispatch_kernel.getResultTypes(), args);
    }
  }

  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Converts lmhlo.fusion op to HAL / XLA:GPU runtime
//===----------------------------------------------------------------------===//

using ConvertFusionOpToHal = ConvertCompiledOpToHal<lmhlo::FusionOp>;
using ConvertFusionOpToApiCall = ConvertCompiledOpToApiCall<lmhlo::FusionOp>;

// Returns Fusion kernel pipeline layout (ABI) inferred from the fusion
// operation body looking at tensor<->memref conversions.
IREE::Input::PipelineLayoutAttr getPipelineLayout(lmhlo::FusionOp op) {
  auto *body = op.getBody(0);

  auto all = [](auto) { return true; };

  auto n_args = llvm::count_if(body->getOps<bufferization::ToTensorOp>(), all);
  auto n_rets = llvm::count_if(body->getOps<memref::TensorStoreOp>(), all);

  return getPipelineLayout(op.getContext(), n_args, n_rets);
}

DispatchArguments getDispatchArguments(lmhlo::FusionOp op,
                                       DeBufferization &state) {
  DispatchArguments args;

  auto *block = op->getBlock();
  auto *body = op.getBody();

  for (auto to_tensor : body->getOps<bufferization::ToTensorOp>()) {
    args.first.push_back(stripReinterpretCast(to_tensor.getMemref()));
    args.second.push_back(state.remapped[block][args.first.back()]);
    assert(args.second.back() && "missing memref to tensor mapping");
  }

  for (auto store : body->getOps<memref::TensorStoreOp>()) {
    args.first.push_back(stripReinterpretCast(store.getMemref()));
    args.second.push_back(state.remapped[block][args.first.back()]);
    assert(args.second.back() && "missing memref to tensor mapping");
  }

  return args;
}

SmallVector<int64_t> getTiedOperands(lmhlo::FusionOp op) {
  SmallVector<int64_t> tied_operands;
  auto *body = op.getBody(0);

  size_t index = 0;

  // Skip regular arguments.
  llvm::for_each(body->getOps<bufferization::ToTensorOp>(),
                 [&](bufferization::ToTensorOp) { index++; });

  // Tie destination-passing style arguments to results.
  llvm::for_each(
      body->getOps<memref::TensorStoreOp>(),
      [&](memref::TensorStoreOp) { tied_operands.push_back(index++); });

  return tied_operands;
}

//===----------------------------------------------------------------------===//
// Converts lmhlo.sort op to to HAL / XLA:GPU runtime
//===----------------------------------------------------------------------===//

using ConvertSortOpToHal = ConvertCompiledOpToHal<lmhlo::SortOp>;
using ConvertSortOpToApiCall = ConvertCompiledOpToApiCall<lmhlo::SortOp>;

IREE::Input::PipelineLayoutAttr getPipelineLayout(lmhlo::SortOp op) {
  auto n_args = op.getInputs().size();
  auto n_rets = op.getOutput().size();
  return getPipelineLayout(op.getContext(), n_args, n_rets);
}

DispatchArguments getDispatchArguments(lmhlo::SortOp op,
                                       DeBufferization &state) {
  DispatchArguments args;

  auto *block = op->getBlock();

  for (auto input : op.getInputs()) {
    args.first.push_back(cast<TypedValue<MemRefType>>(input));
    args.second.push_back(state.remapped[block][args.first.back()]);
    assert(args.second.back() && "missing memref to tensor mapping");
  }

  for (auto output : op.getOutput()) {
    args.first.push_back(cast<TypedValue<MemRefType>>(output));
    args.second.push_back(state.remapped[block][args.first.back()]);
    assert(args.second.back() && "missing memref to tensor mapping");
  }

  return args;
}

SmallVector<int64_t> getTiedOperands(lmhlo::SortOp op) {
  SmallVector<int64_t> tied_operands(op.getOutput().size());
  std::iota(tied_operands.begin(), tied_operands.end(), op.getInputs().size());
  return tied_operands;
}

//===----------------------------------------------------------------------===//
// Converts lmhlo.terminator inside a top level function to a func.return
//===----------------------------------------------------------------------===//

struct TerminatorOpLowering : public OpConversionPattern<lmhlo::TerminatorOp> {
  TerminatorOpLowering(TypeConverter &converter, MLIRContext *ctx,
                       DeBufferization &state, bool add_opt_barrier = true)
      : OpConversionPattern(converter, ctx),
        state(state),
        add_opt_barrier(add_opt_barrier) {}

  LogicalResult matchAndRewrite(
      lmhlo::TerminatorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto *block = op->getBlock();

    auto func = dyn_cast<func::FuncOp>(op->getParentOp());
    if (!func) return rewriter.notifyMatchFailure(op, "unsupported terminator");

    // When lowering for StreamExecutor backend we don't need to add
    // optimization barriers because API calls mutate buffers in place and
    // external functions calls are not dead-code-eliminated.
    if (!add_opt_barrier) {
      rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
      return success();
    }

    // Collect block arguments corresponding to output buffers.
    SmallVector<BlockArgument> results;
    for (unsigned i = 0; i < func.getFunctionType().getNumInputs(); ++i) {
      if (func.getArgAttr(i, "lmhlo.output_index"))
        results.push_back(func.getArgument(i));
    }

    // Find the latest tensors sharing underlying storage with destination
    // passing style arguments.
    llvm::SetVector<Value> updated_tensors;
    for (auto result : results) {
      for (auto memref : state.imported[result]) {
        // Check that we have tensors imported from a memref.
        auto it = state.remapped[block].find(memref);
        if (it != state.remapped[block].end() && it->second.use_empty()) {
          updated_tensors.insert(it->second);
        }
      }
    }

    // Insert optimization barrier to guarantee that all inplace tensor updates
    // threaded through dispatches and custom calls via tied operands will not
    // be dead-code-eliminated because dispatches are pure operations.
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    for (auto updated_tensor : updated_tensors) {
      b.create<IREE::Input::OptimizationBarrierOp>(updated_tensor);
    }

    rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
    return success();
  }

  DeBufferization &state;
  bool add_opt_barrier;
};

}  // namespace

//===----------------------------------------------------------------------===//

void populateCompiledOpsConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &converter,
    IREE::Input::ExecutableSourceOp executable_source,
    ThunkSequence *thunk_sequence, DeBufferization &state) {
  auto *ctx = patterns.getContext();
  patterns.insert<ConvertFusionOpToHal, ConvertSortOpToHal>(
      converter, ctx, executable_source, thunk_sequence, state,
      /*ordinal=*/std::make_shared<int64_t>(0),
      std::make_shared<OptimizationBarriers>());
  patterns.insert<TerminatorOpLowering>(converter, ctx, state);
}

void populateCompiledOpsConversionPatterns(mlir::RewritePatternSet &patterns,
                                           mlir::TypeConverter &converter,
                                           ThunkSequence *thunk_sequence,
                                           DeBufferization &state,
                                           XlaGpuApi &api,
                                           XlaGpuGraphs &graphs) {
  auto *ctx = patterns.getContext();
  patterns.insert<ConvertFusionOpToApiCall, ConvertSortOpToApiCall>(
      converter, ctx, thunk_sequence, state, api, graphs);
  patterns.insert<TerminatorOpLowering>(converter, ctx, state,
                                        /*add_opt_barrier=*/false);
}

}  // namespace xla::gpu
