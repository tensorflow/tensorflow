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

#include "tensorflow/compiler/xla/mlir/backends/openxla/conversion/convert_compiled_ops.h"

#include <stddef.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "third_party/iree/llvm-external-projects/iree-dialects/include/iree-dialects/Dialect/Input/InputOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/backends/openxla/conversion/de_bufferization.h"
#include "tensorflow/compiler/xla/mlir_hlo/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/xla/service/gpu/copy_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/kernel_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/gpu/while_thunk.h"

namespace xla::gpu {

namespace {
using namespace mlir;                 // NOLINT
using namespace mlir::iree_compiler;  // NOLINT

//===----------------------------------------------------------------------===//
// Helper functions to work with ThunkSequence
//===----------------------------------------------------------------------===//

// A helper class to extract thunks compiled from the given operation. It is
// typically a combination of memory copy thunks plus a single device kernel.
template <typename OpTy>
struct CompiledOp {
  OpTy op;
  std::vector<std::unique_ptr<DeviceToDeviceCopyThunk>> memcpy;
  std::unique_ptr<KernelThunk> kernel;
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
      assert(!compiled_op.kernel && "copy after kernel is not suppported");
      compiled_op.memcpy.push_back(std::unique_ptr<DeviceToDeviceCopyThunk>(
          static_cast<DeviceToDeviceCopyThunk *>(thunk.release())));
      continue;
    }

    if (thunk->kind() == Thunk::Kind::kKernel) {
      assert(!compiled_op.kernel && "multiple kernels are not supported");
      compiled_op.kernel = std::unique_ptr<KernelThunk>(
          static_cast<KernelThunk *>(thunk.release()));
      continue;
    }

    return rewriter.notifyMatchFailure(op, "unsupported thunk kind");
  }

  assert(!compiled_op.memcpy.empty() || compiled_op.kernel);
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

template <typename T>
KernelLaunchParams getKernelLaunchParams(CompiledOp<T> &compiled) {
  // Return fake kernel launch parameters when we do not have thunk sequence. We
  // use it only for writing MLIR tests when we do not have thunks.
  if (!compiled.kernel) {
    return KernelLaunchParams(
        "unknown." + std::to_string(unknown_kernel_counter.fetch_add(1)),
        LaunchDimensions(1, 1));
  }

  return KernelLaunchParams(compiled.kernel->kernel_name(),
                            compiled.kernel->launch_dimensions());
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

template <typename T>
DispatchArguments getDispatchArguments(CompiledOp<T> &op,
                                       DeBufferization &state) {
  return op.kernel ? getDispatchArguments(op.kernel.get(), state)
                   : getDispatchArguments(op.op, state);
}

template <typename T>
SmallVector<int64_t> getTiedOperands(CompiledOp<T> &op) {
  return op.kernel ? getTiedOperands(op.kernel.get()) : getTiedOperands(op.op);
}

template <typename T>
IREE::Input::PipelineLayoutAttr getPipelineLayout(CompiledOp<T> &op) {
  return op.kernel ? getPipelineLayout(op.op.getContext(), op.kernel.get())
                   : getPipelineLayout(op.op);
}

//===----------------------------------------------------------------------===//
// Converts compiled op to an iree_input.dispatch operation
//===----------------------------------------------------------------------===//

template <typename OpTy>
struct ConvertCompiledOp : public OpConversionPattern<OpTy> {
  using OpAdaptor = typename OpConversionPattern<OpTy>::OpAdaptor;

  ConvertCompiledOp(TypeConverter &converter, MLIRContext *ctx,
                    IREE::Input::ExecutableSourceOp executable_source,
                    ThunkSequence *thunk_sequence,
                    std::shared_ptr<DeBufferization> state,
                    std::shared_ptr<int64_t> ordinal)
      : OpConversionPattern<OpTy>(converter, ctx),
        executable_source(executable_source.getSymNameAttr()),
        executable_source_body(&executable_source.getBody().front()),
        thunk_sequence(thunk_sequence),
        state(std::move(state)),
        ordinal(std::move(ordinal)) {}

  LogicalResult matchAndRewrite(
      OpTy op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override;

  StringAttr executable_source;
  Block *executable_source_body;
  ThunkSequence *thunk_sequence;
  std::shared_ptr<DeBufferization> state;
  std::shared_ptr<int64_t> ordinal;

  // Keep a mapping from a kernel name to exported function declaration.
  mutable llvm::StringMap<IREE::Input::ExecutableExportOp> exported;
};

template <typename OpTy>
LogicalResult ConvertCompiledOp<OpTy>::matchAndRewrite(
    OpTy op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);

  auto *block = op->getBlock();

  // Extract compiled fusion from the thunk sequence.
  auto compiled_fusion = extractCompiledOp(op, thunk_sequence, rewriter);
  if (failed(compiled_fusion)) return failure();

  // Handle copy operations first, before handling kernel launch.
  for (auto &copy : compiled_fusion->memcpy) {
    auto src_memref = cast<TypedValue<MemRefType>>(copy->source_value());
    auto dst_memref = cast<TypedValue<MemRefType>>(copy->destination_value());

    auto src = state->remapped[block][stripReinterpretCast(src_memref)];
    auto dst = state->remapped[block][stripReinterpretCast(dst_memref)];

    assert(src && "unknown mapping from `src` memref to a tensor");
    assert(dst && "unknown mapping from `dst` memref to a tensor");

    auto rank = dst.getType().getRank();

    // Cast src tensor to dynamic tensors to prevent folding at Flow level.
    // TODO(ezhulenev): Find a solution that does not rely on compiler tricks.
    auto dyn_tensor =
        RankedTensorType::get(SmallVector<int64_t>(rank, ShapedType::kDynamic),
                              dst.getType().getElementType());

    Value dyn_src =
        b.create<IREE::Input::OptimizationBarrierOp>(
             b.create<mlir::tensor::CastOp>(dyn_tensor, src).getResult())
            .getResult(0);

    // Materialize dynamic dimensions for passing them to tensor update op.
    SmallVector<Value> dims = llvm::to_vector(
        llvm::map_range(dst.getType().getShape(), [&](int64_t dim) -> Value {
          return b.create<arith::ConstantIndexOp>(dim);
        }));

    // Update dst tensor with src.
    SmallVector<Value> start_indices(rank, b.create<arith::ConstantIndexOp>(0));
    Value updated = b.create<IREE::Input::TensorUpdateOp>(
        dst, ValueRange(), start_indices, dyn_src, dims);

    state->remapped[block][dst_memref] = cast<TypedValue<TensorType>>(updated);
  }

  // Compiled fusion was a plain copy operation.
  if (thunk_sequence != nullptr && !compiled_fusion->kernel) {
    rewriter.eraseOp(op);
    return success();
  }

  // Get kernel launch parameters from a compiled fusion.
  auto [kernel_name, dims] = getKernelLaunchParams(*compiled_fusion);

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

  // Create `iree_input.executable.export` operation to export device function.
  b.setInsertionPoint(executable_source_body->getTerminator());
  auto &executable_export = exported[kernel_name];
  if (!executable_export) {
    executable_export = b.create<IREE::Input::ExecutableExportOp>(
        /*sym_name=*/b.getStringAttr(kernel_name),
        /*ordinal=*/b.getIndexAttr((*ordinal)++),
        /*layout=*/getPipelineLayout(*compiled_fusion),
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

  auto dispatch_args = getDispatchArguments(*compiled_fusion, *state);
  auto &memrefs = dispatch_args.first;
  auto &tensors = dispatch_args.second;

  // Prepare tied operands and corresponding result types.
  SmallVector<int64_t> tied_operands = getTiedOperands(*compiled_fusion);
  SmallVector<Type> results =
      llvm::to_vector(llvm::map_range(tied_operands, [&](int64_t idx) -> Type {
        return tensors[idx].getType();
      }));

  SmallVector<Value> tensor_vs = llvm::to_vector(
      llvm::map_range(tensors, [&](auto tensor) -> Value { return tensor; }));

  auto dispatch = b.create<IREE::Input::DispatchOp>(
      executable_export, workload, results,
      /*resultDims=*/ValueRange(), tensor_vs,
      /*argumentDims=*/ValueRange(), b.getIndexArrayAttr(tied_operands));

  // Keep track of all tensors updated inplace.
  for (auto result : llvm::enumerate(dispatch.getResults())) {
    auto arg = memrefs[tied_operands[result.index()]];
    state->remapped[block][arg] = cast<TypedValue<TensorType>>(result.value());
  }

  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Converts lmhlo.fusion op to a iree_input.dispatch
//===----------------------------------------------------------------------===//

using ConvertFusionOp = ConvertCompiledOp<lmhlo::FusionOp>;

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
// Converts lmhlo.sort op to a iree_input.dispatch
//===----------------------------------------------------------------------===//

using ConvertSortOp = ConvertCompiledOp<lmhlo::SortOp>;

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
                       std::shared_ptr<DeBufferization> state)
      : OpConversionPattern(converter, ctx), state(std::move(state)) {}

  LogicalResult matchAndRewrite(
      lmhlo::TerminatorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto *block = op->getBlock();

    auto func = dyn_cast<func::FuncOp>(op->getParentOp());
    if (!func) return rewriter.notifyMatchFailure(op, "unsupported terminator");

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
      for (auto memref : state->imported[result]) {
        // Check that we have tensors imported from a memref.
        auto it = state->remapped[block].find(memref);
        if (it != state->remapped[block].end() && it->second.use_empty()) {
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

  std::shared_ptr<DeBufferization> state;
};

}  // namespace

//===----------------------------------------------------------------------===//

void populateCompiledOpsConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &converter,
    IREE::Input::ExecutableSourceOp executable_source,
    ThunkSequence *thunk_sequence, std::shared_ptr<DeBufferization> state) {
  auto *ctx = patterns.getContext();
  patterns.insert<ConvertFusionOp, ConvertSortOp>(
      converter, ctx, executable_source, thunk_sequence, state,
      /*ordinal=*/std::make_shared<int64_t>(0));
  patterns.insert<TerminatorOpLowering>(converter, ctx, state);
}

}  // namespace xla::gpu
