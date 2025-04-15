/* Copyright 2024 The OpenXLA Authors.

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
#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/codegen/device_spec.h"
#include "xla/codegen/emitters/transforms/atomic_rmw_utils.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep

namespace xla {
namespace emitters {
namespace {

#define GEN_PASS_DEF_LOWERTENSORSPASS
#include "xla/codegen/emitters/transforms/passes.h.inc"

using llvm::dyn_cast_or_null;
using mlir::failure;
using mlir::Location;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OpResult;
using mlir::OpRewritePattern;
using mlir::SmallVector;
using mlir::success;
using mlir::Type;
using mlir::TypedValue;
using mlir::TypeRange;
using mlir::UnrealizedConversionCastOp;
using mlir::Value;
using mlir::ValueRange;

namespace arith = ::mlir::arith;
namespace scf = ::mlir::scf;
namespace ml = ::mlir::LLVM;
namespace vector = ::mlir::vector;

Value GetDestinationBuffer(Value dest) {
  while (dest.getDefiningOp()) {
    int result_number = mlir::cast<OpResult>(dest).getResultNumber();
    if (auto insert = dest.getDefiningOp<mlir::tensor::InsertOp>()) {
      dest = insert.getDest();
    } else if (auto scf_if = dest.getDefiningOp<scf::IfOp>()) {
      // Pick one of the branches, they're required to yield the same buffers.
      dest = scf_if.getThenRegion().front().getTerminator()->getOperand(
          result_number);
    } else if (auto scf_for = dest.getDefiningOp<scf::ForOp>()) {
      dest = scf_for.getInitArgs()[result_number];
    } else if (dest.getDefiningOp<UnrealizedConversionCastOp>() ||
               dest.getDefiningOp<gpu::AllocateSharedOp>()) {
      break;
    } else if (auto transfer_write =
                   dest.getDefiningOp<vector::TransferWriteOp>()) {
      dest = transfer_write.getSource();
    } else {
      dest.getDefiningOp()->emitOpError("unsupported dest type");
      return nullptr;
    }
  }
  return dest;
}

std::optional<int> GetAlignmentFromArg(Value addr, ValueRange indices) {
  CHECK_LE(indices.size(), 1) << "Only 0D and 1D tensors are supported";

  // If the offset isn't empty or {0}, we don't return any alignment because
  // computing it isn't trivial and it's unclear that we need to deal with that
  // case in practice.
  auto effective_offset_is_zero = [](ValueRange offsets) -> bool {
    if (offsets.empty()) return true;
    return mlir::matchPattern(offsets[0].getDefiningOp(), mlir::m_Zero());
  };
  if (!effective_offset_is_zero(indices)) return std::nullopt;

  // Try to get the alignment from the function signature.
  auto base = mlir::dyn_cast<mlir::BlockArgument>(addr);
  if (!base) return std::nullopt;
  auto func =
      mlir::dyn_cast<mlir::func::FuncOp>(base.getOwner()->getParentOp());
  if (!func) return std::nullopt;
  auto align_attr =
      func.getArgAttr(base.getArgNumber(), ml::LLVMDialect::getAlignAttrName());
  if (!align_attr) return std::nullopt;
  return align_attr.cast<mlir::IntegerAttr>().getValue().getSExtValue();
}

template <typename Op>
bool IsSupportedTransfer(Op op) {
  return !absl::c_linear_search(op.getInBoundsValues(), false) &&
         op.getVectorType().getRank() == 1 && !op.getMask() &&
         op.getPermutationMap().isMinorIdentity();
}

class RewriteFunctionSignatures : public OpRewritePattern<mlir::func::FuncOp> {
 public:
  RewriteFunctionSignatures(mlir::MLIRContext* context,
                            const DeviceSpec& device_spec)
      : OpRewritePattern<mlir::func::FuncOp>(context),
        device_spec_(device_spec) {}

  LogicalResult matchAndRewrite(
      mlir::func::FuncOp op, mlir::PatternRewriter& rewriter) const override {
    auto is_tensor = [](Type ty) {
      return mlir::isa<mlir::RankedTensorType>(ty);
    };
    if (!llvm::any_of(op.getFunctionType().getInputs(), is_tensor)) {
      return rewriter.notifyMatchFailure(op,
                                         "the function has no input tensors");
    }

    bool some_tensor_result =
        llvm::any_of(op.getFunctionType().getResults(), is_tensor);

    if (!device_spec_.IsCpu()) {
      bool all_tensor_results =
          llvm::all_of(op.getFunctionType().getResults(), is_tensor);
      if (some_tensor_result && !all_tensor_results) {
        op->emitOpError("function has a mix of tensor and non-tensor results");
        return failure();
      }
    }

    TypeRange new_results = op.getFunctionType().getResults();
    if (some_tensor_result) {
      new_results = {};
      auto terminator = op.getFunctionBody().front().getTerminator();
      rewriter.setInsertionPoint(terminator);
      rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(terminator);
    }

    SmallVector<Type> new_operands(op.getFunctionType().getInputs());
    for (auto&& [index, operand] : llvm::enumerate(new_operands)) {
      if (is_tensor(operand)) {
        rewriter.setInsertionPointToStart(&op.getBody().front());
        auto cast = rewriter.create<UnrealizedConversionCastOp>(
            op.getLoc(), operand, op.getArgument(index));
        op.getArgument(index).replaceAllUsesExcept(cast.getResult(0), cast);
        operand = ml::LLVMPointerType::get(op.getContext());
      }
    }

    op.setFunctionType(rewriter.getFunctionType(new_operands, new_results));
    auto& entry = op->getRegion(0).front();
    for (auto [arg, arg_type] : llvm::zip(entry.getArguments(), new_operands)) {
      arg.setType(arg_type);
    }

    return success();
  }

 private:
  const DeviceSpec& device_spec_;
};

Value GetPtr(Value value) {
  if (!mlir::isa<mlir::RankedTensorType>(value.getType())) {
    return nullptr;
  }
  if (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast.getNumOperands() == 1 && cast.getNumResults() == 1 &&
        mlir::isa<ml::LLVMPointerType>(cast.getOperand(0).getType())) {
      return cast.getOperand(0);
    }
  }
  return nullptr;
}

struct RewriteFor : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      scf::ForOp op, mlir::PatternRewriter& rewriter) const override {
    llvm::SmallBitVector inits_to_remove(op.getNumRegionIterArgs(), false);
    SmallVector<Value> new_inits;
    new_inits.reserve(op.getNumResults());
    SmallVector<Value> ptrs;
    ptrs.reserve(op.getNumRegionIterArgs());
    for (auto [index, init] : llvm::enumerate(op.getInitArgs())) {
      Value ptr = GetPtr(init);
      if (ptr) {
        ptrs.push_back(ptr);
        inits_to_remove.set(index);
        continue;
      }
      new_inits.push_back(init);
    }
    if (inits_to_remove.none()) {
      return rewriter.notifyMatchFailure(op, "no args to remove");
    }
    // Create new ForOp with updated init args. The empty body builder is needed
    // to avoid implicit construction of scf.yield in the body block.
    Location loc = op.getLoc();
    auto new_for_op = rewriter.create<scf::ForOp>(
        loc, op.getLowerBound(), op.getUpperBound(), op.getStep(), new_inits,
        [](OpBuilder&, Location, Value, ValueRange) {});
    new_for_op->setAttrs(op->getAttrs());

    // Collect a mapping for block arguments and results. If the init is
    // removed, we can use the init of the original scf.for for replacement,
    // since it was provided by the `builtin.unrealized_conversion_cast` cast to
    // the correct type.
    mlir::Block* new_body = new_for_op.getBody();
    mlir::Block* old_body = op.getBody();
    rewriter.setInsertionPoint(new_body, new_body->begin());

    SmallVector<Value, 4> bb_args_mapping;
    bb_args_mapping.reserve(old_body->getNumArguments());
    bb_args_mapping.push_back(new_for_op.getInductionVar());
    SmallVector<Value, 4> results_replacement;
    results_replacement.reserve(old_body->getNumArguments());
    int num_removed_args = 0;
    for (auto [index, arg] : llvm::enumerate(op.getRegionIterArgs())) {
      if (!inits_to_remove.test(index)) {
        bb_args_mapping.push_back(
            new_for_op.getRegionIterArg(index - num_removed_args));
        results_replacement.push_back(
            new_for_op.getResult(index - num_removed_args));
        continue;
      }
      bb_args_mapping.push_back(op.getInitArgs()[index]);
      results_replacement.push_back(op.getInitArgs()[index]);
      ++num_removed_args;
    }

    // Move the body of the old ForOp to the new one.
    rewriter.mergeBlocks(old_body, new_body, bb_args_mapping);

    // Update the terminator.
    auto new_terminator = mlir::cast<scf::YieldOp>(new_body->getTerminator());
    SmallVector<Value> new_yielded_values;
    new_yielded_values.reserve(new_terminator->getNumOperands());
    rewriter.setInsertionPoint(new_terminator);
    for (auto [index, yielded_value] :
         llvm::enumerate(new_terminator.getResults())) {
      if (inits_to_remove.test(index)) continue;
      new_yielded_values.push_back(yielded_value);
    }
    rewriter.replaceOpWithNewOp<scf::YieldOp>(new_terminator,
                                              new_yielded_values);

    // Replace the op.
    rewriter.replaceOp(op, results_replacement);
    return mlir::success();
  }
};

Value GetLinearIndex(ValueRange indices, mlir::ImplicitLocOpBuilder& b) {
  CHECK_LE(indices.size(), 1) << "Only 0D and 1D tensors are supported";
  auto index = indices.empty() ? b.create<mlir::arith::ConstantIndexOp>(0)
                               : indices.front();
  auto index_ty = b.getIntegerType(
      mlir::DataLayout::closest(b.getInsertionBlock()->getParentOp())
          .getTypeSizeInBits(index.getType()));
  return b.create<mlir::arith::IndexCastUIOp>(index_ty, index);
}

std::tuple<Value, Value> GetI4IndexAndNibble(Value linear_index,
                                             mlir::ImplicitLocOpBuilder& b) {
  Value zero = b.create<mlir::arith::ConstantIntOp>(0, linear_index.getType());
  Value one = b.create<mlir::arith::ConstantIntOp>(1, linear_index.getType());
  Value is_low_nibble = b.create<mlir::arith::CmpIOp>(
      mlir::arith::CmpIPredicate::eq, zero,
      b.create<mlir::arith::AndIOp>(linear_index, one));
  Value i8_index = b.create<mlir::arith::ShRUIOp>(linear_index, one);
  return {i8_index, is_low_nibble};
}

ml::GEPOp CreateGep(TypedValue<mlir::RankedTensorType> tensor,
                    Value linear_index, mlir::ImplicitLocOpBuilder& b) {
  Type element_type = tensor.getType().getElementType();
  if (element_type.isIntOrFloat() &&
      element_type.getIntOrFloatBitWidth() == 4) {
    element_type = b.getI8Type();
  }
  auto ptr = ml::LLVMPointerType::get(b.getContext());
  auto tensor_ptr =
      b.create<UnrealizedConversionCastOp>(ptr, tensor).getResult(0);
  mlir::LLVMTypeConverter converter(b.getContext());
  auto llvm_element_type = converter.convertType(element_type);
  auto gep =
      b.create<ml::GEPOp>(ptr, llvm_element_type, tensor_ptr, linear_index);
  gep.setInbounds(true);
  return gep;
}

ml::GEPOp CreateGep(TypedValue<mlir::RankedTensorType> tensor,
                    ValueRange indices, mlir::ImplicitLocOpBuilder& b) {
  return CreateGep(tensor, GetLinearIndex(indices, b), b);
}

struct RewriteTensorExtract : OpRewritePattern<mlir::tensor::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      mlir::tensor::ExtractOp op,
      mlir::PatternRewriter& rewriter) const override {
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto linear_index = GetLinearIndex(op.getIndices(), b);
    Type element_type = op.getTensor().getType().getElementType();
    Value is_low_nibble = nullptr;
    if (element_type.isIntOrFloat() &&
        element_type.getIntOrFloatBitWidth() == 4) {
      std::tie(linear_index, is_low_nibble) =
          GetI4IndexAndNibble(linear_index, b);
    }

    auto gep = CreateGep(op.getTensor(), linear_index, b);
    auto load =
        rewriter.create<ml::LoadOp>(gep.getLoc(), gep.getElemType(), gep)
            .getResult();

    if (is_low_nibble) {
      auto high_value = b.create<mlir::arith::ShRUIOp>(
          load, b.create<mlir::arith::ConstantIntOp>(4, load.getType()));
      load = b.create<mlir::arith::TruncIOp>(
          rewriter.getI4Type(),
          b.create<mlir::arith::SelectOp>(is_low_nibble, load, high_value));
    }

    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(op, op.getType(),
                                                            load);
    return success();
  }
};

struct RewriteTransferRead : OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      vector::TransferReadOp op,
      mlir::PatternRewriter& rewriter) const override {
    assert(IsSupportedTransfer(op));

    auto source = mlir::dyn_cast<mlir::TypedValue<mlir::RankedTensorType>>(
        op.getSource());
    mlir::Type source_element_type = source.getType().getElementType();

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto linear_index = GetLinearIndex(op.getIndices(), b);

    mlir::VectorType vector_type = op.getVectorType();
    if (vector_type.getElementType().isInteger(1)) {
      vector_type = vector_type.cloneWith(std::nullopt, b.getI8Type());
    }
    mlir::Type gep_element_type = vector_type.getElementType();
    if (gep_element_type.isIntOrFloat() &&
        gep_element_type.getIntOrFloatBitWidth() == 4) {
      linear_index = b.create<arith::ShRUIOp>(
          linear_index,
          b.create<arith::ConstantIntOp>(1, linear_index.getType()));
    }
    auto gep = CreateGep(source, linear_index, b);

    mlir::LLVMTypeConverter converter(b.getContext());
    auto llvm_vector_type = converter.convertType(vector_type);
    auto load = b.create<ml::LoadOp>(llvm_vector_type, gep);
    if (auto alignment = GetAlignmentFromArg(op.getSource(), op.getIndices())) {
      load.setAlignment(*alignment);
    }
    auto loaded = load.getResult();

    if (source_element_type.isInteger(1)) {
      Value zero = b.create<mlir::arith::ConstantOp>(
          mlir::DenseElementsAttr::get(vector_type, b.getI8IntegerAttr(0)));
      loaded = b.create<arith::CmpIOp>(arith::CmpIPredicate::ne, loaded, zero);
    }

    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(op, op.getType(),
                                                            loaded);
    return success();
  }
};

struct RewriteTensorInsert : OpRewritePattern<mlir::tensor::InsertOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      mlir::tensor::InsertOp op,
      mlir::PatternRewriter& rewriter) const override {
    Value dest = GetDestinationBuffer(op.getDest());
    if (!dest) {
      return failure();
    }

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto tensor_dest = mlir::cast<TypedValue<mlir::RankedTensorType>>(dest);
    auto linear_index = GetLinearIndex(op.getIndices(), b);
    auto scalar_value = op.getScalar();

    // For i4 we store 2 values into one byte. This needs special handling here.
    if (tensor_dest.getType().getElementType().isIntOrFloat() &&
        tensor_dest.getType().getElementType().getIntOrFloatBitWidth() == 4) {
      // We need to use directly op.getDest() as input, otherwise the following
      // rewrite might remove the only user of it.
      tensor_dest = op.getDest();
      Value is_low_nibble;
      std::tie(linear_index, is_low_nibble) =
          GetI4IndexAndNibble(linear_index, b);

      // Technically we should half the number of elements when going to i8
      // element type, but it doesn't really matter because we only actually use
      // the element type. Indexing is done by linear index, and GEP ops don't
      // care about the number of elements. The tensor types will disappear
      // completely after the LowerTensors pass.
      Type ty = b.getI8Type();
      Type tensor_ty = tensor_dest.getType().clone(ty);
      auto tensor_dest_i8 =
          b.create<UnrealizedConversionCastOp>(tensor_ty, tensor_dest)
              .getResult(0);
      if (scalar_value.getType() != rewriter.getI4Type()) {
        scalar_value =
            b.create<arith::BitcastOp>(rewriter.getI4Type(), scalar_value);
      }
      scalar_value = b.create<mlir::arith::ExtUIOp>(ty, scalar_value);

      // We need AtomicRMWOp because it can happen that different threads try to
      // access the same memory location.
      auto atomic_rmw = b.create<AtomicRMWOp>(tensor_dest_i8, linear_index);
      mlir::ImplicitLocOpBuilder body_builder(atomic_rmw.getLoc(),
                                              atomic_rmw.getBodyBuilder());
      Value current_value = atomic_rmw.getCurrentValue();
      Value low_updated = body_builder.create<mlir::arith::OrIOp>(
          body_builder.create<mlir::arith::AndIOp>(
              current_value,
              body_builder.create<mlir::arith::ConstantIntOp>(0xf0, ty)),
          body_builder.create<mlir::arith::AndIOp>(
              scalar_value,
              body_builder.create<mlir::arith::ConstantIntOp>(0x0f, ty)));
      Value high_updated = body_builder.create<mlir::arith::OrIOp>(
          body_builder.create<mlir::arith::AndIOp>(
              current_value,
              body_builder.create<mlir::arith::ConstantIntOp>(0x0f, ty)),
          body_builder.create<mlir::arith::ShLIOp>(
              scalar_value,
              body_builder.create<mlir::arith::ConstantIntOp>(4, ty)));
      Value new_value = body_builder.create<mlir::arith::SelectOp>(
          is_low_nibble, low_updated, high_updated);
      body_builder.create<scf::YieldOp>(new_value);
      Value casted_result = b.create<UnrealizedConversionCastOp>(
                                 tensor_dest.getType(), atomic_rmw.getResult())
                                .getResult(0);
      op.replaceAllUsesWith(casted_result);
    } else {
      auto gep = CreateGep(tensor_dest, linear_index, b);
      mlir::LLVMTypeConverter converter(getContext());
      auto llvm_type = converter.convertType(scalar_value.getType());
      scalar_value =
          b.create<UnrealizedConversionCastOp>(llvm_type, scalar_value)
              .getResult(0);
      b.create<ml::StoreOp>(scalar_value, gep);
      op.replaceAllUsesWith(op.getDest());
    }

    op.erase();
    return success();
  }
};

struct RewriteTransferWrite : OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      vector::TransferWriteOp op,
      mlir::PatternRewriter& rewriter) const override {
    assert(IsSupportedTransfer(op));
    Value dest = GetDestinationBuffer(op.getSource());

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto tensor_dest = mlir::cast<TypedValue<mlir::RankedTensorType>>(dest);
    auto linear_index = GetLinearIndex(op.getIndices(), b);

    mlir::Value vector_value = op.getVector();
    mlir::Type vector_element_type = op.getVectorType().getElementType();
    if (vector_element_type.isInteger(1)) {
      vector_value = b.create<arith::ExtUIOp>(
          op.getVectorType().cloneWith(std::nullopt, b.getI8Type()),
          vector_value);
    }
    if (vector_element_type.isIntOrFloat() &&
        vector_element_type.getIntOrFloatBitWidth() == 4) {
      linear_index = b.create<arith::ShRUIOp>(
          linear_index,
          b.create<arith::ConstantIntOp>(1, linear_index.getType()));
    }
    auto gep = CreateGep(tensor_dest, linear_index, b);

    mlir::LLVMTypeConverter converter(getContext());
    auto llvm_type = converter.convertType(vector_value.getType());
    vector_value = b.create<UnrealizedConversionCastOp>(llvm_type, vector_value)
                       .getResult(0);
    b.create<ml::StoreOp>(vector_value, gep);

    rewriter.replaceOp(op, mlir::ValueRange{op.getSource()});
    return success();
  }
};

struct RewriteCall : OpRewritePattern<mlir::func::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      mlir::func::CallOp op, mlir::PatternRewriter& rewriter) const override {
    if (!llvm::any_of(op->getOperandTypes(), [](Type ty) {
          return mlir::isa<mlir::RankedTensorType>(ty);
        })) {
      return rewriter.notifyMatchFailure(op, "the call has no input tensors");
    }

    auto ptr_ty = mlir::LLVM::LLVMPointerType::get(op.getContext());
    llvm::SmallVector<Value, 4> new_operands;
    new_operands.reserve(op.getNumOperands());
    llvm::SmallVector<Type, 4> new_result_types;
    for (const auto&& [index, arg] : llvm::enumerate(op.getOperands())) {
      if (mlir::isa<mlir::RankedTensorType>(arg.getType())) {
        new_operands.push_back(rewriter
                                   .create<mlir::UnrealizedConversionCastOp>(
                                       op.getLoc(), ptr_ty, arg)
                                   .getResult(0));
      } else {
        new_operands.push_back(arg);
      }
    }
    for (const auto result_type : op.getResultTypes()) {
      if (!mlir::isa<mlir::RankedTensorType>(result_type)) {
        new_result_types.push_back(result_type);
      }
    }
    auto new_call = rewriter.create<mlir::func::CallOp>(
        op.getLoc(), op.getCallee(), new_result_types, new_operands);

    if (new_call.getNumResults() == 0) {
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, new_call.getResults());
    }
    return success();
  }
};

ml::GlobalOp CreateGlobalOp(mlir::Attribute value,
                            const std::string& name_prefix,
                            mlir::ShapedType shaped_ty, mlir::ModuleOp module,
                            bool is_constant, int addr_space,
                            mlir::ImplicitLocOpBuilder& b) {
  if (auto elements = mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(value)) {
    // The lowering to LLVM only works for 1d tensors or those with trailing
    // unit dimensions.
    value = elements.reshape(mlir::RankedTensorType::get(
        {elements.getNumElements()}, elements.getElementType()));
  }

  Type element_type = shaped_ty.getElementType();
  int64_t num_elements = shaped_ty.getNumElements();
  // Needed to support complex element type.
  mlir::LLVMTypeConverter converter(b.getContext());
  auto llvm_element_type = converter.convertType(element_type);
  if (value && element_type.isIntOrFloat() &&
      element_type.getIntOrFloatBitWidth() == 4) {
    num_elements = CeilOfRatio<int64_t>(num_elements, 2);
    llvm_element_type = b.getI8Type();
    auto unpacked_data =
        mlir::cast<mlir::DenseElementsAttr>(value).getRawData();
    std::vector<char> packed_data(num_elements);
    absl::Span<char> packed_data_span =
        absl::MakeSpan(packed_data.data(), packed_data.size());
    PackIntN(4, unpacked_data, packed_data_span);
    value = mlir::DenseElementsAttr::getFromRawBuffer(
        mlir::RankedTensorType::get({num_elements}, llvm_element_type),
        packed_data);
  }
  auto array_ty = ml::LLVMArrayType::get(llvm_element_type, num_elements);
  std::string name;
  int index = 0;
  do {
    name = absl::StrCat(name_prefix, index);
    ++index;
  } while (module.lookupSymbol(name));
  b.setInsertionPointToStart(module.getBody());
  return b.create<ml::GlobalOp>(array_ty, is_constant,
                                /*linkage=*/ml::Linkage::Private, name, value,
                                /*alignment=*/0, addr_space);
}

struct RewriteAllocateShared : OpRewritePattern<gpu::AllocateSharedOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      gpu::AllocateSharedOp op,
      mlir::PatternRewriter& rewriter) const override {
    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto shaped_ty = mlir::cast<mlir::ShapedType>(op.getResult().getType());
    constexpr int kGPUSharedMemoryAddrSpace = 3;
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto global =
        CreateGlobalOp(mlir::Attribute{}, "shared_", shaped_ty, module,
                       /*is_constant=*/false, kGPUSharedMemoryAddrSpace, b);

    rewriter.setInsertionPoint(op);
    auto addr = rewriter.create<ml::AddressOfOp>(op.getLoc(), global);
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, op.getResult().getType(),
        rewriter
            .create<ml::AddrSpaceCastOp>(
                op.getLoc(), ml::LLVMPointerType::get(op.getContext()), addr)
            .getResult());
    return success();
  }
};

struct RewriteNonScalarConstants : OpRewritePattern<mlir::arith::ConstantOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::arith::ConstantOp op,
      mlir::PatternRewriter& rewriter) const override {
    if (mlir::isa<mlir::VectorType>(op.getType())) {
      return rewriter.notifyMatchFailure(op, "the op is a vector constant");
    }
    auto shaped_ty = mlir::dyn_cast<mlir::ShapedType>(op.getValue().getType());
    // We only need to rewrite non-scalar constants.
    if (!shaped_ty || shaped_ty.getNumElements() < 2) {
      return rewriter.notifyMatchFailure(
          op, "the op is an effective scalar constant");
    }

    constexpr int kGPUGlobalMemoryAddrSpace = 0;
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto global =
        CreateGlobalOp(op.getValue(), "global_cst_", shaped_ty, module,
                       /*is_constant=*/true, kGPUGlobalMemoryAddrSpace, b);

    rewriter.setInsertionPoint(op);
    auto addr = rewriter.create<ml::AddressOfOp>(op.getLoc(), global);
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, op.getResult().getType(),
        rewriter
            .create<ml::AddrSpaceCastOp>(
                op.getLoc(), ml::LLVMPointerType::get(op.getContext()), addr)
            .getResult());
    return success();
  }
};

struct RewriteSyncThreads : OpRewritePattern<gpu::SyncThreadsOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      gpu::SyncThreadsOp op, mlir::PatternRewriter& rewriter) const override {
    rewriter.create<mlir::gpu::BarrierOp>(op.getLoc());
    rewriter.replaceOp(op, op.getOperands());
    return success();
  }
};

// TODO(jreiffers): Generalize this to support index switches with some used
// results and upstream it as a canonicalization pattern.
struct RemoveUnusedIndexSwitchResults : OpRewritePattern<scf::IndexSwitchOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      scf::IndexSwitchOp op, mlir::PatternRewriter& rewriter) const override {
    if (op->getNumResults() == 0 || !op->use_empty()) {
      return rewriter.notifyMatchFailure(op, "the op has users");
    }

    auto new_op = rewriter.create<scf::IndexSwitchOp>(
        op.getLoc(), mlir::TypeRange{}, op.getArg(), op.getCases(),
        op.getNumCases());
    for (int i = 0; i < op->getNumRegions(); ++i) {
      auto& old_region = op->getRegion(i);
      auto& new_region = new_op->getRegion(i);
      rewriter.mergeBlocks(&old_region.getBlocks().front(),
                           &new_region.emplaceBlock());
      auto yield_op = new_region.getBlocks().front().getTerminator();
      rewriter.modifyOpInPlace(yield_op, [&]() { yield_op->setOperands({}); });
    }
    rewriter.eraseOp(op);
    return success();
  }
};

bool IsAtomicIntegral(Type element_type) {
  if (!element_type.isInteger()) {
    return false;
  }
  unsigned element_bitwidth = element_type.getIntOrFloatBitWidth();
  return element_bitwidth == 32 || element_bitwidth == 64;
}

Value CreateBitcast(mlir::ImplicitLocOpBuilder& b, mlir::Operation* op,
                    Value value, Type ty) {
  if (value.getType().isIntOrFloat() && ty.isIntOrFloat()) {
    return b.create<ml::BitcastOp>(ty, value);
  }

  mlir::LLVMTypeConverter converter(b.getContext());
  // If either type is a complex, we need to go through an alloca, since no
  // direct bitcast from a struct to an int is possible.
  Type llvm_input_ty = converter.convertType(value.getType());
  Type llvm_result_ty = converter.convertType(ty);
  Type ptr_ty = ml::LLVMPointerType::get(b.getContext());
  auto func = op->getParentOfType<mlir::func::FuncOp>();
  // AMDGPU backend needs allocas to be out of loops.
  // Move them to the entry block to be on the safe side.
  auto entry_builder = mlir::ImplicitLocOpBuilder::atBlockBegin(
      b.getLoc(), &func.getBody().front(), b.getListener());

  Value llvm_value =
      b.create<UnrealizedConversionCastOp>(llvm_input_ty, value).getResult(0);
  Value alloca = entry_builder.create<ml::AllocaOp>(
      ptr_ty, llvm_input_ty, b.create<ml::ConstantOp>(b.getI32Type(), 1));
  b.create<ml::StoreOp>(llvm_value, alloca);
  auto result = b.create<ml::LoadOp>(llvm_result_ty, alloca).getResult();
  return b.create<UnrealizedConversionCastOp>(ty, result).getResult(0);
};

class RewriteAtomicRMW : public OpRewritePattern<AtomicRMWOp> {
 public:
  RewriteAtomicRMW(mlir::MLIRContext* context, const DeviceSpec& device_spec)
      : OpRewritePattern<AtomicRMWOp>(context), device_spec_(device_spec) {}

  LogicalResult matchAndRewrite(
      AtomicRMWOp op, mlir::PatternRewriter& rewriter) const override {
    auto modifier_parameters = GetAtomicModifierParameters(op);
    if (modifier_parameters.has_value()) {
      if (mlir::isa<mlir::VectorType>(modifier_parameters->first.getType()) &&
          (!device_spec_.IsNvidiaGpu() ||
           !device_spec_.gpu().cuda_compute_capability().IsAtLeastHopper())) {
        return rewriter.notifyMatchFailure(
            op,
            "atomic vectorization currently only supported on Hopper or later");
      }
    }

    if (!modifier_parameters.has_value() ||
        failed(rewriteAsDirectAtomicRMW(op, modifier_parameters, rewriter))) {
      rewriteAsAtomicCAS(op, rewriter);
    }
    rewriter.replaceOp(op, op.getInput());
    return success();
  }

 private:
  llvm::StringRef determinateScope() const {
    if (device_spec_.IsAmdGpu()) {
      return llvm::StringRef("agent-one-as");
    }
    return llvm::StringRef();
  }

  // Certain computations, such as floating-point addition and integer
  // maximization, can be simply implemented using an LLVM atomic instruction.
  // If "computation" is one of this kind, emits code to do that and returns
  // true; otherwise, returns false.
  LogicalResult rewriteAsDirectAtomicRMW(
      AtomicRMWOp op,
      std::optional<std::pair<mlir::Value, ml::AtomicBinOp>>
          modifier_parameters,
      mlir::PatternRewriter& rewriter) const {
    if (device_spec_.IsCpu()) {
      return failure();  // Unimplemented.
    }

    Value modifier_arg = modifier_parameters->first;
    Type element_type = modifier_arg.getType();
    ml::AtomicBinOp atomic_bin_op = modifier_parameters->second;

    Location loc = op.getLoc();
    auto sync_scope = determinateScope();
    mlir::ImplicitLocOpBuilder b(loc, rewriter);
    Value addr = CreateGep(op.getInput(), op.getIndices(), b);

    switch (atomic_bin_op) {
      case ml::AtomicBinOp::xchg: {
        rewriter.create<ml::StoreOp>(
            loc, modifier_arg, addr,
            /*alignment=*/element_type.getIntOrFloatBitWidth() / 8,
            /*volatile*/ false, /*isNonTemporal=*/false,
            /*isInvariantGroup=*/false, ml::AtomicOrdering::unordered);
        return success();
      }
      case ml::AtomicBinOp::add:
      case ml::AtomicBinOp::max:
      case ml::AtomicBinOp::min:
      case ml::AtomicBinOp::umax:
      case ml::AtomicBinOp::umin: {
        rewriter.create<ml::AtomicRMWOp>(loc, atomic_bin_op, addr, modifier_arg,
                                         ml::AtomicOrdering::monotonic,
                                         sync_scope);
        return success();
      }
      case ml::AtomicBinOp::fadd: {
        // TODO(b/336367154): Introduce an atomic_rmw op with the binOp attr.
        return device_spec_.IsAmdGpu()
                   ? emitAMDAtomicFAdd(
                         loc, modifier_arg, addr,
                         device_spec_.gpu().rocm_compute_capability(), rewriter)
                   : emitNVidiaAtomicFAdd(
                         loc, modifier_arg, addr,
                         device_spec_.gpu().cuda_compute_capability(), rewriter,
                         op);
      }
      case ml::AtomicBinOp::fmax: {
        return rewriteAtomicFMaxAsIntAtomics(loc, modifier_arg, addr,
                                             sync_scope, rewriter);
      }
      default:
        return failure();
    }
    return success();
  }

  LogicalResult emitNVidiaAtomicFAdd(
      Location loc, Value modifier_arg, Value addr,
      const se::CudaComputeCapability& cuda_compute_capability, OpBuilder& b,
      AtomicRMWOp& op) const {
    Type element_type = modifier_arg.getType();
    // "atom.add.f64 requires sm_60 or higher."
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-atom
    bool is_supported_f16_atomic =
        element_type.isF16() &&
        cuda_compute_capability.IsAtLeast(se::CudaComputeCapability::kVolta);
    bool is_supported_bf16_atomic =
        element_type.isBF16() &&
        cuda_compute_capability.IsAtLeast(se::CudaComputeCapability::kHopper);
    bool is_supported_f64_atomic =
        element_type.isF64() &&
        cuda_compute_capability.IsAtLeast(se::CudaComputeCapability::kPascal);
    if (auto vector_type = dyn_cast_or_null<mlir::VectorType>(element_type)) {
      return emitNvidiaVectorizedAtomicFAdd(
          loc, modifier_arg, addr, vector_type, cuda_compute_capability, b);
    }
    if (!element_type.isF32() && !is_supported_f16_atomic &&
        !is_supported_bf16_atomic && !is_supported_f64_atomic) {
      return failure();
    }

    b.create<ml::AtomicRMWOp>(loc, ml::AtomicBinOp::fadd, addr, modifier_arg,
                              ml::AtomicOrdering::monotonic);
    return success();
  }

  LogicalResult emitNvidiaVectorizedAtomicFAdd(
      Location loc, Value modifier_arg, Value addr,
      mlir::VectorType vector_type,
      const se::CudaComputeCapability& cuda_compute_capability,
      OpBuilder& b) const {
    // Hopper supports 2-element and 4-element vectors of f32.
    if (!(vector_type && vector_type.getElementType().isF32() &&
          (vector_type.getNumElements() == 2 ||
           vector_type.getNumElements() == 4) &&
          cuda_compute_capability.IsAtLeast(
              se::CudaComputeCapability::kHopper))) {
      return failure();
    }

    // TODO(https://github.com/llvm/llvm-project/issues/122760): Switch to
    // AtomicRMWOp once the bug is fixed. Currently vectorized AtomicRMWOp
    // lowers incorrectly to PTX due to a bug in NVPTX.  We use inline asm to
    // work around this.
    auto asmDialectAttr =
        ml::AsmDialectAttr::get(b.getContext(), ml::AsmDialect::AD_ATT);
    std::string asm_string;
    std::string constraints;
    if (vector_type.getNumElements() == 2) {
      asm_string = "atom.global.v2.f32.add {$0, $1}, [$2], {$3, $4};";
      constraints = "=f,=f,l,f,f";
    } else if (vector_type.getNumElements() == 4) {
      asm_string =
          "atom.global.v4.f32.add {$0, $1, $2, $3}, [$4], {$5, $6, $7, $8};";
      constraints = "=f,=f,=f,=f,l,f,f,f,f";
    }
    SmallVector<Value> asm_operands{addr};
    for (int i = 0; i < vector_type.getNumElements(); ++i) {
      asm_operands.push_back(b.create<vector::ExtractOp>(
          loc, modifier_arg,
          b.create<arith::ConstantIndexOp>(loc, i).getResult()));
    }
    SmallVector<Type> outputTypes(vector_type.getNumElements(),
                                  vector_type.getElementType());
    auto outputType =
        ml::LLVMStructType::getLiteral(b.getContext(), outputTypes);
    b.create<ml::InlineAsmOp>(loc, outputType, asm_operands, asm_string,
                              constraints,
                              /*has_side_effects=*/true,
                              /*is_align_stack=*/true, asmDialectAttr,
                              /*operand_attrs=*/mlir::ArrayAttr());
    return success();
  }

  LogicalResult emitAMDAtomicFAdd(
      Location loc, Value modifier_arg, Value addr,
      const se::RocmComputeCapability& rocm_compute_capability,
      OpBuilder& b) const {
    Type element_type = modifier_arg.getType();
    if (auto vector_type = dyn_cast_or_null<mlir::VectorType>(element_type)) {
      // TODO(rocm) Don't vectorize atomics if we cannot satisfy 4-byte
      // alignment
      if (!(vector_type.getNumElements() == 2 &&
            (vector_type.getElementType().isF16() ||
             vector_type.getElementType().isBF16()))) {
        return failure();
      }
    } else if (!element_type.isF32() && !element_type.isF16() &&
               !element_type.isBF16() && !element_type.isF64()) {
      return failure();
    }

    if ((element_type.isF16() &&
         rocm_compute_capability.has_packed_fp16_atomics_support()) ||
        (element_type.isBF16() &&
         rocm_compute_capability.has_packed_bf16_atomics_support())) {
      auto packed_type = mlir::VectorType::get({2}, element_type);
      auto i64_type = b.getI64Type();
      auto i32_type = b.getI32Type();
      auto i16_type = b.getI16Type();
      Value addr_int = b.create<ml::PtrToIntOp>(loc, i64_type, addr);
      Value addr_masked = b.create<ml::AndOp>(
          loc, addr_int, b.create<ml::ConstantOp>(loc, i64_type, -4));

      Value offset = b.create<ml::AndOp>(
          loc, b.create<ml::TruncOp>(loc, i32_type, addr_int),
          b.create<ml::ConstantOp>(loc, i32_type, 2));

      Value shift = b.create<ml::MulOp>(
          loc, offset, b.create<ml::ConstantOp>(loc, i32_type, 8));

      Value modifier_int = b.create<ml::BitcastOp>(loc, i16_type, modifier_arg);

      Value modifier_masked = b.create<ml::ShlOp>(
          loc, b.create<ml::ZExtOp>(loc, i32_type, modifier_int), shift);

      constexpr int kGlobalMemory = 1;
      addr = b.create<ml::IntToPtrOp>(
          loc, ml::LLVMPointerType::get(b.getContext(), kGlobalMemory),
          addr_masked);

      modifier_arg = b.create<ml::BitcastOp>(loc, packed_type, modifier_masked);
      element_type = packed_type;
    }

    auto op = b.create<ml::AtomicRMWOp>(
        loc, ml::AtomicBinOp::fadd, addr, modifier_arg,
        ml::AtomicOrdering::monotonic, "agent-one-as");

    auto unitAttr = b.getUnitAttr();
    auto* rocdl =
        op->getContext()->getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
    auto noRemoteMemHelper = rocdl->getNoRemoteMemoryAttrHelper();
    auto noFineMemHelper = rocdl->getNoFineGrainedMemoryAttrHelper();
    auto ignoreDenormalModeHelper = rocdl->getIgnoreDenormalModeAttrHelper();

    noRemoteMemHelper.setAttr(op, unitAttr);
    noFineMemHelper.setAttr(op, unitAttr);
    ignoreDenormalModeHelper.setAttr(op, unitAttr);

    return success();
  }

  LogicalResult rewriteAtomicFMaxAsIntAtomics(Location loc, Value modifier_arg,
                                              Value addr,
                                              llvm::StringRef sync_scope,
                                              OpBuilder& b) const {
    Type element_type = modifier_arg.getType();
    if (!element_type.isF32()) {
      return failure();
    }
    // Evaluating floating max using integer atomics has the limitation of not
    // propagating -NaNs. To handle this, we check if the update value is -NaN
    // and convert it to a positive one by dropping the sign-bit.
    Value current = b.create<ml::LoadOp>(loc, element_type, addr);

    Value current_is_nan =
        b.create<ml::FCmpOp>(loc, ml::FCmpPredicate::uno, current, current);
    auto is_current_nan =
        b.create<scf::IfOp>(loc, /*resultTypes=*/TypeRange{}, current_is_nan,
                            /*addThenBlock=*/true, /*addElseBlock=*/true);
    auto if_current_nan_then_builder =
        OpBuilder::atBlockEnd(is_current_nan.thenBlock(), b.getListener());
    if_current_nan_then_builder.create<scf::YieldOp>(loc);

    auto if_current_nan_else_builder =
        OpBuilder::atBlockEnd(is_current_nan.elseBlock(), b.getListener());
    Value is_modifier_nan = if_current_nan_else_builder.create<ml::FCmpOp>(
        loc, ml::FCmpPredicate::uno, modifier_arg, modifier_arg);
    auto f32_nan = mlir::APFloat::getNaN(mlir::APFloat::IEEEsingle());
    Value nan = if_current_nan_else_builder.create<ml::ConstantOp>(
        loc, b.getF32Type(), f32_nan);
    Value no_negative_nan_source =
        if_current_nan_else_builder.create<ml::SelectOp>(loc, is_modifier_nan,
                                                         nan, modifier_arg);
    Value current_less_than_modifier =
        if_current_nan_else_builder.create<ml::FCmpOp>(
            loc, ml::FCmpPredicate::ult, current, no_negative_nan_source);

    // This check allows us to skip the atomic update all-together at the
    // expense of reading the value in memory for every update. Evaluated
    // against Waymo's benchmarks, adding the check achieves better overall
    // performance.
    auto if_need_update = if_current_nan_else_builder.create<scf::IfOp>(
        loc, /*resultTypes=*/TypeRange{}, current_less_than_modifier,
        /*withElseRegion=*/true,
        /*addElseBlock=*/false);
    if_current_nan_else_builder.create<scf::YieldOp>(loc);

    auto then_builder =
        OpBuilder::atBlockEnd(if_need_update.thenBlock(), b.getListener());
    Value source_float_as_int = then_builder.create<ml::BitcastOp>(
        loc, then_builder.getI32Type(), no_negative_nan_source);
    Value c0 = then_builder.create<ml::ConstantOp>(loc, b.getI32Type(), 0);
    Value is_not_negative = then_builder.create<ml::ICmpOp>(
        loc, ml::ICmpPredicate::sge, source_float_as_int, c0);
    then_builder.create<scf::IfOp>(
        loc, is_not_negative,
        [&](OpBuilder& nested_b, Location nested_loc) {
          // atomicMax((int *)address, __float_as_int(val))
          nested_b.create<ml::AtomicRMWOp>(
              loc, ml::AtomicBinOp::max, addr, source_float_as_int,
              ml::AtomicOrdering::monotonic, sync_scope);
          nested_b.create<scf::YieldOp>(nested_loc);
        },
        [&](OpBuilder& nested_b, Location nested_loc) {
          // atomicMax((int *)address, __float_as_int(val))
          nested_b.create<ml::AtomicRMWOp>(
              loc, ml::AtomicBinOp::umin, addr, source_float_as_int,
              ml::AtomicOrdering::monotonic, sync_scope);
          nested_b.create<scf::YieldOp>(nested_loc);
        });
    then_builder.create<scf::YieldOp>(loc);
    return success();
  }

  // Implements atomic binary operations using atomic compare-and-swap
  // (atomicCAS) as follows:
  //   1. Reads the value from the memory pointed to by output_address and
  //     records it as old_output.
  //   2. Uses old_output as one of the source operand to perform the binary
  //     operation and stores the result in new_output.
  //   3. Calls atomicCAS which implements compare-and-swap as an atomic
  //     operation. In particular, atomicCAS reads the value from the memory
  //     pointed to by output_address, and compares the value with old_output.
  //     If the two values equal, new_output is written to the same memory
  //     location and true is returned to indicate that the atomic operation
  //     succeeds. Otherwise, the new value read from the memory is returned.
  //     In this case, the new value is copied to old_output, and steps 2.
  //     and 3. are repeated until atomicCAS succeeds.
  //
  // On Nvidia GPUs, atomicCAS can only operate on 32 bit and 64 bit integers.
  // If the element type of the binary operation is 32 bits or 64 bits, the
  // integer type of the same size is used for the atomicCAS operation. On the
  // other hand, if the element type is smaller than 32 bits, int32_t is used
  // for the atomicCAS operation. In this case, atomicCAS reads and writes 32
  // bit values from the memory, which is larger than the memory size required
  // by the original atomic binary operation. We mask off the last two bits of
  // the output_address and use the result as an address to read the 32 bit
  // values from the memory. This can avoid out of bound memory accesses if
  // tensor buffers are 4 byte aligned and have a size of 4N, an assumption
  // that the runtime can guarantee.
  void rewriteAsAtomicCAS(AtomicRMWOp op,
                          mlir::PatternRewriter& rewriter) const {
    Location loc = op.getLoc();
    auto input = op.getInput();

    // Use 32-bit atomic type for small input types.
    Type result_ty = op.getResult().getType().getElementType();
    int result_size;
    if (auto complex_ty = mlir::dyn_cast<mlir::ComplexType>(result_ty)) {
      result_size = complex_ty.getElementType().getIntOrFloatBitWidth() * 2;
    } else {
      result_size = result_ty.getIntOrFloatBitWidth();
    }

    bool small_type = result_size < 32;
    Type atomic_ty =
        mlir::IntegerType::get(op.getContext(), small_type ? 32 : result_size);

    // Calculate load address for the input.
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value linear_index = GetLinearIndex(op.getIndices(), b);
    Value is_low_nibble;

    bool is_4_bit_wide =
        result_ty.isIntOrFloat() && result_ty.getIntOrFloatBitWidth() == 4;
    if (is_4_bit_wide) {
      std::tie(linear_index, is_low_nibble) =
          GetI4IndexAndNibble(linear_index, b);
    }
    Value addr = CreateGep(input, linear_index, b);
    Value shift, mask;
    if (small_type) {
      // Update input pointer by discarding the last two bits - i.e. align to
      // 32-bit boundary for small input types (will not result in OOB, as the
      // input alignment is at least 32 bits).
      Type addr_int_ty = rewriter.getI64Type();
      Value addr_int = rewriter.create<ml::PtrToIntOp>(loc, addr_int_ty, addr);
      Value addr_offset = rewriter.create<ml::AndOp>(
          loc, addr_int, rewriter.create<ml::ConstantOp>(loc, addr_int_ty, 3));
      Value index = rewriter.create<ml::MulOp>(
          loc, addr_offset,
          rewriter.create<ml::ConstantOp>(loc, addr_int_ty, -1));
      addr = rewriter.create<ml::GEPOp>(loc, addr.getType(),
                                        rewriter.getI8Type(), addr, index,
                                        /*inbounds=*/true);

      // Calculate the bit shift (assume little-endianness).
      Value offset = rewriter.create<ml::TruncOp>(loc, atomic_ty, addr_offset);
      shift = rewriter.create<ml::MulOp>(
          loc, offset,
          rewriter.create<ml::ConstantOp>(loc, offset.getType(), 8));
      if (is_4_bit_wide) {
        auto c0 = rewriter.create<ml::ConstantOp>(loc, shift.getType(), 0);
        auto c4 = rewriter.create<ml::ConstantOp>(loc, shift.getType(), 4);
        auto subshift =
            rewriter.create<ml::SelectOp>(loc, is_low_nibble, c0, c4);
        shift = rewriter.create<ml::AddOp>(loc, shift, subshift);
      }

      // Compose the update mask.
      Value bits_long = rewriter.create<ml::ConstantOp>(loc, atomic_ty, -1);
      Value bits_short = rewriter.create<ml::ZExtOp>(
          loc, atomic_ty,
          rewriter.create<ml::ConstantOp>(
              loc, rewriter.getIntegerType(result_size), -1));
      mask = rewriter.create<ml::XOrOp>(
          loc, bits_long, rewriter.create<ml::ShlOp>(loc, bits_short, shift));
    }

    // Load initial atomic value and create the loop.
    Value initial = rewriter.create<ml::LoadOp>(loc, atomic_ty, addr);
    rewriter.create<scf::WhileOp>(
        loc, TypeRange{atomic_ty}, ValueRange{initial},
        [&](mlir::OpBuilder& builder, Location loc, ValueRange values) {
          mlir::ImplicitLocOpBuilder b(loc, builder);
          Value old_value = values[0];

          // Convert atomic value to input value.
          Value input_value;
          if (small_type) {
            Value short_value =
                b.create<ml::TruncOp>(b.getIntegerType(result_size),
                                      b.create<ml::LShrOp>(old_value, shift));
            input_value = b.create<ml::BitcastOp>(result_ty, short_value);
          } else {
            input_value = CreateBitcast(b, op, old_value, result_ty);
          }

          // Perform computation on the loaded input value.
          rewriter.mergeBlocks(&op.getComputation().front(), b.getBlock(),
                               {input_value});
          auto yield_op = b.getBlock()->getTerminator();
          Value result = yield_op->getOperand(0);
          rewriter.eraseOp(yield_op);

          // Convert resulting value to atomic value.
          Value new_value;
          if (small_type) {
            Value cast_value = b.create<ml::ZExtOp>(
                atomic_ty, b.create<ml::BitcastOp>(
                               rewriter.getIntegerType(result_size), result));
            new_value =
                b.create<ml::OrOp>(b.create<ml::AndOp>(old_value, mask),
                                   b.create<ml::ShlOp>(cast_value, shift));
          } else {
            new_value = CreateBitcast(b, op, result, atomic_ty);
          }

          auto sync_scope = determinateScope();

          // Try saving the result atomically, retry if failed.
          Value cmpxchg = b.create<ml::AtomicCmpXchgOp>(
              loc, addr, old_value, new_value,
              /*success_ordering=*/ml::AtomicOrdering::monotonic,
              /*failure_ordering=*/ml::AtomicOrdering::monotonic, sync_scope);
          Value next = b.create<ml::ExtractValueOp>(cmpxchg, 0);
          Value ok = b.create<ml::ExtractValueOp>(cmpxchg, 1);
          Value low_bit = b.create<ml::ConstantOp>(b.getOneAttr(b.getI1Type()));
          Value not_ok = b.create<ml::XOrOp>(ok, low_bit);
          b.create<scf::ConditionOp>(not_ok, ValueRange{next});
        },
        [&](mlir::OpBuilder& b, Location loc, ValueRange values) {
          b.create<scf::YieldOp>(loc, values);
        });
  }

  const DeviceSpec& device_spec_;
};

class LowerTensorsPass : public impl::LowerTensorsPassBase<LowerTensorsPass> {
 public:
  explicit LowerTensorsPass(const LowerTensorsPassOptions& options)
      : LowerTensorsPassBase(options) {}

  explicit LowerTensorsPass(const se::DeviceDescription& device_description)
      : device_spec_(device_description) {}

  void runOnOperation() override {
    if (target_type_ == "gpu" && !gpu_device_info_.empty()) {
      se::GpuDeviceInfoProto device_info;
      CHECK(tsl::protobuf::TextFormat::ParseFromString(gpu_device_info_,
                                                       &device_info));
      *device_spec_.mutable_type() = se::DeviceDescription(device_info);
    } else if (target_type_ == "cpu") {
      CHECK(gpu_device_info_.empty());
      *device_spec_.mutable_type() = CpuDeviceSpec{};
    }

    MLIRContext* mlir_context = &getContext();
    mlir::RewritePatternSet tensor_patterns(mlir_context);

    tensor_patterns.add<RewriteAtomicRMW>(mlir_context, device_spec_);
    tensor_patterns
        .add<RewriteAllocateShared, RewriteNonScalarConstants,
             RewriteSyncThreads, RewriteTensorExtract, RewriteTransferRead,
             RewriteTensorInsert, RewriteTransferWrite>(mlir_context);
    if (mlir::failed(mlir::applyPatternsGreedily(getOperation(),
                                                 std::move(tensor_patterns)))) {
      signalPassFailure();
      return;
    }

    mlir::RewritePatternSet function_patterns(mlir_context);
    function_patterns.add<RewriteFunctionSignatures>(mlir_context,
                                                     device_spec_);
    function_patterns
        .add<RewriteCall, RemoveUnusedIndexSwitchResults, RewriteFor>(
            mlir_context);
    scf::ForOp::getCanonicalizationPatterns(function_patterns, mlir_context);
    scf::IfOp::getCanonicalizationPatterns(function_patterns, mlir_context);
    if (mlir::failed(mlir::applyPatternsGreedily(
            getOperation(), std::move(function_patterns)))) {
      signalPassFailure();
      return;
    }

    getOperation()->walk([this](ml::LoadOp load) {
      Value addr = load.getAddr();
      while (auto gep = addr.getDefiningOp<ml::GEPOp>()) {
        addr = gep.getBase();
      }
      while (auto cast = addr.getDefiningOp<UnrealizedConversionCastOp>()) {
        addr = cast.getOperand(0);
      }
      if (addr.getDefiningOp<ml::AddrSpaceCastOp>() ||
          addr.getDefiningOp<ml::AddressOfOp>() ||
          addr.getDefiningOp<ml::AllocaOp>()) {
        // Shared memory, global constant or temporary - no need to annotate
        // anything.
        return;
      }
      if (auto base = mlir::dyn_cast<mlir::BlockArgument>(addr)) {
        if (auto func = mlir::dyn_cast<mlir::func::FuncOp>(
                base.getOwner()->getParentOp())) {
          if (func.getArgAttr(base.getArgNumber(), "xla.invariant")) {
            load.setInvariant(true);
          }
          return;
        }
      }
      if (!device_spec_.IsCpu()) {
        load.emitOpError(
            "load op address is not (a GEP of) a function argument");
        signalPassFailure();
      }
    });
  }

 private:
  DeviceSpec device_spec_;
};

}  // namespace

std::unique_ptr<::mlir::Pass> CreateLowerTensorsPass(
    const std::string& target_type, const std::string& gpu_device_info) {
  LowerTensorsPassOptions options;
  options.gpu_device_info_ = gpu_device_info;
  options.target_type_ = target_type;
  return std::make_unique<LowerTensorsPass>(options);
}

std::unique_ptr<::mlir::Pass> CreateLowerTensorsPass(
    const se::DeviceDescription& device_description) {
  return std::make_unique<LowerTensorsPass>(device_description);
}

}  // namespace emitters
}  // namespace xla
