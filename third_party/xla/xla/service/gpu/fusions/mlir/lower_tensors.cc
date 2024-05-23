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
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

#include "absl/strings/str_cat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"  // from @llvm-project
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"  // from @llvm-project
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Interfaces/DataLayoutInterfaces.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "xla/layout_util.h"
#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_ops.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

#define GEN_PASS_DECL_LOWERTENSORSPASS
#define GEN_PASS_DEF_LOWERTENSORSPASS
#include "xla/service/gpu/fusions/mlir/passes.h.inc"

using mlir::failure;
using mlir::Location;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::success;
using mlir::Type;
using mlir::TypedValue;
using mlir::TypeRange;
using mlir::Value;
using mlir::ValueRange;

namespace arith = ::mlir::arith;
namespace scf = ::mlir::scf;
namespace ml = ::mlir::LLVM;

struct RewriteFunctionSignatures : mlir::OpRewritePattern<mlir::func::FuncOp> {
  using OpRewritePattern::OpRewritePattern;

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
    bool all_tensor_results =
        llvm::all_of(op.getFunctionType().getResults(), is_tensor);
    if (some_tensor_result && !all_tensor_results) {
      op->emitOpError("function has a mix of tensor and non-tensor results");
      return failure();
    }

    TypeRange new_results = op.getFunctionType().getResults();
    if (some_tensor_result) {
      new_results = {};
      auto terminator = op.getFunctionBody().front().getTerminator();
      rewriter.setInsertionPoint(terminator);
      rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(terminator);
    }

    llvm::SmallVector<Type> new_operands(op.getFunctionType().getInputs());
    for (auto&& [index, operand] : llvm::enumerate(new_operands)) {
      if (is_tensor(operand)) {
        rewriter.setInsertionPointToStart(&op.getBody().front());
        auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(
            op.getLoc(), operand, op.getArgument(index));
        op.getArgument(index).replaceAllUsesExcept(cast.getResult(0), cast);
        operand = mlir::LLVM::LLVMPointerType::get(op.getContext());
      }
    }

    op.setFunctionType(rewriter.getFunctionType(new_operands, new_results));
    auto& entry = op->getRegion(0).front();
    for (auto [arg, arg_type] : llvm::zip(entry.getArguments(), new_operands)) {
      arg.setType(arg_type);
    }

    return success();
  }
};

Value GetLinearIndex(TypedValue<mlir::RankedTensorType> tensor,
                     ValueRange indices, mlir::PatternRewriter& rewriter) {
  auto byte_shape = ShapeUtil::MakeShape(U8, tensor.getType().getShape());
  if (auto encoding = tensor.getType().getEncoding()) {
    *byte_shape.mutable_layout() = LayoutUtil::MakeLayout(llvm::to_vector(
        mlir::cast<mlir::DenseElementsAttr>(encoding).getValues<int64_t>()));
  }
  auto linear_shape =
      ShapeUtil::MakeShape(U8, {ShapeUtil::ElementsIn(byte_shape)});
  auto linearize_map =
      GetBitcastMap(byte_shape, linear_shape, tensor.getContext());
  mlir::SmallVector<Value> result;
  rewriter.createOrFold<ApplyIndexingOp>(result, tensor.getLoc(), indices,
                                         ValueRange{}, linearize_map);
  CHECK_EQ(result.size(), 1);
  auto index = result.front();
  auto index_ty = rewriter.getIntegerType(
      mlir::DataLayout::closest(rewriter.getInsertionBlock()->getParentOp())
          .getTypeSizeInBits(index.getType()));
  return rewriter.create<mlir::arith::IndexCastUIOp>(tensor.getLoc(), index_ty,
                                                     index);
}

std::tuple<Value, Value> GetI4IndexAndNibble(Value linear_index,
                                             mlir::ImplicitLocOpBuilder& b) {
  Value one = b.create<mlir::arith::ConstantIntOp>(1, linear_index.getType());
  Value is_low_nibble = b.create<mlir::arith::CmpIOp>(
      mlir::arith::CmpIPredicate::eq, one,
      b.create<mlir::arith::AndIOp>(linear_index, one));
  Value i8_index = b.create<mlir::arith::ShRUIOp>(linear_index, one);
  return {i8_index, is_low_nibble};
}

mlir::LLVM::GEPOp CreateGep(TypedValue<mlir::RankedTensorType> tensor,
                            Value linear_index, mlir::PatternRewriter& rewriter,
                            Type element_type = nullptr) {
  if (!element_type) {
    element_type = tensor.getType().getElementType();
  }
  auto ptr = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
  auto tensor_ptr = rewriter
                        .create<mlir::UnrealizedConversionCastOp>(
                            tensor.getLoc(), ptr, tensor)
                        .getResult(0);
  mlir::LLVMTypeConverter converter(rewriter.getContext());
  auto llvm_element_type = converter.convertType(element_type);
  auto gep = rewriter.create<mlir::LLVM::GEPOp>(
      tensor.getLoc(), ptr, llvm_element_type, tensor_ptr, linear_index);
  gep.setInbounds(true);
  return gep;
}

mlir::LLVM::GEPOp CreateGep(TypedValue<mlir::RankedTensorType> tensor,
                            ValueRange indices,
                            mlir::PatternRewriter& rewriter) {
  return CreateGep(tensor, GetLinearIndex(tensor, indices, rewriter), rewriter);
}

struct RewriteTensorExtract : mlir::OpRewritePattern<mlir::tensor::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      mlir::tensor::ExtractOp op,
      mlir::PatternRewriter& rewriter) const override {
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto linear_index =
        GetLinearIndex(op.getTensor(), op.getIndices(), rewriter);
    Type element_type = op.getTensor().getType().getElementType();
    Value is_low_nibble = nullptr;
    if (element_type == rewriter.getI4Type()) {
      element_type = rewriter.getI8Type();
      std::tie(linear_index, is_low_nibble) =
          GetI4IndexAndNibble(linear_index, b);
    }

    auto gep = CreateGep(op.getTensor(), linear_index, rewriter, element_type);
    auto load =
        rewriter
            .create<mlir::LLVM::LoadOp>(gep.getLoc(), gep.getElemType(), gep)
            .getResult();

    if (is_low_nibble) {
      auto high_value = b.create<mlir::arith::ShRUIOp>(
          load, b.create<mlir::arith::ConstantIntOp>(4, load.getType()));
      load = b.create<mlir::arith::TruncIOp>(
          op.getType(),
          b.create<mlir::arith::SelectOp>(is_low_nibble, load, high_value));
    }

    rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(
        op, op.getType(), load);
    return success();
  }
};

struct RewriteTensorInsert : mlir::OpRewritePattern<mlir::tensor::InsertOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      mlir::tensor::InsertOp op,
      mlir::PatternRewriter& rewriter) const override {
    Value dest = op.getDest();
    while (dest.getDefiningOp()) {
      int result_number = mlir::cast<mlir::OpResult>(dest).getResultNumber();
      if (auto insert = dest.getDefiningOp<mlir::tensor::InsertOp>()) {
        dest = insert.getDest();
      } else if (auto scf_if = dest.getDefiningOp<scf::IfOp>()) {
        // Pick one of the branches, they're required to yield the same buffers.
        dest = scf_if.getThenRegion().front().getTerminator()->getOperand(
            result_number);
      } else if (auto scf_for = dest.getDefiningOp<scf::ForOp>()) {
        dest = scf_for.getInitArgs()[result_number];
      } else if (dest.getDefiningOp<mlir::UnrealizedConversionCastOp>() ||
                 dest.getDefiningOp<AllocateSharedOp>()) {
        break;
      } else {
        return op.emitOpError("unsupported dest type");
      }
    }

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto tensor_dest = mlir::cast<TypedValue<mlir::RankedTensorType>>(dest);
    auto linear_index = GetLinearIndex(tensor_dest, op.getIndices(), rewriter);
    auto element_type = tensor_dest.getType().getElementType();
    Value is_low_nibble = nullptr;

    if (element_type == rewriter.getI4Type()) {
      element_type = rewriter.getI8Type();
      std::tie(linear_index, is_low_nibble) =
          GetI4IndexAndNibble(linear_index, b);
    }

    auto gep = CreateGep(tensor_dest, linear_index, rewriter, element_type);
    auto scalar_value = op.getScalar();

    if (is_low_nibble) {
      Value current_value =
          b.create<mlir::LLVM::LoadOp>(gep.getElemType(), gep);
      auto ty = current_value.getType();
      scalar_value = b.create<mlir::arith::ExtUIOp>(ty, scalar_value);
      Value low_updated = b.create<mlir::arith::OrIOp>(
          b.create<mlir::arith::AndIOp>(
              current_value, b.create<mlir::arith::ConstantIntOp>(0xf0, ty)),
          scalar_value);
      Value high_updated = b.create<mlir::arith::OrIOp>(
          b.create<mlir::arith::AndIOp>(
              current_value, b.create<mlir::arith::ConstantIntOp>(0x0f, ty)),
          b.create<mlir::arith::ShLIOp>(
              scalar_value, b.create<mlir::arith::ConstantIntOp>(4, ty)));
      scalar_value = b.create<mlir::arith::SelectOp>(is_low_nibble, low_updated,
                                                     high_updated);
    }

    mlir::LLVMTypeConverter converter(getContext());
    auto llvm_type = converter.convertType(scalar_value.getType());
    scalar_value = rewriter
                       .create<mlir::UnrealizedConversionCastOp>(
                           gep.getLoc(), llvm_type, scalar_value)
                       .getResult(0);
    rewriter.create<mlir::LLVM::StoreOp>(gep.getLoc(), scalar_value, gep);

    op.replaceAllUsesWith(op.getDest());
    op.erase();
    return success();
  }
};

struct RewriteCall : mlir::OpRewritePattern<mlir::func::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      mlir::func::CallOp op, mlir::PatternRewriter& rewriter) const override {
    if (!llvm::any_of(op->getOperandTypes(), [](Type ty) {
          return mlir::isa<mlir::RankedTensorType>(ty);
        })) {
      return rewriter.notifyMatchFailure(op, "the call has no input tensors");
    }

    for (const auto&& [index, arg] : llvm::enumerate(op.getOperands())) {
      if (mlir::isa<mlir::RankedTensorType>(arg.getType())) {
        op.setOperand(
            index,
            rewriter
                .create<mlir::UnrealizedConversionCastOp>(
                    op.getLoc(),
                    mlir::LLVM::LLVMPointerType::get(op.getContext()), arg)
                .getResult(0));
      }
    }
    return success();
  }
};

mlir::LLVM::GlobalOp CreateGlobalOp(mlir::Attribute value,
                                    const std::string& name_prefix,
                                    mlir::ShapedType shaped_ty,
                                    mlir::ModuleOp module, bool is_constant,
                                    int addr_space,
                                    mlir::ImplicitLocOpBuilder& b) {
  if (auto elements = mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(value)) {
    // The lowering to LLVM only works for 1d tensors or those with trailing
    // unit dimensions.
    value = elements.reshape(mlir::RankedTensorType::get(
        {elements.getNumElements()}, elements.getElementType()));
  }

  Type element_type = shaped_ty.getElementType();
  // Needed to support complex element type.
  mlir::LLVMTypeConverter converter(b.getContext());
  auto llvm_element_type = converter.convertType(element_type);
  auto array_ty = mlir::LLVM::LLVMArrayType::get(llvm_element_type,
                                                 shaped_ty.getNumElements());
  std::string name;
  int index = 0;
  do {
    name = absl::StrCat(name_prefix, index);
    ++index;
  } while (module.lookupSymbol(name));
  b.setInsertionPointToStart(module.getBody());
  return b.create<mlir::LLVM::GlobalOp>(
      array_ty, is_constant,
      /*linkage=*/mlir::LLVM::Linkage::Private, name, value, /*alignment=*/0,
      addr_space);
}

struct RewriteAllocateShared : mlir::OpRewritePattern<AllocateSharedOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      AllocateSharedOp op, mlir::PatternRewriter& rewriter) const override {
    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto shaped_ty = mlir::cast<mlir::ShapedType>(op.getResult().getType());
    constexpr int kGPUSharedMemoryAddrSpace = 3;
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto global =
        CreateGlobalOp(mlir::Attribute{}, "shared_", shaped_ty, module,
                       /*is_constant=*/false, kGPUSharedMemoryAddrSpace, b);

    rewriter.setInsertionPoint(op);
    auto addr = rewriter.create<mlir::LLVM::AddressOfOp>(op.getLoc(), global);
    rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(
        op, op.getResult().getType(),
        rewriter
            .create<mlir::LLVM::AddrSpaceCastOp>(
                op.getLoc(), mlir::LLVM::LLVMPointerType::get(op.getContext()),
                addr)
            .getResult());
    return success();
  }
};

struct RewriteNonScalarConstants
    : mlir::OpRewritePattern<mlir::arith::ConstantOp> {
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
    auto addr = rewriter.create<mlir::LLVM::AddressOfOp>(op.getLoc(), global);
    rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(
        op, op.getResult().getType(),
        rewriter
            .create<mlir::LLVM::AddrSpaceCastOp>(
                op.getLoc(), mlir::LLVM::LLVMPointerType::get(op.getContext()),
                addr)
            .getResult());
    return success();
  }
};

struct RewriteSyncThreads : mlir::OpRewritePattern<SyncThreadsOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      SyncThreadsOp op, mlir::PatternRewriter& rewriter) const override {
    rewriter.create<mlir::gpu::BarrierOp>(op.getLoc());
    rewriter.replaceOp(op, op.getOperands());
    return success();
  }
};

// TODO(jreiffers): Generalize this to support index switches with some used
// results and upstream it as a canonicalization pattern.
struct RemoveUnusedIndexSwitchResults
    : mlir::OpRewritePattern<scf::IndexSwitchOp> {
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

Value CreateBitcast(mlir::ImplicitLocOpBuilder& b, Value value, Type ty) {
  if (value.getType().isIntOrFloat() && ty.isIntOrFloat()) {
    return b.create<ml::BitcastOp>(ty, value);
  }

  mlir::LLVMTypeConverter converter(b.getContext());
  // If either type is a complex, we need to go through an alloca, since no
  // direct bitcast from a struct to an int is possible.
  Type llvm_input_ty = converter.convertType(value.getType());
  Type llvm_result_ty = converter.convertType(ty);
  Type ptr_ty = mlir::LLVM::LLVMPointerType::get(b.getContext());

  Value llvm_value =
      b.create<mlir::UnrealizedConversionCastOp>(llvm_input_ty, value)
          .getResult(0);
  Value alloca = b.create<ml::AllocaOp>(
      ptr_ty, llvm_input_ty, b.create<ml::ConstantOp>(b.getI32Type(), 1));
  b.create<ml::StoreOp>(llvm_value, alloca);
  auto result = b.create<ml::LoadOp>(llvm_result_ty, alloca).getResult();
  return b.create<mlir::UnrealizedConversionCastOp>(ty, result).getResult(0);
};

class RewriteAtomicRMW : public mlir::OpRewritePattern<AtomicRMWOp> {
 public:
  RewriteAtomicRMW(mlir::MLIRContext* context, bool is_amd,
                   const std::string& gpu_arch)
      : mlir::OpRewritePattern<AtomicRMWOp>(context),
        is_amd_(is_amd),
        gpu_arch_(gpu_arch) {}

  LogicalResult matchAndRewrite(
      AtomicRMWOp op, mlir::PatternRewriter& rewriter) const override {
    if (failed(rewriteAsDirectAtomicRMW(op, rewriter))) {
      rewriteAsAtomicCAS(op, rewriter);
    }
    rewriter.replaceOp(op, op.getInput());
    return success();
  }

 private:
  // Returns atomic op modifier and the atomic bin op kind.
  std::optional<std::pair<Value, ml::AtomicBinOp>> GetAtomicModifierParameters(
      AtomicRMWOp op) const {
    Type element_type = op.getInput().getType().getElementType();
    auto& operations = op.getBody()->getOperations();
    auto terminator = op.getBody()->getTerminator();
    if (operations.size() > 2) {
      return std::nullopt;
    }
    // If the body contains only the terminator, then it is an atomic store.
    if (operations.size() == 1) {
      // TODO(b/336367145): Support complex<f32> atomic store.
      if (element_type.isF32() || IsAtomicIntegral(element_type)) {
        return std::make_pair(terminator->getOperand(0), ml::AtomicBinOp::xchg);
      }
      return std::nullopt;
    }
    // Match the kind of the atomic op.
    mlir::Operation* modifier_op = &operations.front();
    std::optional<ml::AtomicBinOp> kind =
        llvm::TypeSwitch<Operation*, std::optional<ml::AtomicBinOp>>(
            modifier_op)
            // Floating-point operations.
            .Case([](arith::AddFOp op) { return ml::AtomicBinOp::fadd; })
            .Case([](arith::MaximumFOp op) { return ml::AtomicBinOp::fmax; })
            .Case([](arith::MinimumFOp op) { return ml::AtomicBinOp::fmin; })
            // Integer operations.
            .Case([&](arith::AddIOp op) {
              return IsAtomicIntegral(element_type)
                         ? std::make_optional(ml::AtomicBinOp::add)
                         : std::nullopt;
            })
            .Case([&](arith::MaxUIOp op) {
              return IsAtomicIntegral(element_type)
                         ? std::make_optional(ml::AtomicBinOp::umax)
                         : std::nullopt;
            })
            .Case([&](arith::MinUIOp op) {
              return IsAtomicIntegral(element_type)
                         ? std::make_optional(ml::AtomicBinOp::umin)
                         : std::nullopt;
            })
            .Case([&](arith::MaxSIOp op) {
              return IsAtomicIntegral(element_type)
                         ? std::make_optional(ml::AtomicBinOp::max)
                         : std::nullopt;
            })
            .Case([&](arith::MinSIOp op) {
              return IsAtomicIntegral(element_type)
                         ? std::make_optional(ml::AtomicBinOp::min)
                         : std::nullopt;
            })
            .Default([](Operation* op) { return std::nullopt; });
    if (!kind.has_value()) {
      return std::nullopt;
    }
    // Find the modifier arg that does not match the argument of `atomic_rmw`
    // body.
    Value block_arg = op.getBody()->getArgument(0);
    Value modifier_arg = modifier_op->getOperand(0) == block_arg
                             ? modifier_op->getOperand(1)
                             : modifier_op->getOperand(0);
    return std::make_pair(modifier_arg, *kind);
  }

  // Certain computations, such as floating-point addition and integer
  // maximization, can be simply implemented using an LLVM atomic instruction.
  // If "computation" is one of this kind, emits code to do that and returns
  // true; otherwise, returns false.
  LogicalResult rewriteAsDirectAtomicRMW(
      AtomicRMWOp op, mlir::PatternRewriter& rewriter) const {
    auto modifier_parameters = GetAtomicModifierParameters(op);
    if (!modifier_parameters.has_value()) {
      return failure();
    }
    Value modifier_arg = modifier_parameters->first;
    Type element_type = modifier_arg.getType();
    ml::AtomicBinOp atomic_bin_op = modifier_parameters->second;

    Location loc = op.getLoc();
    llvm::StringRef sync_scope = is_amd_ ? "agent" : "";
    Value addr = CreateGep(op.getInput(), op.getIndices(), rewriter);

    switch (atomic_bin_op) {
      case ml::AtomicBinOp::xchg: {
        rewriter.create<ml::StoreOp>(
            loc, modifier_arg, addr,
            /*alignment=*/element_type.getIntOrFloatBitWidth(),
            /*volatile*/ false, /*isNonTemporal=*/false,
            ml::AtomicOrdering::unordered);
        return success();
      }
      case ml::AtomicBinOp::add:
      case ml::AtomicBinOp::max:
      case ml::AtomicBinOp::min:
      case ml::AtomicBinOp::umax:
      case ml::AtomicBinOp::umin: {
        rewriter.create<ml::AtomicRMWOp>(loc, atomic_bin_op, addr, modifier_arg,
                                         ml::AtomicOrdering::seq_cst,
                                         sync_scope);
        return success();
      }
      case ml::AtomicBinOp::fadd: {
        // TODO(b/336367154): Introduce an atomic_rmw op with the binOp attr.
        return is_amd_ ? emitAMDAtomicFAdd(loc, modifier_arg, addr, sync_scope,
                                           gpu_arch_, rewriter)
                       : emitNVidiaAtomicFAdd(loc, modifier_arg, addr,
                                              sync_scope, gpu_arch_, rewriter);
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

  LogicalResult emitNVidiaAtomicFAdd(Location loc, Value modifier_arg,
                                     Value addr, llvm::StringRef sync_scope,
                                     llvm::StringRef cuda_arch,
                                     OpBuilder& b) const {
    se::CudaComputeCapability cuda_compute_capability(cuda_arch.str());
    Type element_type = modifier_arg.getType();
    // "atom.add.f64 requires sm_60 or higher."
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-atom
    bool is_supported_f16_atomic =
        element_type.isF16() &&
        cuda_compute_capability.IsAtLeast(se::CudaComputeCapability::VOLTA);
    bool is_supported_f64_atomic =
        element_type.isF64() &&
        cuda_compute_capability.IsAtLeast(se::CudaComputeCapability::PASCAL_);
    if (!element_type.isF32() && !is_supported_f16_atomic &&
        !is_supported_f64_atomic) {
      return failure();
    }
    b.create<ml::AtomicRMWOp>(loc, ml::AtomicBinOp::fadd, addr, modifier_arg,
                              ml::AtomicOrdering::seq_cst, sync_scope);
    return success();
  }

  LogicalResult emitAMDAtomicFAdd(Location loc, Value modifier_arg, Value addr,
                                  llvm::StringRef sync_scope,
                                  llvm::StringRef gcn_arch,
                                  OpBuilder& b) const {
    se::RocmComputeCapability rocm_compute_capability(gcn_arch.str());
    Type element_type = modifier_arg.getType();
    bool is_supported_f16_atomic =
        element_type.isF16() &&
        rocm_compute_capability.has_fp16_atomics_support();
    if (!element_type.isF32() && !is_supported_f16_atomic) {
      return failure();
    }
    constexpr int kGlobalMemory = 1;
    constexpr int kSharedMemory = 3;
    auto addr_type = mlir::cast<ml::LLVMPointerType>(addr.getType());
    // adds to shared memory are always atomic.
    if (addr_type.getAddressSpace() != kSharedMemory) {
      // The compiler will only generate a global_atomic_fadd if the pointer is
      // in global addrspace (1)
      addr = b.create<ml::AddrSpaceCastOp>(
          loc, ml::LLVMPointerType::get(b.getContext(), kGlobalMemory), addr);
    }
    b.create<ml::AtomicRMWOp>(loc, ml::AtomicBinOp::fadd, addr, modifier_arg,
                              ml::AtomicOrdering::seq_cst, sync_scope);
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
              ml::AtomicOrdering::seq_cst, sync_scope);
          nested_b.create<scf::YieldOp>(nested_loc);
        },
        [&](OpBuilder& nested_b, Location nested_loc) {
          // atomicMax((int *)address, __float_as_int(val))
          nested_b.create<ml::AtomicRMWOp>(
              loc, ml::AtomicBinOp::umin, addr, source_float_as_int,
              ml::AtomicOrdering::seq_cst, sync_scope);
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
  //     succeeds. Otherwise, the new value read from the memory is returned. In
  //     this case, the new value is copied to old_output, and steps 2. and 3.
  //     are repeated until atomicCAS succeeds.
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
  // tensor buffers are 4 byte aligned and have a size of 4N, an assumption that
  // the runtime can guarantee.
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
    Value addr = CreateGep(input, op.getIndices(), rewriter);
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
      addr =
          rewriter.create<ml::GEPOp>(loc, addr.getType(), rewriter.getI8Type(),
                                     addr, index, /*inbounds=*/true);

      // Calculate the bit shift (assume little-endianness).
      Value offset = rewriter.create<ml::TruncOp>(loc, atomic_ty, addr_offset);
      shift = rewriter.create<ml::MulOp>(
          loc, offset,
          rewriter.create<ml::ConstantOp>(loc, offset.getType(), 8));

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
            input_value = CreateBitcast(b, old_value, result_ty);
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
            new_value = CreateBitcast(b, result, atomic_ty);
          }

          // Try saving the result atomically, retry if failed.
          Value cmpxchg = b.create<ml::AtomicCmpXchgOp>(
              loc, addr, old_value, new_value,
              /*success_ordering=*/ml::AtomicOrdering::seq_cst,
              /*failure_ordering=*/ml::AtomicOrdering::seq_cst);
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

  bool is_amd_;
  std::string gpu_arch_;
};

class LowerTensorsPass : public impl::LowerTensorsPassBase<LowerTensorsPass> {
 public:
  explicit LowerTensorsPass(const LowerTensorsPassOptions& options)
      : LowerTensorsPassBase(options) {}

  void runOnOperation() override {
    MLIRContext* mlir_context = &getContext();
    mlir::RewritePatternSet tensor_patterns(mlir_context);
    tensor_patterns.add<RewriteAtomicRMW>(mlir_context, is_amd_gpu_, gpu_arch_);
    tensor_patterns
        .add<RewriteAllocateShared, RewriteNonScalarConstants,
             RewriteSyncThreads, RewriteTensorExtract, RewriteTensorInsert>(
            mlir_context);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(
            getOperation(), std::move(tensor_patterns)))) {
      signalPassFailure();
    }

    mlir::RewritePatternSet function_patterns(mlir_context);
    function_patterns.add<RewriteFunctionSignatures, RewriteCall,
                          RemoveUnusedIndexSwitchResults>(mlir_context);
    scf::ForOp::getCanonicalizationPatterns(function_patterns, mlir_context);
    scf::IfOp::getCanonicalizationPatterns(function_patterns, mlir_context);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(
            getOperation(), std::move(function_patterns)))) {
      signalPassFailure();
    }

    getOperation()->walk([this](mlir::LLVM::LoadOp load) {
      Value addr = load.getAddr();
      while (auto gep = addr.getDefiningOp<mlir::LLVM::GEPOp>()) {
        addr = gep.getBase();
      }
      if (addr.getDefiningOp<mlir::LLVM::AddrSpaceCastOp>() ||
          addr.getDefiningOp<mlir::LLVM::AddressOfOp>() ||
          addr.getDefiningOp<mlir::LLVM::AllocaOp>()) {
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
      load.emitOpError("load op address is not (a GEP of) a function argument");
      signalPassFailure();
    });
  }
};

}  // namespace

std::unique_ptr<::mlir::Pass> CreateLowerTensorsPass(
    bool is_amd_gpu, const std::string& gpu_arch) {
  LowerTensorsPassOptions options;
  options.is_amd_gpu_ = is_amd_gpu;
  options.gpu_arch_ = gpu_arch;
  return std::make_unique<LowerTensorsPass>(options);
}

}  // namespace gpu
}  // namespace xla
