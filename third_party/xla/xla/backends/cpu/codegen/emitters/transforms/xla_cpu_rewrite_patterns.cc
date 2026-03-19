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

#include "xla/backends/cpu/codegen/emitters/transforms/xla_cpu_rewrite_patterns.h"

#include <array>
#include <cstdint>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "xla/backends/cpu/codegen/emitters/ir/xla_cpu_ops.h"
#include "xla/backends/cpu/codegen/emitters/ir/xla_cpu_types.h"
#include "xla/codegen/emitters/ir/xla_ops.h"

namespace xla::cpu {
namespace {

static mlir::LLVM::LLVMStructType getNewOrExistingStruct(
    mlir::MLIRContext* ctx, absl::string_view name,
    llvm::ArrayRef<mlir::Type> types) {
  mlir::LLVM::LLVMStructType struct_type =
      mlir::LLVM::LLVMStructType::getIdentified(ctx, name);
  if (struct_type.isIdentified() && struct_type.getBody().equals(types)) {
    return struct_type;
  }
  return mlir::LLVM::LLVMStructType::getNewIdentified(ctx, name, types);
}

static mlir::LLVM::LLVMStructType KernelDim3Type(mlir::MLIRContext* ctx) {
  auto i64 = mlir::IntegerType::get(ctx, 64);
  return getNewOrExistingStruct(ctx, "kernel_dim3", {i64, i64, i64});
}

static mlir::LLVM::LLVMStructType KernelArgType(mlir::MLIRContext* ctx) {
  auto ptr = mlir::LLVM::LLVMPointerType::get(ctx);
  auto i64 = mlir::IntegerType::get(ctx, 64);
  return getNewOrExistingStruct(ctx, "XLA_CPU_KernelArg", {ptr, i64});
}

static mlir::LLVM::LLVMStructType KernelCallFrameType(mlir::MLIRContext* ctx) {
  auto ptr = mlir::LLVM::LLVMPointerType::get(ctx);
  auto i64 = mlir::IntegerType::get(ctx, 64);
  return getNewOrExistingStruct(ctx, "XLA_CPU_KernelCallFrame",
                                {ptr, ptr, i64, ptr});
}

struct LowerLoadOp : public mlir::OpRewritePattern<LoadOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      cpu::LoadOp op, mlir::PatternRewriter& rewriter) const override {
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto ptr = b.getType<mlir::LLVM::LLVMPointerType>();
    auto kernel_call_frame = KernelCallFrameType(b.getContext());
    auto kernel_arg = KernelArgType(b.getContext());

    // Get a pointer to the first `KernelArg` struct.
    auto cast = mlir::UnrealizedConversionCastOp::create(b, op.getLoc(), ptr,
                                                         op.getCallFrame())
                    .getResult(0);
    auto args_gep = mlir::LLVM::GEPOp::create(
        b, ptr, kernel_call_frame, cast,
        llvm::SmallVector<mlir::LLVM::GEPArg, 2>{mlir::LLVM::GEPArg(0),
                                                 mlir::LLVM::GEPArg(3)},
        mlir::LLVM::GEPNoWrapFlags::inbounds);
    auto args_ptr = mlir::LLVM::LoadOp::create(b, ptr, args_gep);
    args_ptr.setInvariant(true);

    // Get a pointer to the `KernelArg` at the given index.
    auto arg_gep = mlir::LLVM::GEPOp::create(
        b, ptr, kernel_arg, args_ptr,
        llvm::SmallVector<mlir::LLVM::GEPArg, 2>{
            mlir::LLVM::GEPArg(op.getIndex()), mlir::LLVM::GEPArg(0)},
        mlir::LLVM::GEPNoWrapFlags::inbounds);
    auto arg_ptr = mlir::LLVM::LoadOp::create(b, ptr, arg_gep);
    arg_ptr.setInvariant(true);

    if (auto dereferenceable = op->getAttrOfType<mlir::IntegerAttr>(
            mlir::LLVM::LLVMDialect::getDereferenceableAttrName())) {
      arg_ptr.setDereferenceable(
          rewriter.getAttr<mlir::LLVM::DereferenceableAttr>(
              dereferenceable.getInt(), false));
    }

    if (auto memref_type = mlir::dyn_cast<mlir::MemRefType>(op.getType())) {
      mlir::LLVMTypeConverter converter(rewriter.getContext());
      mlir::Value memref_desc = mlir::MemRefDescriptor::fromStaticShape(
          b, op.getLoc(), converter, memref_type, arg_ptr);
      auto memref_cast = mlir::UnrealizedConversionCastOp::create(
          b, op.getLoc(), op.getResult().getType(), memref_desc);
      rewriter.replaceOp(op, memref_cast);
    } else {
      auto arg_ptr_cast = mlir::UnrealizedConversionCastOp::create(
          b, op.getLoc(), op.getResult().getType(), arg_ptr.getResult());
      rewriter.replaceOp(op, arg_ptr_cast.getResult(0));
    }
    return mlir::success();
  }
};

struct LowerExtractWorkgroupIdOp
    : public mlir::OpRewritePattern<ExtractWorkgroupIdOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ExtractWorkgroupIdOp op, mlir::PatternRewriter& rewriter) const override {
    mlir::MLIRContext* context = rewriter.getContext();
    mlir::ImplicitLocOpBuilder builder(op.getLoc(), rewriter);

    auto ptr_type = builder.getType<mlir::LLVM::LLVMPointerType>();
    auto kernel_call_frame = KernelCallFrameType(context);
    auto kernel_dim = KernelDim3Type(context);
    auto i64_ty = builder.getI64Type();

    // Get a pointer to the `WorkGroupThread` struct.
    auto cast = mlir::UnrealizedConversionCastOp::create(builder, ptr_type,
                                                         op.getCallFrame())
                    .getResult(0);
    auto workgroup_gep = mlir::LLVM::GEPOp::create(
        builder, ptr_type, kernel_call_frame, cast,
        mlir::ArrayRef<mlir::LLVM::GEPArg>{mlir::LLVM::GEPArg(0),
                                           mlir::LLVM::GEPArg(1)},
        mlir::LLVM::GEPNoWrapFlags::inbounds);
    auto workgroup_ptr =
        mlir::LLVM::LoadOp::create(builder, ptr_type, workgroup_gep);

    int32_t workgroup_dim_idx = static_cast<int32_t>(op.getDimension());
    auto workgroup_dim_gep = mlir::LLVM::GEPOp::create(
        builder, ptr_type, kernel_dim, workgroup_ptr,
        mlir::ArrayRef<mlir::LLVM::GEPArg>{
            mlir::LLVM::GEPArg(0), mlir::LLVM::GEPArg(workgroup_dim_idx)},
        mlir::LLVM::GEPNoWrapFlags::inbounds);
    auto workgroup_dim_load =
        mlir::LLVM::LoadOp::create(builder, i64_ty, workgroup_dim_gep);
    workgroup_dim_load.setInvariant(true);

    mlir::Value workgroup_dim = workgroup_dim_load.getResult();
    auto index_ty = builder.getIntegerType(
        mlir::DataLayout::closest(builder.getInsertionBlock()->getParentOp())
            .getTypeSizeInBits(mlir::IndexType::get(context)));
    if (index_ty != i64_ty) {
      workgroup_dim =
          mlir::LLVM::TruncOp::create(builder, index_ty, workgroup_dim,
                                      mlir::LLVM::IntegerOverflowFlags::nsw);
    }
    auto workgroup_dim_cast = mlir::UnrealizedConversionCastOp::create(
        builder, mlir::IndexType::get(context), workgroup_dim);

    rewriter.replaceOp(op, workgroup_dim_cast.getResult(0));

    return mlir::success();
  }
};

struct LowerWorkGroupIdOp : public mlir::OpRewritePattern<WorkGroupIdOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      WorkGroupIdOp op, mlir::PatternRewriter& rewriter) const override {
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto func = op->getParentOfType<mlir::func::FuncOp>();
    if (func == nullptr) {
      return rewriter.notifyMatchFailure(op, "No parent func found.");
    }

    if (!func->hasAttr("xla.cpu.is_wrapped")) {
      return rewriter.notifyMatchFailure(
          op, "Parent func is not wrapped in a call frame.");
    }

    if (func.getNumArguments() < 3) {
      return rewriter.notifyMatchFailure(op,
                                         "Parent func less than 3 arguments.");
    }

    WorkGroupDimension dim = op.getDimension();
    int32_t workgroup_dim_idx = dim == WorkGroupDimension::x   ? 0
                                : dim == WorkGroupDimension::y ? 1
                                                               : 2;

    auto workgroup_ids = func.getArguments().take_back(3);
    rewriter.replaceOp(op, workgroup_ids[workgroup_dim_idx]);
    return mlir::success();
  }
};

struct LowerSuccessOp : public mlir::OpRewritePattern<SuccessOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      SuccessOp op, mlir::PatternRewriter& rewriter) const override {
    auto elementPtrType =
        mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    rewriter.replaceOpWithNewOp<mlir::LLVM::ZeroOp>(op, elementPtrType);
    return mlir::success();
  }
};

struct RewriteFunctionSignatures : mlir::OpRewritePattern<mlir::func::FuncOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::func::FuncOp op, mlir::PatternRewriter& rewriter) const override {
    auto func_type = op.getFunctionType();
    if (func_type.getNumInputs() != 1 || func_type.getNumResults() != 1 ||
        !mlir::isa<CallFrameType>(func_type.getInput(0)) ||
        !mlir::isa<ErrorType>(func_type.getResult(0))) {
      return rewriter.notifyMatchFailure(
          op, "the function signature does not match the XLA_CPU_Kernel type.");
    }

    auto ptr = rewriter.getType<mlir::LLVM::LLVMPointerType>();
    llvm::SmallVector<mlir::Type> new_operands{ptr};
    rewriter.setInsertionPointToStart(&op.getBody().front());

    auto cast = mlir::UnrealizedConversionCastOp::create(
        rewriter, op.getLoc(), func_type.getInput(0), op.getArgument(0));
    op.getArgument(0).replaceAllUsesExcept(cast.getResult(0), cast);
    op.setFunctionType(rewriter.getFunctionType(new_operands, {ptr}));
    auto& entry = op->getRegion(0).front();
    for (auto [arg, arg_type] : llvm::zip(entry.getArguments(), new_operands)) {
      arg.setType(arg_type);
    }
    return mlir::success();
  }
};

class WrapEntryWithCallFrame
    : public mlir::OpRewritePattern<mlir::func::FuncOp> {
 public:
  WrapEntryWithCallFrame(mlir::MLIRContext* context, int32_t vector_width)
      : OpRewritePattern<mlir::func::FuncOp>(context),
        vector_width_(vector_width) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::func::FuncOp op, mlir::PatternRewriter& rewriter) const override {
    if (!op->hasAttr("xla.entry")) {
      return mlir::failure();
    }

    if (op->hasAttr("xla.cpu.is_wrapped")) {
      return mlir::failure();
    }

    if (mlir::isa<CallFrameType>(op.getArgument(0).getType())) {
      return rewriter.notifyMatchFailure(
          op, "Entry function has a CallFrame argument.");
    }

    if (!op->getUsers().empty()) {
      return rewriter.notifyMatchFailure(op, "Entry function has users.");
    }

    mlir::MLIRContext* context = rewriter.getContext();

    AddWorkGroupArguments(op);

    mlir::ImplicitLocOpBuilder builder(op.getLoc(), rewriter);

    std::string kernel_name(op.getName());
    op.setName(absl::StrCat(kernel_name, "_wrapped"));

    auto call_frame_type = CallFrameType::get(context);
    auto error_type = ErrorType::get(context);
    mlir::func::FuncOp kernel_func = mlir::func::FuncOp::create(
        builder, kernel_name,
        rewriter.getFunctionType({call_frame_type}, {error_type}));

    builder.setInsertionPointToStart(kernel_func.addEntryBlock());

    SetKernelFunctionAttributes(builder, kernel_func, vector_width_);

    mlir::BlockArgument call_frame_arg = kernel_func.getArgument(0);
    llvm::SmallVector<mlir::Value> call_args;
    mlir::ArrayAttr arg_attrs = op.getArgAttrsAttr();
    for (const auto& [idx, arg] :
         llvm::enumerate(op.getArguments().drop_back(3))) {
      mlir::DictionaryAttr arg_attr =
          arg_attrs ? mlir::dyn_cast<mlir::DictionaryAttr>(arg_attrs[idx])
                    : nullptr;
      LoadOp load = LoadOp::create(builder, arg.getType(), call_frame_arg, idx);
      if (arg_attr) {
        load->setAttrs(arg_attr);
      }
      call_args.push_back(load);
    }

    for (auto workgroup_id : {WorkGroupDimension::x, WorkGroupDimension::y,
                              WorkGroupDimension::z}) {
      call_args.push_back(
          ExtractWorkgroupIdOp::create(builder, mlir::IndexType::get(context),
                                       call_frame_arg, workgroup_id));
    }

    // Use func::call here rather than pure call to avoid the entry function
    // being DCEd.
    mlir::func::CallOp::create(builder, op, call_args);

    auto error = cpu::SuccessOp::create(builder, error_type);
    mlir::func::ReturnOp::create(builder, error.getResult());

    op->setAttr("xla.cpu.is_wrapped", builder.getUnitAttr());
    op.setPrivate();
    op->setAttr("llvm.linkage", mlir::LLVM::LinkageAttr::get(
                                    context, mlir::LLVM::Linkage::Internal));
    op->setAttr("always_inline", builder.getUnitAttr());

    return mlir::success();
  }

 private:
  static void AddWorkGroupArguments(mlir::func::FuncOp op) {
    mlir::MLIRContext* context = op.getContext();
    mlir::IndexType index_type = mlir::IndexType::get(context);

    llvm::SmallVector<mlir::Type> new_arg_types(op.getArgumentTypes());
    new_arg_types.push_back(index_type);
    new_arg_types.push_back(index_type);
    new_arg_types.push_back(index_type);

    mlir::FunctionType newFuncType =
        mlir::FunctionType::get(context, new_arg_types, op.getResultTypes());
    op.setFunctionType(newFuncType);
    mlir::Block& entry = op.getBody().front();
    entry.addArgument(index_type, op->getLoc());
    entry.addArgument(index_type, op->getLoc());
    entry.addArgument(index_type, op->getLoc());

    if (op.getArgAttrs().has_value()) {
      llvm::SmallVector<mlir::Attribute> arg_attrs;
      arg_attrs.append(op.getArgAttrs()->begin(), op.getArgAttrs()->end());
      arg_attrs.push_back(mlir::DictionaryAttr{});
      arg_attrs.push_back(mlir::DictionaryAttr{});
      arg_attrs.push_back(mlir::DictionaryAttr{});

      op.setAllArgAttrs(arg_attrs);
    }
  }

  static void SetKernelFunctionAttributes(mlir::Builder& builder,
                                          mlir::func::FuncOp& func,
                                          int32_t vector_width) {
    mlir::MLIRContext* context = func->getContext();

    mlir::ArrayAttr prefer_vector_width_attr = builder.getStrArrayAttr(
        {"prefer-vector-width", absl::StrCat(vector_width)});
    func->setAttr("passthrough",
                  builder.getArrayAttr({prefer_vector_width_attr}));
    func->setAttr(
        "frame_pointer",
        mlir::LLVM::FramePointerKindAttr::get(
            context, mlir::LLVM::framePointerKind::FramePointerKind::All));
    func->setAttr("uwtable_kind",
                  mlir::LLVM::UWTableKindAttr::get(
                      context, mlir::LLVM::uwtable::UWTableKind::Async));
  }

  int32_t vector_width_;
};

// Implementation similar to https://gudok.xyz/transpose/#_64_bit_simd_0_74.

// Example that follows changes in the first row.
// Let rows[i][j] = i * 8 + j

// Round 1: Transpose 2x2 blocks

// rows[0]: [0, 1, 2, 3, 4, 5, 6, 7]
// rows[1]: [8, 9, 10, 11, 12, 13, 14, 15]

// maps to:

// t0: [0, 8, 2, 10, 4, 12, 6, 14]
// t1: [1, 9, 3, 11, 5, 13, 7, 15]

// Round 2: Transpose 4x4 blocks (Swapping 2-element pairs).

// t0: [0, 8, 2, 10, 4, 12, 6, 14]
// t2: [16, 24, 18, 26, 20, 28, 22, 30]

// maps to:

// u0: [0, 8, 16, 24, 4, 12, 20, 28]
// u2: [2, 10, 18, 26, 6, 14, 22, 30]

// Round 3: Transpose 8x8 blocks (Swapping 4-element groups across
// 128-bit lanes).

// u0: [0, 8, 16, 24, 4, 12, 20, 28]
// u4: [32, 40, 48, 56, 36, 44, 52, 60]

// maps to:

// w0: [0, 8, 16, 24, 32, 40, 48, 56]     // Result row 0
// w4: [4, 12, 20, 28, 36, 44, 52, 60]    // Result row 4
mlir::Value Shuffle8x8(mlir::PatternRewriter& rewriter, mlir::Location loc,
                       mlir::Type result_type, mlir::Value source, int m,
                       int n) {
  llvm::SmallVector<mlir::Value> rows;

  for (int row = 0; row < m; ++row) {
    rows.push_back(mlir::vector::ExtractOp::create(rewriter, loc, source, row));
  }

  // Round 1

  llvm::SmallVector<mlir::Value> rows_round1(rows.size());

  // Interleave inside the 2x2 blocks.
  for (const auto i : {0, 2, 4, 6}) {
    constexpr int64_t kRound1Step = 1;
    constexpr std::array<int64_t, 8> kUpperBlockMask = {0, 8,  2, 10,
                                                        4, 12, 6, 14};
    constexpr std::array<int64_t, 8> kLowerBlockMask = {1, 9,  3, 11,
                                                        5, 13, 7, 15};

    rows_round1[i] = mlir::vector::ShuffleOp::create(
        rewriter, loc, rows[i], rows[i + kRound1Step], kUpperBlockMask);

    rows_round1[i + kRound1Step] = mlir::vector::ShuffleOp::create(
        rewriter, loc, rows[i], rows[i + kRound1Step], kLowerBlockMask);
  }

  // Round 2

  llvm::SmallVector<mlir::Value> rows_round2(rows.size());

  // Interleave adjacent 2x2 blocks.
  for (const auto i : {0, 1, 4, 5}) {
    constexpr int64_t kRound2Step = 2;
    constexpr std::array<int64_t, 8> kUpperBlockMask = {0, 1, 8,  9,
                                                        4, 5, 12, 13};
    constexpr std::array<int64_t, 8> kLowerBlockMask = {2, 3, 10, 11,
                                                        6, 7, 14, 15};
    rows_round2[i] = mlir::vector::ShuffleOp::create(
        rewriter, loc, rows_round1[i], rows_round1[i + kRound2Step],
        kUpperBlockMask);

    rows_round2[i + kRound2Step] = mlir::vector::ShuffleOp::create(
        rewriter, loc, rows_round1[i], rows_round1[i + kRound2Step],
        kLowerBlockMask);
  }

  // Round 3

  llvm::SmallVector<mlir::Value> rows_round3(rows.size());

  // Interleave adjacent 4x4 blocks.
  for (const auto i : {0, 1, 2, 3}) {
    constexpr int64_t kRound3Step = 4;
    constexpr std::array<int64_t, 8> kUpperBlockMask = {0, 1, 2,  3,
                                                        8, 9, 10, 11};
    constexpr std::array<int64_t, 8> kLowerBlockMask = {4,  5,  6,  7,
                                                        12, 13, 14, 15};
    rows_round3[i] = mlir::vector::ShuffleOp::create(
        rewriter, loc, rows_round2[i], rows_round2[i + kRound3Step],
        kUpperBlockMask);

    rows_round3[i + kRound3Step] = mlir::vector::ShuffleOp::create(
        rewriter, loc, rows_round2[i], rows_round2[i + kRound3Step],
        kLowerBlockMask);
  }

  mlir::Value result = mlir::ub::PoisonOp::create(rewriter, loc, result_type);

  for (int row_index = 0; row_index < rows_round3.size(); ++row_index) {
    result = mlir::vector::InsertOp::create(
        rewriter, loc, rows_round3[row_index], result, row_index);
  }

  return result;
}

struct LowerVector2DTransposeOp
    : public mlir::OpRewritePattern<mlir::vector::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(
      mlir::vector::TransposeOp op,
      mlir::PatternRewriter& rewriter) const override {
    auto src_gt_one_dims = isTranspose2DSlice(op);
    if (mlir::failed(src_gt_one_dims)) {
      return rewriter.notifyMatchFailure(
          op, "expected transposition on a 2D slice");
    }

    mlir::VectorType srcType = op.getSourceVectorType();
    int64_t m = srcType.getDimSize(std::get<0>(src_gt_one_dims.value()));
    int64_t n = srcType.getDimSize(std::get<1>(src_gt_one_dims.value()));

    if (!(m == 8 && n == 8)) {
      return rewriter.notifyMatchFailure(
          op, "expected transposition on a 8x8 vector");
    }

    // Reshape the n-D input vector with only two dimensions greater than one
    // to a 2-D vector.
    mlir::Location loc = op.getLoc();
    auto reshInputType =
        mlir::VectorType::get({m, n}, srcType.getElementType());
    auto reshInput = mlir::vector::ShapeCastOp::create(
        rewriter, loc, reshInputType, op.getVector());

    auto output_type = mlir::VectorType::get({n, m}, srcType.getElementType());

    auto res = Shuffle8x8(rewriter, loc, output_type, reshInput, m, n);

    rewriter.replaceOpWithNewOp<mlir::vector::ShapeCastOp>(
        op, op.getResultVectorType(), res);

    return mlir::success();
  }
};

}  // namespace

void PopulateXlaCpuConversionPatterns(mlir::RewritePatternSet& patterns,
                                      int32_t vector_width) {
  patterns.add<LowerLoadOp, LowerWorkGroupIdOp, LowerSuccessOp,
               RewriteFunctionSignatures, LowerExtractWorkgroupIdOp,
               LowerVector2DTransposeOp>(patterns.getContext());
  patterns.add<WrapEntryWithCallFrame>(patterns.getContext(), vector_width);
}

}  // namespace xla::cpu
