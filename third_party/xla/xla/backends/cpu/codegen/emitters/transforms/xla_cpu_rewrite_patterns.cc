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

#include <cstdint>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
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
    auto cast = b.create<mlir::UnrealizedConversionCastOp>(op.getLoc(), ptr,
                                                           op.getCallFrame())
                    .getResult(0);
    auto args_gep = b.create<mlir::LLVM::GEPOp>(
        ptr, kernel_call_frame, cast,
        llvm::SmallVector<mlir::LLVM::GEPArg, 2>{mlir::LLVM::GEPArg(0),
                                                 mlir::LLVM::GEPArg(3)},
        mlir::LLVM::GEPNoWrapFlags::inbounds);
    auto args_ptr = b.create<mlir::LLVM::LoadOp>(ptr, args_gep);
    args_ptr.setInvariant(true);

    // Get a pointer to the `KernelArg` at the given index.
    auto arg_gep = b.create<mlir::LLVM::GEPOp>(
        ptr, kernel_arg, args_ptr,
        llvm::SmallVector<mlir::LLVM::GEPArg, 2>{
            mlir::LLVM::GEPArg(op.getIndex()), mlir::LLVM::GEPArg(0)},
        mlir::LLVM::GEPNoWrapFlags::inbounds);
    auto arg_ptr = b.create<mlir::LLVM::LoadOp>(ptr, arg_gep);
    arg_ptr.setInvariant(true);
    arg_ptr->setAttr(mlir::LLVM::LLVMDialect::getAlignAttrName(),
                     b.getIndexAttr(32));

    auto arg_ptr_cast = b.create<mlir::UnrealizedConversionCastOp>(
        op.getLoc(), op->getResult(0).getType(), arg_ptr.getResult());
    rewriter.replaceOp(op, arg_ptr_cast.getResult(0));
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

    auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(
        op.getLoc(), func_type.getInput(0), op.getArgument(0));
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

    auto call_frame_type = CallFrameType::get(context);
    auto error_type = ErrorType::get(context);
    mlir::func::FuncOp kernel_func = builder.create<mlir::func::FuncOp>(
        absl::StrCat(absl::string_view(op.getName()), "_kernel"),
        rewriter.getFunctionType({call_frame_type}, {error_type}));

    builder.setInsertionPointToStart(kernel_func.addEntryBlock());

    SetKernelFunctionAttributes(builder, kernel_func, vector_width_);

    mlir::BlockArgument call_frame_arg = kernel_func.getArgument(0);
    llvm::SmallVector<mlir::Value> call_args;
    for (const auto& [idx, arg] :
         llvm::enumerate(op.getArguments().drop_back(3))) {
      call_args.push_back(
          builder.create<LoadOp>(arg.getType(), call_frame_arg, idx));
    }
    call_args.append(GetWorkGroupIds(call_frame_arg, builder));

    // Use func::call here rather than pure call to avoid the entry function
    // being DCEd.
    builder.create<mlir::func::CallOp>(op, call_args);

    auto error = builder.create<cpu::SuccessOp>(error_type);
    builder.create<mlir::func::ReturnOp>(error.getResult());

    op->setAttr("xla.cpu.is_wrapped", mlir::UnitAttr::get(context));
    op.setPrivate();

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

  static llvm::SmallVector<mlir::Value, 3> GetWorkGroupIds(
      mlir::Value call_frame, mlir::ImplicitLocOpBuilder& builder) {
    mlir::MLIRContext* context = builder.getContext();
    auto ptr = builder.getType<mlir::LLVM::LLVMPointerType>();
    auto kernel_call_frame = KernelCallFrameType(context);
    auto kernel_dim = KernelDim3Type(context);
    auto i64_ty = builder.getIntegerType(
        mlir::DataLayout::closest(builder.getInsertionBlock()->getParentOp())
            .getTypeSizeInBits(builder.getI64Type()));

    // Get a pointer to the `WorkGroupThread` struct.
    auto cast =
        builder.create<mlir::UnrealizedConversionCastOp>(ptr, call_frame)
            .getResult(0);
    auto workgroup_gep = builder.create<mlir::LLVM::GEPOp>(
        ptr, kernel_call_frame, cast,
        mlir::ArrayRef<mlir::LLVM::GEPArg>{mlir::LLVM::GEPArg(0),
                                           mlir::LLVM::GEPArg(1)},
        mlir::LLVM::GEPNoWrapFlags::inbounds);
    auto workgroup_ptr = builder.create<mlir::LLVM::LoadOp>(ptr, workgroup_gep);

    llvm::SmallVector<mlir::Value, 3> workgroup_ids;
    for (int32_t workgroup_dim_idx : {0, 1, 2}) {
      auto workgroup_dim_gep = builder.create<mlir::LLVM::GEPOp>(
          ptr, kernel_dim, workgroup_ptr,
          mlir::ArrayRef<mlir::LLVM::GEPArg>{
              mlir::LLVM::GEPArg(0), mlir::LLVM::GEPArg(workgroup_dim_idx)},
          mlir::LLVM::GEPNoWrapFlags::inbounds);
      auto workgroup_dim_load =
          builder.create<mlir::LLVM::LoadOp>(i64_ty, workgroup_dim_gep);

      mlir::Value workgroup_dim = workgroup_dim_load.getResult();
      auto index_ty = builder.getIntegerType(
          mlir::DataLayout::closest(builder.getInsertionBlock()->getParentOp())
              .getTypeSizeInBits(mlir::IndexType::get(context)));
      if (index_ty != i64_ty) {
        workgroup_dim =
            builder.create<mlir::LLVM::TruncOp>(index_ty, workgroup_dim);
      }
      auto workgroup_dim_cast =
          builder.create<mlir::UnrealizedConversionCastOp>(
              mlir::IndexType::get(context), workgroup_dim);

      workgroup_ids.push_back(workgroup_dim_cast.getResult(0));
    }

    return workgroup_ids;
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

}  // namespace

void PopulateXlaCpuConversionPatterns(mlir::RewritePatternSet& patterns,
                                      int32_t vector_width) {
  patterns.add<LowerLoadOp, LowerWorkGroupIdOp, LowerSuccessOp,
               RewriteFunctionSignatures>(patterns.getContext());
  patterns.add<WrapEntryWithCallFrame>(patterns.getContext(), vector_width);
}

}  // namespace xla::cpu
