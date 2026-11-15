/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/cpu/codegen/emitters/pad_to_static_kernel_emitter.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "xla/backends/cpu/alignment.h"
#include "xla/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "xla/backends/cpu/codegen/symbol_name_util.h"
#include "xla/codegen/emitters/ir/xla_attrs.h.inc"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/kernel_api_builder.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/primitive_util.h"
#include "xla/runtime/work_dimensions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

PadToStaticKernelEmitter::PadToStaticKernelEmitter(
    mlir::MLIRContext& mlir_context, const HloInstruction* instr,
    const BufferAssignment* buffer_assignment, bool use_unique_c_name)
    : mlir_context_(mlir_context),
      instr_(instr),
      buffer_assignment_(buffer_assignment),
      use_unique_c_name_(use_unique_c_name) {}

absl::StatusOr<PadToStaticKernelEmitter::KernelDefinition>
PadToStaticKernelEmitter::EmitKernelDefinition() {
  TF_RET_CHECK(instr_ != nullptr);
  TF_RET_CHECK(instr_->opcode() == HloOpcode::kCustomCall &&
               instr_->custom_call_target() == "PadToStatic")
      << "Expected PadToStatic custom call: " << instr_->ToString();
  TF_RET_CHECK(instr_->operand_count() == 1);
  const Shape& input_shape = instr_->operand(0)->shape();
  TF_RET_CHECK(input_shape.IsArray())
      << "Expected array input shape: " << instr_->ToString();
  PrimitiveType elem_type = input_shape.element_type();
  if (primitive_util::IsSubByteNonPredType(elem_type)) {
    int bit_width = primitive_util::BitWidth(elem_type);
    if (bit_width != 2 && bit_width != 4) {
      return absl::UnimplementedError(
          absl::StrCat("PadToStatic does not support sub-byte type ",
                       primitive_util::LowercasePrimitiveTypeName(elem_type)));
    }
  }
  const int64_t rank = input_shape.dimensions().size();
  TF_RET_CHECK(instr_->shape().IsTuple() &&
               instr_->shape().tuple_shapes().size() ==
                   static_cast<size_t>(1 + rank))
      << "Expected tuple of static array and dimension sizes: "
      << instr_->ToString();

  const HloModule* hlo_module = instr_->GetModule();
  TF_RET_CHECK(hlo_module != nullptr);

  std::string kernel_name(instr_->name());
  if (use_unique_c_name_) {
    ABSL_ASSIGN_OR_RETURN(
        kernel_name,
        ConvertToCName(absl::StrCat(hlo_module->name(), "_", instr_->name())));
  }

  mlir::OpBuilder builder(&mlir_context_);
  auto loc = mlir::NameLoc::get(builder.getStringAttr(kernel_name));
  mlir::OwningOpRef<mlir::ModuleOp> module =
      llvm_ir::CreateMlirModuleOp(loc, kernel_name);

  module->getOperation()->setAttr(
      xla::CpuMemoryRegionNameAttr::name,
      builder.getStringAttr(BuildModuleMemoryRegionName(name(), instr_)));
  emitters::SetIndexDataLayout(*module, *instr_);

  emitters::KernelArguments::BufferAlignment buffer_alignment{
      MinAlign(), MinAlign(), MinAlign()};
  ABSL_ASSIGN_OR_RETURN(mlir::func::FuncOp entry_func,
                   emitters::EmitKernelApi(*module, *instr_, buffer_assignment_,
                                           buffer_alignment, kernel_name));
  SetBackendKind(&mlir_context_, entry_func, BackendKind::kCpu);

  mlir::ImplicitLocOpBuilder b(loc, &mlir_context_);
  b.setInsertionPointToStart(entry_func.addEntryBlock());

  mlir::Value src = entry_func.getArgument(0);
  mlir::Value out_data = entry_func.getArgument(1);
  mlir::ValueRange out_dims = entry_func.getArguments().drop_front(2);

  llvm::SmallVector<mlir::Value> results;
  results.reserve(1 + rank);
  results.push_back(out_data);

  llvm::SmallVector<mlir::Value> dyn_dim_bounds(rank);
  for (int64_t d = 0; d < rank; ++d) {
    mlir::Value dim_i32 = GetDynamicDimSizeOp::create(b, src, d);
    mlir::Value updated_dim = mlir::tensor::InsertOp::create(
        b, dim_i32, out_dims[d], mlir::ValueRange{});
    results.push_back(updated_dim);
    dyn_dim_bounds[d] =
        mlir::arith::IndexCastOp::create(b, b.getIndexType(), dim_i32);
  }

  if (rank == 0) {
    mlir::Value val =
        mlir::tensor::ExtractOp::create(b, src, mlir::ValueRange{});
    results[0] =
        mlir::tensor::InsertOp::create(b, val, out_data, mlir::ValueRange{});
  } else {
    mlir::Value c0 = mlir::arith::ConstantIndexOp::create(b, 0);
    mlir::Value c1 = mlir::arith::ConstantIndexOp::create(b, 1);

    // Loop in physical major to minor order so reads from the packed dynamic
    // buffer advance sequentially by 1 element without integer division.
    llvm::SmallVector<int64_t> major_to_minor(
        llvm::reverse(LayoutUtil::MinorToMajor(input_shape)));
    llvm::SmallVector<mlir::Value> multi_index(rank);

    std::function<std::pair<mlir::Value, mlir::Value>(
        int64_t, mlir::ImplicitLocOpBuilder&, mlir::Value, mlir::Value)>
        emit_loops = [&](int64_t level, mlir::ImplicitLocOpBuilder& loop_b,
                         mlir::Value current_out, mlir::Value current_src_idx)
        -> std::pair<mlir::Value, mlir::Value> {
      if (level == rank) {
        mlir::Value val = LinearExtractOp::create(loop_b, src, current_src_idx);
        mlir::Value next_out = mlir::tensor::InsertOp::create(
            loop_b, val, current_out, multi_index);
        mlir::Value next_src_idx =
            mlir::arith::AddIOp::create(loop_b, current_src_idx, c1);
        return {next_out, next_src_idx};
      }
      int64_t dim = major_to_minor[level];
      auto for_op = mlir::scf::ForOp::create(
          loop_b, c0, dyn_dim_bounds[dim], c1,
          mlir::ValueRange{current_out, current_src_idx},
          [&](mlir::OpBuilder& body_builder, mlir::Location body_loc,
              mlir::Value iv, mlir::ValueRange iter_args) {
            mlir::ImplicitLocOpBuilder nested_b(body_loc, body_builder);
            multi_index[dim] = iv;
            auto [next_out, next_src_idx] =
                emit_loops(level + 1, nested_b, iter_args[0], iter_args[1]);
            mlir::scf::YieldOp::create(
                nested_b, mlir::ValueRange{next_out, next_src_idx});
          });
      return {for_op.getResult(0), for_op.getResult(1)};
    };

    results[0] = emit_loops(0, b, out_data, c0).first;
  }

  mlir::func::ReturnOp::create(b, results);

  ABSL_ASSIGN_OR_RETURN(
      KernelSpec kernel_spec,
      emitters::GetKernelSpec(kernel_name, *instr_, buffer_assignment_,
                              WorkDimensions{}));
  return KernelDefinition(std::move(kernel_spec),
                          MlirKernelSource(nullptr, std::move(module)));
}

}  // namespace xla::cpu
