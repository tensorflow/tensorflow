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

#include "xla/backends/gpu/codegen/slice_to_dynamic.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ArithAttributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialectDecl.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/backends/gpu/codegen/gpu_kernel_spec.h"
#include "xla/backends/gpu/codegen/kernel_compiler.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/kernel_api_builder.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

mlir::Value EmitThreadIdX(mlir::ImplicitLocOpBuilder& b,
                          const LaunchDimensions& launch_dimensions) {
  auto tid = mlir::gpu::ThreadIdOp::create(b, mlir::gpu::Dimension::x);
  tid->setAttr(
      "xla.range",
      b.getIndexArrayAttr(
          {0, static_cast<int64_t>(
                  launch_dimensions.thread_counts_per_block().x - 1)}));
  return tid;
}

mlir::Value EmitBlockId(mlir::ImplicitLocOpBuilder& b, mlir::gpu::Dimension dim,
                        int64_t count) {
  auto bid = mlir::gpu::BlockIdOp::create(b, dim);
  bid->setAttr("xla.range", b.getIndexArrayAttr({0, count - 1}));
  return bid;
}

}  // namespace

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
CreateSliceToDynamicMlirModule(mlir::MLIRContext& mlir_context,
                               const HloCustomCallInstruction* instr,
                               const BufferAssignment* buffer_assignment,
                               absl::string_view entry_function_name,
                               const LaunchDimensions& launch_dimensions) {
  TF_RET_CHECK(instr != nullptr);
  TF_RET_CHECK(instr->opcode() == HloOpcode::kCustomCall &&
               instr->custom_call_target() == "SliceToDynamic")
      << "Expected SliceToDynamic custom call: " << instr->ToString();
  TF_RET_CHECK(instr->shape().IsArray())
      << "Expected array output shape: " << instr->ToString();
  const Shape output_shape = ShapeUtil::MakeStaticShape(instr->shape());
  PrimitiveType elem_type = output_shape.element_type();
  if (primitive_util::IsSubByteNonPredType(elem_type)) {
    int bit_width = primitive_util::BitWidth(elem_type);
    if (bit_width != 2 && bit_width != 4) {
      return absl::UnimplementedError(
          absl::StrCat("SliceToDynamic does not support sub-byte type ",
                       primitive_util::LowercasePrimitiveTypeName(elem_type)));
    }
  }
  const int64_t rank = output_shape.dimensions().size();
  TF_RET_CHECK(instr->operand_count() == 1 + rank)
      << "Expected one dynamic size operand per dimension: "
      << instr->ToString();
  TF_RET_CHECK(ShapeUtil::Compatible(instr->operand(0)->shape(), output_shape))
      << "Expected input shape compatible with output shape: "
      << instr->ToString();
  for (int64_t i = 0; i < rank; ++i) {
    TF_RET_CHECK(Shape::Equal()(instr->operand(i + 1)->shape(),
                                ShapeUtil::MakeScalarShape(S32)));
  }
  TF_RET_CHECK(launch_dimensions.block_counts().z == 1 &&
               launch_dimensions.thread_counts_per_block().y == 1 &&
               launch_dimensions.thread_counts_per_block().z == 1);
  if (!ShapeUtil::IsZeroElementArray(output_shape)) {
    TF_RET_CHECK(launch_dimensions.launch_bound() >=
                 ShapeUtil::ElementsIn(output_shape));
  }

  mlir::OpBuilder builder(&mlir_context);
  mlir::Location loc = mlir::NameLoc::get(builder.getStringAttr(instr->name()));
  mlir::OwningOpRef<mlir::ModuleOp> module = llvm_ir::CreateMlirModuleOp(loc);

  bool force_64_bit =
      launch_dimensions.launch_bound() > std::numeric_limits<int32_t>::max();
  emitters::SetIndexDataLayout(*module, *instr, force_64_bit);

  ABSL_ASSIGN_OR_RETURN(mlir::func::FuncOp entry_func,
                   emitters::EmitKernelApi(*module, *instr, buffer_assignment,
                                           GetDefaultBufferAlignment(),
                                           entry_function_name));
  SetBackendKind(&mlir_context, entry_func, BackendKind::kGpu);

  mlir::ImplicitLocOpBuilder b(loc, &mlir_context);
  b.setInsertionPointToStart(entry_func.addEntryBlock());

  mlir::Value src = entry_func.getArgument(0);
  mlir::ValueRange dyn_dim_args = entry_func.getArguments().slice(1, rank);
  mlir::Value out = entry_func.getArgument(1 + rank);

  if (rank == 0) {
    mlir::Value val =
        mlir::tensor::ExtractOp::create(b, src, mlir::ValueRange{});
    mlir::Value updated_out =
        mlir::tensor::InsertOp::create(b, val, out, mlir::ValueRange{});
    mlir::func::ReturnOp::create(b, updated_out);
    return module;
  }

  llvm::SmallVector<mlir::Value> dyn_dims_i32(rank);
  llvm::SmallVector<mlir::Value> dyn_dim_bounds(rank);
  for (int64_t d = 0; d < rank; ++d) {
    dyn_dims_i32[d] =
        mlir::tensor::ExtractOp::create(b, dyn_dim_args[d], mlir::ValueRange{});
    dyn_dim_bounds[d] =
        mlir::arith::IndexCastOp::create(b, b.getIndexType(), dyn_dims_i32[d]);
  }

  mlir::Value tid_x = EmitThreadIdX(b, launch_dimensions);
  mlir::Value bid_x = EmitBlockId(b, mlir::gpu::Dimension::x,
                                  launch_dimensions.block_counts().x);
  mlir::Value bid_y;
  if (launch_dimensions.block_counts().y > 1) {
    bid_y = EmitBlockId(b, mlir::gpu::Dimension::y,
                        launch_dimensions.block_counts().y);
  }

  mlir::Value c0 = mlir::arith::ConstantIndexOp::create(b, 0);
  mlir::Value is_tid_0 =
      mlir::arith::CmpIOp::create(b, mlir::arith::CmpIPredicate::eq, tid_x, c0);
  mlir::Value is_bid_0 =
      mlir::arith::CmpIOp::create(b, mlir::arith::CmpIPredicate::eq, bid_x, c0);
  mlir::Value is_thread_0 = mlir::arith::AndIOp::create(b, is_tid_0, is_bid_0);
  if (bid_y) {
    mlir::Value is_bid_y_0 = mlir::arith::CmpIOp::create(
        b, mlir::arith::CmpIPredicate::eq, bid_y, c0);
    is_thread_0 = mlir::arith::AndIOp::create(b, is_thread_0, is_bid_y_0);
  }

  auto thread0_if = mlir::scf::IfOp::create(b, out.getType(), is_thread_0,
                                            /*withElseRegion=*/true);
  {
    mlir::ImplicitLocOpBuilder then_b(loc, thread0_if.getThenBodyBuilder());
    mlir::Value updated = out;
    for (int64_t d = 0; d < rank; ++d) {
      updated =
          SetDynamicDimSizeOp::create(then_b, updated, dyn_dims_i32[d], d);
    }
    mlir::scf::YieldOp::create(then_b, updated);

    mlir::ImplicitLocOpBuilder else_b(loc, thread0_if.getElseBodyBuilder());
    mlir::scf::YieldOp::create(else_b, out);
  }
  mlir::Value updated_out = thread0_if.getResult(0);

  if (!ShapeUtil::IsZeroElementArray(output_shape)) {
    mlir::Value dyn_element_total = dyn_dim_bounds[0];
    for (int64_t d = 1; d < rank; ++d) {
      dyn_element_total =
          mlir::arith::MulIOp::create(b, dyn_element_total, dyn_dim_bounds[d]);
    }

    mlir::Value block_idx = bid_x;
    if (bid_y) {
      mlir::Value num_blocks_x = mlir::arith::ConstantIndexOp::create(
          b, launch_dimensions.block_counts().x);
      mlir::Value bid_y_scaled =
          mlir::arith::MulIOp::create(b, bid_y, num_blocks_x);
      block_idx = mlir::arith::AddIOp::create(b, bid_y_scaled, bid_x);
    }
    mlir::Value threads_per_block = mlir::arith::ConstantIndexOp::create(
        b, launch_dimensions.thread_counts_per_block().x);
    mlir::Value block_offset =
        mlir::arith::MulIOp::create(b, block_idx, threads_per_block);
    mlir::Value linear_index =
        mlir::arith::AddIOp::create(b, block_offset, tid_x);

    mlir::Value num_elements = mlir::arith::ConstantIndexOp::create(
        b, ShapeUtil::ElementsIn(output_shape));
    mlir::Value in_static_bounds = mlir::arith::CmpIOp::create(
        b, mlir::arith::CmpIPredicate::ult, linear_index, num_elements);
    mlir::Value in_dyn_bounds = mlir::arith::CmpIOp::create(
        b, mlir::arith::CmpIPredicate::ult, linear_index, dyn_element_total);
    mlir::Value in_bounds =
        mlir::arith::AndIOp::create(b, in_static_bounds, in_dyn_bounds);

    auto copy_if = mlir::scf::IfOp::create(b, out.getType(), in_bounds,
                                           /*withElseRegion=*/true);
    {
      mlir::ImplicitLocOpBuilder then_b(loc, copy_if.getThenBodyBuilder());
      llvm::SmallVector<mlir::Value> dyn_coords(rank);
      mlir::Value remaining = linear_index;
      absl::Span<const int64_t> minor_to_major =
          LayoutUtil::MinorToMajor(output_shape);
      for (int64_t i = 0; i < rank; ++i) {
        int64_t dim = minor_to_major[i];
        if (i < rank - 1) {
          mlir::Value bound = dyn_dim_bounds[dim];
          dyn_coords[dim] =
              mlir::arith::RemUIOp::create(then_b, remaining, bound);
          remaining = mlir::arith::DivUIOp::create(then_b, remaining, bound);
        } else {
          dyn_coords[dim] = remaining;
        }
      }
      mlir::Value val =
          mlir::tensor::ExtractOp::create(then_b, src, dyn_coords);
      mlir::Value inserted =
          LinearInsertOp::create(then_b, val, updated_out, linear_index);
      mlir::scf::YieldOp::create(then_b, inserted);

      mlir::ImplicitLocOpBuilder else_b(loc, copy_if.getElseBodyBuilder());
      mlir::scf::YieldOp::create(else_b, updated_out);
    }
    updated_out = copy_if.getResult(0);
  }

  mlir::func::ReturnOp::create(b, updated_out);
  return module;
}

AsyncThunkSequence EmitSliceToDynamic(const HloCustomCallInstruction* instr,
                                      IrEmitterContext* ir_emitter_context) {
  TF_RET_CHECK(instr != nullptr);
  TF_RET_CHECK(ir_emitter_context != nullptr);
  TF_RET_CHECK(instr->shape().IsArray())
      << "Expected array output shape: " << instr->ToString();

  const Shape output_shape = ShapeUtil::MakeStaticShape(instr->shape());
  constexpr int kUnrollFactor = 1;
  LaunchDimensions launch_dimensions = CalculateLaunchDimensions(
      output_shape, ir_emitter_context->gpu_device_info(), kUnrollFactor);

  ABSL_ASSIGN_OR_RETURN(
      emitters::KernelArguments kernel_arguments,
      emitters::KernelArguments::Create(ir_emitter_context->buffer_assignment(),
                                        GetDefaultBufferAlignment(), instr));

  std::string sanitized_kernel_name =
      ir_emitter_context->GetSanitizedUniqueName(std::string(instr->name()));
  Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context->GetNextThunkId());

  BorrowedMlirContext borrowed_context =
      ir_emitter_context->BorrowMlirContext();
  mlir::MLIRContext* mlir_context = borrowed_context->get();
  mlir_context->appendDialectRegistry(MlirKernelEmitter::GetDialectRegistry());
  mlir_context->loadAllAvailableDialects();

  ABSL_ASSIGN_OR_RETURN(
      mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
      CreateSliceToDynamicMlirModule(*mlir_context, instr,
                                     &ir_emitter_context->buffer_assignment(),
                                     sanitized_kernel_name, launch_dimensions));

  bool use_pdl = IsPdlEnabled(
      instr->GetModule()->config().debug_options(),
      ir_emitter_context->gpu_device_info().gpu_compute_capability());
  GpuKernelSpec spec(std::move(sanitized_kernel_name), launch_dimensions,
                     std::move(kernel_arguments), std::nullopt, use_pdl);
  MlirKernelDefinition kernel_def(
      std::move(spec), MlirKernelSource(nullptr, std::move(mlir_module)));
  return ir_emitter_context->kernel_compiler()
      ->Compile(std::move(thunk_info), std::move(kernel_def),
                std::move(borrowed_context), *instr->GetModule())
      .Map([](std::unique_ptr<Thunk> thunk) {
        return ThunkSequence::Of(std::move(thunk));
      });
}

}  // namespace xla::gpu
