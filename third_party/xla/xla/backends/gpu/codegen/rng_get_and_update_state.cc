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

#include "xla/backends/gpu/codegen/rng_get_and_update_state.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
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

// Use a non zero initial value so the first random number generation passes
// chi square tests.
constexpr uint64_t kRngStateInitialLow = 0x7012395ULL;
constexpr unsigned kGlobalMemoryAddressSpace = 1;
constexpr int64_t kStateAlignmentBytes = 16;

}  // namespace

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
CreateRngGetAndUpdateStateMlirModule(
    mlir::MLIRContext& mlir_context,
    const HloRngGetAndUpdateStateInstruction* instr,
    const BufferAssignment* buffer_assignment,
    absl::string_view entry_function_name) {
  TF_RET_CHECK(instr != nullptr);
  TF_RET_CHECK(instr->opcode() == HloOpcode::kRngGetAndUpdateState)
      << "Expected RngGetAndUpdateState instruction: " << instr->ToString();
  TF_RET_CHECK(instr->operand_count() == 0);
  TF_RET_CHECK(Shape::Equal().IgnoreLayout()(instr->shape(),
                                             ShapeUtil::MakeShape(U64, {2})))
      << "Expected u64[2] output shape: " << instr->ToString();

  mlir_context.loadDialect<mlir::LLVM::LLVMDialect>();

  mlir::OpBuilder builder(&mlir_context);
  mlir::Location loc = mlir::NameLoc::get(builder.getStringAttr(instr->name()));
  mlir::OwningOpRef<mlir::ModuleOp> module = llvm_ir::CreateMlirModuleOp(loc);
  emitters::SetIndexDataLayout(*module, *instr);

  ABSL_ASSIGN_OR_RETURN(mlir::func::FuncOp entry_func,
                   emitters::EmitKernelApi(*module, *instr, buffer_assignment,
                                           GetDefaultBufferAlignment(),
                                           entry_function_name));
  SetBackendKind(&mlir_context, entry_func, BackendKind::kGpu);

  mlir::OpBuilder module_builder =
      mlir::OpBuilder::atBlockBegin(module->getBody());
  mlir::ImplicitLocOpBuilder b(loc, &mlir_context);
  b.setInsertionPointToStart(entry_func.addEntryBlock());

  mlir::Value out = entry_func.getArgument(0);
  auto ptr_type = mlir::LLVM::LLVMPointerType::get(&mlir_context,
                                                   kGlobalMemoryAddressSpace);
  mlir::Type i64_type = b.getI64Type();
  // Represent the 128 bit state as vector<2xi64> so CUDA/ROCm get 16 byte
  // vector loads/stores and SPIR-V avoids i128 without losing state bits.
  mlir::VectorType state_type = mlir::VectorType::get({2}, i64_type);

  auto init_attr = mlir::DenseElementsAttr::get(
      state_type,
      llvm::ArrayRef<int64_t>{static_cast<int64_t>(kRngStateInitialLow), 0});
  mlir::LLVM::GlobalOp::create(
      module_builder, loc, state_type, /*isConstant=*/false,
      mlir::LLVM::Linkage::Private, "rng_state", init_attr,
      kStateAlignmentBytes, kGlobalMemoryAddressSpace);

  mlir::Value state_ptr =
      mlir::LLVM::AddressOfOp::create(b, ptr_type, "rng_state");
  mlir::Value old_state = mlir::LLVM::LoadOp::create(b, state_type, state_ptr,
                                                     kStateAlignmentBytes);
  mlir::Value old_low = mlir::vector::ExtractOp::create(b, old_state, 0);
  mlir::Value old_high = mlir::vector::ExtractOp::create(b, old_state, 1);

  mlir::Value delta =
      mlir::arith::ConstantIntOp::create(b, i64_type, instr->delta());
  auto add_lo = mlir::arith::AddUIExtendedOp::create(b, old_low, delta);
  mlir::Value new_low = add_lo.getSum();
  mlir::Value carry =
      mlir::arith::ExtUIOp::create(b, i64_type, add_lo.getOverflow());
  mlir::Value new_high = mlir::arith::AddIOp::create(b, old_high, carry);

  mlir::Value new_state = mlir::vector::FromElementsOp::create(
      b, state_type, mlir::ValueRange{new_low, new_high});
  mlir::LLVM::StoreOp::create(b, new_state, state_ptr, kStateAlignmentBytes);

  mlir::Value c0 = mlir::arith::ConstantIndexOp::create(b, 0);
  mlir::Value updated =
      mlir::vector::TransferWriteOp::create(
          b, old_state, out, mlir::ValueRange{c0}, llvm::ArrayRef<bool>{true})
          .getResult();
  mlir::func::ReturnOp::create(b, updated);
  return module;
}

AsyncThunkSequence EmitRngGetAndUpdateState(
    const HloRngGetAndUpdateStateInstruction* instr,
    IrEmitterContext* ir_emitter_context) {
  TF_RET_CHECK(instr != nullptr);
  TF_RET_CHECK(ir_emitter_context != nullptr);

  LaunchDimensions launch_dimensions;
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
      CreateRngGetAndUpdateStateMlirModule(
          *mlir_context, instr, &ir_emitter_context->buffer_assignment(),
          sanitized_kernel_name));

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
