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

#include "xla/service/cpu/ir_emitter2.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/cpu/elemental_math_emitter.h"
#include "xla/service/cpu/ir_emitter.h"
#include "xla/service/elemental_ir_emitter.h"
#include "xla/service/llvm_ir/fused_ir_emitter.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/llvm_ir/loop_emitter.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla::cpu {
namespace {

// We do not materialize buffers for tuples at run time, and work only with leaf
// arrays. These are the helper functions to flatten HLO instruction parameters
// and results into a list of leaf shapes.

static std::vector<Shape> FlattenedParameters(const HloInstruction* instr) {
  std::vector<Shape> parameters;
  for (auto* operand : instr->operands()) {
    for (auto& indexed : ShapeUtil::GetLeafShapes(operand->shape())) {
      parameters.push_back(indexed.shape);
    }
  }
  return parameters;
}

static std::vector<Shape> FlattenedResults(const HloInstruction* instr) {
  std::vector<Shape> results;
  for (auto& indexed : ShapeUtil::GetLeafShapes(instr->shape())) {
    results.push_back(indexed.shape);
  }
  return results;
}

// Following struct types correspond to HostKernel C API.
// See: xla/stream_executor/host/host_kernel_c_api.h

static llvm::StructType* Dim3StructTy(llvm::LLVMContext& ctx,
                                      std::string_view name) {
  auto* i64 = llvm::IntegerType::getInt64Ty(ctx);
  return llvm::StructType::create(name, i64, i64, i64);
}

static llvm::StructType* KernelThreadDimTy(llvm::LLVMContext& ctx) {
  return Dim3StructTy(ctx, "SE_HOST_KernelThreadDim");
}

static llvm::StructType* KernelThreadTy(llvm::LLVMContext& ctx) {
  return Dim3StructTy(ctx, "SE_HOST_KernelThread");
}

static llvm::StructType* KernelArgTy(llvm::LLVMContext& ctx) {
  auto* ptr = llvm::PointerType::getUnqual(ctx);
  auto* i64 = llvm::IntegerType::getInt64Ty(ctx);
  return llvm::StructType::create("SE_HOST_KernelArg", ptr, i64);
}

static llvm::StructType* KernelCallFrameTy(llvm::LLVMContext& ctx) {
  auto* ptr = llvm::PointerType::getUnqual(ctx);
  auto* i64 = llvm::IntegerType::getInt64Ty(ctx);
  return llvm::StructType::create("SE_HOST_KernelCallFrame", ptr, ptr, i64,
                                  ptr);
}

static llvm::FunctionType* KernelFunctionTy(llvm::LLVMContext& ctx) {
  return llvm::FunctionType::get(llvm::PointerType::getUnqual(ctx),
                                 llvm::PointerType::getUnqual(ctx),
                                 /*isVarArg=*/false);
}

}  // namespace

//===----------------------------------------------------------------------===//
// ElementalIrEmitter
//===----------------------------------------------------------------------===//

class IrEmitter2::ElementalIrEmitter : public xla::ElementalIrEmitter {
 public:
  ElementalIrEmitter(llvm::Module* module, llvm::IRBuilder<>* b,
                     const HloModule* hlo_module, IrEmitter* nested_ir_emitter,
                     bool fast_min_max)
      : xla::ElementalIrEmitter(module, b),
        hlo_module_(hlo_module),
        nested_ir_emitter_(nested_ir_emitter),
        fast_min_max_(fast_min_max) {}

 protected:
  absl::StatusOr<llvm::Value*> EmitAtan2(PrimitiveType prim_type,
                                         llvm::Value* lhs, llvm::Value* rhs,
                                         absl::string_view) override {
    return xla::cpu::EmitAtan2(module(), *b(), prim_type, lhs, rhs);
  }

  absl::StatusOr<llvm::Value*> EmitTanh(PrimitiveType prim_type,
                                        llvm::Value* value) override {
    return xla::cpu::EmitTanh(module(), *b(), prim_type, value);
  }

  absl::StatusOr<llvm::Value*> EmitErf(PrimitiveType prim_type,
                                       llvm::Value* value) override {
    return xla::cpu::EmitErf(module(), *b(), prim_type, value);
  }

  absl::StatusOr<std::vector<llvm::Value*>> EmitThreadLocalCall(
      const HloComputation& callee, absl::Span<llvm::Value* const> parameters,
      absl::string_view name, bool is_reducer) override {
    // Module must be scheduled to emit thread local computation.
    if (!hlo_module_ || !hlo_module_->has_schedule()) {
      return absl::InternalError(
          "HLO module must be scheduled to emit thread local computation.");
    }
    // Create a nested function for thread local computation if it is not
    // already created. Nested functions are created with internal linkage.
    if (!nested_ir_emitter_->is_computation_emitted(callee, is_reducer)) {
      VLOG(2) << "Emit nested computation: " << callee.name();
      TF_RETURN_IF_ERROR(
          nested_ir_emitter_
              ->EmitComputation(
                  const_cast<HloComputation*>(&callee), name, false,
                  hlo_module_->schedule().sequence(&callee).instructions(),
                  /*allow_reassociation=*/is_reducer)
              .status());
    }

    // Add a thread local call to the nested computation.
    VLOG(2) << "Emit thread local call to: " << callee.name();
    nested_ir_emitter_->b()->SetInsertPoint(b()->GetInsertPoint());
    auto values = nested_ir_emitter_->EmitThreadLocalCall(
        callee, parameters, name, is_reducer, /*in_compute_function=*/false);

    return values;
  }

  bool fast_min_max() override { return fast_min_max_; }

 private:
  const HloModule* hlo_module_;
  IrEmitter* nested_ir_emitter_;
  bool fast_min_max_;
};

//===----------------------------------------------------------------------===//
// IrEmitter2
//===----------------------------------------------------------------------===//

IrEmitter2::IrEmitter2(const HloModule& hlo_module, llvm::Module* module,
                       IrEmitter* nested_ir_emitter)
    : hlo_module_(hlo_module),
      module_(module),
      nested_ir_emitter_(nested_ir_emitter),
      call_frame_ty_(KernelCallFrameTy(module_->getContext())),
      thread_dims_ty_(KernelThreadDimTy(module_->getContext())),
      thread_ty_(KernelThreadTy(module_->getContext())),
      arg_ty_(KernelArgTy(module_->getContext())) {}

bool IrEmitter2::fast_min_max() const {
  return hlo_module_.config().debug_options().xla_cpu_enable_fast_min_max();
}

static absl::Status EmitElementalLoops(
    llvm::IRBuilder<>& b, const HloInstruction* instr,
    const llvm_ir::ElementGenerator& element_generator,
    absl::Span<const llvm_ir::IrArray> results) {
  // We can emit loops for instruction with multiple results only if it is a
  // fusion, reduce or reduce window.
  bool multiple_results = results.size() > 1;
  bool support_multiple_results = instr->opcode() == HloOpcode::kFusion ||
                                  instr->opcode() == HloOpcode::kReduce ||
                                  instr->opcode() == HloOpcode::kReduceWindow;

  if (multiple_results && !support_multiple_results) {
    return Internal(
        "Multi-output host kernels are not supported for %s instruction",
        HloOpcodeString(instr->opcode()));
  }

  if (multiple_results) {
    TF_RETURN_IF_ERROR(llvm_ir::LoopEmitter(element_generator, results, &b)
                           .EmitLoop(llvm_ir::IrName(instr)));
  } else {
    TF_RETURN_IF_ERROR(
        llvm_ir::LoopEmitter(element_generator, results.front(), &b)
            .EmitLoop(llvm_ir::IrName(instr)));
  }

  return absl::OkStatus();
}

absl::StatusOr<IrEmitter2::KernelInfo> IrEmitter2::EmitElementalHostKernel(
    const HloInstruction* instr) {
  VLOG(2) << "Emit elemental host kernel: " << instr->name();

  KernelPrototype kernel_prototype = EmitKernelPrototype(instr);

  llvm::IRBuilder<> b(module_->getContext());
  b.SetInsertPoint(kernel_prototype.function->getEntryBlock().getTerminator());

  ElementalIrEmitter::HloToElementGeneratorMap operand_to_generator;
  for (int64_t i = 0; i < instr->operand_count(); ++i) {
    const HloInstruction* operand = instr->operand(i);
    operand_to_generator[operand] = [&, i](const llvm_ir::IrArray::Index& idx) {
      return kernel_prototype.arguments[i].EmitReadArrayElement(idx, &b);
    };
  }

  ElementalIrEmitter elemental_emitter(module_, &b, &hlo_module_,
                                       nested_ir_emitter_, fast_min_max());
  llvm_ir::ElementGenerator element_generator =
      elemental_emitter.MakeElementGenerator(instr, operand_to_generator);

  TF_RETURN_IF_ERROR(EmitElementalLoops(b, instr, element_generator,
                                        kernel_prototype.results));
  return kernels_.emplace_back(kernel_prototype.function->getName().str());
}

absl::StatusOr<IrEmitter2::KernelInfo> IrEmitter2::EmitFusionHostKernel(
    const HloFusionInstruction* fusion) {
  VLOG(2) << "Emit fusion host kernel: " << fusion->name();

  if (fusion->fusion_kind() != HloInstruction::FusionKind::kLoop) {
    return absl::InternalError(absl::StrCat(
        "Unsupported loop fusion kind for instruction: ", fusion->ToString()));
  }

  KernelPrototype kernel_prototype = EmitKernelPrototype(fusion);

  llvm::IRBuilder<> b(module_->getContext());
  b.SetInsertPoint(kernel_prototype.function->getEntryBlock().getTerminator());

  ElementalIrEmitter elemental_emitter(module_, &b, &hlo_module_,
                                       nested_ir_emitter_, fast_min_max());
  FusedIrEmitter fused_emitter(elemental_emitter);

  for (int i = 0; i < fusion->operand_count(); i++) {
    fused_emitter.BindGenerator(
        *fusion->fused_parameter(i), [&, i](llvm_ir::IrArray::Index idx) {
          return kernel_prototype.arguments[i].EmitReadArrayElement(idx, &b);
        });
  }

  TF_ASSIGN_OR_RETURN(
      auto element_generator,
      fused_emitter.GetGenerator(*fusion->fused_expression_root()));

  TF_RETURN_IF_ERROR(EmitElementalLoops(b, fusion, element_generator,
                                        kernel_prototype.results));
  return kernels_.emplace_back(kernel_prototype.function->getName().str());
}

absl::StatusOr<IrEmitter2::KernelInfo> IrEmitter2::EmitReductionHostKernel(
    const HloInstruction* instr) {
  VLOG(2) << "Emit reduction host kernel: " << instr->name();

  // TODO(ezhulenev): Port vectorized reduction emitter from IrEmitter.
  return EmitElementalHostKernel(instr);
}

//===----------------------------------------------------------------------===//
// Building HostKernel prototypes.
//===----------------------------------------------------------------------===//

IrEmitter2::KernelThreadDims IrEmitter2::EmitKernelThreadDims(
    llvm::IRBuilder<>& b, llvm::Value* call_frame) {
  auto* tdims = b.CreateStructGEP(call_frame_ty_, call_frame, 0, "tdims_gep");
  auto* x_gep = b.CreateStructGEP(thread_dims_ty_, tdims, 0, "tdim_x_gep");
  auto* y_gep = b.CreateStructGEP(thread_dims_ty_, tdims, 1, "tdim_y_gep");
  auto* z_gep = b.CreateStructGEP(thread_dims_ty_, tdims, 2, "tdim_z_gep");

  return {b.CreateLoad(b.getInt64Ty(), x_gep, "tdim_x"),
          b.CreateLoad(b.getInt64Ty(), y_gep, "tdim_y"),
          b.CreateLoad(b.getInt64Ty(), z_gep, "tdim_z")};
}

IrEmitter2::KernelThread IrEmitter2::EmitKernelThread(llvm::IRBuilder<>& b,
                                                      llvm::Value* call_frame) {
  auto* tids = b.CreateStructGEP(call_frame_ty_, call_frame, 1, "tid_gep");
  auto* x_gep = b.CreateStructGEP(thread_ty_, tids, 0, "tid_x_gep");
  auto* y_gep = b.CreateStructGEP(thread_ty_, tids, 1, "tid_y_gep");
  auto* z_gep = b.CreateStructGEP(thread_ty_, tids, 2, "tid_z_gep");

  return {b.CreateLoad(b.getInt64Ty(), x_gep, "tid_x"),
          b.CreateLoad(b.getInt64Ty(), y_gep, "tid_y"),
          b.CreateLoad(b.getInt64Ty(), z_gep, "tid_z")};
}

llvm_ir::IrArray IrEmitter2::EmitKernelArgument(llvm::IRBuilder<>& b,
                                                llvm::Value* call_frame,
                                                int64_t index,
                                                const Shape& shape) {
  llvm::Type* ptr = llvm::PointerType::get(b.getContext(), 0);
  std::string name = absl::StrCat("arg", index);

  auto* args_gep = b.CreateStructGEP(call_frame_ty_, call_frame, 3, "args_gep");
  auto* args = b.CreateLoad(ptr, args_gep, "args");
  auto* data_gep = b.CreateConstGEP2_32(arg_ty_, args, index, 0, name + "_gep");
  auto* data = b.CreateLoad(ptr, data_gep, name);

  return llvm_ir::IrArray(data, llvm_ir::ShapeToIrType(shape, module_), shape);
}

IrEmitter2::KernelPrototype IrEmitter2::EmitKernelPrototype(
    std::string_view name, absl::Span<const Shape> arguments,
    absl::Span<const Shape> results) {
  VLOG(3) << "Emit kernel prototype: " << name
          << ", #arguments=" << arguments.size()
          << ", #results=" << results.size();
  for (const Shape& argument : arguments) {
    VLOG(3) << "  argument: " << argument.ToString(true);
  }
  for (const Shape& result : results) {
    VLOG(3) << "  result: " << result.ToString(true);
  }

  llvm::LLVMContext& ctx = module_->getContext();
  llvm::IRBuilder<> b(ctx);

  // Create a kernel function with HostKernel API.
  llvm::Function* function = llvm::dyn_cast<llvm::Function>(
      module_->getOrInsertFunction(name, KernelFunctionTy(ctx)).getCallee());
  function->setCallingConv(llvm::CallingConv::C);
  function->setDoesNotThrow();

  // Create an entry basic block and set insert point to the end of it.
  b.SetInsertPoint(llvm::BasicBlock::Create(ctx, "", function));

  llvm::Value* call_frame = function->getArg(0);
  // Build thread coordinates from the call frame.
  KernelThreadDims kernel_thread_dims = EmitKernelThreadDims(b, call_frame);
  KernelThread kernel_thread = EmitKernelThread(b, call_frame);

  int64_t idx = 0;

  // IrArrays for the parameters.
  std::vector<llvm_ir::IrArray> ir_arguments;
  for (const Shape& argument : arguments) {
    ir_arguments.push_back(EmitKernelArgument(b, call_frame, idx++, argument));
  }

  // IrArrays for the results.
  std::vector<llvm_ir::IrArray> ir_results;
  for (const Shape& result : results) {
    ir_results.push_back(EmitKernelArgument(b, call_frame, idx++, result));
  }

  // Return null pointer to signal success as we do not support error handling
  // in the compiled host kernel.
  b.CreateRet(
      llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(ctx)));

  return KernelPrototype{function, kernel_thread_dims, kernel_thread,
                         std::move(ir_arguments), std::move(ir_results)};
}

IrEmitter2::KernelPrototype IrEmitter2::EmitKernelPrototype(
    const HloInstruction* instr) {
  return EmitKernelPrototype(instr->name(), FlattenedParameters(instr),
                             FlattenedResults(instr));
}

}  // namespace xla::cpu
