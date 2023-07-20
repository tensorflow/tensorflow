/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_nested.h"

#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_context.h"
#include "tensorflow/compiler/xla/service/llvm_ir/buffer_assignment_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/tuple_ops.h"

namespace xla {
namespace gpu {
namespace {

class IrEmitterNested : public IrEmitter {
 public:
  // Constructs an LLVM IR emitter for a nested HLO computation. `function` is
  // the containing IR function this emitter produces IR to. See
  // IrEmitter::IrEmitter for the meanings of other arguments.
  IrEmitterNested(const HloModuleConfig& hlo_module_config,
                  const HloComputation& nested_computation,
                  IrEmitterContext* ir_emitter_context);

  IrEmitterNested(const IrEmitterNested&) = delete;
  IrEmitterNested& operator=(const IrEmitterNested&) = delete;

  // Overrides the default empty implementation. Binds the given instruction
  // "parameter" with the parameter of the IR function.
  Status HandleParameter(HloInstruction* parameter) override;

  // Generate the code for the computation passed in the constructor, if it
  // wasn't already generated previously.
  // As well as generting the code for the function, emits code for global
  // constants, and also populates related information to 'ir_emitter_context_'
  // for large-constant initializations. Large constants don't get initializers
  // in the generated code and so must be initialized by XLA. The value of these
  // constants will be stored in 'content'. Constants with initializers in the
  // generated code will have empty 'content'.
  //
  // The allocation index for these constants will always be -1 (i.e. doesn't
  // correspond to any allocation)
  StatusOr<llvm::Function*> CodegenNestedComputation();

 protected:
  Status EmitTargetElementLoop(
      const HloInstruction& hlo,
      const llvm_ir::ElementGenerator& element_generator) override;

 private:
  // Emits constants to generated LLVM IR, and also populates related
  // information to 'ir_emitter_context_' for large-constant initializations.
  Status EmitConstants(const HloComputation& computation);

  const HloComputation& nested_computation_;
};

IrEmitterNested::IrEmitterNested(const HloModuleConfig& hlo_module_config,
                                 const HloComputation& nested_computation,
                                 IrEmitterContext* ir_emitter_context)
    : IrEmitter(hlo_module_config, ir_emitter_context, /*is_nested=*/true),
      nested_computation_(nested_computation) {}

// Nested function serves the same purpose on GPU as a thread-local function on
// a CPU.
StatusOr<llvm::Function*> IrEmitterNested::CodegenNestedComputation() {
  std::string function_name = llvm_ir::SanitizeFunctionName(absl::StrCat(
      nested_computation_.name(), "_",
      absl::Hex(reinterpret_cast<intptr_t>(&nested_computation_))));
  auto* function =
      ir_emitter_context_->llvm_module()->getFunction(function_name);
  if (function) return function;

  TF_RETURN_IF_ERROR(EmitConstants(nested_computation_));
  std::vector<const HloInstruction*> io_hlos;
  std::vector<llvm::Type*> argument_types;
  std::vector<int64_t> argument_dereferenceable_bytes;
  const auto& params = nested_computation_.parameter_instructions();
  const auto n = params.size() + 1;
  io_hlos.reserve(n - 1);
  argument_types.reserve(n);
  argument_dereferenceable_bytes.reserve(n);
  for (const HloInstruction* param : params) {
    io_hlos.push_back(param);
    const Shape& param_shape = param->shape();
    argument_types.push_back(
        llvm_ir::ShapeToIrType(param_shape, module_)->getPointerTo());
    int64_t param_size =
        llvm_ir::ByteSizeOf(param_shape, module_->getDataLayout());
    argument_dereferenceable_bytes.push_back(param_size);
  }

  const HloInstruction* root = nested_computation_.root_instruction();
  {
    const Shape& root_shape = root->shape();
    argument_types.push_back(
        llvm_ir::ShapeToIrType(root_shape, module_)->getPointerTo());
    int64_t root_size = llvm_ir::ByteSizeOf(
        root_shape, ir_emitter_context_->llvm_module()->getDataLayout());
    argument_dereferenceable_bytes.push_back(root_size);
  }

  llvm::FunctionType* function_type =
      llvm::FunctionType::get(b_.getVoidTy(), argument_types, false);
  function = llvm::Function::Create(
      function_type,                       // The function type.
      llvm::GlobalValue::InternalLinkage,  // The linkage type.
      function_name,
      ir_emitter_context_->llvm_module());  // The parent LLVM module.
  for (size_t arg_no = 0; arg_no < argument_dereferenceable_bytes.size();
       ++arg_no) {
    int64_t arg_size = argument_dereferenceable_bytes[arg_no];
    if (arg_size > 0) {
      function->addDereferenceableParamAttr(arg_no, arg_size);
    }
  }

  // TODO(b/65380986): Investigate if adding fast math flags for generated
  // kernels makes sense.

  llvm::BasicBlock* entry_bb =
      llvm::BasicBlock::Create(function->getContext(), "entry", function);
  // Emit a "return void" at entry_bb's end, and sets the insert point before
  // that return instruction.
  llvm::ReturnInst* ret_instr =
      llvm::ReturnInst::Create(function->getContext(), entry_bb);
  b_.SetInsertPoint(ret_instr);

  std::vector<const HloInstruction*> non_io_hlos;
  non_io_hlos.push_back(root);
  for (const auto* hlo : nested_computation_.instructions()) {
    if (hlo->opcode() != HloOpcode::kParameter &&
        hlo != nested_computation_.root_instruction()) {
      non_io_hlos.push_back(hlo);
    }
  }
  bindings_.EmitBasePointersForHlos(io_hlos, non_io_hlos);

  TF_RETURN_IF_ERROR(nested_computation_.root_instruction()->Accept(this));
  b_.SetInsertPoint(ret_instr);

  // Function epilogue: copy the output value back.
  {
    // TODO(cheshire) Duplication vs. EmitThreadLocalFunctionEpilogue
    const HloInstruction* root_instruction =
        nested_computation_.root_instruction();
    llvm::Value* root_value = bindings_.GetBasePointer(*root_instruction);
    const Shape& return_shape = root_instruction->shape();

    // Last argument is the out parameter.
    llvm::Argument* out_parameter = std::prev(function->arg_end(), 1);

    if (ShapeUtil::IsScalar(return_shape)) {
      llvm::Value* ret_value =
          Load(llvm_ir::ShapeToIrType(return_shape, module_), root_value,
               "load_ret_value");
      Store(ret_value,
            BitCast(out_parameter, root_value->getType(), "bitcast_ret_value"));
    } else {
      CHECK(return_shape.IsTuple());
      llvm::Type* tuple_type = llvm_ir::ShapeToIrType(return_shape, module_);
      llvm::Type* tuple_type_ptr = tuple_type->getPointerTo();
      llvm::Value* tuple_ptr = BitCast(out_parameter, tuple_type_ptr);

      for (int i = 0; i < return_shape.tuple_shapes_size(); i++) {
        const Shape& element_shape = return_shape.tuple_shapes(i);
        llvm::Value* destination = llvm_ir::EmitGetTupleElement(
            element_shape,
            /*index=*/i,
            /*alignment=*/1, tuple_ptr, tuple_type, &b_);
        llvm::Value* source = llvm_ir::EmitGetTupleElement(
            element_shape,
            /*index=*/i,
            /*alignment=*/1, root_value,
            llvm_ir::ShapeToIrType(root_instruction->shape(), module_), &b_);
        Store(Load(llvm_ir::ShapeToIrType(element_shape, module_), source),
              destination);
      }
    }
  }
  b_.SetInsertPoint(ret_instr);
  return function;
}

Status IrEmitterNested::HandleParameter(HloInstruction* parameter) {
  return OkStatus();
}

Status IrEmitterNested::EmitTargetElementLoop(
    const HloInstruction& hlo,
    const llvm_ir::ElementGenerator& element_generator) {
  // For MOF we give the loop emitter an array for every output it should
  // generate.
  if (hlo.shape().IsTuple()) {
    std::vector<llvm_ir::IrArray> target_arrays =
        ConstructIrArrayForOutputs(hlo);
    TF_RETURN_IF_ERROR(
        llvm_ir::LoopEmitter(element_generator, target_arrays, &b_).EmitLoop());
    llvm_ir::EmitTuple(GetIrArray(hlo, hlo), target_arrays, &b_);
    return OkStatus();
  }
  return llvm_ir::LoopEmitter(element_generator, GetIrArray(hlo, hlo), &b_)
      .EmitLoop();
}

Status IrEmitterNested::EmitConstants(const HloComputation& computation) {
  for (HloInstruction* instr : computation.instructions()) {
    if (instr->opcode() != HloOpcode::kConstant) {
      continue;
    }
    const Literal& literal = instr->literal();

    // These globals will be looked up by name by GpuExecutable so we need to
    // give them an external linkage.  Not all of their uses are visible in
    // the LLVM IR (e.g. TupleThunk) so we can't give then a linkage that
    // merely preserves their names (like available_externally), we also need
    // to ensure that they stick around even if they're "unused".
    //
    // We may have to be more clever here in the future if we notice that we're
    // keeping around too many globals because of their linkage.
    std::string global_name = llvm_ir::ConstantHloToGlobalName(*instr);

    auto base = static_cast<const uint8_t*>(literal.untyped_data());
    ir_emitter_context_->emit_constant(
        literal.element_count(),
        ShapeUtil::ByteSizeOfPrimitiveType(literal.shape().element_type()),

        global_name,
        /*allocation_idx=*/-1,
        llvm::ArrayRef<uint8_t>(base, base + literal.size_bytes()), &b_);
  }
  return OkStatus();
}

// Casts the provided llvm::Value* to the default address space. This is useful
// in particular for generating IR for AMDGPU target, as its kernel variables
// are in address space 5 instead of the default address space.
static llvm::Value* AddrCastToDefault(llvm::Value* arg, llvm::IRBuilder<>& b) {
  llvm::Type* arg_type = arg->getType();
  CHECK(arg_type->isPointerTy());
  if (arg_type->getPointerAddressSpace() != 0) {
    llvm::Type* generic_arg_type = llvm::PointerType::getWithSamePointeeType(
        llvm::cast<llvm::PointerType>(arg_type), 0);
    llvm::Value* addrspacecast_arg =
        b.CreateAddrSpaceCast(arg, generic_arg_type);
    return addrspacecast_arg;
  }
  return arg;
}

}  // namespace

Status CallNestedComputation(llvm::IRBuilder<>* builder,
                             const HloModuleConfig& hlo_module_config,
                             const HloComputation& nested_computation,
                             IrEmitterContext& ir_emitter_context,
                             absl::Span<llvm::Value* const> operands,
                             llvm::Value* output) {
  TF_RET_CHECK(nested_computation.num_parameters() > 0);

  TF_ASSIGN_OR_RETURN(llvm::Function * emitted_function,
                      IrEmitterNested(hlo_module_config, nested_computation,
                                      &ir_emitter_context)
                          .CodegenNestedComputation());

  // Operands are in default address space for non-AMDGPU target.
  // However for AMDGPU target, addrspacecast alloca variables from
  // addrspace 5 to addrspace 0 is needed.
  std::vector<llvm::Value*> arguments;
  absl::c_transform(
      operands, std::back_inserter(arguments),
      [builder](llvm::Value* arg) { return AddrCastToDefault(arg, *builder); });

  llvm::Value* casted_output = AddrCastToDefault(output, *builder);
  arguments.push_back(casted_output);

  builder->CreateCall(emitted_function, arguments);

  return OkStatus();
}

}  // namespace gpu
}  // namespace xla
