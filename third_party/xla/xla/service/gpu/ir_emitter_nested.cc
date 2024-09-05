/* Copyright 2017 The OpenXLA Authors.

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
#include "xla/service/gpu/ir_emitter_nested.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/Casting.h"
#include "llvm/TargetParser/Triple.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/kernel_reuse_cache.h"
#include "xla/service/llvm_ir/buffer_assignment_util.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/kernel_support_library.h"
#include "xla/service/llvm_ir/llvm_loop.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/llvm_ir/loop_emitter.h"
#include "xla/service/llvm_ir/tuple_ops.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

class IrEmitterNested : public IrEmitter {
 public:
  // Constructs an LLVM IR emitter for a nested HLO computation. `function` is
  // the containing IR function this emitter produces IR to. See
  // IrEmitter::IrEmitter for the meanings of other arguments.
  IrEmitterNested(const HloComputation& nested_computation,
                  IrEmitterContext* ir_emitter_context);

  IrEmitterNested(const IrEmitterNested&) = delete;
  IrEmitterNested& operator=(const IrEmitterNested&) = delete;

  // Overrides the default empty implementation. Binds the given instruction
  // "parameter" with the parameter of the IR function.
  absl::Status HandleParameter(HloInstruction* parameter) override;

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
  absl::StatusOr<llvm::Function*> CodegenNestedComputation();

 protected:
  absl::Status EmitTargetElementLoop(
      const HloInstruction& hlo,
      const llvm_ir::ElementGenerator& element_generator) override;

 private:
  // Emits constants to generated LLVM IR, and also populates related
  // information to 'ir_emitter_context_' for large-constant initializations.
  absl::Status EmitConstants(const HloComputation& computation);

  const HloComputation& nested_computation_;
};

IrEmitterNested::IrEmitterNested(const HloComputation& nested_computation,
                                 IrEmitterContext* ir_emitter_context)
    : IrEmitter(ir_emitter_context,
                /*is_nested=*/true),
      nested_computation_(nested_computation) {}

// Nested function serves the same purpose on GPU as a thread-local function on
// a CPU.
absl::StatusOr<llvm::Function*> IrEmitterNested::CodegenNestedComputation() {
  // Include a fingerprint of the HLO in the function name. Currently, codegen
  // is invoked on temporary HLO objects, which means the address of the
  // computation is not necessarily unique.
  std::string fingerprint = GetComputationFingerprint(&nested_computation_, {});
  size_t hash = absl::Hash<std::string>{}(fingerprint);
  std::string function_name = llvm_ir::SanitizeFunctionName(
      absl::StrCat(nested_computation_.name(), "_",
                   absl::Hex(reinterpret_cast<intptr_t>(&nested_computation_)),
                   "_", absl::Hex(hash)));

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
    argument_types.push_back(b_.getPtrTy());
    int64_t param_size =
        llvm_ir::ByteSizeOf(param_shape, module_->getDataLayout());
    argument_dereferenceable_bytes.push_back(param_size);
  }

  const HloInstruction* root = nested_computation_.root_instruction();
  {
    const Shape& root_shape = root->shape();
    argument_types.push_back(b_.getPtrTy());
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
      Store(ret_value, out_parameter);
    } else {
      CHECK(return_shape.IsTuple());
      llvm::Type* tuple_type = llvm_ir::ShapeToIrType(return_shape, module_);

      for (int i = 0; i < return_shape.tuple_shapes_size(); i++) {
        const Shape& element_shape = return_shape.tuple_shapes(i);
        llvm::Value* destination = llvm_ir::EmitGetTupleElement(
            element_shape,
            /*index=*/i,
            /*alignment=*/1, out_parameter, tuple_type, &b_);
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

absl::Status IrEmitterNested::HandleParameter(HloInstruction* parameter) {
  return absl::OkStatus();
}

absl::Status IrEmitterNested::EmitTargetElementLoop(
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
    return absl::OkStatus();
  }
  return llvm_ir::LoopEmitter(element_generator, GetIrArray(hlo, hlo), &b_)
      .EmitLoop();
}

absl::Status IrEmitterNested::EmitConstants(const HloComputation& computation) {
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
        DenseDataIntermediate::Alias(
            absl::MakeSpan(base, base + literal.size_bytes())),
        &b_);
  }
  return absl::OkStatus();
}

// Casts the provided llvm::Value* to the default address space. This is useful
// in particular for generating IR for AMDGPU target, as its kernel variables
// are in address space 5 instead of the default address space.
llvm::Value* AddrCastToDefault(llvm::Value* arg, llvm::IRBuilder<>& b) {
  llvm::Type* arg_type = arg->getType();
  CHECK(arg_type->isPointerTy());
  if (arg_type->getPointerAddressSpace() != 0) {
    llvm::Type* generic_arg_type = llvm::PointerType::get(
        llvm::cast<llvm::PointerType>(arg_type)->getContext(), 0);
    llvm::Value* addrspacecast_arg =
        b.CreateAddrSpaceCast(arg, generic_arg_type);
    return addrspacecast_arg;
  }
  return arg;
}

void EmitAMDGPUAtomicAdd(llvm::IRBuilder<>* builder,
                         llvm::Value* output_address, llvm::Value* source) {
  CHECK(IsAMDGPU(builder->GetInsertBlock()->getModule()));
  auto output_address_type =
      llvm::dyn_cast<llvm::PointerType>(output_address->getType());
  CHECK_NE(output_address_type, nullptr);

  auto output_ptr =
      (output_address_type->getPointerAddressSpace() == 3)
          // adds to shared memory are always atomic.
          ? output_address
          // the compiler will only generate a global_atomic_fadd if the pointer
          // is in global addrspace (1)
          : builder->CreateAddrSpaceCast(
                output_address,
                llvm::PointerType::get(output_address_type->getContext(),
                                       /*AddressSpace=*/1));

  builder->CreateAtomicRMW(
      llvm::AtomicRMWInst::FAdd, output_ptr, source, llvm::MaybeAlign(),
      llvm::AtomicOrdering::SequentiallyConsistent,
      builder->getContext().getOrInsertSyncScopeID("agent"));
}

llvm::SyncScope::ID DetermineSyncScope(llvm::Module* module) {
  return IsAMDGPU(module) ? module->getContext().getOrInsertSyncScopeID("agent")
                          : llvm::SyncScope::System;
}

// A helper method for EmitAtomicOperationForNestedComputation. Certain
// computations, such as floating-point addition and integer maximization, can
// be simply implemented using an LLVM atomic instruction. If "computation" is
// one of this kind, emits code to do that and returns true; otherwise,
// returns false.
bool MaybeEmitDirectAtomicOperation(llvm::IRBuilder<>* builder,
                                    IrEmitterContext& ir_emitter_context,
                                    const HloComputation& computation,
                                    llvm::Value* output_address,
                                    llvm::Value* source_address) {
  CHECK_EQ(2, computation.num_parameters());

  auto* module = builder->GetInsertBlock()->getModule();
  HloOpcode root_opcode = computation.root_instruction()->opcode();
  PrimitiveType element_type =
      computation.root_instruction()->shape().element_type();
  bool is_atomic_integral = element_type == S32 || element_type == U32 ||
                            element_type == S64 || element_type == U64;
  llvm::Value* source =
      builder->CreateLoad(llvm_ir::PrimitiveTypeToIrType(element_type, module),
                          source_address, "source");

  // Just passing along RHS -> atomic store.
  if (computation.instruction_count() == 2 &&
      root_opcode == HloOpcode::kParameter &&
      (element_type == F32 || is_atomic_integral) &&
      computation.root_instruction()->parameter_number() == 1) {
    llvm::StoreInst* store = builder->CreateStore(source, output_address);
    store->setAtomic(llvm::AtomicOrdering::Unordered);
    // Derive a minimum alignment from the type. The optimizer can increase it
    // later.
    store->setAlignment(
        llvm::Align(ShapeUtil::ByteSizeOfPrimitiveType(element_type)));
    return true;
  }

  if (computation.instruction_count() != 3) {
    // We special-case only computations with one computing instruction for now.
    // Such computation has exactly three instructions given it has two
    // parameters.
    return false;
  }

  auto sync_scope = DetermineSyncScope(module);
  if (root_opcode == HloOpcode::kAdd) {
    llvm::Triple target_triple = llvm::Triple(module->getTargetTriple());
    // NVPTX supports atomicAdd on F32 and integer types.
    if (target_triple.isNVPTX()) {
      // "atom.add.f64 requires sm_60 or higher."
      // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-atom
      bool f64_atomic_add_supported =
          ir_emitter_context.cuda_compute_capability().IsAtLeast(
              se::CudaComputeCapability::PASCAL_);
      bool f16_atomic_add_supported =
          ir_emitter_context.cuda_compute_capability().IsAtLeast(
              se::CudaComputeCapability::VOLTA);
      bool bf16_atomic_add_supported =
          ir_emitter_context.cuda_compute_capability().IsAtLeast(
              se::CudaComputeCapability::HOPPER);
      bool atomic_add_supported =
          element_type == F32 ||
          (f16_atomic_add_supported && element_type == F16) ||
          (bf16_atomic_add_supported && element_type == BF16) ||
          (f64_atomic_add_supported && element_type == F64);
      if (atomic_add_supported) {
        builder->CreateAtomicRMW(llvm::AtomicRMWInst::FAdd, output_address,
                                 source, llvm::MaybeAlign(),
                                 llvm::AtomicOrdering::SequentiallyConsistent);
        return true;
      }
    }

    if (target_triple.isAMDGPU() &&
        (element_type == F32 ||
         (element_type == F16 &&
          ir_emitter_context.rocm_compute_capability()
              .has_fp16_atomics_support()))) /* is atomic add supported? */ {
      EmitAMDGPUAtomicAdd(builder, output_address, source);
      return true;
    }

    if (is_atomic_integral) {
      // integral + integral
      builder->CreateAtomicRMW(
          llvm::AtomicRMWInst::Add, output_address, source, llvm::MaybeAlign(),
          llvm::AtomicOrdering::SequentiallyConsistent, sync_scope);
      return true;
    }
  }

  // NVPTX supports atomicMax and atomicMin only on integer types.
  // For float, we can convert to int and use atomicMax. This approach works
  // correctly only when both operands are not -NANs. Currently, we care about
  // this optimization for Scatter use cases.
  if (root_opcode == HloOpcode::kMaximum) {
    if (is_atomic_integral) {
      // max(integral, integral)
      auto opcode = primitive_util::IsSignedIntegralType(element_type)
                        ? llvm::AtomicRMWInst::Max
                        : llvm::AtomicRMWInst::UMax;
      builder->CreateAtomicRMW(
          opcode, output_address, source, llvm::MaybeAlign(),
          llvm::AtomicOrdering::SequentiallyConsistent, sync_scope);
      return true;
    } else if (element_type == F32) {
      // max(float, float) via AtomicMax and AtomicMin on int
      // We use AtomicMax when the update value is positive.
      // We use AtomicMin when the value is negative to produce correct results.
      // The snippet below expresses the emitted code
      // if (!signbit(val)) {
      //   atomicMax((int*)address, __float_as_int(val));
      // } else {
      //   atomicMin((unsigned int*)address, __float_as_uint(val));
      // }

      KernelSupportLibrary ksl(builder, llvm_ir::UnrollMode::kDefaultUnroll);

      llvm::Value* old_output = builder->CreateLoad(
          builder->getFloatTy(), output_address, "old_output");
      auto is_nan_output = builder->CreateFCmpUNO(old_output, old_output);
      ksl.If(
          "is_nan_output", is_nan_output,
          [&]() {
            // Do nothing
          },
          [&]() {
            // Evaluating floating max using integer atomics has the limitation
            // of not propagating -NaNs. To handle this, we check if the update
            // value is -NaN and convert it to a positive one by dropping the
            // sign-bit.
            auto is_nan_source = builder->CreateFCmpUNO(source, source);
            llvm::Value* no_negative_nan_source = builder->CreateSelect(
                is_nan_source, llvm::ConstantFP::getNaN(source->getType()),
                source);

            llvm::Value* old_less_than =
                builder->CreateFCmpULT(old_output, no_negative_nan_source);

            // This check allows us to skip the atomic update all-together at
            // the expense of reading the value in memory for every update.
            // Evaluated against Waymo's benchmarks, adding the check achieves
            // better overall performance.
            ksl.If("need_update", old_less_than, [&]() {
              llvm::Value* source_float_as_int = builder->CreateBitCast(
                  no_negative_nan_source, builder->getInt32Ty());
              llvm::Value* is_not_negative = builder->CreateICmpSGE(
                  source_float_as_int,
                  llvm::ConstantInt::get(builder->getInt32Ty(), 0));
              ksl.If(
                  "not_negative", is_not_negative,
                  [&]() {
                    // atomicMax((int *)address, __float_as_int(val))
                    builder->CreateAtomicRMW(
                        llvm::AtomicRMWInst::Max, output_address,
                        source_float_as_int, llvm::MaybeAlign(),
                        llvm::AtomicOrdering::SequentiallyConsistent,
                        sync_scope);
                  },
                  [&]() {
                    // atomicMin((unsigned int *)address, __float_as_uint(val))
                    builder->CreateAtomicRMW(
                        llvm::AtomicRMWInst::UMin, output_address,
                        source_float_as_int, llvm::MaybeAlign(),
                        llvm::AtomicOrdering::SequentiallyConsistent,
                        sync_scope);
                  });
            });
          });
      return true;
    }
    return false;
  }

  if (root_opcode == HloOpcode::kMinimum && is_atomic_integral) {
    // min(integral, integral)
    auto opcode = primitive_util::IsSignedIntegralType(element_type)
                      ? llvm::AtomicRMWInst::Min
                      : llvm::AtomicRMWInst::UMin;
    builder->CreateAtomicRMW(opcode, output_address, source, llvm::MaybeAlign(),
                             llvm::AtomicOrdering::SequentiallyConsistent,
                             sync_scope);
    return true;
  }

  return false;
}

// Implements atomic binary operations using atomic compare-and-swap
// (atomicCAS) as follows:
//   1. Reads the value from the memory pointed to by output_address and
//     records it as old_output.
//   2. Uses old_output as one of the source operand to perform the binary
//     operation and stores the result in new_output.
//   3. Calls atomicCAS which implements compare-and-swap as an atomic
//     operation. In particular, atomicCAS reads the value from the memory
//     pointed to by output_address, and compares the value with old_output. If
//     the two values equal, new_output is written to the same memory location
//     and true is returned to indicate that the atomic operation succeeds.
//     Otherwise, the new value read from the memory is returned. In this case,
//     the new value is copied to old_output, and steps 2. and 3. are repeated
//     until atomicCAS succeeds.
//
// On Nvidia GPUs, atomicCAS can only operate on 32 bit and 64 bit integers. If
// the element type of the binary operation is 32 bits or 64 bits, the integer
// type of the same size is used for the atomicCAS operation. On the other hand,
// if the element type is smaller than 32 bits, int32_t is used for the
// atomicCAS operation. In this case, atomicCAS reads and writes 32 bit values
// from the memory, which is larger than the memory size required by the
// original atomic binary operation. We mask off the last two bits of the
// output_address and use the result as an address to read the 32 bit values
// from the memory. This can avoid out of bound memory accesses if tensor
// buffers are 4 byte aligned and have a size of 4N, an assumption that the
// runtime can guarantee.
//
// The pseudo code is shown below. Variables *_address are pointers to a memory
// region with a size equal to the size of the atomicCAS operation, with the
// exception that new_output_address is a pointer to a memory region with a size
// equal to the element size of the binary operation.
//
//   element_size = sizeof(element_type);
//   atomic_size = max(32, element_size);
//   cas_new_output_address = alloca(atomic_size);
//   cas_old_output_address = alloca(atomic_size);
//   if (atomic_size != element_size) {
//     atomic_address = output_address & ((int64_t)(-4));
//     new_output_address = cas_new_output_address + (output_address & 3);
//   } else {
//     atomic_address = output_address;
//     new_output_address = cas_new_output_address;
//   }
//
//   *cas_old_output_address = *atomic_address;
//   do {
//     *cas_new_output_address = *cas_old_output_address;
//     *new_output_address = operation(*new_output_address, *source_address);
//     (*cas_old_output_address, success) =
//       atomicCAS(atomic_address, *cas_old_output_address,
//       *cas_new_output_address);
//   } while (!success);
//
absl::Status EmitAtomicOperationUsingCAS(llvm::IRBuilder<>* builder,
                                         IrEmitterContext& ir_emitter_context,
                                         const HloComputation& computation,
                                         llvm::Value* output_address,
                                         llvm::Value* source_address,
                                         llvm::Type* element_type) {
  llvm::PointerType* output_address_type =
      llvm::dyn_cast<llvm::PointerType>(output_address->getType());
  CHECK_NE(output_address_type, nullptr);

  int element_size = llvm_ir::GetSizeInBits(element_type);

  int atomic_size = (element_size < 32) ? 32 : element_size;
  llvm::Type* atomic_type = builder->getIntNTy(atomic_size);
  llvm::Type* atomic_address_type =
      builder->getPtrTy(output_address_type->getPointerAddressSpace());

  // cas_old_output_address and cas_new_output_address point to the scratch
  // memory where we store the old and new values for the repeated atomicCAS
  // operations.
  llvm::AllocaInst* cas_old_output_address = llvm_ir::EmitAllocaAtFunctionEntry(
      atomic_type, "cas_old_output_address", builder);
  llvm::AllocaInst* cas_new_output_address = llvm_ir::EmitAllocaAtFunctionEntry(
      atomic_type, "cas_new_output_address", builder);

  // Emit preparation code to the preheader.
  llvm::BasicBlock* loop_preheader_bb = builder->GetInsertBlock();
  auto* module = loop_preheader_bb->getModule();

  llvm::Value* atomic_memory_address;
  // binop_output_address points to the scratch memory that stores the
  // result of the binary operation.
  llvm::Value* binop_output_address;
  if (element_size < 32) {
    // Assume the element size is an integer number of bytes.
    CHECK_EQ((element_size % sizeof(char)), 0);
    llvm::Type* address_int_type =
        module->getDataLayout().getIntPtrType(output_address_type);
    atomic_memory_address =
        builder->CreatePtrToInt(output_address, address_int_type);
    llvm::Value* mask = llvm::ConstantInt::get(address_int_type, 3);
    llvm::Value* offset = builder->CreateAnd(atomic_memory_address, mask);
    mask = llvm::ConstantInt::get(address_int_type, -4);
    atomic_memory_address = builder->CreateAnd(atomic_memory_address, mask);
    atomic_memory_address =
        builder->CreateIntToPtr(atomic_memory_address, atomic_address_type);
    binop_output_address = builder->CreateAdd(
        builder->CreatePtrToInt(cas_new_output_address, address_int_type),
        offset);
    binop_output_address = builder->CreateIntToPtr(
        binop_output_address,
        builder->getPtrTy(
            cas_new_output_address->getType()->getPointerAddressSpace()));
  } else {
    atomic_memory_address = builder->CreatePointerBitCastOrAddrSpaceCast(
        output_address, atomic_address_type);
    binop_output_address = builder->CreatePointerBitCastOrAddrSpaceCast(
        cas_new_output_address,
        builder->getPtrTy(
            cas_new_output_address->getType()->getPointerAddressSpace()));
  }

  // Use the value from the memory that atomicCAS operates on to initialize
  // cas_old_output.
  llvm::Value* cas_old_output =
      builder->CreateLoad(atomic_type, atomic_memory_address, "cas_old_output");
  builder->CreateStore(cas_old_output, cas_old_output_address);

  llvm::BasicBlock* loop_exit_bb = loop_preheader_bb->splitBasicBlock(
      builder->GetInsertPoint(), "atomic_op_loop_exit");
  llvm::BasicBlock* loop_body_bb =
      llvm::BasicBlock::Create(builder->getContext(), "atomic_op_loop_body",
                               builder->GetInsertBlock()->getParent());
  builder->SetInsertPoint(loop_body_bb);
  // Change preheader's successor from loop_exit_bb to loop_body_bb.
  loop_preheader_bb->getTerminator()->setSuccessor(0, loop_body_bb);

  // Emit the body of the loop that repeatedly invokes atomicCAS.
  //
  // Use cas_old_output to initialize cas_new_output.
  cas_old_output =
      builder->CreateLoad(cas_old_output_address->getAllocatedType(),
                          cas_old_output_address, "cas_old_output");
  builder->CreateStore(cas_old_output, cas_new_output_address);
  // Emits code to calculate new_output = operation(old_output, source);
  TF_RETURN_IF_ERROR(CallNestedComputation(
      builder, ir_emitter_context, computation,
      {binop_output_address, source_address}, binop_output_address));

  llvm::Value* cas_new_output =
      builder->CreateLoad(cas_new_output_address->getAllocatedType(),
                          cas_new_output_address, "cas_new_output");

  // If cas_new_output == cas_old_output, we're not asking for anything to
  // change, so we're done here!
  llvm::Value* old_eq_new =
      builder->CreateICmpEQ(cas_old_output, cas_new_output);
  llvm::BasicBlock* loop_cas_bb =
      llvm::BasicBlock::Create(builder->getContext(), "atomic_op_loop_cas",
                               builder->GetInsertBlock()->getParent());
  builder->CreateCondBr(old_eq_new, loop_exit_bb, loop_cas_bb);
  builder->SetInsertPoint(loop_cas_bb);

  // Emit code to perform the atomicCAS operation
  // (cas_old_output, success) = atomicCAS(memory_address, cas_old_output,
  //                                       cas_new_output);
  llvm::Value* ret_value = builder->CreateAtomicCmpXchg(
      atomic_memory_address, cas_old_output, cas_new_output, llvm::MaybeAlign(),
      llvm::AtomicOrdering::SequentiallyConsistent,
      llvm::AtomicOrdering::SequentiallyConsistent, DetermineSyncScope(module));

  // Extract the memory value returned from atomicCAS and store it as
  // cas_old_output.
  builder->CreateStore(
      builder->CreateExtractValue(ret_value, 0, "cas_old_output"),
      cas_old_output_address);
  // Extract the success bit returned from atomicCAS and generate a
  // conditional branch on the success bit.
  builder->CreateCondBr(builder->CreateExtractValue(ret_value, 1, "success"),
                        loop_exit_bb, loop_body_bb);

  // Set the insertion point to the exit basic block so that the caller of
  // this method can continue emitting code to the right place.
  llvm_ir::SetToFirstInsertPoint(loop_exit_bb, builder);
  return absl::OkStatus();
}

}  // namespace

absl::Status CallNestedComputation(llvm::IRBuilder<>* builder,
                                   IrEmitterContext& ir_emitter_context,
                                   const HloComputation& computation,
                                   absl::Span<llvm::Value* const> operands,
                                   llvm::Value* output) {
  TF_RET_CHECK(computation.num_parameters() > 0);

  TF_ASSIGN_OR_RETURN(llvm::Function * emitted_function,
                      IrEmitterNested(computation, &ir_emitter_context)
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

  return absl::OkStatus();
}

absl::StatusOr<std::vector<llvm::Value*>> CallNestedComputationWithScalars(
    llvm::IRBuilder<>* builder, IrEmitterContext& ir_emitter_context,
    const HloComputation& computation,
    absl::Span<llvm::Value* const> parameter_elements) {
  std::vector<llvm::Value*> parameter_buffers;
  for (llvm::Value* parameter_element : parameter_elements) {
    parameter_buffers.push_back(llvm_ir::EmitAllocaAtFunctionEntry(
        parameter_element->getType(), "parameter_buffer", builder));
    builder->CreateStore(parameter_element, parameter_buffers.back());
  }

  return CallNestedComputationWithScalarAddrs(builder, ir_emitter_context,
                                              computation, parameter_buffers);
}

absl::StatusOr<std::vector<llvm::Value*>> CallNestedComputationWithScalarAddrs(
    llvm::IRBuilder<>* builder, IrEmitterContext& ir_emitter_context,
    const HloComputation& computation,
    absl::Span<llvm::Value* const> parameter_elements_addrs) {
  const Shape& return_shape = computation.root_instruction()->shape();
  llvm::Type* return_buffer_type = llvm_ir::ShapeToIrType(
      return_shape, builder->GetInsertBlock()->getModule());
  llvm::Value* return_buffer = llvm_ir::EmitAllocaAtFunctionEntry(
      return_buffer_type, "return_buffer", builder);

  std::vector<llvm::Value*> allocas_for_returned_scalars;
  if (!return_shape.IsTuple()) {
    allocas_for_returned_scalars.push_back(return_buffer);
  } else {
    allocas_for_returned_scalars =
        llvm_ir::EmitTupleAllocasAtFunctionEntry(return_shape, builder);
    llvm_ir::IrArray tuple_array(return_buffer, return_buffer_type,
                                 return_shape);

    llvm_ir::EmitTuple(tuple_array, allocas_for_returned_scalars, builder);
  }

  TF_RETURN_IF_ERROR(
      CallNestedComputation(builder, ir_emitter_context, computation,
                            parameter_elements_addrs, return_buffer));

  std::vector<llvm::Value*> returned_scalars;
  returned_scalars.reserve(allocas_for_returned_scalars.size());
  for (llvm::Value* addr : allocas_for_returned_scalars) {
    auto alloca = llvm::cast<llvm::AllocaInst>(addr);
    returned_scalars.push_back(
        builder->CreateLoad(alloca->getAllocatedType(), alloca));
  }
  return returned_scalars;
}

absl::Status EmitAtomicOperationForNestedComputation(
    llvm::IRBuilder<>* builder, IrEmitterContext& ir_emitter_context,
    const HloComputation& computation, llvm::Value* output_address,
    llvm::Value* source_address, llvm::Type* element_type) {
  if (computation.num_parameters() != 2) {
    // TODO(b/30258929): We only accept binary computations so far.
    return Unimplemented(
        "We only support atomic functions with exactly two parameters, but "
        "computation %s has %d.",
        computation.name(), computation.num_parameters());
  }

  if (MaybeEmitDirectAtomicOperation(builder, ir_emitter_context, computation,
                                     output_address, source_address)) {
    return absl::OkStatus();
  }

  return EmitAtomicOperationUsingCAS(builder, ir_emitter_context, computation,
                                     output_address, source_address,
                                     element_type);
}

}  // namespace gpu
}  // namespace xla
