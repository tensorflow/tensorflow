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

#include "xla/backends/gpu/codegen/llvm/llvm_emitter.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// IWYU pragma: no_include "llvm/IR/Intrinsics.gen.inc"
#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Casting.h"
#include "llvm/TargetParser/Triple.h"
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/backends/gpu/codegen/llvm/parallel_loop_emitter.h"
#include "xla/backends/gpu/codegen/llvm/sort_util.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/codegen/emitters/computation_fingerprint.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/elemental_ir_emitter.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/hlo_to_ir_bindings.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/llvm_ir/buffer_assignment_util.h"
#include "xla/service/llvm_ir/fused_ir_emitter.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/ir_builder_mixin.h"
#include "xla/service/llvm_ir/kernel_support_library.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/llvm_ir/loop_emitter.h"
#include "xla/service/llvm_ir/tuple_ops.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/fingerprint.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla::gpu {
namespace {

// Emits LLVM IR for a "nested computation" into a non-kernel device function.
//
// IrEmitter generates a non-kernel function with the following parameters:
//
//   - N pointers to the buffers of each of the N parameters to the computation,
//   - a pointer to the output buffer of the computation, and
//   - a pointer to the top-level temp buffer.
absl::Status CallNestedComputation(llvm::IRBuilderBase* builder,
                                   IrEmitterContext& ir_emitter_context,
                                   llvm::Module* llvm_module,
                                   const HloComputation& computation,
                                   absl::Span<llvm::Value* const> operands,
                                   llvm::Value* output);

// Class for translating HLO graphs to LLVM IR for a GPU.
//
// In the unnested variety, each HLO gets its own kernel function, whereas in
// the nested version the whole computation is emitted as one *non-kernel*
// function.
class IrEmitter : public DfsHloVisitorWithDefault,
                  public IrBuilderMixin<IrEmitter> {
 public:
  IrEmitter(const IrEmitter&) = delete;
  IrEmitter& operator=(const IrEmitter&) = delete;

  // Constructs an IrEmitter with the given IrEmitter context.
  // ir_emitter_context is owned by the caller and should outlive the IrEmitter
  // object.
  explicit IrEmitter(IrEmitterContext* ir_emitter_context,
                     llvm::Module* llvm_module, bool is_nested)
      : ir_emitter_context_(ir_emitter_context),
        module_(llvm_module),
        b_(module_->getContext()),
        bindings_(&b_, module_, is_nested) {}

  absl::Status DefaultAction(HloInstruction* hlo) override;

  absl::Status HandleConstant(HloInstruction* constant) override {
    return absl::OkStatus();
  }
  absl::Status HandleGetTupleElement(
      HloInstruction* get_tuple_element) override {
    auto operand = get_tuple_element->operand(0);
    CHECK(bindings_.BoundToIrValue(*operand));
    bindings_.BindHloToIrValue(
        *get_tuple_element,
        llvm_ir::EmitGetTupleElement(
            get_tuple_element->shape(), get_tuple_element->tuple_index(),
            /*alignment=*/1, GetBasePointer(*operand),
            llvm_ir::ShapeToIrType(operand->shape(), module_->getContext()),
            &b_));
    return absl::OkStatus();
  }

  absl::Status HandleConvolution(HloInstruction* convolution) override {
    if (ShapeUtil::IsZeroElementArray(convolution->shape())) {
      // Emit no code for an empty output.
      return absl::OkStatus();
    }
    return Unimplemented(
        "Hit a case for convolution that is not implemented on GPU.");
  }

  absl::Status HandleFft(HloInstruction* fft) override {
    if (ShapeUtil::IsZeroElementArray(fft->shape())) {
      // Emit no code for an empty output.
      return absl::OkStatus();
    }
    return Unimplemented("Hit a case for fft that is not implemented on GPU.");
  }

  absl::Status HandleAllReduce(HloInstruction* crs) override {
    return Unimplemented(
        "AllReduce cannot be nested inside of fusion, map, etc.");
  }
  absl::Status HandleInfeed(HloInstruction*) override {
    return Unimplemented("Infeed is not supported on GPU.");
  }

  absl::Status HandleOutfeed(HloInstruction*) override {
    return Unimplemented("Outfeed is not supported on GPU.");
  }
  absl::Status HandleSend(HloInstruction*) override {
    return Unimplemented("Send is not implemented on GPU");
  }

  absl::Status HandleSendDone(HloInstruction*) override {
    return Unimplemented("Send-Done is not implemented on GPU");
  }

  absl::Status HandleRecv(HloInstruction*) override {
    return Unimplemented("Recv is not implemented on GPU");
  }

  absl::Status HandleRecvDone(HloInstruction*) override {
    return Unimplemented("Recv-done is not implemented on GPU");
  }

  absl::Status HandleScatter(HloInstruction*) override {
    return Unimplemented("Scatter is not implemented on GPUs.");
  }
  absl::Status HandleParameter(HloInstruction* parameter) override {
    return absl::OkStatus();
  }
  absl::Status HandleTuple(HloInstruction* tuple) override {
    std::vector<llvm::Value*> base_ptrs;
    for (const HloInstruction* operand : tuple->operands()) {
      base_ptrs.push_back(GetBasePointer(*operand));
    }
    llvm_ir::EmitTuple(GetIrArray(*tuple, *tuple), base_ptrs, &b_);
    return absl::OkStatus();
  }
  absl::Status HandleCall(HloInstruction* call) override {
    std::vector<llvm::Value*> operand_addresses;
    for (HloInstruction* operand : call->operands()) {
      operand_addresses.push_back(GetBasePointer(*operand));
    }
    return CallNestedComputation(&b_, *ir_emitter_context_, module_,
                                 *call->to_apply(), operand_addresses,
                                 GetBasePointer(*call));
  }

  absl::Status HandleCustomCall(HloInstruction*) override {
    return Unimplemented("custom-call");
  }

  absl::Status HandleBatchNormInference(HloInstruction*) override {
    return Unimplemented(
        "The GPU backend does not implement BatchNormInference directly.  It "
        "should be lowered before IR emission to HLO-soup using "
        "BatchNormRewriter.");
  }

  absl::Status HandleBatchNormTraining(HloInstruction*) override {
    return Unimplemented(
        "The GPU backend does not implement BatchNormTraining directly.  It "
        "should be lowered before IR emission to HLO-soup using "
        "BatchNormRewriter.");
  }

  absl::Status HandleBatchNormGrad(HloInstruction*) override {
    return Unimplemented(
        "The GPU backend does not implement BatchNormGrad directly.  It should "
        "be lowered before IR emission to HLO-soup using BatchNormRewriter.");
  }

  absl::Status HandleAddDependency(HloInstruction* add_dependency) override;

  absl::Status FinishVisit(HloInstruction* root) override {
    return absl::OkStatus();
  }

  llvm::IRBuilderBase* builder() { return &b_; }

  llvm::Module* module() { return module_; }

  // Generate the code for the computation passed in the constructor, if it
  // wasn't already generated previously.
  // As well as generting the code for the function, emits code for global
  // constants, and also populates related information to 'ir_emitter_context'
  // for large-constant initializations. Large constants don't get initializers
  // in the generated code and so must be initialized by XLA. The value of these
  // constants will be stored in 'content'. Constants with initializers in the
  // generated code will have empty 'content'.
  //
  // The allocation index for these constants will always be -1 (i.e. doesn't
  // correspond to any allocation)
  absl::StatusOr<llvm::Function*> CodegenNestedComputation(
      const HloComputation& nested_computation);

 protected:
  // Helper for calling HloToIrBindings::GetIrArray.
  //
  // Gets the IrArray which contains inst.  This array has metadata that makes
  // it valid only within the IR that implements consumer.  If you are
  // implementing an HLO and want to get its own output buffer, call
  // GetIrArray(hlo, hlo).
  llvm_ir::IrArray GetIrArray(const HloInstruction& inst,
                              const HloInstruction& consumer,
                              const ShapeIndex& shape_index = {}) {
    return bindings_.GetIrArray(inst, consumer, shape_index);
  }
  // A convenient helper for calling HloToIrBindings::GetBasePointer.
  llvm::Value* GetBasePointer(const HloInstruction& inst,
                              ShapeIndexView shape_index = {}) const {
    return bindings_.GetBasePointer(inst, shape_index);
  }

  // Generates the IrArray for each output of an hlo instruction and returns
  // a vector containing such IrArrays.
  std::vector<llvm_ir::IrArray> ConstructIrArrayForOutputs(
      const HloInstruction& hlo);

  // Emit a single-threaded or multi-threaded loop that computes every element
  // in the result of the given HLO instruction. This produces a series of
  // nested loops (e.g. one for each dimension of the `hlo`'s shape). The body
  // of the inner-most loop is provided by the body_emitter function.
  absl::Status EmitTargetElementLoop(
      const HloInstruction& hlo,
      const llvm_ir::ElementGenerator& element_generator);

  IrEmitterContext* ir_emitter_context_;
  llvm::Module* module_;

  // The following fields track the IR emission state. According to LLVM memory
  // management rules, their memory is owned by the module.
  llvm::IRBuilder<> b_;

  // Mapping from HLO to its underlying LLVM value.
  HloToIrBindings bindings_;

  // Bind all argument IrArrays of `fusion` to `fused_emitter`.
  void BindFusionArguments(const HloInstruction* fusion,
                           FusedIrEmitter* fused_emitter);
};


absl::Status IrEmitter::DefaultAction(HloInstruction* hlo) {
  ElementalIrEmitter::HloToElementGeneratorMap operand_to_generator;
  for (const HloInstruction* operand : hlo->operands()) {
    operand_to_generator[operand] = [=](const llvm_ir::IrArray::Index& index) {
      return GetIrArray(*operand, *hlo)
          .EmitReadArrayElement(index, &b_, operand->name());
    };
  }
  return EmitTargetElementLoop(
      *hlo, ElementalIrEmitter(module_, &b_)
                .MakeElementGenerator(hlo, operand_to_generator));
}

absl::Status IrEmitter::HandleAddDependency(HloInstruction* add_dependency) {
  VLOG(2) << "HandleAddDependency: " << add_dependency->ToString();
  const HloInstruction* operand = add_dependency->operand(0);
  // Add_Dependency is a no-op, but we still want to bind it to an llvm::Value
  // sometimes, e.g., when it's operand is a constant or a bitcast of a
  // constant.
  if (bindings_.BoundToIrValue(*operand)) {
    bindings_.BindHloToIrValue(*add_dependency, GetBasePointer(*operand));
  }
  return absl::OkStatus();
}

// Casts the provided llvm::Value* to the default address space. This is useful
// in particular for generating IR for AMDGPU target, as its kernel variables
// are in address space 5 instead of the default address space.
llvm::Value* AddrCastToDefault(llvm::Value* arg, llvm::IRBuilderBase& b) {
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

absl::Status CallNestedComputation(llvm::IRBuilderBase* builder,
                                   IrEmitterContext& ir_emitter_context,
                                   llvm::Module* llvm_module,
                                   const HloComputation& computation,
                                   absl::Span<llvm::Value* const> operands,
                                   llvm::Value* output) {
  TF_RET_CHECK(computation.num_parameters() > 0);

  ASSIGN_OR_RETURN(
      llvm::Function * emitted_function,
      IrEmitter(&ir_emitter_context, llvm_module, /*is_nested=*/true)
          .CodegenNestedComputation(computation));

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

std::vector<llvm_ir::IrArray> IrEmitter::ConstructIrArrayForOutputs(
    const HloInstruction& hlo) {
  std::vector<llvm_ir::IrArray> output_arrays;
  if (hlo.shape().IsTuple()) {
    int64_t num_outputs = ShapeUtil::TupleElementCount(hlo.shape());
    output_arrays.reserve(num_outputs);
    for (int64_t i = 0; i < num_outputs; ++i) {
      output_arrays.push_back(GetIrArray(hlo, hlo, {i}));
    }
  } else {
    output_arrays.push_back(GetIrArray(hlo, hlo));
  }
  return output_arrays;
}

void IrEmitter::BindFusionArguments(const HloInstruction* fusion,
                                    FusedIrEmitter* fused_emitter) {
  for (int i = 0; i < fusion->operand_count(); i++) {
    const HloInstruction* operand = fusion->operand(i);
    fused_emitter->BindGenerator(
        *fusion->fused_parameter(i),
        [this, operand, fusion](llvm_ir::IrArray::Index index) {
          return GetIrArray(*operand, *fusion)
              .EmitReadArrayElement(index, &b_, operand->name());
        });
  }
}

// Emits constants to generated LLVM IR, and also populates related information
// to 'ir_emitter_context' for large-constant initializations.
absl::Status EmitConstants(llvm::Module* module,
                           IrEmitterContext* ir_emitter_context,
                           const HloComputation& computation) {
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
    GpuExecutable::ConstantInfo info = AppendGlobalConstant(
        module, literal.element_count(),
        ShapeUtil::ByteSizeOfPrimitiveType(literal.shape().element_type()),
        global_name, /*allocation_idx=*/-1,
        DenseDataIntermediate::Alias(
            absl::MakeSpan(base, base + literal.size_bytes())));
    ir_emitter_context->constants().push_back(std::move(info));
  }
  return absl::OkStatus();
}

// Nested function serves the same purpose on GPU as a thread-local function on
// a CPU.
absl::StatusOr<llvm::Function*> IrEmitter::CodegenNestedComputation(
    const HloComputation& nested_computation) {
  // Include a fingerprint of the HLO in the function name to make the name
  // unique.
  tsl::Fprint128 fingerprint = tsl::Fingerprint128(
      emitters::GetComputationFingerprint(&nested_computation, {}));
  std::string function_name = llvm_ir::SanitizeFunctionName(absl::StrCat(
      nested_computation.name(), "_", fingerprint.low64, fingerprint.high64));

  auto* function = module_->getFunction(function_name);
  if (function) {
    return function;
  }

  RETURN_IF_ERROR(
      EmitConstants(module_, ir_emitter_context_, nested_computation));
  std::vector<const HloInstruction*> io_hlos;
  std::vector<llvm::Type*> argument_types;
  std::vector<int64_t> argument_dereferenceable_bytes;
  const auto& params = nested_computation.parameter_instructions();
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

  const HloInstruction* root = nested_computation.root_instruction();
  {
    const Shape& root_shape = root->shape();
    argument_types.push_back(b_.getPtrTy());
    int64_t root_size =
        llvm_ir::ByteSizeOf(root_shape, module_->getDataLayout());
    argument_dereferenceable_bytes.push_back(root_size);
  }

  llvm::FunctionType* function_type =
      llvm::FunctionType::get(b_.getVoidTy(), argument_types, false);
  function = llvm::Function::Create(
      function_type,                       // The function type.
      llvm::GlobalValue::InternalLinkage,  // The linkage type.
      function_name,
      module_);  // The parent LLVM module.
  for (size_t arg_no = 0; arg_no < argument_dereferenceable_bytes.size();
       ++arg_no) {
    int64_t arg_size = argument_dereferenceable_bytes[arg_no];
    if (arg_size > 0) {
      function->addDereferenceableParamAttr(arg_no, arg_size);
    }
  }

  llvm::BasicBlock* entry_bb =
      llvm::BasicBlock::Create(function->getContext(), "entry", function);
  // Emit a "return void" at entry_bb's end, and sets the insert point before
  // that return instruction.
  llvm::ReturnInst* ret_instr =
      llvm::ReturnInst::Create(function->getContext(), entry_bb);
  b_.SetInsertPoint(ret_instr);

  std::vector<const HloInstruction*> non_io_hlos;
  non_io_hlos.push_back(root);
  for (const auto* hlo : nested_computation.instructions()) {
    if (hlo->opcode() != HloOpcode::kParameter &&
        hlo != nested_computation.root_instruction()) {
      non_io_hlos.push_back(hlo);
    }
  }
  bindings_.EmitBasePointersForHlos(io_hlos, non_io_hlos);

  RETURN_IF_ERROR(nested_computation.root_instruction()->Accept(this));
  b_.SetInsertPoint(ret_instr);

  // Function epilogue: copy the output value back.
  {
    const HloInstruction* root_instruction =
        nested_computation.root_instruction();
    llvm::Value* root_value = bindings_.GetBasePointer(*root_instruction);
    const Shape& return_shape = root_instruction->shape();

    // Last argument is the out parameter.
    llvm::Argument* out_parameter = std::prev(function->arg_end(), 1);

    if (ShapeUtil::IsScalar(return_shape)) {
      llvm::Value* ret_value =
          Load(llvm_ir::ShapeToIrType(return_shape, module_->getContext()),
               root_value, "load_ret_value");
      Store(ret_value, out_parameter);
    } else {
      CHECK(return_shape.IsTuple());
      llvm::Type* tuple_type =
          llvm_ir::ShapeToIrType(return_shape, module_->getContext());

      for (int i = 0; i < return_shape.tuple_shapes().size(); i++) {
        const Shape& element_shape = return_shape.tuple_shapes(i);
        llvm::Value* destination = llvm_ir::EmitGetTupleElement(
            element_shape,
            /*index=*/i,
            /*alignment=*/1, out_parameter, tuple_type, &b_);
        llvm::Value* source = llvm_ir::EmitGetTupleElement(
            element_shape,
            /*index=*/i,
            /*alignment=*/1, root_value,
            llvm_ir::ShapeToIrType(root_instruction->shape(),
                                   module_->getContext()),
            &b_);
        Store(Load(llvm_ir::ShapeToIrType(element_shape, module_->getContext()),
                   source),
              destination);
      }
    }
  }
  b_.SetInsertPoint(ret_instr);
  return function;
}

absl::Status IrEmitter::EmitTargetElementLoop(
    const HloInstruction& hlo,
    const llvm_ir::ElementGenerator& element_generator) {
  // For MOF we give the loop emitter an array for every output it should
  // generate.
  if (hlo.shape().IsTuple()) {
    std::vector<llvm_ir::IrArray> target_arrays =
        ConstructIrArrayForOutputs(hlo);
    RETURN_IF_ERROR(
        llvm_ir::LoopEmitter(element_generator, target_arrays, &b_).EmitLoop());
    llvm_ir::EmitTuple(GetIrArray(hlo, hlo), target_arrays, &b_);
    return absl::OkStatus();
  }
  return llvm_ir::LoopEmitter(element_generator, GetIrArray(hlo, hlo), &b_)
      .EmitLoop();
}

struct KernelThunkInfo {
  std::vector<llvm_ir::IrArray> ir_arrays;
  std::unique_ptr<Thunk> thunk;
};

absl::StatusOr<KernelThunkInfo> BuildKernelThunkForNonFusionOp(
    llvm::Module* llvm_module, const HloInstruction* hlo,
    const BufferAssignment& buffer_assignment, ThunkId thunk_id,
    const se::DeviceDescription& gpu_device_info,
    const std::string& sanitized_kernel_name,

    IrEmitter& ir_emitter, const LaunchDimensions& launch_dimensions) {
  std::string suggested_kernel_name(hlo->name());

  ASSIGN_OR_RETURN(auto kernel_arguments,
                   emitters::KernelArguments::Create(
                       buffer_assignment, GetDefaultBufferAlignment(), hlo));

  VLOG(3) << "Generating (without reuse check): " << suggested_kernel_name;

  ASSIGN_OR_RETURN(
      llvm::Function * kernel,
      BuildKernelPrototype(llvm_module, gpu_device_info, suggested_kernel_name,
                           sanitized_kernel_name, kernel_arguments,
                           launch_dimensions, ir_emitter.builder()));

  auto thunk = std::make_unique<KernelThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(hlo, thunk_id),
      kernel->getName().str(), kernel_arguments, launch_dimensions,
      /*cluster_dim=*/std::nullopt,
      /*shmem_bytes=*/0,
      /*tma_metadata=*/se::gpu::TmaMetadata());

  std::vector<llvm_ir::IrArray> ir_arrays;
  ir_arrays.reserve(kernel_arguments.args().size());
  for (const auto& [kernel_argument, llvm_arg] :
       llvm::zip(kernel_arguments.args(), kernel->args())) {
    llvm::Type* ir_type =
        llvm_ir::ShapeToIrType(kernel_argument.shape(), llvm_arg.getContext());
    llvm_ir::IrArray ir_array(&llvm_arg, ir_type, kernel_argument.shape());

    if (!kernel_argument.written()) {
      ir_array.MarkInvariantOverWholeProgram(&llvm_arg.getContext());
    }
    ir_arrays.push_back(ir_array);
  }
  return {KernelThunkInfo{ir_arrays, std::move(thunk)}};
}

llvm::Value* CreateLoad(llvm::Value* address, llvm::Type* data_type,
                        int alignment_bytes, llvm::IRBuilderBase* b) {
  int data_bytes = data_type->getPrimitiveSizeInBits() /
                   primitive_util::BitWidth(PrimitiveType::U8);
  if (alignment_bytes == 0) {
    return b->CreateLoad(data_type, address);
  }

  int alignment_bitwidth =
      alignment_bytes * primitive_util::BitWidth(PrimitiveType::U8);

  llvm::Value* output = llvm::ConstantInt::get(data_type, 0);
  for (int offset_bytes = 0; offset_bytes < data_bytes;
       offset_bytes += alignment_bytes) {
    llvm::Value* offset_address = b->CreateConstInBoundsGEP1_32(
        b->getInt8Ty(), address, offset_bytes, "offset_address");
    llvm::Value* partial_value = b->CreateLoad(b->getIntNTy(alignment_bitwidth),
                                               offset_address, "partial_value");
    llvm::Value* zextd =
        b->CreateZExt(partial_value, output->getType(), "partial_value_zextd");
    llvm::Value* shifted = b->CreateShl(
        zextd, llvm::ConstantInt::get(b->getInt32Ty(), offset_bytes),
        "partial_input_shifted");
    output = b->CreateAdd(output, shifted, "output_updated");
  }
  return output;
}

void CreateStore(llvm::Value* data, llvm::Value* address, int alignment_bytes,
                 llvm::IRBuilderBase* b) {
  int data_bytes = data->getType()->getPrimitiveSizeInBits() /
                   primitive_util::BitWidth(PrimitiveType::U8);
  CHECK_GE(data_bytes, alignment_bytes);
  if (alignment_bytes == 0) {
    b->CreateStore(data, address);
    return;
  }

  int alignment_bitwidth =
      alignment_bytes * primitive_util::BitWidth(PrimitiveType::U8);

  for (int offset_bytes = 0; offset_bytes < data_bytes;
       offset_bytes += alignment_bytes) {
    llvm::Value* offset_address = b->CreateConstInBoundsGEP1_32(
        b->getInt8Ty(), address, offset_bytes, "offset_address");
    llvm::Value* shifted_partial = b->CreateTrunc(
        b->CreateLShr(data,
                      llvm::ConstantInt::get(b->getInt32Ty(), offset_bytes)),
        b->getIntNTy(alignment_bitwidth), "truncated_value");
    b->CreateStore(shifted_partial, offset_address);
  }
}

}  // namespace

GpuExecutable::ConstantInfo AppendGlobalConstant(
    llvm::Module* module, int64_t num_elements, int64_t bytes_per_element,
    absl::string_view symbol_name, int allocation_idx,
    DenseDataIntermediate content) {
  // LLVM and PTXAS don't deal well with large constants, so we only emit very
  // small constants directly in LLVM IR.  Larger constants are emitted with
  // zero initializers in LLVM IR and are later overwritten when the PTX/CUBIN
  // is loaded.
  bool should_emit_initializer = num_elements <= 1;

  llvm::IRBuilder<> b(module->getContext());
  // Ptxas has issues if the constant allocation is smaller than 64 bytes.
  // TODO(b/253259975): Remove when fixed ptxas version is submitted.
  constexpr int64_t kMinConstAllocationInBytes = 64;
  bool needs_padding =
      num_elements * bytes_per_element < kMinConstAllocationInBytes;

  llvm::ArrayType* global_type = llvm::ArrayType::get(
      b.getInt8Ty(),
      std::max(num_elements * bytes_per_element, kMinConstAllocationInBytes));

  GpuExecutable::ConstantInfo info;
  llvm::Constant* initializer = [&]() -> llvm::Constant* {
    if (!should_emit_initializer) {
      info.content = std::move(content);
      return llvm::ConstantAggregateZero::get(global_type);
    }

    std::vector<uint8_t> padded(kMinConstAllocationInBytes, 0);
    absl::c_copy(content.span(), padded.begin());
    return llvm::ConstantDataArray::get<uint8_t>(
        module->getContext(),
        needs_padding ? llvm::ArrayRef<uint8_t>(padded)
                      : llvm::ArrayRef<uint8_t>(content.span().data(),
                                                content.span().size()));
  }();

  // Explicitly set global addrspace for SPIR backend.
  int addrspace = llvm::Triple(module->getTargetTriple()).isSPIR() ? 1 : 0;
  // These globals will be looked up by name by GpuExecutable so we need to
  // give them an external linkage.  Not all of their uses are visible in
  // the LLVM IR so we can't give then a linkage that merely preserves their
  // names (like available_externally), we also need to ensure that they stick
  // around even if they're "unused".
  //
  // We may have to be more clever here in the future if we notice that we're
  // keeping around too many globals because of their linkage.
  auto* global_for_const = new llvm::GlobalVariable(
      global_type, /*isConstant=*/should_emit_initializer,
      llvm::GlobalValue::ExternalLinkage,
      /*Initializer=*/initializer, symbol_name,
      /*TLMode=*/llvm::GlobalValue::NotThreadLocal,
      /*AddressSpace=*/addrspace,
      /*isExternallyInitialized=*/false);
  global_for_const->setAlignment(llvm::Align(kConstantBufferAlignBytes));
  module->insertGlobalVariable(global_for_const);

  info.symbol_name.assign(symbol_name);
  info.allocation_index = allocation_idx;
  return info;
}

absl::StatusOr<ThunkSequence> EmitBitonicSortLLVMIR(
    const HloSortInstruction* sort, llvm::Module* llvm_module,
    IrEmitterContext* ir_emitter_context) {
  std::string op_name(sort->name());

  IrEmitter ir_emitter(ir_emitter_context, llvm_module, /*nested=*/false);

  int64_t dimension_to_sort = sort->sort_dimension();
  const Shape& keys_shape = sort->operand(0)->shape();
  uint64_t dimension_to_sort_bound = keys_shape.dimensions(dimension_to_sort);
  int64_t num_stages = Log2Ceiling(dimension_to_sort_bound);
  VLOG(2) << op_name << " requires " << num_stages << " stages.";
  CHECK_GE(1ULL << num_stages, dimension_to_sort_bound);
  CHECK_LT(1ULL << (num_stages - 1), dimension_to_sort_bound);

  // Naive C++ code for the outer loops:
  //
  // for (int64_t stage = 0; stage <
  // Log2Ceiling(dimension_to_sort_bound);
  //     ++stage) {
  //   int64_t first_xor_mask = (1LL << (stage + 1)) - 1;
  //   SortInPlace(first_xor_mask);
  //   for (int64_t mask = stage - 1; mask >= 0; --mask) {
  //     int64_t later_xor_mask = 1LL << mask;
  //     SortInPlace(later_xor_mask);
  //   }
  // }
  //
  // This follows the alternative representation of the algorithm
  // described on Wikipedia:
  // https://en.wikipedia.org/wiki/Bitonic_sorter
  //
  // Each mask specifies how to derive from one position in the
  // array the position with which it should be compared (we
  // calculate the xor of the position with the mask). As an
  // optimization, we can move the 'mask' loop to inside the
  // sorting/comparison loop if the comparisons happen within a
  // small block of the array. To make this work, we collect all
  // consecutive masks that are smaller than our chosen power of 2
  // tile size, and pass them to SortInPlace. Each block then
  // processes one tile of data.

  const uint64_t kUnrollFactor = 4;
  // Determine the total element size of all sort operands. We need to choose a
  // tile size such that we have enough shared memory to store a tile of
  // elements from each operand.
  uint64_t total_element_size = 0;
  for (int64_t i = 0; i < sort->operand_count(); ++i) {
    total_element_size += ShapeUtil::ByteSizeOfPrimitiveType(
        sort->operand(i)->shape().element_type());
  }
  const uint64_t kMaxSharedMemoryPerBlock =
      ir_emitter_context->gpu_device_info().shared_memory_per_block();
  uint64_t max_tile_size_fitting_into_shared_memory =
      kMaxSharedMemoryPerBlock / total_element_size;
  const uint64_t kMaxThreadsPerBlock =
      ir_emitter_context->gpu_device_info().threads_per_block_limit();
  // Choose the tile size based on actual amount of elements to sort, the amount
  // of shared memory available, and the maximum number of threads per block.
  uint64_t tile_size =
      std::min(std::min(kMaxThreadsPerBlock * kUnrollFactor,
                        max_tile_size_fitting_into_shared_memory),
               uint64_t{1} << num_stages);
  // The tile size needs to be a power of 2.
  tile_size = uint64_t{1} << Log2Floor(tile_size);

  // If we cannot combine several xor masks together, we don't use
  // tiling, so we calculate the standard launch dimensions for the
  // shape. However we only need to iterate through ~half of the
  // dimension to sort (rounded up to the next highest power of 2),
  // because each iteration compares one pair of elements.
  Shape standard_iteration_shape = keys_shape;
  uint64_t standard_num_iterations_in_sort_dim = 1ULL << (num_stages - 1);
  standard_iteration_shape.set_dimensions(
      dimension_to_sort,
      CeilOfRatio(standard_num_iterations_in_sort_dim, kUnrollFactor));

  LaunchDimensions standard_launch_dimensions = CalculateLaunchDimensions(
      standard_iteration_shape, ir_emitter_context->gpu_device_info());

  // Calculate the launch dimensions for the case where we use
  // tiling. We split the dimension that should be sorted into tiles
  // of size 'tile_size'. This means we first need to round
  // 'dimension_to_sort_bound' up to be a multiple of the tile size.
  uint64_t rounded_bound = RoundUpTo(dimension_to_sort_bound, tile_size);
  Shape iteration_shape = keys_shape;

  // We iterate through the element pairs that should be compared.
  uint64_t num_iterations_in_sort_dim =
      CeilOfRatio(rounded_bound, kUnrollFactor);
  iteration_shape.set_dimensions(dimension_to_sort, num_iterations_in_sort_dim);
  uint64_t num_iterations = ShapeUtil::ElementsIn(iteration_shape);

  // For correctness reasons we need exactly `tile_size` / `kUnrollFactor` many
  // threads per block. Each thread is responsible for copying
  // exactly `kUnrollFactor` many adjacent elements into shared memory, and then
  // does `kUnrollFactor` / 2 many comparisons of two elements taken from shared
  // memory.
  const uint64_t kThreadsPerBlock =
      std::max(uint64_t{1}, tile_size / kUnrollFactor);

  uint64_t num_blocks = CeilOfRatio(num_iterations, kThreadsPerBlock);
  LaunchDimensions tiled_launch_dimensions(num_blocks, kThreadsPerBlock);
  VLOG(2) << absl::StreamFormat("%s launch dims: %d blocks, %d threads/block",
                                op_name, num_blocks, kThreadsPerBlock);
  ThunkSequence thunks;
  bool emit_iota_operands = true;
  auto emit_kernel = [&](absl::Span<const int64_t> xor_masks) {
    VLOG(2) << absl::StreamFormat(
        "%s uses kernel for xor masks [%s]", op_name,
        absl::StrJoin(xor_masks, ", ", [](std::string* out, int64_t xor_mask) {
          absl::StrAppendFormat(out, "0x%x", xor_mask);
        }));
    LaunchDimensions launch_dimensions = xor_masks.size() > 1
                                             ? tiled_launch_dimensions
                                             : standard_launch_dimensions;
    bool is_fusion = sort->parent()->IsFusionComputation();
    const HloInstruction* hlo_with_buffers =
        is_fusion ? sort->parent()->FusionInstruction() : sort;
    ASSIGN_OR_RETURN(KernelThunkInfo kernel_thunk_info,
                     BuildKernelThunkForNonFusionOp(
                         llvm_module, hlo_with_buffers,
                         ir_emitter_context->buffer_assignment(),
                         ir_emitter_context->GetNextThunkId(),
                         ir_emitter_context->gpu_device_info(),
                         ir_emitter_context->GetSanitizedUniqueName(op_name),
                         ir_emitter, launch_dimensions));
    thunks.push_back(std::move(kernel_thunk_info.thunk));

    // The first `operand_count()` elements of `ir_arrays` are the input
    // operands and the rest are the output arrays. Inputs are aliases with
    // outputs, so we need to pass only the outputs to the in-place sort kernel.
    auto output_arrays_span =
        absl::Span<const llvm_ir::IrArray>(kernel_thunk_info.ir_arrays)
            .subspan(hlo_with_buffers->operand_count());

    auto* comparator = sort->called_computations().front();
    auto* builder = ir_emitter.builder();
    auto result = llvm_ir::EmitSortInPlace(
        sort, output_arrays_span, emit_iota_operands, llvm_ir::IrName(op_name),
        xor_masks, ir_emitter.module(), ir_emitter.builder(), launch_dimensions,
        xor_masks.size() > 1 ? num_iterations_in_sort_dim
                             : standard_num_iterations_in_sort_dim,
        tile_size, kUnrollFactor,
        [&](absl::Span<llvm::Value* const> operands, llvm::Value* output) {
          return CallNestedComputation(builder, *ir_emitter_context,
                                       llvm_module, *comparator, operands,
                                       output);
        });
    emit_iota_operands = false;
    return result;
  };
  std::vector<int64_t> xor_masks;
  for (int64_t stage = 0; stage < num_stages; ++stage) {
    for (int64_t mask = stage; mask >= 0; --mask) {
      int64_t xor_mask;
      if (mask == stage) {
        xor_mask = (1LL << (stage + 1)) - 1;
      } else {
        xor_mask = 1LL << mask;
      }
      if (xor_mask >= tile_size) {
        if (!xor_masks.empty()) {
          RETURN_IF_ERROR(emit_kernel(xor_masks));
          xor_masks.clear();
        }
        RETURN_IF_ERROR(emit_kernel({xor_mask}));
      } else {
        xor_masks.push_back(xor_mask);
      }
    }
  }
  if (!xor_masks.empty()) {
    RETURN_IF_ERROR(emit_kernel(xor_masks));
  }
  return thunks;
}

// Input = {dynamic array(with dynamic dimension meta data at the
// end)} Output = {static array, dynamic_dim0, dynamic_dim1}
absl::StatusOr<ThunkSequence> EmitPadToStaticLLVMIR(
    const HloCustomCallInstruction* hlo, llvm::Module* llvm_module,
    IrEmitterContext* ir_emitter_context) {
  std::string ir_name = std::string(hlo->name());

  IrEmitter ir_emitter(ir_emitter_context, llvm_module, /*nested=*/false);

  constexpr int kUnrollFactor = 1;
  const Shape& input_shape = hlo->operand(0)->shape();

  LaunchDimensions launch_dimensions = CalculateLaunchDimensions(
      input_shape, ir_emitter_context->gpu_device_info(), {kUnrollFactor});

  ASSIGN_OR_RETURN(
      KernelThunkInfo kernel_thunk_info,
      BuildKernelThunkForNonFusionOp(
          llvm_module, hlo, ir_emitter_context->buffer_assignment(),
          ir_emitter_context->GetNextThunkId(),
          ir_emitter_context->gpu_device_info(),
          ir_emitter_context->GetSanitizedUniqueName(ir_name), ir_emitter,
          launch_dimensions));
  ThunkSequence thunk_sequence;
  thunk_sequence.push_back(std::move(kernel_thunk_info.thunk));

  const llvm_ir::IrArray& source_array = kernel_thunk_info.ir_arrays[0];
  const llvm_ir::IrArray& output_array = kernel_thunk_info.ir_arrays[1];
  auto output_dim_arrays =
      absl::Span<const llvm_ir::IrArray>(kernel_thunk_info.ir_arrays)
          .subspan(2);

  llvm::Type* index_ty = GetIndexTypeForKernel(
      hlo, launch_dimensions.launch_bound(), ir_emitter.builder());

  // pseudo code for PadToStatic on a 2d array
  //   int* source_array = args[0];
  //   int* dest_array = args[1];
  llvm::Value* source_buffer = source_array.GetBasePointer();

  // TODO(jurahul): input_shape here is the static shape of the
  // input (which has a dynamic shape in XLA). Currently, we are
  // mapping that to a static shaped memref. When we change that to
  // a more appropriate representation in MLIR, fix this code to
  // correctly deduce the static shape backing the dynamically
  // shaped memref.
  int64_t raw_data_size = ShapeUtil::ByteSizeOf(input_shape);

  //   int* dyn_dim0_size = source_array + meta_data_offset;
  //   int* dyn_dim1_size = source_array + meta_data_offset +
  //   sizeof(int);
  std::vector<llvm::Value*> dynamic_dims;
  int alignment = raw_data_size % sizeof(int32_t);
  std::vector<ShapeUtil::IndexedShape> output_shapes =
      ShapeUtil::GetLeafShapes(hlo->shape());

  for (int64_t i = 1; i < output_shapes.size(); ++i) {
    // Dynamic size of each dimension is attached at the end of the
    // source array(operand(0)). We need to extract these value.
    const Shape& dim_shape = output_shapes[i].shape;
    TF_RET_CHECK(Shape::Equal()(dim_shape, ShapeUtil::MakeScalarShape(S32)));

    const int64_t dim_index = i - 1;
    llvm::Value* metadata = ir_emitter.builder()->CreateConstInBoundsGEP1_32(
        ir_emitter.builder()->getInt8Ty(), source_buffer,
        raw_data_size + dim_index * sizeof(int32_t));
    llvm::Value* dyn_dim_size =
        CreateLoad(metadata, ir_emitter.builder()->getInt32Ty(), alignment,
                   ir_emitter.builder());
    dynamic_dims.push_back(dyn_dim_size);
  }

  // only one thread need to store the dynamic index
  //   int thread_id = GetThreadId();
  //   int block_id = GetBlockId();
  //   if (thread_id == 0 && block_id == 0) {
  //     *output[1] = *dyn_dim0_size;
  //     *output[2] = *dyn_dim1_size;
  //   }
  KernelSupportLibrary{ir_emitter.builder()}.If(
      "is_thread_0", IsBlock0Thread0(ir_emitter.builder()), [&] {
        for (int64_t i = 1; i < output_shapes.size(); ++i) {
          const int64_t dim_index = i - 1;
          llvm::Value* dest_dim_size_address =
              output_dim_arrays[dim_index].GetBasePointer();
          // output[i] stores dynamic_dim_(i-1)
          CreateStore(dynamic_dims[dim_index], dest_dim_size_address, alignment,
                      ir_emitter.builder());
        }
      });

  //     int dyn_element_total = 1;
  //     dyn_element_total *= *dyn_dim0_size;
  //     dyn_element_total *= *dyn_dim1_size;
  llvm::Value* dyn_element_total = llvm::ConstantInt::get(index_ty, 1);
  for (llvm::Value* dynamic_dim : dynamic_dims) {
    dyn_element_total = ir_emitter.builder()->CreateMul(
        dyn_element_total,
        ir_emitter.builder()->CreateIntCast(dynamic_dim,
                                            dyn_element_total->getType(),
                                            /*isSigned=*/true),
        /*Name=*/"dyn_element_total_pad");
  }

  //   linear_index = block_id * threads_per_block + thread_id;
  //   if (linear_index < max_num_element) {
  //     Index static_index =
  //         delinerized(linerized_index, static_dim0_size,
  //         static_dim1_size);
  //     if (linerized_index < dyn_element_total) {
  //       Index dyn_index =
  //           delinerized(linerized_index, *dyn_dim0_size,
  //           *dyn_dim1_size);
  //       dest_array[dyn_index.dim0][dyn_index.dim1] =
  //           source_array[static_index.dim0][static_index.dim1];
  //     }
  //   }
  llvm_ir::BodyEmitter body_generator =
      [&](const llvm_ir::IrArray::Index& array_index) -> absl::Status {
    llvm::Value* linearIndex =
        array_index.Linearize(input_shape.dimensions(), ir_emitter.builder());
    auto if_in_dyn_bounds = llvm_ir::EmitIfThenElse(
        ir_emitter.builder()->CreateICmpULT(linearIndex, dyn_element_total),
        llvm_ir::IrName(ir_name, "in_dyn_bounds"), ir_emitter.builder(), false);
    // Set IR builder insertion point to the body of the if
    // structure.
    llvm_ir::SetToFirstInsertPoint(if_in_dyn_bounds.true_block,
                                   ir_emitter.builder());
    llvm_ir::IrArray::Index dyn_index(linearIndex, input_shape,
                                      absl::MakeSpan(dynamic_dims),
                                      ir_emitter.builder());
    output_array.EmitWriteArrayElement(
        dyn_index,
        source_array.EmitReadArrayElement(array_index, ir_emitter.builder(),
                                          /*name=*/""),
        ir_emitter.builder(),
        /*use_linear_index=*/false);
    return absl::OkStatus();
  };

  const Shape& data_shape = hlo->shape().tuple_shapes(0);
  RETURN_IF_ERROR(ParallelLoopEmitter(body_generator, data_shape,
                                      launch_dimensions, ir_emitter.builder(),
                                      {kUnrollFactor})
                      .EmitLoop(ir_name, index_ty));
  return thunk_sequence;
}

// Input = {dynamic array(with dynamic dimension meta data at the
// end)} Output = {static array, dynamic_dim0, dynamic_dim1}
absl::StatusOr<ThunkSequence> EmitSliceToDynamicLLVMIR(
    const HloCustomCallInstruction* hlo, llvm::Module* llvm_module,
    IrEmitterContext* ir_emitter_context) {
  std::string ir_name = std::string(hlo->name());

  IrEmitter ir_emitter(ir_emitter_context, llvm_module, /*nested=*/false);
  constexpr int kUnrollFactor = 1;
  const Shape& input_shape = hlo->operand(0)->shape();

  LaunchDimensions launch_dimensions = CalculateLaunchDimensions(
      input_shape, ir_emitter_context->gpu_device_info(), {kUnrollFactor});
  llvm::Type* index_ty = GetIndexTypeForKernel(
      hlo, launch_dimensions.launch_bound(), ir_emitter.builder());
  ASSIGN_OR_RETURN(
      KernelThunkInfo kernel_thunk_info,
      BuildKernelThunkForNonFusionOp(
          llvm_module, hlo, ir_emitter_context->buffer_assignment(),
          ir_emitter_context->GetNextThunkId(),
          ir_emitter_context->gpu_device_info(),
          ir_emitter_context->GetSanitizedUniqueName(ir_name), ir_emitter,
          launch_dimensions));
  ThunkSequence thunk_sequence;
  thunk_sequence.push_back(std::move(kernel_thunk_info.thunk));

  const Shape& data_shape = ShapeUtil::MakeStaticShape(hlo->shape());
  TF_RET_CHECK(data_shape.IsArray());

  // TODO(jurahul): data_shape here is the static shape of the
  // output (which has a dynamic shape in XLA). Currently, we are
  // mapping that to a static shaped memref. When we change that to
  // a more appropriate representation in MLIR, fix this code to
  // correctly deduce the static shape backing the dynamically
  // shaped memref.

  // calculate the location where metadata needs to be inserted
  //   int* dyn_dim0_size = dest_array + meta_data_offset;
  //   int* dyn_dim1_size = dest_array + meta_data_offset +
  //   sizeof(int);
  int32_t raw_data_size = ShapeUtil::ByteSizeOf(data_shape);

  // pseudo code for sliceToDynamic on a 2d array
  //   int* source_array = args[0];
  //   int* dest_array = args.back();
  const auto& ir_arrays = kernel_thunk_info.ir_arrays;
  const llvm_ir::IrArray& data_array = ir_arrays.back();
  llvm::Value* dest_buffer = data_array.GetBasePointer();

  // Load dynamic dimensions from memory.
  std::vector<llvm::Value*> dynamic_dims;
  int alignment = raw_data_size % sizeof(int32_t);
  for (int64_t i = 1; i < hlo->operand_count(); ++i) {
    llvm::Value* source_buffer = ir_arrays[i].GetBasePointer();
    llvm::Type* source_buffer_pointee_type = ir_arrays[i].GetBasePointeeType();
    llvm::LoadInst* dyn_dim_size = ir_emitter.builder()->CreateLoad(
        source_buffer_pointee_type, source_buffer, "dyn_dim_size");
    dynamic_dims.push_back(dyn_dim_size);
  }

  // only one thread need to store the dynamic index
  //   int thread_id = GetThreadId();
  //   int block_id = GetBlockId();
  //   if (thread_id == 0 && block_id == 0) {
  //     *dyn_dim0_size = *output[1];
  //     *dyn_dim1_size = *output[2];
  //   }
  KernelSupportLibrary{ir_emitter.builder()}.If(
      "is_thread_0", IsBlock0Thread0(ir_emitter.builder()), [&] {
        for (int64_t i = 1; i < hlo->operand_count(); ++i) {
          const int64_t dim_index = i - 1;
          llvm::Value* metadata =
              ir_emitter.builder()->CreateConstInBoundsGEP1_32(
                  ir_emitter.builder()->getInt8Ty(), dest_buffer,
                  raw_data_size + dim_index * sizeof(int32_t));
          // output[i] stores dynamic_dim_(i-1)
          CreateStore(dynamic_dims[dim_index], metadata, alignment,
                      ir_emitter.builder());
        }
      });

  //     int dyn_element_total = 1;
  //     dyn_element_total *= dyn_dim0_size;
  //     dyn_element_total *= dyn_dim1_size;
  llvm::Value* dyn_element_total = llvm::ConstantInt::get(index_ty, 1);
  for (llvm::Value* dynamic_dim : dynamic_dims) {
    dyn_element_total = ir_emitter.builder()->CreateMul(
        dyn_element_total,
        ir_emitter.builder()->CreateIntCast(dynamic_dim,
                                            dyn_element_total->getType(),
                                            /*isSigned=*/true),
        /*Name=*/"dyn_element_total_slice");
  }

  //   linear_index = block_id * threads_per_block + thread_id;
  //   if (linear_index < max_num_element) {
  //     Index static_index =
  //         delinerized(linerized_index, static_dim0_size,
  //         static_dim1_size);
  //     if (linerized_index < dyn_element_total) {
  //       Index dyn_index =
  //           delinerized(linerized_index, *dyn_dim0_size,
  //           *dyn_dim1_size);
  //       dest_array[static_index.dim0][static_index.di] =
  //           source_array[dyn_index.dim0][dyn_index.dim1];
  //     }
  //   }
  llvm_ir::BodyEmitter body_generator =
      [&](const llvm_ir::IrArray::Index& array_index) -> absl::Status {
    llvm::Value* linearIndex =
        array_index.Linearize(input_shape.dimensions(), ir_emitter.builder());
    auto if_in_dyn_bounds = llvm_ir::EmitIfThenElse(
        ir_emitter.builder()->CreateICmpULT(linearIndex, dyn_element_total),
        llvm_ir::IrName(ir_name, "in_dyn_bounds"), ir_emitter.builder(), false);
    // Set IR builder insertion point to the body of the if
    // structure.
    llvm_ir::SetToFirstInsertPoint(if_in_dyn_bounds.true_block,
                                   ir_emitter.builder());
    llvm_ir::IrArray::Index dyn_index(linearIndex, input_shape,
                                      absl::MakeSpan(dynamic_dims),
                                      ir_emitter.builder());

    data_array.EmitWriteArrayElement(
        array_index,
        ir_arrays[0].EmitReadArrayElement(dyn_index, ir_emitter.builder(),
                                          /*name=*/"",
                                          /*use_linear_index=*/false),
        ir_emitter.builder());
    return absl::OkStatus();
  };

  RETURN_IF_ERROR(ParallelLoopEmitter(body_generator, data_shape,
                                      launch_dimensions, ir_emitter.builder(),
                                      {kUnrollFactor})
                      .EmitLoop(ir_name, index_ty));
  return thunk_sequence;
}

absl::StatusOr<ThunkSequence> EmitRngGetAndUpdateStateLLVMIR(
    const HloRngGetAndUpdateStateInstruction* hlo, llvm::Module* llvm_module,
    IrEmitterContext* ir_emitter_context) {
  std::string ir_name = std::string(hlo->name());

  IrEmitter ir_emitter(ir_emitter_context, llvm_module, /*nested=*/false);

  auto& b = *ir_emitter.builder();
  // Emit a kernel to increment the global state for Philox RNG
  // algorithm.
  ASSIGN_OR_RETURN(
      KernelThunkInfo kernel_thunk_info,
      BuildKernelThunkForNonFusionOp(
          llvm_module, hlo, ir_emitter_context->buffer_assignment(),
          ir_emitter_context->GetNextThunkId(),
          ir_emitter_context->gpu_device_info(),
          ir_emitter_context->GetSanitizedUniqueName(ir_name), ir_emitter,
          LaunchDimensions()));
  ThunkSequence thunk_sequence;
  thunk_sequence.push_back(std::move(kernel_thunk_info.thunk));

  auto& ir_arrays = kernel_thunk_info.ir_arrays;
  llvm::Value* old_state =
      llvm_ir::RngGetAndUpdateState(hlo->delta(), llvm_module, &b);
  llvm::Value* output_address = ir_arrays[0].EmitArrayElementAddress(
      llvm_ir::IrArray::Index(
          /*linear=*/b.getInt64(0), hlo->shape(), &b),
      &b, "rng_state_address");
  b.CreateStore(old_state, output_address);
  return thunk_sequence;
}

}  // namespace xla::gpu
