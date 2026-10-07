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

#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_handler_testlib.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/ffi/ffi_attributes_from_backend_config.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_emitter_context.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_handler_registry.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/ffi/attributes.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/logical_buffer.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

int64_t BufferSizeBytes(const BufferValue& buffer) {
  return ShapeUtil::ByteSizeOf(buffer.shape(), sizeof(void*));
}

absl::StatusOr<const HloCustomCallInstruction*> FindCustomCall(
    const HloModule& module, absl::string_view instruction_name) {
  HloComputation& entry = *module.entry_computation();
  const HloInstruction* instruction =
      instruction_name.empty() ? entry.root_instruction()
                               : entry.GetInstructionWithName(instruction_name);
  if (instruction == nullptr) {
    return absl::NotFoundError(absl::StrCat("No instruction named '",
                                            instruction_name,
                                            "' in the entry "
                                            "computation"));
  }
  const auto* custom_call = DynCast<HloCustomCallInstruction>(instruction);
  if (custom_call == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("Instruction '", instruction->name(),
                     "' is not a custom call, it is a ",
                     HloOpcodeString(instruction->opcode())));
  }
  return custom_call;
}

}  // namespace

// Serves a handler the same information `ThunkEmitter` would, but backed by a
// standalone buffer assignment rather than a running compilation.
class NativeCustomCallHandlerTester::ContextImpl
    : public NativeCustomCallEmitterContext {
 public:
  ContextImpl(const HloCustomCallInstruction& instr,
              const BufferAssignment& buffer_assignment,
              const GpuTopology& topology, DebugOptions debug_options)
      : instr_(instr),
        buffer_assignment_(buffer_assignment),
        topology_(topology),
        debug_options_(std::move(debug_options)) {}

  const GpuTopology& GetTargetTopology() const override { return topology_; }

  const stream_executor::DeviceDescription& GetDeviceDescription()
      const override {
    return topology_.gpu_target_config().device_description;
  }

  const DebugOptions& GetDebugOptions() const override {
    return debug_options_;
  }

  Thunk::ThunkInfo GenerateThunkInfo() const override {
    return Thunk::ThunkInfo::WithProfileAnnotation(&instr_,
                                                   ThunkId(next_thunk_id_++));
  }

  absl::StatusOr<BufferAllocation::Slice> GetResultAllocationSlice(
      const ShapeIndex& index) const override {
    return buffer_assignment_.GetUniqueSlice(&instr_, index);
  }

  absl::StatusOr<BufferAllocation::Slice> GetOperandAllocationSlice(
      int64_t operand_index, const ShapeIndex& index) const override {
    ABSL_ASSIGN_OR_RETURN(const HloInstruction* operand, GetOperand(operand_index));
    return buffer_assignment_.GetUniqueSlice(operand, index);
  }

  absl::StatusOr<ShapedSlice> GetResultShapedSlice(
      const ShapeIndex& index) const override {
    return GetShapedSlice(&instr_, index);
  }

  absl::StatusOr<ShapedSlice> GetOperandShapedSlice(
      int64_t operand_index, const ShapeIndex& index) const override {
    ABSL_ASSIGN_OR_RETURN(const HloInstruction* operand, GetOperand(operand_index));
    return GetShapedSlice(operand, index);
  }

  absl::StatusOr<emitters::KernelArguments> CreateKernelArguments(
      absl::Span<const Shape> unmanaged_arguments) const override {
    return emitters::KernelArguments::Create(buffer_assignment_,
                                             GetDefaultBufferAlignment(),
                                             &instr_, unmanaged_arguments);
  }

  absl::StatusOr<xla::ffi::Attributes> GetFfiAttributes() const override {
    return FfiAttributesFromBackendConfig(instr_, mlir_context_);
  }

 private:
  absl::StatusOr<const HloInstruction*> GetOperand(
      int64_t operand_index) const {
    TF_RET_CHECK(operand_index >= 0 && operand_index < instr_.operand_count());
    return instr_.operand(operand_index);
  }

  absl::StatusOr<ShapedSlice> GetShapedSlice(const HloInstruction* instruction,
                                             const ShapeIndex& index) const {
    ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                     buffer_assignment_.GetUniqueSlice(instruction, index));
    ABSL_ASSIGN_OR_RETURN(Shape shape, buffer_assignment_.GetShapeForUniqueSlice(
                                      instruction, index));
    return ShapedSlice{slice, shape};
  }

  const HloCustomCallInstruction& instr_;
  const BufferAssignment& buffer_assignment_;
  const GpuTopology& topology_;
  DebugOptions debug_options_;
  mutable mlir::MLIRContext mlir_context_;
  mutable int64_t next_thunk_id_ = 0;
};

absl::StatusOr<std::unique_ptr<NativeCustomCallHandlerTester>>
NativeCustomCallHandlerTester::Create(absl::string_view hlo_text) {
  return Create(hlo_text, Options());
}

absl::StatusOr<std::unique_ptr<NativeCustomCallHandlerTester>>
NativeCustomCallHandlerTester::Create(absl::string_view hlo_text,
                                      Options options) {
  auto tester = absl::WrapUnique(new NativeCustomCallHandlerTester());

  ABSL_ASSIGN_OR_RETURN(tester->module_, ParseAndReturnUnverifiedModule(hlo_text));
  ABSL_ASSIGN_OR_RETURN(tester->instruction_,
                   FindCustomCall(*tester->module_, options.instruction_name));

  ABSL_ASSIGN_OR_RETURN(
      tester->buffer_assignment_,
      BufferAssigner::Run(
          tester->module_.get(),
          std::make_unique<DependencyHloOrdering>(tester->module_.get()),
          &BufferSizeBytes, &tester->alias_info_,
          [](LogicalBuffer::Color) { return 0; },
          BufferAssigner::Options{/*allocate_buffers_for_constants=*/true}));

  ABSL_ASSIGN_OR_RETURN(stream_executor::GpuTargetConfigProto target_config_proto,
                   GetGpuTargetConfig(options.gpu_model));
  ABSL_ASSIGN_OR_RETURN(GpuTargetConfig target_config,
                   GpuTargetConfig::FromProto(target_config_proto));
  tester->topology_ = std::make_unique<GpuTopology>(
      GetSingleDeviceGpuTopology(target_config.platform_name, target_config));

  tester->context_ = std::make_unique<ContextImpl>(
      *tester->instruction_, *tester->buffer_assignment_, *tester->topology_,
      std::move(options.debug_options));
  return tester;
}

absl::StatusOr<ThunkSequence> NativeCustomCallHandlerTester::EmitThunks()
    const {
  absl::string_view target = instruction_->custom_call_target();
  std::optional<NativeCustomCallHandlerRef> handler =
      NativeCustomCallHandlerRegistry::GetGlobal().Lookup(target);
  if (!handler.has_value()) {
    return absl::NotFoundError(absl::StrCat(
        "No native custom call handler is registered for '", target,
        "'. Is the library that registers it linked into this test?"));
  }
  return EmitThunksWith(*handler);
}

absl::StatusOr<ThunkSequence> NativeCustomCallHandlerTester::EmitThunksWith(
    NativeCustomCallHandlerRef handler) const {
  return handler(*instruction_, *context_);
}

}  // namespace xla::gpu
