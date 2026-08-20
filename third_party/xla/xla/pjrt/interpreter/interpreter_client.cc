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

#include "xla/pjrt/interpreter/interpreter_client.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/die_if_null.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/client/executable_build_options.h"
#include "xla/future.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/evaluator/hlo_evaluator_interface.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/interpreter/interpreter_executable.h"
#include "xla/pjrt/interpreter/interpreter_topology_description.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/utils.h"
#include "xla/runtime/device_id.h"
#include "xla/service/computation_placer.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

bool ShapesMatch(const Shape& expected_shape, const Shape& actual_shape) {
  if (expected_shape.is_dynamic()) {
    return ShapeUtil::DynamicArrayShapeIsCompatible(actual_shape,
                                                    expected_shape);
  }
  return Shape::Equal().MinorToMajorOnlyInLayout()(expected_shape,
                                                   actual_shape);
}

// Handles custom_call ops during evaluation by routing them through the global
// CPU registry used by other CPU-based backends.
absl::StatusOr<Literal> HandleEvaluatorCustomCall(
    const HloInstruction* custom_call, absl::Span<const Literal*> operands) {
  // Find the target C function in the global registry.
  CustomCallTargetRegistry* const registry = CustomCallTargetRegistry::Global();
  void* const target_fn =
      registry->Lookup(custom_call->custom_call_target(), "Host");
  if (target_fn == nullptr) {
    return NotFound("Custom call target '%s' was not registered",
                    custom_call->custom_call_target());
  }

  // Populate pointers to operand and output literal data.
  std::vector<const void*> operand_data;
  operand_data.reserve(operands.size());
  for (const Literal* const literal : operands) {
    operand_data.push_back(literal->untyped_data());
  }
  Literal output = Literal::CreateFromShape(custom_call->shape());
  void* const output_data = output.untyped_data();

  // Call the target function matching the C ABI used by the CPU backends.
  auto* typed_fn = reinterpret_cast<void (*)(void*, const void**)>(target_fn);
  (*typed_fn)(output_data, operand_data.data());

  return std::move(output);
}

// Extract the input literals from the provided buffers.
//
// If there is a tupled argument and the arguments are not tupled, the extracted
// literals will be reconstituted into a tuple. The second element of the
// returned tuple is storage for the tupled literal, if required. Otherwise it
// is nullptr.
absl::StatusOr<std::tuple<std::vector<Literal*>, std::unique_ptr<Literal>>>
ExtractInterpreterInputLiteralsFromBuffers(
    const absl::Span<PjRtBuffer* const> buffers,
    const HloComputation& entry_computation,
    const bool parameter_is_tupled_arguments) {
  std::vector<Literal*> literals;
  for (PjRtBuffer* const buffer : buffers) {
    InterpreterLiteralWrapperBuffer* interpreter_buffer =
        dynamic_cast<InterpreterLiteralWrapperBuffer*>(buffer);
    if (interpreter_buffer == nullptr) {
      return absl::InvalidArgumentError(
          "Interpreter only supports InterpreterLiteralWrapperBuffers");
    }
    literals.push_back(&interpreter_buffer->mutable_literal());
  }

  // Return early if arguments don't need to be re-tupled.
  if (!parameter_is_tupled_arguments) {
    return std::make_tuple(std::move(literals), nullptr);
  }

  if (entry_computation.num_parameters() != 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Interpreter expected a single tupled entry parameter, but got %d.",
        entry_computation.num_parameters()));
  }

  // Re-tuple input arguments. PjRt is commonly used in a mode where the input
  // tuple (if present) is flattened and passed as a vector of argument
  // buffers. The HloEvaluator expects the input to be tupled in these cases.
  std::vector<const Literal*> literal_ptrs;
  literal_ptrs.reserve(literals.size());
  for (const Literal* literal : literals) {
    literal_ptrs.push_back(literal);
  }
  auto tupled_arg_literal =
      std::make_unique<Literal>(LiteralUtil::MakeTuple(literal_ptrs));

  // Replace arg literals with the tupled literal.
  literals.clear();
  literals.push_back(tupled_arg_literal.get());
  return std::make_tuple(std::move(literals), std::move(tupled_arg_literal));
}

}  // namespace

InterpreterLoadedExecutable::InterpreterLoadedExecutable(
    PjRtClient* absl_nonnull client,
    std::shared_ptr<InterpreterExecutable> executable,
    std::unique_ptr<HloEvaluatorInterface> hlo_evaluator,
    std::shared_ptr<DeviceAssignment> device_assignment,
    std::vector<LogicalDeviceIds> addressable_device_logical_ids,
    std::vector<PjRtDevice*> addressable_devices)
    : client_(ABSL_DIE_IF_NULL(client)),
      name_(ABSL_DIE_IF_NULL(executable)->name()),
      executable_(std::move(executable)),
      hlo_evaluator_(std::move(hlo_evaluator)),
      device_assignment_(std::move(device_assignment)),
      addressable_device_logical_ids_(
          std::move(addressable_device_logical_ids)),
      addressable_devices_(std::move(addressable_devices)) {
  if (executable_ && executable_->dynamic_dimension_inference().has_value()) {
    hlo_evaluator_->set_dynamic_dimension_inference(
        &executable_->dynamic_dimension_inference().value());
  }
}

InterpreterExecutable* InterpreterLoadedExecutable::GetExecutable() const {
  absl::MutexLock lock(mutex_);
  return executable_.get();
}

int InterpreterLoadedExecutable::num_replicas() const {
  absl::MutexLock lock(mutex_);
  return executable_ ? executable_->num_replicas() : 1;
}

int InterpreterLoadedExecutable::num_partitions() const {
  absl::MutexLock lock(mutex_);
  return executable_ ? executable_->num_partitions() : 1;
}

absl::string_view InterpreterLoadedExecutable::name() const { return name_; }

absl::StatusOr<std::vector<std::shared_ptr<HloModule>>>
InterpreterLoadedExecutable::GetHloModules() const {
  absl::MutexLock lock(mutex_);
  if (!executable_) {
    return std::vector<std::shared_ptr<HloModule>>{};
  }
  return executable_->GetHloModules();
}

absl::StatusOr<CompileOptions> InterpreterLoadedExecutable::GetCompileOptions()
    const {
  absl::MutexLock lock(mutex_);
  if (!executable_) {
    return absl::InternalError("Executable was deleted.");
  }
  return executable_->GetCompileOptions();
}

void InterpreterLoadedExecutable::Delete() {
  absl::MutexLock lock(mutex_);
  if (hlo_evaluator_ != nullptr) {
    hlo_evaluator_->set_dynamic_dimension_inference(nullptr);
    hlo_evaluator_ = nullptr;
  }
  executable_ = nullptr;
}

bool InterpreterLoadedExecutable::IsDeleted() const {
  absl::MutexLock lock(mutex_);
  return executable_ == nullptr;
}

absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
InterpreterLoadedExecutable::Execute(
    absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
    const ExecuteOptions& options,
    std::optional<std::vector<Future<>>>& returned_futures) const {
  if (device_assignment_ == nullptr) {
    return absl::InvalidArgumentError(
        "Execute expects a non-null device_assignment");
  }
  if (argument_handles.size() != addressable_devices_.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Attempted to execute with %d argument lists when device count is %d "
        "(total replica count: %d, partition count: %d)",
        argument_handles.size(), addressable_devices_.size(), num_replicas(),
        num_partitions()));
  }
  if (addressable_devices_.size() != 1) {
    return absl::InvalidArgumentError(
        "Attempted to execute with multiple devices, but interpreter "
        "only supports single device execution.");
  }

  std::optional<Future<>> returned_future;
  ABSL_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<PjRtBuffer>> replica_result,
      ExecuteSharded(argument_handles[0], addressable_devices_[0], options,
                     returned_future, returned_futures.has_value()));
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> result;
  result.push_back(std::move(replica_result));
  if (returned_futures.has_value()) {
    CHECK(returned_future.has_value())
        << "returned_future must be set because ExecuteSharded was called with "
           "fill_future=true.";
    returned_futures = std::vector<Future<>>({*std::move(returned_future)});
  }
  return result;
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
InterpreterLoadedExecutable::ExecuteSharded(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options, std::optional<Future<>>& returned_future,
    bool fill_future) const {
  if (device_assignment_ == nullptr) {
    return absl::InvalidArgumentError(
        "ExecuteSharded expects a non-null device_assignment");
  }
  std::shared_ptr<InterpreterExecutable> executable;
  {
    absl::MutexLock lock(mutex_);
    if (!executable_ || !executable_->hlo_module()) {
      return absl::InternalError("Executable was deleted.");
    }
    executable = executable_;
  }
  // Since there is only one device, the device should always be the same. Check
  // anyways just to be sure.
  if (device == nullptr ||
      !absl::c_any_of(
          addressable_devices_,
          [needle = device](PjRtDevice* const d) { return d == needle; })) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "ExecuteSharded attempted to execute on device %s, which is not "
        "addressable by this client.",
        device != nullptr
            ? absl::StrCat("id ", device->global_device_id().value())
            : "null"));
  }

  // Apply the ExecuteOptions to the module being executed. The HloEvaluator
  // expects to find the seed in the `HloModuleConfig`.
  const HloModule* hlo_module_to_execute = executable->hlo_module().get();
  std::unique_ptr<HloModule> updated_hlo_module = nullptr;
  if (options.seed != 0) {
    updated_hlo_module = executable->hlo_module()->Clone("");
    updated_hlo_module->mutable_config().set_seed(options.seed);
    hlo_module_to_execute = updated_hlo_module.get();
  }

  // Extract the literals from the arguments.
  const HloComputation& computation =
      *hlo_module_to_execute->entry_computation();
  ABSL_ASSIGN_OR_RETURN(
      const auto literals_and_storage,
      ExtractInterpreterInputLiteralsFromBuffers(
          argument_handles, computation,
          executable->compile_options().parameter_is_tupled_arguments));
  const absl::Span<const Literal* const> literals =
      std::get<0>(literals_and_storage);
  if (computation.num_parameters() != literals.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Mismatch between argument count (%d) and graph parameter count (%d).",
        literals.size(), computation.num_parameters()));
  }

  // Check that the args have the right shape.
  for (int64_t i = 0; i < computation.num_parameters(); ++i) {
    const Shape& expected_shape = computation.parameter_instruction(i)->shape();
    const Shape& actual_shape = literals[i]->shape();
    if (!ShapesMatch(expected_shape, actual_shape)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Shape mismatch on parameter %d. Expected %s but was %s.", i,
          ShapeUtil::HumanStringWithLayout(expected_shape),
          ShapeUtil::HumanStringWithLayout(actual_shape)));
    }
  }

  ABSL_ASSIGN_OR_RETURN(Literal result_literal,
                   Evaluate(computation, literals, options));
  // Shrink the generated dynamic shape into static shape.
  result_literal = result_literal.ToStatic();
  if (fill_future) {
    returned_future = Future<>(absl::OkStatus());
  }

  ABSL_ASSIGN_OR_RETURN(PjRtMemorySpace * memory_space,
                   device->default_memory_space());

  // Transform the result literal back into a one or more
  // InterpreterLiteralWrapperBuffer.
  std::vector<std::unique_ptr<PjRtBuffer>> result;
  if (result_literal.shape().IsTuple()) {
    const int tuple_count = result_literal.shape().tuple_shapes().size();
    result.reserve(tuple_count);
    // DecomposeTuple invalidates result_literal. move(...) to make it obvious.
    std::vector<Literal> tuple_elements =
        std::move(result_literal).DecomposeTuple();
    CHECK(tuple_count == tuple_elements.size())
        << "DecomposedTuple returned the wrong number of elements.";
    for (int i = 0; i < tuple_count; ++i) {
      result.push_back(std::make_unique<InterpreterLiteralWrapperBuffer>(
          client_, memory_space, std::move(tuple_elements[i])));
    }
  } else {
    result.push_back(std::make_unique<InterpreterLiteralWrapperBuffer>(
        client_, memory_space, std::move(result_literal)));
  }
  return result;
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
InterpreterLoadedExecutable::ExecutePortable(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options, std::optional<Future<>>& returned_future,
    bool fill_future) const {
  return absl::UnimplementedError("ExecutePortable is not implemented");
}

absl::StatusOr<Literal> InterpreterLoadedExecutable::Evaluate(
    const HloComputation& computation,
    absl::Span<const Literal* const> arg_literals,
    const ExecuteOptions& options) const {
  absl::MutexLock lock(mutex_);
  if (!hlo_evaluator_) {
    return absl::InternalError("Executable was deleted.");
  }
  if (!options.hlo_output_callbacks.empty()) {
    absl::flat_hash_map<int64_t, const HloOutputCallback*> cb_map;
    for (const auto& cb : options.hlo_output_callbacks) {
      cb_map[cb.callback_id] = &cb;
    }
    hlo_evaluator_->set_eval_literal_handler([cb_map = std::move(cb_map)](
                                                 const HloInstruction* hlo,
                                                 const LiteralSlice& literal) {
      const auto& attr_map = hlo->frontend_attributes().map();
      if (auto it = attr_map.find("_xla_tag"); it != attr_map.end()) {
        int64_t tag_id;
        if (absl::SimpleAtoi(it->second, &tag_id)) {
          if (auto cb_it = cb_map.find(tag_id); cb_it != cb_map.end()) {
            std::shared_ptr<const Literal> shared_literal =
                std::make_shared<const Literal>(literal.Clone());
            cb_it->second->callback(0, 0, absl::MakeSpan(&shared_literal, 1));
          }
        }
      }
    });
  }
  hlo_evaluator_->ResetVisitStates();
  auto result = hlo_evaluator_->Evaluate(computation, arg_literals);
  hlo_evaluator_->set_eval_literal_handler(nullptr);
  return result;
}

InterpreterClient::InterpreterClient()
    : InterpreterClient([]() { return std::make_unique<HloEvaluator>(); }) {}

InterpreterClient::InterpreterClient(
    absl::AnyInvocable<std::unique_ptr<HloEvaluatorInterface>() const>
        hlo_evaluator_factory)
    : hlo_evaluator_factory_(std::move(hlo_evaluator_factory)),
      topology_(std::make_unique<InterpreterTopologyDescription>()),
      interpreter_device_{this},
      interpreter_memory_space_{this},
      devices_({&interpreter_device_}),
      memory_spaces_({&interpreter_memory_space_}) {}

std::optional<PjRtPluginAttributes> InterpreterClient::plugin_attributes()
    const {
  PjRtPluginAttributes attributes =
      PjRtClient::plugin_attributes().value_or(PjRtPluginAttributes());
  attributes.attributes["serialize_with_sdy"] = true;
  return attributes;
}

absl::StatusOr<DeviceAssignment> InterpreterClient::GetDefaultDeviceAssignment(
    int num_replicas, int num_partitions) const {
  return topology_->GetDefaultDeviceAssignment(
      /*process_index=*/0, num_replicas,
      /*num_replicas_per_slice=*/std::nullopt, num_partitions,
      /*multi_slice_config=*/nullptr);
}

absl::StatusOr<DeviceAssignment> InterpreterClient::GetDefaultDeviceAssignment(
    int num_replicas, std::optional<int> num_replicas_per_slice,
    int num_partitions, const MultiSliceConfig* multi_slice_config) const {
  return topology_->GetDefaultDeviceAssignment(
      /*process_index=*/0, num_replicas, num_replicas_per_slice, num_partitions,
      multi_slice_config);
}

absl::StatusOr<Layout> InterpreterClient::GetDefaultLayout(
    PrimitiveType element_type, absl::Span<const int64_t> dims) {
  return topology_->GetDefaultLayout(element_type, dims);
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>> InterpreterClient::Compile(
    const XlaComputation& computation, CompileOptions options) {
  ABSL_ASSIGN_OR_RETURN(const PjRtTopologyDescription* topology,
                   GetTopologyDescription());
  return PjRtCompile(options, computation, *topology, this);
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>> InterpreterClient::Compile(
    MaybeOwningMlirModule module, CompileOptions options) {
  ABSL_ASSIGN_OR_RETURN(const PjRtTopologyDescription* topology,
                   GetTopologyDescription());
  return PjRtCompile(options, std::move(module), *topology, this);
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
InterpreterClient::CompileAndLoad(const XlaComputation& computation,
                                  CompileOptions options) {
  ABSL_ASSIGN_OR_RETURN(const PjRtTopologyDescription* topology,
                   GetTopologyDescription());
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<PjRtExecutable> executable,
                   PjRtCompile(options, computation, *topology, this));
  return Load(std::move(executable), LoadOptions());
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
InterpreterClient::CompileAndLoad(MaybeOwningMlirModule module,
                                  CompileOptions options) {
  ABSL_ASSIGN_OR_RETURN(const PjRtTopologyDescription* topology,
                   GetTopologyDescription());
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<PjRtExecutable> executable,
                   PjRtCompile(options, std::move(module), *topology, this));
  return Load(std::move(executable), LoadOptions());
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> InterpreterClient::Load(
    std::shared_ptr<PjRtExecutable> executable,
    const LoadOptions& load_options) {
  if (executable == nullptr) {
    return absl::InvalidArgumentError("executable cannot be null");
  }
  auto* interpreter_executable =
      dynamic_cast<InterpreterExecutable*>(executable.get());
  if (interpreter_executable == nullptr) {
    return absl::InvalidArgumentError(
        "InterpreterClient::Load expects an InterpreterExecutable");
  }
  auto shared_interpreter_executable =
      std::dynamic_pointer_cast<InterpreterExecutable>(executable);

  auto evaluator = hlo_evaluator_factory_();
  if (interpreter_executable->hlo_module()) {
    evaluator->set_use_fast_path(interpreter_executable->hlo_module()
                                     ->config()
                                     .debug_options()
                                     .xla_hlo_evaluator_use_fast_path());
  }
  evaluator->set_custom_call_handler(HandleEvaluatorCustomCall);

  std::shared_ptr<DeviceAssignment> device_assignment = nullptr;
  int num_replicas = 0, num_partitions = 0;
  CompileOptions compile_options = interpreter_executable->compile_options();
  ABSL_RETURN_IF_ERROR(ParseDeviceAssignmentCompileOptions(
      compile_options.compile_portable_executable,
      &compile_options.executable_build_options,
      [this](int num_replicas, int num_partitions) {
        return GetDefaultDeviceAssignment(num_replicas, num_partitions);
      },
      &num_replicas, &num_partitions, &device_assignment));
  if (device_assignment == nullptr) {
    return absl::InternalError("device_assignment is nullptr");
  }
  if (num_replicas != 1 || num_partitions != 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("num_replicas and num_partitions must be 1. "
                        "num_replicas: %d, num_partitions: %d",
                        num_replicas, num_partitions));
  }
  std::vector<PjRtLoadedExecutable::LogicalDeviceIds>
      addressable_device_logical_ids;
  std::vector<PjRtDevice*> addressable_devices;
  PjRtLoadedExecutable::LogicalDeviceIds logical_device_ids;
  logical_device_ids.replica = 0;
  logical_device_ids.partition = 0;
  addressable_device_logical_ids.push_back(std::move(logical_device_ids));
  addressable_devices.push_back(&interpreter_device_);

  return std::make_unique<InterpreterLoadedExecutable>(
      this, std::move(shared_interpreter_executable), std::move(evaluator),
      std::move(device_assignment), std::move(addressable_device_logical_ids),
      std::move(addressable_devices));
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>>
InterpreterClient::DeserializeExecutable(
    absl::string_view serialized, std::optional<CompileOptions> options) {
  return InterpreterExecutable::Deserialize(serialized, *topology_,
                                            std::move(options));
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
InterpreterClient::LoadSerializedExecutable(
    absl::string_view serialized, std::optional<CompileOptions> options,
    const LoadOptions& load_options) {
  ABSL_ASSIGN_OR_RETURN(auto executable,
                   DeserializeExecutable(serialized, std::move(options)));
  return Load(std::move(executable), load_options);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
InterpreterClient::BufferFromHostLiteral(const LiteralSlice& literal,
                                         PjRtMemorySpace* memory_space,
                                         const Layout* device_layout) {
  if (memory_space == nullptr) {
    memory_space = &interpreter_memory_space_;
  }
  if (device_layout == nullptr) {
    return std::make_unique<InterpreterLiteralWrapperBuffer>(
        memory_space->client(), memory_space, literal);
  }
  Literal device_literal = literal.Relayout(*device_layout);
  return std::make_unique<InterpreterLiteralWrapperBuffer>(
      memory_space->client(), memory_space, std::move(device_literal));
}

absl::StatusOr<PjRtDevice*> InterpreterClient::LookupDevice(
    GlobalDeviceId global_device_id) const {
  if (global_device_id.value() < 0 ||
      global_device_id.value() >= devices_.size()) {
    return InvalidArgument("No matching device found for device_id %d",
                           global_device_id.value());
  }
  return devices_[global_device_id.value()];
}

absl::StatusOr<PjRtDevice*> InterpreterClient::LookupAddressableDevice(
    LocalDeviceId local_device_id) const {
  if (local_device_id.value() < 0 ||
      local_device_id.value() >= addressable_devices().size()) {
    return InvalidArgument(
        "No matching addressable device found for local_device_id %d",
        local_device_id.value());
  }
  return addressable_devices()[local_device_id.value()];
}

}  // namespace xla
