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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "nanobind/stl/unique_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/kernel_emitter.h"
#include "xla/codegen/kernel_source.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/testlib/kernel_runner.h"
#include "xla/comparison_util.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/python/nb_absl_inlined_vector.h"  // IWYU pragma: keep
#include "xla/python/nb_absl_span.h"  // IWYU pragma: keep
#include "xla/service/buffer_assignment.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

namespace {

// Use `std::vector<Literal*>` instead of `absl::Span<Literal*>` to take
// advantage of the built in bindings.
void KernelRunnerCall(KernelRunner* kernel_runner,
                      std::vector<Literal*> literals) {
  absl::Status status = kernel_runner->Call(absl::MakeSpan(literals));
  if (!status.ok()) {
    throw std::runtime_error(std::string(status.message()));
  }
}

// Need this helper as Literal requires an explicit clone.
std::unique_ptr<HloInstruction> CreateConstantHloInstruction(
    const Literal& literal) {
  return HloInstruction::CreateConstant(literal.Clone());
}

std::unique_ptr<HloInstruction> CreateDot(
    const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
    const DotDimensionNumbers& dimension_numbers) {
  return HloInstruction::CreateDot(shape, lhs, rhs, dimension_numbers,
                                   PrecisionConfig());
}

std::unique_ptr<HloInstruction> CreateComparisonHloInstruction(
    const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
    Comparison::Direction direction) {
  return HloInstruction::CreateCompare(shape, lhs, rhs, direction);
}

std::unique_ptr<HloInstruction> CreateCallHloInstruction(
    const Shape& shape, std::vector<HloInstruction*> operands,
    HloComputation* computation) {
  return HloInstruction::CreateCall(shape, operands, computation);
}

HloModuleConfig DefaultHloModuleConfigWithDebugOptions() {
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsFromFlags());
  return config;
}

// A dummy kernel runner that implements a simple elementwise add.
class DummyAddKernelRunner final : public KernelRunner {
 public:
  absl::Status Call(absl::Span<const Argument> arguments) override {
    if (arguments.size() != 3) {
      return InvalidArgument("Expected 3 arguments, got %u", arguments.size());
    }

    if (arguments[0].size() != arguments[1].size()) {
      return InvalidArgument(
          "Expected argument 0 to be the same size as argument "
          "1, got %u and %u",
          arguments[0].size(), arguments[1].size());
    }

    if (arguments[1].size() != arguments[2].size()) {
      return InvalidArgument(
          "Expected argument 1 to be the same size as argument "
          "2, got %u and %u",
          arguments[1].size(), arguments[2].size());
    }

    constexpr size_t element_bytes = sizeof(int32_t);

    if ((arguments[0].size() % element_bytes) != 0) {
      return InvalidArgument(
          "Expected arguments to be a multiple of %u bytes, got %u",
          element_bytes, arguments[0].size());
    }

    size_t num_elements = arguments[0].size() / element_bytes;

    auto* in_arg1 = reinterpret_cast<const int32_t*>(arguments[0].data());
    auto* in_arg2 = reinterpret_cast<const int32_t*>(arguments[1].data());
    auto* out_arg = reinterpret_cast<int32_t*>(arguments[2].data());

    for (int i = 0; i < num_elements; ++i) {
      out_arg[i] = in_arg1[i] + in_arg2[i];
    }

    return absl::OkStatus();
  }
};

std::unique_ptr<HloComputation> BuildComputation(
    std::unique_ptr<HloInstruction> root, nanobind::args instructions) {
  HloComputation::Builder builder(absl::StrCat(root->name(), "_computation"));
  for (nanobind::handle handle : instructions) {
    builder.AddInstruction(
        nanobind::cast<std::unique_ptr<HloInstruction>>(handle));
  }
  builder.AddInstruction(std::move(root));

  // Annoyingly if we don't clone the computation, nanobind thinks
  // that the object has been destroyed and will raise an exception
  // after we call get_root_instruction. See
  // https://github.com/wjakob/nanobind/issues/879.
  // TODO(willfroom): Remove the clone once the nanobind bug is
  // fixed and integrated.
  return builder.Build()->Clone();
}

}  // namespace

NB_MODULE(_extension, kernel_runner_module) {
  namespace nb = nanobind;

  nb::class_<KernelSource>(kernel_runner_module, "KernelSource")
      .def("__str__", &KernelSource::ToString);

  nb::class_<KernelSpec> kernel_spec(kernel_runner_module, "KernelSpec");

  nb::class_<KernelDefinition>(kernel_runner_module, "KernelDefinition")
      .def("spec", &KernelDefinition::spec, nb::rv_policy::reference_internal)
      .def("source", &KernelDefinition::source,
           nb::rv_policy::reference_internal);

  nb::class_<KernelEmitter>(kernel_runner_module, "KernelEmitter")
      .def("emit_kernel_definition", [](KernelEmitter* self) {
        absl::StatusOr<KernelDefinition> definition =
            self->EmitKernelDefinition();
        if (!definition.ok()) {
          throw std::runtime_error(std::string(definition.status().message()));
        }
        return std::move(definition).value();
      });

  nb::class_<KernelRunner>(kernel_runner_module, "KernelRunner")
      .def("call", &KernelRunnerCall);

  nb::class_<DummyAddKernelRunner, KernelRunner>(kernel_runner_module,
                                                 "DummyAddKernelRunner")
      .def(nb::init<>());

  nb::enum_<HloOpcode> hlo_opcode(kernel_runner_module, "HloOpcode");
#define DECLARE_ENUM(enum_name, opcode_name, ...)                          \
  hlo_opcode.value(absl::StrReplaceAll(opcode_name, {{"-", "_"}}).c_str(), \
                   HloOpcode::enum_name);
  HLO_OPCODE_LIST(DECLARE_ENUM)
#undef DECLARE_ENUM

  kernel_runner_module.def("opcode_arity", &HloOpcodeArity);

  nb::enum_<Comparison::Direction>(kernel_runner_module, "ComparisonDirection")
      .value("kEq", Comparison::Direction::kEq)
      .value("kNe", Comparison::Direction::kNe)
      .value("kGe", Comparison::Direction::kGe)
      .value("kGt", Comparison::Direction::kGt)
      .value("kLe", Comparison::Direction::kLe)
      .value("kLt", Comparison::Direction::kLt);

  nb::class_<DotDimensionNumbers>(kernel_runner_module, "DotDimensionNumbers")
      .def(
          "__init__",
          [](DotDimensionNumbers* self,
             std::vector<int64_t> lhs_contracting_dims,
             std::vector<int64_t> rhs_contracting_dims,
             std::vector<int64_t> lhs_batch_dims,
             std::vector<int64_t> rhs_batch_dims) {
            new (self) DotDimensionNumbers();
            self->mutable_lhs_contracting_dimensions()->Assign(
                lhs_contracting_dims.begin(), lhs_contracting_dims.end());
            self->mutable_rhs_contracting_dimensions()->Assign(
                rhs_contracting_dims.begin(), rhs_contracting_dims.end());

            self->mutable_lhs_batch_dimensions()->Assign(lhs_batch_dims.begin(),
                                                         lhs_batch_dims.end());
            self->mutable_rhs_batch_dimensions()->Assign(rhs_batch_dims.begin(),
                                                         rhs_batch_dims.end());
          },
          nb::arg("lhs_contracting_dims"), nb::arg("rhs_contracting_dims"),
          nb::arg("lhs_batch_dims") = std::vector<int64_t>{},
          nb::arg("rhs_batch_dims") = std::vector<int64_t>{});

  nb::class_<HloInstruction> hlo_instruction(kernel_runner_module,
                                             "HloInstruction");
  // Factory methods
  hlo_instruction
      .def_static("create_parameter", &HloInstruction::CreateParameter)
      .def_static("create_constant", &CreateConstantHloInstruction)
      .def_static("create_dot", &CreateDot, nb::keep_alive<0, 2>(),
                  nb::keep_alive<0, 3>())
      .def_static("create_unary", &HloInstruction::CreateUnary,
                  nb::keep_alive<0, 3>())
      .def_static("create_binary", &HloInstruction::CreateBinary,
                  nb::keep_alive<0, 3>(), nb::keep_alive<0, 4>())
      .def_static("create_ternary", &HloInstruction::CreateTernary,
                  nb::keep_alive<0, 3>(), nb::keep_alive<0, 4>(),
                  nb::keep_alive<0, 5>())
      .def_static("create_variadic", &HloInstruction::CreateVariadic,
                  nb::keep_alive<0, 3>())
      .def_static("create_compare", &CreateComparisonHloInstruction,
                  nb::keep_alive<0, 2>(), nb::keep_alive<0, 3>())
      .def_static("create_concatenate", &HloInstruction::CreateConcatenate,
                  nb::keep_alive<0, 2>())
      .def_static("create_call", &CreateCallHloInstruction,
                  nb::keep_alive<0, 1>(), nb::keep_alive<0, 2>(),
                  nb::keep_alive<0, 3>())
      .def("name", &HloInstruction::name);

  nb::class_<HloComputation>(kernel_runner_module, "HloComputation")
      .def("__str__",
           nb::overload_cast<>(&HloComputation::ToString, nb::const_));

  // Accessors
  hlo_instruction.def("opcode", &HloInstruction::opcode);
  hlo_instruction.def("shape", &HloInstruction::shape);
  hlo_instruction.def("operands", &HloInstruction::operands,
                      nb::rv_policy::reference_internal);
  hlo_instruction.def(
      "__str__", [](const HloInstruction& self) { return self.ToString(); });

  nb::class_<BufferAssignment>(kernel_runner_module, "BufferAssignment")
      .def("__str__", &BufferAssignment::ToString);

  nb::class_<HloSchedule>(kernel_runner_module, "HloSchedule")
      .def("__str__", &HloSchedule::ToString);

  kernel_runner_module.def("build_hlo_computation", &BuildComputation);

  nb::class_<HloModuleConfig>(kernel_runner_module, "HloModuleConfig")
      .def(nb::new_(&DefaultHloModuleConfigWithDebugOptions));

  nb::class_<HloModule>(kernel_runner_module, "HloModule")
      .def("__init__",
           [](HloModule* self, absl::string_view name) {
             new (self) HloModule(std::string(name),
                                  DefaultHloModuleConfigWithDebugOptions());
           })
      .def_static("parse_from_string",
                  [](absl::string_view str) {
                    absl::StatusOr<std::unique_ptr<HloModule>> hlo_module =
                        ParseAndReturnUnverifiedModule(
                            str, DefaultHloModuleConfigWithDebugOptions());

                    if (!hlo_module.ok()) {
                      throw std::runtime_error(
                          std::string(hlo_module.status().message()));
                    }

                    return std::move(hlo_module).value();
                  })
      .def("add_entry_computation",
           [](HloModule* self, std::unique_ptr<HloComputation> computation) {
             self->AddEntryComputation(std::move(computation));
           })
      .def("add_computation",
           [](HloModule* self, std::unique_ptr<HloComputation> computation) {
             self->AddEmbeddedComputation(std::move(computation));
           })
      .def("set_schedule",
           [](HloModule& self, HloSchedule schedule) {
             absl::Status status = self.set_schedule(std::move(schedule));
             if (!status.ok()) {
               throw std::runtime_error(std::string(status.message()));
             }
           })
      .def(
          "get_root_instruction",
          [](HloModule* self) {
            return self->entry_computation()->root_instruction();
          },
          nb::rv_policy::reference_internal)
      .def("get_config", &HloModule::config)
      .def("__str__", nb::overload_cast<>(&HloModule::ToString, nb::const_));
}

}  // namespace xla
