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

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "nanobind/stl/tuple.h"  // IWYU pragma: keep
#include "nanobind/stl/unique_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "xla/backends/cpu/codegen/computation_kernel_emitter.h"
#include "xla/backends/cpu/codegen/dot/dot_kernel_emitter.h"
#include "xla/backends/cpu/codegen/elemental/concatenate_kernel_emitter.h"
#include "xla/backends/cpu/codegen/elemental/elemental_kernel_emitter.h"
#include "xla/backends/cpu/codegen/emitters/cpu_scatter_emitter.h"
#include "xla/backends/cpu/codegen/fusion_compiler.h"
#include "xla/backends/cpu/codegen/fusion_emitter.h"
#include "xla/backends/cpu/codegen/jit_compiler.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/backends/cpu/testlib/kernel_runner.h"
#include "xla/backends/cpu/testlib/llvm_ir_kernel_emitter.h"
#include "xla/backends/cpu/testlib/mlir_kernel_emitter.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/llvm_ir_kernel_source.h"
#include "xla/codegen/llvm_kernel_definition.h"
#include "xla/codegen/llvm_kernel_emitter.h"
#include "xla/codegen/mlir_kernel_definition.h"
#include "xla/codegen/mlir_kernel_emitter.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/codegen/testlib/kernel_runner.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/runtime/work_group.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/cpu_compiler.h"
#include "xla/service/cpu/fusion_wrapper.h"
#include "xla/service/hlo_module_config.h"

namespace xla::cpu {

namespace nb = nanobind;

void ImportBaseClasses(const nb::module_& kernel_runner_module) {
  absl::string_view module_name =
      nb::borrow<nb::str>(nb::getattr(kernel_runner_module, "__name__"))
          .c_str();

  // Sequentially strip the module name until we get to the base xla module.
  absl::string_view cpu_testlib_module =
      module_name.substr(0, module_name.find_last_of('.'));
  absl::string_view cpu_module =
      cpu_testlib_module.substr(0, cpu_testlib_module.find_last_of('.'));
  absl::string_view backends_module =
      cpu_module.substr(0, cpu_module.find_last_of('.'));
  absl::string_view xla_module =
      backends_module.substr(0, backends_module.find_last_of('.'));

  nb::module_::import_(absl::StrCat(xla_module, ".codegen.testlib").c_str());
}

NB_MODULE(_extension, kernel_runner_module) {
  // We depend on the base classes so must import them before python tries to
  // register the derived versions.
  ImportBaseClasses(kernel_runner_module);

  // Use a tuple and cast to NumWorkGroups to take advantage of built in
  // bindings.
  using NbNumWorkGroups = std::tuple<uint64_t, uint64_t, uint64_t>;
  nb::class_<LlvmTestKernelEmitter, LlvmKernelEmitter>(kernel_runner_module,
                                                       "LlvmTestKernelEmitter")
      .def("__init__", [](LlvmKernelEmitter* self, absl::string_view ir,
                          absl::string_view kernel_name,
                          NbNumWorkGroups num_workgroups) {
        new (self)
            LlvmTestKernelEmitter(ir, kernel_name,
                                  NumWorkGroups{std::get<0>(num_workgroups),
                                                std::get<1>(num_workgroups),
                                                std::get<2>(num_workgroups)},
                                  {});
      });

  nb::class_<MlirTestKernelEmitter, MlirKernelEmitter>(kernel_runner_module,
                                                       "MlirTestKernelEmitter")
      .def("__init__", [](MlirKernelEmitter* self, absl::string_view ir,
                          absl::string_view kernel_name,
                          NbNumWorkGroups num_workgroups) {
        new (self)
            MlirTestKernelEmitter(ir, kernel_name,
                                  NumWorkGroups{std::get<0>(num_workgroups),
                                                std::get<1>(num_workgroups),
                                                std::get<2>(num_workgroups)},
                                  {});
      });

  kernel_runner_module.def("lower_to_llvm", [](MlirKernelSource& source) {
    absl::StatusOr<LlvmIrKernelSource> llvm_ir_kernel_source =
        LowerToLlvm(source);

    if (!llvm_ir_kernel_source.ok()) {
      throw std::runtime_error(
          std::string(llvm_ir_kernel_source.status().message()));
    }

    return std::move(llvm_ir_kernel_source).value();
  });

  nb::class_<CpuCompiler>(kernel_runner_module, "HloCompiler")
      .def(nb::init<>())
      .def("create_buffer_assignment",
           [](const CpuCompiler& self, const HloModule& hlo_module) {
             absl::StatusOr<std::unique_ptr<BufferAssignment>>
                 buffer_assignment = self.CreateBufferAssignment(hlo_module);

             if (!buffer_assignment.ok()) {
               throw std::runtime_error(
                   std::string(buffer_assignment.status().message()));
             }

             return std::move(buffer_assignment).value();
           })
      .def("create_hlo_schedule", [](const CpuCompiler& self,
                                     const HloModule& hlo_module) {
        absl::StatusOr<HloSchedule> schedule =
            self.CreateHloSchedule(hlo_module);

        if (!schedule.ok()) {
          throw std::runtime_error(std::string(schedule.status().message()));
        }

        return std::move(schedule).value();
      });

  nb::class_<mlir::MLIRContext>(kernel_runner_module, "MLIRContext")
      .def(nb::new_([] { return FusionCompiler::CreateContext(); }));

  nb::class_<TargetMachineFeatures>(kernel_runner_module,
                                    "TargetMachineFeatures")
      .def("__str__", &TargetMachineFeatures::get_target_feature_string);

  nb::class_<ElementalKernelEmitter, LlvmKernelEmitter>(
      kernel_runner_module, "ElementalKernelEmitter")
      .def(nb::init<const HloInstruction*, const BufferAssignment*,
                    const TargetMachineFeatures*>(),
           nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>(),
           nb::keep_alive<1, 4>());

  nb::class_<DotKernelEmitter, LlvmKernelEmitter>(kernel_runner_module,
                                                  "DotKernelEmitter")
      .def(nb::init<const HloInstruction*, const BufferAssignment*,
                    const TargetMachineFeatures*>(),
           nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>(),
           nb::keep_alive<1, 4>());

  nb::class_<ConcatenateKernelEmitter, LlvmKernelEmitter>(
      kernel_runner_module, "ConcatenateKernelEmitter")
      .def(nb::init<const HloInstruction*, const BufferAssignment*,
                    const TargetMachineFeatures*>(),
           nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>(),
           nb::keep_alive<1, 4>());

  nb::class_<ComputationKernelEmitter, LlvmKernelEmitter>(
      kernel_runner_module, "ComputationKernelEmitter")
      .def(nb::init<const HloInstruction*, const BufferAssignment*,
                    const TargetMachineFeatures*>(),
           nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>(),
           nb::keep_alive<1, 4>());

  nb::class_<CpuScatterFusion, MlirKernelEmitter>(kernel_runner_module,
                                                  "ScatterKernelEmitter")
      .def(
          "__init__",
          [](CpuScatterFusion* self, const HloFusionInstruction* instruction,
             const BufferAssignment* bufffer_assignment) {
            new (self) CpuScatterFusion(*bufffer_assignment, instruction);
          },
          nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>());

  kernel_runner_module.def(
      "emit_fusion_kernel",
      [](mlir::MLIRContext& context, const HloFusionInstruction& fusion,
         const BufferAssignment* buffer_assignment) {
        absl::StatusOr<MlirKernelDefinition> kernel_definition =
            EmitFusionKernel(context, fusion, buffer_assignment);
        if (!kernel_definition.ok()) {
          throw std::runtime_error(kernel_definition.status().ToString());
        }
        return std::move(kernel_definition).value();
      },
      nb::keep_alive<0, 1>(), nb::keep_alive<0, 2>(), nb::keep_alive<0, 3>());

  nb::class_<JitCompiler>(kernel_runner_module, "JitCompiler")
      .def(nb::new_([](const HloModuleConfig& config) {
             absl::StatusOr<JitCompiler> compiler =
                 KernelRunner::CreateJitCompiler(config);

             if (!compiler.ok()) {
               throw std::runtime_error(
                   std::string(compiler.status().message()));
             }

             return std::make_unique<JitCompiler>(
                 JitCompiler(std::move(compiler).value()));
           }),
           nb::arg("config"))
      .def(
          "get_target_machine",
          [](JitCompiler* self) {
            return std::make_unique<TargetMachineFeatures>(
                self->target_machine());
          },
          nb::rv_policy::reference_internal);

  nb::class_<KernelRunner, xla::KernelRunner>(kernel_runner_module,
                                              "KernelRunner")
      .def_static(
          "create",
          [](std::unique_ptr<MlirKernelDefinition,
                             nb::deleter<MlirKernelDefinition>>
                 kernel_definition,
             std::unique_ptr<JitCompiler, nb::deleter<JitCompiler>>
                 jit_compiler) {
            absl::StatusOr<KernelRunner> runner = KernelRunner::Create(
                std::move(*kernel_definition), std::move(*jit_compiler));

            if (!runner.ok()) {
              throw std::runtime_error(std::string(runner.status().ToString()));
            }

            return *std::move(runner);
          })
      .def_static(
          "create", [](std::unique_ptr<LlvmKernelDefinition,
                                       nb::deleter<LlvmKernelDefinition>>
                           kernel_definition,
                       std::unique_ptr<JitCompiler, nb::deleter<JitCompiler>>
                           jit_compiler) {
            absl::StatusOr<KernelRunner> runner = KernelRunner::Create(
                std::move(*kernel_definition), std::move(*jit_compiler));

            if (!runner.ok()) {
              throw std::runtime_error(std::string(runner.status().ToString()));
            }

            return *std::move(runner);
          });

  kernel_runner_module.def(
      "run_fusion_wrapper_pass",
      [](std::unique_ptr<HloModule, nb::deleter<HloModule>> hlo_module) {
        FusionWrapper fusion_wrapper;
        absl::StatusOr<bool> result = fusion_wrapper.Run(hlo_module.get());
        if (!result.ok()) {
          throw std::runtime_error(std::string(result.status().message()));
        }

        return hlo_module->Clone();
      });
}

}  // namespace xla::cpu
