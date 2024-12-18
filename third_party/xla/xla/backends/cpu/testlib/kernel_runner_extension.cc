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
#include "nanobind/nanobind.h"
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "nanobind/stl/tuple.h"  // IWYU pragma: keep
#include "nanobind/stl/unique_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "xla/backends/cpu/testlib/elemental_kernel_emitter.h"
#include "xla/backends/cpu/testlib/kernel_runner.h"
#include "xla/backends/cpu/testlib/llvm_ir_kernel_emitter.h"
#include "xla/backends/cpu/testlib/llvm_ir_kernel_spec.h"
#include "xla/codegen/kernel_emitter.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/testlib/kernel_runner.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/stream_executor/launch_dim.h"

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

  nb::class_<LlvmIrKernelSpec, KernelSpec> give_me_a_name(kernel_runner_module,
                                                          "LlvmIrKernelSpec");

  // Use a tuple and cast to ThreadDim to take advantage of built in bindings.
  using NbThreadDim = std::tuple<uint64_t, uint64_t, uint64_t>;
  nb::class_<LlvmIrKernelEmitter, KernelEmitter>(kernel_runner_module,
                                                 "LlvmIrKernelEmitter")
      .def("__init__", [](LlvmIrKernelEmitter* self, absl::string_view ir,
                          absl::string_view kernel_name,
                          NbThreadDim thread_dim) {
        new (self) LlvmIrKernelEmitter(
            ir, kernel_name,
            se::ThreadDim{std::get<0>(thread_dim), std::get<1>(thread_dim),
                          std::get<2>(thread_dim)},
            {});
      });

  nb::class_<ElementalKernelEmitter, KernelEmitter>(kernel_runner_module,
                                                    "ElementalKernelEmitter")
      .def(nb::init<std::unique_ptr<HloInstruction>>());

  nb::class_<KernelRunner, xla::KernelRunner>(kernel_runner_module,
                                              "KernelRunner")
      .def_static("create", [](std::unique_ptr<LlvmIrKernelSpec> kernel_spec) {
        absl::StatusOr<KernelRunner> runner =
            KernelRunner::Create(std::move(*kernel_spec));

        if (!runner.ok()) {
          throw std::runtime_error(std::string(runner.status().message()));
        }

        return std::move(runner).value();
      });
}

}  // namespace xla::cpu
