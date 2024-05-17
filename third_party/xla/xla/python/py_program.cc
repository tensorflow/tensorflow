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

#include "xla/python/py_program.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "third_party/nanobind/include/nanobind/nanobind.h"
#include "third_party/nanobind/include/nanobind/stl/string.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/unique_ptr.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/vector.h"  // IWYU pragma: keep
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt/plugin_program.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "tsl/platform/statusor.h"

namespace xla {

namespace nb = ::nanobind;

namespace {

absl::StatusOr<std::unique_ptr<xla::ifrt::Program>> MakePluginProgramFromString(
    std::string data) {
  auto plugin_program = std::make_unique<xla::ifrt::PluginProgram>();
  plugin_program->data = std::move(data);
  return plugin_program;
}

absl::StatusOr<std::unique_ptr<xla::ifrt::Program>> MakePluginProgramFromBytes(
    nb::bytes data) {
  auto plugin_program = std::make_unique<xla::ifrt::PluginProgram>();
  plugin_program->data = std::string(data.c_str(), data.size());
  return plugin_program;
}

absl::StatusOr<std::unique_ptr<ifrt::CompileOptions>>
MakePluginCompileOptions() {
  return std::make_unique<ifrt::PluginCompileOptions>();
}

absl::StatusOr<std::unique_ptr<ifrt::Program>> MakeHloProgram(
    absl::string_view mlir_module) {
  auto context = std::make_unique<mlir::MLIRContext>();
  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                      ParseMlirModuleString(mlir_module, *context));
  return std::make_unique<xla::ifrt::HloProgram>(std::move(context),
                                                 std::move(module));
}

absl::StatusOr<std::unique_ptr<ifrt::Program>> MakeHloProgramFromString(
    std::string mlir_module) {
  return MakeHloProgram(mlir_module);
}

absl::StatusOr<std::unique_ptr<ifrt::Program>> MakeHloProgramFromBytes(
    nb::bytes mlir_module) {
  return MakeHloProgram(
      absl::string_view(mlir_module.c_str(), mlir_module.size()));
}

absl::StatusOr<std::unique_ptr<ifrt::CompileOptions>> MakeXlaCompileOptions(
    CompileOptions options, std::vector<nb::capsule> host_callbacks) {
  std::vector<tsl::RCReference<ifrt::LoadedHostCallback>>
      ifrt_loaded_host_callbacks;
  ifrt_loaded_host_callbacks.reserve(host_callbacks.size());
  // Extract `ifrt::LoadedHostCallback`s from host callback capsules that were
  // created by `PyClient::MakePythonCallbackUsingHostSendAndRecv()` or
  // `PyClient::GetEmitPythonCallbackDescriptor()`.
  for (auto& host_callback : host_callbacks) {
    ifrt_loaded_host_callbacks.push_back(tsl::FormRef(
        static_cast<ifrt::LoadedHostCallback*>(host_callback.data())));
  }
  return std::make_unique<ifrt::XlaCompileOptions>(
      std::move(options), std::move(ifrt_loaded_host_callbacks));
}

}  // namespace

void BuildIfrtProgramsSubmodule(nanobind::module_& m) {
  auto sub_module = m.def_submodule("ifrt_programs");
  nb::class_<xla::ifrt::Program> ifrt_program_base_class(sub_module, "Program");
  nb::class_<xla::ifrt::CompileOptions> ifrt_compile_options_base_class(
      sub_module, "CompileOptions");
  sub_module
      .def("make_hlo_program",
           xla::ValueOrThrowWrapper(MakeHloProgramFromString))
      .def("make_hlo_program",
           xla::ValueOrThrowWrapper(MakeHloProgramFromBytes))
      .def("make_plugin_program",
           xla::ValueOrThrowWrapper(MakePluginProgramFromString))
      .def("make_plugin_program",
           xla::ValueOrThrowWrapper(MakePluginProgramFromBytes))
      .def("make_xla_compile_options",
           xla::ValueOrThrowWrapper(MakeXlaCompileOptions))
      .def("make_plugin_compile_options",
           xla::ValueOrThrowWrapper(MakePluginCompileOptions));
}

}  // namespace xla
