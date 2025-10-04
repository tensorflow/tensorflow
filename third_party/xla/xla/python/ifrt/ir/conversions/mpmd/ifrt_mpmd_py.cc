/* Copyright 2025 The OpenXLA Authors.

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

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"  // IWYU pragma: keep; Needed to allow MlirModule -> ModuleOp.
#include "mlir/CAPI/IR.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "nanobind/nanobind.h"
#include "pybind11/detail/common.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11_abseil/absl_casters.h"
#include "xla/pjrt/status_casters.h"  // IWYU pragma: keep; Needed for ValueOrThrow
#include "xla/python/ifrt/ir/conversions/mpmd/lower_to_ifrt.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::ifrt::mpmd {

PYBIND11_MODULE(ifrt_mpmd_py, m) {
  m.def(
      "lower_to_ifrt",
      [](MlirModule module) -> void {
        return xla::ThrowIfError(LowerToIfrt(unwrap(module)));
      },
      py::arg("module"));

  m.def("get_compile_options",
        [](MlirModule c_module,
           const absl::flat_hash_map<std::string, const EnvOptionsOverride>&
               compile_options_overrides) -> absl::StatusOr<py::dict> {
          auto module = unwrap(c_module);
          TF_ASSIGN_OR_RETURN(
              auto compile_options_map,
              GetCompileOptions(module, compile_options_overrides));
          py::dict out;
          for (const auto& [name, options] : compile_options_map) {
            out[py::cast(name)] = py::reinterpret_steal<py::object>(
                nanobind::cast(options).release().ptr());
          }
          return out;
        });
}

}  // namespace xla::ifrt::mpmd
