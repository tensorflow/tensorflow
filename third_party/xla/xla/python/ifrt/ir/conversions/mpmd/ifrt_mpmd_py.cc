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
#include "mlir/Bindings/Python/NanobindAdaptors.h"  // IWYU pragma: keep; Needed to allow MlirModule -> ModuleOp.
#include "mlir/CAPI/IR.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/pair.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/variant.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "xla/pjrt/status_casters.h"  // IWYU pragma: keep; Needed for ValueOrThrow
#include "xla/python/ifrt/ir/conversions/mpmd/lower_to_ifrt.h"
#include "xla/python/nb_absl_flat_hash_map.h"  // IWYU pragma: keep

namespace nb = nanobind;

namespace xla::ifrt::mpmd {

NB_MODULE(ifrt_mpmd_py, m) {
  m.def(
      "lower_to_ifrt",
      [](MlirModule module) -> void {
        return xla::ThrowIfError(LowerToIfrt(unwrap(module)));
      },
      nb::arg("module"));

  m.def("get_compile_options",
        [](MlirModule c_module,
           const absl::flat_hash_map<std::string, const EnvOptionsOverride>&
               compile_options_overrides) -> absl::StatusOr<nb::dict> {
          auto module = unwrap(c_module);
          auto compile_options_map = ValueOrThrow(
              GetCompileOptions(module, compile_options_overrides));
          nb::dict out;
          for (const auto& [name, options] : compile_options_map) {
            out[nb::cast(name)] =
                nb::steal<nb::object>(nanobind::cast(options).release().ptr());
          }
          return out;
        });
}

}  // namespace xla::ifrt::mpmd
