/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/Registration.h"
#include "mlir-hlo-c/Dialects.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace py = pybind11;

PYBIND11_MODULE(_mlirHlo, m) {
  m.doc() = "mlir-hlo main python extension";

  m.def(
      "register_mhlo_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle mhloDialect = mlirGetDialectHandle__mhlo__();
        mlirDialectHandleRegisterDialect(mhloDialect, context);
        if (load) {
          mlirDialectHandleLoadDialect(mhloDialect, context);
        }
      },
      py::arg("context"), py::arg("load") = true);

  m.def(
      "register_chlo_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle chloDialect = mlirGetDialectHandle__chlo__();
        mlirDialectHandleRegisterDialect(chloDialect, context);
        if (load) {
          mlirDialectHandleLoadDialect(chloDialect, context);
        }
      },
      py::arg("context"), py::arg("load") = true);
}
