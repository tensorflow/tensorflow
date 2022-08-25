/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The StableHLO Authors.
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

#include "integrations/c/ChloAttributes.h"
#include "integrations/c/ChloDialect.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace py = pybind11;

namespace {

auto toPyString(MlirStringRef mlirStringRef) {
  return py::str(mlirStringRef.data, mlirStringRef.length);
}

}  // namespace

PYBIND11_MODULE(_chlo, m) {
  m.doc() = "chlo main python extension";

  //
  // Dialects.
  //

  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle dialect = mlirGetDialectHandle__chlo__();
        mlirDialectHandleRegisterDialect(dialect, context);
        if (load) {
          mlirDialectHandleLoadDialect(dialect, context);
        }
      },
      py::arg("context"), py::arg("load") = true);

  //
  // Attributes.
  //

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "ComparisonDirectionAttr", chloAttributeIsAComparisonDirectionAttr)
      .def_classmethod(
          "get",
          [](py::object cls, const std::string &direction, MlirContext ctx) {
            return cls(chloComparisonDirectionAttrGet(
                ctx, mlirStringRefCreate(direction.c_str(), direction.size())));
          },
          py::arg("cls"), py::arg("comparison_direction"),
          py::arg("context") = py::none(),
          "Creates a ComparisonDirection attribute with the given direction.")
      .def_property_readonly("comparison_direction", [](MlirAttribute self) {
        return toPyString(chloComparisonDirectionAttrGetDirection(self));
      });

  mlir::python::adaptors::mlir_attribute_subclass(
      m, "ComparisonTypeAttr", chloAttributeIsAComparisonTypeAttr)
      .def_classmethod(
          "get",
          [](py::object cls, const std::string &type, MlirContext ctx) {
            return cls(chloComparisonTypeAttrGet(
                ctx, mlirStringRefCreate(type.c_str(), type.size())));
          },
          py::arg("cls"), py::arg("comparison_type"),
          py::arg("context") = py::none(),
          "Creates a ComparisonType attribute with the given type.")
      .def_property_readonly("comparison_type", [](MlirAttribute self) {
        return toPyString(chloComparisonTypeAttrGetType(self));
      });
}
