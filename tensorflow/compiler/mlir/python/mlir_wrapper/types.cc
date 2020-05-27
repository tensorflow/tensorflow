/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/python/mlir_wrapper/mlir_wrapper.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

void init_types(py::module& m) {
  // Type
  py::class_<mlir::Type> Type(m, "Type");
  Type.def("getKind", &mlir::Type::getKind);

  // Type Enums
  py::enum_<mlir::StandardTypes::Kind>(Type, "StandardTypes_Kind")
      .value("BF16", mlir::StandardTypes::BF16);

  // Type Sub-classes
  py::class_<mlir::FunctionType, mlir::Type>(m, "FunctionType")
      .def("getResults",
           [](mlir::FunctionType& ft) { return ft.getResults().vec(); });

  py::class_<mlir::FloatType, mlir::Type>(m, "FloatType")
      .def("get", &mlir::FloatType::get);

  py::class_<mlir::IntegerType, mlir::Type>(m, "IntegerType")
      .def("get", py::overload_cast<unsigned, mlir::MLIRContext*>(
                      &mlir::IntegerType::get));

  py::class_<mlir::UnrankedTensorType, mlir::Type>(m, "UnrankedTensorType")
      .def("get", &mlir::UnrankedTensorType::get);

  py::class_<mlir::RankedTensorType, mlir::Type>(m, "RankedTensorType")
      .def("get", [](std::vector<int64_t> shape, mlir::Type ty) {
        return mlir::RankedTensorType::get(mlir::ArrayRef<int64_t>(shape), ty);
      });
}
