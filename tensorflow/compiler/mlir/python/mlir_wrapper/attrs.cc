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

#include <cstdint>

#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/python/mlir_wrapper/mlir_wrapper.h"

void init_attrs(py::module& m) {
  py::class_<mlir::Attribute>(m, "Attribute");
  py::class_<mlir::IntegerAttr, mlir::Attribute>(m, "IntegerAttr")
      .def("get",
           py::overload_cast<mlir::Type, int64_t>(&mlir::IntegerAttr::get));
}
