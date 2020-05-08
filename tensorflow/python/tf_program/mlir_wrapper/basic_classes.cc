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

#include "llvm/Support/FileCheck.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "tensorflow/python/tf_program/mlir_wrapper/mlir_util.h"

void init_basic_classes(py::module& m) {
  py::class_<mlir::MLIRContext>(m, "MLIRContext").def(py::init<>());

  py::class_<mlir::Location>(m, "Location");

  py::class_<mlir::UnknownLoc>(m, "UnknownLoc")
      .def("get", &mlir::UnknownLoc::get);

  py::class_<mlir::Region>(m, "Region")
      .def("back", &mlir::Region::back, py::return_value_policy::reference)
      .def("front", &mlir::Region::front, py::return_value_policy::reference)
      .def("add_block", [](mlir::Region& r) { r.push_back(new mlir::Block); })
      .def("push_back", &mlir::Region::push_back)
      .def("size", [](mlir::Region& r) { return r.getBlocks().size(); })
      .def("front", &mlir::Region::front, py::return_value_policy::reference);
  py::class_<mlir::Block::iterator>(m, "Block_Iterator");
  py::class_<mlir::Block>(m, "Block")
      .def("new", ([]() { return new mlir::Block; }),
           py::return_value_policy::reference)
      .def("end", &mlir::Block::end)
      .def("addArgument", &mlir::Block::addArgument);

  py::class_<mlir::Value>(m, "Value").def("getType", &mlir::Value::getType);
  py::class_<mlir::OpResult, mlir::Value>(m, "OpResult");
  py::class_<mlir::BlockArgument, mlir::Value>(m, "BlockArgument");
}
