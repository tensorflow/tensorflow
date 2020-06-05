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

#include "mlir/IR/Builders.h"  // from @llvm-project

#include "tensorflow/compiler/mlir/python/mlir_wrapper/mlir_wrapper.h"

void init_builders(py::module& m) {
  py::class_<mlir::Builder>(m, "Builder")
      .def(py::init<mlir::MLIRContext*>())
      .def("getFunctionType",
           [](mlir::Builder& b, std::vector<mlir::Type> inputs,
              std::vector<mlir::Type> outputs) {
             return b.getFunctionType(llvm::ArrayRef<mlir::Type>(inputs),
                                      llvm::ArrayRef<mlir::Type>(outputs));
           });
  py::class_<mlir::OpBuilder>(m, "OpBuilder")
      .def(py::init<mlir::MLIRContext*>())
      .def(py::init<mlir::Region&>())
      .def(py::init<mlir::Operation*>())
      .def(py::init<mlir::Block*, mlir::Block::iterator>())
      .def("getUnknownLoc", &mlir::OpBuilder::getUnknownLoc)
      .def("setInsertionPoint",
           py::overload_cast<mlir::Block*, mlir::Block::iterator>(
               &mlir::OpBuilder::setInsertionPoint))
      .def("saveInsertionPoint", &mlir::OpBuilder::saveInsertionPoint)
      .def("restoreInsertionPoint", &mlir::OpBuilder::restoreInsertionPoint)
      .def(
          "createOperation",
          [](mlir::OpBuilder& opb, mlir::OperationState& state) {
            return opb.createOperation(state);
          },
          py::return_value_policy::reference)
      .def("getContext", &mlir::OpBuilder::getContext,
           py::return_value_policy::reference);

  py::class_<mlir::OpBuilder::InsertPoint>(m, "OpBuilder_InsertionPoint")
      .def("getBlock", &mlir::OpBuilder::InsertPoint::getBlock);
}
