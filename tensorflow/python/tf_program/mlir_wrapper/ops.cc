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

#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "mlir/IR/Function.h"
#include "mlir/IR/Operation.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/python/tf_program/mlir_wrapper/mlir_util.h"

void init_ops(py::module& m) {
  py::class_<mlir::Operation, std::unique_ptr<mlir::Operation, py::nodelete>>(
      m, "Operation")
      .def("getRegion", &mlir::Operation::getRegion,
           py::return_value_policy::reference)
      .def("getResult", &mlir::Operation::getResult)
      .def("dump", &mlir::Operation::dump)
      .def("getNumResults", &mlir::Operation::getNumResults);

  py::class_<mlir::OperationState>(m, "OperationState")
      .def(py::init([](mlir::Location loc, std::string name) {
        return mlir::OperationState(loc, llvm::StringRef(name));
      }))
      .def("addTypes",
           [](mlir::OperationState& state, std::vector<mlir::Type> tys) {
             state.addTypes(mlir::ArrayRef<mlir::Type>(tys));
           })
      .def("addOperands",
           [](mlir::OperationState& os, std::vector<mlir::Value> ops) {
             os.addOperands(mlir::ArrayRef<mlir::Value>(ops));
           })
      .def("addRegion", py::overload_cast<>(&mlir::OperationState::addRegion),
           py::return_value_policy::reference);

  py::class_<mlir::ModuleOp>(m, "ModuleOp")
      .def("create",
           [](mlir::Location loc) { return mlir::ModuleOp::create(loc); })
      .def("push_back",
           [](mlir::ModuleOp& m, mlir::FuncOp f) { m.push_back(f); })
      .def("dump", &mlir::ModuleOp::dump)
      .def("getAsStr", [](mlir::ModuleOp& m) {
        std::string str;
        llvm::raw_string_ostream os(str);
        m.print(os);
        return os.str();
      });

  py::class_<mlir::FuncOp>(m, "FuncOp")
      .def("create",
           [](mlir::Location location, std::string name,
              mlir::FunctionType type) {
             auto func = mlir::FuncOp::create(location, name, type);
             func.addEntryBlock();
             return func;
           })
      .def(
          "getBody",
          [](mlir::FuncOp& f) -> mlir::Region& { return f.getBody(); },
          py::return_value_policy::reference)
      .def("getArguments",
           [](mlir::FuncOp& f) { return f.getArguments().vec(); })
      .def("getName", [](mlir::FuncOp& f) { return f.getName().str(); })
      .def("getType", &mlir::FuncOp::getType);

  py::class_<mlir::ReturnOp>(m, "ReturnOp")
      .def("create",
           [](mlir::OpBuilder& opb, mlir::Location loc,
              std::vector<mlir::Value> values) -> mlir::Operation* {
             return opb
                 .create<mlir::ReturnOp>(loc,
                                         mlir::ArrayRef<mlir::Value>(values))
                 .getOperation();
           });

  // mlir::TF::AddOp
  py::class_<mlir::TF::AddV2Op>(m, "Tf_AddV2Op")
      .def("create",
           [](mlir::OpBuilder& opb, mlir::Location loc, mlir::Value x,
              mlir::Value y) -> mlir::Operation* {
             return opb.create<mlir::TF::AddV2Op>(loc, x, y).getOperation();
           });

  py::class_<mlir::TF::AnyOp>(m, "Tf_AnyOp")
      .def("create",
           [](mlir::OpBuilder& opb, mlir::Location loc, mlir::Value input,
              mlir::Value reduction_indices,
              bool keep_dims = false) -> mlir::Operation* {
             return opb
                 .create<mlir::TF::AnyOp>(loc, opb.getI1Type(), input,
                                          reduction_indices, keep_dims)
                 .getOperation();
           });

  // mlir::TF::ConstOp
  py::class_<mlir::TF::ConstOp>(m, "Tf_ConstOp")
      .def("create",
           [](mlir::OpBuilder& opb, mlir::Location loc,
              mlir::Attribute value) -> mlir::Operation* {
             return opb.create<mlir::TF::ConstOp>(loc, value).getOperation();
           });

  // mlir::TF::EqualOp
  py::class_<mlir::TF::EqualOp>(m, "Tf_EqualOp")
      .def("create",
           [](mlir::OpBuilder& opb, mlir::Location loc, mlir::Value x,
              mlir::Value y) -> mlir::Operation* {
             return opb
                 .create<mlir::TF::EqualOp>(loc, x, y, opb.getBoolAttr(true))
                 .getOperation();
           });

  // mlir::TF::GreaterEqualOp
  py::class_<mlir::TF::GreaterEqualOp>(m, "Tf_GreaterEqualOp")
      .def("create",
           [](mlir::OpBuilder& opb, mlir::Location loc, mlir::Value x,
              mlir::Value y) -> mlir::Operation* {
             return opb.create<mlir::TF::GreaterEqualOp>(loc, x, y)
                 .getOperation();
           });

  // mlir::TF::GreaterOp
  py::class_<mlir::TF::GreaterOp>(m, "Tf_GreaterOp")
      .def("create",
           [](mlir::OpBuilder& opb, mlir::Location loc, mlir::Value x,
              mlir::Value y) -> mlir::Operation* {
             return opb.create<mlir::TF::GreaterOp>(loc, x, y).getOperation();
           });

  // mlir::TF::LegacyCallOp
  py::class_<mlir::TF::LegacyCallOp>(m, "Tf_LegacyCallOp")
      .def("create",
           [](mlir::OpBuilder& opb, mlir::Location loc,
              std::vector<mlir::Type> output, std::vector<mlir::Value> args,
              std::string f) -> mlir::Operation* {
             return opb
                 .create<mlir::TF::LegacyCallOp>(
                     loc, mlir::ArrayRef<mlir::Type>(output),
                     mlir::ArrayRef<mlir::Value>(args), mlir::StringRef(f))
                 .getOperation();
           });

  // mlir::TF::LessEqualOp
  py::class_<mlir::TF::LessEqualOp>(m, "Tf_LessEqualOp")
      .def("create",
           [](mlir::OpBuilder& opb, mlir::Location loc, mlir::Value x,
              mlir::Value y) -> mlir::Operation* {
             return opb.create<mlir::TF::LessEqualOp>(loc, x, y).getOperation();
           });

  // mlir::TF::LessOp
  py::class_<mlir::TF::LessOp>(m, "Tf_LessOp")
      .def("create",
           [](mlir::OpBuilder& opb, mlir::Location loc, mlir::Value x,
              mlir::Value y) -> mlir::Operation* {
             return opb.create<mlir::TF::LessOp>(loc, x, y).getOperation();
           });

  // mlir::TF::NegOp
  py::class_<mlir::TF::NegOp>(m, "Tf_NegOp")
      .def("create",
           [](mlir::OpBuilder& opb, mlir::Location loc,
              mlir::Value x) -> mlir::Operation* {
             return opb.create<mlir::TF::NegOp>(loc, x).getOperation();
           });

  py::class_<mlir::TF::NotEqualOp>(m, "Tf_NotEqualOp")
      .def("create", [](mlir::OpBuilder& opb, mlir::Location loc, mlir::Value x,
                        mlir::Value y) {
        return opb
            .create<mlir::TF::NotEqualOp>(
                loc, x, y, mlir::BoolAttr::get(true, opb.getContext()))
            .getOperation();
      });

  // mlir::TF::SubOp
  py::class_<mlir::TF::SubOp>(m, "Tf_SubOp")
      .def("create",
           [](mlir::OpBuilder& opb, mlir::Location loc, mlir::Value x,
              mlir::Value y) -> mlir::Operation* {
             return opb.create<mlir::TF::SubOp>(loc, x, y).getOperation();
           });
}
