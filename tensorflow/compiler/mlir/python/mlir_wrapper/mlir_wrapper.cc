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

#include "tensorflow/compiler/mlir/python/mlir_wrapper/mlir_wrapper.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

PYBIND11_MODULE(mlir_wrapper, m) {
  m.def("registerDialects", []() {
    mlir::registerDialect<mlir::TF::TensorFlowDialect>();
    mlir::registerDialect<mlir::tf_executor::TensorFlowExecutorDialect>();
    mlir::registerDialect<mlir::StandardOpsDialect>();
  });

  init_basic_classes(m);
  init_types(m);
  init_builders(m);
  init_ops(m);
  init_attrs(m);
}
