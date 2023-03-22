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

#ifndef PYTHON_TEST_ATTRS
#define PYTHON_TEST_ATTRS

include "mlir/IR/OpBase.td"

def PythonTestAttrsDialect : Dialect {
  let name = "python_test_attrs";
  let hasRegionArgAttrVerify = 1;
  let cppNamespace = "::mlir::tfrt";
  let description = [{
    The python test attributes dialect is intended to verify attributes which
    are used to annotate function arguments so that the python test knows which
    values it needs to pass to a function which expects dynamic shapes.
  }];

  let extraClassDeclaration = [{
    static std::string GetStaticTypeAttrName() {
      return (getDialectNamespace() + ".static_type").str();
    }
    static std::string GetShapeValueAttrName() {
      return (getDialectNamespace() + ".shape_value").str();
    }
  }];
}

#endif  // PYTHON_TEST_ATTRS
