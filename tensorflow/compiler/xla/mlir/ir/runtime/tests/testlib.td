/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_TESTLIB
#define XLA_TESTLIB

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// XLA Runtime Testlib dialect definitions.
//===----------------------------------------------------------------------===//

def TestlibDialect : Dialect {
  let name = "testlib";

  let description = [{
    XLA Runtime Testlib dialect for writing tests for the runtime features.
  }];

  let cppNamespace = "::xla::runtime";

  let useDefaultAttributePrinterParser = 1;

  let useDefaultTypePrinterParser = 1;

  let emitAccessorPrefix = kEmitAccessorPrefix_Both;
}

include "testlib_attrs.td"
include "testlib_enums.td"
include "testlib_types.td"

#endif // XLA_TESTLIB
