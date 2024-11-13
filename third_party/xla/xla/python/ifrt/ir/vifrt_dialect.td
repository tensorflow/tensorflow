/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_IR_VIFRT_DIALECT_TD_
#define XLA_PYTHON_IFRT_IR_VIFRT_DIALECT_TD_

def Vifrt_Dialect : Dialect {
  let name = "vifrt";
  let summary = "Versioned IFRT dialect";
  let cppNamespace = "::xla::ifrt";

  let description = [{
    A versioned copy of the IFRT IR dialect that is used for forward and
    backward compatible serialization/deserialization.

    Version log:
      0.1.0: Initial IFRT IR stability guarantees.
  }];

  let useDefaultAttributePrinterParser = 0;
  let useDefaultTypePrinterParser = 0;
  let usePropertiesForAttributes = 1;
}

#endif  // XLA_PYTHON_IFRT_IR_VIFRT_DIALECT_TD_