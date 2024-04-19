/* Copyright 2022 The OpenXLA Authors.

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

//===- rt_interfaces.td ---------------------------------------------------===//
//
// Attribute and type interfaces for the `rt` dialect.
//
//===----------------------------------------------------------------------===//

#ifdef RT_INTERFACES
#else
#define RT_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Attribute interfaces
//===----------------------------------------------------------------------===//

class RT_AttrInterface<string name> : AttrInterface<name> {
  let cppNamespace = "xla::runtime";
}

def RT_TraceAnnotationAttrInterface
  : RT_AttrInterface<"TraceAnnotationAttrInterface"> {
  let description = [{
    This is a marker for attributes that can be passed to the XLA runtime
    tracing custom calls (see `rt.trace` operation).
  }];
}

#endif // RT_INTERFACES
