/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This header file defines common utils used by TFLite transformation
// passes to work with op attributes.

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_UTILS_ATTRIBUTE_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_UTILS_ATTRIBUTE_UTILS_H_

#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir

namespace mlir {
namespace TFL {

// Returns true if none of the three attributes are empty.
inline bool HasAll3Attrs(Attribute a, Attribute b, Attribute c) {
  return a != Attribute() && b != Attribute() && c != Attribute();
}

// Returns the single float element from an ElementsAttr. Returns empty
// attribute if the number of elements in the attribute is not 1 or the
// element isn't a float attribute.
FloatAttr ExtractSingleElementAsFloat(ElementsAttr attr);

// Returns the single float element if the input is an ElementsAttr, or return
// itself as a float element. Returns empty attribute if the number of elements
// in the attribute is not 1, the element or itself isn't a float attribute.
FloatAttr GetSingleElementAsFloatOrSelf(Attribute attr);

// Returns the single integer element from an ElementsAttr. Returns empty
// attribute if the number of elements in the attribute is not 1 or the
// element isn't a integer attribute.
IntegerAttr ExtractSingleElementAsInteger(ElementsAttr attr);

}  // end namespace TFL
}  // end namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_UTILS_ATTRIBUTE_UTILS_H_
