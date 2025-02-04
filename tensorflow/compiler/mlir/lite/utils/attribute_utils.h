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

#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/DialectResourceBlobManager.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"  // from @llvm-project

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

// Returns the values of the given ElementsAttr as a ArrayRef of ElementType.
// Returns {std::nullopt} if the attribute is empty.
// TODO(b/707702324): Add support for type like IntergAttr and APInt, etc. This
// is a temporary solution to unblock the LiteRT. Ideally MLIR should provide
// a common API to access the values of an DenseResourceElementsAttr.
template <typename ElementType>
inline llvm::ArrayRef<ElementType> GetValues(ElementsAttr attr) {
  if (auto dense_elements_attr = dyn_cast<DenseElementsAttr>(attr)) {
    auto raw_data = dense_elements_attr.getRawData();
    if (raw_data.empty()) {
      return {};
    }
    return llvm::ArrayRef<ElementType>(
        reinterpret_cast<const ElementType *>(raw_data.data()),
        raw_data.size() / sizeof(ElementType));
  } else if (auto dense_resource_elements_attr =
                 dyn_cast<DenseResourceElementsAttr>(attr)) {
    if (AsmResourceBlob *blob =
            dense_resource_elements_attr.getRawHandle().getBlob())
      return blob->getDataAs<ElementType>();
    return {};
  }
  return {};
}

}  // end namespace TFL
}  // end namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_UTILS_ATTRIBUTE_UTILS_H_
