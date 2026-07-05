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

#include <sys/stat.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/DialectResourceBlobManager.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/IR/Types.h"  // from @llvm-project
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

/// Check the information for a C++ data type, check if this type is valid for
/// the current attribute. This method is used to verify specific type
/// invariants that the templatized 'getValues' method cannot.
bool IsValidIntOrFloat(Type type, int64_t data_elt_size, bool is_int,
                       bool is_signed);

/// Type trait used to check if the given type T is a potentially valid C++
/// floating point type that can be used to access the underlying element
/// types of a DenseElementsAttr.
template <typename T>
struct is_valid_cpp_fp_type {
  /// The type is a valid floating point type if it is a builtin floating
  /// point type, or is a potentially user defined floating point type. The
  /// latter allows for supporting users that have custom types defined for
  /// bfloat16/half/etc.
  static constexpr bool value = llvm::is_one_of<T, float, double>::value ||
                                (std::numeric_limits<T>::is_specialized &&
                                 !std::numeric_limits<T>::is_integer);
};

/// Return the bit width which ElementsAttr should use for this type.
size_t GetDenseElementBitWidth(Type elt_type);

// Returns the values of the given ElementsAttr as a ArrayRef of ElementType.
// Returns {std::nullopt} if the attribute is empty.
// TODO(b/707702324): Add support for type like IntergAttr and APInt, etc. This
// is a temporary solution to unblock the LiteRT. Ideally MLIR should provide
// a common API to access the values of an DenseResourceElementsAttr.
template <typename ElementType>
using IntFloatValueTemplateCheckT =
    std::enable_if_t<(!std::is_same<ElementType, bool>::value &&
                      std::numeric_limits<ElementType>::is_integer) ||
                     is_valid_cpp_fp_type<ElementType>::value>;
template <typename ElementType,
          typename = IntFloatValueTemplateCheckT<ElementType>>
inline ArrayRef<ElementType> GetValues(ElementsAttr attr) {
  Type element_type = attr.getElementType();

  // Check if the element type is not valid for the given ElementType, return
  // empty ArrayRef.
  if (!IsValidIntOrFloat(element_type, sizeof(ElementType),
                         std::numeric_limits<ElementType>::is_integer,
                         std::numeric_limits<ElementType>::is_signed)) {
    assert(false && "Incompatible dtype expected from the given ElementsAttr");
  }

  if (auto dense_elements_attr = dyn_cast<DenseElementsAttr>(attr)) {
    auto raw_data = dense_elements_attr.getRawData();
    if (raw_data.empty()) {
      return {};
    }
    return llvm::ArrayRef<ElementType>(
        reinterpret_cast<const ElementType*>(raw_data.data()),
        raw_data.size() / sizeof(ElementType));
  } else if (auto dense_resource_elements_attr =
                 dyn_cast<DenseResourceElementsAttr>(attr)) {
    if (AsmResourceBlob* blob =
            dense_resource_elements_attr.getRawHandle().getBlob())
      return blob->getDataAs<ElementType>();
    return {};
  }
  return {};
}

}  // end namespace TFL
}  // end namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_UTILS_ATTRIBUTE_UTILS_H_
