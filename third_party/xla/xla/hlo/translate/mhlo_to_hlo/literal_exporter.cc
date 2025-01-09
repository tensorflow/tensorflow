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

#include "xla/hlo/translate/mhlo_to_hlo/literal_exporter.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/APInt.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "xla/array.h"
#include "xla/hlo/translate/mhlo_to_hlo/type_to_shape.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"

namespace mlir {
namespace mhlo {

template <typename T>
xla::Array<T> ArrayFromDenseElementsAttr(mlir::DenseElementsAttr dense_attr) {
  constexpr xla::PrimitiveType type =
      xla::primitive_util::NativeToPrimitiveType<T>();
  xla::Shape shape = xla::TypeToShape(dense_attr.getType());
  xla::Array<T> array(shape.dimensions());
  if constexpr (!xla::primitive_util::IsSubByteNonPredType(type)) {
    array.SetValues(dense_attr.getValues<T>());
  } else {
    // The only way to get subbyte integers from getValues() is to get them as
    // APInts.
    auto values = dense_attr.getValues<llvm::APInt>();
    for (int i = 0; i < values.size(); i++) {
      if constexpr (xla::primitive_util::IsUnsignedIntegralType(type)) {
        array.data()[i] = T{values[i].getZExtValue()};
      } else {
        static_assert(xla::primitive_util::IsSignedIntegralType(type));
        array.data()[i] = T{values[i].getSExtValue()};
      }
    }
  }
  return array;
}

absl::StatusOr<xla::Literal> CreateLiteralFromAttribute(mlir::ElementsAttr attr,
                                                        xla::Layout layout) {
  auto dense_attr = mlir::dyn_cast<mlir::DenseElementsAttr>(attr);
  if (!dense_attr)
    return absl::UnimplementedError("Only dense elements attr are supported");

  xla::Shape shape = xla::TypeToShape(dense_attr.getType());

  return xla::primitive_util::PrimitiveTypeSwitch<absl::StatusOr<xla::Literal>>(
      [&](auto primitive_type_constant) -> absl::StatusOr<xla::Literal> {
        if constexpr (xla::primitive_util::IsArrayType(
                          primitive_type_constant)) {
          using cpp_type =
              xla::primitive_util::NativeTypeOf<primitive_type_constant>;
          xla::Array<cpp_type> source_data =
              ArrayFromDenseElementsAttr<cpp_type>(dense_attr);
          if (layout.minor_to_major().empty()) {
            return xla::LiteralUtil::CreateFromArray(source_data);
          }
          return xla::LiteralUtil::CreateFromArrayWithLayout(source_data,
                                                             layout);
        }
        return absl::InternalError(absl::StrCat(  // NOLINT
            "Unsupported type: ",
            xla::PrimitiveType_Name(shape.element_type())));
      },
      shape.element_type());
}

}  // namespace mhlo
}  // namespace mlir
