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

#include <algorithm>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/translate/mhlo_to_hlo/type_to_shape.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace mlir {
namespace mhlo {
namespace {

// Writes the elements of `dense_attr`, in row major order, into `dest`, which
// holds exactly one slot per element. A splat fills every slot with its one
// value, so the order of `dest` does not matter for it.
template <typename T>
absl::Status CopyDenseElementsAttr(DenseElementsAttr dense_attr,
                                   absl::Span<T> dest) {
  constexpr xla::PrimitiveType type =
      xla::primitive_util::NativeToPrimitiveType<T>();
  TF_RET_CHECK(dense_attr.getNumElements() ==
               static_cast<int64_t>(dest.size()));
  // Handle all splats with a single shared std::fill.
  if (dense_attr.isSplat()) {
    T splat_value = [&]() -> T {
      if constexpr (!xla::primitive_util::IsSubByteNonPredType(type)) {
        return dense_attr.getSplatValue<T>();
      } else if constexpr (xla::primitive_util::IsMXType(type)) {
        return T::FromRep(dense_attr.getSplatValue<llvm::APFloat>()
                              .bitcastToAPInt()
                              .getZExtValue());
      } else {
        // DenseElementsAttr stores a sub byte element in its own byte with the
        // value in the low bits (see getFromRawBuffer); the T constructor
        // keeps exactly those bits.
        return T(dense_attr.getRawData().front());
      }
    }();
    std::fill(dest.begin(), dest.end(), splat_value);
    return absl::OkStatus();
  }

  // Handle non splat attributes.
  if constexpr (!xla::primitive_util::IsSubByteNonPredType(type)) {
    auto values = dense_attr.getValues<T>();
    std::copy(values.begin(), values.end(), dest.begin());
  } else if constexpr (xla::primitive_util::IsMXType(type)) {
    // Bitcast MX floating point types from APFloat.
    auto values = dense_attr.getValues<llvm::APFloat>();
    std::transform(values.begin(), values.end(), dest.begin(),
                   [](const llvm::APFloat& value) {
                     return T::FromRep(value.bitcastToAPInt().getZExtValue());
                   });
  } else {
    // One byte per element, the value in the low bits.
    static_assert(xla::is_intN_v<T>);
    llvm::ArrayRef<char> raw = dense_attr.getRawData();
    TF_RET_CHECK(raw.size() == dest.size())
        << "Unexpected raw storage size " << raw.size() << " for "
        << dest.size() << " elements of " << xla::PrimitiveType_Name(type);
    std::transform(raw.begin(), raw.end(), dest.begin(),
                   [](char byte) { return T(byte); });
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<xla::Literal> CreateLiteralFromAttribute(mlir::ElementsAttr attr,
                                                        xla::Layout layout) {
  auto dense_attr = mlir::dyn_cast<mlir::DenseElementsAttr>(attr);
  if (!dense_attr) {
    return absl::UnimplementedError("Only dense elements attr are supported");
  }

  xla::Shape shape = xla::TypeToShape(dense_attr.getType());

  return xla::primitive_util::PrimitiveTypeSwitch<absl::StatusOr<xla::Literal>>(
      [&](auto primitive_type_constant) -> absl::StatusOr<xla::Literal> {
        if constexpr (xla::primitive_util::IsArrayType(
                          primitive_type_constant)) {
          using cpp_type =
              xla::primitive_util::NativeTypeOf<primitive_type_constant>;
          // The attribute lists its elements in row major order, which is the
          // element order of a literal with a descending layout.
          const xla::Shape row_major_shape =
              xla::ShapeUtil::MakeShapeWithDescendingLayout(
                  shape.element_type(), shape.dimensions());
          const xla::Shape literal_shape =
              layout.minor_to_major().empty()
                  ? row_major_shape
                  : xla::ShapeUtil::MakeShapeWithDenseLayout(
                        shape.element_type(), shape.dimensions(),
                        layout.minor_to_major());
          // If it's a splat, we can safely allocate the target layout directly.
          const xla::Shape initial_shape =
              dense_attr.isSplat() ? literal_shape : row_major_shape;
          xla::Literal initial_literal(initial_shape);
          ABSL_RETURN_IF_ERROR(CopyDenseElementsAttr<cpp_type>(
              dense_attr, initial_literal.data<cpp_type>()));
          if (xla::ShapeUtil::Equal(literal_shape, initial_shape)) {
            return initial_literal;
          }
          return initial_literal.Relayout(literal_shape);
        }
        return absl::InternalError(absl::StrCat(  // NOLINT
            "Unsupported type: ",
            xla::PrimitiveType_Name(shape.element_type())));
      },
      shape.element_type());
}

}  // namespace mhlo
}  // namespace mlir
