/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <numeric>
#include <vector>

#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/numeric.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace xla {

namespace {

template <typename T>
XlaOp MakeIota(XlaBuilder* builder, int64 size) {
  std::vector<T> values(size);
  for (int64 i = 0; i < size; ++i) {
    values[i] = static_cast<T>(i);
  }
  return ConstantR1<T>(builder, values);
}

}  // namespace

XlaOp Iota(XlaBuilder* builder, PrimitiveType type, int64 size) {
  switch (type) {
    case S8:
      return MakeIota<int8>(builder, size);
    case S16:
      return MakeIota<int16>(builder, size);
    case S32:
      return MakeIota<int32>(builder, size);
    case S64:
      return MakeIota<int64>(builder, size);
    case U8:
      return MakeIota<uint8>(builder, size);
    case U16:
      return MakeIota<uint16>(builder, size);
    case U32:
      return MakeIota<uint32>(builder, size);
    case U64:
      return MakeIota<uint64>(builder, size);
    case BF16:
      return MakeIota<bfloat16>(builder, size);
    case F16:
      return MakeIota<half>(builder, size);
    case F32:
      return MakeIota<float>(builder, size);
    case F64:
      return MakeIota<double>(builder, size);
    case C64:
      return MakeIota<complex64>(builder, size);
    default:
      return builder->ReportError(
          InvalidArgument("Unimplemented type for Iota: %s.",
                          PrimitiveType_Name(type).c_str()));
  }
}

XlaOp IdentityMatrix(XlaBuilder* builder, PrimitiveType type, int64 m,
                     int64 n) {
  auto a = Iota(builder, type, m);
  auto b = Iota(builder, type, n);
  auto indicator = Eq(a, Broadcast(b, {m}), /*broadcast_dimensions=*/{0});
  return ConvertElementType(indicator, type);
}

XlaOp Diagonal(XlaOp x) {
  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));
    const int64 n_dims = ShapeUtil::Rank(shape);
    TF_RET_CHECK(n_dims >= 2);
    const int64 n = shape.dimensions(n_dims - 1);
    const int64 m = shape.dimensions(n_dims - 2);
    tensorflow::gtl::ArraySlice<int64> major_dims(
        AsInt64Slice(shape.dimensions()), /*pos=*/0, /*len=*/n_dims - 2);
    auto a = Iota(builder, U32, n);
    auto b = Iota(builder, U32, m);
    auto indicator = Eq(a, Broadcast(b, {n}), /*broadcast_dimensions=*/{0});
    auto mask = Broadcast(indicator, major_dims);
    XlaComputation add =
        CreateScalarAddComputation(shape.element_type(), builder);
    auto diag = Reduce(Select(mask, x, Zeros(builder, shape)), ScalarLike(x, 0),
                       add, {n_dims - 1});
    return diag;
  });
}

}  // namespace xla
