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

#include "tensorflow/compiler/tf2xla/lib/util.h"

#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

xla::XlaOp Zeros(xla::XlaBuilder* builder, const xla::Shape& shape) {
  return xla::Broadcast(
      xla::ConstantLiteral(builder,
                           xla::LiteralUtil::Zero(shape.element_type())),
      xla::AsInt64Slice(shape.dimensions()));
}

xla::XlaOp FloatLiteral(xla::XlaBuilder* builder, xla::PrimitiveType type,
                        double value) {
  switch (type) {
    case xla::F16:
      return xla::ConstantR0<xla::half>(builder, static_cast<xla::half>(value));
      break;
    case xla::BF16:
      return xla::ConstantR0<bfloat16>(builder, static_cast<bfloat16>(value));
      break;
    case xla::F32:
      return xla::ConstantR0<float>(builder, static_cast<float>(value));
      break;
    case xla::F64:
      return xla::ConstantR0<double>(builder, value);
      break;
    case xla::C64:
      return xla::ConstantR0<xla::complex64>(builder, value);
      break;
    default:
      LOG(FATAL) << "unhandled element type " << type;
  }
}

xla::XlaOp IntegerLiteral(xla::XlaBuilder* builder, xla::PrimitiveType type,
                          int64 value) {
  xla::Literal literal;
  switch (type) {
    case xla::U8:
      literal = xla::LiteralUtil::CreateR0<uint8>(value);
      break;
    case xla::U32:
      literal = xla::LiteralUtil::CreateR0<uint32>(value);
      break;
    case xla::U64:
      literal = xla::LiteralUtil::CreateR0<uint64>(value);
      break;
    case xla::S8:
      literal = xla::LiteralUtil::CreateR0<int8>(value);
      break;
    case xla::S32:
      literal = xla::LiteralUtil::CreateR0<int32>(value);
      break;
    case xla::S64:
      literal = xla::LiteralUtil::CreateR0<int64>(value);
      break;
    case xla::F32:
      literal = xla::LiteralUtil::CreateR0<float>(value);
      break;
    case xla::F64:
      literal = xla::LiteralUtil::CreateR0<double>(value);
      break;
    case xla::C64:
      literal = xla::LiteralUtil::CreateR0<complex64>(value);
      break;
    case xla::PRED:
      LOG(FATAL) << "pred element type is not integral";
    case xla::S16:
    case xla::U16:
      LOG(FATAL) << "u16/s16 literals not yet implemented";
    case xla::BF16:
      literal =
          xla::LiteralUtil::CreateR0<bfloat16>(static_cast<bfloat16>(value));
      break;
    case xla::F16:
      literal =
          xla::LiteralUtil::CreateR0<xla::half>(static_cast<xla::half>(value));
      break;
    case xla::TUPLE:
      LOG(FATAL) << "tuple element type is not integral";
    case xla::OPAQUE:
      LOG(FATAL) << "opaque element type is not integral";
    default:
      LOG(FATAL) << "unhandled element type " << type;
  }
  return xla::ConstantLiteral(builder, literal);
}

xla::XlaOp SliceInMinorDims(xla::XlaOp x, absl::Span<const int64> start,
                            absl::Span<const int64> end) {
  xla::XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> xla::StatusOr<xla::XlaOp> {
    TF_RET_CHECK(start.size() == end.size());
    int64 n_minor_dims = start.size();

    TF_ASSIGN_OR_RETURN(xla::Shape shape, builder->GetShape(x));

    const int64 n_dims = xla::ShapeUtil::Rank(shape);
    TF_RET_CHECK(n_minor_dims <= n_dims);
    auto major_dims = xla::AsInt64Slice(shape.dimensions())
                          .subspan(
                              /*pos=*/0,
                              /*len=*/n_dims - n_minor_dims);

    // Prepends 0s in the major dim
    std::vector<int64> padded_start(n_dims, 0);
    std::copy(start.begin(), start.end(),
              padded_start.begin() + major_dims.size());

    // Prepends the shape of the major dims.
    std::vector<int64> padded_end(n_dims);
    std::copy(major_dims.begin(), major_dims.end(), padded_end.begin());
    std::copy(end.begin(), end.end(), padded_end.begin() + major_dims.size());

    std::vector<int64> strides(n_dims, 1);
    return xla::Slice(x, padded_start, padded_end, strides);
  });
}

std::vector<int64> ConcatVectors(absl::Span<const int64> xs,
                                 absl::Span<const int64> ys) {
  std::vector<int64> output(xs.size() + ys.size());
  std::copy(xs.begin(), xs.end(), output.begin());
  std::copy(ys.begin(), ys.end(), output.begin() + xs.size());
  return output;
}

xla::XlaOp DynamicSliceInMinorDims(xla::XlaOp x,
                                   absl::Span<const xla::XlaOp> starts,
                                   absl::Span<const int64> sizes) {
  xla::XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> xla::StatusOr<xla::XlaOp> {
    TF_ASSIGN_OR_RETURN(xla::Shape shape, builder->GetShape(x));
    const int64 n_dims = xla::ShapeUtil::Rank(shape);
    int64 n_minor_dims = starts.size();
    TF_RET_CHECK(n_minor_dims == sizes.size());
    TF_RET_CHECK(n_minor_dims <= n_dims);
    auto major_dims = xla::AsInt64Slice(shape.dimensions())
                          .subspan(
                              /*pos=*/0,
                              /*len=*/n_dims - sizes.size());
    auto padded_starts = PrependZerosInMajorDims(x, starts);
    auto padded_sizes = ConcatVectors(major_dims, sizes);
    return xla::DynamicSlice(x, padded_starts, padded_sizes);
  });
}

xla::XlaOp UpdateSlice(xla::XlaOp x, xla::XlaOp update,
                       absl::Span<const int64> start) {
  xla::XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> xla::StatusOr<xla::XlaOp> {
    // TODO(phawkins): make int64 work on all backends, remove the int32 cast.
    std::vector<int32> start_as_int32(start.begin(), start.end());
    auto start_constant = xla::ConstantR1<int32>(builder, start_as_int32);
    TF_ASSIGN_OR_RETURN(xla::Shape shape, builder->GetShape(x));
    const int64 n_dims = xla::ShapeUtil::Rank(shape);
    TF_ASSIGN_OR_RETURN(xla::Shape start_constant_shape,
                        builder->GetShape(start_constant));
    const int64 start_length =
        xla::ShapeUtil::GetDimension(start_constant_shape, -1);
    TF_RET_CHECK(start_length == n_dims);
    return xla::DynamicUpdateSlice(x, update, start_constant);
  });
}

xla::XlaOp UpdateSliceInMinorDims(xla::XlaOp x, xla::XlaOp update,
                                  absl::Span<const int64> start) {
  xla::XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> xla::StatusOr<xla::XlaOp> {
    TF_ASSIGN_OR_RETURN(xla::Shape shape, builder->GetShape(x));
    const int64 n_dims = xla::ShapeUtil::Rank(shape);
    const int64 n_minor_dims = start.size();
    TF_RET_CHECK(n_minor_dims <= n_dims);
    std::vector<int64> padded_start(n_dims, 0);
    std::copy(start.begin(), start.end(),
              padded_start.begin() + (n_dims - n_minor_dims));
    return UpdateSlice(x, update, padded_start);
  });
}

xla::XlaOp DynamicUpdateSliceInMinorDims(xla::XlaOp x, xla::XlaOp update,
                                         absl::Span<const xla::XlaOp> starts) {
  auto padded_starts = PrependZerosInMajorDims(x, starts);
  return xla::DynamicUpdateSlice(x, update, padded_starts);
}

xla::XlaOp PrependZerosInMajorDims(xla::XlaOp x,
                                   absl::Span<const xla::XlaOp> starts) {
  xla::XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> xla::StatusOr<xla::XlaOp> {
    TF_ASSIGN_OR_RETURN(xla::Shape shape, builder->GetShape(x));
    const int64 n_dims = xla::ShapeUtil::Rank(shape);
    auto zero = xla::Reshape(xla::ConstantR0<int32>(builder, 0), {1});
    std::vector<xla::XlaOp> padded_starts(n_dims, zero);
    for (int i = 0; i < starts.size(); ++i) {
      padded_starts[n_dims - starts.size() + i] = xla::Reshape(starts[i], {1});
    }
    return xla::ConcatInDim(builder, padded_starts, 0);
  });
}

xla::XlaOp TransposeInMinorDims(xla::XlaOp x) {
  xla::XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> xla::StatusOr<xla::XlaOp> {
    TF_ASSIGN_OR_RETURN(xla::Shape shape, builder->GetShape(x));
    const int64 n_dims = xla::ShapeUtil::Rank(shape);
    TF_RET_CHECK(n_dims >= 2);
    std::vector<int64> permutation(n_dims);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::swap(permutation[n_dims - 1], permutation[n_dims - 2]);
    return xla::Transpose(x, permutation);
  });
}

xla::XlaOp MaybeTransposeInMinorDims(xla::XlaOp x, bool transpose) {
  return transpose ? TransposeInMinorDims(x) : x;
}

xla::XlaOp MaybeConjugate(xla::XlaOp x, bool conjugate) {
  xla::XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> xla::StatusOr<xla::XlaOp> {
    TF_ASSIGN_OR_RETURN(xla::Shape shape, builder->GetShape(x));
    auto perform_conj = shape.element_type() == xla::C64 && conjugate;
    return perform_conj ? xla::Conj(x) : x;
  });
}

}  // namespace tensorflow
