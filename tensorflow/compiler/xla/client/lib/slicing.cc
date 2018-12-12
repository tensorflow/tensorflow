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

#include "tensorflow/compiler/xla/client/lib/slicing.h"

namespace xla {

XlaOp SliceInMinorDims(XlaOp x, absl::Span<const int64> start,
                       absl::Span<const int64> end) {
  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_RET_CHECK(start.size() == end.size());
    int64 n_minor_dims = start.size();

    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));

    const int64 n_dims = ShapeUtil::Rank(shape);
    TF_RET_CHECK(n_minor_dims <= n_dims);
    auto major_dims = AsInt64Slice(shape.dimensions())
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
    return Slice(x, padded_start, padded_end, strides);
  });
}

XlaOp UpdateSlice(XlaOp x, XlaOp update, absl::Span<const int64> start) {
  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    // TODO(phawkins): make int64 work on all backends, remove the int32 cast.
    std::vector<int32> start_as_int32(start.begin(), start.end());
    auto start_constant = ConstantR1<int32>(builder, start_as_int32);
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));
    const int64 n_dims = ShapeUtil::Rank(shape);
    TF_ASSIGN_OR_RETURN(Shape start_constant_shape,
                        builder->GetShape(start_constant));
    const int64 start_length =
        ShapeUtil::GetDimension(start_constant_shape, -1);
    TF_RET_CHECK(start_length == n_dims);
    return DynamicUpdateSlice(x, update, start_constant);
  });
}

XlaOp UpdateSliceInMinorDims(XlaOp x, XlaOp update,
                             absl::Span<const int64> start) {
  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));
    const int64 n_dims = ShapeUtil::Rank(shape);
    const int64 n_minor_dims = start.size();
    TF_RET_CHECK(n_minor_dims <= n_dims);
    std::vector<int64> padded_start(n_dims, 0);
    std::copy(start.begin(), start.end(),
              padded_start.begin() + (n_dims - n_minor_dims));
    return UpdateSlice(x, update, padded_start);
  });
}

namespace {

std::vector<int64> ConcatVectors(absl::Span<const int64> xs,
                                 absl::Span<const int64> ys) {
  std::vector<int64> output(xs.size() + ys.size());
  std::copy(xs.begin(), xs.end(), output.begin());
  std::copy(ys.begin(), ys.end(), output.begin() + xs.size());
  return output;
}

XlaOp PrependZerosInMajorDims(XlaOp x, absl::Span<const XlaOp> starts) {
  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));
    const int64 n_dims = ShapeUtil::Rank(shape);
    auto zero = Reshape(ConstantR0<int32>(builder, 0), {1});
    std::vector<XlaOp> padded_starts(n_dims, zero);
    for (int i = 0; i < starts.size(); ++i) {
      padded_starts[n_dims - starts.size() + i] = Reshape(starts[i], {1});
    }
    return ConcatInDim(builder, padded_starts, 0);
  });
}

}  // namespace

XlaOp DynamicSliceInMinorDims(XlaOp x, absl::Span<const XlaOp> starts,
                              absl::Span<const int64> sizes) {
  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));
    const int64 n_dims = ShapeUtil::Rank(shape);
    int64 n_minor_dims = starts.size();
    TF_RET_CHECK(n_minor_dims == sizes.size());
    TF_RET_CHECK(n_minor_dims <= n_dims);
    auto major_dims = AsInt64Slice(shape.dimensions())
                          .subspan(
                              /*pos=*/0,
                              /*len=*/n_dims - sizes.size());
    auto padded_starts = PrependZerosInMajorDims(x, starts);
    auto padded_sizes = ConcatVectors(major_dims, sizes);
    return DynamicSlice(x, padded_starts, padded_sizes);
  });
}

XlaOp DynamicUpdateSliceInMinorDims(XlaOp x, XlaOp update,
                                    absl::Span<const XlaOp> starts) {
  auto padded_starts = PrependZerosInMajorDims(x, starts);
  return DynamicUpdateSlice(x, update, padded_starts);
}

}  // namespace xla
