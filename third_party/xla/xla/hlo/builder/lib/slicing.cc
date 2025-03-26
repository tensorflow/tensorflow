/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/hlo/builder/lib/slicing.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/builder/lib/arithmetic.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {

XlaOp SliceInMinorDims(XlaOp x, absl::Span<const int64_t> start,
                       absl::Span<const int64_t> end) {
  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> absl::StatusOr<XlaOp> {
    TF_RET_CHECK(start.size() == end.size());
    int64_t n_minor_dims = start.size();

    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));

    const int64_t n_dims = shape.dimensions_size();
    TF_RET_CHECK(n_minor_dims <= n_dims);
    auto major_dims = shape.dimensions().subspan(
        /*pos=*/0,
        /*len=*/n_dims - n_minor_dims);

    // Prepends 0s in the major dim
    std::vector<int64_t> padded_start(n_dims, 0);
    std::copy(start.begin(), start.end(),
              padded_start.begin() + major_dims.size());

    // Prepends the shape of the major dims.
    std::vector<int64_t> padded_end(n_dims);
    std::copy(major_dims.begin(), major_dims.end(), padded_end.begin());
    std::copy(end.begin(), end.end(), padded_end.begin() + major_dims.size());

    std::vector<int64_t> strides(n_dims, 1);
    return Slice(x, padded_start, padded_end, strides);
  });
}

XlaOp UpdateSlice(XlaOp x, XlaOp update, absl::Span<const int64_t> start) {
  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> absl::StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));
    const int64_t n_dims = shape.dimensions_size();
    const int64_t start_size = start.size();
    TF_RET_CHECK(start_size == n_dims);

    // TODO(phawkins): make int64_t work on all backends, remove the int32_t
    // cast.
    std::vector<int32_t> start_as_int32(start.begin(), start.end());
    std::vector<XlaOp> start_ops(start.size());
    for (int i = 0, end = start.size(); i < end; ++i) {
      start_ops[i] = ConstantR0(builder, start_as_int32[i]);
    }
    return DynamicUpdateSlice(x, update, start_ops);
  });
}

XlaOp UpdateSliceInMinorDims(XlaOp x, XlaOp update,
                             absl::Span<const int64_t> start) {
  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> absl::StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));
    const int64_t n_dims = shape.dimensions_size();
    const int64_t n_minor_dims = start.size();
    TF_RET_CHECK(n_minor_dims <= n_dims);
    std::vector<int64_t> padded_start(n_dims, 0);
    std::copy(start.begin(), start.end(),
              padded_start.begin() + (n_dims - n_minor_dims));
    return UpdateSlice(x, update, padded_start);
  });
}

namespace {

std::vector<int64_t> ConcatVectors(absl::Span<const int64_t> xs,
                                   absl::Span<const int64_t> ys) {
  std::vector<int64_t> output(xs.size() + ys.size());
  std::copy(xs.begin(), xs.end(), output.begin());
  std::copy(ys.begin(), ys.end(), output.begin() + xs.size());
  return output;
}

absl::StatusOr<std::vector<XlaOp>> PrependZerosInMajorDims(
    XlaOp x, absl::Span<const XlaOp> starts) {
  XlaBuilder* builder = x.builder();
  TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));
  const int64_t n_dims = shape.dimensions_size();
  auto zero = ConstantR0<int32_t>(builder, 0);
  std::vector<XlaOp> padded_starts(n_dims, zero);
  for (int i = 0; i < starts.size(); ++i) {
    padded_starts[n_dims - starts.size() + i] = starts[i];
  }
  return padded_starts;
}

}  // namespace

XlaOp DynamicSliceInMinorDims(XlaOp x, absl::Span<const XlaOp> starts,
                              absl::Span<const int64_t> sizes) {
  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> absl::StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));
    const int64_t n_dims = shape.dimensions_size();
    int64_t n_minor_dims = starts.size();
    TF_RET_CHECK(n_minor_dims == sizes.size());
    TF_RET_CHECK(n_minor_dims <= n_dims);
    auto major_dims = shape.dimensions().subspan(
        /*pos=*/0,
        /*len=*/n_dims - sizes.size());
    TF_ASSIGN_OR_RETURN(auto padded_starts, PrependZerosInMajorDims(x, starts));
    auto padded_sizes = ConcatVectors(major_dims, sizes);
    return DynamicSlice(x, padded_starts, padded_sizes);
  });
}

XlaOp DynamicUpdateSliceInMinorDims(XlaOp x, XlaOp update,
                                    absl::Span<const XlaOp> starts) {
  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> absl::StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(auto padded_starts, PrependZerosInMajorDims(x, starts));
    return DynamicUpdateSlice(x, update, padded_starts);
  });
}

XlaOp TorchGather(XlaOp input, XlaOp index, int64_t dim, bool sparse) {
  XlaBuilder* builder = input.builder();
  return builder->ReportErrorOrReturn([&]() -> absl::StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape index_shape, builder->GetShape(index));
    TF_ASSIGN_OR_RETURN(Shape input_shape, builder->GetShape(input));
    if (ShapeUtil::ElementHasBitWidth(index_shape, 64) &&
        input_shape.dimensions(dim) < std::numeric_limits<uint32_t>::max()) {
      index = ConvertElementType(index, U32);
      index_shape.set_element_type(U32);
    }
    if (index_shape.dimensions_size() == 1) {
      return TorchIndexSelect(input, index, 0);
    }
    if (!sparse) {
      std::vector<int64_t> index_broadcast_dims;
      std::vector<int64_t> input_broadcast_dims;
      std::vector<int64_t> sizes;
      sizes.reserve(index_shape.dimensions_size());
      for (int64_t i = 0; i < index_shape.dimensions_size(); ++i) {
        if (i < dim) {
          input_broadcast_dims.push_back(i);
          index_broadcast_dims.push_back(i);
        } else if (i == dim) {
          sizes.push_back(input_shape.dimensions(i));
          input_broadcast_dims.push_back(i);
          index_broadcast_dims.push_back(i + 1);
        } else {
          input_broadcast_dims.push_back(i + 1);
          index_broadcast_dims.push_back(i + 1);
        }
        sizes.push_back(index_shape.dimensions(i));
      }
      auto mask = Eq(
          BroadcastInDim(index, sizes, index_broadcast_dims),
          Iota(builder, ShapeUtil::MakeShape(index_shape.element_type(), sizes),
               dim));
      auto masked_input = Select(
          mask, BroadcastInDim(input, sizes, input_broadcast_dims),
          Zeros(builder,
                ShapeUtil::MakeShape(input_shape.element_type(), sizes)));
      return Reduce(masked_input, Zero(builder, input_shape.element_type()),
                    CreateScalarIdentityWithZeroComputation(
                        input_shape.element_type(), builder),
                    {dim});
    }

    ShapeUtil::AppendMajorDimension(1, &index_shape);
    std::vector<XlaOp> to_concat;

    to_concat.reserve(input_shape.dimensions_size());
    for (int64_t i = 0; i < input_shape.dimensions_size(); ++i) {
      if (i == dim) {
        to_concat.push_back(Reshape(index, index_shape.dimensions()));
      } else {
        to_concat.push_back(Iota(builder, index_shape, i));
      }
    }
    XlaOp gather_indices =
        ConcatInDim(builder, to_concat, input_shape.dimensions_size());
    std::vector<int64_t> slice_sizes(input_shape.dimensions_size(), 1);
    GatherDimensionNumbers gather_dnums;
    gather_dnums.set_index_vector_dim(input_shape.dimensions_size());
    for (int64_t i = 0; i < input_shape.dimensions_size(); ++i) {
      gather_dnums.add_collapsed_slice_dims(i);
      gather_dnums.add_start_index_map(i);
    }
    return Gather(input, gather_indices, gather_dnums, slice_sizes);
  });
}

XlaOp TorchScatterDense(XlaOp input, XlaOp index, XlaOp src, int64_t dim,
                        const std::function<XlaOp(XlaOp, XlaOp)>& combiner) {
  XlaBuilder* builder = input.builder();
  return builder->ReportErrorOrReturn([&]() -> absl::StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape index_shape, builder->GetShape(index));
    TF_ASSIGN_OR_RETURN(Shape input_shape, builder->GetShape(input));
    std::vector<int64_t> index_broadcast_dims;
    std::vector<int64_t> sizes;
    const auto rank = index_shape.dimensions_size();
    sizes.reserve(rank + 1);
    for (int64_t i = 0; i < index_shape.dimensions_size(); ++i) {
      if (i < dim) {
        index_broadcast_dims.push_back(i);
      } else {
        if (i == dim) {
          sizes.push_back(input_shape.dimensions(i));
        }
        index_broadcast_dims.push_back(i + 1);
      }
      sizes.push_back(index_shape.dimensions(i));
    }
    auto mask =
        Eq(BroadcastInDim(index, sizes, index_broadcast_dims),
           Iota(builder,
                ShapeUtil::MakeShape(index_shape.element_type(), sizes), dim));
    auto masked_src =
        Select(mask, BroadcastInDim(src, sizes, index_broadcast_dims),
               Zeros(builder,
                     ShapeUtil::MakeShape(input_shape.element_type(), sizes)));

    return combiner(
        input,
        Reduce(masked_src, Zero(builder, input_shape.element_type()),
               CreateScalarComputation("reducer", input_shape.element_type(),
                                       builder, combiner),
               {dim + 1}));
  });
}

XlaOp TorchIndexSelect(XlaOp input, XlaOp index, int64_t dim,
                       int64_t batch_dims) {
  XlaBuilder* builder = input.builder();
  return builder->ReportErrorOrReturn([&]() -> absl::StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape input_shape, builder->GetShape(input));
    TF_ASSIGN_OR_RETURN(Shape index_shape, builder->GetShape(index));
    if (dim < batch_dims) {
      return InvalidArgument(
          "Gather dim must be greater than or equal to the number of batch "
          "dims");
    }
    if (ShapeUtil::ElementHasBitWidth(index_shape, 64) &&
        input_shape.dimensions(dim) < std::numeric_limits<uint32_t>::max()) {
      index = ConvertElementType(index, U32);
      index_shape.set_element_type(U32);
    }
    std::vector<int64_t> slice_sizes = SpanToVector(input_shape.dimensions());
    GatherDimensionNumbers gather_dnums;
    gather_dnums.set_index_vector_dim(index_shape.dimensions_size());
    if (batch_dims > 0) {
      ShapeUtil::AppendMajorDimension(1, &index_shape);
      std::vector<XlaOp> to_concat;
      to_concat.reserve(batch_dims + 1);
      xla::Shape iota_shape = xla::ShapeUtil::MakeStaticShape(index_shape);
      for (int64_t batch_dim = 0; batch_dim < batch_dims; ++batch_dim) {
        to_concat.push_back(Iota(builder, iota_shape, batch_dim));
      }
      to_concat.push_back(Reshape(index, index_shape.dimensions()));
      index = ConcatInDim(builder, to_concat, gather_dnums.index_vector_dim());
    }
    for (int64_t i = 0; i < input_shape.dimensions_size(); ++i) {
      if (i < batch_dims || i == dim) {
        slice_sizes[i] = std::min<int64_t>(slice_sizes[i], 1);
        gather_dnums.add_collapsed_slice_dims(i);
        gather_dnums.add_start_index_map(i);
      } else {
        if (i < dim) {
          gather_dnums.add_offset_dims(i);
        } else {
          gather_dnums.add_offset_dims(i + gather_dnums.index_vector_dim() -
                                       (1 + batch_dims));
        }
      }
    }
    return Gather(input, index, gather_dnums, slice_sizes);
  });
}

}  // namespace xla
