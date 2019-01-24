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

#include "tensorflow/compiler/xla/client/lib/matrix.h"

#include <array>
#include <numeric>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

XlaOp IdentityMatrix(XlaBuilder* builder, PrimitiveType type, int64 m,
                     int64 n) {
  auto a = Iota(builder, U32, m);
  auto b = Iota(builder, U32, n);
  auto indicator = Eq(a, Broadcast(b, {m}), /*broadcast_dimensions=*/{0});
  return ConvertElementType(indicator, type);
}

XlaOp GetMatrixDiagonal(XlaOp x) {
  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));
    const int64 n_dims = shape.rank();
    TF_RET_CHECK(n_dims >= 2);
    const int64 m = shape.dimensions(n_dims - 2);
    const int64 n = shape.dimensions(n_dims - 1);
    absl::Span<const int64> major_dims =
        AsInt64Slice(shape.dimensions()).subspan(/*pos=*/0, /*len=*/n_dims - 2);
    auto a = Iota(builder, U32, n);
    auto b = Iota(builder, U32, m);
    auto indicator = Eq(b, Broadcast(a, {m}), /*broadcast_dimensions=*/{0});
    auto mask = Broadcast(indicator, major_dims);

    // TPUs don't support S64 add reduction at the moment. But fortunately
    // OR-reductions work just as well for integers.
    XlaComputation reducer =
        primitive_util::IsIntegralType(shape.element_type())
            ? CreateScalarOrComputation(shape.element_type(), builder)
            : CreateScalarAddComputation(shape.element_type(), builder);

    return Reduce(Select(mask, x, Zeros(builder, shape)), ScalarLike(x, 0),
                  reducer, {m >= n ? n_dims - 2 : n_dims - 1});
  });
}

XlaOp TriangleMask(XlaOp x, int diagonal) {
  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));
    const int64 n_dims = shape.rank();
    TF_RET_CHECK(n_dims >= 2);
    const int64 m = shape.dimensions(n_dims - 2);
    const int64 n = shape.dimensions(n_dims - 1);
    absl::Span<const int64> major_dims =
        AsInt64Slice(shape.dimensions()).subspan(/*pos=*/0, /*len=*/n_dims - 2);
    auto a = Iota(builder, S32, n);
    auto b = Iota(builder, S32, m) + ConstantR0<int32>(builder, diagonal);
    XlaOp indicator;
    indicator = Ge(b, Broadcast(a, {m}), /*broadcast_dimensions=*/{0});
    return Broadcast(indicator, major_dims);
  });
}

XlaOp Triangle(XlaOp x, bool lower) {
  return lower ? Select(TriangleMask(x, 0), x, ZerosLike(x))
               : Select(TriangleMask(x, -1), ZerosLike(x), x);
}

XlaOp UpperTriangle(XlaOp x) { return Triangle(x, false); }

XlaOp LowerTriangle(XlaOp x) { return Triangle(x, true); }

Status ValidateEinsumNumericDimensions(absl::Span<const int64> x_config,
                                       absl::Span<const int64> y_config,
                                       absl::Span<const int64> output_config) {
  for (auto dim : output_config) {
    if (absl::c_linear_search(x_config, dim) ||
        absl::c_linear_search(y_config, dim)) {
      if (absl::c_count(output_config, dim) > 1) {
        return InvalidArgument("Einsum has repeated output dimension.");
      }
      continue;
    }
    return InvalidArgument(
        "Einsum has output dimension without corresponding input dimension.");
  }
  for (auto dim : x_config) {
    if (absl::c_linear_search(y_config, dim) ||
        absl::c_linear_search(output_config, dim)) {
      if (absl::c_count(x_config, dim) > 1) {
        return InvalidArgument("Einsum has repeated lhs dimension.");
      }
      continue;
    }
    return InvalidArgument(
        "Einsum has lhs dimension without corresponding rhs or output "
        "dimension.");
  }
  for (auto dim : y_config) {
    if (absl::c_linear_search(x_config, dim) ||
        absl::c_linear_search(output_config, dim)) {
      if (absl::c_count(y_config, dim) > 1) {
        return InvalidArgument("Einsum has repeated rhs dimension.");
      }
      continue;
    }
    return InvalidArgument(
        "Einsum has rhs dimension without corresponding lhs or output "
        "dimension.");
  }
  return Status::OK();
}

xla::XlaOp Einsum(xla::XlaOp x, absl::Span<const int64> x_config, xla::XlaOp y,
                  absl::Span<const int64> y_config,
                  absl::Span<const int64> output_config,
                  xla::PrecisionConfig::Precision precision) {
  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_RETURN_IF_ERROR(
        ValidateEinsumNumericDimensions(x_config, y_config, output_config));
    const int64 x_rank = x_config.size();
    const int64 y_rank = y_config.size();
    const int64 output_rank = output_config.size();
    absl::flat_hash_set<int64> x_map;
    absl::flat_hash_set<int64> y_map;
    absl::flat_hash_set<int64> output_map;

    auto find = [&](const absl::flat_hash_set<int64>& map, int64 d) {
      return map.count(d) != 0;
    };

    auto insert = [&](absl::flat_hash_set<int64>& map, char d) {
      CHECK(!find(map, d));
      map.insert(d);
    };

    for (auto d : x_config) {
      insert(x_map, d);
    }

    for (auto d : y_config) {
      insert(y_map, d);
    }

    for (auto d : output_config) {
      insert(output_map, d);
    }

    DotDimensionNumbers dnums;
    std::vector<int64> lhs_outer_dims;
    auto is_batch_dim = [&](int64 d) {
      return find(x_map, d) && find(y_map, d) && find(output_map, d);
    };
    auto is_contracting = [&](int64 d) {
      return find(x_map, d) && find(y_map, d);
    };
    auto rhs_dimension_number = [&](int64 d) {
      return absl::c_find(y_config, d) - y_config.begin();
    };
    for (int64 i = 0; i < x_rank; ++i) {
      auto dim_name = x_config[i];
      if (is_batch_dim(dim_name)) {
        dnums.add_lhs_batch_dimensions(i);
        dnums.add_rhs_batch_dimensions(rhs_dimension_number(dim_name));
      } else if (is_contracting(dim_name)) {
        dnums.add_lhs_contracting_dimensions(i);
        dnums.add_rhs_contracting_dimensions(rhs_dimension_number(dim_name));
      } else {
        lhs_outer_dims.push_back(i);
      }
    }

    std::vector<int64> rhs_outer_dims;
    for (int64 i = 0; i < y_rank; ++i) {
      auto dim_name = y_config[i];
      if (!is_batch_dim(dim_name) && !is_contracting(dim_name)) {
        rhs_outer_dims.push_back(i);
      }
    }

    auto output_dimension_number = [&](char d) {
      return absl::c_find(output_config, d) - output_config.begin();
    };

    std::vector<int64> output_dims;
    output_dims.reserve(output_rank);
    for (auto d : dnums.lhs_batch_dimensions()) {
      output_dims.push_back(output_dimension_number(x_config[d]));
    }
    for (auto d : lhs_outer_dims) {
      output_dims.push_back(output_dimension_number(x_config[d]));
    }
    for (auto d : rhs_outer_dims) {
      output_dims.push_back(output_dimension_number(y_config[d]));
    }

    std::vector<int64> transpose_dims(output_rank);
    for (int64 i = 0; i < output_rank; ++i) {
      transpose_dims[output_dims[i]] = i;
    }

    PrecisionConfig precision_proto;
    precision_proto.add_operand_precision(precision);
    precision_proto.add_operand_precision(precision);
    return Transpose(DotGeneral(x, y, dnums, &precision_proto), transpose_dims);
  });
}

XlaOp BatchDot(XlaOp x, XlaOp y, PrecisionConfig::Precision precision) {
  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape x_shape, builder->GetShape(x));
    TF_ASSIGN_OR_RETURN(Shape y_shape, builder->GetShape(y));

    // The batch dimensions must be equal and the matrix dimensions must be
    // valid.
    std::vector<int64> batch_dimension_numbers;
    const int ndims = x_shape.rank();
    batch_dimension_numbers.reserve(ndims - 2);
    for (int i = 0; i < ndims - 2; ++i) {
      batch_dimension_numbers.push_back(i);
    }
    std::vector<int64> x_config = batch_dimension_numbers;
    x_config.push_back(ndims - 2);
    x_config.push_back(ndims);
    std::vector<int64> y_config = batch_dimension_numbers;
    y_config.push_back(ndims);
    y_config.push_back(ndims - 1);
    std::vector<int64> output_config = batch_dimension_numbers;
    output_config.push_back(ndims - 2);
    output_config.push_back(ndims - 1);
    return Einsum(x, x_config, y, y_config, output_config, precision);
  });
}

StatusOr<std::array<std::vector<int64>, 3>> ParseEinsumString(
    absl::string_view einsum_config) {
  std::array<std::vector<int64>, 3> einsum_config_numeric;
  std::vector<absl::string_view> main_split =
      absl::StrSplit(einsum_config, ',');

  if (main_split.size() != 2) {
    return InvalidArgument("Expected one \",\" in einsum_config.");
  }

  auto maybe_invalid_character = [](char d) {
    if (absl::ascii_isalpha(d)) {
      return Status::OK();
    }
    if (d == '.') {
      return InvalidArgument("Unsupported \"...\" or \".\" in einsum config.");
    }
    return InvalidArgument("Unexpected character in einsum config.");
  };

  auto& x_config = einsum_config_numeric[0];
  x_config.reserve(main_split[0].size());
  for (auto d : main_split[0]) {
    TF_RETURN_IF_ERROR(maybe_invalid_character(d));
    x_config.push_back(static_cast<int64>(d));
  }
  std::vector<absl::string_view> y_output_split =
      absl::StrSplit(main_split[1], "->");
  if (y_output_split.size() != 2) {
    return InvalidArgument("Expected one \"->\" in einsum_config.");
  }
  auto& y_config = einsum_config_numeric[1];
  y_config.reserve(y_output_split[0].size());
  for (auto d : y_output_split[0]) {
    TF_RETURN_IF_ERROR(maybe_invalid_character(d));
    y_config.push_back(static_cast<int64>(d));
  }
  auto& output_config = einsum_config_numeric[2];
  output_config.reserve(y_output_split[1].size());
  for (auto d : y_output_split[1]) {
    TF_RETURN_IF_ERROR(maybe_invalid_character(d));
    output_config.push_back(static_cast<int64>(d));
  }
  return einsum_config_numeric;
}

XlaOp Einsum(XlaOp x, XlaOp y, absl::string_view einsum_config,
             PrecisionConfig::Precision precision) {
  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(auto einsum_config_numeric,
                        ParseEinsumString(einsum_config));
    return Einsum(x, einsum_config_numeric[0], y, einsum_config_numeric[1],
                  einsum_config_numeric[2], precision);
  });
}

XlaOp TransposeInMinorDims(XlaOp x) {
  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));
    const int64 n_dims = shape.rank();
    TF_RET_CHECK(n_dims >= 2);
    std::vector<int64> permutation(n_dims);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::swap(permutation[n_dims - 1], permutation[n_dims - 2]);
    return Transpose(x, permutation);
  });
}

XlaOp MaybeTransposeInMinorDims(XlaOp x, bool transpose) {
  return transpose ? TransposeInMinorDims(x) : x;
}
}  // namespace xla
