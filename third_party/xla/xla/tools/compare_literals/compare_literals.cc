/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/tools/compare_literals/compare_literals.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/shape_util.h"
#include "xla/tools/compare_literals/element_comparator.h"
#include "xla/tsl/platform/env.h"
#include "xla/xla_data.pb.h"

namespace xla::compare_literals {
namespace {

template <typename NativeT>
ComparisonResult CompareArrayValues(const LiteralSlice& clean,
                                    const LiteralSlice& dirty,
                                    const ComparisonOptions& options) {
  ElementComparator<NativeT> comparator(options,
                                        ShapeUtil::ElementsIn(clean.shape()));

  if (LayoutUtil::Equal(dirty.shape().layout(), clean.shape().layout()) &&
      clean.shape().layout().element_size_in_bits() == 0 &&
      clean.shape().is_static() && dirty.shape().is_static()) {
    absl::Span<const NativeT> clean_span = clean.data<NativeT>();
    absl::Span<const NativeT> dirty_span = dirty.data<NativeT>();
    const int64_t num_elements = clean_span.size();
    for (int64_t i = 0; i < num_elements; ++i) {
      comparator.RecordElement(i, clean_span[i], dirty_span[i]);
    }
  } else {
    std::vector<int64_t> multi_index(clean.shape().dimensions_size(), 0);
    const int64_t num_elements = ShapeUtil::ElementsIn(clean.shape());
    for (int64_t i = 0; i < num_elements; ++i) {
      comparator.RecordElement(i, clean.Get<NativeT>(multi_index),
                               dirty.Get<NativeT>(multi_index));
      for (int d = static_cast<int>(multi_index.size()) - 1; d >= 0; --d) {
        if (++multi_index[d] < clean.shape().dimensions(d)) {
          break;
        }
        multi_index[d] = 0;
      }
    }
  }

  return comparator.Finalize();
}

}  // namespace

absl::StatusOr<ComparisonResult> CompareLiterals(
    const LiteralSlice& clean, const LiteralSlice& dirty,
    const ComparisonOptions& options) {
  if (!ShapeUtil::Compatible(clean.shape(), dirty.shape())) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Shapes must be equal; clean: %s, dirty: %s",
                        ShapeUtil::HumanString(clean.shape()),
                        ShapeUtil::HumanString(dirty.shape())));
  }

  if (!clean.shape().IsArray()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Only array literals are supported; got: ",
                     ShapeUtil::HumanString(clean.shape())));
  }

  if (!primitive_util::IsArrayType(clean.shape().element_type())) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported element type for literal comparison: ",
                     primitive_util::LowercasePrimitiveTypeName(
                         clean.shape().element_type())));
  }

  ComparisonResult result = primitive_util::ArrayTypeSwitch(
      [&](auto type_constant) -> ComparisonResult {
        using NativeT = primitive_util::NativeTypeOf<type_constant>;
        return CompareArrayValues<NativeT>(clean, dirty, options);
      },
      clean.shape().element_type());

  result.element_type =
      primitive_util::LowercasePrimitiveTypeName(clean.shape().element_type());
  result.shape_str = ShapeUtil::HumanString(clean.shape());
  return result;
}

absl::StatusOr<ComparisonResult> CompareLiteralProtos(
    const LiteralProto& clean_proto, const LiteralProto& dirty_proto,
    const ComparisonOptions& options) {
  ABSL_ASSIGN_OR_RETURN(Literal clean, Literal::CreateFromProto(clean_proto));
  ABSL_ASSIGN_OR_RETURN(Literal dirty, Literal::CreateFromProto(dirty_proto));
  return CompareLiterals(clean, dirty, options);
}

absl::StatusOr<ComparisonResult> CompareLiteralFiles(
    absl::string_view clean_file, absl::string_view dirty_file,
    const ComparisonOptions& options) {
  LiteralProto clean_proto;
  ABSL_RETURN_IF_ERROR(
      tsl::ReadBinaryProto(tsl::Env::Default(), clean_file, &clean_proto))
      << absl::StrCat("Failed to read clean literal file '", clean_file, "'");

  LiteralProto dirty_proto;
  ABSL_RETURN_IF_ERROR(
      tsl::ReadBinaryProto(tsl::Env::Default(), dirty_file, &dirty_proto))
      << absl::StrCat("Failed to read dirty literal file '", dirty_file, "'");

  return CompareLiteralProtos(clean_proto, dirty_proto, options);
}

}  // namespace xla::compare_literals
