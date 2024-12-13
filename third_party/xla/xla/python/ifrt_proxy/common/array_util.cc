// Copyright 2023 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xla/python/ifrt_proxy/common/array_util.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt_proxy/common/array_util.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace proxy {

namespace {

std::string StridesAsStr(const ArrayMemRegion::ByteStrides& strides) {
  if (!strides.has_value()) return "strides{nullopt}";
  return absl::StrCat("strides{", absl::StrJoin(*strides, ","), "}");
}

}  // namespace

absl::StatusOr<std::vector<int64_t>> DefaultByteStrides(const DType dtype,
                                                        const Shape& shape) {
  if (!dtype.byte_size().has_value()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported data type to query byte-strides for: ",
                     dtype.DebugString()));
  }
  std::vector<int64_t> result(shape.dims().size());
  int64_t stride = *dtype.byte_size();
  for (int i = static_cast<int>(shape.dims().size()) - 1; i >= 0; --i) {
    result[i] = stride;
    stride *= shape.dims()[i];
  }
  return result;
}

absl::StatusOr<ArrayMemRegion> ArrayMemRegion::FromZerothElementPointer(
    const void* zeroth_element, const DType dtype, const Shape& shape,
    ByteStrides byte_strides) {
  if (!dtype.byte_size().has_value()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported data type to construct ArrayMemRegion: ",
                     dtype.DebugString()));
  }
  // Below, we return an error for all situations where the zeroth_element
  // is different from mem_region_start.
  void* const mem_region_start = const_cast<void*>(zeroth_element);

  if (!byte_strides.has_value() ||
      (byte_strides->empty() && shape.dims().empty())) {
    return ArrayMemRegion(mem_region_start,
                          dtype.byte_size().value() * shape.num_elements());
  }
  if (shape.num_elements() == 0) {
    return ArrayMemRegion(mem_region_start, 0);
  }
  if (shape.dims().size() != byte_strides->size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Shape has different dimensions from byte_strides: ",
                     shape.DebugString(), " vs ", StridesAsStr(byte_strides)));
  }
  // Logic based on
  // https://numpy.org/doc/stable/reference/generated/numpy.ndarray.strides.html
  //
  // So long as all strides are positive, the array's memory region begins at
  // the zeroth element, and the last element of the array is farthest off from
  // the beginning. We use the offset of the last element of the array to
  // calculate the memory region. Note that this reasoning does not apply to
  // negative strides, since the zeroth element can then be in the middle of the
  // memory region (as an example, consider shape=[10, 10] and
  // element_strides=[10,-1]).
  uint64_t last_element_byte_offset = 0;
  for (int i = 0; i < byte_strides->size(); ++i) {
    int stride = (*byte_strides)[i];
    if (shape.dims()[i] < 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("A shape dimension is negative: ", shape.DebugString()));
    } else if (shape.dims()[i] == 1) {
      // The stride shouldn't matter in this case, so continue without checking
      // validity of the given stride.
      continue;
    } else if (stride <= 0) {
      return absl::UnimplementedError(
          absl::StrCat("Negative or zero strides are not fully supported: ",
                       StridesAsStr(byte_strides)));
    } else if (stride % dtype.byte_size().value() != 0) {
      return absl::UnimplementedError(absl::StrCat(
          "byte_stride[", i, "] is not a multiple of the data-type's size: ",
          StridesAsStr(byte_strides), ", dtype=", dtype.DebugString()));
    } else {
      // `shape.dims()[i]` cannot be negative (we explicitly check for this
      // above) or zero (we return early for `shape.num_elements() == 0`).
      DCHECK_GT(shape.dims()[i], 0);
      last_element_byte_offset += (stride * (shape.dims()[i] - 1));
    }
  }
  return ArrayMemRegion(mem_region_start,
                        last_element_byte_offset + dtype.byte_size().value());
}

absl::StatusOr<ArrayMemRegion> ArrayMemRegion::FromMinimalMemRegion(
    absl::string_view mem_region, const DType dtype, const Shape& shape,
    ByteStrides byte_strides) {
  // FromZerothElementPointer() currently returns an error for any situation
  // where the zeroth_element will is not equal to the place where the minimal
  // memory region starts.
  TF_ASSIGN_OR_RETURN(
      auto result,
      FromZerothElementPointer(mem_region.data(), dtype, shape, byte_strides));

  if (result.mem_region().size() != mem_region.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Incorrect size ", result.mem_region().size(), " vs ",
                     mem_region.size(), "; is provided memory region minimal? ",
                     dtype.DebugString(), " ", shape.DebugString(), " ",
                     StridesAsStr(byte_strides)));
  }
  CHECK_EQ(result.mem_region().data(), mem_region.data());
  return result;
}

absl::string_view ArrayMemRegion::mem_region() const {
  return absl::string_view(static_cast<char*>(mem_region_start_), nbytes_);
}

void* ArrayMemRegion::zeroth_element() const {
  // ArrayMemRegion cannot yet be constructed for situations where the
  // zeroth element pointer is different from mem_region_start_.
  return mem_region_start_;
}

absl::StatusOr<std::unique_ptr<std::string>> SerializeStringHostBuffer(
    absl::Span<const absl::Cord> cords) {
  proto::StringArrayContents string_array_proto;
  for (const auto& c : cords) {
    string_array_proto.add_strings(std::string(c));
  }
  return std::make_unique<std::string>(string_array_proto.SerializeAsString());
}

absl::StatusOr<std::vector<absl::Cord>> DeserializeStringHostBufferFromString(
    const std::string& serialized_string_buffer) {
  proto::StringArrayContents string_array_proto;
  if (!string_array_proto.ParseFromString(serialized_string_buffer)) {
    return absl::InvalidArgumentError(
        "Failed to parse serialized string buffer");
  }

  std::vector<absl::Cord> result;
  result.reserve(string_array_proto.strings_size());
  for (const auto& s : string_array_proto.strings()) {
    result.push_back(absl::Cord(s));
  }
  return result;
}

absl::Status DeserializeFromCordIntoPreallocatedStringHostBuffer(
    const absl::Cord& serialized_string_buffer,
    absl::Cord* preallocated_buffer) {
  proto::StringArrayContents string_array_proto;

#if defined(PLATFORM_GOOGLE)
  if (!string_array_proto.ParseFromString(serialized_string_buffer)) {
#else
  if (!string_array_proto.ParseFromString(  // No absl::Cord support in OSS.
          std::string(serialized_string_buffer))) {
#endif
    return absl::InvalidArgumentError(
        "Failed to parse serialized string buffer");
  }

  auto* current_cord = preallocated_buffer;
  for (const auto& s : string_array_proto.strings()) {
    *current_cord = s;
    ++current_cord;
  }
  return absl::OkStatus();
}

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
