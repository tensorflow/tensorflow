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

#include "xla/python/ifrt_proxy/common/types.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/sharding_serdes.h"
#include "xla/python/ifrt_proxy/common/types.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace proxy {

DType FromDTypeProto(proto::DType dtype_proto) {
  switch (dtype_proto) {
    case proto::DType::DTYPE_PRED:
      return DType(DType::Kind::kPred);
    case proto::DType::DTYPE_TOKEN:
      return DType(DType::Kind::kToken);
#define CASE(X)                 \
  case proto::DType::DTYPE_##X: \
    return DType(DType::Kind::k##X);
      CASE(S4);
      CASE(S8);
      CASE(S16);
      CASE(S32);
      CASE(S64);
      CASE(U4);
      CASE(U8);
      CASE(U16);
      CASE(U32);
      CASE(U64);
      CASE(F16);
      CASE(F32);
      CASE(F64);
      CASE(BF16);
      CASE(C64);
      CASE(C128);
      CASE(F8E4M3FN);
      CASE(F8E4M3B11FNUZ);
      CASE(F8E4M3FNUZ);
      CASE(F8E5M2);
      CASE(F8E5M2FNUZ);
#undef CASE
    default:
      return DType(DType::Kind::kInvalid);
  }
}

proto::DType ToDTypeProto(DType dtype) {
  switch (dtype.kind()) {
    case DType::Kind::kPred:
      return proto::DType::DTYPE_PRED;
    case DType::Kind::kToken:
      return proto::DType::DTYPE_TOKEN;
#define CASE(X)           \
  case DType::Kind::k##X: \
    return proto::DType::DTYPE_##X;
      CASE(S4);
      CASE(S8);
      CASE(S16);
      CASE(S32);
      CASE(S64);
      CASE(U4);
      CASE(U8);
      CASE(U16);
      CASE(U32);
      CASE(U64);
      CASE(F16);
      CASE(F32);
      CASE(F64);
      CASE(BF16);
      CASE(C64);
      CASE(C128);
      CASE(F8E4M3FN);
      CASE(F8E4M3B11FNUZ);
      CASE(F8E4M3FNUZ);
      CASE(F8E5M2);
      CASE(F8E5M2FNUZ);
#undef CASE
    default:
      return proto::DType::DTYPE_UNSPECIFIED;
  }
}

Shape FromShapeProto(const proto::Shape& shape_proto) {
  return Shape(shape_proto.dimensions());
}

proto::Shape ToShapeProto(const Shape& shape) {
  proto::Shape shape_proto;
  for (int64_t dim : shape.dims()) {
    shape_proto.add_dimensions(dim);
  }
  return shape_proto;
}

absl::StatusOr<xla::PjRtValueType> FromVariantProto(
    const proto::Variant& variant_proto) {
  switch (variant_proto.value_case()) {
    case proto::Variant::kStringValue:
      return variant_proto.string_value();
    case proto::Variant::kInt64Value:
      return variant_proto.int64_value();
    case proto::Variant::kInt64List: {
      const auto& values = variant_proto.int64_list().values();
      return std::vector<int64_t>(values.begin(), values.end());
    }
    case proto::Variant::kFloatValue:
      return variant_proto.float_value();
    default:
      return absl::UnimplementedError(absl::StrCat(
          "Unknown xla.ifrt.proto.Variant case: ", variant_proto.value_case()));
  }
}

absl::StatusOr<proto::Variant> ToVariantProto(const xla::PjRtValueType& value) {
  proto::Variant variant;
  if (auto* s = std::get_if<std::string>(&value)) {
    variant.set_string_value(*s);
  } else if (auto* i = std::get_if<int64_t>(&value)) {
    variant.set_int64_value(*i);
  } else if (auto* is = std::get_if<std::vector<int64_t>>(&value)) {
    for (const int64_t i : *is) {
      variant.mutable_int64_list()->add_values(i);
    }
  } else if (auto* f = std::get_if<float>(&value)) {
    variant.set_float_value(*f);
  } else {
    return absl::UnimplementedError("Unknown xla::PjRtValueType type");
  }
  return variant;
}

absl::StatusOr<std::shared_ptr<const Sharding>> FromShardingProto(
    DeviceList::LookupDeviceFunc lookup_device,
    const proto::Sharding& sharding_proto) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<Serializable> sharding,
                      Deserialize(sharding_proto.serialized_sharding(),
                                  std::make_unique<DeserializeShardingOptions>(
                                      std::move(lookup_device))));
  return std::shared_ptr<const Sharding>(
      llvm::cast<Sharding>(sharding.release()));
}

absl::StatusOr<proto::Sharding> ToShardingProto(const Sharding& sharding) {
  proto::Sharding sharding_proto;
  TF_ASSIGN_OR_RETURN(*sharding_proto.mutable_serialized_sharding(),
                      Serialize(const_cast<Sharding&>(sharding)));
  return sharding_proto;
}

proto::ArrayCopySemantics ToArrayCopySemanticsProto(ArrayCopySemantics s) {
  switch (s) {
    case ArrayCopySemantics::kAlwaysCopy:
      return proto::ARRAY_COPY_SEMANTICS_ALWAYS_COPY;
    case ArrayCopySemantics::kDonateInput:
      return proto::ARRAY_COPY_SEMANTICS_DONATE_INPUT;
    case ArrayCopySemantics::kReuseInput:
      return proto::ARRAY_COPY_SEMANTICS_REUSE_INPUT;
  }
}

absl::StatusOr<ArrayCopySemantics> FromArrayCopySemanticsProto(
    proto::ArrayCopySemantics s) {
  MakeArrayFromHostBufferRequest req;
  switch (s) {
    case proto::ARRAY_COPY_SEMANTICS_ALWAYS_COPY:
      return ArrayCopySemantics::kAlwaysCopy;
    case proto::ARRAY_COPY_SEMANTICS_DONATE_INPUT:
      return ArrayCopySemantics::kDonateInput;
    case proto::ARRAY_COPY_SEMANTICS_REUSE_INPUT:
      return ArrayCopySemantics::kReuseInput;
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unhandled proto-enum value ", s, ":",
                       proto::ArrayCopySemantics_Name(s)));
  }
}

std::vector<int64_t> FromByteStridesProto(const proto::ByteStrides& strides) {
  std::vector<int64_t> result;
  result.reserve(strides.strides_size());
  for (auto x : strides.strides()) {
    result.push_back(x);
  }
  return result;
}

proto::ByteStrides ToByteStridesProto(const absl::Span<const int64_t> strides) {
  proto::ByteStrides result;
  for (auto x : strides) {
    result.add_strides(x);
  }
  return result;
}

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
