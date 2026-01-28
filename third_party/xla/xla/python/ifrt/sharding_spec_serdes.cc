/* Copyright 2025 The OpenXLA Authors.

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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/ir/sharding_param.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding_spec.h"
#include "xla/python/ifrt/sharding_spec_serdes.pb.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {

namespace {

// Serialization/deserialization for `SingleDeviceShardingSpec`.
class SingleDeviceShardingSpecSerDes
    : public llvm::RTTIExtends<SingleDeviceShardingSpecSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    return "xla::ifrt::SingleDeviceShardingSpec";
  }

  absl::StatusOr<std::string> Serialize(
      const Serializable& serializable,
      std::unique_ptr<SerializeOptions> options) override {
    const SerDesVersion version = GetRequestedSerDesVersion(options.get());
    if (version.version_number() < SerDesVersionNumber(0)) {
      return absl::FailedPreconditionError(
          absl::StrCat("Unsupported ", version.version_number(),
                       " for SingleDeviceShardingSpec serialization"));
    }
    SingleDeviceShardingSpecProto proto;
    proto.set_version_number(SerDesVersionNumber(0).value());
    return proto.SerializeAsString();
  }

  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions> options) override {
    SingleDeviceShardingSpecProto proto;
    if (!proto.ParseFromString(serialized)) {
      return absl::InvalidArgumentError(
          "Failed to parse serialized SingleDeviceShardingSpec");
    }
    const SerDesVersionNumber version_number(proto.version_number());
    if (version_number != SerDesVersionNumber(0)) {
      return absl::FailedPreconditionError(
          absl::StrCat("Unsupported ", version_number,
                       " for SingleDeviceShardingSpec deserialization"));
    }
    return SingleDeviceShardingSpec::Create();
  }

  static char ID;  // NOLINT
};

// Serialization/deserialization for `OpaqueShardingSpec`.
class OpaqueShardingSpecSerDes
    : public llvm::RTTIExtends<OpaqueShardingSpecSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    return "xla::ifrt::OpaqueShardingSpec";
  }

  absl::StatusOr<std::string> Serialize(
      const Serializable& serializable,
      std::unique_ptr<SerializeOptions> options) override {
    const SerDesVersion version = GetRequestedSerDesVersion(options.get());
    if (version.version_number() < SerDesVersionNumber(0)) {
      return absl::FailedPreconditionError(
          absl::StrCat("Unsupported ", version.version_number(),
                       " for OpaqueShardingSpec serialization"));
    }
    const OpaqueShardingSpec& sharding_spec =
        llvm::cast<OpaqueShardingSpec>(serializable);
    OpaqueShardingSpecProto proto;
    proto.set_version_number(SerDesVersionNumber(0).value());
    proto.set_num_shards(sharding_spec.num_shards());
    return proto.SerializeAsString();
  }

  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions> options) override {
    OpaqueShardingSpecProto proto;
    if (!proto.ParseFromString(serialized)) {
      return absl::InvalidArgumentError(
          "Failed to parse serialized OpaqueShardingSpec");
    }
    const SerDesVersionNumber version_number(proto.version_number());
    if (version_number != SerDesVersionNumber(0)) {
      return absl::FailedPreconditionError(
          absl::StrCat("Unsupported ", version_number,
                       " for OpaqueShardingSpec deserialization"));
    }
    return OpaqueShardingSpec::Create(proto.num_shards());
  }

  static char ID;  // NOLINT
};

// Serialization/deserialization for `ConcreteShardingSpec`.
class ConcreteShardingSpecSerDes
    : public llvm::RTTIExtends<ConcreteShardingSpecSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    return "xla::ifrt::ConcreteShardingSpec";
  }

  absl::StatusOr<std::string> Serialize(
      const Serializable& serializable,
      std::unique_ptr<SerializeOptions> options) override {
    const SerDesVersion version = GetRequestedSerDesVersion(options.get());
    if (version.version_number() < SerDesVersionNumber(0)) {
      return absl::FailedPreconditionError(
          absl::StrCat("Unsupported ", version.version_number(),
                       " for ConcreteShardingSpec serialization"));
    }
    const ConcreteShardingSpec& sharding_spec =
        llvm::cast<ConcreteShardingSpec>(serializable);
    if (sharding_spec.index_domains().has_value()) {
      return absl::UnimplementedError(
          "Index domains are not yet supported in ConcreteShardingSpec "
          "serialization");
    }
    ConcreteShardingSpecProto proto;
    proto.set_version_number(SerDesVersionNumber(0).value());
    if (sharding_spec.has_static_shape()) {
      sharding_spec.shape().ToProto(*proto.mutable_shape(), version);
      for (const Shape& shape : sharding_spec.shard_shapes()) {
        *proto.add_shard_shapes() = shape.ToProto(version);
      }
    } else {
      sharding_spec.dynamic_shape().ToProto(*proto.mutable_dynamic_shape(),
                                            version);
      for (const DynamicShape& dynamic_shape :
           sharding_spec.shard_dynamic_shapes()) {
        dynamic_shape.ToProto(*proto.add_shard_dynamic_shapes(), version);
      }
    }
    return proto.SerializeAsString();
  }

  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions> options) override {
    ConcreteShardingSpecProto proto;
    if (!proto.ParseFromString(serialized)) {
      return absl::InvalidArgumentError(
          "Failed to parse serialized ConcreteShardingSpec");
    }
    const SerDesVersionNumber version_number(proto.version_number());
    if (version_number != SerDesVersionNumber(0)) {
      return absl::FailedPreconditionError(
          absl::StrCat("Unsupported ", version_number,
                       " for ConcreteShardingSpec deserialization"));
    }
    if (proto.has_shape()) {
      TF_ASSIGN_OR_RETURN(auto shape, Shape::FromProto(proto.shape()));
      std::vector<Shape> shard_shapes;
      shard_shapes.reserve(proto.shard_shapes_size());
      for (const auto& shard_shape_proto : proto.shard_shapes()) {
        TF_ASSIGN_OR_RETURN(auto shard_shape,
                            Shape::FromProto(shard_shape_proto));
        shard_shapes.push_back(std::move(shard_shape));
      }
      return ConcreteShardingSpec::Create(std::move(shape),
                                          std::move(shard_shapes));
    }
    if (!proto.has_dynamic_shape()) {
      return absl::InvalidArgumentError(
          "ConcreteShardingSpec must have Shape or DynamicShape.");
    }
    TF_ASSIGN_OR_RETURN(auto dynamic_shape,
                        DynamicShape::FromProto(proto.dynamic_shape()));
    std::vector<DynamicShape> shard_dynamic_shapes;
    shard_dynamic_shapes.reserve(proto.shard_dynamic_shapes_size());
    for (const auto& shard_dynamic_shape_proto : proto.shard_dynamic_shapes()) {
      TF_ASSIGN_OR_RETURN(auto dynamic_shape,
                          DynamicShape::FromProto(shard_dynamic_shape_proto));
      shard_dynamic_shapes.push_back(std::move(dynamic_shape));
    }
    return ConcreteShardingSpec::Create(std::move(dynamic_shape),
                                        std::move(shard_dynamic_shapes));
  }

  static char ID;  // NOLINT
};

// Serialization/deserialization for `ConcreteEvenShardingSpec`.
class ConcreteEvenShardingSpecSerDes
    : public llvm::RTTIExtends<ConcreteEvenShardingSpecSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    return "xla::ifrt::ConcreteEvenShardingSpec";
  }

  absl::StatusOr<std::string> Serialize(
      const Serializable& serializable,
      std::unique_ptr<SerializeOptions> options) override {
    const SerDesVersion version = GetRequestedSerDesVersion(options.get());
    if (version.version_number() < SerDesVersionNumber(0)) {
      return absl::FailedPreconditionError(
          absl::StrCat("Unsupported ", version.version_number(),
                       " for ConcreteEvenShardingSpec serialization"));
    }
    const ConcreteEvenShardingSpec& sharding_spec =
        llvm::cast<ConcreteEvenShardingSpec>(serializable);
    ConcreteEvenShardingSpecProto proto;
    proto.set_version_number(SerDesVersionNumber(0).value());
    proto.set_num_shards(sharding_spec.num_shards());
    sharding_spec.shape().ToProto(*proto.mutable_shape(), version);
    sharding_spec.shard_shape().ToProto(*proto.mutable_shard_shape(), version);
    proto.set_is_fully_replicated(sharding_spec.IsFullyReplicated());
    return proto.SerializeAsString();
  }

  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions> options) override {
    ConcreteEvenShardingSpecProto proto;
    if (!proto.ParseFromString(serialized)) {
      return absl::InvalidArgumentError(
          "Failed to parse serialized ConcreteEvenShardingSpec");
    }
    const SerDesVersionNumber version_number(proto.version_number());
    if (version_number != SerDesVersionNumber(0)) {
      return absl::FailedPreconditionError(
          absl::StrCat("Unsupported ", version_number,
                       " for ConcreteEvenShardingSpec deserialization"));
    }
    TF_ASSIGN_OR_RETURN(auto shape, Shape::FromProto(proto.shape()));
    TF_ASSIGN_OR_RETURN(auto shard_shape,
                        Shape::FromProto(proto.shard_shape()));
    return ConcreteEvenShardingSpec::Create(
        proto.num_shards(), std::move(shape), std::move(shard_shape),
        proto.is_fully_replicated());
  }

  static char ID;  // NOLINT
};

class ShardingParamShardingSpecSerDes
    : public llvm::RTTIExtends<ShardingParamShardingSpecSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    return "xla::ifrt::ShardingParamShardingSpec";
  }

  absl::StatusOr<std::string> Serialize(
      const Serializable& serializable,
      std::unique_ptr<SerializeOptions> options) override {
    const SerDesVersion version = GetRequestedSerDesVersion(options.get());
    if (version.version_number() < SerDesVersionNumber(0)) {
      return absl::FailedPreconditionError(
          absl::StrCat("Unsupported ", version.version_number(),
                       " for ShardingParamShardingSpec serialization"));
    }
    const ShardingParamShardingSpec& sharding_spec =
        llvm::cast<ShardingParamShardingSpec>(serializable);
    ShardingParamShardingSpecProto proto;
    proto.set_version_number(SerDesVersionNumber(0).value());
    TF_RETURN_IF_ERROR(sharding_spec.sharding_param().ToProto(
        *proto.mutable_sharding_param(), version));
    return proto.SerializeAsString();
  }

  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions> options) override {
    ShardingParamShardingSpecProto proto;
    if (!proto.ParseFromString(serialized)) {
      return absl::InvalidArgumentError(
          "Failed to parse serialized ShardingParamShardingSpec");
    }
    const SerDesVersionNumber version_number(proto.version_number());
    if (version_number != SerDesVersionNumber(0)) {
      return absl::FailedPreconditionError(
          absl::StrCat("Unsupported ", version_number,
                       " for ShardingParamShardingSpec deserialization"));
    }
    TF_ASSIGN_OR_RETURN(ShardingParam sharding_param,
                        ShardingParam::FromProto(proto.sharding_param()));
    return ShardingParamShardingSpec::Create(std::move(sharding_param));
  }

  static char ID;  // NOLINT
};

[[maybe_unused]] char SingleDeviceShardingSpecSerDes::ID = 0;   // NOLINT
[[maybe_unused]] char OpaqueShardingSpecSerDes::ID = 0;         // NOLINT
[[maybe_unused]] char ConcreteShardingSpecSerDes::ID = 0;       // NOLINT
[[maybe_unused]] char ConcreteEvenShardingSpecSerDes::ID = 0;   // NOLINT
[[maybe_unused]] char ShardingParamShardingSpecSerDes::ID = 0;  // NOLINT

// clang-format off
bool register_single_device_sharding_spec_serdes = ([]{
  RegisterSerDes<SingleDeviceShardingSpec>(
      std::make_unique<SingleDeviceShardingSpecSerDes>());
}(), true);

bool register_opaque_sharding_spec_serdes = ([]{
  RegisterSerDes<OpaqueShardingSpec>(
      std::make_unique<OpaqueShardingSpecSerDes>());
}(), true);

bool register_concrete_sharding_spec_serdes = ([]{
  RegisterSerDes<ConcreteShardingSpec>(
      std::make_unique<ConcreteShardingSpecSerDes>());
}(), true);

bool register_concrete_even_sharding_spec_serdes = ([]{
  RegisterSerDes<ConcreteEvenShardingSpec>(
      std::make_unique<ConcreteEvenShardingSpecSerDes>());
}(), true);

bool register_sharding_param_sharding_spec_serdes = ([]{
  RegisterSerDes<ShardingParamShardingSpec>(
      std::make_unique<ShardingParamShardingSpecSerDes>());
}(), true);
// clang-format on

}  // namespace

}  // namespace ifrt
}  // namespace xla
