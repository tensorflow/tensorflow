/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/python/ifrt/sharding_serdes.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/python/ifrt/device.h"
#include "tensorflow/compiler/xla/python/ifrt/memory.h"
#include "tensorflow/compiler/xla/python/ifrt/serdes.h"
#include "tensorflow/compiler/xla/python/ifrt/shape.h"
#include "tensorflow/compiler/xla/python/ifrt/sharding.h"
#include "tensorflow/compiler/xla/python/ifrt/sharding.pb.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {

char DeserializeShardingOptions::ID = 0;

namespace {

// Serialization/deserialization for `SingleDeviceSharding`.
class SingleDeviceShardingSerDes
    : public llvm::RTTIExtends<SingleDeviceShardingSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    return "xla::ifrt::SingleDeviceSharding";
  }

  absl::StatusOr<std::string> Serialize(Serializable& serializable) override {
    const SingleDeviceSharding& sharding =
        llvm::cast<SingleDeviceSharding>(serializable);
    SingleDeviceShardingProto proto;
    proto.set_device_id(sharding.devices().front()->id());
    if (sharding.memory_kind().memory_kind().has_value()) {
      proto.set_memory_kind(std::string(*sharding.memory_kind().memory_kind()));
    }
    return proto.SerializeAsString();
  }

  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions> options) override {
    TF_ASSIGN_OR_RETURN(auto deserialize_sharding_options,
                        GetDeserializeShardingOptions(std::move(options)));
    SingleDeviceShardingProto proto;
    if (!proto.ParseFromString(serialized)) {
      return absl::InvalidArgumentError(
          "Failed to parse serialized SimpleDeviceSharding");
    }
    TF_ASSIGN_OR_RETURN(
        Device * device,
        deserialize_sharding_options->lookup_device(proto.device_id()));
    MemoryKind memory_kind;
    if (proto.has_memory_kind()) {
      memory_kind = MemoryKind(proto.memory_kind());
    }
    return SingleDeviceSharding::Create(device, memory_kind);
  }

  static char ID;  // NOLINT
};

// Serialization/deserialization for `OpaqueSharding`.
class OpaqueShardingSerDes
    : public llvm::RTTIExtends<OpaqueShardingSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    return "xla::ifrt::OpaqueSharding";
  }

  absl::StatusOr<std::string> Serialize(Serializable& serializable) override {
    const OpaqueSharding& sharding = llvm::cast<OpaqueSharding>(serializable);
    OpaqueShardingProto proto;
    *proto.mutable_devices() = sharding.devices().ToProto();
    if (sharding.memory_kind().memory_kind().has_value()) {
      proto.set_memory_kind(std::string(*sharding.memory_kind().memory_kind()));
    }
    return proto.SerializeAsString();
  }

  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions> options) override {
    TF_ASSIGN_OR_RETURN(auto deserialize_sharding_options,
                        GetDeserializeShardingOptions(std::move(options)));

    OpaqueShardingProto proto;
    if (!proto.ParseFromString(serialized)) {
      return absl::InvalidArgumentError(
          "Failed to parse serialized OpaqueSharding");
    }
    TF_ASSIGN_OR_RETURN(
        auto devices,
        DeviceList::FromProto(deserialize_sharding_options->lookup_device,
                              proto.devices()));
    MemoryKind memory_kind;
    if (proto.has_memory_kind()) {
      memory_kind = MemoryKind(proto.memory_kind());
    }
    return OpaqueSharding::Create(std::move(devices), memory_kind);
  }

  static char ID;  // NOLINT
};

// Serialization/deserialization for `ConcreteSharding`.
class ConcreteShardingSerDes
    : public llvm::RTTIExtends<ConcreteShardingSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    return "xla::ifrt::ConcreteSharding";
  }

  absl::StatusOr<std::string> Serialize(Serializable& serializable) override {
    const ConcreteSharding& sharding =
        llvm::cast<ConcreteSharding>(serializable);
    ConcreteShardingProto proto;
    *proto.mutable_devices() = sharding.devices().ToProto();
    if (sharding.memory_kind().memory_kind().has_value()) {
      proto.set_memory_kind(std::string(*sharding.memory_kind().memory_kind()));
    }
    *proto.mutable_shape() = sharding.shape().ToProto();
    for (const Shape& shape : sharding.shard_shapes()) {
      *proto.add_shard_shapes() = shape.ToProto();
    }
    return proto.SerializeAsString();
  }

  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions> options) override {
    TF_ASSIGN_OR_RETURN(auto deserialize_sharding_options,
                        GetDeserializeShardingOptions(std::move(options)));

    ConcreteShardingProto proto;
    if (!proto.ParseFromString(serialized)) {
      return absl::InvalidArgumentError(
          "Failed to parse serialized ConcreteSharding");
    }
    TF_ASSIGN_OR_RETURN(
        auto devices,
        DeviceList::FromProto(deserialize_sharding_options->lookup_device,
                              proto.devices()));
    MemoryKind memory_kind;
    if (proto.has_memory_kind()) {
      memory_kind = MemoryKind(proto.memory_kind());
    }
    TF_ASSIGN_OR_RETURN(auto shape, Shape::FromProto(proto.shape()));
    std::vector<Shape> shard_shapes;
    shard_shapes.reserve(proto.shard_shapes_size());
    for (const auto& shard_shape_proto : proto.shard_shapes()) {
      TF_ASSIGN_OR_RETURN(auto shard_shape,
                          Shape::FromProto(shard_shape_proto));
      shard_shapes.push_back(std::move(shard_shape));
    }
    return ConcreteSharding::Create(std::move(devices), memory_kind,
                                    std::move(shape), std::move(shard_shapes));
  }

  static char ID;  // NOLINT
};

// Serialization/deserialization for `ConcreteEvenSharding`.
class ConcreteEvenShardingSerDes
    : public llvm::RTTIExtends<ConcreteEvenShardingSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    return "xla::ifrt::ConcreteEvenSharding";
  }

  absl::StatusOr<std::string> Serialize(Serializable& serializable) override {
    const ConcreteEvenSharding& sharding =
        llvm::cast<ConcreteEvenSharding>(serializable);
    ConcreteEvenShardingProto proto;
    *proto.mutable_devices() = sharding.devices().ToProto();
    if (sharding.memory_kind().memory_kind().has_value()) {
      proto.set_memory_kind(std::string(*sharding.memory_kind().memory_kind()));
    }
    *proto.mutable_shape() = sharding.shape().ToProto();
    *proto.mutable_shard_shape() = sharding.shard_shape().ToProto();
    return proto.SerializeAsString();
  }

  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions> options) override {
    TF_ASSIGN_OR_RETURN(auto deserialize_sharding_options,
                        GetDeserializeShardingOptions(std::move(options)));

    ConcreteEvenShardingProto proto;
    if (!proto.ParseFromString(serialized)) {
      return absl::InvalidArgumentError(
          "Failed to parse serialized ConcreteEvenSharding");
    }
    TF_ASSIGN_OR_RETURN(
        auto devices,
        DeviceList::FromProto(deserialize_sharding_options->lookup_device,
                              proto.devices()));
    MemoryKind memory_kind;
    if (proto.has_memory_kind()) {
      memory_kind = MemoryKind(proto.memory_kind());
    }
    TF_ASSIGN_OR_RETURN(auto shape, Shape::FromProto(proto.shape()));
    TF_ASSIGN_OR_RETURN(auto shard_shape,
                        Shape::FromProto(proto.shard_shape()));
    return ConcreteEvenSharding::Create(std::move(devices), memory_kind,
                                        std::move(shape),
                                        std::move(shard_shape));
  }

  static char ID;  // NOLINT
};

// TODO(hyeontaek): Implement `ShardingParamShardingSerDes`.

[[maybe_unused]] char SingleDeviceShardingSerDes::ID = 0;  // NOLINT
[[maybe_unused]] char OpaqueShardingSerDes::ID = 0;        // NOLINT
[[maybe_unused]] char ConcreteShardingSerDes::ID = 0;      // NOLINT
[[maybe_unused]] char ConcreteEvenShardingSerDes::ID = 0;  // NOLINT

// clang-format off
bool register_single_device_sharding_serdes = ([]{
  RegisterSerDes<SingleDeviceSharding>(
      std::make_unique<SingleDeviceShardingSerDes>());
}(), true);

bool register_opaque_sharding_serdes = ([]{
  RegisterSerDes<OpaqueSharding>(
      std::make_unique<OpaqueShardingSerDes>());
}(), true);

bool register_concrete_sharding_serdes = ([]{
  RegisterSerDes<ConcreteSharding>(
      std::make_unique<ConcreteShardingSerDes>());
}(), true);

bool register_concrete_even_sharding_serdes = ([]{
  RegisterSerDes<ConcreteEvenSharding>(
      std::make_unique<ConcreteEvenShardingSerDes>());
}(), true);
// clang-format on

}  // namespace

StatusOr<std::unique_ptr<DeserializeShardingOptions>>
GetDeserializeShardingOptions(std::unique_ptr<DeserializeOptions> options) {
  if (!llvm::isa<DeserializeShardingOptions>(options.get())) {
    return xla::InvalidArgument("options must be DeserializeShardingOptions");
  }
  return std::unique_ptr<DeserializeShardingOptions>(
      static_cast<DeserializeShardingOptions*>(options.release()));
}

}  // namespace ifrt
}  // namespace xla
