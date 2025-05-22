/* Copyright 2023 The OpenXLA Authors.

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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/pjrt_ifrt/xla_sharding.h"
#include "xla/python/pjrt_ifrt/xla_sharding.pb.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {

namespace {

// Serialization/deserialization for `HloSharding`.
class HloShardingSerDes : public llvm::RTTIExtends<HloSharding, SerDes> {
 public:
  absl::string_view type_name() const override {
    return "xla::ifrt::HloSharding";
  }

  absl::StatusOr<std::string> Serialize(
      const Serializable& serializable,
      std::unique_ptr<SerializeOptions>) override {
    const HloSharding& sharding = llvm::cast<HloSharding>(serializable);
    HloShardingProto proto;
    *proto.mutable_devices() = sharding.devices()->ToProto();
    if (sharding.memory_kind().memory_kind().has_value()) {
      proto.set_memory_kind(std::string(*sharding.memory_kind().memory_kind()));
    }
    *proto.mutable_xla_op_sharding() = sharding.xla_hlo_sharding().ToProto();
    return proto.SerializeAsString();
  }

  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions> options) override {
    const auto* deserialize_sharding_options =
        llvm::cast<DeserializeShardingOptions>(options.get());

    HloShardingProto proto;
    if (!proto.ParseFromString(serialized)) {
      return absl::InvalidArgumentError(
          "Failed to parse serialized HloSharding");
    }
    TF_ASSIGN_OR_RETURN(auto devices, DeviceList::FromProto(
                                          deserialize_sharding_options->client,
                                          proto.devices()));
    MemoryKind memory_kind;
    if (proto.has_memory_kind()) {
      memory_kind = MemoryKind(proto.memory_kind());
    }
    TF_ASSIGN_OR_RETURN(auto xla_hlo_sharding,
                        xla::HloSharding::FromProto(proto.xla_op_sharding()));
    return HloSharding::Create(std::move(devices), memory_kind,
                               std::move(xla_hlo_sharding));
  }

  static char ID;  // NOLINT
};

[[maybe_unused]] char HloShardingSerDes::ID = 0;  // NOLINT

// clang-format off
bool register_hlo_sharding_serdes = ([] {
  RegisterSerDes<HloSharding>(
      std::make_unique<HloShardingSerDes>());
}(), true);
// clang-format on

}  // namespace
}  // namespace ifrt
}  // namespace xla
