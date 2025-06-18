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

#ifndef XLA_PYTHON_COMPILE_ONLY_IFRT_CLIENT_H_
#define XLA_PYTHON_COMPILE_ONLY_IFRT_CLIENT_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/basic_device_list.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/program.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/topology.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/python/ifrt/value.h"
#include "xla/python/pjrt_ifrt/pjrt_attribute_map_util.h"
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"
#include "xla/python/pjrt_ifrt/pjrt_topology.h"
#include "xla/service/computation_placer.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

class CompileOnlyMemory
    : public llvm::RTTIExtends<CompileOnlyMemory, ifrt::Memory> {
 public:
  explicit CompileOnlyMemory(
      int id, const PjRtMemorySpaceDescription* memory_description,
      ifrt::Device* device)
      : id_(id),
        kind_(memory_description->kind()),
        debug_string_(absl::StrFormat("CompileOnlyMemory(id=%d, kind=%s)", id,
                                      memory_description->kind())),
        device_(device) {}

  ifrt::MemoryId Id() const override { return ifrt::MemoryId(id_); }

  const ifrt::MemoryKind& Kind() const override { return kind_; }

  absl::string_view ToString() const override { return debug_string_; }
  absl::string_view DebugString() const override { return debug_string_; }

  absl::Span<ifrt::Device* const> Devices() const override {
    return absl::Span<ifrt::Device* const>{&device_, 1};
  }

  static char ID;  // NOLINT

 private:
  int id_;
  ifrt::MemoryKind kind_;
  std::string debug_string_;
  ifrt::Device* device_;
};

class CompileOnlyDevice
    : public llvm::RTTIExtends<CompileOnlyDevice, ifrt::Device> {
 public:
  explicit CompileOnlyDevice(const PjRtDeviceDescription* description)
      : description_(std::move(description)),
        attributes_(ifrt::FromPjRtAttributeMap(description_->Attributes())) {}

  const PjRtDeviceDescription& description() const { return *description_; }

  ifrt::Client* client() const override { return nullptr; }
  bool IsAddressable() const override { return false; }
  ifrt::DeviceId Id() const override {
    return ifrt::DeviceId(description_->id());
  }

  int ProcessIndex() const override { return description_->process_index(); }

  absl::string_view Kind() const override {
    return description_->device_kind();
  }

  absl::string_view ToString() const override {
    return description_->ToString();
  }

  absl::string_view DebugString() const override {
    return description_->DebugString();
  }

  absl::Span<ifrt::Memory* const> Memories() const override {
    return unowned_memories_;
  }
  absl::StatusOr<ifrt::Memory*> DefaultMemory() const override {
    if (default_memory_) {
      return default_memory_;
    }
    return Unimplemented("DefaultMemory is not supported");
  }

  const ifrt::AttributeMap& Attributes() const override { return attributes_; }

  void AttachMemory(std::unique_ptr<ifrt::Memory> memory) {
    unowned_memories_.push_back(memory.get());
    owned_memories_.push_back(std::move(memory));
  }

  void SetDefaultMemory(ifrt::Memory* memory) { default_memory_ = memory; }

  static char ID;  // NOLINT

 private:
  const PjRtDeviceDescription* description_;
  ifrt::AttributeMap attributes_;
  ifrt::Memory* default_memory_ = nullptr;
  std::vector<ifrt::Memory*> unowned_memories_;
  std::vector<std::unique_ptr<ifrt::Memory>> owned_memories_;
};

class CompileOnlyIfrtCompiler final
    : public llvm::RTTIExtends<CompileOnlyIfrtCompiler, ifrt::Compiler> {
 public:
  absl::StatusOr<ifrt::LoadedExecutableRef> CompileAndLoad(
      std::unique_ptr<ifrt::Program> program,
      std::unique_ptr<ifrt::CompileOptions> options) override {
    return Unimplemented("Compile not implemented.");
  }

  absl::StatusOr<ifrt::ExecutableRef> Compile(
      std::unique_ptr<ifrt::Program> program, const ifrt::Topology& topology,
      std::unique_ptr<ifrt::CompileOptions> options) override {
    return Unimplemented("Compile not implemented.");
  }

  absl::StatusOr<ifrt::LoadedExecutableRef> DeserializeLoadedExecutable(
      absl::string_view serialized,
      std::unique_ptr<ifrt::DeserializeExecutableOptions> options) override {
    return Unimplemented("DeserializeLoadedExecutable not implemented.");
  }

  static char ID;  // NOLINT
};

class CompileOnlyIfRtClient final
    : public llvm::RTTIExtends<CompileOnlyIfRtClient, ifrt::Client> {
 public:
  explicit CompileOnlyIfRtClient(std::shared_ptr<ifrt::PjRtTopology> topology)
      : topology_(std::move(topology)),
        descriptions_(topology_->DeviceDescriptions()),
        attributes_(ifrt::AttributeMap::Map()) {
    int offset = 0;
    for (auto& description : descriptions_) {
      owned_devices_.push_back(
          std::make_unique<CompileOnlyDevice>(description.get()));
      auto* device = owned_devices_.back().get();
      devices_.push_back(device);
      if (description->process_index() == process_index()) {
        auto default_memory = description->default_memory_space();
        for (auto* memory_description : description->memory_spaces()) {
          auto memory = std::make_unique<CompileOnlyMemory>(
              offset, memory_description, device);
          if (default_memory.ok() && memory_description == *default_memory) {
            device->SetDefaultMemory(memory.get());
          }
          device->AttachMemory(std::move(memory));
          ++offset;
        }
      }
    }
  }

  absl::StatusOr<xla::ifrt::ArrayRef> MakeArrayFromHostBuffer(
      const void* data, xla::ifrt::DType dtype, xla::ifrt::Shape shape,
      std::optional<absl::Span<const int64_t>> byte_strides,
      xla::ifrt::ShardingRef sharding, HostBufferSemantics semantics,
      std::function<void()> on_done_with_host_buffer,
      tsl::RCReference<xla::ifrt::UserContext> user_context) override {
    return Unimplemented(
        "MakeArrayFromHostBuffer not available with compile-only client.");
  }

  absl::StatusOr<std::vector<ifrt::ArrayRef>> MakeArraysFromHostBufferShards(
      absl::Span<MakeArraysFromHostBufferShardsSpec> specs,
      HostBufferSemantics semantics,
      tsl::RCReference<xla::ifrt::UserContext> user_context) override {
    return Unimplemented(
        "MakeArraysFromHostBufferShards not available with compile-only "
        "client.");
  }

  absl::StatusOr<std::vector<ifrt::ArrayRef>> MakeErrorArrays(
      const absl::Status& error, absl::Span<const ifrt::ArraySpec> array_specs,
      tsl::RCReference<ifrt::UserContext> user_context) override {
    return Unimplemented(
        "MakeErrorArrays not available with compile-only client.");
  }

  absl::StatusOr<ifrt::ArrayRef> AssembleArrayFromSingleDeviceArrays(
      ifrt::DType dtype, ifrt::Shape shape, ifrt::ShardingRef sharding,
      absl::Span<ifrt::ArrayRef> arrays,
      ifrt::ArrayCopySemantics array_copy_semantics,
      ifrt::SingleDeviceShardSemantics single_device_shard_semantics) override {
    return Unimplemented(
        "AssembleArrayFromSingleDeviceArrays not available with compile-only "
        "client.");
  }

  absl::StatusOr<std::vector<ifrt::ArrayRef>> CopyArrays(
      absl::Span<ifrt::ArrayRef> arrays,
      std::optional<ifrt::DeviceListRef> devices,
      std::optional<ifrt::MemoryKind> memory_kind,
      ifrt::ArrayCopySemantics semantics) override {
    return Unimplemented("CopyArrays not available with compile-only client.");
  }

  absl::StatusOr<std::vector<ifrt::ArrayRef>> RemapArrays(
      const ifrt::RemapPlan& plan, absl::Span<ifrt::ArrayRef> arrays,
      ifrt::ArrayCopySemantics semantics) override {
    return Unimplemented("RemapArrays not available with compile-only client.");
  }

  ifrt::Future<> GetReadyFuture(
      absl::Span<const ifrt::ValueRef> values) override {
    return ifrt::Future<>(Unimplemented(
        "GetReadyFuture not available with compile-only client."));
  }

  absl::StatusOr<tsl::RCReference<ifrt::Tuple>> MakeTuple(
      absl::Span<ifrt::ValueRef> values) override {
    return Unimplemented("MakeTuple not available with compile-only client.");
  }

  absl::string_view runtime_type() const override {
    return "compile_only_runtime";
  }

  absl::string_view platform_name() const override {
    return topology_->platform_name();
  }
  absl::string_view platform_version() const override {
    return topology_->platform_version();
  }
  ifrt::PlatformId platform_id() const override {
    return topology_->platform_id();
  }
  const ifrt::AttributeMap& Attributes() const override { return attributes_; }

  int device_count() const override { return devices().size(); }
  int addressable_device_count() const override { return 0; }
  absl::Span<ifrt::Device* const> devices() const override { return devices_; }
  absl::Span<ifrt::Device* const> addressable_devices() const override {
    return {};
  }
  int process_index() const override { return 0; }
  absl::Span<xla::ifrt::Device* const> GetAllDevices() const override {
    return devices_;
  }
  absl::StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override {
    return Unimplemented(
        "GetDefaultDeviceAssignment not available with compile-only client.");
  }
  absl::StatusOr<ifrt::Device*> LookupDevice(
      ifrt::DeviceId device_id) const override {
    return Unimplemented(
        "LookupDevice not available with compile-only client.");
  }

  absl::StatusOr<ifrt::Device*> LookupAddressableDevice(
      int local_hardware_id) const override {
    return Unimplemented(
        "LookupAddressableDevice not available with compile-only client.");
  }

  ifrt::DeviceListRef MakeDeviceList(
      absl::Span<ifrt::Device* const> devices) const override {
    return ifrt::BasicDeviceList::Create(devices);
  }

  ifrt::Compiler* GetDefaultCompiler() override { return &default_compiler_; }

  tsl::RCReference<xla::ifrt::UserContext> CreateUserContext() override {
    return tsl::RCReference<xla::ifrt::UserContext>();
  }

  static char ID;  // NOLINT

  const ifrt::PjRtTopology& topology() const { return *topology_; }

  absl::StatusOr<std::shared_ptr<ifrt::Topology>> GetTopologyForDevices(
      const xla::ifrt::DeviceListRef& devices) const override {
    return topology_;
  }

  absl::StatusOr<std::shared_ptr<const PjRtLayout>> GetDefaultLayout(
      ifrt::DType dtype, absl::Span<const int64_t> dims, ifrt::Device* device,
      ifrt::MemoryKind memory_kind) const override {
    if (memory_kind == ifrt::MemoryKind(UnpinnedHostMemorySpace::kKind)) {
      return std::make_shared<PjRtLayout>(
          LayoutUtil::MakeDescendingLayout(dims.size()));
    }
    TF_ASSIGN_OR_RETURN(PrimitiveType element_type, ToPrimitiveType(dtype));
    TF_ASSIGN_OR_RETURN(xla::Layout layout,
                        topology_->GetDefaultLayout(element_type, dims));
    return std::make_shared<PjRtLayout>(std::move(layout));
  }

 private:
  CompileOnlyIfrtCompiler default_compiler_;
  std::shared_ptr<ifrt::PjRtTopology> topology_;
  std::vector<std::unique_ptr<const PjRtDeviceDescription>> descriptions_;
  ifrt::AttributeMap attributes_;
  std::vector<std::unique_ptr<CompileOnlyDevice>> owned_devices_;
  std::vector<ifrt::Device*> devices_;
};

}  // namespace xla

#endif  // XLA_PYTHON_COMPILE_ONLY_IFRT_CLIENT_H_
