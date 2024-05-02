/*
 * Copyright 2023 The OpenXLA Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef XLA_PYTHON_IFRT_PROXY_CLIENT_CLIENT_H_
#define XLA_PYTHON_IFRT_PROXY_CLIENT_CLIENT_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/tuple.h"
#include "xla/python/ifrt/value.h"
#include "xla/python/ifrt_proxy/client/compiler.h"
#include "xla/python/ifrt_proxy/client/device.h"
#include "xla/python/ifrt_proxy/client/memory.h"
#include "xla/python/ifrt_proxy/client/rpc_helper.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {
namespace proxy {

// Implementation of the xla::ifrt::Client interface.
class Client final : public llvm::RTTIExtends<Client, xla::ifrt::Client> {
 public:
  static absl::StatusOr<std::unique_ptr<Client>> Create(
      std::shared_ptr<RpcHelper> rpc_helper, InitResponse init_response);

  ~Client() override;

  absl::StatusOr<tsl::RCReference<xla::ifrt::Array>> MakeArrayFromHostBuffer(
      const void* data, DType dtype, Shape shape,
      std::optional<absl::Span<const int64_t>> byte_strides,
      std::shared_ptr<const Sharding> sharding, HostBufferSemantics semantics,
      std::function<void()> on_done_with_host_buffer) override;

  absl::StatusOr<tsl::RCReference<xla::ifrt::Array>>
  AssembleArrayFromSingleDeviceArrays(
      Shape shape, std::shared_ptr<const Sharding> sharding,
      absl::Span<tsl::RCReference<xla::ifrt::Array>> arrays,
      ArrayCopySemantics semantics) override;

  absl::StatusOr<tsl::RCReference<Tuple>> MakeTuple(
      absl::Span<tsl::RCReference<Value>> values) override {
    return absl::UnimplementedError(
        "MakeTuple is not supported for the IFRT proxy client.");
  }

  absl::string_view runtime_type() const override { return runtime_type_; }
  absl::string_view platform_name() const override { return platform_name_; }
  absl::string_view platform_version() const override {
    return platform_version_;
  }
  PlatformId platform_id() const override { return platform_id_; }
  absl::flat_hash_map<std::string, ClientAttribute> attributes()
      const override {
    // TODO(b/309059940): Forward the backend attributes to the client.
    return {};
  }
  int device_count() const override { return devices().size(); }
  int addressable_device_count() const override {
    return addressable_devices().size();
  }
  absl::Span<xla::ifrt::Device* const> devices() const override {
    return device_ptrs_;
  }
  absl::Span<xla::ifrt::Device* const> addressable_devices() const override {
    return addressable_device_ptrs_;
  }
  int process_index() const override { return process_index_; }
  absl::StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;
  absl::StatusOr<xla::ifrt::Device*> LookupDevice(
      DeviceId device_id) const override;
  absl::StatusOr<xla::ifrt::Device*> LookupAddressableDevice(
      int local_hardware_id) const override {
    return absl::UnimplementedError(
        "LookupAddressableDevice is not supported for the IFRT proxy client.");
  }
  xla::ifrt::Compiler* GetDefaultCompiler() override {
    return &default_compiler_;
  }
  absl::StatusOr<std::shared_ptr<const xla::PjRtTopologyDescription>>
  GetTopologyForDevices(const xla::ifrt::DeviceList& devices) const override {
    return absl::UnimplementedError(
        "GetTopologyForDevices is not supported for the IFRT proxy client.");
  }
  absl::StatusOr<std::unique_ptr<xla::PjRtLayout>> GetDefaultLayoutForDevice(
      xla::ifrt::DType dtype, absl::Span<const int64_t> dims,
      xla::ifrt::Device* device) const override {
    return absl::UnimplementedError(
        "GetDefaultLayout is not supported for the IFRT proxy client.");
  }

  // For llvm::RTTIExtends.
  static char ID;  // NOLINT

 private:
  Client(std::shared_ptr<RpcHelper> rpc_helper, uint64_t session_id,
         std::string platform_name, std::string platform_version,
         uint64_t platform_id, uint64_t process_index, std::string runtime_type,
         absl::flat_hash_map<int, std::unique_ptr<Device>> devices,
         std::vector<xla::ifrt::Device*> device_ptrs,
         std::vector<xla::ifrt::Device*> addressable_device_ptrs,
         absl::flat_hash_map<int, std::unique_ptr<Memory>> memories);

  // rpc_helper_ will be referenced by various IFRT objects whose lifetime is
  // managed by the layer above the IFRT interface, so shared_ptr is
  // appropriate.
  const std::shared_ptr<RpcHelper> rpc_helper_;

  const std::string platform_name_;
  const std::string platform_version_;
  const uint64_t platform_id_;
  const uint64_t process_index_;
  const std::string runtime_type_;

  const absl::flat_hash_map<int, std::unique_ptr<Device>> devices_;
  const std::vector<xla::ifrt::Device*> device_ptrs_;
  const std::vector<xla::ifrt::Device*> addressable_device_ptrs_;

  const absl::flat_hash_map<int, std::unique_ptr<Memory>> memories_;

  Compiler default_compiler_;
};

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_CLIENT_CLIENT_H_
