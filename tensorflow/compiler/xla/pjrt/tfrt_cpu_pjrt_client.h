/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_TFRT_CPU_PJRT_CLIENT_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_TFRT_CPU_PJRT_CLIENT_H_

#include <memory>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/layout.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/semaphore.h"
#include "tensorflow/compiler/xla/pjrt/worker_thread.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime

namespace xla {

class TfrtCpuDevice : public PjRtDevice {
 public:
  TfrtCpuDevice(int id, bool asynchronous);

  void SetClient(PjRtClient* client) {
    CHECK(client_ == nullptr);
    client_ = client;
  }

  PjRtClient* client() const override { return client_; }

  bool IsAddressable() const override {
    return process_index() == client()->process_index();
  }

  int id() const override { return id_; }

  int process_index() const override { return 0; }

  // Used as `device_ordinal`.
  int local_hardware_id() const override { return id_; }

  absl::string_view device_kind() const override;

  std::string DebugString() const override;

  Status TransferToInfeed(const LiteralSlice& literal) override;

  Status TransferFromOutfeed(MutableBorrowingLiteral literal) override;

  // Returns a semaphore for admission control on inflight computations.
  Semaphore& max_inflight_computations_semaphore() {
    return max_inflight_computations_semaphore_;
  }

 private:
  int id_;
  PjRtClient* client_ = nullptr;

  // TODO(zhangqiaorjc): Optimize semaphore related overhead.
  // Semaphore used to limit how many programs can be enqueued by the host
  // ahead of the device.
  Semaphore max_inflight_computations_semaphore_;
};

class TfrtCpuExecutable;

class TfrtCpuClient : public PjRtClient {
 public:
  TfrtCpuClient(int process_index,
                std::vector<std::unique_ptr<TfrtCpuDevice>> devices,
                std::unique_ptr<tfrt::HostContext> host_ctx);

  int process_index() const override { return process_index_; }

  int device_count() const override { return devices_.size(); }

  int addressable_device_count() const override {
    return addressable_devices_.size();
  }

  absl::Span<PjRtDevice* const> devices() const override { return devices_; }

  absl::Span<PjRtDevice* const> addressable_devices() const override {
    return addressable_devices_;
  }

  StatusOr<PjRtDevice*> LookupDevice(int device_id) const override;

  StatusOr<PjRtDevice*> LookupAddressableDevice(
      int local_hardware_id) const override;

  PjRtPlatformId platform_id() const override {
    return tensorflow::Fingerprint64(kCpuName);
  }

  absl::string_view platform_name() const override { return kCpuName; }

  absl::string_view platform_version() const override { return "<unknown>"; }

  StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;

  StatusOr<std::unique_ptr<HloCostAnalysis>> GetHloCostAnalysis() override;

  StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      const XlaComputation& computation, CompileOptions options) override {
    return Unimplemented("Compile");
  }

  StatusOr<absl::optional<std::string>> ExecutableFingerprint(
      const PjRtExecutable& executable) const override;

  StatusOr<std::unique_ptr<PjRtBuffer>> CreateUninitializedBuffer(
      const Shape& shape, PjRtDevice* device) override {
    return Unimplemented("CreateUninitializedBuffer");
  }

  StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBuffer(
      const void* data, const Shape& shape,
      HostBufferSemantics host_buffer_semantics,
      std::function<void()> on_done_with_host_buffer,
      PjRtDevice* device) override {
    return Unimplemented("BufferFromHostBuffer");
  }

  StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostLiteral(
      const LiteralSlice& literal, PjRtDevice* device) override {
    return Unimplemented("BufferFromHostLiteral");
  }

  void MakeCrossHostReceiveBuffers(
      absl::Span<const Shape> shapes, PjRtDevice* device,
      PjRtCrossHostRecvNotifier&& notifier) override {
    LOG(FATAL) << "MakeCrossHostReceiveBuffers not implemented.";
  }

  StatusOr<std::unique_ptr<PjRtBuffer>> CreateViewOfDeviceBuffer(
      void* device_ptr, const Shape& shape, PjRtDevice* device,
      std::function<void()> on_delete_callback) override {
    return Unimplemented("CreateViewOfDeviceBuffer not implemented.");
  }

  StatusOr<ChannelHandle> CreateChannelHandle() override {
    return Unimplemented("CreateChannelHandle not implemented.");
  }
  StatusOr<ChannelHandle> CreateDeviceToHostChannelHandle() override {
    return Unimplemented("CreateDeviceToHostChannelHandle not implemented.");
  }
  StatusOr<ChannelHandle> CreateHostToDeviceChannelHandle() override {
    return Unimplemented("CreateHostToDeviceChannelHandle not implemented.");
  }

  Status Defragment(absl::Span<PjRtBuffer* const> buffers,
                    absl::Span<PjRtExecutable* const> executables) override {
    return Unimplemented("Defragment not implemented.");
  }

  tfrt::HostContext* GetHostContext() const { return host_ctx_.get(); }

  Eigen::ThreadPoolDevice* eigen_intraop_device() const {
    return eigen_intraop_device_.get();
  }

 private:
  int process_index_;
  // Includes all devices, including non-addressable devices.
  std::vector<std::unique_ptr<TfrtCpuDevice>> owned_devices_;
  // Pointers to `owned_devices_`.
  std::vector<PjRtDevice*> devices_;
  // Maps Device::id() to the corresponding Device. Includes all devices.
  absl::flat_hash_map<int, TfrtCpuDevice*> id_to_device_;
  // Addressable devices indexed by core_id.
  std::vector<PjRtDevice*> addressable_devices_;
  std::unique_ptr<tfrt::HostContext> host_ctx_;
  std::unique_ptr<ComputationPlacer> computation_placer_;

  // TODO(zhangqiaorjc): Use tfrt::compat::EigenHostContextThreadPool.
  std::unique_ptr<tensorflow::thread::ThreadPool> eigen_intraop_pool_;
  std::unique_ptr<Eigen::ThreadPoolDevice> eigen_intraop_device_;
};

StatusOr<std::unique_ptr<PjRtClient>> GetTfrtCpuClient(bool asynchronous);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_TFRT_CPU_PJRT_CLIENT_H_
