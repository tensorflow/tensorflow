/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_PJRT_INTERPRETER_INTERPRETER_CLIENT_H_
#define XLA_PJRT_INTERPRETER_INTERPRETER_CLIENT_H_

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/die_if_null.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/evaluator/hlo_evaluator_interface.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/interpreter/interpreter_executable.h"
#include "xla/pjrt/interpreter/interpreter_topology_description.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/scoped_async_tracking_event.h"
#include "xla/runtime/chip_id.h"
#include "xla/runtime/device_id.h"
#include "xla/service/device_assignment.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

class InterpreterMemorySpace final : public PjRtMemorySpace {
 public:
  explicit InterpreterMemorySpace(PjRtClient* absl_nonnull client)
      : client_(ABSL_DIE_IF_NULL(client)) {}

  PjRtClient* client() const override { return client_; }

  absl::Span<PjRtDevice* const> devices() const override {
    return client_->devices();
  }

  int id() const override { return 0; }

  absl::string_view kind() const override { return "interpreter"; }

  int kind_id() const override { return 0; }

  absl::string_view DebugString() const override { return "interpreter:0"; }

  absl::string_view ToString() const override {
    return "InterpreterMemorySpace(id=0)";
  }

  PJRT_Memory* ToCApiPtr() override { return capi_delegator_.ToCApiPtr(); }

 private:
  PjRtClient* client_ = nullptr;
  PjRtMemorySpaceCApiDelegator capi_delegator_{this};
};

class InterpreterDevice final : public PjRtDevice {
 public:
  explicit InterpreterDevice(PjRtClient* absl_nonnull client)
      : client_(ABSL_DIE_IF_NULL(client)) {}

  // Return the client that owns this device.
  PjRtClient* client() const override { return client_; }

  bool IsAddressable() const override { return true; }

  const InterpreterDescription& description() const override {
    return InterpreterDescription::Singleton();
  }

  LocalDeviceId local_device_id() const override { return LocalDeviceId(0); }

  LocalChipId local_hardware_id() const override { return LocalChipId(0); }

  std::unique_ptr<ScopedAsyncTrackingEvent> CreateAsyncTrackingEvent(
      absl::string_view description) const override {
    return nullptr;
  }

  absl::Status TransferToInfeed(const LiteralSlice& literal) override {
    return Unimplemented("Interpreter does not support transfer to infeed.");
  }

  absl::Status TransferFromOutfeed(MutableBorrowingLiteral literal) override {
    return Unimplemented("Interpreter does not support transfer from outfeed.");
  }

  absl::Span<PjRtMemorySpace* const> memory_spaces() const override {
    return client_->memory_spaces();
  }

  absl::StatusOr<PjRtMemorySpace*> memory_space_by_kind(
      absl::string_view memory_space_kind) const override {
    // TODO(slebedev): Consider returning a memory space with the given kind.
    return default_memory_space();
  }

  absl::StatusOr<PjRtMemorySpace*> default_memory_space() const override {
    return client_->memory_spaces().front();
  }

 private:
  PjRtClient* client_ = nullptr;
};

// A buffer that wraps a Literal.
class InterpreterLiteralWrapperBuffer final : public PjRtBuffer {
 public:
  InterpreterLiteralWrapperBuffer(PjRtClient* absl_nonnull client,
                                  PjRtMemorySpace* absl_nonnull memory_space,
                                  const LiteralSlice& literal)
      : client_(client),
        memory_space_(memory_space),
        literal_(literal.Clone()) {}
  InterpreterLiteralWrapperBuffer(PjRtClient* absl_nonnull client,
                                  PjRtMemorySpace* absl_nonnull memory_space,
                                  Literal literal)
      : client_(client),
        memory_space_(memory_space),
        literal_(std::move(literal)) {}

  const Shape& on_device_shape() const override { return literal_.shape(); }

  PjRtMemorySpace* memory_space() const override { return memory_space_; }

  PjRtDevice* device() const override {
    if (memory_space_ != nullptr && !memory_space_->devices().empty()) {
      return memory_space_->devices().front();
    }
    if (client_ != nullptr && !client_->devices().empty()) {
      return client_->devices().front();
    }
    return nullptr;
  }

  PjRtClient* client() const override { return client_; }

  absl::StatusOr<std::unique_ptr<ExternalReference>> AcquireExternalReference()
      override {
    return absl::UnimplementedError(
        "AcquireExternalReference not supported by "
        "InterpreterLiteralWrapperBuffer.");
  }

  Future<> ToLiteral(MutableLiteralBase* literal) override {
    return Future<>(ShapeUtil::ForEachSubshapeWithStatus(
        literal_.shape(),
        [&](const Shape& subshape, const ShapeIndex& index) -> absl::Status {
          if (!subshape.IsArray()) {
            return absl::OkStatus();
          }
          const int64_t src_size = literal_.size_bytes(index);
          const int64_t dst_size = literal->size_bytes(index);
          if (src_size != dst_size) {
            return absl::FailedPreconditionError(absl::StrFormat(
                "Cannot copy between buffers of different sizes: "
                "Source size is %d bytes, "
                "destination size is %d bytes.",
                src_size, dst_size));
          }
          std::memcpy(/*dst=*/literal->untyped_data(index),
                      /*src=*/literal_.untyped_data(index), dst_size);
          return absl::OkStatus();
        }));
  }

  Future<> LazyToLiteral(
      absl::AnyInvocable<Future<MutableLiteralBase*>() &&> generator) override {
    // Underlying buffer is always ready, so we can immediately call the
    // generator.
    Future<MutableLiteralBase*> future = std::move(generator)();
    const absl::StatusOr<MutableLiteralBase*>& literal = future.Await();
    if (!literal.ok()) {
      return Future<>(literal.status());
    }
    return ToLiteral(*literal);
  }

  absl::StatusOr<size_t> GetOnDeviceSizeInBytes() const override {
    return literal_.size_bytes();
  }

  Future<> CopyRawToHost(void* dst, int64_t offset,
                         int64_t transfer_size) override {
    return Future<>(absl::UnimplementedError(
        "CopyRawToHost not supported by InterpreterLiteralWrapperBuffer."));
  }

  void Delete() override {
    // Delete does not need to do anything for this type of buffer.
    //
    // This buffer does not support ownership transfers of the underlying
    // buffer. The buffer memory is owned by the Literal field, deleted when
    // this buffer's object is deleted.
    is_deleted_ = true;
  }

  absl::StatusOr<std::unique_ptr<ExternalReference>>
  ReleaseDeviceMemoryOwnership(bool wait_for_operations_to_complete) override {
    return absl::UnimplementedError(
        "ReleaseDeviceMemoryOwnership not supported by "
        "InterpreterLiteralWrapperBuffer.");
  }

  bool IsDeleted() const override { return is_deleted_; }

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CopyToMemorySpace(
      PjRtMemorySpace* dst_memory_space) override {
    return absl::UnimplementedError(
        "CopyToMemorySpace not supported by "
        "InterpreterLiteralWrapperBuffer.");
  }

  void CopyToRemoteDevice(Future<std::string> serialized_descriptor,
                          RemoteSendCallback on_done) override {
    LOG(ERROR) << "InterpreterLiteralWrapperBuffer::CopyToRemoteDevice was "
                  "called but is not implemented.";
  }

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> Bitcast(
      PrimitiveType element_type, absl::Span<const int64_t> dims,
      const Layout* device_layout) override {
    return absl::UnimplementedError(
        "Bitcast not supported by InterpreterLiteralWrapperBuffer.");
  }

  Future<> GetReadyFuture() override { return Future<>(absl::OkStatus()); }

  bool IsOnCpu() const override { return true; }

  const Literal& literal() const { return literal_; }
  Literal& mutable_literal() { return literal_; }

 private:
  PjRtClient* client_ = nullptr;
  PjRtMemorySpace* memory_space_ = nullptr;
  Literal literal_;
  bool is_deleted_ = false;
};

class InterpreterLoadedExecutable final : public PjRtLoadedExecutable {
 public:
  explicit InterpreterLoadedExecutable(
      PjRtClient* absl_nonnull client,
      std::shared_ptr<InterpreterExecutable> executable,
      std::unique_ptr<HloEvaluatorInterface> hlo_evaluator,
      std::shared_ptr<DeviceAssignment> device_assignment,
      std::vector<LogicalDeviceIds> addressable_device_logical_ids,
      std::vector<PjRtDevice*> addressable_devices);

  InterpreterExecutable* GetExecutable() const override;

  int num_replicas() const override;

  int num_partitions() const override;

  int64_t SizeOfGeneratedCodeInBytes() const override { return -1; }

  absl::string_view name() const override;

  absl::StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const override;

  absl::StatusOr<struct CompileOptions> GetCompileOptions() const override;

  PjRtClient* client() const override { return client_; }

  const DeviceAssignment& device_assignment() const override {
    return *device_assignment_;
  }

  absl::Span<const LogicalDeviceIds> addressable_device_logical_ids()
      const override {
    return addressable_device_logical_ids_;
  }

  absl::Span<PjRtDevice* const> addressable_devices() const override {
    return addressable_devices_;
  }

  absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>> Execute(
      absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
      const ExecuteOptions& options,
      std::optional<std::vector<Future<>>>& returned_futures) const override;

  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteSharded(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options, std::optional<Future<>>& returned_future,
      bool fill_future) const override;

  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecutePortable(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options, std::optional<Future<>>& returned_future,
      bool fill_future) const override;

  void Delete() override;

  bool IsDeleted() const override;

 private:
  absl::StatusOr<Literal> Evaluate(
      const HloComputation& computation,
      absl::Span<const Literal* const> arg_literals,
      const ExecuteOptions& options) const ABSL_LOCKS_EXCLUDED(mutex_);

  PjRtClient* client_ = nullptr;
  std::string name_;
  mutable absl::Mutex mutex_;
  std::shared_ptr<InterpreterExecutable> executable_ ABSL_GUARDED_BY(mutex_);
  std::unique_ptr<HloEvaluatorInterface> hlo_evaluator_ ABSL_GUARDED_BY(mutex_);
  std::shared_ptr<DeviceAssignment> device_assignment_;
  std::vector<LogicalDeviceIds> addressable_device_logical_ids_;
  std::vector<PjRtDevice*> addressable_devices_;
};

class InterpreterClient final : public PjRtClient {
 public:
  InterpreterClient();
  explicit InterpreterClient(
      absl::AnyInvocable<std::unique_ptr<HloEvaluatorInterface>() const>
          hlo_evaluator_factory);
  // Not copyable or movable
  InterpreterClient(const InterpreterClient&) = delete;
  InterpreterClient& operator=(const InterpreterClient&) = delete;
  InterpreterClient(InterpreterClient&&) = delete;
  InterpreterClient& operator=(InterpreterClient&&) = delete;

  static Shape DeviceShapeRepresentation(const Shape& shape) { return shape; }

  static int64_t ShapeSizeBytes(const Shape& shape) {
    if (shape.IsOpaque()) {
      return sizeof(void*);
    }
    return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
  }

  int process_index() const override { return 0; }

  int device_count() const override { return devices().size(); }

  int addressable_device_count() const override {
    return addressable_devices().size();
  }

  absl::Span<PjRtDevice* const> devices() const override { return devices_; }

  absl::Span<PjRtDevice* const> addressable_devices() const override {
    return devices_;
  }

  absl::Span<PjRtMemorySpace* const> memory_spaces() const override {
    return memory_spaces_;
  }

  PjRtPlatformId platform_id() const override { return xla::InterpreterId(); }

  absl::string_view platform_name() const override {
    return xla::InterpreterName();
  }

  absl::string_view platform_version() const override {
    return topology_->platform_version();
  }

  std::optional<PjRtPluginAttributes> plugin_attributes() const override;

  absl::StatusOr<const PjRtTopologyDescription*> GetTopologyDescription()
      const override {
    return topology_.get();
  }

  absl::StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;

  absl::StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, std::optional<int> num_replicas_per_slice,
      int num_partitions,
      const MultiSliceConfig* multi_slice_config) const override;

  absl::StatusOr<Layout> GetDefaultLayout(
      PrimitiveType element_type, absl::Span<const int64_t> dims) override;

  absl::StatusOr<std::unique_ptr<HloCostAnalysis>> GetHloCostAnalysis()
      const override {
    return std::make_unique<HloCostAnalysis>(ShapeSizeBytes);
  }

  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      const XlaComputation& computation, CompileOptions options) override;

  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      MaybeOwningMlirModule module, CompileOptions options) override;

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileAndLoad(
      const XlaComputation& computation, CompileOptions options) override;

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileAndLoad(
      MaybeOwningMlirModule module, CompileOptions options) override;

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Load(
      std::shared_ptr<PjRtExecutable> executable,
      const LoadOptions& load_options) override;

  absl::StatusOr<std::unique_ptr<PjRtExecutable>> DeserializeExecutable(
      absl::string_view serialized,
      std::optional<CompileOptions>&& options) override;

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
  LoadSerializedExecutable(absl::string_view serialized,
                           std::optional<CompileOptions> options,
                           const LoadOptions& load_options) override;

  using PjRtClient::BufferFromHostLiteral;
  absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostLiteral(
      const LiteralSlice& literal, PjRtMemorySpace* memory_space,
      const Layout* device_layout) override;

  absl::StatusOr<PjRtDevice*> LookupDevice(
      GlobalDeviceId global_device_id) const override;

  absl::StatusOr<PjRtDevice*> LookupAddressableDevice(
      LocalDeviceId local_device_id) const override;

 private:
  absl::AnyInvocable<std::unique_ptr<HloEvaluatorInterface>() const>
      hlo_evaluator_factory_;
  std::unique_ptr<InterpreterTopologyDescription> topology_;
  InterpreterDevice interpreter_device_;
  InterpreterMemorySpace interpreter_memory_space_;
  // Pointer array of devices (just one) so that we can create a span of it.
  // Similarly for memory spaces.
  std::array<PjRtDevice*, 1> devices_;
  std::array<PjRtMemorySpace*, 1> memory_spaces_;
};

}  // namespace xla

#endif  // XLA_PJRT_INTERPRETER_INTERPRETER_CLIENT_H_
