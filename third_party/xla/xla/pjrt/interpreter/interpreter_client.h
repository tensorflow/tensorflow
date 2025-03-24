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
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/die_if_null.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/service/computation_placer.h"
#include "xla/service/dynamic_dimension_inference.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/fingerprint.h"

namespace xla {

class InterpreterDescription final : public PjRtDeviceDescription {
 public:
  static const InterpreterDescription& Singleton();

  int id() const override { return 0; }

  int process_index() const override { return 0; }

  absl::string_view device_kind() const override { return "interpreter"; }

  absl::string_view DebugString() const override { return "interpreter:0"; }

  absl::string_view ToString() const override {
    return "InterpreterDevice(id=0)";
  }

  const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& Attributes()
      const override {
    return attributes_;
  }

 private:
  InterpreterDescription() = default;
  absl::flat_hash_map<std::string, PjRtDeviceAttribute> attributes_;
};

class InterpreterMemorySpace final : public PjRtMemorySpace {
 public:
  explicit InterpreterMemorySpace(absl::Nonnull<PjRtClient*> client)
      : client_(ABSL_DIE_IF_NULL(client)) {}

  PjRtClient* client() const override { return client_; }

  absl::Span<PjRtDevice* const> devices() const override {
    return client_->devices();
  }

  int id() const override { return 0; };

  absl::string_view kind() const override { return "interpreter"; };

  int kind_id() const override { return 0; };

  absl::string_view DebugString() const override { return "interpreter:0"; }

  absl::string_view ToString() const override {
    return "InterpreterMemorySpace(id=0)";
  }

 private:
  PjRtClient* client_ = nullptr;
};

class InterpreterDevice final : public PjRtDevice {
 public:
  explicit InterpreterDevice(absl::Nonnull<PjRtClient*> client)
      : client_(ABSL_DIE_IF_NULL(client)) {}

  // Return the client that owns this device.
  PjRtClient* client() const override { return client_; }

  bool IsAddressable() const override { return true; };

  const InterpreterDescription& description() const override {
    return InterpreterDescription::Singleton();
  }

  PjRtLocalDeviceId local_device_id() const override {
    return PjRtLocalDeviceId(0);
  }

  PjRtLocalHardwareId local_hardware_id() const override {
    return PjRtLocalHardwareId(0);
  }

  std::unique_ptr<ScopedAsyncTrackingEvent> CreateAsyncTrackingEvent(
      absl::string_view description) const override {
    return nullptr;
  }

  absl::Status TransferToInfeed(const LiteralSlice& literal) override {
    return Unimplemented("Interpreter does not suppot transfer to infeed.");
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
  InterpreterLiteralWrapperBuffer(absl::Nonnull<PjRtClient*> client,
                                  absl::Nonnull<PjRtMemorySpace*> memory_space,
                                  const LiteralSlice& literal)
      : client_(client),
        memory_space_(memory_space),
        literal_(literal.Clone()) {}
  InterpreterLiteralWrapperBuffer(absl::Nonnull<PjRtClient*> client,
                                  absl::Nonnull<PjRtMemorySpace*> memory_space,
                                  Literal literal)
      : client_(client),
        memory_space_(memory_space),
        literal_(std::move(literal)) {}

  const Shape& on_device_shape() const override { return literal_.shape(); }

  PjRtMemorySpace* memory_space() const override { return memory_space_; }

  PjRtDevice* device() const override { return nullptr; }

  PjRtClient* client() const override { return client_; }

  absl::StatusOr<std::unique_ptr<ExternalReference>> AcquireExternalReference()
      override {
    return absl::UnimplementedError(
        "AcquireExternalReference not supported by "
        "InterpreterLiteralWrapperBuffer.");
  }

  PjRtFuture<> ToLiteral(MutableLiteralBase* literal) override {
    return PjRtFuture<>(ShapeUtil::ForEachSubshapeWithStatus(
        literal_.shape(),
        [&](const Shape& subshape, const ShapeIndex& index) -> absl::Status {
          if (!subshape.IsArray()) {
            return absl::OkStatus();
          }
          const int64_t src_size = literal_.size_bytes(index);
          const int64_t dst_size = literal->size_bytes(index);
          if (src_size < dst_size) {
            return absl::FailedPreconditionError(
                absl::StrFormat("Cannot copy more data than available: Tried "
                                "to copy %d bytes, "
                                "but only %d bytes are available (%d < %d).",
                                dst_size, src_size, src_size, dst_size));
          }
          std::memcpy(/*dst=*/literal->untyped_data(index),
                      /*src=*/literal_.untyped_data(index), dst_size);
          return absl::OkStatus();
        }));
  }

  PjRtFuture<> LazyToLiteral(
      absl::AnyInvocable<absl::StatusOr<MutableLiteralBase*>() &&> generator)
      override {
    // Underlying buffer is always ready, so we can immediately call the
    // generator.
    absl::StatusOr<MutableLiteralBase*> literal = std::move(generator)();
    if (!literal.ok()) {
      return PjRtFuture<>(literal.status());
    }
    return ToLiteral(*literal);
  }

  absl::StatusOr<size_t> GetOnDeviceSizeInBytes() const override {
    return literal_.size_bytes();
  }

  PjRtFuture<> CopyRawToHost(void* dst, int64_t offset,
                             int64_t transfer_size) override {
    return PjRtFuture<>(absl::UnimplementedError(
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

  bool IsDeleted() override { return is_deleted_; }

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CopyToMemorySpace(
      PjRtMemorySpace* dst_memory_space) override {
    return absl::UnimplementedError(
        "CopyToMemorySpace not supported by "
        "InterpreterLiteralWrapperBuffer.");
  }

  void CopyToRemoteDevice(PjRtFuture<std::string> serialized_descriptor,
                          RemoteSendCallback on_done) override {
    LOG(ERROR) << "InterpreterLiteralWrapperBuffer::CopyToRemoteDevice was "
                  "called but is not implemented.";
  }

  PjRtFuture<> GetReadyFuture() override {
    return PjRtFuture<>(absl::OkStatus());
  }

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
      absl::Nonnull<PjRtClient*> client, std::unique_ptr<HloModule> hlo_module,
      std::unique_ptr<HloEvaluator> hlo_evaluator,
      std::optional<DynamicDimensionInference> dynamic_dimension_inference,
      std::shared_ptr<DeviceAssignment> device_assignment,
      CompileOptions compile_options,
      std::vector<LogicalDeviceIds> addressable_device_logical_ids,
      std::vector<PjRtDevice*> addressable_devices)
      : client_(ABSL_DIE_IF_NULL(client)),
        hlo_module_(std::move(hlo_module)),
        hlo_evaluator_(std::move(hlo_evaluator)),
        dynamic_dimension_inference_(std::move(dynamic_dimension_inference)),
        device_assignment_(std::move(device_assignment)),
        compile_options_(std::move(compile_options)),
        addressable_device_logical_ids_(
            std::move(addressable_device_logical_ids)),
        addressable_devices_(std::move(addressable_devices)) {
    if (dynamic_dimension_inference_.has_value()) {
      hlo_evaluator_->set_dynamic_dimension_inference(
          &dynamic_dimension_inference_.value());
    }
  }

  int num_replicas() const override {
    return hlo_module_->config().replica_count();
  }

  int num_partitions() const override {
    return hlo_module_->config().num_partitions();
  }

  int64_t SizeOfGeneratedCodeInBytes() const override { return -1; }

  absl::string_view name() const override { return hlo_module_->name(); }

  absl::StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const override {
    std::vector<std::shared_ptr<HloModule>> hlo_modules;
    hlo_modules.push_back(hlo_module_);
    return hlo_modules;
  }

  absl::StatusOr<std::vector<std::vector<absl::string_view>>>
  GetOutputMemoryKinds() const override {
    return absl::UnimplementedError("GetOutputMemoryKinds is not supported.");
  }

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
      std::optional<std::vector<PjRtFuture<>>>& returned_futures) override;

  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteSharded(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      std::optional<PjRtFuture<>>& returned_future, bool fill_future) override;

  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecutePortable(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      std::optional<PjRtFuture<>>& returned_future, bool fill_future) override;

  void Delete() override { hlo_module_ = nullptr; }

  bool IsDeleted() override { return hlo_module_ == nullptr; }

 private:
  absl::StatusOr<Literal> Evaluate(
      const HloComputation& computation,
      absl::Span<const Literal* const> arg_literals)
      ABSL_LOCKS_EXCLUDED(hlo_evaluator_lock_);

  PjRtClient* client_ = nullptr;
  std::shared_ptr<HloModule> hlo_module_;
  mutable absl::Mutex hlo_evaluator_lock_;
  std::unique_ptr<HloEvaluator> hlo_evaluator_
      ABSL_PT_GUARDED_BY(hlo_evaluator_lock_);
  std::optional<DynamicDimensionInference> dynamic_dimension_inference_;
  std::shared_ptr<DeviceAssignment> device_assignment_;
  CompileOptions compile_options_;
  std::vector<LogicalDeviceIds> addressable_device_logical_ids_;
  std::vector<PjRtDevice*> addressable_devices_;
};

class InterpreterClient final : public PjRtClient {
 public:
  InterpreterClient()
      : InterpreterClient([]() { return std::make_unique<HloEvaluator>(); }) {}
  explicit InterpreterClient(
      absl::AnyInvocable<std::unique_ptr<HloEvaluator>() const>
          hlo_evaluator_factory)
      : hlo_evaluator_factory_(std::move(hlo_evaluator_factory)),
        interpreter_device_{this},
        interpreter_memory_space_{this},
        devices_({&interpreter_device_}),
        memory_spaces_({&interpreter_memory_space_}) {}
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

  PjRtPlatformId platform_id() const override {
    static const PjRtPlatformId kPlatformId = tsl::Fingerprint64("interpreter");
    return kPlatformId;
  }

  absl::string_view platform_name() const override { return "interpreter"; }

  absl::string_view platform_version() const override { return "<unknown>"; }

  absl::StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;

  absl::StatusOr<Layout> GetDefaultLayout(
      PrimitiveType element_type, absl::Span<const int64_t> dims) override;

  absl::StatusOr<std::unique_ptr<HloCostAnalysis>> GetHloCostAnalysis()
      const override {
    return std::make_unique<HloCostAnalysis>(ShapeSizeBytes);
  }

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileAndLoad(
      const XlaComputation& computation, CompileOptions options) override;

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileAndLoad(
      mlir::ModuleOp module, CompileOptions options) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostLiteral(
      const LiteralSlice& literal, PjRtMemorySpace* memory_space) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostLiteral(
      const LiteralSlice& literal, PjRtMemorySpace* memory_space,
      const Layout* device_layout) override;

 private:
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileInternal(
      const XlaComputation& computation,
      const std::vector<const Shape*>& argument_shapes,
      LayoutCanonicalizationCallback layout_canonicalization_callback,
      CompileOptions options);
  absl::StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> hlo_module);
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> RunBackend(
      std::unique_ptr<HloModule> hlo_module, CompileOptions& options);

  absl::AnyInvocable<std::unique_ptr<HloEvaluator>() const>
      hlo_evaluator_factory_;
  InterpreterDevice interpreter_device_;
  InterpreterMemorySpace interpreter_memory_space_;
  // Pointer array of devices (just one) so that we can create a span of it.
  // Similarly for memory spaces.
  std::array<PjRtDevice*, 1> devices_;
  std::array<PjRtMemorySpace*, 1> memory_spaces_;
};
}  // namespace xla

#endif  // XLA_PJRT_INTERPRETER_INTERPRETER_CLIENT_H_
