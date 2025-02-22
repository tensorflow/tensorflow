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

#include "xla/pjrt/plugin/stablehlo_reference/buffer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/DebugStringHelper.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_utils.h"
#include "xla/hlo/translate/mhlo_to_hlo/literal_exporter.h"
#include "xla/hlo/translate/mhlo_to_hlo/type_to_shape.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/plugin/stablehlo_reference/logging.h"
#include "xla/shape.h"
#include "xla/util.h"

namespace mlir::stablehlo {

using xla::MutableLiteralBase;
using xla::PjRtBuffer;
using xla::PjRtClient;
using xla::PjRtDevice;
using xla::PjRtFuture;
using xla::PjRtMemorySpace;
using xla::PjRtPlatformId;
using xla::Shape;

#define UNIMPLEMENTED(name) \
  xla::Unimplemented("MlirPjrtBuffer::" #name " is not implemented")

class MlirPjrtBuffer : public PjRtBuffer {
 public:
  MlirPjrtBuffer(const Shape& shape, PjRtMemorySpace* memory_space)
      : xla::PjRtBuffer(),
        context_(),
        buffer_(),
        shape_(shape),
        memory_space_(memory_space) {
    TRACE_ME_MEMBER;
  }

  class MlirClonedExternalReference : public ExternalReference {
   public:
    explicit MlirClonedExternalReference(PjRtBuffer* buffer,
                                         PjRtMemorySpace* memory_space)
        : buffer_() {
      TRACE_ME_MEMBER;
      auto mlir_buffer = GetAttributeFromBuffer(buffer);
      if (!mlir_buffer.ok()) {
        LOG(ERROR) << "Could not get attribute from buffer: "
                   << mlir_buffer.status();
      }
      buffer_ =
          CreateMlirBufferFromAttribute(mlir_buffer.value(), memory_space);
      data_ptr_ = (void*)mlir_buffer.value().getRawData().data();
    }

   private:
    std::unique_ptr<PjRtBuffer> buffer_;
  };

  // All buffers are managed by the MLIR Context
  ~MlirPjrtBuffer() override = default;

  MlirPjrtBuffer(const MlirPjrtBuffer&) = delete;
  MlirPjrtBuffer(MlirPjrtBuffer&&) = delete;
  MlirPjrtBuffer& operator=(const MlirPjrtBuffer&) = delete;
  MlirPjrtBuffer& operator=(MlirPjrtBuffer&&) = delete;

  static std::unique_ptr<MlirPjrtBuffer> CreateFromLiteral(
      const xla::LiteralSlice& literal, xla::PjRtMemorySpace* memory_space) {
    TRACE_ME;
    LOG(INFO) << "CreateFromLiteral: " << literal.ToString() << "\n";
    auto buffer =
        std::make_unique<MlirPjrtBuffer>(literal.shape(), memory_space);
    LOG(INFO) << "CreateFromLiteral -> " << (void*)buffer.get() << "\n";
    mlir::Builder builder(&buffer->context_);
    auto attr = xla::CreateDenseElementsAttrFromLiteral(literal, builder);
    if (!attr.ok()) {
      LOG(ERROR) << "Could not create dense elements attr from literal: "
                 << attr.status();
      return nullptr;
    }
    buffer->buffer_ = attr.value();
    return buffer;
  }

  static std::unique_ptr<MlirPjrtBuffer> CreateFromAttribute(
      DenseElementsAttr attr, xla::PjRtMemorySpace* memory_space) {
    TRACE_ME;

    // MLIR type to xla shape:
    Shape shape = xla::TypeToShape(attr.getType());
    auto buffer = std::make_unique<MlirPjrtBuffer>(shape, memory_space);
    buffer->buffer_ = CloneIntoContext(attr, buffer->context_);
    LOG(INFO) << "CreateFromAttribute(" << ToString(attr) << ") -> "
              << (void*)buffer.get() << "\n";
    return buffer;
  }

  const Shape& on_device_shape() const override {
    TRACE_ME_MEMBER;
    return shape_;
  }
  absl::StatusOr<Shape> logical_on_device_shape() override {
    TRACE_ME_MEMBER;
    return shape_;
  }

  PjRtPlatformId platform_id() const {
    TRACE_ME_MEMBER;
    return client()->platform_id();
  }
  absl::string_view platform_name() const {
    TRACE_ME_MEMBER;
    return client()->platform_name();
  }

  bool IsEmptyTuple() const {
    TRACE_ME_MEMBER;
    return shape_.IsTuple() && shape_.tuple_shapes().empty();
  }

  // Buffer knows device + client per older design, should only need
  // memory_space.
  PjRtMemorySpace* memory_space() const override {
    TRACE_ME_MEMBER;
    return memory_space_;
  }
  PjRtDevice* device() const override {
    TRACE_ME_MEMBER;
    return memory_space_->devices().front();
  }
  PjRtClient* client() const override {
    TRACE_ME_MEMBER;
    return memory_space_->client();
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer::ExternalReference>>
  AcquireExternalReference() override {
    TRACE_ME_MEMBER;
    return std::make_unique<MlirClonedExternalReference>(this, memory_space_);
  }

  xla::PjRtFuture<> ToLiteral(xla::MutableLiteralBase* literal) override {
    TRACE_ME_MEMBER;
    if (IsEmptyTuple()) {
      return PjRtFuture<>(
          xla::InvalidArgument("ToLiteral called on empty tuple"));
    }

    absl::StatusOr<xla::Literal> to_copy =
        mhlo::CreateLiteralFromAttribute(buffer_, {});
    if (!to_copy.ok()) return PjRtFuture<>(to_copy.status());

    // Synchronous! To make async, make the buffer, start the copy, and return a
    // future that is ready when the copy is done.
    auto status = literal->CopyFrom(to_copy.value());
    if (!status.ok()) return PjRtFuture<>(status);
    return PjRtFuture<>(absl::OkStatus());
  }

  PjRtFuture<> LazyToLiteral(
      absl::AnyInvocable<absl::StatusOr<MutableLiteralBase*>() &&> generator)
      override {
    TRACE_ME_MEMBER;
    auto buffer = std::move(generator)();
    if (!buffer.ok()) return PjRtFuture<>(buffer.status());
    return ToLiteral(buffer.value());
  }

  absl::StatusOr<size_t> GetOnDeviceSizeInBytes() const override {
    // This is needed by AcquireExternalReference, for framework figuring out
    // how to read the underlying buffer data.
    TRACE_ME_MEMBER;
    if (!buffer_) return 0;
    return buffer_.getRawData().size();
  }

  PjRtFuture<> CopyRawToHost(void* dst, int64_t offset,
                             int64_t transfer_size) override {
    TRACE_ME_MEMBER;
    return PjRtFuture<>(UNIMPLEMENTED(CopyRawToHost));
  }

  absl::StatusOr<std::unique_ptr<ExternalReference>>
  ReleaseDeviceMemoryOwnership(bool wait_for_operations_to_complete) override {
    TRACE_ME_MEMBER;
    auto external_ref = AcquireExternalReference();
    Delete();
    return external_ref;
  }

  // Remove the buffer if deleted.
  // Note: deleted and uninitialized appear the same in this scenario.
  // Consider changing to mlir::NoneType when deleted.
  void Delete() override {
    TRACE_ME_MEMBER;
    buffer_ = {};
  }

  bool IsDeleted() override {
    TRACE_ME_MEMBER;
    return !buffer_;
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> CopyToDevice(
      xla::PjRtDevice* dst_device) override {
    TRACE_ME_MEMBER;
    return CopyToMemorySpace(
        dst_device->default_memory_space().value_or(nullptr));
  }

  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> CopyToMemorySpace(
      xla::PjRtMemorySpace* dst_memory_space) override {
    TRACE_ME_MEMBER;
    return CreateMlirBufferFromAttribute(buffer_, dst_memory_space);
  }

  void CopyToRemoteDevice(
      xla::PjRtFuture<std::string> serialized_descriptor,
      xla::PjRtBuffer::RemoteSendCallback on_done) override {
    TRACE_ME_MEMBER;
    on_done(UNIMPLEMENTED(CopyToRemoteDevice), false);
  }

  void CopyToRemoteDeviceScattered(
      xla::PjRtFuture<std::vector<std::string>> serialized_descriptors,
      std::vector<RemoteSendCallback> callbacks,
      const xla::PjRtBuffer::ScatterDetails& scatter_details) override {
    TRACE_ME_MEMBER;
    for (auto cb : callbacks) {
      cb(UNIMPLEMENTED(CopyToRemoteDeviceScattered), false);
    }
  }

  xla::PjRtFuture<> GetReadyFuture() override {
    TRACE_ME_MEMBER;
    LOG(INFO) << "GetReadyFuture(" << (void*)this << ")\n";
    // Synchronous! To make async, have the device make a buffer with a ready
    // future that is ready when the computation is done / buffer is ready.
    return PjRtFuture<>(absl::OkStatus());
  }

  bool IsOnCpu() const override {
    // If buffer is on CPU, it will be shared with framework via
    // GetExternalReference, lse it is copied back to host.
    // Since we are using reference interpreter, we are running on CPU in a
    // shared memory space.
    TRACE_ME_MEMBER;
    return false;
  }

  mlir::DenseElementsAttr GetBufferAttribute() const { return buffer_; }

 private:
  MLIRContext context_;
  mlir::DenseElementsAttr buffer_;

  xla::Shape shape_;
  PjRtMemorySpace* memory_space_;
};

std::unique_ptr<PjRtBuffer> CreateMlirBufferFromLiteral(
    const xla::LiteralSlice& literal, xla::PjRtMemorySpace* memory_space) {
  TRACE_ME;
  return MlirPjrtBuffer::CreateFromLiteral(literal, memory_space);
}

std::unique_ptr<PjRtBuffer> CreateMlirBufferFromAttribute(
    DenseElementsAttr attr, xla::PjRtMemorySpace* memory_space) {
  TRACE_ME;
  return MlirPjrtBuffer::CreateFromAttribute(attr, memory_space);
}

std::unique_ptr<PjRtBuffer> CreateMlirBufferUninitizlied(
    const Shape& shape, PjRtMemorySpace* memory_space) {
  TRACE_ME;
  return std::make_unique<MlirPjrtBuffer>(shape, memory_space);
}

absl::StatusOr<mlir::DenseElementsAttr> GetAttributeFromBuffer(
    xla::PjRtBuffer* buffer) {
  TRACE_ME;
  if (buffer == nullptr || buffer->IsDeleted()) {
    return xla::InvalidArgument("Buffer is null or deleted");
  }
  auto mlir_buffer = dynamic_cast<MlirPjrtBuffer*>(buffer);
  if (mlir_buffer == nullptr) {
    return xla::InvalidArgument("Buffer is not a MlirPjrtBuffer");
  }
  LOG(INFO) << "GetAttributeFromBuffer(" << (void*)buffer << ") -> "
            << ToString(mlir_buffer->GetBufferAttribute()) << "\n";
  return mlir_buffer->GetBufferAttribute();
}

DenseElementsAttr CloneIntoContext(DenseElementsAttr attr,
                                   MLIRContext& context) {
  Type type = mlir::parseType(mlir::debugString(attr.getType()), &context);
  return DenseElementsAttr::getFromRawBuffer(llvm::cast<ShapedType>(type),
                                             attr.getRawData());
}

}  // namespace mlir::stablehlo
