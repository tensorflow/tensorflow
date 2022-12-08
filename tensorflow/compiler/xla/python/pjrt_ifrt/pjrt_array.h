/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_PJRT_IFRT_PJRT_ARRAY_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_PJRT_IFRT_PJRT_ARRAY_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "tensorflow/compiler/xla/python/ifrt/array.h"
#include "tensorflow/compiler/xla/python/ifrt/client.h"
#include "tensorflow/compiler/xla/python/pjrt_ifrt/pjrt_client.h"

namespace xla {
namespace ifrt {

// Converts IFRT `DType` into `xla::PrimitiveType`.
StatusOr<xla::PrimitiveType> ToPrimitiveType(DType dtype);

// Converts `xla::PrimitiveType` into IFRT `DType`.
StatusOr<DType> ToDType(xla::PrimitiveType primitive_type);

// `Array` implementation that wraps a list of `xla::PjRtBuffer`s.
class PjRtArray final : public llvm::RTTIExtends<PjRtArray, Array> {
 public:
  static constexpr int kPjRtBufferInlineSize = 1;
  using PjRtBuffers =
      absl::InlinedVector<std::shared_ptr<PjRtBuffer>, kPjRtBufferInlineSize>;

  // General array construction.
  static StatusOr<std::unique_ptr<Array>> Create(
      Client* client, DType dtype, Shape shape,
      std::shared_ptr<const Sharding> sharding, PjRtBuffers pjrt_buffers);

  // Shorthand for a single-shard array construction.
  static StatusOr<std::unique_ptr<Array>> Create(
      Client* client, std::shared_ptr<PjRtBuffer> pjrt_buffer);
  static StatusOr<std::unique_ptr<Array>> Create(
      Client* client, std::unique_ptr<PjRtBuffer> pjrt_buffer);

  // Shorthand for a multi-shard array construction using OpaqueSharding.
  // TODO(hyeontaek): Remove this once IFRT Sharding and JAX Sharding is unified
  // so that OpaqueSharding can be replaced with a real Sharding.
  static StatusOr<std::unique_ptr<Array>> Create(Client* client, Shape shape,
                                                 PjRtBuffers pjrt_buffers);

  absl::Span<const std::shared_ptr<PjRtBuffer>> pjrt_buffers() const {
    DCHECK(this);
    return pjrt_buffers_;
  }
  absl::Span<std::shared_ptr<PjRtBuffer>> pjrt_buffers() {
    DCHECK(this);
    return absl::MakeSpan(pjrt_buffers_);
  }
  PjRtBuffer* pjrt_buffer(int device_id) const {
    DCHECK(this);
    return pjrt_buffers_[device_id].get();
  }

  // Array implementation.

  ~PjRtArray() override = default;

  Client* client() const override {
    DCHECK(this);
    return const_cast<PjRtClient*>(client_);
  }

  DType dtype() const override {
    DCHECK(this);
    return dtype_;
  }
  const Shape& shape() const override {
    DCHECK(this);
    return shape_;
  }
  const Sharding& sharding() const override {
    DCHECK(this);
    return *sharding_;
  }
  std::shared_ptr<const Sharding> shared_ptr_sharding() const override {
    DCHECK(this);
    return sharding_;
  }

  StatusOr<std::vector<std::unique_ptr<Array>>>
  DisassembleIntoSingleDeviceArrays(ArrayCopySemantics semantics) override;

  ABSL_MUST_USE_RESULT
  Future<Status> CopyToHostBuffer(
      void* data, std::optional<absl::Span<const int64_t>> byte_strides,
      ArrayCopySemantics semantics) override;

  StatusOr<std::unique_ptr<Array>> Reshard(
      std::shared_ptr<const Sharding> new_sharding,
      ArrayCopySemantics semantics) override;

  Future<Status> GetReadyFuture() const override;

  Future<Status> Delete() override;
  bool IsDeleted() const override;

  std::string DebugString() const override;

  static char ID;  // NOLINT

 private:
  PjRtArray(PjRtClient* client, DType dtype, Shape shape,
            std::shared_ptr<const Sharding> sharding, PjRtBuffers pjrt_buffers);

  PjRtClient* client_;
  DType dtype_;
  Shape shape_;
  std::shared_ptr<const Sharding> sharding_;
  PjRtBuffers pjrt_buffers_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_PJRT_IFRT_PJRT_ARRAY_H_
