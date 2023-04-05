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
#include "tfrt/concurrency/ref_count.h"  // from @tf_runtime

namespace xla {
namespace ifrt {

// Converts IFRT `DType` into `xla::PrimitiveType`.
StatusOr<xla::PrimitiveType> ToPrimitiveType(DType dtype);

// Converts `xla::PrimitiveType` into IFRT `DType`.
StatusOr<DType> ToDType(xla::PrimitiveType primitive_type);

// PjRt-compatible `Array` interface that wraps a list of `xla::PjRtBuffer`s.
class PjRtCompatibleArray
    : public llvm::RTTIExtends<PjRtCompatibleArray, Array> {
 public:
  // APIs that allow direct access to `PjRtBuffer` for PjRt-only operations.
  virtual absl::Span<const std::shared_ptr<PjRtBuffer>> pjrt_buffers() = 0;
  virtual StatusOr<absl::Span<std::shared_ptr<PjRtBuffer>>>
  mutable_pjrt_buffers() = 0;

  static char ID;  // NOLINT
};

// `Array` implementation that wraps a list of `xla::PjRtBuffer`s.
class PjRtArray final
    : public llvm::RTTIExtends<PjRtArray, PjRtCompatibleArray> {
 public:
  static constexpr int kPjRtBufferInlineSize = 1;
  using PjRtBuffers =
      absl::InlinedVector<std::shared_ptr<PjRtBuffer>, kPjRtBufferInlineSize>;

  // General array construction.
  static StatusOr<tsl::RCReference<PjRtArray>> Create(
      PjRtCompatibleClient* client, DType dtype, Shape shape,
      std::shared_ptr<const Sharding> sharding, PjRtBuffers pjrt_buffers);

  // Shorthand for a single-shard array construction.
  static StatusOr<tsl::RCReference<PjRtArray>> Create(
      PjRtCompatibleClient* client, std::shared_ptr<PjRtBuffer> pjrt_buffer);

  // Shorthand for a multi-shard array construction using OpaqueSharding.
  // TODO(hyeontaek): Remove this once IFRT Sharding and JAX Sharding is unified
  // so that OpaqueSharding can be replaced with a real Sharding.
  static StatusOr<tsl::RCReference<PjRtArray>> Create(
      PjRtCompatibleClient* client, Shape shape, PjRtBuffers pjrt_buffers);

  // PjRtCompatibleArray implementation.

  absl::Span<const std::shared_ptr<PjRtBuffer>> pjrt_buffers() override {
    DCHECK(this);
    return pjrt_buffers_;
  }
  StatusOr<absl::Span<std::shared_ptr<PjRtBuffer>>> mutable_pjrt_buffers()
      override {
    DCHECK(this);
    return absl::MakeSpan(pjrt_buffers_);
  }

  // Array implementation.

  ~PjRtArray() override = default;

  PjRtCompatibleClient* client() const override {
    DCHECK(this);
    return client_;
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

  StatusOr<std::vector<tsl::RCReference<Array>>>
  DisassembleIntoSingleDeviceArrays(ArrayCopySemantics semantics) override;

  ABSL_MUST_USE_RESULT
  Future<Status> CopyToHostBuffer(
      void* data, std::optional<absl::Span<const int64_t>> byte_strides,
      ArrayCopySemantics semantics) override;

  StatusOr<tsl::RCReference<Array>> Reshard(
      std::shared_ptr<const Sharding> new_sharding,
      ArrayCopySemantics semantics) override;

  Future<Status> GetReadyFuture() const override;

  Future<Status> Delete() override;
  bool IsDeleted() const override;

  std::string DebugString() const override;

  static char ID;  // NOLINT

 private:
  PjRtArray(PjRtCompatibleClient* client, DType dtype, Shape shape,
            std::shared_ptr<const Sharding> sharding, PjRtBuffers pjrt_buffers);

  template <typename T, typename... Args>
  friend tsl::RCReference<T> tsl::MakeRef(Args&&... args);

  PjRtCompatibleClient* client_;
  DType dtype_;
  Shape shape_;
  std::shared_ptr<const Sharding> sharding_;
  PjRtBuffers pjrt_buffers_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_PJRT_IFRT_PJRT_ARRAY_H_
