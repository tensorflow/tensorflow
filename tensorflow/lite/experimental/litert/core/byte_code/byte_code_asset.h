// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_BYTE_CODE_BYTE_CODE_ASSET_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_BYTE_CODE_BYTE_CODE_ASSET_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"

////////////////////////////////////////////////////////////////////////////////
// ByteCodeAsset
//
// Byte code assets contain the results of plugins compiling partitions of the
// model. A particular asset may be referenced by one or more ops in the final
// graph. In serial form, byte code assets may require storage outside of the
// ops custom options themselves.
////////////////////////////////////////////////////////////////////////////////

namespace litert::internal {

// Base byte code asset class. All assets pertain to a specific backend and
// contain a single bytecode buffer.
class ByteCodeAsset {
 public:
  using RawByteCode = uint8_t[];

  // Construct a byte code asset for a specific backend.
  explicit ByteCodeAsset(std::string backend_id)
      : backend_id_(std::move(backend_id)) {}

  // Backend id associated with this asset.
  absl::string_view BackendId() const { return backend_id_; }

  // Bytecode buffer associated with this asset.
  virtual BufferRef<uint8_t> ByteCode() const = 0;

  // Serialize this asset into the given model. See derived classes for possible
  // side effects.
  virtual LiteRtStatus Serialize(LiteRtModelT& model) const = 0;

  // TODO also move post process step into this class

  virtual ~ByteCodeAsset() = default;

 private:
  std::string backend_id_;
};

// An asset that is shared by multiple ops.
class SharedByteCodeAsset : public ByteCodeAsset {
 private:
  using Base = ByteCodeAsset;

 public:
  using SharedByteCode = std::shared_ptr<Base::RawByteCode>;
  SharedByteCodeAsset(std::string backend_id, std::string entry_point,
                      SharedByteCode shared_bytecode,
                      size_t shared_bytecode_size);

  // Add an op that will use this byte code asset.
  void AddCaller(LiteRtOp op, std::string entry_point);

  // Serialize this asset into the given model. This sets the custom options of
  // the ops that were added as callers. Also stores the bytecode in the model
  // metadata.
  LiteRtStatus Serialize(LiteRtModelT& model) const override;

  // Get a reference to the bytecode buffer.
  BufferRef<uint8_t> ByteCode() const override {
    return BufferRef<uint8_t>(shared_bytecode_.get(), shared_bytecode_size_);
  }

  SharedByteCodeAsset(const SharedByteCodeAsset&) = delete;
  SharedByteCodeAsset& operator=(const SharedByteCodeAsset&) = delete;
  SharedByteCodeAsset(SharedByteCodeAsset&&) = default;
  SharedByteCodeAsset& operator=(SharedByteCodeAsset&&) = default;

 private:
  std::string MakeMetadataKey() const;

  std::string entry_point_;
  SharedByteCode shared_bytecode_;
  size_t shared_bytecode_size_;
  std::vector<std::pair<LiteRtOp, std::string>> ops_;
};

// An asset that is referenced by only a single op.
class UniqueByteCodeAsset : public ByteCodeAsset {
 private:
  using Base = ByteCodeAsset;

 public:
  using OwnedByteCode = std::unique_ptr<Base::RawByteCode>;

  UniqueByteCodeAsset(std::string backend_id, LiteRtOp op,
                      std::string entry_point, OwnedByteCode owned_bytecode,
                      size_t owned_bytecode_size);

  // Serialize this asset into the given model. This sets the custom options of
  // the op that was passed in the constructor. Also stores the bytecode in the
  // model metadata.
  LiteRtStatus Serialize(LiteRtModelT& model) const override;

  // Get a reference to the bytecode buffer.
  BufferRef<uint8_t> ByteCode() const override {
    return BufferRef<uint8_t>(owned_bytecode_.get(), owned_bytecode_size_);
  }

  UniqueByteCodeAsset(const UniqueByteCodeAsset&) = delete;
  UniqueByteCodeAsset& operator=(const UniqueByteCodeAsset&) = delete;
  UniqueByteCodeAsset(UniqueByteCodeAsset&&) = default;
  UniqueByteCodeAsset& operator=(UniqueByteCodeAsset&&) = default;

 private:
  std::string MakeMetadataKey() const;

  LiteRtOp callers_;
  std::string entry_point_;
  OwnedByteCode owned_bytecode_;
  size_t owned_bytecode_size_;
};

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_BYTE_CODE_BYTE_CODE_ASSET_H_
