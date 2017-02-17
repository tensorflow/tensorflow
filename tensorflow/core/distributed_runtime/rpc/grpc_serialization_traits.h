/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_SERIALIZATION_TRAITS_H_
#define THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_SERIALIZATION_TRAITS_H_

#include "grpc++/impl/codegen/proto_utils.h"

namespace grpc {

namespace tensorflow_helper {

const int kGrpcBufferWriterMaxBufferLength = 8192;

class GrpcBufferWriter GRPC_FINAL
    : public ::grpc::protobuf::io::ZeroCopyOutputStream {
 public:
  explicit GrpcBufferWriter(grpc_byte_buffer** bp, int block_size)
      : block_size_(block_size), byte_count_(0), have_backup_(false) {
    *bp = g_core_codegen_interface->grpc_raw_byte_buffer_create(NULL, 0);
    slice_buffer_ = &(*bp)->data.raw.slice_buffer;
  }

  ~GrpcBufferWriter() GRPC_OVERRIDE {
    if (have_backup_) {
      g_core_codegen_interface->gpr_slice_unref(backup_slice_);
    }
  }

  bool Next(void** data, int* size) GRPC_OVERRIDE {
    if (have_backup_) {
      slice_ = backup_slice_;
      have_backup_ = false;
    } else {
      slice_ = g_core_codegen_interface->gpr_slice_malloc(block_size_);
    }
    *data = GPR_SLICE_START_PTR(slice_);
    // On win x64, int is only 32bit
    GPR_CODEGEN_ASSERT(GPR_SLICE_LENGTH(slice_) <= INT_MAX);
    byte_count_ += * size = (int)GPR_SLICE_LENGTH(slice_);
    g_core_codegen_interface->gpr_slice_buffer_add(slice_buffer_, slice_);
    return true;
  }

  void BackUp(int count) GRPC_OVERRIDE {
    g_core_codegen_interface->gpr_slice_buffer_pop(slice_buffer_);
    if (count == block_size_) {
      backup_slice_ = slice_;
    } else {
      backup_slice_ = g_core_codegen_interface->gpr_slice_split_tail(
          &slice_, GPR_SLICE_LENGTH(slice_) - count);
      g_core_codegen_interface->gpr_slice_buffer_add(slice_buffer_, slice_);
    }
    have_backup_ = true;
    byte_count_ -= count;
  }

  grpc::protobuf::int64 ByteCount() const GRPC_OVERRIDE { return byte_count_; }

 private:
  const int block_size_;
  int64_t byte_count_;
  gpr_slice_buffer* slice_buffer_;
  bool have_backup_;
  gpr_slice backup_slice_;
  gpr_slice slice_;
};

class GrpcBufferReader GRPC_FINAL
    : public ::grpc::protobuf::io::ZeroCopyInputStream {
  typedef void (CoreCodegenInterface::*OldReaderInitAPI)(
      grpc_byte_buffer_reader* reader, grpc_byte_buffer* buffer);
  typedef int (CoreCodegenInterface::*NewReaderInitAPI)(
      grpc_byte_buffer_reader* reader, grpc_byte_buffer* buffer);
  void ReaderInit(OldReaderInitAPI ptr, grpc_byte_buffer_reader* reader,
                  grpc_byte_buffer* buffer) {
    (g_core_codegen_interface->*ptr)(reader, buffer);
  }
  void ReaderInit(NewReaderInitAPI ptr, grpc_byte_buffer_reader* reader,
                  grpc_byte_buffer* buffer) {
    int result = (g_core_codegen_interface->*ptr)(reader, buffer);
    (void)result;
  }

 public:
  explicit GrpcBufferReader(grpc_byte_buffer* buffer)
      : byte_count_(0), backup_count_(0) {
    ReaderInit(&CoreCodegenInterface::grpc_byte_buffer_reader_init, &reader_,
               buffer);
  }
  ~GrpcBufferReader() GRPC_OVERRIDE {
    g_core_codegen_interface->grpc_byte_buffer_reader_destroy(&reader_);
  }

  bool Next(const void** data, int* size) GRPC_OVERRIDE {
    if (backup_count_ > 0) {
      *data = GPR_SLICE_START_PTR(slice_) + GPR_SLICE_LENGTH(slice_) -
              backup_count_;
      GPR_CODEGEN_ASSERT(backup_count_ <= INT_MAX);
      *size = (int)backup_count_;
      backup_count_ = 0;
      return true;
    }
    if (!g_core_codegen_interface->grpc_byte_buffer_reader_next(&reader_,
                                                                &slice_)) {
      return false;
    }
    g_core_codegen_interface->gpr_slice_unref(slice_);
    *data = GPR_SLICE_START_PTR(slice_);
    // On win x64, int is only 32bit
    GPR_CODEGEN_ASSERT(GPR_SLICE_LENGTH(slice_) <= INT_MAX);
    byte_count_ += * size = (int)GPR_SLICE_LENGTH(slice_);
    return true;
  }

  void BackUp(int count) GRPC_OVERRIDE { backup_count_ = count; }

  bool Skip(int count) GRPC_OVERRIDE {
    const void* data;
    int size;
    while (Next(&data, &size)) {
      if (size >= count) {
        BackUp(size - count);
        return true;
      }
      // size < count;
      count -= size;
    }
    // error or we have too large count;
    return false;
  }

  grpc::protobuf::int64 ByteCount() const GRPC_OVERRIDE {
    return byte_count_ - backup_count_;
  }

 private:
  int64_t byte_count_;
  int64_t backup_count_;
  grpc_byte_buffer_reader reader_;
  gpr_slice slice_;
};

}  // namespace tensorflow_helper

// Defines specialized serialization/deserialization routines that
// default to allowing a 2GB max message size.
//
// To instantiate this template for a particular type `T`, use
// `TF_GRPC_ALLOW_UNLIMITED_MESSAGE_SIZE(T)`, as defined below.
template <typename T>
class UnlimitedSizeProtoSerializationTraits {
 public:
  static Status Serialize(const T& msg, grpc_byte_buffer** bp,
                          bool* own_buffer) {
    *own_buffer = true;
    int byte_size = msg.ByteSize();
    if (byte_size < 0) {
      return Status(StatusCode::INTERNAL, "Message length was negative");
    } else if (byte_size <=
               tensorflow_helper::kGrpcBufferWriterMaxBufferLength) {
      gpr_slice slice = g_core_codegen_interface->gpr_slice_malloc(byte_size);
      GPR_CODEGEN_ASSERT(
          GPR_SLICE_END_PTR(slice) ==
          msg.SerializeWithCachedSizesToArray(GPR_SLICE_START_PTR(slice)));
      *bp = g_core_codegen_interface->grpc_raw_byte_buffer_create(&slice, 1);
      g_core_codegen_interface->gpr_slice_unref(slice);
      return g_core_codegen_interface->ok();
    } else {
      tensorflow_helper::GrpcBufferWriter writer(
          bp, tensorflow_helper::kGrpcBufferWriterMaxBufferLength);
      return msg.SerializeToZeroCopyStream(&writer)
                 ? g_core_codegen_interface->ok()
                 : Status(StatusCode::INTERNAL, "Failed to serialize message");
    }
  }

  static Status Deserialize(grpc_byte_buffer* buffer, T* msg,
                            int max_message_size = INT_MAX) {
    if (buffer == nullptr) {
      return Status(StatusCode::INTERNAL, "No payload");
    }
    Status result = g_core_codegen_interface->ok();
    {
      tensorflow_helper::GrpcBufferReader reader(buffer);
      ::grpc::protobuf::io::CodedInputStream decoder(&reader);
      if (max_message_size == 0) {
        // NOTE(mrry): Override maximum message size to 2GB.
        decoder.SetTotalBytesLimit(INT_MAX, INT_MAX);
      } else {
        decoder.SetTotalBytesLimit(max_message_size, max_message_size);
      }
      if (!msg->ParseFromCodedStream(&decoder)) {
        result = Status(StatusCode::INTERNAL, msg->InitializationErrorString());
      }
      if (!decoder.ConsumedEntireMessage()) {
        result = Status(StatusCode::INTERNAL, "Did not read entire message");
      }
    }
    g_core_codegen_interface->grpc_byte_buffer_destroy(buffer);
    return result;
  }
};

}  // namespace grpc

// For the given protobuf message type `MessageType`, specializes the
// gRPC serialization and deserialization such that the default
// maximum message size is 2GB.
#define TF_GRPC_ALLOW_UNLIMITED_MESSAGE_SIZE(MessageType)             \
  namespace grpc {                                                    \
  template <>                                                         \
  class SerializationTraits<MessageType>                              \
      : public UnlimitedSizeProtoSerializationTraits<MessageType> {}; \
  }  // namespace grpc

#endif  // THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_SERIALIZATION_TRAITS_H_
