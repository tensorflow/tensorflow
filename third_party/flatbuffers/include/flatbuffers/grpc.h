/*
 * Copyright 2014 Google Inc. All rights reserved.
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

#ifndef FLATBUFFERS_GRPC_H_
#define FLATBUFFERS_GRPC_H_

// Helper functionality to glue FlatBuffers and GRPC.

#include "flatbuffers/flatbuffers.h"
#include "grpc++/support/byte_buffer.h"
#include "grpc/byte_buffer_reader.h"

namespace flatbuffers {
namespace grpc {

// Message is a typed wrapper around a buffer that manages the underlying
// `grpc_slice` and also provides flatbuffers-specific helpers such as `Verify`
// and `GetRoot`. Since it is backed by a `grpc_slice`, the underlying buffer
// is refcounted and ownership is be managed automatically.
template<class T> class Message {
 public:
  Message() : slice_(grpc_empty_slice()) {}

  Message(grpc_slice slice, bool add_ref)
      : slice_(add_ref ? grpc_slice_ref(slice) : slice) {}

  Message &operator=(const Message &other) = delete;

  Message(Message &&other) : slice_(other.slice_) {
    other.slice_ = grpc_empty_slice();
  }

  Message(const Message &other) = delete;

  Message &operator=(Message &&other) {
    grpc_slice_unref(slice_);
    slice_ = other.slice_;
    other.slice_ = grpc_empty_slice();
    return *this;
  }

  ~Message() { grpc_slice_unref(slice_); }

  const uint8_t *mutable_data() const { return GRPC_SLICE_START_PTR(slice_); }

  const uint8_t *data() const { return GRPC_SLICE_START_PTR(slice_); }

  size_t size() const { return GRPC_SLICE_LENGTH(slice_); }

  bool Verify() const {
    Verifier verifier(data(), size());
    return verifier.VerifyBuffer<T>(nullptr);
  }

  T *GetMutableRoot() { return flatbuffers::GetMutableRoot<T>(mutable_data()); }

  const T *GetRoot() const { return flatbuffers::GetRoot<T>(data()); }

  // This is only intended for serializer use, or if you know what you're doing
  const grpc_slice &BorrowSlice() const { return slice_; }

 private:
  grpc_slice slice_;
};

class MessageBuilder;

// SliceAllocator is a gRPC-specific allocator that uses the `grpc_slice`
// refcounted slices to manage memory ownership. This makes it easy and
// efficient to transfer buffers to gRPC.
class SliceAllocator : public Allocator {
 public:
  SliceAllocator() : slice_(grpc_empty_slice()) {}

  SliceAllocator(const SliceAllocator &other) = delete;
  SliceAllocator &operator=(const SliceAllocator &other) = delete;

  SliceAllocator(SliceAllocator &&other)
    : slice_(grpc_empty_slice()) {
    // default-construct and swap idiom
    swap(other);
  }

  SliceAllocator &operator=(SliceAllocator &&other) {
    // move-construct and swap idiom
    SliceAllocator temp(std::move(other));
    swap(temp);
    return *this;
  }

  void swap(SliceAllocator &other) {
    using std::swap;
    swap(slice_, other.slice_);
  }

  virtual ~SliceAllocator() { grpc_slice_unref(slice_); }

  virtual uint8_t *allocate(size_t size) override {
    FLATBUFFERS_ASSERT(GRPC_SLICE_IS_EMPTY(slice_));
    slice_ = grpc_slice_malloc(size);
    return GRPC_SLICE_START_PTR(slice_);
  }

  virtual void deallocate(uint8_t *p, size_t size) override {
    FLATBUFFERS_ASSERT(p == GRPC_SLICE_START_PTR(slice_));
    FLATBUFFERS_ASSERT(size == GRPC_SLICE_LENGTH(slice_));
    grpc_slice_unref(slice_);
    slice_ = grpc_empty_slice();
  }

  virtual uint8_t *reallocate_downward(uint8_t *old_p, size_t old_size,
                                       size_t new_size, size_t in_use_back,
                                       size_t in_use_front) override {
    FLATBUFFERS_ASSERT(old_p == GRPC_SLICE_START_PTR(slice_));
    FLATBUFFERS_ASSERT(old_size == GRPC_SLICE_LENGTH(slice_));
    FLATBUFFERS_ASSERT(new_size > old_size);
    grpc_slice old_slice = slice_;
    grpc_slice new_slice = grpc_slice_malloc(new_size);
    uint8_t *new_p = GRPC_SLICE_START_PTR(new_slice);
    memcpy_downward(old_p, old_size, new_p, new_size, in_use_back,
                    in_use_front);
    slice_ = new_slice;
    grpc_slice_unref(old_slice);
    return new_p;
  }

 private:
  grpc_slice &get_slice(uint8_t *p, size_t size) {
    FLATBUFFERS_ASSERT(p == GRPC_SLICE_START_PTR(slice_));
    FLATBUFFERS_ASSERT(size == GRPC_SLICE_LENGTH(slice_));
    return slice_;
  }

  grpc_slice slice_;

  friend class MessageBuilder;
};

// SliceAllocatorMember is a hack to ensure that the MessageBuilder's
// slice_allocator_ member is constructed before the FlatBufferBuilder, since
// the allocator is used in the FlatBufferBuilder ctor.
namespace detail {
struct SliceAllocatorMember {
  SliceAllocator slice_allocator_;
};
}  // namespace detail

// MessageBuilder is a gRPC-specific FlatBufferBuilder that uses SliceAllocator
// to allocate gRPC buffers.
class MessageBuilder : private detail::SliceAllocatorMember,
                       public FlatBufferBuilder {
 public:
  explicit MessageBuilder(uoffset_t initial_size = 1024)
    : FlatBufferBuilder(initial_size, &slice_allocator_, false) {}

  MessageBuilder(const MessageBuilder &other) = delete;
  MessageBuilder &operator=(const MessageBuilder &other) = delete;

  MessageBuilder(MessageBuilder &&other)
    : FlatBufferBuilder(1024, &slice_allocator_, false) {
    // Default construct and swap idiom.
    Swap(other);
  }

  /// Create a MessageBuilder from a FlatBufferBuilder.
  explicit MessageBuilder(FlatBufferBuilder &&src, void (*dealloc)(void*, size_t) = &DefaultAllocator::dealloc)
    : FlatBufferBuilder(1024, &slice_allocator_, false) {
    src.Swap(*this);
    src.SwapBufAllocator(*this);
    if (buf_.capacity()) {
      uint8_t *buf = buf_.scratch_data();       // pointer to memory
      size_t capacity = buf_.capacity();        // size of memory
      slice_allocator_.slice_ = grpc_slice_new_with_len(buf, capacity, dealloc);
    }
    else {
      slice_allocator_.slice_ = grpc_empty_slice();
    }
  }

  /// Move-assign a FlatBufferBuilder to a MessageBuilder.
  /// Only FlatBufferBuilder with default allocator (basically, nullptr) is supported.
  MessageBuilder &operator=(FlatBufferBuilder &&src) {
    // Move construct a temporary and swap
    MessageBuilder temp(std::move(src));
    Swap(temp);
    return *this;
  }

  MessageBuilder &operator=(MessageBuilder &&other) {
    // Move construct a temporary and swap
    MessageBuilder temp(std::move(other));
    Swap(temp);
    return *this;
  }

  void Swap(MessageBuilder &other) {
    slice_allocator_.swap(other.slice_allocator_);
    FlatBufferBuilder::Swap(other);
    // After swapping the FlatBufferBuilder, we swap back the allocator, which restores
    // the original allocator back in place. This is necessary because MessageBuilder's
    // allocator is its own member (SliceAllocatorMember). The allocator passed to
    // FlatBufferBuilder::vector_downward must point to this member.
    buf_.swap_allocator(other.buf_);
  }

  // Releases the ownership of the buffer pointer.
  // Returns the size, offset, and the original grpc_slice that
  // allocated the buffer. Also see grpc_slice_unref().
  uint8_t *ReleaseRaw(size_t &size, size_t &offset, grpc_slice &slice) {
    uint8_t *buf = FlatBufferBuilder::ReleaseRaw(size, offset);
    slice = slice_allocator_.slice_;
    slice_allocator_.slice_ = grpc_empty_slice();
    return buf;
  }

  ~MessageBuilder() {}

  // GetMessage extracts the subslice of the buffer corresponding to the
  // flatbuffers-encoded region and wraps it in a `Message<T>` to handle buffer
  // ownership.
  template<class T> Message<T> GetMessage() {
    auto buf_data = buf_.scratch_data();       // pointer to memory
    auto buf_size = buf_.capacity();  // size of memory
    auto msg_data = buf_.data();      // pointer to msg
    auto msg_size = buf_.size();      // size of msg
    // Do some sanity checks on data/size
    FLATBUFFERS_ASSERT(msg_data);
    FLATBUFFERS_ASSERT(msg_size);
    FLATBUFFERS_ASSERT(msg_data >= buf_data);
    FLATBUFFERS_ASSERT(msg_data + msg_size <= buf_data + buf_size);
    // Calculate offsets from the buffer start
    auto begin = msg_data - buf_data;
    auto end = begin + msg_size;
    // Get the slice we are working with (no refcount change)
    grpc_slice slice = slice_allocator_.get_slice(buf_data, buf_size);
    // Extract a subslice of the existing slice (increment refcount)
    grpc_slice subslice = grpc_slice_sub(slice, begin, end);
    // Wrap the subslice in a `Message<T>`, but don't increment refcount
    Message<T> msg(subslice, false);
    return msg;
  }

  template<class T> Message<T> ReleaseMessage() {
    Message<T> msg = GetMessage<T>();
    Reset();
    return msg;
  }

 private:
  // SliceAllocator slice_allocator_;  // part of SliceAllocatorMember
};

}  // namespace grpc
}  // namespace flatbuffers

namespace grpc {

template<class T> class SerializationTraits<flatbuffers::grpc::Message<T>> {
 public:
  static grpc::Status Serialize(const flatbuffers::grpc::Message<T> &msg,
                                grpc_byte_buffer **buffer, bool *own_buffer) {
    // We are passed in a `Message<T>`, which is a wrapper around a
    // `grpc_slice`. We extract it here using `BorrowSlice()`. The const cast
    // is necesary because the `grpc_raw_byte_buffer_create` func expects
    // non-const slices in order to increment their refcounts.
    grpc_slice *slice = const_cast<grpc_slice *>(&msg.BorrowSlice());
    // Now use `grpc_raw_byte_buffer_create` to package the single slice into a
    // `grpc_byte_buffer`, incrementing the refcount in the process.
    *buffer = grpc_raw_byte_buffer_create(slice, 1);
    *own_buffer = true;
    return grpc::Status::OK;
  }

  // Deserialize by pulling the
  static grpc::Status Deserialize(grpc_byte_buffer *buffer,
                                  flatbuffers::grpc::Message<T> *msg) {
    if (!buffer) {
      return ::grpc::Status(::grpc::StatusCode::INTERNAL, "No payload");
    }
    // Check if this is a single uncompressed slice.
    if ((buffer->type == GRPC_BB_RAW) &&
        (buffer->data.raw.compression == GRPC_COMPRESS_NONE) &&
        (buffer->data.raw.slice_buffer.count == 1)) {
      // If it is, then we can reference the `grpc_slice` directly.
      grpc_slice slice = buffer->data.raw.slice_buffer.slices[0];
      // We wrap a `Message<T>` around the slice, incrementing the refcount.
      *msg = flatbuffers::grpc::Message<T>(slice, true);
    } else {
      // Otherwise, we need to use `grpc_byte_buffer_reader_readall` to read
      // `buffer` into a single contiguous `grpc_slice`. The gRPC reader gives
      // us back a new slice with the refcount already incremented.
      grpc_byte_buffer_reader reader;
      grpc_byte_buffer_reader_init(&reader, buffer);
      grpc_slice slice = grpc_byte_buffer_reader_readall(&reader);
      grpc_byte_buffer_reader_destroy(&reader);
      // We wrap a `Message<T>` around the slice, but dont increment refcount
      *msg = flatbuffers::grpc::Message<T>(slice, false);
    }
    grpc_byte_buffer_destroy(buffer);
#if FLATBUFFERS_GRPC_DISABLE_AUTO_VERIFICATION
    return ::grpc::Status::OK;
#else
    if (msg->Verify()) {
      return ::grpc::Status::OK;
    } else {
      return ::grpc::Status(::grpc::StatusCode::INTERNAL,
                            "Message verification failed");
    }
#endif
  }
};

}  // namespace grpc

#endif  // FLATBUFFERS_GRPC_H_
