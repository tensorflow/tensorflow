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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_BUFFER_REF_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_BUFFER_REF_H_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <ostream>
#include <tuple>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace litert {

//===----------------------------------------------------------------------===//
//
//                                                            << BUFFER REF >>
//
// Read, read/write, and owning views of buffers of arbitrary byte width types.
//
// Serialized model artifacts and assets are frequently large strings that with
// (annoyingly) non-standard char type and left padded. The following classes
// simplify handling such buffers in an efficient copy free manner. They also
// provide read and write left-padded aware interpretebility through standard
// signed char strings types. This is used for making manual edits to flatbuffer
// metadata or dierctly to serialized flatbuffer.
// NOTE: std::basic_xxx<unsigned char> not supported by our C++ toolchain.
//
// Pre-allocated buffers can be transferred to these classes or allocation can
// be internalized. XBufferRefs can be implictly upcasted to non-owning
// read/write or read-only to provide other routines with an appropriate view of
// the data. E.g.:
//
// ```
// void ReadBuffer(BufferRef r_buf) { std::cerr << r_buf.StrView(); }
// void WriteToBuffer(MutableBufferRef rw_buf) { rw_buf.WriteTo("SomeData"); }
// ...
// OwningBuffer<uint8_t> buf(size);
// WriteToBuffer(buf); // Implicitly convert to read/write with no ownership.
// ReadBuffer(buf); // Implicitly convert to read-only.
// ```
//
//===----------------------------------------------------------------------===//

// Allocation/Deallocation behavior for owning buffer refs. An allocator is a
// trivially constructible/destructible object that overrides () for allocating
// and freeing memory.

// Malloc/free based memory.
template <typename ByteT = uint8_t>
struct Mallocator {
  void operator()(ByteT* d) {
    if (d != nullptr) {
      free(d);
    }
  }

  ByteT* operator()(size_t bytes) {
    return reinterpret_cast<ByteT*>(malloc(bytes));
  }
};

// New/delete based memory.
template <typename ByteT = uint8_t>
struct Newlocator {
  void operator()(ByteT* d) {
    if (d != nullptr) {
      delete[] d;
    }
  }

  ByteT* operator()(size_t bytes) { return new ByteT[bytes]; }
};

//
// Read-Only Bytes
//

// Immutable and non-owning view of a buffer.
template <typename ByteT = uint8_t>
class BufferRef {
 public:
  using TupleT = std::tuple<const ByteT* const, const size_t, const size_t>;

  // Null buffer.
  BufferRef() : size_(0), offset_(0), data_(nullptr) {}

  // Construct from already allocated buffer. Methods will only expose
  // data[offset, offset + size].
  BufferRef(const ByteT* data, size_t size, size_t offset = 0)
      : size_(size), offset_(offset), data_(const_cast<ByteT*>(data)) {}
  BufferRef(const void* data, size_t size, size_t offset = 0)
      : size_(size),
        offset_(offset),
        data_(const_cast<ByteT*>(reinterpret_cast<const ByteT*>(data))) {}
  explicit BufferRef(absl::Span<const ByteT> data)
      : size_(data.size()),
        offset_(0),
        data_(const_cast<ByteT*>(data.data())) {}

  // Start of actual data.
  const ByteT* Data() const { return data_ + offset_; }

  // Size of actual data.
  size_t Size() const { return size_ - offset_; }

  // Get buffer details in tuple form.
  TupleT Get() const { return TupleT(data_, size_, offset_); }

  // Start of actual data as signed char. Might not be null terminated.
  const char* StrData() const { return reinterpret_cast<const char*>(Data()); }

  // Convenience view of actual data as a string. Makes null terminated.
  absl::string_view StrView() const {
    return absl::string_view(StrData(), Size());
  }

  // Const view of actual data.
  absl::Span<const ByteT> Span() const {
    return absl::MakeConstSpan(Data(), Size());
  }

  // Copy the buffer data to a vector.
  std::vector<ByteT> ToVec() const {
    return std::vector<ByteT>(StrData(), StrData() + Size());
  }

  // Write the string data to a stream.
  void WriteStr(std::ostream& out) const { out.write(StrData(), Size()); }

  // Print info about this buffer.
  void Dump(std::ostream& out) const {
    out << absl::StreamFormat("%s[%lu:%lu]\n", TypeName(), offset_, size_);
  }

  BufferRef(const BufferRef& other) = default;
  BufferRef& operator=(const BufferRef& other) = default;

  virtual ~BufferRef() = default;

 protected:
  size_t size_;
  size_t offset_;
  ByteT* data_ = nullptr;

  // Debug name.
  virtual absl::string_view TypeName() const { return "BufferRef"; }
};
template <typename ByteT = uint8_t>
BufferRef(const ByteT*, size_t, size_t) -> BufferRef<ByteT>;

//
// Read-Write Non-Owning Bytes
//

// Writeable (but still non-owning) version of BufferRef.
template <typename ByteT>
class MutableBufferRef : public BufferRef<ByteT> {
 public:
  using TupleT = std::tuple<ByteT* const, const size_t, const size_t>;

  // Null buffer.
  MutableBufferRef()
      : BufferRef<ByteT>((ByteT*)nullptr, /*size*/ 0, /*offset*/ 0) {}

  // Create a mutable view from pre-allocated non-const buffer.
  MutableBufferRef(ByteT* data, size_t size, size_t offset = 0)
      : BufferRef<ByteT>(data, size, offset) {}
  MutableBufferRef(void* data, size_t size, size_t offset = 0)
      : BufferRef<ByteT>(data, size, offset) {}
  explicit MutableBufferRef(absl::Span<ByteT> data) : BufferRef<ByteT>(data) {}
  explicit MutableBufferRef(absl::Span<const ByteT> data) = delete;
  MutableBufferRef(const ByteT*, size_t, size_t) = delete;
  MutableBufferRef(const void*, size_t, size_t) = delete;

  // Mutable start of actual data.
  ByteT* Data() { return this->data_ + this->offset_; }

  // Get the mutable start of actual data as a char pointer.
  char* StrData() { return reinterpret_cast<char*>(Data()); }

  // Get buffer info in tuple form.
  TupleT Get() { return TupleT(this->data_, this->size_, this->offset_); }

  // Mutable span of actual data.
  absl::Span<ByteT> Span() { return absl::MakeSpan(Data(), this->Size()); }

  // Write string into the actual buffer at offset. Returns false if the entire
  // string cannot fit into the actual buffer.
  bool WriteInto(absl::string_view str, size_t offset = 0) {
    if (str.size() > this->Size() - offset) {
      return false;
    }
    std::memcpy(Data() + offset, str.data(), str.size());
    return true;
  }

  MutableBufferRef(const MutableBufferRef& other) = default;
  MutableBufferRef& operator=(const MutableBufferRef& other) = default;

 protected:
  // Debug name.
  absl::string_view TypeName() const override { return "MutableBufferRef"; }
};
template <typename ByteT>
MutableBufferRef(ByteT*, size_t, size_t) -> MutableBufferRef<ByteT>;

//
// Read-Write Owning Bytes
//

// Writable and owning buffer reference. Can allocate new buffers internally and
// take ownership of existing buffers. Does not support resizing.
template <typename ByteT = uint8_t, class Allocator = Newlocator<ByteT>>
class OwningBufferRef : public MutableBufferRef<ByteT> {
 public:
  using TupleT = std::tuple<ByteT* const, const size_t, const size_t>;
  using WeakTupleT = std::tuple<ByteT*&, size_t&, size_t&>;

  // Null buffer.
  OwningBufferRef()
      : MutableBufferRef<ByteT>(/*data*/ (ByteT*)nullptr, /*size*/ 0,
                                /*offset*/ 0) {}

  // Initialize a new buffer reference and allocate internally.
  explicit OwningBufferRef(size_t size)
      : MutableBufferRef<ByteT>(/*data*/ (ByteT*)nullptr, size, /*offset*/ 0) {
    this->data_ = (ByteT*)Allocator()(size);
  }

  // Take ownership of given buffer.
  OwningBufferRef(ByteT* data, size_t size, size_t offset = 0)
      : MutableBufferRef<ByteT>(data, size, offset) {}
  OwningBufferRef(void* data, size_t size, size_t offset = 0)
      : MutableBufferRef<ByteT>(data, size, offset) {}
  explicit OwningBufferRef(absl::Span<ByteT> data)
      : MutableBufferRef<ByteT>(data) {}

  // Copy the given buffer.
  OwningBufferRef(const ByteT* data, size_t size)
      : MutableBufferRef<ByteT>(/*data*/ (ByteT*)nullptr, size,
                                /*offset*/ 0) {
    this->data_ = (ByteT*)Allocator()(size);
    std::memcpy(this->data_, data, size);
  }
  explicit OwningBufferRef(absl::Span<const ByteT> data)
      : OwningBufferRef<ByteT, Allocator>(data.data(), data.size()) {}

  // Copy data from givens string.
  explicit OwningBufferRef(absl::string_view data)
      : OwningBufferRef<ByteT, Allocator>(
            reinterpret_cast<const ByteT*>(data.data()), data.size()) {}

  // Copy data from given c-style string.
  explicit OwningBufferRef(const char* data)
      : OwningBufferRef<ByteT, Allocator>(absl::string_view(data)) {}

  // Drop reference to any owned memory.
  void Drop() {
    this->data_ = nullptr;
    this->size_ = 0;
    this->offset_ = 0;
  }

  // Get the buffer details and drop references to them.
  TupleT Release() {
    auto res = std::make_tuple(this->data_, this->size_, this->offset_);
    Drop();
    return res;
  }

  // Get weak references to buffer data. Takes ownership of anything that
  // is swapped in.
  WeakTupleT GetWeak() {
    return WeakTupleT(this->data_, this->size_, this->offset_);
  }

  // Free any owned memory.
  void Reset() {
    Allocator()(this->data_);
    Drop();
  }

  // Reset any existing data and copy in given ro buffer.
  void Assign(const ByteT* buf, size_t size, size_t offset = 0) {
    Reset();
    this->size_ = size;
    this->data_ = (ByteT*)Allocator()(this->size_);
    std::memcpy(this->data_, buf, this->size_);
    this->offset_ = offset;
  }

  OwningBufferRef(OwningBufferRef&& other)
      : MutableBufferRef<ByteT>(other.data_, other.size_, other.offset_) {
    other.Drop();
  }

  OwningBufferRef& operator=(OwningBufferRef&& other) {
    if (this != &other) {
      Reset();
      this->data_ = other.data_;
      this->size_ = other.size_;
      this->offset_ = other.offset_;
      other.Drop();
    }
    return *this;
  }

  OwningBufferRef(const OwningBufferRef& other)
      : MutableBufferRef<ByteT>(/*data*/ (ByteT*)nullptr, other.size_,
                                other.offset_) {
    Assign(other.data_, other.size_, other.offset_);
  }

  OwningBufferRef& operator=(const OwningBufferRef& other) {
    Assign(other.data_, other.size_, other.offset_);
    return *this;
  }

  ~OwningBufferRef() override { Reset(); }

 protected:
  // Debug string.
  absl::string_view TypeName() const override { return "OwningBufferRef"; }
};

template <typename ByteT = uint8_t, class Allocator = Newlocator<ByteT>>
OwningBufferRef(const ByteT*, size_t) -> OwningBufferRef<ByteT, Allocator>;

template <typename ByteT = uint8_t, class Allocator = Newlocator<ByteT>>
OwningBufferRef(ByteT*, size_t) -> OwningBufferRef<ByteT, Allocator>;

template <typename ByteT = char, class Allocator = Newlocator<ByteT>>
OwningBufferRef(const char*) -> OwningBufferRef<ByteT, Allocator>;

}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_BUFFER_REF_H_
