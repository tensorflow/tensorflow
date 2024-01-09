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
#ifndef TENSORFLOW_CORE_TFRT_MLRT_BYTECODE_BYTECODE_H_
#define TENSORFLOW_CORE_TFRT_MLRT_BYTECODE_BYTECODE_H_

// This file defines bytecode primitives that can be used to build bytecode
// structures. This library is C++17 compliant and portable for different
// platforms. It should be also as effcient as plain C++ structs on common
// platforms.
//
// Usage:
//
// class CustomStruct {
//  public:
//    // The actual storage of this CustomStruct should be defined as a member
//    // struct of this class. Defining storage struct is almost as simple as
//    // defining a plain C++ struct;
//    struct Storage {
//      using Self = Storage;
//      // DEFINE_BYTECODE_FIELD will generate helpers for reading and
//      constructing
//      // the field in bytecode.
//      DEFINE_BYTECODE_FIELD(uint32_t, x);
//      DEFINE_BYTECODE_FIELD(bc::Vector<uint32_t>, y);
//    };
//
//    // If the storage involves indirection like std::vector, a member class
//    // Constructor should be also provided.
//    class Constructor {
//      public:
//        // The Constructor will use `allocator` to allocate indirect storage,
//        // though the direct storage is assumed to be already allocated using
//        // the same allocator starting at `address`.
//        explicit Constructor(Allocator* allocator, BcAddr_t address)
//          : allocator_(allocator), address_(address) {}
//
//      // Setting trivial fields only need to call construct_<field_name>
//      // provided by DEFINE_BYTECODE_FIELD.
//      void set_x(uint32_t x) {
//        Storage::construct_x(allocator_, address_, x);
//      }
//
//      // Setting non-trivial fields only need to call construct_<field_name>
//      // provided by DEFINE_BYTECODE_FIELD and also return the field's
//      constructor. bc::Vector<uint32_t>::Constructor construct_y(size_t
//      y_size) {
//        return Storage::construct_y(allocator_, address_, y_size);
//      }
//
//      BcAddr_t address() const { return address_; }
//
//      private:
//        bc::Allocator* allocator_;
//        BcAddr_t address_;
//    };
//    using NonTrivialConstructorType = Constructor;
//
//    explicit CustomStruct(const char* p) : p_(p) {}
//
//    // Reading fields needs only calling read_<field_name> methods provided by
//    // DEFINE_BYTECODE_FIELD.
//    uint32_t x() const { return Storage::read_x(p_); }
//    bc::Vector<uint32_t> y() const { return Storage::read_y(p_); }
//
//    private:
//      // The CustomStruct can contain only the pointer to the actual memory
//      // blob. So fields need not be touched if not necessary, which would
//      // otherwise incurs overhead.
//      const char* p_;
// };

#include <cstddef>
#include <cstring>
#include <iterator>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"

namespace mlrt {
namespace bc {

using BcAddr_t = uint64_t;

class Buffer {
 public:
  char* Get(BcAddr_t address) {
    DCHECK_LT(address, buffer_.size());
    return &buffer_.at(address);
  }

  char* data() { return buffer_.data(); }
  const char* data() const { return buffer_.data(); }
  size_t size() const { return buffer_.size(); }
  bool empty() const { return buffer_.empty(); }

  void shrink_to_fit() { buffer_.shrink_to_fit(); }

 private:
  static_assert(alignof(std::max_align_t) >= 8,
                "The bytecode buffer needs to be at least 8-byte aligned.");
  std::vector<char> buffer_;

  friend class Allocator;
};

class Allocator {
 public:
  explicit Allocator(Buffer* buffer) : buffer_(buffer) {
    DCHECK(buffer != nullptr);
  }

  BcAddr_t Allocate(size_t size, size_t alignment) {
    DCHECK_LE(alignment, 8);

    // Calculate the next buffer size that is greater or equal to the previous
    // buffer size, and is also aligned to `alignment`.
    size_t next_align =
        (buffer_->buffer_.size() + alignment - 1) / alignment * alignment;

    buffer_->buffer_.resize(next_align + size);

    return next_align;
  }

  template <typename T>
  BcAddr_t Allocate() {
    static_assert(std::is_trivial<T>::value, "T must be trivial.");
    return Allocate(sizeof(T), alignof(T));
  }

  size_t size() const { return buffer_->size(); }

  char* raw(BcAddr_t address) { return buffer_->Get(address); }

 private:
  Buffer* buffer_;
};

// AccessTraits encapsulates the fundamental Read() and Construct() methods for
// reading and constructing bytecode data structures.

// AccessTraits specialized for trivial types.
template <typename T, typename Enable = void>
struct AccessTraits {
  using StorageType = T;
  static_assert(std::is_trivial<StorageType>::value,
                "StorageType must be trivial.");

  using ConstructorType = void;

  static T Read(const char* p) {
    // To be compliant with C++ standard on object lifetime and strict aliasing
    // rules, we have to copy the data from memory to construct a new object.
    // This is fine on most platforms as the copy can be optimized away,
    // assuming `p` is sufficiently aligned.
    T value;
    std::memcpy(&value, p, sizeof(T));
    return value;
  }

  template <typename... Args>
  static BcAddr_t Construct(Allocator* allocator, BcAddr_t address,
                            Args&&... args) {
    // Similar to Read(), memcpy is used to serialize data to bytecode.
    T value(std::forward<Args>(args)...);
    std::memcpy(allocator->raw(address), &value, sizeof(T));
    return address;
  }

  // Place the bytes directly for this trivial type T. It also supports placing
  // bytes for a contiguous array of T. The number of bytes, `size` must not be
  // greater than `num` * sizeof(T).
  static void Place(Allocator* allocator, BcAddr_t address, const char* data,
                    size_t size, size_t num = 1) {
    CHECK_LE(size, num * sizeof(T));  // Crash Ok
    std::memcpy(allocator->raw(address), data, size);
  }
};

// AccessTraits specialized for non-trivial types.
template <typename T>
struct AccessTraits<T, std::void_t<typename T::NonTrivialConstructorType>> {
  // Non-trivial types should provide a member struct `StorageType` to
  // specify the storage layout.
  using StorageType = typename T::StorageType;
  static_assert(std::is_trivial<StorageType>::value,
                "StorageType must be trivial.");

  // Non-trivial types should provide a member type `NonTrivialConstructorType`
  // for constructing storages.
  using ConstructorType = typename T::NonTrivialConstructorType;

  static T Read(const char* p) {
    // Reading non-trivial types is simply constructing the bytecode type with
    // the pointer to the memory blob. All reading methods are encapsulated in
    // `T`.
    return T(p);
  }

  template <typename... Args>
  static ConstructorType Construct(Allocator* allocator, BcAddr_t address,
                                   Args&&... args) {
    // Constructing non-trivial types is simply creating the corresponding
    // constructor.
    return ConstructorType(allocator, address, std::forward<Args>(args)...);
  }
};

// The bytecode counterparts of malloc() and operator new() are also provided.
template <typename T>
BcAddr_t Allocate(Allocator* allocator) {
  return allocator->Allocate<typename AccessTraits<T>::StorageType>();
}
template <typename T, typename... Args>
auto New(Allocator* allocator, Args&&... args) {
  auto address = Allocate<T>(allocator);
  return AccessTraits<T>::Construct(allocator, address,
                                    std::forward<Args>(args)...);
}

// The iterator for reading bytecode data. It uses AccessTraits<T>::Read() for
// reading the data. It is an input iterator as we cannot return the type-safe
// reference to the data in bytecode in a C++ compliant way due to object
// lifetime and strict aliasing rule.
template <typename T>
class ReadIterator {
  using StorageType = typename AccessTraits<T>::StorageType;

 public:
  using difference_type = std::ptrdiff_t;
  using value_type = std::remove_cv_t<T>;
  using pointer = void;
  using reference = value_type;
  using iterator_category = std::input_iterator_tag;

  explicit ReadIterator(const char* data) : data_(data) {}

  const char* data() const { return data_; }

  value_type operator*() const { return AccessTraits<T>::Read(data_); }

  ReadIterator& operator++() {
    data_ += sizeof(StorageType);
    return *this;
  }

  ReadIterator operator++(int) {
    ReadIterator r = *this;
    data_ += sizeof(StorageType);
    return r;
  }

  ReadIterator& operator+=(difference_type offset) {
    data_ += offset * sizeof(StorageType);
    return *this;
  }

  ReadIterator operator+(difference_type offset) const {
    ReadIterator r = *this;
    r += offset;
    return r;
  }

  ReadIterator& operator--() {
    data_ -= sizeof(StorageType);
    return *this;
  }

  ReadIterator operator--(int) {
    ReadIterator r = *this;
    data_ -= sizeof(StorageType);
    return r;
  }

  ReadIterator& operator-=(difference_type offset) {
    data_ -= offset * sizeof(StorageType);
    return *this;
  }

  ReadIterator operator-(difference_type offset) const {
    ReadIterator r = *this;
    r -= offset;
    return r;
  }

  difference_type operator-(const ReadIterator& other) const {
    DCHECK_EQ((data_ - other.data_) % sizeof(StorageType), 0);
    return (data_ - other.data_) / sizeof(StorageType);
  }

  friend bool operator==(const ReadIterator& a, const ReadIterator& b) {
    return a.data_ == b.data_;
  }

  friend bool operator!=(const ReadIterator& a, const ReadIterator& b) {
    return !(a == b);
  }

  friend bool operator<(const ReadIterator& a, const ReadIterator& b) {
    return a.data_ < b.data_;
  }

  friend bool operator<=(const ReadIterator& a, const ReadIterator& b) {
    return a.data_ <= b.data_;
  }

  friend bool operator>(const ReadIterator& a, const ReadIterator& b) {
    return a.data_ > b.data_;
  }

  friend bool operator>=(const ReadIterator& a, const ReadIterator& b) {
    return a.data_ >= b.data_;
  }

 private:
  const char* data_ = nullptr;
};

// DEFINE_BYTECODE_FIELD provides helper functions for reading and constructing
// member fields in bytecode.
#define DEFINE_BYTECODE_FIELD(Type, name)                                   \
  typename ::mlrt::bc::AccessTraits<Type>::StorageType name;                \
  static const char* name##_pointer(const char* base) {                     \
    return base + offsetof(Self, name);                                     \
  }                                                                         \
  static ::mlrt::bc::BcAddr_t name##_address(::mlrt::bc::BcAddr_t base) {   \
    return base + offsetof(Self, name);                                     \
  }                                                                         \
  static Type read_##name(const char* base) {                               \
    return ::mlrt::bc::AccessTraits<Type>::Read(name##_pointer(base));      \
  }                                                                         \
  template <typename... Args>                                               \
  static auto construct_##name(::mlrt::bc::Allocator* allocator,            \
                               ::mlrt::bc::BcAddr_t base, Args&&... args) { \
    return ::mlrt::bc::AccessTraits<Type>::Construct(                       \
        allocator, name##_address(base), std::forward<Args>(args)...);      \
  }                                                                         \
  static_assert(                                                            \
      std::is_trivial<                                                      \
          typename ::mlrt::bc::AccessTraits<Type>::StorageType>::value,     \
      "Bytecode storage types must be trivial.")

// Defines a bytecode vector.
template <typename T, typename SizeType = uint32_t>
class Vector {
 public:
  struct Storage {
    using Self = Storage;
    DEFINE_BYTECODE_FIELD(SizeType, size);
    DEFINE_BYTECODE_FIELD(SizeType, offset);
  };
  static_assert(std::is_trivial<Storage>::value, "StorageType is trivial");
  static_assert(std::is_standard_layout<Storage>::value,
                "StorageType has standard layout");
  static_assert(sizeof(Storage) == 2 * sizeof(SizeType));
  static_assert(alignof(Storage) == alignof(SizeType));

  using StorageType = Storage;
  using ElementStorageType = typename AccessTraits<T>::StorageType;

  using value_type = T;
  using iterator = ReadIterator<T>;
  using const_iterator = iterator;

  class Constructor {
   public:
    Constructor(Allocator* allocator, BcAddr_t address, size_t size)
        : allocator_(allocator), address_(address) {
      DCHECK_GE(allocator->size(), address + sizeof(StorageType));
      size_t data_start = allocator->Allocate(size * sizeof(ElementStorageType),
                                              alignof(ElementStorageType));

      CHECK_LT(size, std::numeric_limits<SizeType>::max());  // Crash Ok
      CHECK_LT(data_start - address,                         // Crash Ok
               std::numeric_limits<SizeType>::max());
      storage_.size = size;
      storage_.offset = data_start - address;
      AccessTraits<StorageType>::Construct(allocator, address, storage_);
    }

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    Constructor(Allocator* allocator, BcAddr_t address,
                const std::vector<T>& vec)
        : Constructor(allocator, address, vec.size()) {
      Assign(vec.begin(), vec.end());
    }

    template <typename... Args>
    auto ConstructAt(size_t index, Args&&... args) {
      DCHECK_LT(index, size());
      return AccessTraits<T>::Construct(allocator_, GetElementAddress(index),
                                        std::forward<Args>(args)...);
    }

    template <typename V>
    void Assign(std::initializer_list<V> ilist) {
      DCHECK_EQ(ilist.size(), size());
      Assign(ilist.begin(), ilist.end());
    }

    template <typename Range>
    void Assign(const Range& range) {
      DCHECK_EQ(std::distance(std::begin(range), std::end(range)), size());
      Assign(std::begin(range), std::end(range));
    }

    template <typename Iter>
    void Assign(Iter begin, Iter end) {
      size_t i = 0;
      for (; begin != end; ++begin) {
        ConstructAt(i++, *begin);
      }
      DCHECK_EQ(i, size());
    }

    // If T is a trivial inplace type like int32_t, we can place the bytes for
    // this vector directly instead of constructing the elements one by one.
    template <
        typename U = T,
        typename std::enable_if<
            std::is_same_v<typename AccessTraits<U>::ConstructorType, void>,
            int>::type = 0>
    void Place(const char* data, size_t size) {
      AccessTraits<U>::Place(allocator_, address_ + storage_.offset, data, size,
                             storage_.size);
    }

    // TODO(chky): Implement iterators for construction.

    size_t size() const { return storage_.size; }
    BcAddr_t address() const { return address_; }

   private:
    BcAddr_t GetElementAddress(size_t index) const {
      return address_ + storage_.offset + index * sizeof(ElementStorageType);
    }

    Allocator* allocator_;
    BcAddr_t address_;
    Vector::Storage storage_;
  };
  using NonTrivialConstructorType = Constructor;

  explicit Vector(const char* p) : p_(p) {
    static_assert(!std::is_trivial_v<Vector>);
    DCHECK(p_ != nullptr);
  }
  Vector() {
    static_assert(!std::is_trivial_v<Vector>);
    static Storage kEmptyStorage{0, 0};
    p_ = reinterpret_cast<const char*>(&kEmptyStorage);
  }

  const char* data() const { return p_ + offset(); }

  size_t size() const { return StorageType::read_size(p_); }
  bool empty() const { return size() == 0; }

  iterator begin() const { return iterator(data()); }
  iterator end() const {
    return iterator(data() + size() * sizeof(ElementStorageType));
  }

  T operator[](size_t index) const {
    DCHECK_LT(index, size());
    auto iter = begin();
    iter += index;
    return *iter;
  }

 private:
  SizeType offset() const { return StorageType::read_offset(p_); }

  const char* p_;
};

class String : public Vector<char, uint64_t> {
 public:
  using Base = Vector<char, uint64_t>;
  using Base::Base;

  class Constructor : public Base::Constructor {
   public:
    using Base::Constructor::Assign;

    Constructor(Allocator* allocator, BcAddr_t address, absl::string_view str)
        : Base::Constructor(allocator, address, str.size()) {
      Assign(str.begin(), str.end());
    }
  };
  using NonTrivialConstructorType = Constructor;

  using Base::data;
  using Base::size;

  std::string str() const { return std::string(data(), size()); }
  absl::string_view Get() const { return absl::string_view(data(), size()); }

  operator absl::string_view() const {  // NOLINT
    return absl::string_view(data(), size());
  }

  friend bool operator==(String x, absl::string_view y) { return x.Get() == y; }
  friend bool operator==(absl::string_view x, String y) { return x == y.Get(); }
};

}  // namespace bc
}  // namespace mlrt

#endif  // TENSORFLOW_CORE_TFRT_MLRT_BYTECODE_BYTECODE_H_
