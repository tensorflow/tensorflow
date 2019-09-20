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

#ifndef FLATBUFFERS_H_
#define FLATBUFFERS_H_

#include "flatbuffers/base.h"

#if defined(FLATBUFFERS_NAN_DEFAULTS)
#include <cmath>
#endif

namespace flatbuffers {
// Generic 'operator==' with conditional specialisations.
// T e - new value of a scalar field.
// T def - default of scalar (is known at compile-time).
template<typename T> inline bool IsTheSameAs(T e, T def) { return e == def; }

#if defined(FLATBUFFERS_NAN_DEFAULTS) && \
    defined(FLATBUFFERS_HAS_NEW_STRTOD) && (FLATBUFFERS_HAS_NEW_STRTOD > 0)
// Like `operator==(e, def)` with weak NaN if T=(float|double).
template<typename T> inline bool IsFloatTheSameAs(T e, T def) {
  return (e == def) || ((def != def) && (e != e));
}
template<> inline bool IsTheSameAs<float>(float e, float def) {
  return IsFloatTheSameAs(e, def);
}
template<> inline bool IsTheSameAs<double>(double e, double def) {
  return IsFloatTheSameAs(e, def);
}
#endif

// Wrapper for uoffset_t to allow safe template specialization.
// Value is allowed to be 0 to indicate a null object (see e.g. AddOffset).
template<typename T> struct Offset {
  uoffset_t o;
  Offset() : o(0) {}
  Offset(uoffset_t _o) : o(_o) {}
  Offset<void> Union() const { return Offset<void>(o); }
  bool IsNull() const { return !o; }
};

inline void EndianCheck() {
  int endiantest = 1;
  // If this fails, see FLATBUFFERS_LITTLEENDIAN above.
  FLATBUFFERS_ASSERT(*reinterpret_cast<char *>(&endiantest) ==
                     FLATBUFFERS_LITTLEENDIAN);
  (void)endiantest;
}

template<typename T> FLATBUFFERS_CONSTEXPR size_t AlignOf() {
  // clang-format off
  #ifdef _MSC_VER
    return __alignof(T);
  #else
    #ifndef alignof
      return __alignof__(T);
    #else
      return alignof(T);
    #endif
  #endif
  // clang-format on
}

// When we read serialized data from memory, in the case of most scalars,
// we want to just read T, but in the case of Offset, we want to actually
// perform the indirection and return a pointer.
// The template specialization below does just that.
// It is wrapped in a struct since function templates can't overload on the
// return type like this.
// The typedef is for the convenience of callers of this function
// (avoiding the need for a trailing return decltype)
template<typename T> struct IndirectHelper {
  typedef T return_type;
  typedef T mutable_return_type;
  static const size_t element_stride = sizeof(T);
  static return_type Read(const uint8_t *p, uoffset_t i) {
    return EndianScalar((reinterpret_cast<const T *>(p))[i]);
  }
};
template<typename T> struct IndirectHelper<Offset<T>> {
  typedef const T *return_type;
  typedef T *mutable_return_type;
  static const size_t element_stride = sizeof(uoffset_t);
  static return_type Read(const uint8_t *p, uoffset_t i) {
    p += i * sizeof(uoffset_t);
    return reinterpret_cast<return_type>(p + ReadScalar<uoffset_t>(p));
  }
};
template<typename T> struct IndirectHelper<const T *> {
  typedef const T *return_type;
  typedef T *mutable_return_type;
  static const size_t element_stride = sizeof(T);
  static return_type Read(const uint8_t *p, uoffset_t i) {
    return reinterpret_cast<const T *>(p + i * sizeof(T));
  }
};

// An STL compatible iterator implementation for Vector below, effectively
// calling Get() for every element.
template<typename T, typename IT> struct VectorIterator {
  typedef std::random_access_iterator_tag iterator_category;
  typedef IT value_type;
  typedef ptrdiff_t difference_type;
  typedef IT *pointer;
  typedef IT &reference;

  VectorIterator(const uint8_t *data, uoffset_t i)
      : data_(data + IndirectHelper<T>::element_stride * i) {}
  VectorIterator(const VectorIterator &other) : data_(other.data_) {}
  VectorIterator() : data_(nullptr) {}

  VectorIterator &operator=(const VectorIterator &other) {
    data_ = other.data_;
    return *this;
  }

  // clang-format off
  #if !defined(FLATBUFFERS_CPP98_STL)
  VectorIterator &operator=(VectorIterator &&other) {
    data_ = other.data_;
    return *this;
  }
  #endif  // !defined(FLATBUFFERS_CPP98_STL)
  // clang-format on

  bool operator==(const VectorIterator &other) const {
    return data_ == other.data_;
  }

  bool operator<(const VectorIterator &other) const {
    return data_ < other.data_;
  }

  bool operator!=(const VectorIterator &other) const {
    return data_ != other.data_;
  }

  difference_type operator-(const VectorIterator &other) const {
    return (data_ - other.data_) / IndirectHelper<T>::element_stride;
  }

  IT operator*() const { return IndirectHelper<T>::Read(data_, 0); }

  IT operator->() const { return IndirectHelper<T>::Read(data_, 0); }

  VectorIterator &operator++() {
    data_ += IndirectHelper<T>::element_stride;
    return *this;
  }

  VectorIterator operator++(int) {
    VectorIterator temp(data_, 0);
    data_ += IndirectHelper<T>::element_stride;
    return temp;
  }

  VectorIterator operator+(const uoffset_t &offset) const {
    return VectorIterator(data_ + offset * IndirectHelper<T>::element_stride,
                          0);
  }

  VectorIterator &operator+=(const uoffset_t &offset) {
    data_ += offset * IndirectHelper<T>::element_stride;
    return *this;
  }

  VectorIterator &operator--() {
    data_ -= IndirectHelper<T>::element_stride;
    return *this;
  }

  VectorIterator operator--(int) {
    VectorIterator temp(data_, 0);
    data_ -= IndirectHelper<T>::element_stride;
    return temp;
  }

  VectorIterator operator-(const uoffset_t &offset) const {
    return VectorIterator(data_ - offset * IndirectHelper<T>::element_stride,
                          0);
  }

  VectorIterator &operator-=(const uoffset_t &offset) {
    data_ -= offset * IndirectHelper<T>::element_stride;
    return *this;
  }

 private:
  const uint8_t *data_;
};

template<typename Iterator> struct VectorReverseIterator :
  public std::reverse_iterator<Iterator> {

  explicit VectorReverseIterator(Iterator iter) :
    std::reverse_iterator<Iterator>(iter) {}

  typename Iterator::value_type operator*() const {
    return *(std::reverse_iterator<Iterator>::current);
  }

  typename Iterator::value_type operator->() const {
    return *(std::reverse_iterator<Iterator>::current);
  }
};

struct String;

// This is used as a helper type for accessing vectors.
// Vector::data() assumes the vector elements start after the length field.
template<typename T> class Vector {
 public:
  typedef VectorIterator<T, typename IndirectHelper<T>::mutable_return_type>
      iterator;
  typedef VectorIterator<T, typename IndirectHelper<T>::return_type>
      const_iterator;
  typedef VectorReverseIterator<iterator> reverse_iterator;
  typedef VectorReverseIterator<const_iterator> const_reverse_iterator;

  uoffset_t size() const { return EndianScalar(length_); }

  // Deprecated: use size(). Here for backwards compatibility.
  FLATBUFFERS_ATTRIBUTE(deprecated("use size() instead"))
  uoffset_t Length() const { return size(); }

  typedef typename IndirectHelper<T>::return_type return_type;
  typedef typename IndirectHelper<T>::mutable_return_type mutable_return_type;

  return_type Get(uoffset_t i) const {
    FLATBUFFERS_ASSERT(i < size());
    return IndirectHelper<T>::Read(Data(), i);
  }

  return_type operator[](uoffset_t i) const { return Get(i); }

  // If this is a Vector of enums, T will be its storage type, not the enum
  // type. This function makes it convenient to retrieve value with enum
  // type E.
  template<typename E> E GetEnum(uoffset_t i) const {
    return static_cast<E>(Get(i));
  }

  // If this a vector of unions, this does the cast for you. There's no check
  // to make sure this is the right type!
  template<typename U> const U *GetAs(uoffset_t i) const {
    return reinterpret_cast<const U *>(Get(i));
  }

  // If this a vector of unions, this does the cast for you. There's no check
  // to make sure this is actually a string!
  const String *GetAsString(uoffset_t i) const {
    return reinterpret_cast<const String *>(Get(i));
  }

  const void *GetStructFromOffset(size_t o) const {
    return reinterpret_cast<const void *>(Data() + o);
  }

  iterator begin() { return iterator(Data(), 0); }
  const_iterator begin() const { return const_iterator(Data(), 0); }

  iterator end() { return iterator(Data(), size()); }
  const_iterator end() const { return const_iterator(Data(), size()); }

  reverse_iterator rbegin() { return reverse_iterator(end() - 1); }
  const_reverse_iterator rbegin() const { return const_reverse_iterator(end() - 1); }

  reverse_iterator rend() { return reverse_iterator(begin() - 1); }
  const_reverse_iterator rend() const { return const_reverse_iterator(begin() - 1); }

  const_iterator cbegin() const { return begin(); }

  const_iterator cend() const { return end(); }

  const_reverse_iterator crbegin() const { return rbegin(); }

  const_reverse_iterator crend() const { return rend(); }

  // Change elements if you have a non-const pointer to this object.
  // Scalars only. See reflection.h, and the documentation.
  void Mutate(uoffset_t i, const T &val) {
    FLATBUFFERS_ASSERT(i < size());
    WriteScalar(data() + i, val);
  }

  // Change an element of a vector of tables (or strings).
  // "val" points to the new table/string, as you can obtain from
  // e.g. reflection::AddFlatBuffer().
  void MutateOffset(uoffset_t i, const uint8_t *val) {
    FLATBUFFERS_ASSERT(i < size());
    static_assert(sizeof(T) == sizeof(uoffset_t), "Unrelated types");
    WriteScalar(data() + i,
                static_cast<uoffset_t>(val - (Data() + i * sizeof(uoffset_t))));
  }

  // Get a mutable pointer to tables/strings inside this vector.
  mutable_return_type GetMutableObject(uoffset_t i) const {
    FLATBUFFERS_ASSERT(i < size());
    return const_cast<mutable_return_type>(IndirectHelper<T>::Read(Data(), i));
  }

  // The raw data in little endian format. Use with care.
  const uint8_t *Data() const {
    return reinterpret_cast<const uint8_t *>(&length_ + 1);
  }

  uint8_t *Data() { return reinterpret_cast<uint8_t *>(&length_ + 1); }

  // Similarly, but typed, much like std::vector::data
  const T *data() const { return reinterpret_cast<const T *>(Data()); }
  T *data() { return reinterpret_cast<T *>(Data()); }

  template<typename K> return_type LookupByKey(K key) const {
    void *search_result = std::bsearch(
        &key, Data(), size(), IndirectHelper<T>::element_stride, KeyCompare<K>);

    if (!search_result) {
      return nullptr;  // Key not found.
    }

    const uint8_t *element = reinterpret_cast<const uint8_t *>(search_result);

    return IndirectHelper<T>::Read(element, 0);
  }

 protected:
  // This class is only used to access pre-existing data. Don't ever
  // try to construct these manually.
  Vector();

  uoffset_t length_;

 private:
  // This class is a pointer. Copying will therefore create an invalid object.
  // Private and unimplemented copy constructor.
  Vector(const Vector &);

  template<typename K> static int KeyCompare(const void *ap, const void *bp) {
    const K *key = reinterpret_cast<const K *>(ap);
    const uint8_t *data = reinterpret_cast<const uint8_t *>(bp);
    auto table = IndirectHelper<T>::Read(data, 0);

    // std::bsearch compares with the operands transposed, so we negate the
    // result here.
    return -table->KeyCompareWithValue(*key);
  }
};

// Represent a vector much like the template above, but in this case we
// don't know what the element types are (used with reflection.h).
class VectorOfAny {
 public:
  uoffset_t size() const { return EndianScalar(length_); }

  const uint8_t *Data() const {
    return reinterpret_cast<const uint8_t *>(&length_ + 1);
  }
  uint8_t *Data() { return reinterpret_cast<uint8_t *>(&length_ + 1); }

 protected:
  VectorOfAny();

  uoffset_t length_;

 private:
  VectorOfAny(const VectorOfAny &);
};

#ifndef FLATBUFFERS_CPP98_STL
template<typename T, typename U>
Vector<Offset<T>> *VectorCast(Vector<Offset<U>> *ptr) {
  static_assert(std::is_base_of<T, U>::value, "Unrelated types");
  return reinterpret_cast<Vector<Offset<T>> *>(ptr);
}

template<typename T, typename U>
const Vector<Offset<T>> *VectorCast(const Vector<Offset<U>> *ptr) {
  static_assert(std::is_base_of<T, U>::value, "Unrelated types");
  return reinterpret_cast<const Vector<Offset<T>> *>(ptr);
}
#endif

// Convenient helper function to get the length of any vector, regardless
// of whether it is null or not (the field is not set).
template<typename T> static inline size_t VectorLength(const Vector<T> *v) {
  return v ? v->size() : 0;
}

// This is used as a helper type for accessing arrays.
template<typename T, uint16_t length> class Array {
 public:
  typedef VectorIterator<T, typename IndirectHelper<T>::return_type>
      const_iterator;
  typedef VectorReverseIterator<const_iterator> const_reverse_iterator;

  typedef typename IndirectHelper<T>::return_type return_type;

  FLATBUFFERS_CONSTEXPR uint16_t size() const { return length; }

  return_type Get(uoffset_t i) const {
    FLATBUFFERS_ASSERT(i < size());
    return IndirectHelper<T>::Read(Data(), i);
  }

  return_type operator[](uoffset_t i) const { return Get(i); }

  const_iterator begin() const { return const_iterator(Data(), 0); }
  const_iterator end() const { return const_iterator(Data(), size()); }

  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(end());
  }
  const_reverse_iterator rend() const { return const_reverse_iterator(end()); }

  const_iterator cbegin() const { return begin(); }
  const_iterator cend() const { return end(); }

  const_reverse_iterator crbegin() const { return rbegin(); }
  const_reverse_iterator crend() const { return rend(); }

  // Change elements if you have a non-const pointer to this object.
  void Mutate(uoffset_t i, const T &val) {
    FLATBUFFERS_ASSERT(i < size());
    WriteScalar(data() + i, val);
  }

  // Get a mutable pointer to elements inside this array.
  // @note This method should be only used to mutate arrays of structs followed
  //  by a @p Mutate operation. For primitive types use @p Mutate directly.
  // @warning Assignments and reads to/from the dereferenced pointer are not
  //  automatically converted to the correct endianness.
  T *GetMutablePointer(uoffset_t i) const {
    FLATBUFFERS_ASSERT(i < size());
    return const_cast<T *>(&data()[i]);
  }

  // The raw data in little endian format. Use with care.
  const uint8_t *Data() const { return data_; }

  uint8_t *Data() { return data_; }

  // Similarly, but typed, much like std::vector::data
  const T *data() const { return reinterpret_cast<const T *>(Data()); }
  T *data() { return reinterpret_cast<T *>(Data()); }

 protected:
  // This class is only used to access pre-existing data. Don't ever
  // try to construct these manually.
  // 'constexpr' allows us to use 'size()' at compile time.
  // @note Must not use 'FLATBUFFERS_CONSTEXPR' here, as const is not allowed on
  //  a constructor.
#if defined(__cpp_constexpr)
  constexpr Array();
#else
  Array();
#endif

  uint8_t data_[length * sizeof(T)];

 private:
  // This class is a pointer. Copying will therefore create an invalid object.
  // Private and unimplemented copy constructor.
  Array(const Array &);
};

// Lexicographically compare two strings (possibly containing nulls), and
// return true if the first is less than the second.
static inline bool StringLessThan(const char *a_data, uoffset_t a_size,
                                  const char *b_data, uoffset_t b_size) {
  const auto cmp = memcmp(a_data, b_data, (std::min)(a_size, b_size));
  return cmp == 0 ? a_size < b_size : cmp < 0;
}

struct String : public Vector<char> {
  const char *c_str() const { return reinterpret_cast<const char *>(Data()); }
  std::string str() const { return std::string(c_str(), size()); }

  // clang-format off
  #ifdef FLATBUFFERS_HAS_STRING_VIEW
  flatbuffers::string_view string_view() const {
    return flatbuffers::string_view(c_str(), size());
  }
  #endif // FLATBUFFERS_HAS_STRING_VIEW
  // clang-format on

  bool operator<(const String &o) const {
    return StringLessThan(this->data(), this->size(), o.data(), o.size());
  }
};

// Convenience function to get std::string from a String returning an empty
// string on null pointer.
static inline std::string GetString(const String * str) {
  return str ? str->str() : "";
}

// Convenience function to get char* from a String returning an empty string on
// null pointer.
static inline const char * GetCstring(const String * str) {
  return str ? str->c_str() : "";
}

// Allocator interface. This is flatbuffers-specific and meant only for
// `vector_downward` usage.
class Allocator {
 public:
  virtual ~Allocator() {}

  // Allocate `size` bytes of memory.
  virtual uint8_t *allocate(size_t size) = 0;

  // Deallocate `size` bytes of memory at `p` allocated by this allocator.
  virtual void deallocate(uint8_t *p, size_t size) = 0;

  // Reallocate `new_size` bytes of memory, replacing the old region of size
  // `old_size` at `p`. In contrast to a normal realloc, this grows downwards,
  // and is intended specifcally for `vector_downward` use.
  // `in_use_back` and `in_use_front` indicate how much of `old_size` is
  // actually in use at each end, and needs to be copied.
  virtual uint8_t *reallocate_downward(uint8_t *old_p, size_t old_size,
                                       size_t new_size, size_t in_use_back,
                                       size_t in_use_front) {
    FLATBUFFERS_ASSERT(new_size > old_size);  // vector_downward only grows
    uint8_t *new_p = allocate(new_size);
    memcpy_downward(old_p, old_size, new_p, new_size, in_use_back,
                    in_use_front);
    deallocate(old_p, old_size);
    return new_p;
  }

 protected:
  // Called by `reallocate_downward` to copy memory from `old_p` of `old_size`
  // to `new_p` of `new_size`. Only memory of size `in_use_front` and
  // `in_use_back` will be copied from the front and back of the old memory
  // allocation.
  void memcpy_downward(uint8_t *old_p, size_t old_size,
                       uint8_t *new_p, size_t new_size,
                       size_t in_use_back, size_t in_use_front) {
    memcpy(new_p + new_size - in_use_back, old_p + old_size - in_use_back,
           in_use_back);
    memcpy(new_p, old_p, in_use_front);
  }
};

// DefaultAllocator uses new/delete to allocate memory regions
class DefaultAllocator : public Allocator {
 public:
  uint8_t *allocate(size_t size) FLATBUFFERS_OVERRIDE {
    return new uint8_t[size];
  }

  void deallocate(uint8_t *p, size_t) FLATBUFFERS_OVERRIDE {
    delete[] p;
  }

  static void dealloc(void *p, size_t) {
    delete[] static_cast<uint8_t *>(p);
  }
};

// These functions allow for a null allocator to mean use the default allocator,
// as used by DetachedBuffer and vector_downward below.
// This is to avoid having a statically or dynamically allocated default
// allocator, or having to move it between the classes that may own it.
inline uint8_t *Allocate(Allocator *allocator, size_t size) {
  return allocator ? allocator->allocate(size)
                   : DefaultAllocator().allocate(size);
}

inline void Deallocate(Allocator *allocator, uint8_t *p, size_t size) {
  if (allocator) allocator->deallocate(p, size);
  else DefaultAllocator().deallocate(p, size);
}

inline uint8_t *ReallocateDownward(Allocator *allocator, uint8_t *old_p,
                                   size_t old_size, size_t new_size,
                                   size_t in_use_back, size_t in_use_front) {
  return allocator
      ? allocator->reallocate_downward(old_p, old_size, new_size,
                                       in_use_back, in_use_front)
      : DefaultAllocator().reallocate_downward(old_p, old_size, new_size,
                                               in_use_back, in_use_front);
}

// DetachedBuffer is a finished flatbuffer memory region, detached from its
// builder. The original memory region and allocator are also stored so that
// the DetachedBuffer can manage the memory lifetime.
class DetachedBuffer {
 public:
  DetachedBuffer()
      : allocator_(nullptr),
        own_allocator_(false),
        buf_(nullptr),
        reserved_(0),
        cur_(nullptr),
        size_(0) {}

  DetachedBuffer(Allocator *allocator, bool own_allocator, uint8_t *buf,
                 size_t reserved, uint8_t *cur, size_t sz)
      : allocator_(allocator),
        own_allocator_(own_allocator),
        buf_(buf),
        reserved_(reserved),
        cur_(cur),
        size_(sz) {}

  // clang-format off
  #if !defined(FLATBUFFERS_CPP98_STL)
  // clang-format on
  DetachedBuffer(DetachedBuffer &&other)
      : allocator_(other.allocator_),
        own_allocator_(other.own_allocator_),
        buf_(other.buf_),
        reserved_(other.reserved_),
        cur_(other.cur_),
        size_(other.size_) {
    other.reset();
  }
  // clang-format off
  #endif  // !defined(FLATBUFFERS_CPP98_STL)
  // clang-format on

  // clang-format off
  #if !defined(FLATBUFFERS_CPP98_STL)
  // clang-format on
  DetachedBuffer &operator=(DetachedBuffer &&other) {
    destroy();

    allocator_ = other.allocator_;
    own_allocator_ = other.own_allocator_;
    buf_ = other.buf_;
    reserved_ = other.reserved_;
    cur_ = other.cur_;
    size_ = other.size_;

    other.reset();

    return *this;
  }
  // clang-format off
  #endif  // !defined(FLATBUFFERS_CPP98_STL)
  // clang-format on

  ~DetachedBuffer() { destroy(); }

  const uint8_t *data() const { return cur_; }

  uint8_t *data() { return cur_; }

  size_t size() const { return size_; }

  // clang-format off
  #if 0  // disabled for now due to the ordering of classes in this header
  template <class T>
  bool Verify() const {
    Verifier verifier(data(), size());
    return verifier.Verify<T>(nullptr);
  }

  template <class T>
  const T* GetRoot() const {
    return flatbuffers::GetRoot<T>(data());
  }

  template <class T>
  T* GetRoot() {
    return flatbuffers::GetRoot<T>(data());
  }
  #endif
  // clang-format on

  // clang-format off
  #if !defined(FLATBUFFERS_CPP98_STL)
  // clang-format on
  // These may change access mode, leave these at end of public section
  FLATBUFFERS_DELETE_FUNC(DetachedBuffer(const DetachedBuffer &other))
  FLATBUFFERS_DELETE_FUNC(
      DetachedBuffer &operator=(const DetachedBuffer &other))
  // clang-format off
  #endif  // !defined(FLATBUFFERS_CPP98_STL)
  // clang-format on

protected:
  Allocator *allocator_;
  bool own_allocator_;
  uint8_t *buf_;
  size_t reserved_;
  uint8_t *cur_;
  size_t size_;

  inline void destroy() {
    if (buf_) Deallocate(allocator_, buf_, reserved_);
    if (own_allocator_ && allocator_) { delete allocator_; }
    reset();
  }

  inline void reset() {
    allocator_ = nullptr;
    own_allocator_ = false;
    buf_ = nullptr;
    reserved_ = 0;
    cur_ = nullptr;
    size_ = 0;
  }
};

// This is a minimal replication of std::vector<uint8_t> functionality,
// except growing from higher to lower addresses. i.e push_back() inserts data
// in the lowest address in the vector.
// Since this vector leaves the lower part unused, we support a "scratch-pad"
// that can be stored there for temporary data, to share the allocated space.
// Essentially, this supports 2 std::vectors in a single buffer.
class vector_downward {
 public:
  explicit vector_downward(size_t initial_size,
                           Allocator *allocator,
                           bool own_allocator,
                           size_t buffer_minalign)
      : allocator_(allocator),
        own_allocator_(own_allocator),
        initial_size_(initial_size),
        buffer_minalign_(buffer_minalign),
        reserved_(0),
        buf_(nullptr),
        cur_(nullptr),
        scratch_(nullptr) {}

  // clang-format off
  #if !defined(FLATBUFFERS_CPP98_STL)
  vector_downward(vector_downward &&other)
  #else
  vector_downward(vector_downward &other)
  #endif  // defined(FLATBUFFERS_CPP98_STL)
  // clang-format on
    : allocator_(other.allocator_),
      own_allocator_(other.own_allocator_),
      initial_size_(other.initial_size_),
      buffer_minalign_(other.buffer_minalign_),
      reserved_(other.reserved_),
      buf_(other.buf_),
      cur_(other.cur_),
      scratch_(other.scratch_) {
    // No change in other.allocator_
    // No change in other.initial_size_
    // No change in other.buffer_minalign_
    other.own_allocator_ = false;
    other.reserved_ = 0;
    other.buf_ = nullptr;
    other.cur_ = nullptr;
    other.scratch_ = nullptr;
  }

  // clang-format off
  #if !defined(FLATBUFFERS_CPP98_STL)
  // clang-format on
  vector_downward &operator=(vector_downward &&other) {
    // Move construct a temporary and swap idiom
    vector_downward temp(std::move(other));
    swap(temp);
    return *this;
  }
  // clang-format off
  #endif  // defined(FLATBUFFERS_CPP98_STL)
  // clang-format on

  ~vector_downward() {
    clear_buffer();
    clear_allocator();
  }

  void reset() {
    clear_buffer();
    clear();
  }

  void clear() {
    if (buf_) {
      cur_ = buf_ + reserved_;
    } else {
      reserved_ = 0;
      cur_ = nullptr;
    }
    clear_scratch();
  }

  void clear_scratch() {
    scratch_ = buf_;
  }

  void clear_allocator() {
    if (own_allocator_ && allocator_) { delete allocator_; }
    allocator_ = nullptr;
    own_allocator_ = false;
  }

  void clear_buffer() {
    if (buf_) Deallocate(allocator_, buf_, reserved_);
    buf_ = nullptr;
  }

  // Relinquish the pointer to the caller.
  uint8_t *release_raw(size_t &allocated_bytes, size_t &offset) {
    auto *buf = buf_;
    allocated_bytes = reserved_;
    offset = static_cast<size_t>(cur_ - buf_);

    // release_raw only relinquishes the buffer ownership.
    // Does not deallocate or reset the allocator. Destructor will do that.
    buf_ = nullptr;
    clear();
    return buf;
  }

  // Relinquish the pointer to the caller.
  DetachedBuffer release() {
    // allocator ownership (if any) is transferred to DetachedBuffer.
    DetachedBuffer fb(allocator_, own_allocator_, buf_, reserved_, cur_,
                      size());
    if (own_allocator_) {
      allocator_ = nullptr;
      own_allocator_ = false;
    }
    buf_ = nullptr;
    clear();
    return fb;
  }

  size_t ensure_space(size_t len) {
    FLATBUFFERS_ASSERT(cur_ >= scratch_ && scratch_ >= buf_);
    if (len > static_cast<size_t>(cur_ - scratch_)) { reallocate(len); }
    // Beyond this, signed offsets may not have enough range:
    // (FlatBuffers > 2GB not supported).
    FLATBUFFERS_ASSERT(size() < FLATBUFFERS_MAX_BUFFER_SIZE);
    return len;
  }

  inline uint8_t *make_space(size_t len) {
    size_t space = ensure_space(len);
    cur_ -= space;
    return cur_;
  }

  // Returns nullptr if using the DefaultAllocator.
  Allocator *get_custom_allocator() { return allocator_; }

  uoffset_t size() const {
    return static_cast<uoffset_t>(reserved_ - (cur_ - buf_));
  }

  uoffset_t scratch_size() const {
    return static_cast<uoffset_t>(scratch_ - buf_);
  }

  size_t capacity() const { return reserved_; }

  uint8_t *data() const {
    FLATBUFFERS_ASSERT(cur_);
    return cur_;
  }

  uint8_t *scratch_data() const {
    FLATBUFFERS_ASSERT(buf_);
    return buf_;
  }

  uint8_t *scratch_end() const {
    FLATBUFFERS_ASSERT(scratch_);
    return scratch_;
  }

  uint8_t *data_at(size_t offset) const { return buf_ + reserved_ - offset; }

  void push(const uint8_t *bytes, size_t num) {
    if (num > 0) { memcpy(make_space(num), bytes, num); }
  }

  // Specialized version of push() that avoids memcpy call for small data.
  template<typename T> void push_small(const T &little_endian_t) {
    make_space(sizeof(T));
    *reinterpret_cast<T *>(cur_) = little_endian_t;
  }

  template<typename T> void scratch_push_small(const T &t) {
    ensure_space(sizeof(T));
    *reinterpret_cast<T *>(scratch_) = t;
    scratch_ += sizeof(T);
  }

  // fill() is most frequently called with small byte counts (<= 4),
  // which is why we're using loops rather than calling memset.
  void fill(size_t zero_pad_bytes) {
    make_space(zero_pad_bytes);
    for (size_t i = 0; i < zero_pad_bytes; i++) cur_[i] = 0;
  }

  // Version for when we know the size is larger.
  // Precondition: zero_pad_bytes > 0
  void fill_big(size_t zero_pad_bytes) {
    memset(make_space(zero_pad_bytes), 0, zero_pad_bytes);
  }

  void pop(size_t bytes_to_remove) { cur_ += bytes_to_remove; }
  void scratch_pop(size_t bytes_to_remove) { scratch_ -= bytes_to_remove; }

  void swap(vector_downward &other) {
    using std::swap;
    swap(allocator_, other.allocator_);
    swap(own_allocator_, other.own_allocator_);
    swap(initial_size_, other.initial_size_);
    swap(buffer_minalign_, other.buffer_minalign_);
    swap(reserved_, other.reserved_);
    swap(buf_, other.buf_);
    swap(cur_, other.cur_);
    swap(scratch_, other.scratch_);
  }

  void swap_allocator(vector_downward &other) {
    using std::swap;
    swap(allocator_, other.allocator_);
    swap(own_allocator_, other.own_allocator_);
  }

 private:
  // You shouldn't really be copying instances of this class.
  FLATBUFFERS_DELETE_FUNC(vector_downward(const vector_downward &))
  FLATBUFFERS_DELETE_FUNC(vector_downward &operator=(const vector_downward &))

  Allocator *allocator_;
  bool own_allocator_;
  size_t initial_size_;
  size_t buffer_minalign_;
  size_t reserved_;
  uint8_t *buf_;
  uint8_t *cur_;  // Points at location between empty (below) and used (above).
  uint8_t *scratch_;  // Points to the end of the scratchpad in use.

  void reallocate(size_t len) {
    auto old_reserved = reserved_;
    auto old_size = size();
    auto old_scratch_size = scratch_size();
    reserved_ += (std::max)(len,
                            old_reserved ? old_reserved / 2 : initial_size_);
    reserved_ = (reserved_ + buffer_minalign_ - 1) & ~(buffer_minalign_ - 1);
    if (buf_) {
      buf_ = ReallocateDownward(allocator_, buf_, old_reserved, reserved_,
                                old_size, old_scratch_size);
    } else {
      buf_ = Allocate(allocator_, reserved_);
    }
    cur_ = buf_ + reserved_ - old_size;
    scratch_ = buf_ + old_scratch_size;
  }
};

// Converts a Field ID to a virtual table offset.
inline voffset_t FieldIndexToOffset(voffset_t field_id) {
  // Should correspond to what EndTable() below builds up.
  const int fixed_fields = 2;  // Vtable size and Object Size.
  return static_cast<voffset_t>((field_id + fixed_fields) * sizeof(voffset_t));
}

template<typename T, typename Alloc>
const T *data(const std::vector<T, Alloc> &v) {
  // Eventually the returned pointer gets passed down to memcpy, so
  // we need it to be non-null to avoid undefined behavior.
  static uint8_t t;
  return v.empty() ? reinterpret_cast<const T*>(&t) : &v.front();
}
template<typename T, typename Alloc> T *data(std::vector<T, Alloc> &v) {
  // Eventually the returned pointer gets passed down to memcpy, so
  // we need it to be non-null to avoid undefined behavior.
  static uint8_t t;
  return v.empty() ? reinterpret_cast<T*>(&t) : &v.front();
}

/// @endcond

/// @addtogroup flatbuffers_cpp_api
/// @{
/// @class FlatBufferBuilder
/// @brief Helper class to hold data needed in creation of a FlatBuffer.
/// To serialize data, you typically call one of the `Create*()` functions in
/// the generated code, which in turn call a sequence of `StartTable`/
/// `PushElement`/`AddElement`/`EndTable`, or the builtin `CreateString`/
/// `CreateVector` functions. Do this is depth-first order to build up a tree to
/// the root. `Finish()` wraps up the buffer ready for transport.
class FlatBufferBuilder {
 public:
  /// @brief Default constructor for FlatBufferBuilder.
  /// @param[in] initial_size The initial size of the buffer, in bytes. Defaults
  /// to `1024`.
  /// @param[in] allocator An `Allocator` to use. If null will use
  /// `DefaultAllocator`.
  /// @param[in] own_allocator Whether the builder/vector should own the
  /// allocator. Defaults to / `false`.
  /// @param[in] buffer_minalign Force the buffer to be aligned to the given
  /// minimum alignment upon reallocation. Only needed if you intend to store
  /// types with custom alignment AND you wish to read the buffer in-place
  /// directly after creation.
  explicit FlatBufferBuilder(size_t initial_size = 1024,
                             Allocator *allocator = nullptr,
                             bool own_allocator = false,
                             size_t buffer_minalign =
                                 AlignOf<largest_scalar_t>())
      : buf_(initial_size, allocator, own_allocator, buffer_minalign),
        num_field_loc(0),
        max_voffset_(0),
        nested(false),
        finished(false),
        minalign_(1),
        force_defaults_(false),
        dedup_vtables_(true),
        string_pool(nullptr) {
    EndianCheck();
  }

  // clang-format off
  /// @brief Move constructor for FlatBufferBuilder.
  #if !defined(FLATBUFFERS_CPP98_STL)
  FlatBufferBuilder(FlatBufferBuilder &&other)
  #else
  FlatBufferBuilder(FlatBufferBuilder &other)
  #endif  // #if !defined(FLATBUFFERS_CPP98_STL)
    : buf_(1024, nullptr, false, AlignOf<largest_scalar_t>()),
      num_field_loc(0),
      max_voffset_(0),
      nested(false),
      finished(false),
      minalign_(1),
      force_defaults_(false),
      dedup_vtables_(true),
      string_pool(nullptr) {
    EndianCheck();
    // Default construct and swap idiom.
    // Lack of delegating constructors in vs2010 makes it more verbose than needed.
    Swap(other);
  }
  // clang-format on

  // clang-format off
  #if !defined(FLATBUFFERS_CPP98_STL)
  // clang-format on
  /// @brief Move assignment operator for FlatBufferBuilder.
  FlatBufferBuilder &operator=(FlatBufferBuilder &&other) {
    // Move construct a temporary and swap idiom
    FlatBufferBuilder temp(std::move(other));
    Swap(temp);
    return *this;
  }
  // clang-format off
  #endif  // defined(FLATBUFFERS_CPP98_STL)
  // clang-format on

  void Swap(FlatBufferBuilder &other) {
    using std::swap;
    buf_.swap(other.buf_);
    swap(num_field_loc, other.num_field_loc);
    swap(max_voffset_, other.max_voffset_);
    swap(nested, other.nested);
    swap(finished, other.finished);
    swap(minalign_, other.minalign_);
    swap(force_defaults_, other.force_defaults_);
    swap(dedup_vtables_, other.dedup_vtables_);
    swap(string_pool, other.string_pool);
  }

  ~FlatBufferBuilder() {
    if (string_pool) delete string_pool;
  }

  void Reset() {
    Clear();       // clear builder state
    buf_.reset();  // deallocate buffer
  }

  /// @brief Reset all the state in this FlatBufferBuilder so it can be reused
  /// to construct another buffer.
  void Clear() {
    ClearOffsets();
    buf_.clear();
    nested = false;
    finished = false;
    minalign_ = 1;
    if (string_pool) string_pool->clear();
  }

  /// @brief The current size of the serialized buffer, counting from the end.
  /// @return Returns an `uoffset_t` with the current size of the buffer.
  uoffset_t GetSize() const { return buf_.size(); }

  /// @brief Get the serialized buffer (after you call `Finish()`).
  /// @return Returns an `uint8_t` pointer to the FlatBuffer data inside the
  /// buffer.
  uint8_t *GetBufferPointer() const {
    Finished();
    return buf_.data();
  }

  /// @brief Get a pointer to an unfinished buffer.
  /// @return Returns a `uint8_t` pointer to the unfinished buffer.
  uint8_t *GetCurrentBufferPointer() const { return buf_.data(); }

  /// @brief Get the released pointer to the serialized buffer.
  /// @warning Do NOT attempt to use this FlatBufferBuilder afterwards!
  /// @return A `FlatBuffer` that owns the buffer and its allocator and
  /// behaves similar to a `unique_ptr` with a deleter.
  FLATBUFFERS_ATTRIBUTE(deprecated("use Release() instead")) DetachedBuffer
  ReleaseBufferPointer() {
    Finished();
    return buf_.release();
  }

  /// @brief Get the released DetachedBuffer.
  /// @return A `DetachedBuffer` that owns the buffer and its allocator.
  DetachedBuffer Release() {
    Finished();
    return buf_.release();
  }

  /// @brief Get the released pointer to the serialized buffer.
  /// @param size The size of the memory block containing
  /// the serialized `FlatBuffer`.
  /// @param offset The offset from the released pointer where the finished
  /// `FlatBuffer` starts.
  /// @return A raw pointer to the start of the memory block containing
  /// the serialized `FlatBuffer`.
  /// @remark If the allocator is owned, it gets deleted when the destructor is called..
  uint8_t *ReleaseRaw(size_t &size, size_t &offset) {
    Finished();
    return buf_.release_raw(size, offset);
  }

  /// @brief get the minimum alignment this buffer needs to be accessed
  /// properly. This is only known once all elements have been written (after
  /// you call Finish()). You can use this information if you need to embed
  /// a FlatBuffer in some other buffer, such that you can later read it
  /// without first having to copy it into its own buffer.
  size_t GetBufferMinAlignment() {
    Finished();
    return minalign_;
  }

  /// @cond FLATBUFFERS_INTERNAL
  void Finished() const {
    // If you get this assert, you're attempting to get access a buffer
    // which hasn't been finished yet. Be sure to call
    // FlatBufferBuilder::Finish with your root table.
    // If you really need to access an unfinished buffer, call
    // GetCurrentBufferPointer instead.
    FLATBUFFERS_ASSERT(finished);
  }
  /// @endcond

  /// @brief In order to save space, fields that are set to their default value
  /// don't get serialized into the buffer.
  /// @param[in] fd When set to `true`, always serializes default values that are set.
  /// Optional fields which are not set explicitly, will still not be serialized.
  void ForceDefaults(bool fd) { force_defaults_ = fd; }

  /// @brief By default vtables are deduped in order to save space.
  /// @param[in] dedup When set to `true`, dedup vtables.
  void DedupVtables(bool dedup) { dedup_vtables_ = dedup; }

  /// @cond FLATBUFFERS_INTERNAL
  void Pad(size_t num_bytes) { buf_.fill(num_bytes); }

  void TrackMinAlign(size_t elem_size) {
    if (elem_size > minalign_) minalign_ = elem_size;
  }

  void Align(size_t elem_size) {
    TrackMinAlign(elem_size);
    buf_.fill(PaddingBytes(buf_.size(), elem_size));
  }

  void PushFlatBuffer(const uint8_t *bytes, size_t size) {
    PushBytes(bytes, size);
    finished = true;
  }

  void PushBytes(const uint8_t *bytes, size_t size) { buf_.push(bytes, size); }

  void PopBytes(size_t amount) { buf_.pop(amount); }

  template<typename T> void AssertScalarT() {
    // The code assumes power of 2 sizes and endian-swap-ability.
    static_assert(flatbuffers::is_scalar<T>::value, "T must be a scalar type");
  }

  // Write a single aligned scalar to the buffer
  template<typename T> uoffset_t PushElement(T element) {
    AssertScalarT<T>();
    T litle_endian_element = EndianScalar(element);
    Align(sizeof(T));
    buf_.push_small(litle_endian_element);
    return GetSize();
  }

  template<typename T> uoffset_t PushElement(Offset<T> off) {
    // Special case for offsets: see ReferTo below.
    return PushElement(ReferTo(off.o));
  }

  // When writing fields, we track where they are, so we can create correct
  // vtables later.
  void TrackField(voffset_t field, uoffset_t off) {
    FieldLoc fl = { off, field };
    buf_.scratch_push_small(fl);
    num_field_loc++;
    max_voffset_ = (std::max)(max_voffset_, field);
  }

  // Like PushElement, but additionally tracks the field this represents.
  template<typename T> void AddElement(voffset_t field, T e, T def) {
    // We don't serialize values equal to the default.
    if (IsTheSameAs(e, def) && !force_defaults_) return;
    auto off = PushElement(e);
    TrackField(field, off);
  }

  template<typename T> void AddOffset(voffset_t field, Offset<T> off) {
    if (off.IsNull()) return;  // Don't store.
    AddElement(field, ReferTo(off.o), static_cast<uoffset_t>(0));
  }

  template<typename T> void AddStruct(voffset_t field, const T *structptr) {
    if (!structptr) return;  // Default, don't store.
    Align(AlignOf<T>());
    buf_.push_small(*structptr);
    TrackField(field, GetSize());
  }

  void AddStructOffset(voffset_t field, uoffset_t off) {
    TrackField(field, off);
  }

  // Offsets initially are relative to the end of the buffer (downwards).
  // This function converts them to be relative to the current location
  // in the buffer (when stored here), pointing upwards.
  uoffset_t ReferTo(uoffset_t off) {
    // Align to ensure GetSize() below is correct.
    Align(sizeof(uoffset_t));
    // Offset must refer to something already in buffer.
    FLATBUFFERS_ASSERT(off && off <= GetSize());
    return GetSize() - off + static_cast<uoffset_t>(sizeof(uoffset_t));
  }

  void NotNested() {
    // If you hit this, you're trying to construct a Table/Vector/String
    // during the construction of its parent table (between the MyTableBuilder
    // and table.Finish().
    // Move the creation of these sub-objects to above the MyTableBuilder to
    // not get this assert.
    // Ignoring this assert may appear to work in simple cases, but the reason
    // it is here is that storing objects in-line may cause vtable offsets
    // to not fit anymore. It also leads to vtable duplication.
    FLATBUFFERS_ASSERT(!nested);
    // If you hit this, fields were added outside the scope of a table.
    FLATBUFFERS_ASSERT(!num_field_loc);
  }

  // From generated code (or from the parser), we call StartTable/EndTable
  // with a sequence of AddElement calls in between.
  uoffset_t StartTable() {
    NotNested();
    nested = true;
    return GetSize();
  }

  // This finishes one serialized object by generating the vtable if it's a
  // table, comparing it against existing vtables, and writing the
  // resulting vtable offset.
  uoffset_t EndTable(uoffset_t start) {
    // If you get this assert, a corresponding StartTable wasn't called.
    FLATBUFFERS_ASSERT(nested);
    // Write the vtable offset, which is the start of any Table.
    // We fill it's value later.
    auto vtableoffsetloc = PushElement<soffset_t>(0);
    // Write a vtable, which consists entirely of voffset_t elements.
    // It starts with the number of offsets, followed by a type id, followed
    // by the offsets themselves. In reverse:
    // Include space for the last offset and ensure empty tables have a
    // minimum size.
    max_voffset_ =
        (std::max)(static_cast<voffset_t>(max_voffset_ + sizeof(voffset_t)),
                   FieldIndexToOffset(0));
    buf_.fill_big(max_voffset_);
    auto table_object_size = vtableoffsetloc - start;
    // Vtable use 16bit offsets.
    FLATBUFFERS_ASSERT(table_object_size < 0x10000);
    WriteScalar<voffset_t>(buf_.data() + sizeof(voffset_t),
                           static_cast<voffset_t>(table_object_size));
    WriteScalar<voffset_t>(buf_.data(), max_voffset_);
    // Write the offsets into the table
    for (auto it = buf_.scratch_end() - num_field_loc * sizeof(FieldLoc);
         it < buf_.scratch_end(); it += sizeof(FieldLoc)) {
      auto field_location = reinterpret_cast<FieldLoc *>(it);
      auto pos = static_cast<voffset_t>(vtableoffsetloc - field_location->off);
      // If this asserts, it means you've set a field twice.
      FLATBUFFERS_ASSERT(
          !ReadScalar<voffset_t>(buf_.data() + field_location->id));
      WriteScalar<voffset_t>(buf_.data() + field_location->id, pos);
    }
    ClearOffsets();
    auto vt1 = reinterpret_cast<voffset_t *>(buf_.data());
    auto vt1_size = ReadScalar<voffset_t>(vt1);
    auto vt_use = GetSize();
    // See if we already have generated a vtable with this exact same
    // layout before. If so, make it point to the old one, remove this one.
    if (dedup_vtables_) {
      for (auto it = buf_.scratch_data(); it < buf_.scratch_end();
           it += sizeof(uoffset_t)) {
        auto vt_offset_ptr = reinterpret_cast<uoffset_t *>(it);
        auto vt2 = reinterpret_cast<voffset_t *>(buf_.data_at(*vt_offset_ptr));
        auto vt2_size = *vt2;
        if (vt1_size != vt2_size || 0 != memcmp(vt2, vt1, vt1_size)) continue;
        vt_use = *vt_offset_ptr;
        buf_.pop(GetSize() - vtableoffsetloc);
        break;
      }
    }
    // If this is a new vtable, remember it.
    if (vt_use == GetSize()) { buf_.scratch_push_small(vt_use); }
    // Fill the vtable offset we created above.
    // The offset points from the beginning of the object to where the
    // vtable is stored.
    // Offsets default direction is downward in memory for future format
    // flexibility (storing all vtables at the start of the file).
    WriteScalar(buf_.data_at(vtableoffsetloc),
                static_cast<soffset_t>(vt_use) -
                    static_cast<soffset_t>(vtableoffsetloc));

    nested = false;
    return vtableoffsetloc;
  }

  FLATBUFFERS_ATTRIBUTE(deprecated("call the version above instead"))
  uoffset_t EndTable(uoffset_t start, voffset_t /*numfields*/) {
    return EndTable(start);
  }

  // This checks a required field has been set in a given table that has
  // just been constructed.
  template<typename T> void Required(Offset<T> table, voffset_t field);

  uoffset_t StartStruct(size_t alignment) {
    Align(alignment);
    return GetSize();
  }

  uoffset_t EndStruct() { return GetSize(); }

  void ClearOffsets() {
    buf_.scratch_pop(num_field_loc * sizeof(FieldLoc));
    num_field_loc = 0;
    max_voffset_ = 0;
  }

  // Aligns such that when "len" bytes are written, an object can be written
  // after it with "alignment" without padding.
  void PreAlign(size_t len, size_t alignment) {
    TrackMinAlign(alignment);
    buf_.fill(PaddingBytes(GetSize() + len, alignment));
  }
  template<typename T> void PreAlign(size_t len) {
    AssertScalarT<T>();
    PreAlign(len, sizeof(T));
  }
  /// @endcond

  /// @brief Store a string in the buffer, which can contain any binary data.
  /// @param[in] str A const char pointer to the data to be stored as a string.
  /// @param[in] len The number of bytes that should be stored from `str`.
  /// @return Returns the offset in the buffer where the string starts.
  Offset<String> CreateString(const char *str, size_t len) {
    NotNested();
    PreAlign<uoffset_t>(len + 1);  // Always 0-terminated.
    buf_.fill(1);
    PushBytes(reinterpret_cast<const uint8_t *>(str), len);
    PushElement(static_cast<uoffset_t>(len));
    return Offset<String>(GetSize());
  }

  /// @brief Store a string in the buffer, which is null-terminated.
  /// @param[in] str A const char pointer to a C-string to add to the buffer.
  /// @return Returns the offset in the buffer where the string starts.
  Offset<String> CreateString(const char *str) {
    return CreateString(str, strlen(str));
  }

  /// @brief Store a string in the buffer, which is null-terminated.
  /// @param[in] str A char pointer to a C-string to add to the buffer.
  /// @return Returns the offset in the buffer where the string starts.
  Offset<String> CreateString(char *str) {
    return CreateString(str, strlen(str));
  }

  /// @brief Store a string in the buffer, which can contain any binary data.
  /// @param[in] str A const reference to a std::string to store in the buffer.
  /// @return Returns the offset in the buffer where the string starts.
  Offset<String> CreateString(const std::string &str) {
    return CreateString(str.c_str(), str.length());
  }

  // clang-format off
  #ifdef FLATBUFFERS_HAS_STRING_VIEW
  /// @brief Store a string in the buffer, which can contain any binary data.
  /// @param[in] str A const string_view to copy in to the buffer.
  /// @return Returns the offset in the buffer where the string starts.
  Offset<String> CreateString(flatbuffers::string_view str) {
    return CreateString(str.data(), str.size());
  }
  #endif // FLATBUFFERS_HAS_STRING_VIEW
  // clang-format on

  /// @brief Store a string in the buffer, which can contain any binary data.
  /// @param[in] str A const pointer to a `String` struct to add to the buffer.
  /// @return Returns the offset in the buffer where the string starts
  Offset<String> CreateString(const String *str) {
    return str ? CreateString(str->c_str(), str->size()) : 0;
  }

  /// @brief Store a string in the buffer, which can contain any binary data.
  /// @param[in] str A const reference to a std::string like type with support
  /// of T::c_str() and T::length() to store in the buffer.
  /// @return Returns the offset in the buffer where the string starts.
  template<typename T> Offset<String> CreateString(const T &str) {
    return CreateString(str.c_str(), str.length());
  }

  /// @brief Store a string in the buffer, which can contain any binary data.
  /// If a string with this exact contents has already been serialized before,
  /// instead simply returns the offset of the existing string.
  /// @param[in] str A const char pointer to the data to be stored as a string.
  /// @param[in] len The number of bytes that should be stored from `str`.
  /// @return Returns the offset in the buffer where the string starts.
  Offset<String> CreateSharedString(const char *str, size_t len) {
    if (!string_pool)
      string_pool = new StringOffsetMap(StringOffsetCompare(buf_));
    auto size_before_string = buf_.size();
    // Must first serialize the string, since the set is all offsets into
    // buffer.
    auto off = CreateString(str, len);
    auto it = string_pool->find(off);
    // If it exists we reuse existing serialized data!
    if (it != string_pool->end()) {
      // We can remove the string we serialized.
      buf_.pop(buf_.size() - size_before_string);
      return *it;
    }
    // Record this string for future use.
    string_pool->insert(off);
    return off;
  }

  /// @brief Store a string in the buffer, which null-terminated.
  /// If a string with this exact contents has already been serialized before,
  /// instead simply returns the offset of the existing string.
  /// @param[in] str A const char pointer to a C-string to add to the buffer.
  /// @return Returns the offset in the buffer where the string starts.
  Offset<String> CreateSharedString(const char *str) {
    return CreateSharedString(str, strlen(str));
  }

  /// @brief Store a string in the buffer, which can contain any binary data.
  /// If a string with this exact contents has already been serialized before,
  /// instead simply returns the offset of the existing string.
  /// @param[in] str A const reference to a std::string to store in the buffer.
  /// @return Returns the offset in the buffer where the string starts.
  Offset<String> CreateSharedString(const std::string &str) {
    return CreateSharedString(str.c_str(), str.length());
  }

  /// @brief Store a string in the buffer, which can contain any binary data.
  /// If a string with this exact contents has already been serialized before,
  /// instead simply returns the offset of the existing string.
  /// @param[in] str A const pointer to a `String` struct to add to the buffer.
  /// @return Returns the offset in the buffer where the string starts
  Offset<String> CreateSharedString(const String *str) {
    return CreateSharedString(str->c_str(), str->size());
  }

  /// @cond FLATBUFFERS_INTERNAL
  uoffset_t EndVector(size_t len) {
    FLATBUFFERS_ASSERT(nested);  // Hit if no corresponding StartVector.
    nested = false;
    return PushElement(static_cast<uoffset_t>(len));
  }

  void StartVector(size_t len, size_t elemsize) {
    NotNested();
    nested = true;
    PreAlign<uoffset_t>(len * elemsize);
    PreAlign(len * elemsize, elemsize);  // Just in case elemsize > uoffset_t.
  }

  // Call this right before StartVector/CreateVector if you want to force the
  // alignment to be something different than what the element size would
  // normally dictate.
  // This is useful when storing a nested_flatbuffer in a vector of bytes,
  // or when storing SIMD floats, etc.
  void ForceVectorAlignment(size_t len, size_t elemsize, size_t alignment) {
    PreAlign(len * elemsize, alignment);
  }

  // Similar to ForceVectorAlignment but for String fields.
  void ForceStringAlignment(size_t len, size_t alignment) {
    PreAlign((len + 1) * sizeof(char), alignment);
  }

  /// @endcond

  /// @brief Serialize an array into a FlatBuffer `vector`.
  /// @tparam T The data type of the array elements.
  /// @param[in] v A pointer to the array of type `T` to serialize into the
  /// buffer as a `vector`.
  /// @param[in] len The number of elements to serialize.
  /// @return Returns a typed `Offset` into the serialized data indicating
  /// where the vector is stored.
  template<typename T> Offset<Vector<T>> CreateVector(const T *v, size_t len) {
    // If this assert hits, you're specifying a template argument that is
    // causing the wrong overload to be selected, remove it.
    AssertScalarT<T>();
    StartVector(len, sizeof(T));
    // clang-format off
    #if FLATBUFFERS_LITTLEENDIAN
      PushBytes(reinterpret_cast<const uint8_t *>(v), len * sizeof(T));
    #else
      if (sizeof(T) == 1) {
        PushBytes(reinterpret_cast<const uint8_t *>(v), len);
      } else {
        for (auto i = len; i > 0; ) {
          PushElement(v[--i]);
        }
      }
    #endif
    // clang-format on
    return Offset<Vector<T>>(EndVector(len));
  }

  template<typename T>
  Offset<Vector<Offset<T>>> CreateVector(const Offset<T> *v, size_t len) {
    StartVector(len, sizeof(Offset<T>));
    for (auto i = len; i > 0;) { PushElement(v[--i]); }
    return Offset<Vector<Offset<T>>>(EndVector(len));
  }

  /// @brief Serialize a `std::vector` into a FlatBuffer `vector`.
  /// @tparam T The data type of the `std::vector` elements.
  /// @param v A const reference to the `std::vector` to serialize into the
  /// buffer as a `vector`.
  /// @return Returns a typed `Offset` into the serialized data indicating
  /// where the vector is stored.
  template<typename T> Offset<Vector<T>> CreateVector(const std::vector<T> &v) {
    return CreateVector(data(v), v.size());
  }

  // vector<bool> may be implemented using a bit-set, so we can't access it as
  // an array. Instead, read elements manually.
  // Background: https://isocpp.org/blog/2012/11/on-vectorbool
  Offset<Vector<uint8_t>> CreateVector(const std::vector<bool> &v) {
    StartVector(v.size(), sizeof(uint8_t));
    for (auto i = v.size(); i > 0;) {
      PushElement(static_cast<uint8_t>(v[--i]));
    }
    return Offset<Vector<uint8_t>>(EndVector(v.size()));
  }

  // clang-format off
  #ifndef FLATBUFFERS_CPP98_STL
  /// @brief Serialize values returned by a function into a FlatBuffer `vector`.
  /// This is a convenience function that takes care of iteration for you.
  /// @tparam T The data type of the `std::vector` elements.
  /// @param f A function that takes the current iteration 0..vector_size-1 and
  /// returns any type that you can construct a FlatBuffers vector out of.
  /// @return Returns a typed `Offset` into the serialized data indicating
  /// where the vector is stored.
  template<typename T> Offset<Vector<T>> CreateVector(size_t vector_size,
      const std::function<T (size_t i)> &f) {
    std::vector<T> elems(vector_size);
    for (size_t i = 0; i < vector_size; i++) elems[i] = f(i);
    return CreateVector(elems);
  }
  #endif
  // clang-format on

  /// @brief Serialize values returned by a function into a FlatBuffer `vector`.
  /// This is a convenience function that takes care of iteration for you.
  /// @tparam T The data type of the `std::vector` elements.
  /// @param f A function that takes the current iteration 0..vector_size-1,
  /// and the state parameter returning any type that you can construct a
  /// FlatBuffers vector out of.
  /// @param state State passed to f.
  /// @return Returns a typed `Offset` into the serialized data indicating
  /// where the vector is stored.
  template<typename T, typename F, typename S>
  Offset<Vector<T>> CreateVector(size_t vector_size, F f, S *state) {
    std::vector<T> elems(vector_size);
    for (size_t i = 0; i < vector_size; i++) elems[i] = f(i, state);
    return CreateVector(elems);
  }

  /// @brief Serialize a `std::vector<std::string>` into a FlatBuffer `vector`.
  /// This is a convenience function for a common case.
  /// @param v A const reference to the `std::vector` to serialize into the
  /// buffer as a `vector`.
  /// @return Returns a typed `Offset` into the serialized data indicating
  /// where the vector is stored.
  Offset<Vector<Offset<String>>> CreateVectorOfStrings(
      const std::vector<std::string> &v) {
    std::vector<Offset<String>> offsets(v.size());
    for (size_t i = 0; i < v.size(); i++) offsets[i] = CreateString(v[i]);
    return CreateVector(offsets);
  }

  /// @brief Serialize an array of structs into a FlatBuffer `vector`.
  /// @tparam T The data type of the struct array elements.
  /// @param[in] v A pointer to the array of type `T` to serialize into the
  /// buffer as a `vector`.
  /// @param[in] len The number of elements to serialize.
  /// @return Returns a typed `Offset` into the serialized data indicating
  /// where the vector is stored.
  template<typename T>
  Offset<Vector<const T *>> CreateVectorOfStructs(const T *v, size_t len) {
    StartVector(len * sizeof(T) / AlignOf<T>(), AlignOf<T>());
    PushBytes(reinterpret_cast<const uint8_t *>(v), sizeof(T) * len);
    return Offset<Vector<const T *>>(EndVector(len));
  }

  /// @brief Serialize an array of native structs into a FlatBuffer `vector`.
  /// @tparam T The data type of the struct array elements.
  /// @tparam S The data type of the native struct array elements.
  /// @param[in] v A pointer to the array of type `S` to serialize into the
  /// buffer as a `vector`.
  /// @param[in] len The number of elements to serialize.
  /// @return Returns a typed `Offset` into the serialized data indicating
  /// where the vector is stored.
  template<typename T, typename S>
  Offset<Vector<const T *>> CreateVectorOfNativeStructs(const S *v,
                                                        size_t len) {
    extern T Pack(const S &);
    std::vector<T> vv(len);
    std::transform(v, v + len, vv.begin(), Pack);
    return CreateVectorOfStructs<T>(data(vv), vv.size());
  }

  // clang-format off
  #ifndef FLATBUFFERS_CPP98_STL
  /// @brief Serialize an array of structs into a FlatBuffer `vector`.
  /// @tparam T The data type of the struct array elements.
  /// @param[in] filler A function that takes the current iteration 0..vector_size-1
  /// and a pointer to the struct that must be filled.
  /// @return Returns a typed `Offset` into the serialized data indicating
  /// where the vector is stored.
  /// This is mostly useful when flatbuffers are generated with mutation
  /// accessors.
  template<typename T> Offset<Vector<const T *>> CreateVectorOfStructs(
      size_t vector_size, const std::function<void(size_t i, T *)> &filler) {
    T* structs = StartVectorOfStructs<T>(vector_size);
    for (size_t i = 0; i < vector_size; i++) {
      filler(i, structs);
      structs++;
    }
    return EndVectorOfStructs<T>(vector_size);
  }
  #endif
  // clang-format on

  /// @brief Serialize an array of structs into a FlatBuffer `vector`.
  /// @tparam T The data type of the struct array elements.
  /// @param[in] f A function that takes the current iteration 0..vector_size-1,
  /// a pointer to the struct that must be filled and the state argument.
  /// @param[in] state Arbitrary state to pass to f.
  /// @return Returns a typed `Offset` into the serialized data indicating
  /// where the vector is stored.
  /// This is mostly useful when flatbuffers are generated with mutation
  /// accessors.
  template<typename T, typename F, typename S>
  Offset<Vector<const T *>> CreateVectorOfStructs(size_t vector_size, F f,
                                                  S *state) {
    T *structs = StartVectorOfStructs<T>(vector_size);
    for (size_t i = 0; i < vector_size; i++) {
      f(i, structs, state);
      structs++;
    }
    return EndVectorOfStructs<T>(vector_size);
  }

  /// @brief Serialize a `std::vector` of structs into a FlatBuffer `vector`.
  /// @tparam T The data type of the `std::vector` struct elements.
  /// @param[in] v A const reference to the `std::vector` of structs to
  /// serialize into the buffer as a `vector`.
  /// @return Returns a typed `Offset` into the serialized data indicating
  /// where the vector is stored.
  template<typename T, typename Alloc>
  Offset<Vector<const T *>> CreateVectorOfStructs(
      const std::vector<T, Alloc> &v) {
    return CreateVectorOfStructs(data(v), v.size());
  }

  /// @brief Serialize a `std::vector` of native structs into a FlatBuffer
  /// `vector`.
  /// @tparam T The data type of the `std::vector` struct elements.
  /// @tparam S The data type of the `std::vector` native struct elements.
  /// @param[in]] v A const reference to the `std::vector` of structs to
  /// serialize into the buffer as a `vector`.
  /// @return Returns a typed `Offset` into the serialized data indicating
  /// where the vector is stored.
  template<typename T, typename S>
  Offset<Vector<const T *>> CreateVectorOfNativeStructs(
      const std::vector<S> &v) {
    return CreateVectorOfNativeStructs<T, S>(data(v), v.size());
  }

  /// @cond FLATBUFFERS_INTERNAL
  template<typename T> struct StructKeyComparator {
    bool operator()(const T &a, const T &b) const {
      return a.KeyCompareLessThan(&b);
    }

   private:
    StructKeyComparator &operator=(const StructKeyComparator &);
  };
  /// @endcond

  /// @brief Serialize a `std::vector` of structs into a FlatBuffer `vector`
  /// in sorted order.
  /// @tparam T The data type of the `std::vector` struct elements.
  /// @param[in] v A const reference to the `std::vector` of structs to
  /// serialize into the buffer as a `vector`.
  /// @return Returns a typed `Offset` into the serialized data indicating
  /// where the vector is stored.
  template<typename T>
  Offset<Vector<const T *>> CreateVectorOfSortedStructs(std::vector<T> *v) {
    return CreateVectorOfSortedStructs(data(*v), v->size());
  }

  /// @brief Serialize a `std::vector` of native structs into a FlatBuffer
  /// `vector` in sorted order.
  /// @tparam T The data type of the `std::vector` struct elements.
  /// @tparam S The data type of the `std::vector` native struct elements.
  /// @param[in] v A const reference to the `std::vector` of structs to
  /// serialize into the buffer as a `vector`.
  /// @return Returns a typed `Offset` into the serialized data indicating
  /// where the vector is stored.
  template<typename T, typename S>
  Offset<Vector<const T *>> CreateVectorOfSortedNativeStructs(
      std::vector<S> *v) {
    return CreateVectorOfSortedNativeStructs<T, S>(data(*v), v->size());
  }

  /// @brief Serialize an array of structs into a FlatBuffer `vector` in sorted
  /// order.
  /// @tparam T The data type of the struct array elements.
  /// @param[in] v A pointer to the array of type `T` to serialize into the
  /// buffer as a `vector`.
  /// @param[in] len The number of elements to serialize.
  /// @return Returns a typed `Offset` into the serialized data indicating
  /// where the vector is stored.
  template<typename T>
  Offset<Vector<const T *>> CreateVectorOfSortedStructs(T *v, size_t len) {
    std::sort(v, v + len, StructKeyComparator<T>());
    return CreateVectorOfStructs(v, len);
  }

  /// @brief Serialize an array of native structs into a FlatBuffer `vector` in
  /// sorted order.
  /// @tparam T The data type of the struct array elements.
  /// @tparam S The data type of the native struct array elements.
  /// @param[in] v A pointer to the array of type `S` to serialize into the
  /// buffer as a `vector`.
  /// @param[in] len The number of elements to serialize.
  /// @return Returns a typed `Offset` into the serialized data indicating
  /// where the vector is stored.
  template<typename T, typename S>
  Offset<Vector<const T *>> CreateVectorOfSortedNativeStructs(S *v,
                                                              size_t len) {
    extern T Pack(const S &);
    typedef T (*Pack_t)(const S &);
    std::vector<T> vv(len);
    std::transform(v, v + len, vv.begin(), static_cast<Pack_t&>(Pack));
    return CreateVectorOfSortedStructs<T>(vv, len);
  }

  /// @cond FLATBUFFERS_INTERNAL
  template<typename T> struct TableKeyComparator {
    TableKeyComparator(vector_downward &buf) : buf_(buf) {}
    bool operator()(const Offset<T> &a, const Offset<T> &b) const {
      auto table_a = reinterpret_cast<T *>(buf_.data_at(a.o));
      auto table_b = reinterpret_cast<T *>(buf_.data_at(b.o));
      return table_a->KeyCompareLessThan(table_b);
    }
    vector_downward &buf_;

   private:
    TableKeyComparator &operator=(const TableKeyComparator &);
  };
  /// @endcond

  /// @brief Serialize an array of `table` offsets as a `vector` in the buffer
  /// in sorted order.
  /// @tparam T The data type that the offset refers to.
  /// @param[in] v An array of type `Offset<T>` that contains the `table`
  /// offsets to store in the buffer in sorted order.
  /// @param[in] len The number of elements to store in the `vector`.
  /// @return Returns a typed `Offset` into the serialized data indicating
  /// where the vector is stored.
  template<typename T>
  Offset<Vector<Offset<T>>> CreateVectorOfSortedTables(Offset<T> *v,
                                                       size_t len) {
    std::sort(v, v + len, TableKeyComparator<T>(buf_));
    return CreateVector(v, len);
  }

  /// @brief Serialize an array of `table` offsets as a `vector` in the buffer
  /// in sorted order.
  /// @tparam T The data type that the offset refers to.
  /// @param[in] v An array of type `Offset<T>` that contains the `table`
  /// offsets to store in the buffer in sorted order.
  /// @return Returns a typed `Offset` into the serialized data indicating
  /// where the vector is stored.
  template<typename T>
  Offset<Vector<Offset<T>>> CreateVectorOfSortedTables(
      std::vector<Offset<T>> *v) {
    return CreateVectorOfSortedTables(data(*v), v->size());
  }

  /// @brief Specialized version of `CreateVector` for non-copying use cases.
  /// Write the data any time later to the returned buffer pointer `buf`.
  /// @param[in] len The number of elements to store in the `vector`.
  /// @param[in] elemsize The size of each element in the `vector`.
  /// @param[out] buf A pointer to a `uint8_t` pointer that can be
  /// written to at a later time to serialize the data into a `vector`
  /// in the buffer.
  uoffset_t CreateUninitializedVector(size_t len, size_t elemsize,
                                      uint8_t **buf) {
    NotNested();
    StartVector(len, elemsize);
    buf_.make_space(len * elemsize);
    auto vec_start = GetSize();
    auto vec_end = EndVector(len);
    *buf = buf_.data_at(vec_start);
    return vec_end;
  }

  /// @brief Specialized version of `CreateVector` for non-copying use cases.
  /// Write the data any time later to the returned buffer pointer `buf`.
  /// @tparam T The data type of the data that will be stored in the buffer
  /// as a `vector`.
  /// @param[in] len The number of elements to store in the `vector`.
  /// @param[out] buf A pointer to a pointer of type `T` that can be
  /// written to at a later time to serialize the data into a `vector`
  /// in the buffer.
  template<typename T>
  Offset<Vector<T>> CreateUninitializedVector(size_t len, T **buf) {
    AssertScalarT<T>();
    return CreateUninitializedVector(len, sizeof(T),
                                     reinterpret_cast<uint8_t **>(buf));
  }

  template<typename T>
  Offset<Vector<const T*>> CreateUninitializedVectorOfStructs(size_t len, T **buf) {
    return CreateUninitializedVector(len, sizeof(T),
                                     reinterpret_cast<uint8_t **>(buf));
  }


  // @brief Create a vector of scalar type T given as input a vector of scalar
  // type U, useful with e.g. pre "enum class" enums, or any existing scalar
  // data of the wrong type.
  template<typename T, typename U>
  Offset<Vector<T>> CreateVectorScalarCast(const U *v, size_t len) {
    AssertScalarT<T>();
    AssertScalarT<U>();
    StartVector(len, sizeof(T));
    for (auto i = len; i > 0;) { PushElement(static_cast<T>(v[--i])); }
    return Offset<Vector<T>>(EndVector(len));
  }

  /// @brief Write a struct by itself, typically to be part of a union.
  template<typename T> Offset<const T *> CreateStruct(const T &structobj) {
    NotNested();
    Align(AlignOf<T>());
    buf_.push_small(structobj);
    return Offset<const T *>(GetSize());
  }

  /// @brief The length of a FlatBuffer file header.
  static const size_t kFileIdentifierLength = 4;

  /// @brief Finish serializing a buffer by writing the root offset.
  /// @param[in] file_identifier If a `file_identifier` is given, the buffer
  /// will be prefixed with a standard FlatBuffers file header.
  template<typename T>
  void Finish(Offset<T> root, const char *file_identifier = nullptr) {
    Finish(root.o, file_identifier, false);
  }

  /// @brief Finish a buffer with a 32 bit size field pre-fixed (size of the
  /// buffer following the size field). These buffers are NOT compatible
  /// with standard buffers created by Finish, i.e. you can't call GetRoot
  /// on them, you have to use GetSizePrefixedRoot instead.
  /// All >32 bit quantities in this buffer will be aligned when the whole
  /// size pre-fixed buffer is aligned.
  /// These kinds of buffers are useful for creating a stream of FlatBuffers.
  template<typename T>
  void FinishSizePrefixed(Offset<T> root,
                          const char *file_identifier = nullptr) {
    Finish(root.o, file_identifier, true);
  }

  void SwapBufAllocator(FlatBufferBuilder &other) {
    buf_.swap_allocator(other.buf_);
  }

protected:

  // You shouldn't really be copying instances of this class.
  FlatBufferBuilder(const FlatBufferBuilder &);
  FlatBufferBuilder &operator=(const FlatBufferBuilder &);

  void Finish(uoffset_t root, const char *file_identifier, bool size_prefix) {
    NotNested();
    buf_.clear_scratch();
    // This will cause the whole buffer to be aligned.
    PreAlign((size_prefix ? sizeof(uoffset_t) : 0) + sizeof(uoffset_t) +
                 (file_identifier ? kFileIdentifierLength : 0),
             minalign_);
    if (file_identifier) {
      FLATBUFFERS_ASSERT(strlen(file_identifier) == kFileIdentifierLength);
      PushBytes(reinterpret_cast<const uint8_t *>(file_identifier),
                kFileIdentifierLength);
    }
    PushElement(ReferTo(root));  // Location of root.
    if (size_prefix) { PushElement(GetSize()); }
    finished = true;
  }

  struct FieldLoc {
    uoffset_t off;
    voffset_t id;
  };

  vector_downward buf_;

  // Accumulating offsets of table members while it is being built.
  // We store these in the scratch pad of buf_, after the vtable offsets.
  uoffset_t num_field_loc;
  // Track how much of the vtable is in use, so we can output the most compact
  // possible vtable.
  voffset_t max_voffset_;

  // Ensure objects are not nested.
  bool nested;

  // Ensure the buffer is finished before it is being accessed.
  bool finished;

  size_t minalign_;

  bool force_defaults_;  // Serialize values equal to their defaults anyway.

  bool dedup_vtables_;

  struct StringOffsetCompare {
    StringOffsetCompare(const vector_downward &buf) : buf_(&buf) {}
    bool operator()(const Offset<String> &a, const Offset<String> &b) const {
      auto stra = reinterpret_cast<const String *>(buf_->data_at(a.o));
      auto strb = reinterpret_cast<const String *>(buf_->data_at(b.o));
      return StringLessThan(stra->data(), stra->size(),
                            strb->data(), strb->size());
    }
    const vector_downward *buf_;
  };

  // For use with CreateSharedString. Instantiated on first use only.
  typedef std::set<Offset<String>, StringOffsetCompare> StringOffsetMap;
  StringOffsetMap *string_pool;

 private:
  // Allocates space for a vector of structures.
  // Must be completed with EndVectorOfStructs().
  template<typename T> T *StartVectorOfStructs(size_t vector_size) {
    StartVector(vector_size * sizeof(T) / AlignOf<T>(), AlignOf<T>());
    return reinterpret_cast<T *>(buf_.make_space(vector_size * sizeof(T)));
  }

  // End the vector of structues in the flatbuffers.
  // Vector should have previously be started with StartVectorOfStructs().
  template<typename T>
  Offset<Vector<const T *>> EndVectorOfStructs(size_t vector_size) {
    return Offset<Vector<const T *>>(EndVector(vector_size));
  }
};
/// @}

/// @cond FLATBUFFERS_INTERNAL
// Helpers to get a typed pointer to the root object contained in the buffer.
template<typename T> T *GetMutableRoot(void *buf) {
  EndianCheck();
  return reinterpret_cast<T *>(
      reinterpret_cast<uint8_t *>(buf) +
      EndianScalar(*reinterpret_cast<uoffset_t *>(buf)));
}

template<typename T> const T *GetRoot(const void *buf) {
  return GetMutableRoot<T>(const_cast<void *>(buf));
}

template<typename T> const T *GetSizePrefixedRoot(const void *buf) {
  return GetRoot<T>(reinterpret_cast<const uint8_t *>(buf) + sizeof(uoffset_t));
}

/// Helpers to get a typed pointer to objects that are currently being built.
/// @warning Creating new objects will lead to reallocations and invalidates
/// the pointer!
template<typename T>
T *GetMutableTemporaryPointer(FlatBufferBuilder &fbb, Offset<T> offset) {
  return reinterpret_cast<T *>(fbb.GetCurrentBufferPointer() + fbb.GetSize() -
                               offset.o);
}

template<typename T>
const T *GetTemporaryPointer(FlatBufferBuilder &fbb, Offset<T> offset) {
  return GetMutableTemporaryPointer<T>(fbb, offset);
}

/// @brief Get a pointer to the the file_identifier section of the buffer.
/// @return Returns a const char pointer to the start of the file_identifier
/// characters in the buffer.  The returned char * has length
/// 'flatbuffers::FlatBufferBuilder::kFileIdentifierLength'.
/// This function is UNDEFINED for FlatBuffers whose schema does not include
/// a file_identifier (likely points at padding or the start of a the root
/// vtable).
inline const char *GetBufferIdentifier(const void *buf, bool size_prefixed = false) {
  return reinterpret_cast<const char *>(buf) +
         ((size_prefixed) ? 2 * sizeof(uoffset_t) : sizeof(uoffset_t));
}

// Helper to see if the identifier in a buffer has the expected value.
inline bool BufferHasIdentifier(const void *buf, const char *identifier, bool size_prefixed = false) {
  return strncmp(GetBufferIdentifier(buf, size_prefixed), identifier,
                 FlatBufferBuilder::kFileIdentifierLength) == 0;
}

// Helper class to verify the integrity of a FlatBuffer
class Verifier FLATBUFFERS_FINAL_CLASS {
 public:
  Verifier(const uint8_t *buf, size_t buf_len, uoffset_t _max_depth = 64,
           uoffset_t _max_tables = 1000000, bool _check_alignment = true)
      : buf_(buf),
        size_(buf_len),
        depth_(0),
        max_depth_(_max_depth),
        num_tables_(0),
        max_tables_(_max_tables),
        upper_bound_(0),
        check_alignment_(_check_alignment)
  {
    FLATBUFFERS_ASSERT(size_ < FLATBUFFERS_MAX_BUFFER_SIZE);
  }

  // Central location where any verification failures register.
  bool Check(bool ok) const {
    // clang-format off
    #ifdef FLATBUFFERS_DEBUG_VERIFICATION_FAILURE
      FLATBUFFERS_ASSERT(ok);
    #endif
    #ifdef FLATBUFFERS_TRACK_VERIFIER_BUFFER_SIZE
      if (!ok)
        upper_bound_ = 0;
    #endif
    // clang-format on
    return ok;
  }

  // Verify any range within the buffer.
  bool Verify(size_t elem, size_t elem_len) const {
    // clang-format off
    #ifdef FLATBUFFERS_TRACK_VERIFIER_BUFFER_SIZE
      auto upper_bound = elem + elem_len;
      if (upper_bound_ < upper_bound)
        upper_bound_ =  upper_bound;
    #endif
    // clang-format on
    return Check(elem_len < size_ && elem <= size_ - elem_len);
  }

  template<typename T> bool VerifyAlignment(size_t elem) const {
    return (elem & (sizeof(T) - 1)) == 0 || !check_alignment_;
  }

  // Verify a range indicated by sizeof(T).
  template<typename T> bool Verify(size_t elem) const {
    return VerifyAlignment<T>(elem) && Verify(elem, sizeof(T));
  }

  // Verify relative to a known-good base pointer.
  bool Verify(const uint8_t *base, voffset_t elem_off, size_t elem_len) const {
    return Verify(static_cast<size_t>(base - buf_) + elem_off, elem_len);
  }

  template<typename T> bool Verify(const uint8_t *base, voffset_t elem_off)
      const {
    return Verify(static_cast<size_t>(base - buf_) + elem_off, sizeof(T));
  }

  // Verify a pointer (may be NULL) of a table type.
  template<typename T> bool VerifyTable(const T *table) {
    return !table || table->Verify(*this);
  }

  // Verify a pointer (may be NULL) of any vector type.
  template<typename T> bool VerifyVector(const Vector<T> *vec) const {
    return !vec || VerifyVectorOrString(reinterpret_cast<const uint8_t *>(vec),
                                        sizeof(T));
  }

  // Verify a pointer (may be NULL) of a vector to struct.
  template<typename T> bool VerifyVector(const Vector<const T *> *vec) const {
    return VerifyVector(reinterpret_cast<const Vector<T> *>(vec));
  }

  // Verify a pointer (may be NULL) to string.
  bool VerifyString(const String *str) const {
    size_t end;
    return !str ||
           (VerifyVectorOrString(reinterpret_cast<const uint8_t *>(str),
                                 1, &end) &&
            Verify(end, 1) &&      // Must have terminator
            Check(buf_[end] == '\0'));  // Terminating byte must be 0.
  }

  // Common code between vectors and strings.
  bool VerifyVectorOrString(const uint8_t *vec, size_t elem_size,
                    size_t *end = nullptr) const {
    auto veco = static_cast<size_t>(vec - buf_);
    // Check we can read the size field.
    if (!Verify<uoffset_t>(veco)) return false;
    // Check the whole array. If this is a string, the byte past the array
    // must be 0.
    auto size = ReadScalar<uoffset_t>(vec);
    auto max_elems = FLATBUFFERS_MAX_BUFFER_SIZE / elem_size;
    if (!Check(size < max_elems))
      return false;  // Protect against byte_size overflowing.
    auto byte_size = sizeof(size) + elem_size * size;
    if (end) *end = veco + byte_size;
    return Verify(veco, byte_size);
  }

  // Special case for string contents, after the above has been called.
  bool VerifyVectorOfStrings(const Vector<Offset<String>> *vec) const {
    if (vec) {
      for (uoffset_t i = 0; i < vec->size(); i++) {
        if (!VerifyString(vec->Get(i))) return false;
      }
    }
    return true;
  }

  // Special case for table contents, after the above has been called.
  template<typename T> bool VerifyVectorOfTables(const Vector<Offset<T>> *vec) {
    if (vec) {
      for (uoffset_t i = 0; i < vec->size(); i++) {
        if (!vec->Get(i)->Verify(*this)) return false;
      }
    }
    return true;
  }

  bool VerifyTableStart(const uint8_t *table) {
    // Check the vtable offset.
    auto tableo = static_cast<size_t>(table - buf_);
    if (!Verify<soffset_t>(tableo)) return false;
    // This offset may be signed, but doing the substraction unsigned always
    // gives the result we want.
    auto vtableo = tableo - static_cast<size_t>(ReadScalar<soffset_t>(table));
    // Check the vtable size field, then check vtable fits in its entirety.
    return VerifyComplexity() && Verify<voffset_t>(vtableo) &&
           VerifyAlignment<voffset_t>(ReadScalar<voffset_t>(buf_ + vtableo)) &&
           Verify(vtableo, ReadScalar<voffset_t>(buf_ + vtableo));
  }

  template<typename T>
  bool VerifyBufferFromStart(const char *identifier, size_t start) {
    if (identifier &&
        (size_ < 2 * sizeof(flatbuffers::uoffset_t) ||
         !BufferHasIdentifier(buf_ + start, identifier))) {
      return false;
    }

    // Call T::Verify, which must be in the generated code for this type.
    auto o = VerifyOffset(start);
    return o && reinterpret_cast<const T *>(buf_ + start + o)->Verify(*this)
    // clang-format off
    #ifdef FLATBUFFERS_TRACK_VERIFIER_BUFFER_SIZE
           && GetComputedSize()
    #endif
        ;
    // clang-format on
  }

  // Verify this whole buffer, starting with root type T.
  template<typename T> bool VerifyBuffer() { return VerifyBuffer<T>(nullptr); }

  template<typename T> bool VerifyBuffer(const char *identifier) {
    return VerifyBufferFromStart<T>(identifier, 0);
  }

  template<typename T> bool VerifySizePrefixedBuffer(const char *identifier) {
    return Verify<uoffset_t>(0U) &&
           ReadScalar<uoffset_t>(buf_) == size_ - sizeof(uoffset_t) &&
           VerifyBufferFromStart<T>(identifier, sizeof(uoffset_t));
  }

  uoffset_t VerifyOffset(size_t start) const {
    if (!Verify<uoffset_t>(start)) return 0;
    auto o = ReadScalar<uoffset_t>(buf_ + start);
    // May not point to itself.
    if (!Check(o != 0)) return 0;
    // Can't wrap around / buffers are max 2GB.
    if (!Check(static_cast<soffset_t>(o) >= 0)) return 0;
    // Must be inside the buffer to create a pointer from it (pointer outside
    // buffer is UB).
    if (!Verify(start + o, 1)) return 0;
    return o;
  }

  uoffset_t VerifyOffset(const uint8_t *base, voffset_t start) const {
    return VerifyOffset(static_cast<size_t>(base - buf_) + start);
  }

  // Called at the start of a table to increase counters measuring data
  // structure depth and amount, and possibly bails out with false if
  // limits set by the constructor have been hit. Needs to be balanced
  // with EndTable().
  bool VerifyComplexity() {
    depth_++;
    num_tables_++;
    return Check(depth_ <= max_depth_ && num_tables_ <= max_tables_);
  }

  // Called at the end of a table to pop the depth count.
  bool EndTable() {
    depth_--;
    return true;
  }

  // Returns the message size in bytes
  size_t GetComputedSize() const {
    // clang-format off
    #ifdef FLATBUFFERS_TRACK_VERIFIER_BUFFER_SIZE
      uintptr_t size = upper_bound_;
      // Align the size to uoffset_t
      size = (size - 1 + sizeof(uoffset_t)) & ~(sizeof(uoffset_t) - 1);
      return (size > size_) ?  0 : size;
    #else
      // Must turn on FLATBUFFERS_TRACK_VERIFIER_BUFFER_SIZE for this to work.
      (void)upper_bound_;
      FLATBUFFERS_ASSERT(false);
      return 0;
    #endif
    // clang-format on
  }

 private:
  const uint8_t *buf_;
  size_t size_;
  uoffset_t depth_;
  uoffset_t max_depth_;
  uoffset_t num_tables_;
  uoffset_t max_tables_;
  mutable size_t upper_bound_;
  bool check_alignment_;
};

// Convenient way to bundle a buffer and its length, to pass it around
// typed by its root.
// A BufferRef does not own its buffer.
struct BufferRefBase {};  // for std::is_base_of
template<typename T> struct BufferRef : BufferRefBase {
  BufferRef() : buf(nullptr), len(0), must_free(false) {}
  BufferRef(uint8_t *_buf, uoffset_t _len)
      : buf(_buf), len(_len), must_free(false) {}

  ~BufferRef() {
    if (must_free) free(buf);
  }

  const T *GetRoot() const { return flatbuffers::GetRoot<T>(buf); }

  bool Verify() {
    Verifier verifier(buf, len);
    return verifier.VerifyBuffer<T>(nullptr);
  }

  uint8_t *buf;
  uoffset_t len;
  bool must_free;
};

// "structs" are flat structures that do not have an offset table, thus
// always have all members present and do not support forwards/backwards
// compatible extensions.

class Struct FLATBUFFERS_FINAL_CLASS {
 public:
  template<typename T> T GetField(uoffset_t o) const {
    return ReadScalar<T>(&data_[o]);
  }

  template<typename T> T GetStruct(uoffset_t o) const {
    return reinterpret_cast<T>(&data_[o]);
  }

  const uint8_t *GetAddressOf(uoffset_t o) const { return &data_[o]; }
  uint8_t *GetAddressOf(uoffset_t o) { return &data_[o]; }

 private:
  uint8_t data_[1];
};

// "tables" use an offset table (possibly shared) that allows fields to be
// omitted and added at will, but uses an extra indirection to read.
class Table {
 public:
  const uint8_t *GetVTable() const {
    return data_ - ReadScalar<soffset_t>(data_);
  }

  // This gets the field offset for any of the functions below it, or 0
  // if the field was not present.
  voffset_t GetOptionalFieldOffset(voffset_t field) const {
    // The vtable offset is always at the start.
    auto vtable = GetVTable();
    // The first element is the size of the vtable (fields + type id + itself).
    auto vtsize = ReadScalar<voffset_t>(vtable);
    // If the field we're accessing is outside the vtable, we're reading older
    // data, so it's the same as if the offset was 0 (not present).
    return field < vtsize ? ReadScalar<voffset_t>(vtable + field) : 0;
  }

  template<typename T> T GetField(voffset_t field, T defaultval) const {
    auto field_offset = GetOptionalFieldOffset(field);
    return field_offset ? ReadScalar<T>(data_ + field_offset) : defaultval;
  }

  template<typename P> P GetPointer(voffset_t field) {
    auto field_offset = GetOptionalFieldOffset(field);
    auto p = data_ + field_offset;
    return field_offset ? reinterpret_cast<P>(p + ReadScalar<uoffset_t>(p))
                        : nullptr;
  }
  template<typename P> P GetPointer(voffset_t field) const {
    return const_cast<Table *>(this)->GetPointer<P>(field);
  }

  template<typename P> P GetStruct(voffset_t field) const {
    auto field_offset = GetOptionalFieldOffset(field);
    auto p = const_cast<uint8_t *>(data_ + field_offset);
    return field_offset ? reinterpret_cast<P>(p) : nullptr;
  }

  template<typename T> bool SetField(voffset_t field, T val, T def) {
    auto field_offset = GetOptionalFieldOffset(field);
    if (!field_offset) return IsTheSameAs(val, def);
    WriteScalar(data_ + field_offset, val);
    return true;
  }

  bool SetPointer(voffset_t field, const uint8_t *val) {
    auto field_offset = GetOptionalFieldOffset(field);
    if (!field_offset) return false;
    WriteScalar(data_ + field_offset,
                static_cast<uoffset_t>(val - (data_ + field_offset)));
    return true;
  }

  uint8_t *GetAddressOf(voffset_t field) {
    auto field_offset = GetOptionalFieldOffset(field);
    return field_offset ? data_ + field_offset : nullptr;
  }
  const uint8_t *GetAddressOf(voffset_t field) const {
    return const_cast<Table *>(this)->GetAddressOf(field);
  }

  bool CheckField(voffset_t field) const {
    return GetOptionalFieldOffset(field) != 0;
  }

  // Verify the vtable of this table.
  // Call this once per table, followed by VerifyField once per field.
  bool VerifyTableStart(Verifier &verifier) const {
    return verifier.VerifyTableStart(data_);
  }

  // Verify a particular field.
  template<typename T>
  bool VerifyField(const Verifier &verifier, voffset_t field) const {
    // Calling GetOptionalFieldOffset should be safe now thanks to
    // VerifyTable().
    auto field_offset = GetOptionalFieldOffset(field);
    // Check the actual field.
    return !field_offset || verifier.Verify<T>(data_, field_offset);
  }

  // VerifyField for required fields.
  template<typename T>
  bool VerifyFieldRequired(const Verifier &verifier, voffset_t field) const {
    auto field_offset = GetOptionalFieldOffset(field);
    return verifier.Check(field_offset != 0) &&
           verifier.Verify<T>(data_, field_offset);
  }

  // Versions for offsets.
  bool VerifyOffset(const Verifier &verifier, voffset_t field) const {
    auto field_offset = GetOptionalFieldOffset(field);
    return !field_offset || verifier.VerifyOffset(data_, field_offset);
  }

  bool VerifyOffsetRequired(const Verifier &verifier, voffset_t field) const {
    auto field_offset = GetOptionalFieldOffset(field);
    return verifier.Check(field_offset != 0) &&
           verifier.VerifyOffset(data_, field_offset);
  }

 private:
  // private constructor & copy constructor: you obtain instances of this
  // class by pointing to existing data only
  Table();
  Table(const Table &other);

  uint8_t data_[1];
};

template<typename T> void FlatBufferBuilder::Required(Offset<T> table,
                                                      voffset_t field) {
  auto table_ptr = reinterpret_cast<const Table *>(buf_.data_at(table.o));
  bool ok = table_ptr->GetOptionalFieldOffset(field) != 0;
  // If this fails, the caller will show what field needs to be set.
  FLATBUFFERS_ASSERT(ok);
  (void)ok;
}

/// @brief This can compute the start of a FlatBuffer from a root pointer, i.e.
/// it is the opposite transformation of GetRoot().
/// This may be useful if you want to pass on a root and have the recipient
/// delete the buffer afterwards.
inline const uint8_t *GetBufferStartFromRootPointer(const void *root) {
  auto table = reinterpret_cast<const Table *>(root);
  auto vtable = table->GetVTable();
  // Either the vtable is before the root or after the root.
  auto start = (std::min)(vtable, reinterpret_cast<const uint8_t *>(root));
  // Align to at least sizeof(uoffset_t).
  start = reinterpret_cast<const uint8_t *>(reinterpret_cast<uintptr_t>(start) &
                                            ~(sizeof(uoffset_t) - 1));
  // Additionally, there may be a file_identifier in the buffer, and the root
  // offset. The buffer may have been aligned to any size between
  // sizeof(uoffset_t) and FLATBUFFERS_MAX_ALIGNMENT (see "force_align").
  // Sadly, the exact alignment is only known when constructing the buffer,
  // since it depends on the presence of values with said alignment properties.
  // So instead, we simply look at the next uoffset_t values (root,
  // file_identifier, and alignment padding) to see which points to the root.
  // None of the other values can "impersonate" the root since they will either
  // be 0 or four ASCII characters.
  static_assert(FlatBufferBuilder::kFileIdentifierLength == sizeof(uoffset_t),
                "file_identifier is assumed to be the same size as uoffset_t");
  for (auto possible_roots = FLATBUFFERS_MAX_ALIGNMENT / sizeof(uoffset_t) + 1;
       possible_roots; possible_roots--) {
    start -= sizeof(uoffset_t);
    if (ReadScalar<uoffset_t>(start) + start ==
        reinterpret_cast<const uint8_t *>(root))
      return start;
  }
  // We didn't find the root, either the "root" passed isn't really a root,
  // or the buffer is corrupt.
  // Assert, because calling this function with bad data may cause reads
  // outside of buffer boundaries.
  FLATBUFFERS_ASSERT(false);
  return nullptr;
}

/// @brief This return the prefixed size of a FlatBuffer.
inline uoffset_t GetPrefixedSize(const uint8_t* buf){ return ReadScalar<uoffset_t>(buf); }

// Base class for native objects (FlatBuffer data de-serialized into native
// C++ data structures).
// Contains no functionality, purely documentative.
struct NativeTable {};

/// @brief Function types to be used with resolving hashes into objects and
/// back again. The resolver gets a pointer to a field inside an object API
/// object that is of the type specified in the schema using the attribute
/// `cpp_type` (it is thus important whatever you write to this address
/// matches that type). The value of this field is initially null, so you
/// may choose to implement a delayed binding lookup using this function
/// if you wish. The resolver does the opposite lookup, for when the object
/// is being serialized again.
typedef uint64_t hash_value_t;
// clang-format off
#ifdef FLATBUFFERS_CPP98_STL
  typedef void (*resolver_function_t)(void **pointer_adr, hash_value_t hash);
  typedef hash_value_t (*rehasher_function_t)(void *pointer);
#else
  typedef std::function<void (void **pointer_adr, hash_value_t hash)>
          resolver_function_t;
  typedef std::function<hash_value_t (void *pointer)> rehasher_function_t;
#endif
// clang-format on

// Helper function to test if a field is present, using any of the field
// enums in the generated code.
// `table` must be a generated table type. Since this is a template parameter,
// this is not typechecked to be a subclass of Table, so beware!
// Note: this function will return false for fields equal to the default
// value, since they're not stored in the buffer (unless force_defaults was
// used).
template<typename T>
bool IsFieldPresent(const T *table, typename T::FlatBuffersVTableOffset field) {
  // Cast, since Table is a private baseclass of any table types.
  return reinterpret_cast<const Table *>(table)->CheckField(
      static_cast<voffset_t>(field));
}

// Utility function for reverse lookups on the EnumNames*() functions
// (in the generated C++ code)
// names must be NULL terminated.
inline int LookupEnum(const char **names, const char *name) {
  for (const char **p = names; *p; p++)
    if (!strcmp(*p, name)) return static_cast<int>(p - names);
  return -1;
}

// These macros allow us to layout a struct with a guarantee that they'll end
// up looking the same on different compilers and platforms.
// It does this by disallowing the compiler to do any padding, and then
// does padding itself by inserting extra padding fields that make every
// element aligned to its own size.
// Additionally, it manually sets the alignment of the struct as a whole,
// which is typically its largest element, or a custom size set in the schema
// by the force_align attribute.
// These are used in the generated code only.

// clang-format off
#if defined(_MSC_VER)
  #define FLATBUFFERS_MANUALLY_ALIGNED_STRUCT(alignment) \
    __pragma(pack(1)) \
    struct __declspec(align(alignment))
  #define FLATBUFFERS_STRUCT_END(name, size) \
    __pragma(pack()) \
    static_assert(sizeof(name) == size, "compiler breaks packing rules")
#elif defined(__GNUC__) || defined(__clang__) || defined(__ICCARM__)
  #define FLATBUFFERS_MANUALLY_ALIGNED_STRUCT(alignment) \
    _Pragma("pack(1)") \
    struct __attribute__((aligned(alignment)))
  #define FLATBUFFERS_STRUCT_END(name, size) \
    _Pragma("pack()") \
    static_assert(sizeof(name) == size, "compiler breaks packing rules")
#else
  #error Unknown compiler, please define structure alignment macros
#endif
// clang-format on

// Minimal reflection via code generation.
// Besides full-fat reflection (see reflection.h) and parsing/printing by
// loading schemas (see idl.h), we can also have code generation for mimimal
// reflection data which allows pretty-printing and other uses without needing
// a schema or a parser.
// Generate code with --reflect-types (types only) or --reflect-names (names
// also) to enable.
// See minireflect.h for utilities using this functionality.

// These types are organized slightly differently as the ones in idl.h.
enum SequenceType { ST_TABLE, ST_STRUCT, ST_UNION, ST_ENUM };

// Scalars have the same order as in idl.h
// clang-format off
#define FLATBUFFERS_GEN_ELEMENTARY_TYPES(ET) \
  ET(ET_UTYPE) \
  ET(ET_BOOL) \
  ET(ET_CHAR) \
  ET(ET_UCHAR) \
  ET(ET_SHORT) \
  ET(ET_USHORT) \
  ET(ET_INT) \
  ET(ET_UINT) \
  ET(ET_LONG) \
  ET(ET_ULONG) \
  ET(ET_FLOAT) \
  ET(ET_DOUBLE) \
  ET(ET_STRING) \
  ET(ET_SEQUENCE)  // See SequenceType.

enum ElementaryType {
  #define FLATBUFFERS_ET(E) E,
    FLATBUFFERS_GEN_ELEMENTARY_TYPES(FLATBUFFERS_ET)
  #undef FLATBUFFERS_ET
};

inline const char * const *ElementaryTypeNames() {
  static const char * const names[] = {
    #define FLATBUFFERS_ET(E) #E,
      FLATBUFFERS_GEN_ELEMENTARY_TYPES(FLATBUFFERS_ET)
    #undef FLATBUFFERS_ET
  };
  return names;
}
// clang-format on

// Basic type info cost just 16bits per field!
struct TypeCode {
  uint16_t base_type : 4;  // ElementaryType
  uint16_t is_vector : 1;
  int16_t sequence_ref : 11;  // Index into type_refs below, or -1 for none.
};

static_assert(sizeof(TypeCode) == 2, "TypeCode");

struct TypeTable;

// Signature of the static method present in each type.
typedef const TypeTable *(*TypeFunction)();

struct TypeTable {
  SequenceType st;
  size_t num_elems;  // of type_codes, values, names (but not type_refs).
  const TypeCode *type_codes;  // num_elems count
  const TypeFunction *type_refs;  // less than num_elems entries (see TypeCode).
  const int64_t *values;  // Only set for non-consecutive enum/union or structs.
  const char * const *names;     // Only set if compiled with --reflect-names.
};

// String which identifies the current version of FlatBuffers.
// flatbuffer_version_string is used by Google developers to identify which
// applications uploaded to Google Play are using this library.  This allows
// the development team at Google to determine the popularity of the library.
// How it works: Applications that are uploaded to the Google Play Store are
// scanned for this version string.  We track which applications are using it
// to measure popularity.  You are free to remove it (of course) but we would
// appreciate if you left it in.

// Weak linkage is culled by VS & doesn't work on cygwin.
// clang-format off
#if !defined(_WIN32) && !defined(__CYGWIN__)

extern volatile __attribute__((weak)) const char *flatbuffer_version_string;
volatile __attribute__((weak)) const char *flatbuffer_version_string =
  "FlatBuffers "
  FLATBUFFERS_STRING(FLATBUFFERS_VERSION_MAJOR) "."
  FLATBUFFERS_STRING(FLATBUFFERS_VERSION_MINOR) "."
  FLATBUFFERS_STRING(FLATBUFFERS_VERSION_REVISION);

#endif  // !defined(_WIN32) && !defined(__CYGWIN__)

#define FLATBUFFERS_DEFINE_BITMASK_OPERATORS(E, T)\
    inline E operator | (E lhs, E rhs){\
        return E(T(lhs) | T(rhs));\
    }\
    inline E operator & (E lhs, E rhs){\
        return E(T(lhs) & T(rhs));\
    }\
    inline E operator ^ (E lhs, E rhs){\
        return E(T(lhs) ^ T(rhs));\
    }\
    inline E operator ~ (E lhs){\
        return E(~T(lhs));\
    }\
    inline E operator |= (E &lhs, E rhs){\
        lhs = lhs | rhs;\
        return lhs;\
    }\
    inline E operator &= (E &lhs, E rhs){\
        lhs = lhs & rhs;\
        return lhs;\
    }\
    inline E operator ^= (E &lhs, E rhs){\
        lhs = lhs ^ rhs;\
        return lhs;\
    }\
    inline bool operator !(E rhs) \
    {\
        return !bool(T(rhs)); \
    }
/// @endcond
}  // namespace flatbuffers

// clang-format on

#endif  // FLATBUFFERS_H_
