// An InlinedVector<T,N,A> is like a std::vector<T,A>, except that storage
// for sequences of length <= N are provided inline without requiring
// any heap allocation.  Typically N is very small (e.g., 4) so that
// sequences that are expected to be short do not require allocations.
//
// Only some of the std::vector<> operations are currently implemented.
// Other operations may be added as needed to facilitate migrating
// code that uses std::vector<> to InlinedVector<>.
//
// NOTE: If you want an inlined version to replace use of a
// std::vector<bool>, consider using util::bitmap::InlinedBitVector<NBITS>
// in util/bitmap/inlined_bitvector.h
//
// TODO(billydonahue): change size_t to size_type where appropriate.

#ifndef TENSORFLOW_LIB_GTL_INLINED_VECTOR_H_
#define TENSORFLOW_LIB_GTL_INLINED_VECTOR_H_

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <algorithm>
#include <iterator>
#include <memory>
#include <type_traits>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/lib/gtl/manual_constructor.h"

#include <initializer_list>  // NOLINT(build/include_order)

namespace tensorflow {
namespace gtl {

template <typename T, int N, typename A = std::allocator<T> >
class InlinedVector {
 public:
  typedef A allocator_type;
  typedef typename allocator_type::value_type value_type;
  typedef typename allocator_type::pointer pointer;
  typedef typename allocator_type::const_pointer const_pointer;
  typedef typename allocator_type::reference reference;
  typedef typename allocator_type::const_reference const_reference;
  typedef typename allocator_type::size_type size_type;
  typedef typename allocator_type::difference_type difference_type;
  typedef pointer iterator;
  typedef const_pointer const_iterator;

  // Create an empty vector
  InlinedVector();
  explicit InlinedVector(const allocator_type& alloc);

  // Create a vector with n copies of value_type().
  explicit InlinedVector(size_t n);

  // Create a vector with n copies of elem
  InlinedVector(size_t n, const value_type& elem,
                const allocator_type& alloc = allocator_type());

  // Create and initialize with the elements [range_start .. range_end).
  // The unused enable_if argument restricts this constructor so that it is
  // elided when value_type is an integral type.  This prevents ambiguous
  // interpretation between a call to this constructor with two integral
  // arguments and a call to the preceding (n, elem) constructor.
  template <typename InputIterator>
  InlinedVector(
      InputIterator range_start, InputIterator range_end,
      const allocator_type& alloc = allocator_type(),
      typename std::enable_if<!std::is_integral<InputIterator>::value>::type* =
          NULL)
      : allocator_and_tag_(alloc) {
    AppendRange(range_start, range_end);
  }

  InlinedVector(std::initializer_list<value_type> init,
                const allocator_type& alloc = allocator_type())
      : allocator_and_tag_(alloc) {
    AppendRange(init.begin(), init.end());
  }

  InlinedVector(const InlinedVector& v);

  ~InlinedVector() { clear(); }

  InlinedVector& operator=(const InlinedVector& v) {
    // Optimized to avoid reallocation.
    // Prefer reassignment to copy construction for elements.
    if (size() < v.size()) {  // grow
      reserve(v.size());
      std::copy(v.begin(), v.begin() + size(), begin());
      std::copy(v.begin() + size(), v.end(), std::back_inserter(*this));
    } else {  // maybe shrink
      erase(begin() + v.size(), end());
      std::copy(v.begin(), v.end(), begin());
    }
    return *this;
  }

  size_t size() const {
    return allocated() ? allocation().size() : tag().size();
  }

  bool empty() const { return (size() == 0); }

  // Return number of elements that can be stored in vector
  // without requiring a reallocation of underlying memory
  size_t capacity() const { return allocated() ? allocation().capacity() : N; }

  // Return a pointer to the underlying array.
  // Only result[0,size()-1] are defined.
  const_pointer data() const {
    return allocated() ? allocated_space() : inlined_space();
  }
  pointer data() { return allocated() ? allocated_space() : inlined_space(); }

  // An older name for the more standard-friendly .data().
  const_pointer array() const { return data(); }
  pointer mutable_array() { return data(); }

  // Remove all elements
  void clear() {
    size_t s = size();
    if (allocated()) {
      DestroyAllocated(allocated_space(), allocated_space() + s);
      allocation().Dealloc(allocator());
    } else {
      DestroyInlined(inlined_space(), inlined_space() + s);
    }
    tag() = Tag();
  }

  // Return the ith element
  // REQUIRES: 0 <= i < size()
  const value_type& at(size_t i) const {
    DCHECK_LT(i, size());
    return array()[i];
  }
  const value_type& operator[](size_t i) const {
    DCHECK_LT(i, size());
    return array()[i];
  }

  // Return a non-const reference to the ith element
  // REQUIRES: 0 <= i < size()
  value_type& at(size_t i) {
    DCHECK_LT(i, size());
    return mutable_array()[i];
  }
  value_type& operator[](size_t i) {
    DCHECK_LT(i, size());
    return mutable_array()[i];
  }

  value_type& back() {
    DCHECK(!empty());
    return at(size() - 1);
  }

  const value_type& back() const {
    DCHECK(!empty());
    return at(size() - 1);
  }

  value_type& front() {
    DCHECK(!empty());
    return at(0);
  }

  const value_type& front() const {
    DCHECK(!empty());
    return at(0);
  }

  // Append t to the vector.
  // Increases size() by one.
  // Amortized complexity: O(1)
  // Worst-case complexity: O(size())
  void push_back(const value_type& t) {
    size_t s = size();
    DCHECK_LE(s, capacity());
    if (s == capacity()) {
      return GrowAndPushBack(t);
    }
    DCHECK_LT(s, capacity());

    if (allocated()) {
      ConstructAllocated(allocated_space() + s, t);
    } else {
      ConstructInlined(inlined_space() + s, t);
    }

    set_size_internal(s + 1);
  }

  void pop_back() {
    DCHECK(!empty());
    size_t s = size();
    if (allocated()) {
      DestroyAllocated(allocated_space() + s - 1, allocated_space() + s);
    } else {
      DestroyInlined(inlined_space() + s - 1, inlined_space() + s);
    }
    set_size_internal(s - 1);
  }

  // Resizes the vector to contain "n" elements.
  // If "n" is smaller than the initial size, extra elements are destroyed.
  // If "n" is larger than the initial size, enough copies of "elem"
  // are appended to increase the size to "n". If "elem" is omitted,
  // new elements are value-initialized.
  void resize(size_t n);
  void resize(size_t n, const value_type& elem);

  iterator begin() { return mutable_array(); }
  const_iterator begin() const { return array(); }

  iterator end() { return mutable_array() + size(); }
  const_iterator end() const { return array() + size(); }

  iterator insert(iterator pos, const value_type& v);

  iterator erase(iterator pos) {
    DCHECK_LT(pos, end());
    DCHECK_GE(pos, begin());
    std::copy(pos + 1, end(), pos);
    pop_back();
    return pos;
  }

  iterator erase(iterator first, iterator last);

  // Enlarges the underlying representation so it can hold at least
  // "n" elements without reallocation.
  // Does not change size() or the actual contents of the vector.
  void reserve(size_t n) {
    if (n > capacity()) {
      // Make room for new elements
      EnlargeBy(n - size());
    }
  }

  // Swap the contents of *this with other.
  // REQUIRES: value_type is swappable and copyable.
  void swap(InlinedVector& other);

  allocator_type get_allocator() const { return allocator(); }

 private:
  struct AllocatorTraits {
    typedef typename allocator_type::value_type value_type;
    typedef typename allocator_type::pointer pointer;
    typedef typename allocator_type::size_type size_type;

    static void construct(allocator_type& a,  // NOLINT(runtime/references)
                          pointer p) {
      // Tricky: do we support non-copyable types, or support allocators
      // that do special things with construct()? Non-copyable types are
      // needed today, so they are more important. When we sort out the
      // Android NDK C++11 problem, we will be able to use the proper
      // std::allocator_traits<A>::construct(p, ...).
      //
      // a.construct(p, value_type());
      new (p) value_type();
    }
    static void construct(allocator_type& a,  // NOLINT(runtime/references)
                          pointer p, const value_type& t) {
      a.construct(p, t);
    }
    static void destroy(allocator_type& a,  // NOLINT(runtime/references)
                        pointer p) {
      a.destroy(p);
    }
    static pointer allocate(allocator_type& a,  // NOLINT(runtime/references)
                            size_type n) {
      return a.allocate(n);
    }
    static void deallocate(allocator_type& a,  // NOLINT(runtime/references)
                           pointer p, size_type n) {
      a.deallocate(p, n);
    }
  };

  // If the vector is inlined, holds the size of the vector.
  // If the vector is allocated, holds the special value kAllocated,
  // and the size is stored in the vector's Allocation.
  class Tag {
   public:
    Tag() : size_(0) {}
    size_t size() const { return size_; }
    void set_size(size_t n) { size_ = n; }
    bool allocated() const { return size_ == kAllocated; }
    void set_allocated() { size_ = kAllocated; }

   private:
    static const size_t kAllocated = -1;
    size_t size_;
  };

  // Derives from allocator_type to use the empty base class optimization.
  // If the allocator_type is stateless, we can 'store'
  // our instance of it for free.
  class AllocatorAndTag : private allocator_type {
   public:
    explicit AllocatorAndTag(const allocator_type& a, Tag t = Tag())
        : allocator_type(a), tag_(t) {}
    Tag& tag() { return tag_; }
    const Tag& tag() const { return tag_; }
    allocator_type& allocator() { return *this; }
    const allocator_type& allocator() const { return *this; }

   private:
    Tag tag_;
  };

  class Allocation {
   public:
    Allocation(allocator_type& a,  // NOLINT(runtime/references)
               size_t capacity)
        : size_(0),
          capacity_(capacity),
          buffer_(AllocatorTraits::allocate(a, capacity_)) {}

    void Dealloc(allocator_type& a) {  // NOLINT(runtime/references)
      AllocatorTraits::deallocate(a, buffer(), capacity());
    }

    size_t size() const { return size_; }
    void set_size(size_t s) { size_ = s; }
    size_t capacity() const { return capacity_; }
    const value_type* buffer() const { return buffer_; }
    value_type* buffer() { return buffer_; }

   private:
    size_t size_;
    size_t capacity_;
    value_type* buffer_;
  };

  const Tag& tag() const { return allocator_and_tag_.tag(); }
  Tag& tag() { return allocator_and_tag_.tag(); }

  Allocation& allocation() { return *rep_.allocation_storage.allocation.get(); }
  const Allocation& allocation() const {
    return *rep_.allocation_storage.allocation.get();
  }
  void init_allocation(const Allocation& allocation) {
    rep_.allocation_storage.allocation.Init(allocation);
  }

  value_type* inlined_space() { return rep_.inlined_storage.inlined[0].get(); }
  const value_type* inlined_space() const {
    return rep_.inlined_storage.inlined[0].get();
  }

  value_type* allocated_space() { return allocation().buffer(); }
  const value_type* allocated_space() const { return allocation().buffer(); }

  const allocator_type& allocator() const {
    return allocator_and_tag_.allocator();
  }
  allocator_type& allocator() { return allocator_and_tag_.allocator(); }

  bool allocated() const { return tag().allocated(); }
  void set_allocated() { return tag().set_allocated(); }

  void set_size_internal(size_t n) {
    if (allocated()) {
      allocation().set_size(n);
    } else {
      tag().set_size(n);
    }
  }

  // Enlarge the underlying representation so we can store size_ + delta elems.
  // The size is not changed, and any newly added memory is not initialized.
  void EnlargeBy(size_t delta);

  void ResetAllocation(Allocation new_allocation) {
    if (allocated()) {
      DestroyAllocated(allocated_space(), allocated_space() + size());
      DCHECK_EQ(begin(), allocated_space());
      allocation().Dealloc(allocator());
      allocation() = new_allocation;
    } else {
      DestroyInlined(inlined_space(), inlined_space() + size());
      init_allocation(new_allocation);  // bug: only init once
      set_allocated();
    }
  }

  void GrowAndPushBack(const value_type& t) {
    DCHECK_EQ(size(), capacity());
    const size_t s = size();

    Allocation new_allocation(allocator(), 2 * capacity());
    new_allocation.set_size(s + 1);

    UninitializedCopyAllocated(array(), array() + s, new_allocation.buffer());
    ConstructAllocated(new_allocation.buffer() + s, t);

    ResetAllocation(new_allocation);
  }

  void InitAssign(size_t n);
  void InitAssign(size_t n, const value_type& t);

  void ConstructInlined(pointer p) { new (p) value_type(); }

  void ConstructInlined(pointer p, const value_type& t) {
    new (p) value_type(t);
  }

  void ConstructAllocated(pointer p) {
    AllocatorTraits::construct(allocator(), p);
  }
  void ConstructAllocated(pointer p, const value_type& t) {
    AllocatorTraits::construct(allocator(), p, t);
  }

  template <typename Iter>
  void UninitializedCopyInlined(Iter src, Iter src_last, value_type* dst) {
    std::uninitialized_copy(src, src_last, dst);
  }

  template <typename Iter>
  void UninitializedCopyAllocated(Iter src, Iter src_last, value_type* dst) {
    for (; src != src_last; ++dst, ++src) ConstructAllocated(dst, *src);
  }

  void UninitializedFillInlined(value_type* dst, value_type* dst_last) {
    for (; dst != dst_last; ++dst) ConstructInlined(dst);
  }
  void UninitializedFillInlined(value_type* dst, value_type* dst_last,
                                const value_type& t) {
    std::uninitialized_fill(dst, dst_last, t);
  }

  void UninitializedFillAllocated(value_type* dst, value_type* dst_last) {
    for (; dst != dst_last; ++dst) ConstructAllocated(dst);
  }
  void UninitializedFillAllocated(value_type* dst, value_type* dst_last,
                                  const value_type& t) {
    for (; dst != dst_last; ++dst) ConstructAllocated(dst, t);
  }

  // Destroy [ptr, ptr_last) in place.
  void DestroyInlined(value_type* ptr, value_type* ptr_last);
  void DestroyAllocated(value_type* ptr, value_type* ptr_last);

  template <typename Iter>
  void AppendRange(Iter first, Iter last, std::input_iterator_tag);

  // Faster path for forward iterators.
  template <typename Iter>
  void AppendRange(Iter first, Iter last, std::forward_iterator_tag);

  template <typename Iter>
  void AppendRange(Iter first, Iter last);

  AllocatorAndTag allocator_and_tag_;

  // Either the inlined or allocated representation
  union Rep {
    // Use struct to perform indirection that solves a bizarre compilation
    // error on Visual Studio (all known versions).
    struct {
      tensorflow::ManualConstructor<value_type> inlined[N];
    } inlined_storage;
    struct {
      tensorflow::ManualConstructor<Allocation> allocation;
    } allocation_storage;
  } rep_;
};

template <typename T, int N, typename A>
const size_t InlinedVector<T, N, A>::Tag::kAllocated;

template <typename T, int N, typename A>
inline void swap(InlinedVector<T, N, A>& a, InlinedVector<T, N, A>& b) {
  a.swap(b);
}

template <typename T, int N, typename A>
inline bool operator==(const InlinedVector<T, N, A>& a,
                       const InlinedVector<T, N, A>& b) {
  return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

template <typename T, int N, typename A>
inline bool operator!=(const InlinedVector<T, N, A>& a,
                       const InlinedVector<T, N, A>& b) {
  return !(a == b);
}

template <typename T, int N, typename A>
inline bool operator<(const InlinedVector<T, N, A>& a,
                      const InlinedVector<T, N, A>& b) {
  return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
}

template <typename T, int N, typename A>
inline bool operator>(const InlinedVector<T, N, A>& a,
                      const InlinedVector<T, N, A>& b) {
  return b < a;
}

template <typename T, int N, typename A>
inline bool operator<=(const InlinedVector<T, N, A>& a,
                       const InlinedVector<T, N, A>& b) {
  return !(b < a);
}

template <typename T, int N, typename A>
inline bool operator>=(const InlinedVector<T, N, A>& a,
                       const InlinedVector<T, N, A>& b) {
  return !(a < b);
}

// ========================================
// Implementation

template <typename T, int N, typename A>
inline InlinedVector<T, N, A>::InlinedVector()
    : allocator_and_tag_(allocator_type()) {}

template <typename T, int N, typename A>
inline InlinedVector<T, N, A>::InlinedVector(const allocator_type& alloc)
    : allocator_and_tag_(alloc) {}

template <typename T, int N, typename A>
inline InlinedVector<T, N, A>::InlinedVector(size_t n)
    : allocator_and_tag_(allocator_type()) {
  InitAssign(n);
}

template <typename T, int N, typename A>
inline InlinedVector<T, N, A>::InlinedVector(size_t n, const value_type& elem,
                                             const allocator_type& alloc)
    : allocator_and_tag_(alloc) {
  InitAssign(n, elem);
}

template <typename T, int N, typename A>
inline InlinedVector<T, N, A>::InlinedVector(const InlinedVector& v)
    : allocator_and_tag_(v.allocator()) {
  reserve(v.size());
  if (allocated()) {
    UninitializedCopyAllocated(v.begin(), v.end(), allocated_space());
  } else {
    UninitializedCopyInlined(v.begin(), v.end(), inlined_space());
  }
  set_size_internal(v.size());
}

template <typename T, int N, typename A>
inline void InlinedVector<T, N, A>::InitAssign(size_t n, const value_type& t) {
  if (n > static_cast<size_t>(N)) {
    Allocation new_allocation(allocator(), n);
    init_allocation(new_allocation);
    set_allocated();
    UninitializedFillAllocated(allocated_space(), allocated_space() + n, t);
  } else {
    UninitializedFillInlined(inlined_space(), inlined_space() + n, t);
  }
  set_size_internal(n);
}

template <typename T, int N, typename A>
inline void InlinedVector<T, N, A>::InitAssign(size_t n) {
  if (n > static_cast<size_t>(N)) {
    Allocation new_allocation(allocator(), n);
    init_allocation(new_allocation);
    set_allocated();
    UninitializedFillAllocated(allocated_space(), allocated_space() + n);
  } else {
    UninitializedFillInlined(inlined_space(), inlined_space() + n);
  }
  set_size_internal(n);
}

template <typename T, int N, typename A>
inline void InlinedVector<T, N, A>::resize(size_t n) {
  size_t s = size();
  if (n < s) {
    erase(begin() + n, end());
    return;
  }
  reserve(n);
  DCHECK_GE(capacity(), n);

  // Fill new space with elements constructed in-place.
  if (allocated()) {
    UninitializedFillAllocated(allocated_space() + s, allocated_space() + n);
  } else {
    UninitializedFillInlined(inlined_space() + s, inlined_space() + n);
  }
  set_size_internal(n);
}

template <typename T, int N, typename A>
inline void InlinedVector<T, N, A>::resize(size_t n, const value_type& elem) {
  size_t s = size();
  if (n < s) {
    erase(begin() + n, end());
    return;
  }
  reserve(n);
  DCHECK_GE(capacity(), n);

  // Fill new space with copies of 'elem'.
  if (allocated()) {
    UninitializedFillAllocated(allocated_space() + s, allocated_space() + n,
                               elem);
  } else {
    UninitializedFillInlined(inlined_space() + s, inlined_space() + n, elem);
  }
  set_size_internal(n);
}

template <typename T, int N, typename A>
typename InlinedVector<T, N, A>::iterator InlinedVector<T, N, A>::insert(
    iterator pos, const value_type& v) {
  DCHECK_GE(pos, begin());
  DCHECK_LE(pos, end());
  if (pos == end()) {
    push_back(v);
    return end() - 1;
  }
  size_t s = size();
  size_t idx = std::distance(begin(), pos);
  if (s == capacity()) {
    EnlargeBy(1);
  }
  CHECK_LT(s, capacity());
  pos = begin() + idx;  // Reset 'pos' into a post-enlarge iterator.

  if (allocated()) {
    ConstructAllocated(allocated_space() + s, *(allocated_space() + s - 1));
    std::copy_backward(pos, allocated_space() + s - 1, allocated_space() + s);
  } else {
    ConstructInlined(inlined_space() + s, *(inlined_space() + s - 1));
    std::copy_backward(pos, inlined_space() + s - 1, inlined_space() + s);
  }

  *pos = v;

  set_size_internal(s + 1);
  return pos;
}

template <typename T, int N, typename A>
typename InlinedVector<T, N, A>::iterator InlinedVector<T, N, A>::erase(
    iterator first, iterator last) {
  DCHECK_LE(begin(), first);
  DCHECK_LE(first, last);
  DCHECK_LE(last, end());

  size_t s = size();
  ptrdiff_t erase_gap = std::distance(first, last);

  if (allocated()) {
    std::copy(last, allocated_space() + s, first);
    DestroyAllocated(allocated_space() + s - erase_gap, allocated_space() + s);
  } else {
    std::copy(last, inlined_space() + s, first);
    DestroyInlined(inlined_space() + s - erase_gap, inlined_space() + s);
  }

  set_size_internal(size() - erase_gap);

  return first;
}

template <typename T, int N, typename A>
void InlinedVector<T, N, A>::swap(InlinedVector& other) {
  using std::swap;  // Augment ADL with std::swap.
  if (&other == this) {
    return;
  }
  if (allocated() && other.allocated()) {
    // Both out of line, so just swap the tag, allocation, and allocator.
    swap(tag(), other.tag());
    swap(allocation(), other.allocation());
    swap(allocator(), other.allocator());
    return;
  }
  if (!allocated() && !other.allocated()) {
    // Both inlined: swap up to smaller size, then move remaining elements.
    InlinedVector* a = this;
    InlinedVector* b = &other;
    if (size() < other.size()) {
      swap(a, b);
    }

    const size_t a_size = a->size();
    const size_t b_size = b->size();
    DCHECK_GE(a_size, b_size);
    // 'a' is larger. Swap the elements up to the smaller array size.
    std::swap_ranges(a->inlined_space(), a->inlined_space() + b_size,
                     b->inlined_space());

    // Move the remaining elements: A[b_size,a_size) -> B[b_size,a_size)
    b->UninitializedCopyInlined(a->inlined_space() + b_size,
                                a->inlined_space() + a_size,
                                b->inlined_space() + b_size);
    a->DestroyInlined(a->inlined_space() + b_size, a->inlined_space() + a_size);

    swap(a->tag(), b->tag());
    swap(a->allocator(), b->allocator());
    DCHECK_EQ(b->size(), a_size);
    DCHECK_EQ(a->size(), b_size);
    return;
  }
  // One is out of line, one is inline.
  // We first move the elements from the inlined vector into the
  // inlined space in the other vector.  We then put the other vector's
  // pointer/capacity into the originally inlined vector and swap
  // the tags.
  InlinedVector* a = this;
  InlinedVector* b = &other;
  if (a->allocated()) {
    swap(a, b);
  }
  DCHECK(!a->allocated());
  DCHECK(b->allocated());
  const size_t a_size = a->size();
  const size_t b_size = b->size();

  // Made Local copies of size(), don't need tag() accurate anymore
  swap(a->tag(), b->tag());

  // Copy b_allocation out before b's union gets clobbered by inline_space.
  Allocation b_allocation = b->allocation();

  b->UninitializedCopyInlined(a->inlined_space(), a->inlined_space() + a_size,
                              b->inlined_space());
  a->DestroyInlined(a->inlined_space(), a->inlined_space() + a_size);

  a->allocation() = b_allocation;

  if (a->allocator() != b->allocator()) {
    swap(a->allocator(), b->allocator());
  }

  DCHECK_EQ(b->size(), a_size);
  DCHECK_EQ(a->size(), b_size);
}

template <typename T, int N, typename A>
void InlinedVector<T, N, A>::EnlargeBy(size_t delta) {
  const size_t s = size();
  DCHECK_LE(s, capacity());

  size_t target = std::max(static_cast<size_t>(N), s + delta);

  // Compute new capacity by repeatedly doubling current capacity
  // TODO(psrc): Check and avoid overflow?
  size_t new_capacity = capacity();
  while (new_capacity < target) {
    new_capacity <<= 1;
  }

  Allocation new_allocation(allocator(), new_capacity);
  new_allocation.set_size(s);

  UninitializedCopyAllocated(array(), array() + s, new_allocation.buffer());

  ResetAllocation(new_allocation);
}

template <typename T, int N, typename A>
inline void InlinedVector<T, N, A>::DestroyInlined(value_type* ptr,
                                                   value_type* ptr_last) {
  for (value_type* p = ptr; p != ptr_last; ++p) {
    p->~value_type();
  }

// Overwrite unused memory with 0xab so we can catch uninitialized usage.
// Cast to void* to tell the compiler that we don't care that we might be
// scribbling on a vtable pointer.
#ifndef NDEBUG
  if (ptr != ptr_last) {
    memset(reinterpret_cast<void*>(ptr), 0xab, sizeof(*ptr) * (ptr_last - ptr));
  }
#endif
}

template <typename T, int N, typename A>
inline void InlinedVector<T, N, A>::DestroyAllocated(value_type* ptr,
                                                     value_type* ptr_last) {
  for (value_type* p = ptr; p != ptr_last; ++p) {
    AllocatorTraits::destroy(allocator(), p);
  }

// Overwrite unused memory with 0xab so we can catch uninitialized usage.
// Cast to void* to tell the compiler that we don't care that we might be
// scribbling on a vtable pointer.
#ifndef NDEBUG
  if (ptr != ptr_last) {
    memset(reinterpret_cast<void*>(ptr), 0xab, sizeof(*ptr) * (ptr_last - ptr));
  }
#endif
}

template <typename T, int N, typename A>
template <typename Iter>
inline void InlinedVector<T, N, A>::AppendRange(Iter first, Iter last,
                                                std::input_iterator_tag) {
  std::copy(first, last, std::back_inserter(*this));
}

template <typename T, int N, typename A>
template <typename Iter>
inline void InlinedVector<T, N, A>::AppendRange(Iter first, Iter last,
                                                std::forward_iterator_tag) {
  typedef typename std::iterator_traits<Iter>::difference_type Length;
  Length length = std::distance(first, last);
  reserve(size() + length);
  if (allocated()) {
    UninitializedCopyAllocated(first, last, allocated_space() + size());
  } else {
    UninitializedCopyInlined(first, last, inlined_space() + size());
  }
  set_size_internal(size() + length);
}

template <typename T, int N, typename A>
template <typename Iter>
inline void InlinedVector<T, N, A>::AppendRange(Iter first, Iter last) {
  typedef typename std::iterator_traits<Iter>::iterator_category IterTag;
  AppendRange(first, last, IterTag());
}

}  // namespace gtl
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_GTL_INLINED_VECTOR_H_
