//===- STLExtras.h - STL-like extensions that are used by MLIR --*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file contains stuff that should be arguably sunk down to the LLVM
// Support/STLExtras.h file over time.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_STLEXTRAS_H
#define MLIR_SUPPORT_STLEXTRAS_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {

namespace detail {
template <typename RangeT>
using ValueOfRange = typename std::remove_reference<decltype(
    *std::begin(std::declval<RangeT &>()))>::type;
} // end namespace detail

/// An STL-style algorithm similar to std::for_each that applies a second
/// functor between every pair of elements.
///
/// This provides the control flow logic to, for example, print a
/// comma-separated list:
/// \code
///   interleave(names.begin(), names.end(),
///              [&](StringRef name) { os << name; },
///              [&] { os << ", "; });
/// \endcode
template <typename ForwardIterator, typename UnaryFunctor,
          typename NullaryFunctor,
          typename = typename std::enable_if<
              !std::is_constructible<StringRef, UnaryFunctor>::value &&
              !std::is_constructible<StringRef, NullaryFunctor>::value>::type>
inline void interleave(ForwardIterator begin, ForwardIterator end,
                       UnaryFunctor each_fn, NullaryFunctor between_fn) {
  if (begin == end)
    return;
  each_fn(*begin);
  ++begin;
  for (; begin != end; ++begin) {
    between_fn();
    each_fn(*begin);
  }
}

template <typename Container, typename UnaryFunctor, typename NullaryFunctor,
          typename = typename std::enable_if<
              !std::is_constructible<StringRef, UnaryFunctor>::value &&
              !std::is_constructible<StringRef, NullaryFunctor>::value>::type>
inline void interleave(const Container &c, UnaryFunctor each_fn,
                       NullaryFunctor between_fn) {
  interleave(c.begin(), c.end(), each_fn, between_fn);
}

/// Overload of interleave for the common case of string separator.
template <typename Container, typename UnaryFunctor, typename raw_ostream,
          typename T = detail::ValueOfRange<Container>>
inline void interleave(const Container &c, raw_ostream &os,
                       UnaryFunctor each_fn, const StringRef &separator) {
  interleave(c.begin(), c.end(), each_fn, [&] { os << separator; });
}
template <typename Container, typename raw_ostream,
          typename T = detail::ValueOfRange<Container>>
inline void interleave(const Container &c, raw_ostream &os,
                       const StringRef &separator) {
  interleave(
      c, os, [&](const T &a) { os << a; }, separator);
}

template <typename Container, typename UnaryFunctor, typename raw_ostream,
          typename T = detail::ValueOfRange<Container>>
inline void interleaveComma(const Container &c, raw_ostream &os,
                            UnaryFunctor each_fn) {
  interleave(c, os, each_fn, ", ");
}
template <typename Container, typename raw_ostream,
          typename T = detail::ValueOfRange<Container>>
inline void interleaveComma(const Container &c, raw_ostream &os) {
  interleaveComma(c, os, [&](const T &a) { os << a; });
}

/// A special type used to provide an address for a given class that can act as
/// a unique identifier during pass registration.
/// Note: We specify an explicit alignment here to allow use with PointerIntPair
/// and other utilities/data structures that require a known pointer alignment.
struct alignas(8) ClassID {
  template <typename T> static ClassID *getID() {
    static ClassID id;
    return &id;
  }
  template <template <typename T> class Trait> static ClassID *getID() {
    static ClassID id;
    return &id;
  }
};

/// Utilities for detecting if a given trait holds for some set of arguments
/// 'Args'. For example, the given trait could be used to detect if a given type
/// has a copy assignment operator:
///   template<class T>
///   using has_copy_assign_t = decltype(std::declval<T&>()
///                                                 = std::declval<const T&>());
///   bool fooHasCopyAssign = is_detected<has_copy_assign_t, FooClass>::value;
namespace detail {
template <typename...> using void_t = void;
template <class, template <class...> class Op, class... Args> struct detector {
  using value_t = std::false_type;
};
template <template <class...> class Op, class... Args>
struct detector<void_t<Op<Args...>>, Op, Args...> {
  using value_t = std::true_type;
};
} // end namespace detail

template <template <class...> class Op, class... Args>
using is_detected = typename detail::detector<void, Op, Args...>::value_t;

/// Check if a Callable type can be invoked with the given set of arg types.
namespace detail {
template <typename Callable, typename... Args>
using is_invocable =
    decltype(std::declval<Callable &>()(std::declval<Args>()...));
} // namespace detail

template <typename Callable, typename... Args>
using is_invocable = is_detected<detail::is_invocable, Callable, Args...>;

//===----------------------------------------------------------------------===//
//     Extra additions to <iterator>
//===----------------------------------------------------------------------===//

/// A utility class used to implement an iterator that contains some base object
/// and an index. The iterator moves the index but keeps the base constant.
template <typename DerivedT, typename BaseT, typename T,
          typename PointerT = T *, typename ReferenceT = T &>
class indexed_accessor_iterator
    : public llvm::iterator_facade_base<DerivedT,
                                        std::random_access_iterator_tag, T,
                                        std::ptrdiff_t, PointerT, ReferenceT> {
public:
  ptrdiff_t operator-(const indexed_accessor_iterator &rhs) const {
    assert(base == rhs.base && "incompatible iterators");
    return index - rhs.index;
  }
  bool operator==(const indexed_accessor_iterator &rhs) const {
    return base == rhs.base && index == rhs.index;
  }
  bool operator<(const indexed_accessor_iterator &rhs) const {
    assert(base == rhs.base && "incompatible iterators");
    return index < rhs.index;
  }

  DerivedT &operator+=(ptrdiff_t offset) {
    this->index += offset;
    return static_cast<DerivedT &>(*this);
  }
  DerivedT &operator-=(ptrdiff_t offset) {
    this->index -= offset;
    return static_cast<DerivedT &>(*this);
  }

  /// Returns the current index of the iterator.
  ptrdiff_t getIndex() const { return index; }

  /// Returns the current base of the iterator.
  const BaseT &getBase() const { return base; }

protected:
  indexed_accessor_iterator(BaseT base, ptrdiff_t index)
      : base(base), index(index) {}
  BaseT base;
  ptrdiff_t index;
};

namespace detail {
/// The class represents the base of a range of indexed_accessor_iterators. It
/// provides support for many different range functionalities, e.g.
/// drop_front/slice/etc.. Derived range classes must implement the following
/// static methods:
///   * ReferenceT dereference_iterator(const BaseT &base, ptrdiff_t index)
///     - Derefence an iterator pointing to the base object at the given index.
///   * BaseT offset_base(const BaseT &base, ptrdiff_t index)
///     - Return a new base that is offset from the provide base by 'index'
///       elements.
template <typename DerivedT, typename BaseT, typename T,
          typename PointerT = T *, typename ReferenceT = T &>
class indexed_accessor_range_base {
public:
  /// An iterator element of this range.
  class iterator : public indexed_accessor_iterator<iterator, BaseT, T,
                                                    PointerT, ReferenceT> {
  public:
    // Index into this iterator, invoking a static method on the derived type.
    ReferenceT operator*() const {
      return DerivedT::dereference_iterator(this->getBase(), this->getIndex());
    }

  private:
    iterator(BaseT owner, ptrdiff_t curIndex)
        : indexed_accessor_iterator<iterator, BaseT, T, PointerT, ReferenceT>(
              owner, curIndex) {}

    /// Allow access to the constructor.
    friend indexed_accessor_range_base<DerivedT, BaseT, T, PointerT,
                                       ReferenceT>;
  };

  iterator begin() const { return iterator(base, 0); }
  iterator end() const { return iterator(base, count); }
  ReferenceT operator[](unsigned index) const {
    assert(index < size() && "invalid index for value range");
    return *std::next(begin(), index);
  }

  /// Return the size of this range.
  size_t size() const { return count; }

  /// Return if the range is empty.
  bool empty() const { return size() == 0; }

  /// Drop the first N elements, and keep M elements.
  DerivedT slice(unsigned n, unsigned m) const {
    assert(n + m <= size() && "invalid size specifiers");
    return DerivedT(DerivedT::offset_base(base, n), m);
  }

  /// Drop the first n elements.
  DerivedT drop_front(unsigned n = 1) const {
    assert(size() >= n && "Dropping more elements than exist");
    return slice(n, size() - n);
  }
  /// Drop the last n elements.
  DerivedT drop_back(unsigned n = 1) const {
    assert(size() >= n && "Dropping more elements than exist");
    return DerivedT(base, size() - n);
  }

protected:
  indexed_accessor_range_base(BaseT base, ptrdiff_t count)
      : base(base), count(count) {}
  indexed_accessor_range_base(const indexed_accessor_range_base &) = default;
  indexed_accessor_range_base(indexed_accessor_range_base &&) = default;
  indexed_accessor_range_base &
  operator=(const indexed_accessor_range_base &) = default;

  /// The base that owns the provided range of values.
  BaseT base;
  /// The size from the owning range.
  ptrdiff_t count;
};
} // end namespace detail

/// This class provides an implementation of a range of
/// indexed_accessor_iterators where the base is not indexable. Ranges with
/// bases that are offsetable should derive from indexed_accessor_range_base
/// instead. Derived range classes are expected to implement the following
/// static method:
///   * ReferenceT dereference_iterator(const BaseT &base, ptrdiff_t index)
///     - Derefence an iterator pointing to a parent base at the given index.
template <typename DerivedT, typename BaseT, typename T,
          typename PointerT = T *, typename ReferenceT = T &>
class indexed_accessor_range
    : public detail::indexed_accessor_range_base<
          indexed_accessor_range<DerivedT, BaseT, T, PointerT, ReferenceT>,
          std::pair<BaseT, ptrdiff_t>, T, PointerT, ReferenceT> {
protected:
  indexed_accessor_range(BaseT base, ptrdiff_t startIndex, ptrdiff_t count)
      : detail::indexed_accessor_range_base<
            DerivedT, std::pair<BaseT, ptrdiff_t>, T, PointerT, ReferenceT>(
            std::make_pair(base, startIndex), count) {}

private:
  /// See `detail::indexed_accessor_range_base` for details.
  static std::pair<BaseT, ptrdiff_t>
  offset_base(const std::pair<BaseT, ptrdiff_t> &base, ptrdiff_t index) {
    // We encode the internal base as a pair of the derived base and a start
    // index into the derived base.
    return std::make_pair(base.first, base.second + index);
  }
  /// See `detail::indexed_accessor_range_base` for details.
  static ReferenceT
  dereference_iterator(const std::pair<BaseT, ptrdiff_t> &base,
                       ptrdiff_t index) {
    return DerivedT::dereference_iterator(base.first, base.second + index);
  }

  /// Allow access to `offset_base` and `dereference_iterator`.
  friend detail::indexed_accessor_range_base<
      indexed_accessor_range<DerivedT, BaseT, T, PointerT, ReferenceT>,
      std::pair<BaseT, ptrdiff_t>, T, PointerT, ReferenceT>;
};

/// Given a container of pairs, return a range over the second elements.
template <typename ContainerTy> auto make_second_range(ContainerTy &&c) {
  return llvm::map_range(
      std::forward<ContainerTy>(c),
      [](decltype((*std::begin(c))) elt) -> decltype((elt.second)) {
        return elt.second;
      });
}

/// Returns true of the given range only contains a single element.
template <typename ContainerTy> bool has_single_element(ContainerTy &&c) {
  auto it = std::begin(c), e = std::end(c);
  return it != e && std::next(it) == e;
}
} // end namespace mlir

// Allow tuples to be usable as DenseMap keys.
// TODO: Move this to upstream LLVM.

/// Simplistic combination of 32-bit hash values into 32-bit hash values.
/// This function is taken from llvm/ADT/DenseMapInfo.h.
static inline unsigned llvm_combineHashValue(unsigned a, unsigned b) {
  uint64_t key = (uint64_t)a << 32 | (uint64_t)b;
  key += ~(key << 32);
  key ^= (key >> 22);
  key += ~(key << 13);
  key ^= (key >> 8);
  key += (key << 3);
  key ^= (key >> 15);
  key += ~(key << 27);
  key ^= (key >> 31);
  return (unsigned)key;
}

namespace llvm {
template <typename... Ts> struct DenseMapInfo<std::tuple<Ts...>> {
  using Tuple = std::tuple<Ts...>;

  static inline Tuple getEmptyKey() {
    return Tuple(DenseMapInfo<Ts>::getEmptyKey()...);
  }

  static inline Tuple getTombstoneKey() {
    return Tuple(DenseMapInfo<Ts>::getTombstoneKey()...);
  }

  template <unsigned I>
  static unsigned getHashValueImpl(const Tuple &values, std::false_type) {
    using EltType = typename std::tuple_element<I, Tuple>::type;
    std::integral_constant<bool, I + 1 == sizeof...(Ts)> atEnd;
    return llvm_combineHashValue(
        DenseMapInfo<EltType>::getHashValue(std::get<I>(values)),
        getHashValueImpl<I + 1>(values, atEnd));
  }

  template <unsigned I>
  static unsigned getHashValueImpl(const Tuple &values, std::true_type) {
    return 0;
  }

  static unsigned getHashValue(const std::tuple<Ts...> &values) {
    std::integral_constant<bool, 0 == sizeof...(Ts)> atEnd;
    return getHashValueImpl<0>(values, atEnd);
  }

  template <unsigned I>
  static bool isEqualImpl(const Tuple &lhs, const Tuple &rhs, std::false_type) {
    using EltType = typename std::tuple_element<I, Tuple>::type;
    std::integral_constant<bool, I + 1 == sizeof...(Ts)> atEnd;
    return DenseMapInfo<EltType>::isEqual(std::get<I>(lhs), std::get<I>(rhs)) &&
           isEqualImpl<I + 1>(lhs, rhs, atEnd);
  }

  template <unsigned I>
  static bool isEqualImpl(const Tuple &lhs, const Tuple &rhs, std::true_type) {
    return true;
  }

  static bool isEqual(const Tuple &lhs, const Tuple &rhs) {
    std::integral_constant<bool, 0 == sizeof...(Ts)> atEnd;
    return isEqualImpl<0>(lhs, rhs, atEnd);
  }
};

} // end namespace llvm

#endif // MLIR_SUPPORT_STLEXTRAS_H
