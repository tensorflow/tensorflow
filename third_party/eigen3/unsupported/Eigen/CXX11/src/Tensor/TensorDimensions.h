// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_DIMENSIONS_H
#define EIGEN_CXX11_TENSOR_TENSOR_DIMENSIONS_H


namespace Eigen {

/** \internal
  *
  * \class TensorDimensions
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Set of classes used to encode and store the dimensions of a Tensor.
  *
  * The Sizes class encodes as part of the type the number of dimensions and the
  * sizes corresponding to each dimension. It uses no storage space since it is
  * entirely known at compile time.
  * The DSizes class is its dynamic sibling: the number of dimensions is known
  * at compile time but the sizes are set during execution.
  *
  * \sa Tensor
  */

// Can't use std::pairs on cuda devices
template <typename Index> struct IndexPair {
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE IndexPair() : first(0), second(0) { }
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE IndexPair(Index f, Index s) : first(f), second(s) { }
  Index first;
  Index second;
};

// Boilerplate code
namespace internal {

template<std::size_t n, typename Dimension> struct dget {
  static const std::size_t value = get<n, typename Dimension::Base>::value;
};


template<typename Index, std::size_t NumIndices, std::size_t n, bool RowMajor>
struct fixed_size_tensor_index_linearization_helper
{
  template <typename Dimensions> EIGEN_DEVICE_FUNC
  static inline Index run(array<Index, NumIndices> const& indices,
                          const Dimensions& dimensions)
  {
    return array_get<RowMajor ? n - 1 : (NumIndices - n)>(indices) +
        dget<RowMajor ? n - 1 : (NumIndices - n), Dimensions>::value *
        fixed_size_tensor_index_linearization_helper<Index, NumIndices, n - 1, RowMajor>::run(indices, dimensions);
  }
};

template<typename Index, std::size_t NumIndices, bool RowMajor>
struct fixed_size_tensor_index_linearization_helper<Index, NumIndices, 0, RowMajor>
{
  template <typename Dimensions> EIGEN_DEVICE_FUNC
  static inline Index run(array<Index, NumIndices> const& indices,
                          const Dimensions&)
  {
    return 0;
  }
};

template<typename Index, std::size_t n>
struct fixed_size_tensor_index_extraction_helper
{
  template <typename Dimensions> EIGEN_DEVICE_FUNC
  static inline Index run(const Index index,
                          const Dimensions& dimensions)
  {
    const Index mult = (index == n) ? 1 : 0;
    return array_get<n>(dimensions) * mult +
        fixed_size_tensor_index_extraction_helper<Index, n - 1>::run(index, dimensions);
  }
};

template<typename Index>
struct fixed_size_tensor_index_extraction_helper<Index, 0>
{
  template <typename Dimensions> EIGEN_DEVICE_FUNC
  static inline Index run(const Index index,
                          const Dimensions& dimensions)
  {
    const Index mult = (index == 0) ? 1 : 0;
    return array_get<0>(dimensions) * mult;
  }
};

}  // end namespace internal


// Fixed size
#ifndef EIGEN_EMULATE_CXX11_META_H
template <typename std::size_t... Indices>
struct Sizes : internal::numeric_list<std::size_t, Indices...> {
  typedef internal::numeric_list<std::size_t, Indices...> Base;
  static const std::size_t total_size = internal::arg_prod(Indices...);

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t rank() const {
    return Base::count;
  }

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE std::size_t TotalSize() {
    return internal::arg_prod(Indices...);
  }

  Sizes() { }
  template <typename DenseIndex>
  explicit Sizes(const array<DenseIndex, Base::count>& /*indices*/) {
    // todo: add assertion
  }
#ifdef EIGEN_HAS_VARIADIC_TEMPLATES
  template <typename... DenseIndex> Sizes(DenseIndex...) { }
  explicit Sizes(std::initializer_list<std::size_t> /*l*/) {
    // todo: add assertion
  }
#endif

  template <typename T> Sizes& operator = (const T& /*other*/) {
    // add assertion failure if the size of other is different
    return *this;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE std::ptrdiff_t operator[] (const int index) const {
    return internal::fixed_size_tensor_index_extraction_helper<std::ptrdiff_t, Base::count - 1>::run(index, *this);
  }

  template <typename DenseIndex> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  size_t IndexOfColMajor(const array<DenseIndex, Base::count>& indices) const {
    return internal::fixed_size_tensor_index_linearization_helper<DenseIndex, Base::count, Base::count, false>::run(indices, *static_cast<const Base*>(this));
  }
  template <typename DenseIndex> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  size_t IndexOfRowMajor(const array<DenseIndex, Base::count>& indices) const {
    return internal::fixed_size_tensor_index_linearization_helper<DenseIndex, Base::count, Base::count, true>::run(indices, *static_cast<const Base*>(this));
  }
};

namespace internal {
template <typename std::size_t... Indices>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE std::size_t array_prod(const Sizes<Indices...>&) {
  return Sizes<Indices...>::total_size;
}
}

#else

template <std::size_t n>
struct non_zero_size {
  typedef internal::type2val<std::size_t, n> type;
};
template <>
struct non_zero_size<0> {
  typedef internal::null_type type;
};

template <std::size_t V1=0, std::size_t V2=0, std::size_t V3=0, std::size_t V4=0, std::size_t V5=0> struct Sizes {
  typedef typename internal::make_type_list<typename non_zero_size<V1>::type, typename non_zero_size<V2>::type, typename non_zero_size<V3>::type, typename non_zero_size<V4>::type, typename non_zero_size<V5>::type >::type Base;
  static const size_t count = Base::count;
  static const std::size_t total_size = internal::arg_prod<Base>::value;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t rank() const {
    return count;
  }

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t TotalSize() {
    return internal::arg_prod<Base>::value;
  }

  Sizes() { }
  template <typename DenseIndex>
  explicit Sizes(const array<DenseIndex, Base::count>& indices) {
    // todo: add assertion
  }
#ifdef EIGEN_HAS_VARIADIC_TEMPLATES
  template <typename... DenseIndex> Sizes(DenseIndex... indices) { }
  explicit Sizes(std::initializer_list<std::size_t> l) {
    // todo: add assertion
  }
#else
  EIGEN_DEVICE_FUNC explicit Sizes(const DenseIndex i0) {
  }
  EIGEN_DEVICE_FUNC explicit Sizes(const DenseIndex i0, const DenseIndex i1) {
  }
  EIGEN_DEVICE_FUNC explicit Sizes(const DenseIndex i0, const DenseIndex i1, const DenseIndex i2) {
  }
  EIGEN_DEVICE_FUNC explicit Sizes(const DenseIndex i0, const DenseIndex i1, const DenseIndex i2, const DenseIndex i3) {
  }
  EIGEN_DEVICE_FUNC explicit Sizes(const DenseIndex i0, const DenseIndex i1, const DenseIndex i2, const DenseIndex i3, const DenseIndex i4) {
  }
#endif

  template <typename T> Sizes& operator = (const T& other) {
    // to do: check the size of other
    return *this;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE std::size_t operator[] (const int index) const {
    switch (index) {
      case 0:
        return internal::get<0, Base>::value;
      case 1:
        return internal::get<1, Base>::value;
      case 2:
        return internal::get<2, Base>::value;
      case 3:
        return internal::get<3, Base>::value;
      case 4:
        return internal::get<4, Base>::value;
      default:
        eigen_assert(false && "index overflow");
        return static_cast<std::size_t>(-1);
    }
  }

  template <typename DenseIndex> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  size_t IndexOfColMajor(const array<DenseIndex, Base::count>& indices) const {
    return internal::fixed_size_tensor_index_linearization_helper<DenseIndex, Base::count, Base::count, false>::run(indices, *this);
  }
  template <typename DenseIndex> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  size_t IndexOfRowMajor(const array<DenseIndex, Base::count>& indices) const {
    return internal::fixed_size_tensor_index_linearization_helper<DenseIndex, Base::count, Base::count, true>::run(indices, *this);
  }
};

namespace internal {
template <std::size_t V1, std::size_t V2, std::size_t V3, std::size_t V4, std::size_t V5>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE std::size_t array_prod(const Sizes<V1, V2, V3, V4, V5>&) {
  return Sizes<V1, V2, V3, V4, V5>::total_size;
}
}

#endif

// Boilerplate
namespace internal {
template<typename Index, std::size_t NumIndices, std::size_t n, bool RowMajor>
struct tensor_index_linearization_helper
{
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  Index run(array<Index, NumIndices> const& indices, array<Index, NumIndices> const& dimensions)
  {
    return array_get<RowMajor ? n : (NumIndices - n - 1)>(indices) +
      array_get<RowMajor ? n : (NumIndices - n - 1)>(dimensions) *
        tensor_index_linearization_helper<Index, NumIndices, n - 1, RowMajor>::run(indices, dimensions);
  }
};

template<typename Index, std::size_t NumIndices, bool RowMajor>
struct tensor_index_linearization_helper<Index, NumIndices, 0, RowMajor>
{
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  Index run(array<Index, NumIndices> const& indices, array<Index, NumIndices> const&)
  {
    return array_get<RowMajor ? 0 : NumIndices - 1>(indices);
  }
};
}  // end namespace internal



// Dynamic size
template <typename DenseIndex, std::size_t NumDims>
struct DSizes : array<DenseIndex, NumDims> {
  typedef array<DenseIndex, NumDims> Base;
  static const std::size_t count = NumDims;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t rank() const {
    return NumDims;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t TotalSize() const {
    return internal::array_prod(*static_cast<const Base*>(this));
  }

  EIGEN_DEVICE_FUNC DSizes() {
    for (int i = 0 ; i < NumDims; ++i) {
      (*this)[i] = 0;
    }
  }
  EIGEN_DEVICE_FUNC DSizes(const array<DenseIndex, NumDims>& a) : Base(a) { }

  EIGEN_DEVICE_FUNC DSizes(const DimensionList<DenseIndex, NumDims>& a) {
    for (int i = 0 ; i < NumDims; ++i) {
      (*this)[i] = a[i];
    }
  }

#ifndef EIGEN_EMULATE_CXX11_META_H
  template <typename std::size_t... Indices>
  EIGEN_DEVICE_FUNC DSizes(const Sizes<Indices...>& a) {
    for (int i = 0 ; i < NumDims; ++i) {
      (*this)[i] = a[i];
    }
  }
#else
  template <std::size_t V1, std::size_t V2, std::size_t V3, std::size_t V4, std::size_t V5>
  EIGEN_DEVICE_FUNC DSizes(const Sizes<V1, V2, V3, V4, V5>& a) {
    for (int i = 0 ; i < NumDims; ++i) {
      (*this)[i] = a[i];
    }
  }
#endif

#ifdef EIGEN_HAS_VARIADIC_TEMPLATES
  template<typename... IndexTypes> EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE explicit DSizes(DenseIndex firstDimension, IndexTypes... otherDimensions) {
    EIGEN_STATIC_ASSERT(sizeof...(otherDimensions) + 1 == NumDims, YOU_MADE_A_PROGRAMMING_MISTAKE)
    (*this) = array<DenseIndex, NumDims>{{firstDimension, otherDimensions...}};
  }
#else
  EIGEN_DEVICE_FUNC explicit DSizes(const DenseIndex i0) {
    eigen_assert(NumDims == 1);
    (*this)[0] = i0;
  }
  EIGEN_DEVICE_FUNC explicit DSizes(const DenseIndex i0, const DenseIndex i1) {
    eigen_assert(NumDims == 2);
    (*this)[0] = i0;
    (*this)[1] = i1;
  }
  EIGEN_DEVICE_FUNC explicit DSizes(const DenseIndex i0, const DenseIndex i1, const DenseIndex i2) {
    eigen_assert(NumDims == 3);
    (*this)[0] = i0;
    (*this)[1] = i1;
    (*this)[2] = i2;
  }
  EIGEN_DEVICE_FUNC explicit DSizes(const DenseIndex i0, const DenseIndex i1, const DenseIndex i2, const DenseIndex i3) {
    eigen_assert(NumDims == 4);
    (*this)[0] = i0;
    (*this)[1] = i1;
    (*this)[2] = i2;
    (*this)[3] = i3;
  }
  EIGEN_DEVICE_FUNC explicit DSizes(const DenseIndex i0, const DenseIndex i1, const DenseIndex i2, const DenseIndex i3, const DenseIndex i4) {
    eigen_assert(NumDims == 5);
    (*this)[0] = i0;
    (*this)[1] = i1;
    (*this)[2] = i2;
    (*this)[3] = i3;
    (*this)[4] = i4;
  }
#endif

  EIGEN_DEVICE_FUNC DSizes& operator = (const array<DenseIndex, NumDims>& other) {
    *static_cast<Base*>(this) = other;
    return *this;
  }

  // A constexpr would be so much better here
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t IndexOfColMajor(const array<DenseIndex, NumDims>& indices) const {
    return internal::tensor_index_linearization_helper<DenseIndex, NumDims, NumDims - 1, false>::run(indices, *static_cast<const Base*>(this));
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t IndexOfRowMajor(const array<DenseIndex, NumDims>& indices) const {
    return internal::tensor_index_linearization_helper<DenseIndex, NumDims, NumDims - 1, true>::run(indices, *static_cast<const Base*>(this));
  }
};




// Boilerplate
namespace internal {
template<typename Index, std::size_t NumIndices, std::size_t n, bool RowMajor>
struct tensor_vsize_index_linearization_helper
{
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  Index run(array<Index, NumIndices> const& indices, std::vector<DenseIndex> const& dimensions)
  {
    return array_get<RowMajor ? n : (NumIndices - n - 1)>(indices) +
      array_get<RowMajor ? n : (NumIndices - n - 1)>(dimensions) *
        tensor_vsize_index_linearization_helper<Index, NumIndices, n - 1, RowMajor>::run(indices, dimensions);
  }
};

template<typename Index, std::size_t NumIndices, bool RowMajor>
struct tensor_vsize_index_linearization_helper<Index, NumIndices, 0, RowMajor>
{
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  Index run(array<Index, NumIndices> const& indices, std::vector<DenseIndex> const&)
  {
    return array_get<RowMajor ? 0 : NumIndices - 1>(indices);
  }
};
}  // end namespace internal


template <typename DenseIndex>
struct VSizes : std::vector<DenseIndex> {
  typedef std::vector<DenseIndex> Base;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t rank() const {
    return Base::size();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t TotalSize() const {
    return internal::array_prod(*static_cast<const Base*>(this));
  }

  EIGEN_DEVICE_FUNC VSizes() { }
  EIGEN_DEVICE_FUNC explicit VSizes(const std::vector<DenseIndex>& a) : Base(a) { }

  template <std::size_t NumDims>
  EIGEN_DEVICE_FUNC explicit VSizes(const array<DenseIndex, NumDims>& a) {
    this->resize(NumDims);
    for (int i = 0; i < NumDims; ++i) {
      (*this)[i] = a[i];
    }
  }
  template <std::size_t NumDims>
  EIGEN_DEVICE_FUNC explicit VSizes(const DSizes<DenseIndex, NumDims>& a) {
    this->resize(NumDims);
    for (int i = 0; i < NumDims; ++i) {
      (*this)[i] = a[i];
    }
  }

  EIGEN_DEVICE_FUNC explicit VSizes(const DenseIndex i0) {
    this->resize(1);
    (*this)[0] = i0;
  }
  EIGEN_DEVICE_FUNC explicit VSizes(const DenseIndex i0, const DenseIndex i1) {
    this->resize(2);
    (*this)[0] = i0;
    (*this)[1] = i1;
  }
  EIGEN_DEVICE_FUNC explicit VSizes(const DenseIndex i0, const DenseIndex i1, const DenseIndex i2) {
    this->resize(3);
    (*this)[0] = i0;
    (*this)[1] = i1;
    (*this)[2] = i2;
  }
  EIGEN_DEVICE_FUNC explicit VSizes(const DenseIndex i0, const DenseIndex i1, const DenseIndex i2, const DenseIndex i3) {
    this->resize(4);
    (*this)[0] = i0;
    (*this)[1] = i1;
    (*this)[2] = i2;
    (*this)[3] = i3;
  }
  EIGEN_DEVICE_FUNC explicit VSizes(const DenseIndex i0, const DenseIndex i1, const DenseIndex i2, const DenseIndex i3, const DenseIndex i4) {
    this->resize(5);
    (*this)[0] = i0;
    (*this)[1] = i1;
    (*this)[2] = i2;
    (*this)[3] = i3;
    (*this)[4] = i4;
  }

  EIGEN_DEVICE_FUNC VSizes& operator = (const std::vector<DenseIndex>& other) {
    *static_cast<Base*>(this) = other;
    return *this;
  }
  template <std::size_t NumDims>
  EIGEN_DEVICE_FUNC VSizes& operator = (const array<DenseIndex, NumDims>& a) {
    this->resize(NumDims);
    for (int i = 0; i < NumDims; ++i) {
      (*this)[i] = a[i];
    }
    return *this;
  }
  template <std::size_t NumDims>
  EIGEN_DEVICE_FUNC VSizes& operator = (const DSizes<DenseIndex, NumDims>& a) {
    this->resize(NumDims);
    for (int i = 0; i < NumDims; ++i) {
      (*this)[i] = a[i];
    }
    return *this;
  }

  // A constexpr would be so much better here
  template <std::size_t NumDims>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t IndexOfColMajor(const array<DenseIndex, NumDims>& indices) const {
    return internal::tensor_vsize_index_linearization_helper<DenseIndex, NumDims, NumDims - 1, false>::run(indices, *static_cast<const Base*>(this));
  }
  template <std::size_t NumDims>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t IndexOfRowMajor(const array<DenseIndex, NumDims>& indices) const {
    return internal::tensor_vsize_index_linearization_helper<DenseIndex, NumDims, NumDims - 1, true>::run(indices, *static_cast<const Base*>(this));
  }
};


// Boilerplate
namespace internal {
template <typename DenseIndex>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE DenseIndex array_prod(const VSizes<DenseIndex>& sizes) {
  DenseIndex total_size = 1;
  for (int i = 0; i < sizes.size(); ++i) {
    total_size *= sizes[i];
  }
  return total_size;
};
}

namespace internal {

template <typename DenseIndex, std::size_t NumDims> struct array_size<const DSizes<DenseIndex, NumDims> > {
  static const size_t value = NumDims;
};
template <typename DenseIndex, std::size_t NumDims> struct array_size<DSizes<DenseIndex, NumDims> > {
  static const size_t value = NumDims;
};
template <typename DenseIndex>
struct array_size<VSizes<DenseIndex> > {
  static const ptrdiff_t value = -1;
};
#ifndef EIGEN_EMULATE_CXX11_META_H
template <typename std::size_t... Indices> struct array_size<const Sizes<Indices...> > {
static const size_t value = Sizes<Indices...>::count;
};
template <typename std::size_t... Indices> struct array_size<Sizes<Indices...> > {
static const size_t value = Sizes<Indices...>::count;
};
template <std::size_t n, typename std::size_t... Indices> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE std::ptrdiff_t array_get(const Sizes<Indices...>&) {
  return get<n, internal::numeric_list<std::size_t, Indices...> >::value;
}
#else
template <std::size_t V1, std::size_t V2, std::size_t V3, std::size_t V4, std::size_t V5> struct array_size<const Sizes<V1,V2,V3,V4,V5> > {
  static const size_t value = Sizes<V1,V2,V3,V4,V5>::count;
};
template <std::size_t V1, std::size_t V2, std::size_t V3, std::size_t V4, std::size_t V5> struct array_size<Sizes<V1,V2,V3,V4,V5> > {
  static const size_t value = Sizes<V1,V2,V3,V4,V5>::count;
};
template <std::size_t n, std::size_t V1, std::size_t V2, std::size_t V3, std::size_t V4, std::size_t V5> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE std::ptrdiff_t array_get(const Sizes<V1,V2,V3,V4,V5>& a) {
  return get<n, typename Sizes<V1,V2,V3,V4,V5>::Base>::value;
}

#endif


template <typename Dims1, typename Dims2, size_t n, size_t m>
struct sizes_match_below_dim {
  static inline bool run(Dims1& dims1, Dims2& dims2) {
    return false;
  }
};
template <typename Dims1, typename Dims2, size_t n>
struct sizes_match_below_dim<Dims1, Dims2, n, n> {
  static inline bool run(Dims1& dims1, Dims2& dims2) {
    return (array_get<n-1>(dims1) == array_get<n-1>(dims2)) &
        sizes_match_below_dim<Dims1, Dims2, n-1, n-1>::run(dims1, dims2);
  }
};
template <typename Dims1, typename Dims2>
struct sizes_match_below_dim<Dims1, Dims2, 0, 0> {
  static inline bool run(Dims1& dims1, Dims2& dims2) {
    return true;
  }
};

} // end namespace internal


template <typename Dims1, typename Dims2>
bool dimensions_match(Dims1& dims1, Dims2& dims2) {
  return internal::sizes_match_below_dim<Dims1, Dims2, internal::array_size<Dims1>::value, internal::array_size<Dims2>::value>::run(dims1, dims2);
}

template <typename IndexType, typename Dims2>
bool dimensions_match(const VSizes<IndexType>& dims1, Dims2& dims2) {
  if (dims1.size() != internal::array_size<Dims2>::value) {
    return false;
  }
  for (int i = 0; i < internal::array_size<Dims2>::value; ++i) {
    if (dims1[i] != dims2[i]) {
      return false;
    }
  }
  return true;
}

template <typename Dims1, typename IndexType>
bool dimensions_match(Dims1& dims1, const VSizes<IndexType>& dims2) {
  if (internal::array_size<Dims1>::value != dims2.size()) {
    return false;
  }
  for (int i = 0; i < internal::array_size<Dims1>::value; ++i) {
    if (dims1[i] != dims2[i]) {
      return false;
    }
  }
  return true;
}

template <typename IndexType>
bool dimensions_match(const VSizes<IndexType>& dims1, const VSizes<IndexType>& dims2) {
  return dims1 == dims2;
}

} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_DIMENSIONS_H
