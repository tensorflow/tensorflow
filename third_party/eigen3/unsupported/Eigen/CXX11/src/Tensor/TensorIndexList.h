// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_INDEX_LIST_H
#define EIGEN_CXX11_TENSOR_TENSOR_INDEX_LIST_H

#if defined(EIGEN_HAS_CONSTEXPR) && defined(EIGEN_HAS_VARIADIC_TEMPLATES)

#define EIGEN_HAS_INDEX_LIST

namespace Eigen {

/** \internal
  *
  * \class TensorIndexList
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Set of classes used to encode a set of Tensor dimensions/indices.
  *
  * The indices in the list can be known at compile time or at runtime. A mix
  * of static and dynamic indices can also be provided if needed. The tensor
  * code will attempt to take advantage of the indices that are known at
  * compile time to optimize the code it generates.
  *
  * This functionality requires a c++11 compliant compiler. If your compiler
  * is older you need to use arrays of indices instead.
  *
  * Several examples are provided in the cxx11_tensor_index_list.cpp file.
  *
  * \sa Tensor
  */

template <DenseIndex n>
struct type2index {
  static const DenseIndex value = n;
  constexpr operator DenseIndex() const { return n; }
  void set(DenseIndex val) {
    eigen_assert(val == n);
  }
};

namespace internal {
template <typename T>
void update_value(T& val, DenseIndex new_val) {
  val = new_val;
}
template <DenseIndex n>
void update_value(type2index<n>& val, DenseIndex new_val) {
  val.set(new_val);
}

template <typename T>
struct is_compile_time_constant {
  static constexpr bool value = false;
};

template <DenseIndex idx>
struct is_compile_time_constant<type2index<idx> > {
  static constexpr bool value = true;
};
template <DenseIndex idx>
struct is_compile_time_constant<const type2index<idx> > {
  static constexpr bool value = true;
};
template <DenseIndex idx>
struct is_compile_time_constant<type2index<idx>& > {
  static constexpr bool value = true;
};
template <DenseIndex idx>
struct is_compile_time_constant<const type2index<idx>& > {
  static constexpr bool value = true;
};

template <DenseIndex Idx>
struct tuple_coeff {
  template <typename... T>
  static constexpr DenseIndex get(const DenseIndex i, const std::tuple<T...>& t) {
    return std::get<Idx>(t) * (i == Idx) + tuple_coeff<Idx-1>::get(i, t) * (i != Idx);
  }
  template <typename... T>
  static void set(const DenseIndex i, std::tuple<T...>& t, const DenseIndex value) {
    if (i == Idx) {
      update_value(std::get<Idx>(t), value);
    } else {
      tuple_coeff<Idx-1>::set(i, t, value);
    }
  }

  template <typename... T>
  static constexpr bool value_known_statically(const DenseIndex i, const std::tuple<T...>& t) {
    return ((i == Idx) & is_compile_time_constant<typename std::tuple_element<Idx, std::tuple<T...> >::type>::value) ||
        tuple_coeff<Idx-1>::value_known_statically(i, t);
  }

  template <typename... T>
  static constexpr bool values_up_to_known_statically(const std::tuple<T...>& t) {
    return is_compile_time_constant<typename std::tuple_element<Idx, std::tuple<T...> >::type>::value &&
        tuple_coeff<Idx-1>::values_up_to_known_statically(t);
  }

  template <typename... T>
  static constexpr bool values_up_to_statically_known_to_increase(const std::tuple<T...>& t) {
    return is_compile_time_constant<typename std::tuple_element<Idx, std::tuple<T...> >::type>::value &&
           is_compile_time_constant<typename std::tuple_element<Idx-1, std::tuple<T...> >::type>::value &&
           std::get<Idx>(t) > std::get<Idx-1>(t) &&
           tuple_coeff<Idx-1>::values_up_to_statically_known_to_increase(t);
  }
};

template <>
struct tuple_coeff<0> {
  template <typename... T>
  static constexpr DenseIndex get(const DenseIndex i, const std::tuple<T...>& t) {
    //  eigen_assert (i == 0);  // gcc fails to compile assertions in constexpr
    return std::get<0>(t) * (i == 0);
  }
  template <typename... T>
  static void set(const DenseIndex i, std::tuple<T...>& t, const DenseIndex value) {
    eigen_assert (i == 0);
    update_value(std::get<0>(t), value);
  }
  template <typename... T>
  static constexpr bool value_known_statically(const DenseIndex i, const std::tuple<T...>& t) {
    //    eigen_assert (i == 0);  // gcc fails to compile assertions in constexpr
    return is_compile_time_constant<typename std::tuple_element<0, std::tuple<T...> >::type>::value & (i == 0);
  }

  template <typename... T>
  static constexpr bool values_up_to_known_statically(const std::tuple<T...>& t) {
    return is_compile_time_constant<typename std::tuple_element<0, std::tuple<T...> >::type>::value;
  }

  template <typename... T>
  static constexpr bool values_up_to_statically_known_to_increase(const std::tuple<T...>& t) {
    return true;
  }
};
}  // namespace internal


template<typename FirstType, typename... OtherTypes>
struct IndexList : std::tuple<FirstType, OtherTypes...> {
  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC constexpr DenseIndex operator[] (const DenseIndex i) const {
    return internal::tuple_coeff<std::tuple_size<std::tuple<FirstType, OtherTypes...> >::value-1>::get(i, *this);
  }
  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC void set(const DenseIndex i, const DenseIndex value) {
    return internal::tuple_coeff<std::tuple_size<std::tuple<FirstType, OtherTypes...> >::value-1>::set(i, *this, value);
  }

  constexpr IndexList(const std::tuple<FirstType, OtherTypes...>& other) : std::tuple<FirstType, OtherTypes...>(other) { }
  constexpr IndexList() : std::tuple<FirstType, OtherTypes...>() { }

  constexpr bool value_known_statically(const DenseIndex i) const {
    return internal::tuple_coeff<std::tuple_size<std::tuple<FirstType, OtherTypes...> >::value-1>::value_known_statically(i, *this);
  }
  constexpr bool all_values_known_statically() const {
    return internal::tuple_coeff<std::tuple_size<std::tuple<FirstType, OtherTypes...> >::value-1>::values_up_to_known_statically(*this);
  }

  constexpr bool values_statically_known_to_increase() const {
    return internal::tuple_coeff<std::tuple_size<std::tuple<FirstType, OtherTypes...> >::value-1>::values_up_to_statically_known_to_increase(*this);
  }
};


template<typename FirstType, typename... OtherTypes>
constexpr IndexList<FirstType, OtherTypes...> make_index_list(FirstType val1, OtherTypes... other_vals) {
  return std::make_tuple(val1, other_vals...);
}


namespace internal {

template<typename FirstType, typename... OtherTypes> size_t array_prod(const IndexList<FirstType, OtherTypes...>& sizes) {
  size_t result = 1;
  for (int i = 0; i < array_size<IndexList<FirstType, OtherTypes...> >::value; ++i) {
    result *= sizes[i];
  }
  return result;
};

template<typename FirstType, typename... OtherTypes> struct array_size<IndexList<FirstType, OtherTypes...> > {
  static const size_t value = std::tuple_size<std::tuple<FirstType, OtherTypes...> >::value;
};
template<typename FirstType, typename... OtherTypes> struct array_size<const IndexList<FirstType, OtherTypes...> > {
  static const size_t value = std::tuple_size<std::tuple<FirstType, OtherTypes...> >::value;
};

template<DenseIndex n, typename FirstType, typename... OtherTypes> constexpr DenseIndex array_get(IndexList<FirstType, OtherTypes...>& a) {
  return std::get<n>(a);
}
template<DenseIndex n, typename FirstType, typename... OtherTypes> constexpr DenseIndex array_get(const IndexList<FirstType, OtherTypes...>& a) {
  return std::get<n>(a);
}

template <typename T>
struct index_known_statically {
  constexpr bool operator() (DenseIndex) const {
    return false;
  }
};

template <typename FirstType, typename... OtherTypes>
struct index_known_statically<IndexList<FirstType, OtherTypes...> > {
  constexpr bool operator() (const DenseIndex i) const {
    return IndexList<FirstType, OtherTypes...>().value_known_statically(i);
  }
};

template <typename FirstType, typename... OtherTypes>
struct index_known_statically<const IndexList<FirstType, OtherTypes...> > {
  constexpr bool operator() (const DenseIndex i) const {
    return IndexList<FirstType, OtherTypes...>().value_known_statically(i);
  }
};

template <typename T>
struct all_indices_known_statically {
  constexpr bool operator() () const {
    return false;
  }
};

template <typename FirstType, typename... OtherTypes>
struct all_indices_known_statically<IndexList<FirstType, OtherTypes...> > {
  constexpr bool operator() () const {
    return IndexList<FirstType, OtherTypes...>().all_values_known_statically();
  }
};

template <typename FirstType, typename... OtherTypes>
struct all_indices_known_statically<const IndexList<FirstType, OtherTypes...> > {
  constexpr bool operator() () const {
    return IndexList<FirstType, OtherTypes...>().all_values_known_statically();
  }
};

template <typename T>
struct indices_statically_known_to_increase {
  constexpr bool operator() () const {
    return false;
  }
};

template <typename FirstType, typename... OtherTypes>
struct indices_statically_known_to_increase<IndexList<FirstType, OtherTypes...> > {
  constexpr bool operator() () const {
    return IndexList<FirstType, OtherTypes...>().values_statically_known_to_increase();
  }
};

template <typename FirstType, typename... OtherTypes>
struct indices_statically_known_to_increase<const IndexList<FirstType, OtherTypes...> > {
  constexpr bool operator() () const {
    return IndexList<FirstType, OtherTypes...>().values_statically_known_to_increase();
  }
};

template <typename Tx>
struct index_statically_eq {
  constexpr bool operator() (DenseIndex, DenseIndex) const {
    return false;
  }
};

template <typename FirstType, typename... OtherTypes>
struct index_statically_eq<IndexList<FirstType, OtherTypes...> > {
  constexpr bool operator() (const DenseIndex i, const DenseIndex value) const {
    return IndexList<FirstType, OtherTypes...>().value_known_statically(i) &&
        IndexList<FirstType, OtherTypes...>()[i] == value;
  }
};

template <typename FirstType, typename... OtherTypes>
struct index_statically_eq<const IndexList<FirstType, OtherTypes...> > {
  constexpr bool operator() (const DenseIndex i, const DenseIndex value) const {
    return IndexList<FirstType, OtherTypes...>().value_known_statically(i) &&
        IndexList<FirstType, OtherTypes...>()[i] == value;
  }
};

template <typename T>
struct index_statically_ne {
  constexpr bool operator() (DenseIndex, DenseIndex) const {
  return false;
  }
};

template <typename FirstType, typename... OtherTypes>
struct index_statically_ne<IndexList<FirstType, OtherTypes...> > {
  constexpr bool operator() (const DenseIndex i, const DenseIndex value) const {
    return IndexList<FirstType, OtherTypes...>().value_known_statically(i) &&
        IndexList<FirstType, OtherTypes...>()[i] != value;
  }
};

template <typename FirstType, typename... OtherTypes>
struct index_statically_ne<const IndexList<FirstType, OtherTypes...> > {
  constexpr bool operator() (const DenseIndex i, const DenseIndex value) const {
    return IndexList<FirstType, OtherTypes...>().value_known_statically(i) &&
        IndexList<FirstType, OtherTypes...>()[i] != value;
  }
};


template <typename T>
struct index_statically_gt {
  constexpr bool operator() (DenseIndex, DenseIndex) const {
  return false;
  }
};

template <typename FirstType, typename... OtherTypes>
struct index_statically_gt<IndexList<FirstType, OtherTypes...> > {
  constexpr bool operator() (const DenseIndex i, const DenseIndex value) const {
    return IndexList<FirstType, OtherTypes...>().value_known_statically(i) &&
        IndexList<FirstType, OtherTypes...>()[i] > value;
  }
};

template <typename FirstType, typename... OtherTypes>
struct index_statically_gt<const IndexList<FirstType, OtherTypes...> > {
  constexpr bool operator() (const DenseIndex i, const DenseIndex value) const {
    return IndexList<FirstType, OtherTypes...>().value_known_statically(i) &&
        IndexList<FirstType, OtherTypes...>()[i] > value;
  }
};

template <typename T>
struct index_statically_lt {
  constexpr bool operator() (DenseIndex, DenseIndex) const {
  return false;
  }
};

template <typename FirstType, typename... OtherTypes>
struct index_statically_lt<IndexList<FirstType, OtherTypes...> > {
  constexpr bool operator() (const DenseIndex i, const DenseIndex value) const {
    return IndexList<FirstType, OtherTypes...>().value_known_statically(i) &&
        IndexList<FirstType, OtherTypes...>()[i] < value;
  }
};

template <typename FirstType, typename... OtherTypes>
struct index_statically_lt<const IndexList<FirstType, OtherTypes...> > {
  constexpr bool operator() (const DenseIndex i, const DenseIndex value) const {
    return IndexList<FirstType, OtherTypes...>().value_known_statically(i) &&
        IndexList<FirstType, OtherTypes...>()[i] < value;
  }
};

}  // end namespace internal
}  // end namespace Eigen

#else

namespace Eigen {
namespace internal {

// No C++11 support
template <typename T>
struct index_known_statically {
  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC bool operator() (DenseIndex) const{
    return false;
  }
};

template <typename T>
struct all_indices_known_statically {
  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC bool operator() () const {
    return false;
  }
};

template <typename T>
struct indices_statically_known_to_increase {
  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC bool operator() () const {
    return false;
  }
};

template <typename T>
struct index_statically_eq {
  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC bool operator() (DenseIndex, DenseIndex) const{
    return false;
  }
};

template <typename T>
struct index_statically_ne {
  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC bool operator() (DenseIndex, DenseIndex) const{
    return false;
  }
};

template <typename T>
struct index_statically_gt {
  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC bool operator() (DenseIndex, DenseIndex) const{
    return false;
  }
};

template <typename T>
struct index_statically_lt {
  EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC bool operator() (DenseIndex, DenseIndex) const{
    return false;
  }
};

}  // end namespace internal
}  // end namespace Eigen

#endif

#endif // EIGEN_CXX11_TENSOR_TENSOR_INDEX_LIST_H
