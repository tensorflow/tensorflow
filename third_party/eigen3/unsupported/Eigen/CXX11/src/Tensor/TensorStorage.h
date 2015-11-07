// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2013 Christian Seiler <christian@iwakd.de>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSORSTORAGE_H
#define EIGEN_CXX11_TENSOR_TENSORSTORAGE_H

#ifdef EIGEN_TENSOR_STORAGE_CTOR_PLUGIN
  #define EIGEN_INTERNAL_TENSOR_STORAGE_CTOR_PLUGIN EIGEN_TENSOR_STORAGE_CTOR_PLUGIN;
#else
  #define EIGEN_INTERNAL_TENSOR_STORAGE_CTOR_PLUGIN
#endif

namespace Eigen {

/** \internal
  *
  * \class TensorStorage
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Stores the data of a tensor
  *
  * This class stores the data of fixed-size, dynamic-size or mixed tensors
  * in a way as compact as possible.
  *
  * \sa Tensor
  */
template<typename T, typename Dimensions, int Options_> class TensorStorage;


// Pure fixed-size storage
template<typename T, int Options_, typename FixedDimensions>
class TensorStorage<T, FixedDimensions, Options_>
{
 private:
  static const std::size_t Size = FixedDimensions::total_size;

  EIGEN_ALIGN_DEFAULT T m_data[Size];
  FixedDimensions m_dimensions;

 public:
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE TensorStorage() {
    EIGEN_STATIC_ASSERT(Size == FixedDimensions::total_size, YOU_MADE_A_PROGRAMMING_MISTAKE)
  }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE T *data() { return m_data; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const T *data() const { return m_data; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const FixedDimensions& dimensions() const { return m_dimensions; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE DenseIndex size() const { return m_dimensions.TotalSize(); }
};


// pure dynamic
template<typename T, int Options_, typename IndexType, std::size_t NumIndices_>
class TensorStorage<T, DSizes<IndexType, NumIndices_>, Options_>
{
  public:
    typedef IndexType Index;
    typedef DSizes<IndexType, NumIndices_> Dimensions;
    typedef TensorStorage<T, DSizes<IndexType, NumIndices_>, Options_> Self;

    EIGEN_DEVICE_FUNC TensorStorage()
      : m_data(NumIndices_ ? 0 : internal::conditional_aligned_new_auto<T,(Options_&DontAlign)==0>(1))
      , m_dimensions() {}

    EIGEN_DEVICE_FUNC TensorStorage(internal::constructor_without_unaligned_array_assert)
      : m_data(NumIndices_ ? 0 : internal::conditional_aligned_new_auto<T,(Options_&DontAlign)==0>(1))
      , m_dimensions(internal::template repeat<NumIndices_, Index>(0)) {}

    EIGEN_DEVICE_FUNC TensorStorage(Index size, const array<Index, NumIndices_>& dimensions)
        : m_data(internal::conditional_aligned_new_auto<T,(Options_&DontAlign)==0>(size)), m_dimensions(dimensions)
      { EIGEN_INTERNAL_TENSOR_STORAGE_CTOR_PLUGIN }

    EIGEN_DEVICE_FUNC TensorStorage(const Self& other)
      : m_data(internal::conditional_aligned_new_auto<T,(Options_&DontAlign)==0>(internal::array_prod(other.m_dimensions)))
      , m_dimensions(other.m_dimensions)
    {
      internal::smart_copy(other.m_data, other.m_data+internal::array_prod(other.m_dimensions), m_data);
    }
    EIGEN_DEVICE_FUNC Self& operator=(const Self& other)
    {
      if (this != &other) {
        Self tmp(other);
        this->swap(tmp);
      }
      return *this;
    }

    EIGEN_DEVICE_FUNC  ~TensorStorage() { internal::conditional_aligned_delete_auto<T,(Options_&DontAlign)==0>(m_data, internal::array_prod(m_dimensions)); }
    EIGEN_DEVICE_FUNC  void swap(Self& other)
    { numext::swap(m_data,other.m_data); numext::swap(m_dimensions,other.m_dimensions); }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const {return m_dimensions;}

    EIGEN_DEVICE_FUNC void resize(Index size, const array<Index, NumIndices_>& nbDimensions)
    {
      const Index currentSz = internal::array_prod(m_dimensions);
      if(size != currentSz)
      {
        internal::conditional_aligned_delete_auto<T,(Options_&DontAlign)==0>(m_data, currentSz);
        if (size)
          m_data = internal::conditional_aligned_new_auto<T,(Options_&DontAlign)==0>(size);
        else
          m_data = 0;
        EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN
      }
      m_dimensions = nbDimensions;
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T *data() { return m_data; }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const T *data() const { return m_data; }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index size() const { return m_dimensions.TotalSize(); }

 private:
  T *m_data;
  Dimensions m_dimensions;
};


// pure dynamic
template<typename T, int Options_>
class TensorStorage<T, VSizes<DenseIndex>, Options_>
{
    T* m_data;
    VSizes<DenseIndex> m_dimensions;
    typedef TensorStorage<T, VSizes<DenseIndex>, Options_> Self_;

  public:
    EIGEN_DEVICE_FUNC TensorStorage() : m_data(0), m_dimensions() {}

    template <DenseIndex NumDims>
    EIGEN_DEVICE_FUNC TensorStorage(const array<DenseIndex, NumDims>& dimensions)
      {
        m_dimensions.resize(NumDims);
        for (int i = 0; i < NumDims; ++i) {
          m_dimensions[i] = dimensions[i];
        }
        const DenseIndex size = array_prod(dimensions);
        m_data = internal::conditional_managed_new_auto<T,(Options_&DontAlign)==0,(Options_&AllocateUVM)>(size);
        EIGEN_INTERNAL_TENSOR_STORAGE_CTOR_PLUGIN
      }

    EIGEN_DEVICE_FUNC TensorStorage(const std::vector<DenseIndex>& dimensions)
        : m_dimensions(dimensions)
      {
        const DenseIndex size = internal::array_prod(dimensions);
        m_data = internal::conditional_managed_new_auto<T,(Options_&DontAlign)==0,(Options_&AllocateUVM)>(size);
        EIGEN_INTERNAL_TENSOR_STORAGE_CTOR_PLUGIN
      }

#ifdef EIGEN_HAS_VARIADIC_TEMPLATES
    template<typename... IndexTypes> EIGEN_DEVICE_FUNC
    TensorStorage(IndexTypes... dimensions) {
      const int NumDims = sizeof...(dimensions);
      m_dimensions.resize(NumDims);
      const array<DenseIndex, NumDims> dim{{dimensions...}};
      DenseIndex size = 1;
      for (int i = 0; i < NumDims; ++i) {
        size *= dim[i];
        m_dimensions[i] = dim[i];
      }
      m_data = internal::conditional_managed_new_auto<T,(Options_&DontAlign)==0,(Options_&AllocateUVM)>(size);
      EIGEN_INTERNAL_TENSOR_STORAGE_CTOR_PLUGIN
    }
#endif

    EIGEN_DEVICE_FUNC TensorStorage(const Self_& other)
      : m_data(internal::conditional_managed_new_auto<T,(Options_&DontAlign)==0,(Options_&AllocateUVM)>(internal::array_prod(other.m_dimensions)))
      , m_dimensions(other.m_dimensions)
    {
      internal::smart_copy(other.m_data, other.m_data+internal::array_prod(other.m_dimensions), m_data);
    }

    EIGEN_DEVICE_FUNC Self_& operator=(const Self_& other)
    {
      if (this != &other) {
        Self_ tmp(other);
        this->swap(tmp);
      }
      return *this;
    }

    EIGEN_DEVICE_FUNC ~TensorStorage()
    {
      internal::conditional_managed_delete_auto<T,(Options_&DontAlign)==0,(Options_&AllocateUVM)>(m_data, internal::array_prod(m_dimensions));
    }

    EIGEN_DEVICE_FUNC void swap(Self_& other)
    { std::swap(m_data,other.m_data); std::swap(m_dimensions,other.m_dimensions); }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const VSizes<DenseIndex>& dimensions() const { return m_dimensions; }

    template <typename NewDimensions> EIGEN_DEVICE_FUNC
    void resize(DenseIndex size, const NewDimensions& nbDimensions)
    {
      const DenseIndex currentSz = internal::array_prod(m_dimensions);
      if(size != currentSz)
      {
        internal::conditional_managed_delete_auto<T,(Options_&DontAlign)==0,(Options_&AllocateUVM)>(m_data, currentSz);
        if (size)
          m_data = internal::conditional_managed_new_auto<T,(Options_&DontAlign)==0,(Options_&AllocateUVM)>(size);
        else
          m_data = 0;
        EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN
      }
      m_dimensions.resize(internal::array_size<NewDimensions>::value);
      for (int i = 0; i < internal::array_size<NewDimensions>::value; ++i) {
        m_dimensions[i] = nbDimensions[i];
      }
    }
    EIGEN_DEVICE_FUNC void resize(DenseIndex size, const std::vector<DenseIndex>& nbDimensions)
    {
      const DenseIndex currentSz = internal::array_prod(m_dimensions);
      if(size != currentSz)
      {
        internal::conditional_managed_delete_auto<T,(Options_&DontAlign)==0,(Options_&AllocateUVM)>(m_data, currentSz);
        if (size)
          m_data = internal::conditional_managed_new_auto<T,(Options_&DontAlign)==0,(Options_&AllocateUVM)>(size);
        else
          m_data = 0;
        EIGEN_INTERNAL_DENSE_STORAGE_CTOR_PLUGIN
      }
      m_dimensions = nbDimensions;
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T *data() { return m_data; }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const T *data() const { return m_data; }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE DenseIndex size() const { return m_dimensions.TotalSize(); }
};

} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSORSTORAGE_H
