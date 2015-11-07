// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_VAR_DIM_H
#define EIGEN_CXX11_TENSOR_TENSOR_VAR_DIM_H

namespace Eigen {

/** \class Tensor
  * \ingroup CXX11_Tensor_Module
  *
  * \brief A version of the tensor class that supports a variable number of dimensions.
  *
  * The variable equivalent of
  * Eigen::Tensor<float, 3> t(3, 5, 7);
  * is
  * Eigen::TensorVarDim<float> t(3, 5, 7);
  */

template<typename Scalar_, int Options_, typename IndexType_>
class TensorVarDim : public TensorBase<TensorVarDim<Scalar_, Options_, IndexType_> >
{
  public:
    typedef TensorVarDim<Scalar_, Options_, IndexType_> Self;
    typedef TensorBase<TensorVarDim<Scalar_, Options_, IndexType_> > Base;
    typedef typename Eigen::internal::nested<Self>::type Nested;
    typedef typename internal::traits<Self>::StorageKind StorageKind;
    typedef typename internal::traits<Self>::Index Index;
    typedef Scalar_ Scalar;
    typedef typename internal::packet_traits<Scalar>::type Packet;
    typedef typename NumTraits<Scalar>::Real RealScalar;
    typedef typename Base::CoeffReturnType CoeffReturnType;
    typedef typename Base::PacketReturnType PacketReturnType;

    enum {
      IsAligned = bool(EIGEN_ALIGN) & !(Options_ & DontAlign),
      PacketAccess = (internal::packet_traits<Scalar>::size > 1),
      BlockAccess = false,
      Layout = Options_ & RowMajor ? RowMajor : ColMajor,
      // disabled for now as the number of coefficients is not known by the
      // caller at compile time.
      CoordAccess = false,
    };

    static const int Options = Options_;

    static const Index NumIndices = Dynamic;

    typedef VSizes<Index> Dimensions;

  protected:
    TensorStorage<Scalar, VSizes<Index>, Options_> m_storage;

  public:
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index                         rank() const { return m_storage.dimensions().size(); }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index                         dimension(std::size_t n) const { return m_storage.dimensions()[n]; }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions()    const { return m_storage.dimensions(); }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index                         size()                   const { return m_storage.size(); }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar                        *data()                        { return m_storage.data(); }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar                  *data()                  const { return m_storage.data(); }

    // This makes EIGEN_INITIALIZE_COEFFS_IF_THAT_OPTION_IS_ENABLED
    // work, because that uses base().coeffRef() - and we don't yet
    // implement a similar class hierarchy
    inline Self& base()             { return *this; }
    inline const Self& base() const { return *this; }

#ifdef EIGEN_HAS_VARIADIC_TEMPLATES
    template<typename... IndexTypes>
    EIGEN_DEVICE_FUNC inline const Scalar& coeff(Index firstIndex, Index secondIndex, IndexTypes... otherIndices) const
    {
      // The number of indices used to access a tensor coefficient must be equal to the rank of the tensor.
      static const std::size_t NumIndices = sizeof...(otherIndices) + 2;
      return coeff(array<Index, NumIndices>{{firstIndex, secondIndex, otherIndices...}});
    }
#endif

    template <std::size_t NumIndices>
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar& coeff(const array<Index, NumIndices>& indices) const
    {
      eigen_internal_assert(checkIndexRange(indices));
      return m_storage.data()[linearizedIndex(indices)];
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar& coeff(Index index) const
    {
      eigen_internal_assert(index >= 0 && index < size());
      return m_storage.data()[index];
    }

#ifdef EIGEN_HAS_VARIADIC_TEMPLATES
    template<typename... IndexTypes>
    inline Scalar& coeffRef(Index firstIndex, Index secondIndex, IndexTypes... otherIndices)
    {
      static const std::size_t NumIndices = sizeof...(otherIndices) + 2;
      return coeffRef(array<Index, NumIndices>{{firstIndex, secondIndex, otherIndices...}});
    }
#endif

    template <std::size_t NumIndices>
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar& coeffRef(const array<Index, NumIndices>& indices)
    {
      eigen_internal_assert(checkIndexRange(indices));
      return m_storage.data()[linearizedIndex(indices)];
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar& coeffRef(Index index)
    {
      eigen_internal_assert(index >= 0 && index < size());
      return m_storage.data()[index];
    }

#ifdef EIGEN_HAS_VARIADIC_TEMPLATES
    template<typename... IndexTypes>
    inline const Scalar& operator()(Index firstIndex, Index secondIndex, IndexTypes... otherIndices) const
    {
      // The number of indices used to access a tensor coefficient must be equal to the rank of the tensor.
      static const std::size_t NumIndices = sizeof...(otherIndices) + 2;
      return this->operator()(array<Index, NumIndices>{{firstIndex, secondIndex, otherIndices...}});
    }
#endif

    template <std::size_t NumIndices>
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar& operator()(const array<Index, NumIndices>& indices) const
    {
      eigen_assert(checkIndexRange(indices));
      return coeff(indices);
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar& operator()(Index index) const
    {
      eigen_internal_assert(index >= 0 && index < size());
      return coeff(index);
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar& operator[](Index index) const
    {
      return coeff(index);
    }

#ifdef EIGEN_HAS_VARIADIC_TEMPLATES
    template<typename... IndexTypes>
    inline Scalar& operator()(Index firstIndex, IndexTypes... otherIndices)
    {
      // The number of indices used to access a tensor coefficient must be equal to the rank of the tensor.
      static const size_t NumIndices = sizeof...(otherIndices) + 1;
      return operator()(array<Index, NumIndices>{{firstIndex, otherIndices...}});
    }
#endif

    template <std::size_t NumIndices>
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar& operator()(const array<Index, NumIndices>& indices)
    {
      eigen_assert(checkIndexRange(indices));
      return coeffRef(indices);
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar& operator()(Index index)
    {
      eigen_assert(index >= 0 && index < size());
      return coeffRef(index);
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar& operator[](Index index)
    {
      return coeffRef(index);
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE TensorVarDim()
      : m_storage()
    {
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE TensorVarDim(const Self& other)
      : m_storage(other.m_storage)
    {
    }

#ifdef EIGEN_HAS_VARIADIC_TEMPLATES
    template<typename... IndexTypes>
    EIGEN_STRONG_INLINE TensorVarDim(Index firstDimension, IndexTypes... otherDimensions)
        : m_storage(firstDimension, otherDimensions...)
    {
    }
#endif

    EIGEN_STRONG_INLINE explicit TensorVarDim(const std::vector<Index>& dimensions)
        : m_storage(dimensions)
    {
      EIGEN_INITIALIZE_COEFFS_IF_THAT_OPTION_IS_ENABLED
    }

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE TensorVarDim(const TensorBase<OtherDerived, ReadOnlyAccessors>& other)
    {
      typedef TensorAssignOp<TensorVarDim, const OtherDerived> Assign;
      Assign assign(*this, other.derived());
      resize(TensorEvaluator<const Assign, DefaultDevice>(assign, DefaultDevice()).dimensions());
      internal::TensorExecutor<const Assign, DefaultDevice>::run(assign, DefaultDevice());
    }
    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE TensorVarDim(const TensorBase<OtherDerived, WriteAccessors>& other)
    {
      typedef TensorAssignOp<TensorVarDim, const OtherDerived> Assign;
      Assign assign(*this, other.derived());
      resize(TensorEvaluator<const Assign, DefaultDevice>(assign, DefaultDevice()).dimensions());
      internal::TensorExecutor<const Assign, DefaultDevice>::run(assign, DefaultDevice());
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE TensorVarDim& operator=(const TensorVarDim& other)
    {
      typedef TensorAssignOp<TensorVarDim, const TensorVarDim> Assign;
      Assign assign(*this, other);
      resize(TensorEvaluator<const Assign, DefaultDevice>(assign, DefaultDevice()).dimensions());
      internal::TensorExecutor<const Assign, DefaultDevice>::run(assign, DefaultDevice());
      return *this;
    }
    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE TensorVarDim& operator=(const OtherDerived& other)
    {
      typedef TensorAssignOp<TensorVarDim, const OtherDerived> Assign;
      Assign assign(*this, other);
      resize(TensorEvaluator<const Assign, DefaultDevice>(assign, DefaultDevice()).dimensions());
      internal::TensorExecutor<const Assign, DefaultDevice>::run(assign, DefaultDevice());
      return *this;
    }

#ifdef EIGEN_HAS_VARIADIC_TEMPLATES
    template<typename... IndexTypes>
    void resize(Index firstDimension, IndexTypes... otherDimensions)
    {
      // The number of dimensions used to resize a tensor must be equal to the rank of the tensor.
      EIGEN_STATIC_ASSERT(sizeof...(otherDimensions) + 1 == NumIndices, YOU_MADE_A_PROGRAMMING_MISTAKE)
      static const std::size_t NumIndices = sizeof...(otherDimensions) + 1;
      resize(array<Index, NumIndices>{{firstDimension, otherDimensions...}});
    }
#endif

    template <size_t NumIndices>
    void resize(const array<Index, NumIndices>& dimensions)
    {
      Index size = Index(1);
      for (std::size_t i = 0; i < NumIndices; i++) {
        internal::check_rows_cols_for_overflow<Dynamic>::run(size, dimensions[i]);
        size *= dimensions[i];
      }
      #ifdef EIGEN_INITIALIZE_COEFFS
        bool size_changed = size != this->size();
        m_storage.resize(size, dimensions);
        if(size_changed) EIGEN_INITIALIZE_COEFFS_IF_THAT_OPTION_IS_ENABLED
      #else
        m_storage.resize(size, dimensions);
      #endif
    }
    void resize(const std::vector<Index>& dimensions)
    {
      Index size = Index(1);
      for (std::size_t i = 0; i < dimensions.size(); i++) {
        internal::check_rows_cols_for_overflow<Dynamic>::run(size, dimensions[i]);
        size *= dimensions[i];
      }
      #ifdef EIGEN_INITIALIZE_COEFFS
        bool size_changed = size != this->size();
        m_storage.resize(size, dimensions);
        if(size_changed) EIGEN_INITIALIZE_COEFFS_IF_THAT_OPTION_IS_ENABLED
      #else
        m_storage.resize(size, dimensions);
      #endif
    }

  protected:
    template <std::size_t NumIndices>
    bool checkIndexRange(const array<Index, NumIndices>& indices) const
    {
      /*     using internal::array_apply_and_reduce;
      using internal::array_zip_and_reduce;
      using internal::greater_equal_zero_op;
      using internal::logical_and_op;
      using internal::lesser_op;

      return
        // check whether the indices are all >= 0
        array_apply_and_reduce<logical_and_op, greater_equal_zero_op>(indices) &&
        // check whether the indices fit in the dimensions
        array_zip_and_reduce<logical_and_op, lesser_op>(indices, m_storage.dimensions());
      */
      return true;
    }

    template <std::size_t NumIndices>
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index linearizedIndex(const array<Index, NumIndices>& indices) const
    {
      if (Options&RowMajor) {
        return m_storage.dimensions().IndexOfRowMajor(indices);
      } else {
        return m_storage.dimensions().IndexOfColMajor(indices);
      }
    }
};

} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_VAR_DIM_H
