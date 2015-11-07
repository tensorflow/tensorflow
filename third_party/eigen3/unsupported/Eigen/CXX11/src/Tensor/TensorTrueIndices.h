// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Eugene Brevdo <ebrevdo@google.com>
//                    Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_TRUE_INDICES_H
#define EIGEN_CXX11_TENSOR_TENSOR_TRUE_INDICES_H
namespace Eigen {

/** \class TensorTrueIndices
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Tensor provide indices of true values class.
  *
  */
namespace internal {
template<typename XprType>
struct traits<TensorTrueIndicesOp<XprType> > : public traits<XprType>
{
  typedef DenseIndex Scalar;
  typedef DenseIndex CoeffReturnType;
  typedef traits<XprType> XprTraits;
  //typedef typename packet_traits<Scalar>::type Packet;
  typedef typename XprTraits::StorageKind StorageKind;
  typedef typename XprTraits::Index Index;
  typedef typename XprType::Nested Nested;
  typedef typename remove_reference<Nested>::type _Nested;
  static const int NumDimensions = 2; // XprTraits::NumDimensions;
  static const int Layout = XprTraits::Layout;
};

template<typename XprType>
struct eval<TensorTrueIndicesOp<XprType>, Eigen::Dense>
{
  typedef const TensorTrueIndicesOp<XprType>& type;
};

template<typename XprType>
struct nested<TensorTrueIndicesOp<XprType>, 1,
            typename eval<TensorTrueIndicesOp<XprType> >::type>
{
  typedef TensorTrueIndicesOp<XprType> type;
};

}  // end namespace internal

template<typename XprType>
class TensorTrueIndicesOp : public TensorBase<TensorTrueIndicesOp<XprType>, WriteAccessors>
{
  public:
    typedef typename Eigen::internal::traits<TensorTrueIndicesOp>::Scalar Scalar;
    //typedef typename Eigen::internal::traits<TensorTrueIndicesOp>::Packet Packet;
    typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
    typedef typename Eigen::internal::traits<TensorTrueIndicesOp>::CoeffReturnType CoeffReturnType;
    typedef typename internal::packet_traits<CoeffReturnType>::type PacketReturnType;
    typedef typename Eigen::internal::nested<TensorTrueIndicesOp>::type Nested;
    typedef typename Eigen::internal::traits<TensorTrueIndicesOp>::StorageKind
                                                                    StorageKind;
    typedef typename Eigen::internal::traits<TensorTrueIndicesOp>::Index Index;

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorTrueIndicesOp(
        const XprType& expr, const CoeffReturnType& not_found = -1)
        : m_xpr(expr), m_not_found(not_found) {
    }

    EIGEN_DEVICE_FUNC
    const CoeffReturnType& not_found() const { return m_not_found; }

    EIGEN_DEVICE_FUNC
    const typename internal::remove_all<typename XprType::Nested>::type&
    expression() const { return m_xpr; }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE TensorTrueIndicesOp& operator = (const TensorTrueIndicesOp& other)
    {
      typedef TensorAssignOp<TensorTrueIndicesOp, const TensorTrueIndicesOp> Assign;
      Assign assign(*this, other);
      internal::TensorExecutor<const Assign, DefaultDevice>::run(
          assign, DefaultDevice());
      return *this;
    }

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE TensorTrueIndicesOp& operator = (const OtherDerived& other)
    {
      typedef TensorAssignOp<TensorTrueIndicesOp, const OtherDerived> Assign;
      Assign assign(*this, other);
      internal::TensorExecutor<const Assign, DefaultDevice>::run(
          assign, DefaultDevice());
      return *this;
    }

  protected:
    typename XprType::Nested m_xpr;
    CoeffReturnType m_not_found;
};

// Eval as rvalue
template<typename ArgType, typename Device>
struct TensorEvaluator<const TensorTrueIndicesOp<ArgType>, Device>
{
  typedef TensorTrueIndicesOp<ArgType> XprType;
  typedef typename XprType::Index InputIndex;
  typedef typename XprType::Index Index;
  static const int NumDims = 2;
  typedef DSizes<Index, 2> Dimensions;
  typedef typename TensorEvaluator<ArgType, Device>::Dimensions InputDimensions;
  static const int NumInputDims = internal::array_size<InputDimensions>::value;

  enum {
    IsAligned = true,
    PacketAccess = TensorEvaluator<ArgType, Device>::PacketAccess,
    BlockAccess = false,
    Layout = TensorEvaluator<ArgType, Device>::Layout,
    CoordAccess = false,  // to be implemented
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op,
                                                        const Device& device)
      : m_impl(op.expression(), device), m_not_found(op.not_found())
  {
    // Store original dimensions
    m_orig_dimensions = m_impl.dimensions();

    // Calculate output dimensions
    m_dimensions[0] = m_orig_dimensions.TotalSize();
    m_dimensions[1] = NumInputDims;

    // Calculate strides of input expression
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      m_strides[0] = 1;
      for (int i = 1; i < NumInputDims; ++i) {
        m_strides[i] = m_strides[i-1] * m_orig_dimensions[i-1];
      }
    } else {
      m_strides[NumInputDims-1] = 1;
      for (int i = NumInputDims - 2; i >= 0; --i) {
        m_strides[i] = m_strides[i+1] * m_orig_dimensions[i+1];
      }
    }
  }

  typedef typename XprType::Scalar Scalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  const Dimensions& dimensions() const { return m_dimensions; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(Scalar*) {
    m_impl.evalSubExprsIfNeeded(NULL);
    return true;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
    m_impl.cleanup();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE InputIndex origIndices(
      Index index) const {
    eigen_assert(index < dimensions().TotalSize());
    Index inputIndex = 0;
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      inputIndex = index % m_dimensions[0];
    } else {
      inputIndex = index / m_dimensions[1];
    }
    return inputIndex;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int whichDim(
      Index index) const {
    eigen_assert(index < dimensions().TotalSize());
    int inputDim = 0;
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      inputDim = index / m_dimensions[0];
    } else {
      inputDim = index % m_dimensions[1];
    }
    return inputDim;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType origDim(
      int dim, InputIndex index) const {
    eigen_assert(index < m_orig_dimensions.TotalSize());
    eigen_assert(dim > -1 && dim < m_orig_dimensions.size());
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      for (int i = NumInputDims - 1; i > 0; --i) {
        Index idx = index / m_strides[i];
        if (i == dim) return idx;  // Found our dimension
        index -= idx * m_strides[i];
      }
      return index;
    } else {
      for (int i = 0; i < NumInputDims - 1; ++i) {
        Index idx = index / m_strides[i];
        if (i == dim) return idx;  // Found our dimension
        index -= idx * m_strides[i];
      }
      return index;
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(
      Index index) const  {
    InputIndex orig_index = origIndices(index);
    if (m_impl.coeff(orig_index))
      return origDim(whichDim(index), orig_index);
    else {
      return m_not_found;
    }
  }

  template<int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  PacketReturnType packet(Index index) const
  {
    const int packetSize = internal::unpacket_traits<PacketReturnType>::size;
    EIGEN_STATIC_ASSERT(packetSize > 1, YOU_MADE_A_PROGRAMMING_MISTAKE)
    eigen_assert(index+packetSize-1 < dimensions().TotalSize());

    // TODO(ndjaitly): write a better packing routine that uses
    // local structure.
    EIGEN_ALIGN_DEFAULT typename internal::remove_const<CoeffReturnType>::type
                                                            values[packetSize];
    for (int i = 0; i < packetSize; ++i) {
      values[i] = coeff(index+i);
    }
    PacketReturnType rslt = internal::pload<PacketReturnType>(values);
    return rslt;
  }

  EIGEN_DEVICE_FUNC Scalar* data() const { return NULL; }

 protected:
  InputDimensions m_orig_dimensions;
  Dimensions m_dimensions;
  TensorEvaluator<ArgType, Device> m_impl;
  array<Index, NumInputDims> m_strides;
  CoeffReturnType m_not_found;
};

}  // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_TRUE_INDICES_H
