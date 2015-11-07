// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_GENERATOR_H
#define EIGEN_CXX11_TENSOR_TENSOR_GENERATOR_H

namespace Eigen {

/** \class TensorGenerator
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Tensor generator class.
  *
  *
  */
namespace internal {
template<typename Generator, typename XprType>
struct traits<TensorGeneratorOp<Generator, XprType> > : public traits<XprType>
{
  typedef typename XprType::Scalar Scalar;
  typedef traits<XprType> XprTraits;
  typedef typename packet_traits<Scalar>::type Packet;
  typedef typename XprTraits::StorageKind StorageKind;
  typedef typename XprTraits::Index Index;
  typedef typename XprType::Nested Nested;
  typedef typename remove_reference<Nested>::type _Nested;
  static const int NumDimensions = XprTraits::NumDimensions;
  static const int Layout = XprTraits::Layout;
};

template<typename Generator, typename XprType>
struct eval<TensorGeneratorOp<Generator, XprType>, Eigen::Dense>
{
  typedef const TensorGeneratorOp<Generator, XprType>& type;
};

template<typename Generator, typename XprType>
struct nested<TensorGeneratorOp<Generator, XprType>, 1, typename eval<TensorGeneratorOp<Generator, XprType> >::type>
{
  typedef TensorGeneratorOp<Generator, XprType> type;
};

}  // end namespace internal



template<typename Generator, typename XprType>
class TensorGeneratorOp : public TensorBase<TensorGeneratorOp<Generator, XprType>, ReadOnlyAccessors>
{
  public:
  typedef typename Eigen::internal::traits<TensorGeneratorOp>::Scalar Scalar;
  typedef typename Eigen::internal::traits<TensorGeneratorOp>::Packet Packet;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;
  typedef typename Eigen::internal::nested<TensorGeneratorOp>::type Nested;
  typedef typename Eigen::internal::traits<TensorGeneratorOp>::StorageKind StorageKind;
  typedef typename Eigen::internal::traits<TensorGeneratorOp>::Index Index;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorGeneratorOp(const XprType& expr, const Generator& generator)
      : m_xpr(expr), m_generator(generator) {}

    EIGEN_DEVICE_FUNC
    const Generator& generator() const { return m_generator; }

    EIGEN_DEVICE_FUNC
    const typename internal::remove_all<typename XprType::Nested>::type&
    expression() const { return m_xpr; }

  protected:
    typename XprType::Nested m_xpr;
    const Generator m_generator;
};


// Eval as rvalue
template<typename Generator, typename ArgType, typename Device>
struct TensorEvaluator<const TensorGeneratorOp<Generator, ArgType>, Device>
{
  typedef TensorGeneratorOp<Generator, ArgType> XprType;
  typedef typename XprType::Index Index;
  typedef typename TensorEvaluator<ArgType, Device>::Dimensions Dimensions;
  static const int NumDims = internal::array_size<Dimensions>::value;
  typedef typename XprType::Scalar Scalar;

  enum {
    IsAligned = false,
    PacketAccess = (internal::packet_traits<Scalar>::size > 1),
    BlockAccess = false,
    Layout = TensorEvaluator<ArgType, Device>::Layout,
    CoordAccess = false,  // to be implemented
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device)
      : m_generator(op.generator())
  {
    TensorEvaluator<ArgType, Device> impl(op.expression(), device);
    m_dimensions = impl.dimensions();

    if (NumDims > 0) {
      if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
        m_strides[0] = 1;
        for (int i = 1; i < NumDims; ++i) {
          m_strides[i] = m_strides[i - 1] * m_dimensions[i - 1];
        }
      } else {
        m_strides[NumDims - 1] = 1;
        for (int i = NumDims - 2; i >= 0; --i) {
          m_strides[i] = m_strides[i + 1] * m_dimensions[i + 1];
        }
      }
    }
  }

  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_dimensions; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(Scalar* /*data*/) {
    return true;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const
  {
    array<Index, NumDims> coords;
    extract_coordinates(index, coords);
    return m_generator(coords);
  }

  template<int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType packet(Index index) const
  {
    const int packetSize = internal::unpacket_traits<PacketReturnType>::size;
    EIGEN_STATIC_ASSERT(packetSize > 1, YOU_MADE_A_PROGRAMMING_MISTAKE)
    eigen_assert(index+packetSize-1 < dimensions().TotalSize());

    EIGEN_ALIGN_DEFAULT typename internal::remove_const<CoeffReturnType>::type values[packetSize];
    for (int i = 0; i < packetSize; ++i) {
      values[i] = coeff(index+i);
    }
    PacketReturnType rslt = internal::pload<PacketReturnType>(values);
    return rslt;
  }

  EIGEN_DEVICE_FUNC Scalar* data() const { return NULL; }

 protected:
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  void extract_coordinates(Index index, array<Index, NumDims>& coords) const {
    if (NumDims > 0) {
      if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
        for (int i = NumDims - 1; i > 0; --i) {
          const Index idx = index / m_strides[i];
          index -= idx * m_strides[i];
          coords[i] = idx;
        }
        coords[0] = index;
      } else {
        for (int i = 0; i < NumDims - 1; ++i) {
          const Index idx = index / m_strides[i];
          index -= idx * m_strides[i];
          coords[i] = idx;
        }
        coords[NumDims-1] = index;
      }
    }
  }

  Dimensions m_dimensions;
  array<Index, NumDims> m_strides;
  Generator m_generator;
};

} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_GENERATOR_H
