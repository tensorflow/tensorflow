// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_EVALUATOR_H
#define EIGEN_CXX11_TENSOR_TENSOR_EVALUATOR_H

namespace Eigen {

/** \class TensorEvaluator
  * \ingroup CXX11_Tensor_Module
  *
  * \brief The tensor evaluator classes.
  *
  * These classes are responsible for the evaluation of the tensor expression.
  *
  * TODO: add support for more types of expressions, in particular expressions
  * leading to lvalues (slicing, reshaping, etc...)
  */

// Generic evaluator
template<typename Derived, typename Device>
struct TensorEvaluator
{
  typedef typename Derived::Index Index;
  typedef typename Derived::Scalar Scalar;
  typedef typename Derived::Scalar CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, Device>::type PacketReturnType;
  typedef typename Derived::Dimensions Dimensions;

  // NumDimensions is -1 for variable dim tensors
  static const int NumCoords = internal::traits<Derived>::NumDimensions;
  static const int SafeNumCoords = NumCoords >= 0 ? NumCoords : 0;

  enum {
    IsAligned = Derived::IsAligned,
    PacketAccess = Derived::PacketAccess,
    BlockAccess = internal::is_arithmetic<
                      typename internal::remove_const<Scalar>::type>::value &&
                  NumCoords >= 0,
    Layout = Derived::Layout,
    CoordAccess = NumCoords >= 0,
  };

  typedef typename internal::TensorBlock<
      Index, typename internal::remove_const<Scalar>::type, SafeNumCoords, Layout>
      TensorBlock;
  typedef typename internal::TensorBlockReader<
      Index, typename internal::remove_const<Scalar>::type, SafeNumCoords, Layout,
      PacketAccess> TensorBlockReader;
  typedef typename internal::TensorBlockWriter<
      Index, typename internal::remove_const<Scalar>::type, SafeNumCoords, Layout,
      PacketAccess> TensorBlockWriter;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  TensorEvaluator(const Derived& m, const Device& device)
      : m_data(const_cast<Scalar*>(m.data())),
        m_dims(m.dimensions()),
        m_device(device) {}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_dims; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(CoeffReturnType* dest) {
    if (dest) {
      m_device.memcpy((void*)dest, m_data, sizeof(Scalar) * m_dims.TotalSize());
      return false;
    }
    return true;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() { }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const {
    eigen_assert(m_data);
    return m_data[index];
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar& coeffRef(Index index) {
    eigen_assert(m_data);
    return m_data[index];
  }

  template<int LoadMode> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  PacketReturnType packet(Index index) const
  {
    return internal::ploadt<PacketReturnType, LoadMode>(m_data + index);
  }

  template <int StoreMode> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  void writePacket(Index index, const PacketReturnType& x)
  {
    return internal::pstoret<Scalar, PacketReturnType, StoreMode>(m_data + index, x);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(const array<Index, SafeNumCoords>& coords) const {
    eigen_assert(m_data);
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      return m_data[m_dims.IndexOfColMajor(coords)];
    } else {
      return m_data[m_dims.IndexOfRowMajor(coords)];
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar& coeffRef(const array<Index, SafeNumCoords>& coords) {
    eigen_assert(m_data);
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      return m_data[m_dims.IndexOfColMajor(coords)];
    } else {
      return m_data[m_dims.IndexOfRowMajor(coords)];
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void getResourceRequirements(
      std::vector<internal::TensorOpResourceRequirements>* resources) const {
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void block(TensorBlock* block) const {
    assert(m_data != NULL);
    TensorBlockReader::Run(block, m_data);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void writeBlock(
      const TensorBlock& block) {
    assert(m_data != NULL);
    TensorBlockWriter::Run(block, m_data);
  }

  EIGEN_DEVICE_FUNC Scalar* data() const { return m_data; }

 protected:
  Scalar* m_data;
  Dimensions m_dims;
  const Device& m_device;
};


namespace {
template <typename T> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
T loadConstant(const T* address) {
  return *address;

}
// Use the texture cache on CUDA devices whenever possible
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
template <> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
float loadConstant(const float* address) {
  return __ldg(address);
}
template <> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
double loadConstant(const double* address) {
  return __ldg(address);


}
#endif
}


// Default evaluator for rvalues
template<typename Derived, typename Device>
struct TensorEvaluator<const Derived, Device>
{
  typedef typename Derived::Index Index;
  typedef typename Derived::Scalar Scalar;
  typedef typename Derived::Scalar CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, Device>::type PacketReturnType;
  typedef typename Derived::Dimensions Dimensions;

  // NumDimensions is -1 for variable dim tensors
  static const int NumCoords = internal::traits<Derived>::NumDimensions;
  static const int SafeNumCoords = NumCoords >= 0 ? NumCoords : 0;

  enum {
    IsAligned = Derived::IsAligned,
    PacketAccess = Derived::PacketAccess,
    BlockAccess = internal::is_arithmetic<
                      typename internal::remove_const<Scalar>::type>::value &&
                  NumCoords >= 0,
    Layout = Derived::Layout,
    CoordAccess = NumCoords >= 0,
  };

  // TODO(andydavis) Add block/writeBlock accessors to Tensor and TensorMap so
  // we can default BlockAccess to true above.
  typedef typename internal::TensorBlock<
      Index, typename internal::remove_const<Scalar>::type, SafeNumCoords, Layout>
      TensorBlock;
  typedef typename internal::TensorBlockReader<
      Index, typename internal::remove_const<Scalar>::type, SafeNumCoords, Layout,
      PacketAccess> TensorBlockReader;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const Derived& m, const Device& device)
      : m_data(m.data()), m_dims(m.dimensions()), m_device(device)
  { }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_dims; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(CoeffReturnType* data) {
    if (internal::is_arithmetic<typename internal::remove_const<Scalar>::type>::value && data) {
      m_device.memcpy((void*)data, m_data, m_dims.TotalSize() * sizeof(Scalar));
      return false;
    }
    return true;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() { }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const {
    eigen_assert(m_data);
    return loadConstant(m_data+index);
  }

  template<int LoadMode> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  PacketReturnType packet(Index index) const
  {
    return internal::ploadt_ro<PacketReturnType, LoadMode>(m_data + index);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(const array<Index, SafeNumCoords>& coords) const {
    eigen_assert(m_data);
    const Index index = (static_cast<int>(Layout) == static_cast<int>(ColMajor)) ? m_dims.IndexOfColMajor(coords)
                        : m_dims.IndexOfRowMajor(coords);
    return loadConstant(m_data+index);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void getResourceRequirements(
      std::vector<internal::TensorOpResourceRequirements>* resources) const {
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void block(TensorBlock* block) const {
    assert(m_data != NULL);
    TensorBlockReader::Run(block, m_data);
  }

  EIGEN_DEVICE_FUNC const Scalar* data() const { return m_data; }

 protected:
  const Scalar* m_data;
  Dimensions m_dims;
  const Device& m_device;
};




// -------------------- CwiseNullaryOp --------------------

template<typename NullaryOp, typename ArgType, typename Device>
struct TensorEvaluator<const TensorCwiseNullaryOp<NullaryOp, ArgType>, Device>
{
  typedef TensorCwiseNullaryOp<NullaryOp, ArgType> XprType;

  enum {
    IsAligned = true,
    PacketAccess = internal::functor_traits<NullaryOp>::PacketAccess,
    BlockAccess = false,
    Layout = TensorEvaluator<ArgType, Device>::Layout,
    CoordAccess = false,  // to be implemented
  };

  EIGEN_DEVICE_FUNC
  TensorEvaluator(const XprType& op, const Device& device)
      : m_functor(op.functor()), m_argImpl(op.nestedExpression(), device)
  { }

  typedef typename XprType::Index Index;
  typedef typename XprType::Scalar Scalar;
  typedef typename internal::traits<XprType>::Scalar CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, Device>::type PacketReturnType;
  typedef typename TensorEvaluator<ArgType, Device>::Dimensions Dimensions;

  EIGEN_DEVICE_FUNC const Dimensions& dimensions() const { return m_argImpl.dimensions(); }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(CoeffReturnType*) { return true; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() { }

  EIGEN_DEVICE_FUNC CoeffReturnType coeff(Index index) const
  {
    return m_functor(index);
  }

  template<int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType packet(Index index) const
  {
    return m_functor.packetOp(index);
  }

  EIGEN_DEVICE_FUNC CoeffReturnType* data() const { return NULL; }

 private:
  const NullaryOp m_functor;
  TensorEvaluator<ArgType, Device> m_argImpl;
};



// -------------------- CwiseUnaryOp --------------------

template<typename UnaryOp, typename ArgType, typename Device>
struct TensorEvaluator<const TensorCwiseUnaryOp<UnaryOp, ArgType>, Device>
{
  typedef TensorCwiseUnaryOp<UnaryOp, ArgType> XprType;

  enum {
    IsAligned = TensorEvaluator<ArgType, Device>::IsAligned,
    PacketAccess = TensorEvaluator<ArgType, Device>::PacketAccess &
                   internal::functor_traits<UnaryOp>::PacketAccess,
    BlockAccess = false,
    Layout = TensorEvaluator<ArgType, Device>::Layout,
    CoordAccess = false,  // to be implemented
  };

  EIGEN_DEVICE_FUNC TensorEvaluator(const XprType& op, const Device& device)
    : m_functor(op.functor()),
      m_argImpl(op.nestedExpression(), device)
  { }

  typedef typename XprType::Index Index;
  typedef typename XprType::Scalar Scalar;
  typedef typename internal::traits<XprType>::Scalar CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, Device>::type PacketReturnType;
  typedef typename TensorEvaluator<ArgType, Device>::Dimensions Dimensions;

  EIGEN_DEVICE_FUNC const Dimensions& dimensions() const { return m_argImpl.dimensions(); }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(Scalar*) {
    m_argImpl.evalSubExprsIfNeeded(NULL);
    return true;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
    m_argImpl.cleanup();
  }

  EIGEN_DEVICE_FUNC CoeffReturnType coeff(Index index) const
  {
    return m_functor(m_argImpl.coeff(index));
  }

  template<int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType packet(Index index) const
  {
    return m_functor.packetOp(m_argImpl.template packet<LoadMode>(index));
  }

  EIGEN_DEVICE_FUNC CoeffReturnType* data() const { return NULL; }

 private:
  const UnaryOp m_functor;
  TensorEvaluator<ArgType, Device> m_argImpl;
};


// -------------------- CwiseBinaryOp --------------------

template<typename BinaryOp, typename LeftArgType, typename RightArgType, typename Device>
struct TensorEvaluator<const TensorCwiseBinaryOp<BinaryOp, LeftArgType, RightArgType>, Device>
{
  typedef TensorCwiseBinaryOp<BinaryOp, LeftArgType, RightArgType> XprType;

  enum {
    IsAligned = TensorEvaluator<LeftArgType, Device>::IsAligned &
                TensorEvaluator<RightArgType, Device>::IsAligned,
    PacketAccess = TensorEvaluator<LeftArgType, Device>::PacketAccess &
                   TensorEvaluator<RightArgType, Device>::PacketAccess &
                   internal::functor_traits<BinaryOp>::PacketAccess,
    BlockAccess = false,
    Layout = TensorEvaluator<LeftArgType, Device>::Layout,
    CoordAccess = false,  // to be implemented
  };

  EIGEN_DEVICE_FUNC TensorEvaluator(const XprType& op, const Device& device)
    : m_functor(op.functor()),
      m_leftImpl(op.lhsExpression(), device),
      m_rightImpl(op.rhsExpression(), device)
  {
    EIGEN_STATIC_ASSERT((static_cast<int>(TensorEvaluator<LeftArgType, Device>::Layout) == static_cast<int>(TensorEvaluator<RightArgType, Device>::Layout) || internal::traits<XprType>::NumDimensions <= 1), YOU_MADE_A_PROGRAMMING_MISTAKE);
    eigen_assert(dimensions_match(m_leftImpl.dimensions(), m_rightImpl.dimensions()));
  }

  typedef typename XprType::Index Index;
  typedef typename XprType::Scalar Scalar;
  typedef typename internal::traits<XprType>::Scalar CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, Device>::type PacketReturnType;
  typedef typename TensorEvaluator<LeftArgType, Device>::Dimensions Dimensions;

  EIGEN_DEVICE_FUNC const Dimensions& dimensions() const
  {
    // TODO: use right impl instead if right impl dimensions are known at compile time.
    return m_leftImpl.dimensions();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(CoeffReturnType*) {
    m_leftImpl.evalSubExprsIfNeeded(NULL);
    m_rightImpl.evalSubExprsIfNeeded(NULL);
    return true;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
    m_leftImpl.cleanup();
    m_rightImpl.cleanup();
  }

  EIGEN_DEVICE_FUNC CoeffReturnType coeff(Index index) const
  {
    return m_functor(m_leftImpl.coeff(index), m_rightImpl.coeff(index));
  }
  template<int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType packet(Index index) const
  {
    return m_functor.packetOp(m_leftImpl.template packet<LoadMode>(index), m_rightImpl.template packet<LoadMode>(index));
  }

  EIGEN_DEVICE_FUNC CoeffReturnType* data() const { return NULL; }

 private:
  const BinaryOp m_functor;
  TensorEvaluator<LeftArgType, Device> m_leftImpl;
  TensorEvaluator<RightArgType, Device> m_rightImpl;
};


// -------------------- SelectOp --------------------

template<typename IfArgType, typename ThenArgType, typename ElseArgType, typename Device>
struct TensorEvaluator<const TensorSelectOp<IfArgType, ThenArgType, ElseArgType>, Device>
{
  typedef TensorSelectOp<IfArgType, ThenArgType, ElseArgType> XprType;
  typedef typename XprType::Scalar Scalar;

  enum {
    IsAligned = TensorEvaluator<ThenArgType, Device>::IsAligned &
                TensorEvaluator<ElseArgType, Device>::IsAligned,
    PacketAccess = TensorEvaluator<ThenArgType, Device>::PacketAccess &
                   TensorEvaluator<ElseArgType, Device>::PacketAccess &
                   internal::packet_traits<Scalar>::HasBlend,
    BlockAccess = false,
    Layout = TensorEvaluator<IfArgType, Device>::Layout,
    CoordAccess = false,  // to be implemented
  };

  EIGEN_DEVICE_FUNC TensorEvaluator(const XprType& op, const Device& device)
    : m_condImpl(op.ifExpression(), device),
      m_thenImpl(op.thenExpression(), device),
      m_elseImpl(op.elseExpression(), device)
  {
    EIGEN_STATIC_ASSERT((static_cast<int>(TensorEvaluator<IfArgType, Device>::Layout) == static_cast<int>(TensorEvaluator<ThenArgType, Device>::Layout)), YOU_MADE_A_PROGRAMMING_MISTAKE);
    EIGEN_STATIC_ASSERT((static_cast<int>(TensorEvaluator<IfArgType, Device>::Layout) == static_cast<int>(TensorEvaluator<ElseArgType, Device>::Layout)), YOU_MADE_A_PROGRAMMING_MISTAKE);
    eigen_assert(dimensions_match(m_condImpl.dimensions(), m_thenImpl.dimensions()));
    eigen_assert(dimensions_match(m_thenImpl.dimensions(), m_elseImpl.dimensions()));
  }

  typedef typename XprType::Index Index;
  typedef typename internal::traits<XprType>::Scalar CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, Device>::type PacketReturnType;
  typedef typename TensorEvaluator<IfArgType, Device>::Dimensions Dimensions;

  EIGEN_DEVICE_FUNC const Dimensions& dimensions() const
  {
    // TODO: use then or else impl instead if they happen to be known at compile time.
    return m_condImpl.dimensions();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(CoeffReturnType*) {
    m_condImpl.evalSubExprsIfNeeded(NULL);
    m_thenImpl.evalSubExprsIfNeeded(NULL);
    m_elseImpl.evalSubExprsIfNeeded(NULL);
    return true;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
    m_condImpl.cleanup();
    m_thenImpl.cleanup();
    m_elseImpl.cleanup();
  }

  EIGEN_DEVICE_FUNC CoeffReturnType coeff(Index index) const
  {
    return m_condImpl.coeff(index) ? m_thenImpl.coeff(index) : m_elseImpl.coeff(index);
  }
  template<int LoadMode>
  EIGEN_DEVICE_FUNC PacketReturnType packet(Index index) const
  {
    const int PacketSize = internal::unpacket_traits<PacketReturnType>::size;
    internal::Selector<PacketSize> select;
    for (Index i = 0; i < PacketSize; ++i) {
      select.select[i] = m_condImpl.coeff(index+i);
    }
    return internal::pblend(select,
                            m_thenImpl.template packet<LoadMode>(index),
                            m_elseImpl.template packet<LoadMode>(index));
  }

  EIGEN_DEVICE_FUNC CoeffReturnType* data() const { return NULL; }

 private:
  TensorEvaluator<IfArgType, Device> m_condImpl;
  TensorEvaluator<ThenArgType, Device> m_thenImpl;
  TensorEvaluator<ElseArgType, Device> m_elseImpl;
};

} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_EVALUATOR_H
