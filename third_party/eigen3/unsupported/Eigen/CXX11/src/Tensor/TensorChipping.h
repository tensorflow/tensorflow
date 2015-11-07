// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_CHIPPING_H
#define EIGEN_CXX11_TENSOR_TENSOR_CHIPPING_H

namespace Eigen {

/** \class TensorKChippingReshaping
  * \ingroup CXX11_Tensor_Module
  *
  * \brief A chip is a thin slice, corresponding to a column or a row in a 2-d tensor.
  *
  *
  */

namespace internal {
template<DenseIndex DimId, typename XprType>
struct traits<TensorChippingOp<DimId, XprType> > : public traits<XprType>
{
  typedef typename XprType::Scalar Scalar;
  typedef traits<XprType> XprTraits;
  typedef typename XprTraits::StorageKind StorageKind;
  typedef typename XprTraits::Index Index;
  typedef typename XprType::Nested Nested;
  typedef typename remove_reference<Nested>::type _Nested;
  static const int NumDimensions = XprTraits::NumDimensions - 1;
  static const int Layout = XprTraits::Layout;
};

template<DenseIndex DimId, typename XprType>
struct eval<TensorChippingOp<DimId, XprType>, Eigen::Dense>
{
  typedef const TensorChippingOp<DimId, XprType>& type;
};

template<DenseIndex DimId, typename XprType>
struct nested<TensorChippingOp<DimId, XprType>, 1, typename eval<TensorChippingOp<DimId, XprType> >::type>
{
  typedef TensorChippingOp<DimId, XprType> type;
};

template <DenseIndex DimId>
struct DimensionId
{
  DimensionId(DenseIndex dim) {
    eigen_assert(dim == DimId);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE DenseIndex actualDim() const {
    return DimId;
  }
};
template <>
struct DimensionId<Dynamic>
{
  DimensionId(DenseIndex dim) : actual_dim(dim) {
    eigen_assert(dim >= 0);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE DenseIndex actualDim() const {
    return actual_dim;
  }
 private:
  const DenseIndex actual_dim;
};


}  // end namespace internal



template<DenseIndex DimId, typename XprType>
class TensorChippingOp : public TensorBase<TensorChippingOp<DimId, XprType> >
{
  public:
  typedef typename Eigen::internal::traits<TensorChippingOp>::Scalar Scalar;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename Eigen::internal::nested<TensorChippingOp>::type Nested;
  typedef typename Eigen::internal::traits<TensorChippingOp>::StorageKind StorageKind;
  typedef typename Eigen::internal::traits<TensorChippingOp>::Index Index;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorChippingOp(const XprType& expr, const Index offset, const Index dim)
      : m_xpr(expr), m_offset(offset), m_dim(dim) {
  }

  EIGEN_DEVICE_FUNC
  const Index offset() const { return m_offset; }
  EIGEN_DEVICE_FUNC
  const Index dim() const { return m_dim.actualDim(); }

  EIGEN_DEVICE_FUNC
  const typename internal::remove_all<typename XprType::Nested>::type&
  expression() const { return m_xpr; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE TensorChippingOp& operator = (const TensorChippingOp& other)
  {
    typedef TensorAssignOp<TensorChippingOp, const TensorChippingOp> Assign;
    Assign assign(*this, other);
    internal::TensorExecutor<const Assign, DefaultDevice>::run(assign, DefaultDevice());
    return *this;
  }

  template<typename OtherDerived>
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE TensorChippingOp& operator = (const OtherDerived& other)
  {
    typedef TensorAssignOp<TensorChippingOp, const OtherDerived> Assign;
    Assign assign(*this, other);
    internal::TensorExecutor<const Assign, DefaultDevice>::run(assign, DefaultDevice());
    return *this;
  }

  protected:
    typename XprType::Nested m_xpr;
    const Index m_offset;
    const internal::DimensionId<DimId> m_dim;
};


// Eval as rvalue
template<DenseIndex DimId, typename ArgType, typename Device>
struct TensorEvaluator<const TensorChippingOp<DimId, ArgType>, Device>
{
  typedef TensorChippingOp<DimId, ArgType> XprType;
  static const int NumInputDims = internal::array_size<typename TensorEvaluator<ArgType, Device>::Dimensions>::value;
  static const int NumDims = NumInputDims-1;
  typedef typename XprType::Index Index;
  typedef DSizes<Index, NumDims> Dimensions;
  typedef typename XprType::Scalar Scalar;
  typedef typename internal::remove_const<Scalar>::type ScalarNonConst;

  enum {
    // Alignment can't be guaranteed at compile time since it depends on the
    // slice offsets.
    IsAligned = false,
    PacketAccess = TensorEvaluator<ArgType, Device>::PacketAccess,
    BlockAccess = TensorEvaluator<ArgType, Device>::BlockAccess,
    Layout = TensorEvaluator<ArgType, Device>::Layout,
    CoordAccess = false,  // to be implemented
  };

  typedef internal::TensorBlock<Index, ScalarNonConst, NumInputDims, Layout>
    InputTensorBlock;
  typedef internal::TensorBlock<Index, ScalarNonConst, NumDims, Layout>
    OutputTensorBlock;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device)
      : m_impl(op.expression(), device), m_dim(op.dim()), m_device(device)
  {
    EIGEN_STATIC_ASSERT(NumInputDims >= 1, YOU_MADE_A_PROGRAMMING_MISTAKE);
    eigen_assert(NumInputDims > m_dim.actualDim());
    const typename TensorEvaluator<ArgType, Device>::Dimensions& input_dims = m_impl.dimensions();
    eigen_assert(op.offset() < input_dims[m_dim.actualDim()]);

    int j = 0;
    for (int i = 0; i < NumInputDims; ++i) {
      if (i != m_dim.actualDim()) {
        m_dimensions[j] = input_dims[i];
        ++j;
      }
    }

    m_stride = 1;
    m_inputStride = 1;
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      for (int i = 0; i < m_dim.actualDim(); ++i) {
        m_stride *= input_dims[i];
        m_inputStride *= input_dims[i];
      }
    } else {
      for (int i = NumInputDims-1; i > m_dim.actualDim(); --i) {
        m_stride *= input_dims[i];
        m_inputStride *= input_dims[i];
      }
    }
    m_inputStride *= input_dims[m_dim.actualDim()];
    m_inputOffset = m_stride * op.offset();

    if (BlockAccess) {
      if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
        m_inputStrides[0] = 1;
        for (int i = 1; i < NumInputDims; ++i) {
          m_inputStrides[i] = m_inputStrides[i - 1] * input_dims[i - 1];
        }
      } else {
        m_inputStrides[NumInputDims - 1] = 1;
        for (int i = NumInputDims - 2; i >= 0; --i) {
          m_inputStrides[i] = m_inputStrides[i + 1] * input_dims[i + 1];
        }
      }

      m_block_total_size_max = numext::maxi(static_cast<std::size_t>(1),
                                            device.lastLevelCacheSize() /
                                            sizeof(Scalar));
    }
  }

  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, Device>::type PacketReturnType;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_dimensions; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(Scalar* /*data*/) {
    m_impl.evalSubExprsIfNeeded(NULL);
    return true;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
    m_impl.cleanup();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const
  {
    return m_impl.coeff(srcCoeff(index));
  }

  template<int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType packet(Index index) const
  {
    const int packetSize = internal::unpacket_traits<PacketReturnType>::size;
    EIGEN_STATIC_ASSERT(packetSize > 1, YOU_MADE_A_PROGRAMMING_MISTAKE)
    eigen_assert(index+packetSize-1 < dimensions().TotalSize());

    if ((static_cast<int>(Layout) == static_cast<int>(ColMajor) &&
         m_dim.actualDim() == 0) ||
        (static_cast<int>(Layout) == static_cast<int>(RowMajor) &&
         m_dim.actualDim() == NumInputDims - 1)) {
      // m_stride is equal to 1, so let's avoid the integer division.
      eigen_assert(m_stride == 1);
      Index inputIndex = index * m_inputStride + m_inputOffset;
      EIGEN_ALIGN_DEFAULT typename internal::remove_const<CoeffReturnType>::type values[packetSize];
      for (int i = 0; i < packetSize; ++i) {
        values[i] = m_impl.coeff(inputIndex);
        inputIndex += m_inputStride;
      }
      PacketReturnType rslt = internal::pload<PacketReturnType>(values);
      return rslt;
    } else if ((static_cast<int>(Layout) == static_cast<int>(ColMajor) &&
                m_dim.actualDim() == NumInputDims - 1) ||
               (static_cast<int>(Layout) == static_cast<int>(RowMajor) &&
                m_dim.actualDim() == 0)) {
      // m_stride is aways greater than index, so let's avoid the integer division.
      eigen_assert(m_stride > index);
      return m_impl.template packet<LoadMode>(index + m_inputOffset);
    } else {
      const Index idx = index / m_stride;
      const Index rem = index - idx * m_stride;
      if (rem + packetSize <= m_stride) {
        Index inputIndex = idx * m_inputStride + m_inputOffset + rem;
        return m_impl.template packet<LoadMode>(inputIndex);
      } else {
        // Cross the stride boundary. Fallback to slow path.
        EIGEN_ALIGN_DEFAULT typename internal::remove_const<CoeffReturnType>::type values[packetSize];
        for (int i = 0; i < packetSize; ++i) {
          values[i] = coeff(index);
          ++index;
        }
        PacketReturnType rslt = internal::pload<PacketReturnType>(values);
        return rslt;
      }
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void getResourceRequirements(
      std::vector<internal::TensorOpResourceRequirements>* resources) const {
    resources->push_back(internal::TensorOpResourceRequirements(
        internal::kSkewedInnerDims, m_block_total_size_max));
    m_impl.getResourceRequirements(resources);
  }

  // TODO(andydavis) Reduce the overhead of this function (experiment with
  // using a fixed block size).
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void block(
      OutputTensorBlock* output_block) const {
    // Calculate input block sizes.
    const DSizes<Index, NumDims>& output_block_sizes =
        output_block->block_sizes();
    const DSizes<Index, NumDims>& output_block_strides =
        output_block->block_strides();
    const Index chip_dim = m_dim.actualDim();
    DSizes<Index, NumInputDims> input_block_sizes;
    DSizes<Index, NumInputDims> input_block_strides;
    for (Index i = 0; i < NumInputDims; ++i) {
      if (i < chip_dim) {
        input_block_sizes[i] = output_block_sizes[i];
        input_block_strides[i] = output_block_strides[i];
      } else if (i > chip_dim) {
        input_block_sizes[i] = output_block_sizes[i - 1];
        input_block_strides[i] = output_block_strides[i - 1];
      } else {
        input_block_sizes[i] = 1;
      }
    }
    // Fix up input_block_stride for chip dimension.
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      if (chip_dim == 0) {
        input_block_strides[chip_dim] = 1;
      } else {
        input_block_strides[chip_dim] = input_block_strides[chip_dim - 1] *
            input_block_sizes[chip_dim - 1];
      }
    } else {
      if (chip_dim == NumInputDims - 1) {
        input_block_strides[chip_dim] = 1;
      } else {
        input_block_strides[chip_dim] = input_block_strides[chip_dim + 1] *
            input_block_sizes[chip_dim + 1];
      }
    }
    // Instantiate and read input block from input tensor.
    InputTensorBlock input_block(srcCoeff(output_block->first_coeff_index()),
                                 input_block_sizes,
                                 input_block_strides,
                                 m_inputStrides,
                                 output_block->data());
    m_impl.block(&input_block);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType* data() const {
    CoeffReturnType* result = const_cast<CoeffReturnType*>(m_impl.data());
    if (((static_cast<int>(Layout) == static_cast<int>(ColMajor) &&
          m_dim.actualDim() == NumDims) ||
         (static_cast<int>(Layout) == static_cast<int>(RowMajor) &&
          m_dim.actualDim() == 0)) &&
        result) {
      return result + m_inputOffset;
    } else {
      return NULL;
    }
  }

 protected:
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index srcCoeff(Index index) const
  {
    Index inputIndex;
    if ((static_cast<int>(Layout) == static_cast<int>(ColMajor) &&
         m_dim.actualDim() == 0) ||
        (static_cast<int>(Layout) == static_cast<int>(RowMajor) &&
         m_dim.actualDim() == NumInputDims - 1)) {
      // m_stride is equal to 1, so let's avoid the integer division.
      eigen_assert(m_stride == 1);
      inputIndex = index * m_inputStride + m_inputOffset;
    } else if ((static_cast<int>(Layout) == static_cast<int>(ColMajor) &&
                m_dim.actualDim() == NumInputDims - 1) ||
               (static_cast<int>(Layout) == static_cast<int>(RowMajor) &&
                m_dim.actualDim() == 0)) {
      // m_stride is aways greater than index, so let's avoid the integer division.
      eigen_assert(m_stride > index);
      inputIndex = index + m_inputOffset;
    } else {
      const Index idx = index / m_stride;
      inputIndex = idx * m_inputStride + m_inputOffset;
      index -= idx * m_stride;
      inputIndex += index;
    }
    return inputIndex;
  }

  Dimensions m_dimensions;
  Index m_stride;
  Index m_inputOffset;
  Index m_inputStride;
  DSizes<Index, NumInputDims> m_inputStrides;
  TensorEvaluator<ArgType, Device> m_impl;
  const internal::DimensionId<DimId> m_dim;
  const Device& m_device;
  std::size_t m_block_total_size_max;
};


// Eval as lvalue
template<DenseIndex DimId, typename ArgType, typename Device>
struct TensorEvaluator<TensorChippingOp<DimId, ArgType>, Device>
  : public TensorEvaluator<const TensorChippingOp<DimId, ArgType>, Device>
{
  typedef TensorEvaluator<const TensorChippingOp<DimId, ArgType>, Device> Base;
  typedef TensorChippingOp<DimId, ArgType> XprType;
  static const int NumInputDims = internal::array_size<typename TensorEvaluator<ArgType, Device>::Dimensions>::value;
  static const int NumDims = NumInputDims-1;
  typedef typename XprType::Index Index;
  typedef DSizes<Index, NumDims> Dimensions;
  typedef typename XprType::Scalar Scalar;

  enum {
    IsAligned = false,
    PacketAccess = TensorEvaluator<ArgType, Device>::PacketAccess,
    BlockAccess = TensorEvaluator<ArgType, Device>::BlockAccess,
    Layout = TensorEvaluator<ArgType, Device>::Layout,
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device)
    : Base(op, device)
    { }

  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, Device>::type PacketReturnType;
  typedef typename internal::remove_const<Scalar>::type ScalarNonConst;
  typedef internal::TensorBlock<Index, ScalarNonConst, NumInputDims, Layout>
    InputTensorBlock;
  typedef internal::TensorBlock<Index, ScalarNonConst, NumDims, Layout>
    OutputTensorBlock;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType& coeffRef(Index index)
  {
    return this->m_impl.coeffRef(this->srcCoeff(index));
  }

  template <int StoreMode> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  void writePacket(Index index, const PacketReturnType& x)
  {
    static const int packetSize = internal::unpacket_traits<PacketReturnType>::size;
    EIGEN_STATIC_ASSERT(packetSize > 1, YOU_MADE_A_PROGRAMMING_MISTAKE)

    if ((static_cast<int>(this->Layout) == static_cast<int>(ColMajor) &&
         this->m_dim.actualDim() == 0) ||
        (static_cast<int>(this->Layout) == static_cast<int>(RowMajor) &&
         this->m_dim.actualDim() == NumInputDims - 1)) {
      // m_stride is equal to 1, so let's avoid the integer division.
      eigen_assert(this->m_stride == 1);
      EIGEN_ALIGN_DEFAULT typename internal::remove_const<CoeffReturnType>::type values[packetSize];
      internal::pstore<CoeffReturnType, PacketReturnType>(values, x);
      Index inputIndex = index * this->m_inputStride + this->m_inputOffset;
      for (int i = 0; i < packetSize; ++i) {
        this->m_impl.coeffRef(inputIndex) = values[i];
        inputIndex += this->m_inputStride;
      }
    } else if ((static_cast<int>(this->Layout) == static_cast<int>(ColMajor) &&
                this->m_dim.actualDim() == NumInputDims - 1) ||
               (static_cast<int>(this->Layout) == static_cast<int>(RowMajor) &&
                this->m_dim.actualDim() == 0)) {
      // m_stride is aways greater than index, so let's avoid the integer division.
      eigen_assert(this->m_stride > index);
      this->m_impl.template writePacket<StoreMode>(index + this->m_inputOffset, x);
    } else {
      const Index idx = index / this->m_stride;
      const Index rem = index - idx * this->m_stride;
      if (rem + packetSize <= this->m_stride) {
        const Index inputIndex = idx * this->m_inputStride + this->m_inputOffset + rem;
        this->m_impl.template writePacket<StoreMode>(inputIndex, x);
      } else {
        // Cross stride boundary. Fallback to slow path.
        EIGEN_ALIGN_DEFAULT typename internal::remove_const<CoeffReturnType>::type values[packetSize];
        internal::pstore<CoeffReturnType, PacketReturnType>(values, x);
        for (int i = 0; i < packetSize; ++i) {
          this->coeffRef(index) = values[i];
          ++index;
        }
      }
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void writeBlock(
      const OutputTensorBlock& output_block) {
    // Calculate input block sizes.
    const DSizes<Index, NumDims>& output_block_sizes =
        output_block.block_sizes();
    const DSizes<Index, NumDims>& output_block_strides =
        output_block.block_strides();
    const Index chip_dim = this->m_dim.actualDim();
    DSizes<Index, NumInputDims> input_block_sizes;
    DSizes<Index, NumInputDims> input_block_strides;
    for (Index i = 0; i < NumInputDims; ++i) {
      if (i < chip_dim) {
        input_block_sizes[i] = output_block_sizes[i];
        input_block_strides[i] = output_block_strides[i];
      } else if (i > chip_dim) {
        input_block_sizes[i] = output_block_sizes[i - 1];
        input_block_strides[i] = output_block_strides[i - 1];
      } else {
        input_block_sizes[i] = 1;
      }
    }
    // Fix up input_block_stride for chip dimension.
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      if (chip_dim == 0) {
        input_block_strides[chip_dim] = 1;
      } else {
        input_block_strides[chip_dim] = input_block_strides[chip_dim - 1] *
            input_block_sizes[chip_dim - 1];
      }
    } else {
      if (chip_dim == NumInputDims - 1) {
        input_block_strides[chip_dim] = 1;
      } else {
        input_block_strides[chip_dim] = input_block_strides[chip_dim - 1] *
            input_block_sizes[chip_dim - 1];
      }
    }
    // Write input block.
    this->m_impl.writeBlock(
        InputTensorBlock(this->srcCoeff(output_block.first_coeff_index()),
                         input_block_sizes,
                         input_block_strides,
                         this->m_inputStrides,
                         const_cast<ScalarNonConst*>(output_block.data())));
  }

};


} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_CHIPPING_H
