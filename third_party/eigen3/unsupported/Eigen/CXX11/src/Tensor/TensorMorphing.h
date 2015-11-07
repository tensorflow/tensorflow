// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_MORPHING_H
#define EIGEN_CXX11_TENSOR_TENSOR_MORPHING_H

namespace Eigen {

/** \class TensorReshaping
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Tensor reshaping class.
  *
  *
  */
namespace internal {
template<typename NewDimensions, typename XprType>
struct traits<TensorReshapingOp<NewDimensions, XprType> > : public traits<XprType>
{
  typedef typename XprType::Scalar Scalar;
  typedef traits<XprType> XprTraits;
  typedef typename packet_traits<Scalar>::type Packet;
  typedef typename XprTraits::StorageKind StorageKind;
  typedef typename XprTraits::Index Index;
  typedef typename XprType::Nested Nested;
  typedef typename remove_reference<Nested>::type _Nested;
  static const int NumDimensions = array_size<NewDimensions>::value;
  static const int Layout = XprTraits::Layout;
};

template<typename NewDimensions, typename XprType>
struct eval<TensorReshapingOp<NewDimensions, XprType>, Eigen::Dense>
{
  typedef const TensorReshapingOp<NewDimensions, XprType>& type;
};

template<typename NewDimensions, typename XprType>
struct nested<TensorReshapingOp<NewDimensions, XprType>, 1, typename eval<TensorReshapingOp<NewDimensions, XprType> >::type>
{
  typedef TensorReshapingOp<NewDimensions, XprType> type;
};

}  // end namespace internal



template<typename NewDimensions, typename XprType>
class TensorReshapingOp : public TensorBase<TensorReshapingOp<NewDimensions, XprType>, WriteAccessors>
{
  public:
  typedef typename Eigen::internal::traits<TensorReshapingOp>::Scalar Scalar;
  typedef typename Eigen::internal::traits<TensorReshapingOp>::Packet Packet;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef typename internal::remove_const<typename XprType::CoeffReturnType>::type CoeffReturnType;
  typedef typename internal::remove_const<typename XprType::PacketReturnType>::type PacketReturnType;
  typedef typename Eigen::internal::nested<TensorReshapingOp>::type Nested;
  typedef typename Eigen::internal::traits<TensorReshapingOp>::StorageKind StorageKind;
  typedef typename Eigen::internal::traits<TensorReshapingOp>::Index Index;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorReshapingOp(const XprType& expr, const NewDimensions& dims)
      : m_xpr(expr), m_dims(dims) {}

    EIGEN_DEVICE_FUNC
    const NewDimensions& dimensions() const { return m_dims; }

    EIGEN_DEVICE_FUNC
    const typename internal::remove_all<typename XprType::Nested>::type&
    expression() const { return m_xpr; }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE TensorReshapingOp& operator = (const TensorReshapingOp& other)
    {
      typedef TensorAssignOp<TensorReshapingOp, const TensorReshapingOp> Assign;
      Assign assign(*this, other);
      internal::TensorExecutor<const Assign, DefaultDevice>::run(
          assign, DefaultDevice());
      return *this;
    }

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE TensorReshapingOp& operator = (const OtherDerived& other)
    {
      typedef TensorAssignOp<TensorReshapingOp, const OtherDerived> Assign;
      Assign assign(*this, other);
      internal::TensorExecutor<const Assign, DefaultDevice>::run(
          assign, DefaultDevice());
      return *this;
    }

  protected:
    typename XprType::Nested m_xpr;
    const NewDimensions m_dims;
};


// Eval as rvalue
template<typename NewDimensions, typename ArgType, typename Device>
struct TensorEvaluator<const TensorReshapingOp<NewDimensions, ArgType>, Device>
{
  typedef TensorReshapingOp<NewDimensions, ArgType> XprType;
  typedef NewDimensions Dimensions;

  enum {
    IsAligned = TensorEvaluator<ArgType, Device>::IsAligned,
    PacketAccess = TensorEvaluator<ArgType, Device>::PacketAccess,
    // TODO(andydavis) Re-enable BlockAccess when the performance issue
    // with block-based reshape is resolved.
    BlockAccess = false,
    Layout = TensorEvaluator<ArgType, Device>::Layout,
    CoordAccess = false,  // to be implemented
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device)
      : m_impl(op.expression(), device), m_dimensions(op.dimensions())
  {
    // The total size of the reshaped tensor must be equal to the total size
    // of the input tensor.
    eigen_assert(internal::array_prod(m_impl.dimensions()) == internal::array_prod(op.dimensions()));

    if (BlockAccess) {
      const typename TensorEvaluator<ArgType, Device>::Dimensions& input_dims =
          m_impl.dimensions();
      if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
        m_outputStrides[0] = 1;
        for (int i = 1; i < NumOutputDims; ++i) {
          m_outputStrides[i] = m_outputStrides[i - 1] * m_dimensions[i - 1];
        }
        m_inputStrides[0] = 1;
        for (int i = 1; i < NumInputDims; ++i) {
          m_inputStrides[i] = m_inputStrides[i - 1] * input_dims[i - 1];
        }
      } else {
#ifdef __CUDACC__
        // TODO(andydavis) Remove the following line of code when associated
        // nvcc bug b/22973013 is fixed.
        for (int i = 0; i < 1; ++i) {}
#endif
        m_outputStrides[NumOutputDims - 1] = 1;
        for (int i = NumOutputDims - 2; i >= 0; --i) {
          m_outputStrides[i] = m_outputStrides[i + 1] * m_dimensions[i + 1];
        }
        m_inputStrides[NumInputDims - 1] = 1;
        for (int i = NumInputDims - 2; i >= 0; --i) {
          m_inputStrides[i] = m_inputStrides[i + 1] * input_dims[i + 1];
        }
      }
    }
  }

  typedef typename XprType::Index Index;
  typedef typename XprType::Scalar Scalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;
  static const std::size_t NumOutputDims =
      internal::array_size<Dimensions>::value;
  static const std::size_t NumInputDims = internal::array_size<
    typename TensorEvaluator<ArgType, Device>::Dimensions>::value;
  typedef typename internal::TensorBlock<
    Index, typename internal::remove_const<Scalar>::type, NumOutputDims, Layout>
  OutputTensorBlock;
  typedef typename internal::TensorBlock<
    Index, typename internal::remove_const<Scalar>::type, NumInputDims, Layout>
  InputTensorBlock;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_dimensions; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(CoeffReturnType* data) {
    return m_impl.evalSubExprsIfNeeded(data);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
    m_impl.cleanup();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const
  {
    return m_impl.coeff(index);
  }

  template<int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType packet(Index index) const
  {
    return m_impl.template packet<LoadMode>(index);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void getResourceRequirements(
      std::vector<internal::TensorOpResourceRequirements>* resources) const {
    m_impl.getResourceRequirements(resources);
  }

  // TODO(andydavis) Reduce the overhead of this function.
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void block(
      OutputTensorBlock* output_block) const {
    // Calculate output block unit-stride inner dimension length.
    const DSizes<Index, NumOutputDims>& output_block_sizes =
        output_block->block_sizes();
    Index output_inner_dim_size = 1;
    Index output_outer_dim_start = NumOutputDims;
    for (Index i = 0; i < NumOutputDims; ++i) {
      const Index dim = static_cast<int>(Layout) == static_cast<int>(ColMajor)
          ? i : NumOutputDims - i - 1;
      output_inner_dim_size *= output_block_sizes[dim];
      if (output_block_sizes[dim] < m_dimensions[dim]) {
        output_outer_dim_start = i + 1;
        break;
      }
    }

    // Initialize output block iterator state.
    struct BlockIteratorState {
      Index stride;
      Index span;
      Index size;
      Index count;
    };
    array<BlockIteratorState, NumOutputDims> block_iter_state;

    for (Index i = 0; i < NumOutputDims; ++i) {
      const Index dim = static_cast<int>(Layout) == static_cast<int>(ColMajor)
          ? i : NumOutputDims - i - 1;
      block_iter_state[i].size = output_block_sizes[dim];
      block_iter_state[i].stride = m_outputStrides[dim];
      block_iter_state[i].span =
          block_iter_state[i].stride * (block_iter_state[i].size - 1);
      block_iter_state[i].count = 0;
    }

    const Index output_outer_dim_size = output_block_sizes.TotalSize() /
        output_inner_dim_size;
    const typename TensorEvaluator<ArgType, Device>::Dimensions& input_dims =
        m_impl.dimensions();

    Index index = output_block->first_coeff_index();
    for (Index outer_idx = 0; outer_idx < output_outer_dim_size; ++outer_idx) {
      Index inner_idx = 0;
      while (inner_idx < output_inner_dim_size) {
        // Calculate input coords based on 'index'.
        array<Index, NumInputDims> input_coords;
        Index idx = index;
        if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
          for (int i = NumInputDims - 1; i > 0; --i) {
            input_coords[i] = idx / m_inputStrides[i];
            idx -= input_coords[i] * m_inputStrides[i];
          }
          input_coords[0] = idx;
        } else {
          for (int i = 0; i < NumInputDims - 1; ++i) {
            input_coords[i] = idx / m_inputStrides[i];
            idx -= input_coords[i] * m_inputStrides[i];
          }
          input_coords[NumInputDims - 1] = idx;
        }

        // Calculate target input block shape, using at most
        // 'output_inner_dim_size' coefficients along the input block's inner
        // dimensions.
        DSizes<Index, NumInputDims> input_block_sizes;
        Index num_to_allocate = output_inner_dim_size - inner_idx;
        for (Index i = 0; i < NumInputDims; ++i) {
          const Index dim =
              static_cast<int>(Layout) == static_cast<int>(ColMajor)
              ? i : NumInputDims - i - 1;
          input_block_sizes[dim] = numext::mini(
              num_to_allocate, (static_cast<Index>(input_dims[dim]) -
                                input_coords[dim]));
          if (input_coords[dim] == 0) {
            num_to_allocate /= input_block_sizes[dim];
          } else {
            num_to_allocate = 1;
          }
        }

        // Calculate input block strides.
        DSizes<Index, NumInputDims> input_block_strides;
        if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
          input_block_strides[0] = 1;
          for (int i = 1; i < NumInputDims; ++i) {
            input_block_strides[i] = input_block_strides[i - 1] *
                input_block_sizes[i - 1];
          }
        } else {
          input_block_strides[NumInputDims - 1] = 1;
          for (int i = NumInputDims - 2; i >= 0; --i) {
            input_block_strides[i] = input_block_strides[i + 1] *
                input_block_sizes[i + 1];
          }
        }

        // Instantiate and read input block from input tensor.
        InputTensorBlock input_block(index, input_block_sizes,
                                     input_block_strides, m_inputStrides,
                                     output_block->data() + outer_idx *
                                     output_inner_dim_size + inner_idx);

        m_impl.block(&input_block);

        const Index input_block_total_size = input_block_sizes.TotalSize();
        index += input_block_total_size;
        inner_idx += input_block_total_size;
      }
      eigen_assert(inner_idx == output_inner_dim_size);
      index -= output_inner_dim_size;
      // Update index.
      for (Index i = output_outer_dim_start; i < NumOutputDims; ++i) {
        if (++block_iter_state[i].count < block_iter_state[i].size) {
          index += block_iter_state[i].stride;
          break;
        }
        block_iter_state[i].count = 0;
        index -= block_iter_state[i].span;
      }
    }
  }

  EIGEN_DEVICE_FUNC Scalar* data() const { return const_cast<Scalar*>(m_impl.data()); }

  EIGEN_DEVICE_FUNC const TensorEvaluator<ArgType, Device>& impl() const { return m_impl; }

 protected:
  TensorEvaluator<ArgType, Device> m_impl;
  NewDimensions m_dimensions;
  DSizes<Index, NumOutputDims> m_outputStrides;
  DSizes<Index, NumInputDims> m_inputStrides;
};


// Eval as lvalue
template<typename NewDimensions, typename ArgType, typename Device>
  struct TensorEvaluator<TensorReshapingOp<NewDimensions, ArgType>, Device>
  : public TensorEvaluator<const TensorReshapingOp<NewDimensions, ArgType>, Device>

{
  typedef TensorEvaluator<const TensorReshapingOp<NewDimensions, ArgType>, Device> Base;
  typedef TensorReshapingOp<NewDimensions, ArgType> XprType;
  typedef NewDimensions Dimensions;

  enum {
    IsAligned = TensorEvaluator<ArgType, Device>::IsAligned,
    PacketAccess = TensorEvaluator<ArgType, Device>::PacketAccess,
    BlockAccess = false,
    Layout = TensorEvaluator<ArgType, Device>::Layout,
    CoordAccess = false,  // to be implemented
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device)
    : Base(op, device)
  { }

  typedef typename XprType::Index Index;
  typedef typename XprType::Scalar Scalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType& coeffRef(Index index)
  {
    return this->m_impl.coeffRef(index);
  }
  template <int StoreMode> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  void writePacket(Index index, const PacketReturnType& x)
  {
    this->m_impl.template writePacket<StoreMode>(index, x);
  }
};


/** \class TensorSlicing
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Tensor slicing class.
  *
  *
  */
namespace internal {
template<typename StartIndices, typename Sizes, typename XprType>
struct traits<TensorSlicingOp<StartIndices, Sizes, XprType> > : public traits<XprType>
{
  typedef typename XprType::Scalar Scalar;
  typedef traits<XprType> XprTraits;
  typedef typename packet_traits<Scalar>::type Packet;
  typedef typename XprTraits::StorageKind StorageKind;
  typedef typename XprTraits::Index Index;
  typedef typename XprType::Nested Nested;
  typedef typename remove_reference<Nested>::type _Nested;
  static const int NumDimensions = array_size<StartIndices>::value;
  static const int Layout = XprTraits::Layout;
};

template<typename StartIndices, typename Sizes, typename XprType>
struct eval<TensorSlicingOp<StartIndices, Sizes, XprType>, Eigen::Dense>
{
  typedef const TensorSlicingOp<StartIndices, Sizes, XprType>& type;
};

template<typename StartIndices, typename Sizes, typename XprType>
struct nested<TensorSlicingOp<StartIndices, Sizes, XprType>, 1, typename eval<TensorSlicingOp<StartIndices, Sizes, XprType> >::type>
{
  typedef TensorSlicingOp<StartIndices, Sizes, XprType> type;
};

}  // end namespace internal



template<typename StartIndices, typename Sizes, typename XprType>
class TensorSlicingOp : public TensorBase<TensorSlicingOp<StartIndices, Sizes, XprType> >
{
  public:
  typedef typename Eigen::internal::traits<TensorSlicingOp>::Scalar Scalar;
  typedef typename Eigen::internal::traits<TensorSlicingOp>::Packet Packet;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;
  typedef typename Eigen::internal::nested<TensorSlicingOp>::type Nested;
  typedef typename Eigen::internal::traits<TensorSlicingOp>::StorageKind StorageKind;
  typedef typename Eigen::internal::traits<TensorSlicingOp>::Index Index;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorSlicingOp(const XprType& expr, const StartIndices& indices, const Sizes& sizes)
      : m_xpr(expr), m_indices(indices), m_sizes(sizes) {}

    EIGEN_DEVICE_FUNC
    const StartIndices& startIndices() const { return m_indices; }
    EIGEN_DEVICE_FUNC
    const Sizes& sizes() const { return m_sizes; }

    EIGEN_DEVICE_FUNC
    const typename internal::remove_all<typename XprType::Nested>::type&
    expression() const { return m_xpr; }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE TensorSlicingOp& operator = (const TensorSlicingOp& other)
    {
      typedef TensorAssignOp<TensorSlicingOp, const TensorSlicingOp> Assign;
      Assign assign(*this, other);
      internal::TensorExecutor<const Assign, DefaultDevice>::run(
          assign, DefaultDevice());
      return *this;
    }

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE TensorSlicingOp& operator = (const OtherDerived& other)
    {
      typedef TensorAssignOp<TensorSlicingOp, const OtherDerived> Assign;
      Assign assign(*this, other);
      internal::TensorExecutor<const Assign, DefaultDevice>::run(
          assign, DefaultDevice());
      return *this;
    }

  protected:
    typename XprType::Nested m_xpr;
    const StartIndices m_indices;
    const Sizes m_sizes;
};


// Eval as rvalue
template<typename StartIndices, typename Sizes, typename ArgType, typename Device>
struct TensorEvaluator<const TensorSlicingOp<StartIndices, Sizes, ArgType>, Device>
{
  typedef TensorSlicingOp<StartIndices, Sizes, ArgType> XprType;
  static const int NumDims = internal::array_size<Sizes>::value;

  enum {
    // Alignment can't be guaranteed at compile time since it depends on the
    // slice offsets and sizes.
    IsAligned = /*TensorEvaluator<ArgType, Device>::IsAligned*/ false,
    PacketAccess = TensorEvaluator<ArgType, Device>::PacketAccess,
    BlockAccess = TensorEvaluator<ArgType, Device>::BlockAccess,
    Layout = TensorEvaluator<ArgType, Device>::Layout,
    CoordAccess = TensorEvaluator<ArgType, Device>::CoordAccess,
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device)
      : m_impl(op.expression(), device), m_device(device), m_dimensions(op.sizes()), m_offsets(op.startIndices())
  {
    for (int i = 0; i < internal::array_size<Dimensions>::value; ++i) {
      eigen_assert(m_impl.dimensions()[i] >= op.sizes()[i] + op.startIndices()[i]);
    }

    const typename TensorEvaluator<ArgType, Device>::Dimensions& input_dims = m_impl.dimensions();
    const Sizes& output_dims = op.sizes();
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      m_inputStrides[0] = 1;
      for (int i = 1; i < NumDims; ++i) {
        m_inputStrides[i] = m_inputStrides[i-1] * input_dims[i-1];
      }

      // Don't initialize m_fastOutputStrides[0] since it won't ever be accessed.
      m_outputStrides[0] = 1;
      for (int i = 1; i < NumDims; ++i) {
        m_outputStrides[i] = m_outputStrides[i-1] * output_dims[i-1];
        m_fastOutputStrides[i] = internal::TensorIntDivisor<Index>(m_outputStrides[i]);
      }
    } else {
      m_inputStrides[NumDims-1] = 1;
      for (int i = NumDims - 2; i >= 0; --i) {
        m_inputStrides[i] = m_inputStrides[i+1] * input_dims[i+1];
      }

      m_outputStrides[NumDims-1] = 1;
      for (int i = NumDims - 2; i >= 0; --i) {
        m_outputStrides[i] = m_outputStrides[i+1] * output_dims[i+1];
        m_fastOutputStrides[i] = internal::TensorIntDivisor<Index>(m_outputStrides[i]);
      }
    }

    m_block_total_size_max = numext::maxi(static_cast<std::size_t>(1),
                                          device.lastLevelCacheSize() /
                                          sizeof(Scalar));
  }

  typedef typename XprType::Index Index;
  typedef typename XprType::Scalar Scalar;
  typedef typename internal::remove_const<Scalar>::type ScalarNonConst;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;
  typedef Sizes Dimensions;
  typedef internal::TensorBlock<Index, ScalarNonConst, NumDims, Layout>
    TensorBlock;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_dimensions; }


  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(CoeffReturnType* data) {
    m_impl.evalSubExprsIfNeeded(NULL);
    if (internal::is_arithmetic<typename internal::remove_const<Scalar>::type>::value && data && m_impl.data()) {
      Index contiguous_values = 1;
      if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
        for (int i = 0; i < NumDims; ++i) {
          contiguous_values *= dimensions()[i];
          if (dimensions()[i] != m_impl.dimensions()[i]) {
            break;
          }
        }
      } else {
        for (int i = NumDims-1; i >= 0; --i) {
          contiguous_values *= dimensions()[i];
          if (dimensions()[i] != m_impl.dimensions()[i]) {
            break;
          }
        }
      }
      // Use memcpy if it's going to be faster than using the regular evaluation.
      if (contiguous_values > m_device.memcpyThreshold()) {
        Scalar* src = (Scalar*)m_impl.data();
        for (int i = 0; i < internal::array_prod(dimensions()); i += contiguous_values) {
          Index offset = srcCoeff(i);
          m_device.memcpy((void*)(data+i), src+offset, contiguous_values * sizeof(Scalar));
        }
        return false;
      }
    }
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
        eigen_assert(index+packetSize-1 < internal::array_prod(dimensions()));

    Index inputIndices[] = {0, 0};
    Index indices[] = {index, index + packetSize - 1};
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      for (int i = NumDims - 1; i > 0; --i) {
        const Index idx0 = indices[0] / m_fastOutputStrides[i];
        const Index idx1 = indices[1] / m_fastOutputStrides[i];
        inputIndices[0] += (idx0 + m_offsets[i]) * m_inputStrides[i];
        inputIndices[1] += (idx1 + m_offsets[i]) * m_inputStrides[i];
        indices[0] -= idx0 * m_outputStrides[i];
        indices[1] -= idx1 * m_outputStrides[i];
      }
      inputIndices[0] += (indices[0] + m_offsets[0]);
      inputIndices[1] += (indices[1] + m_offsets[0]);
    } else {
      for (int i = 0; i < NumDims - 1; ++i) {
        const Index idx0 = indices[0] / m_fastOutputStrides[i];
        const Index idx1 = indices[1] / m_fastOutputStrides[i];
        inputIndices[0] += (idx0 + m_offsets[i]) * m_inputStrides[i];
        inputIndices[1] += (idx1 + m_offsets[i]) * m_inputStrides[i];
        indices[0] -= idx0 * m_outputStrides[i];
        indices[1] -= idx1 * m_outputStrides[i];
      }
      inputIndices[0] += (indices[0] + m_offsets[NumDims-1]);
      inputIndices[1] += (indices[1] + m_offsets[NumDims-1]);
    }
    if (inputIndices[1] - inputIndices[0] == packetSize - 1) {
      PacketReturnType rslt = m_impl.template packet<Unaligned>(inputIndices[0]);
      return rslt;
    }
    else {
      typename internal::remove_const<CoeffReturnType>::type values[packetSize];
      values[0] = m_impl.coeff(inputIndices[0]);
      values[packetSize-1] = m_impl.coeff(inputIndices[1]);
      for (int i = 1; i < packetSize-1; ++i) {
        values[i] = coeff(index+i);
      }
      PacketReturnType rslt = internal::pload<PacketReturnType>(values);
      return rslt;
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(const array<Index, NumDims>& coords)
  {
    array<Index, NumDims> inputCoords;
    for (int i = 0; i < NumDims; ++i) {
      inputCoords = coords[i] + this->m_offsets[i];
    }
    return m_impl.coeff(inputCoords);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void getResourceRequirements(
      std::vector<internal::TensorOpResourceRequirements>* resources) const {
    resources->push_back(internal::TensorOpResourceRequirements(
        internal::kSkewedInnerDims, m_block_total_size_max));
    m_impl.getResourceRequirements(resources);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void block(
      TensorBlock* output_block) const {
    TensorBlock input_block(srcCoeff(output_block->first_coeff_index()),
                            output_block->block_sizes(),
                            output_block->block_strides(),
                            m_inputStrides,
                            output_block->data());
    m_impl.block(&input_block);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar* data() const {
    Scalar* result = m_impl.data();
    if (result) {
      Index offset = 0;
      if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
        for (int i = 0; i < NumDims; ++i) {
          if (m_dimensions[i] != m_impl.dimensions()[i]) {
            offset += m_offsets[i] * m_inputStrides[i];
            for (int j = i+1; j < NumDims; ++j) {
              if (m_dimensions[j] > 1) {
                return NULL;
              }
              offset += m_offsets[j] * m_inputStrides[j];
            }
            break;
          }
        }
      } else {
        for (int i = NumDims - 1; i >= 0; --i) {
          if (m_dimensions[i] != m_impl.dimensions()[i]) {
            offset += m_offsets[i] * m_inputStrides[i];
            for (int j = i-1; j >= 0; --j) {
              if (m_dimensions[j] > 1) {
                return NULL;
              }
              offset += m_offsets[j] * m_inputStrides[j];
            }
            break;
          }
        }
      }
      return result + offset;
    }
    return NULL;
  }

 protected:
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index srcCoeff(Index index) const
  {
    Index inputIndex = 0;
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      for (int i = NumDims - 1; i > 0; --i) {
        const Index idx = index / m_fastOutputStrides[i];
        inputIndex += (idx + m_offsets[i]) * m_inputStrides[i];
        index -= idx * m_outputStrides[i];
      }
      inputIndex += (index + m_offsets[0]);
    } else {
      for (int i = 0; i < NumDims - 1; ++i) {
        const Index idx = index / m_fastOutputStrides[i];
        inputIndex += (idx + m_offsets[i]) * m_inputStrides[i];
        index -= idx * m_outputStrides[i];
      }
      inputIndex += (index + m_offsets[NumDims-1]);
    }
    return inputIndex;
  }

  array<Index, NumDims> m_outputStrides;
  array<internal::TensorIntDivisor<Index>, NumDims> m_fastOutputStrides;
  array<Index, NumDims> m_inputStrides;
  TensorEvaluator<ArgType, Device> m_impl;
  const Device& m_device;
  Dimensions m_dimensions;
  const StartIndices m_offsets;
  std::size_t m_block_total_size_max;
};


// Eval as lvalue
template<typename StartIndices, typename Sizes, typename ArgType, typename Device>
struct TensorEvaluator<TensorSlicingOp<StartIndices, Sizes, ArgType>, Device>
  : public TensorEvaluator<const TensorSlicingOp<StartIndices, Sizes, ArgType>, Device>
{
  typedef TensorEvaluator<const TensorSlicingOp<StartIndices, Sizes, ArgType>, Device> Base;
  typedef TensorSlicingOp<StartIndices, Sizes, ArgType> XprType;
  static const int NumDims = internal::array_size<Sizes>::value;

  enum {
    IsAligned = /*TensorEvaluator<ArgType, Device>::IsAligned*/ false,
    PacketAccess = TensorEvaluator<ArgType, Device>::PacketAccess,
    BlockAccess = TensorEvaluator<ArgType, Device>::BlockAccess,
    Layout = TensorEvaluator<ArgType, Device>::Layout,
    CoordAccess = TensorEvaluator<ArgType, Device>::CoordAccess,
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device)
    : Base(op, device)
    { }

  typedef typename XprType::Index Index;
  typedef typename XprType::Scalar Scalar;
  typedef typename internal::remove_const<Scalar>::type ScalarNonConst;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;
  typedef Sizes Dimensions;
  typedef internal::TensorBlock<Index, ScalarNonConst, NumDims, Layout>
    TensorBlock;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType& coeffRef(Index index)
  {
    return this->m_impl.coeffRef(this->srcCoeff(index));
  }

  template <int StoreMode> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  void writePacket(Index index, const PacketReturnType& x)
  {
    const int packetSize = internal::unpacket_traits<PacketReturnType>::size;
    Index inputIndices[] = {0, 0};
    Index indices[] = {index, index + packetSize - 1};
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      for (int i = NumDims - 1; i > 0; --i) {
        const Index idx0 = indices[0] / this->m_fastOutputStrides[i];
        const Index idx1 = indices[1] / this->m_fastOutputStrides[i];
        inputIndices[0] += (idx0 + this->m_offsets[i]) * this->m_inputStrides[i];
        inputIndices[1] += (idx1 + this->m_offsets[i]) * this->m_inputStrides[i];
        indices[0] -= idx0 * this->m_outputStrides[i];
        indices[1] -= idx1 * this->m_outputStrides[i];
      }
      inputIndices[0] += (indices[0] + this->m_offsets[0]);
      inputIndices[1] += (indices[1] + this->m_offsets[0]);
    } else {
      for (int i = 0; i < NumDims - 1; ++i) {
        const Index idx0 = indices[0] / this->m_fastOutputStrides[i];
        const Index idx1 = indices[1] / this->m_fastOutputStrides[i];
        inputIndices[0] += (idx0 + this->m_offsets[i]) * this->m_inputStrides[i];
        inputIndices[1] += (idx1 + this->m_offsets[i]) * this->m_inputStrides[i];
        indices[0] -= idx0 * this->m_outputStrides[i];
        indices[1] -= idx1 * this->m_outputStrides[i];
      }
      inputIndices[0] += (indices[0] + this->m_offsets[NumDims-1]);
      inputIndices[1] += (indices[1] + this->m_offsets[NumDims-1]);
    }
    if (inputIndices[1] - inputIndices[0] == packetSize - 1) {
      this->m_impl.template writePacket<StoreMode>(inputIndices[0], x);
    }
    else {
      EIGEN_ALIGN_DEFAULT CoeffReturnType values[packetSize];
      internal::pstore<CoeffReturnType, PacketReturnType>(values, x);
      this->m_impl.coeffRef(inputIndices[0]) = values[0];
      this->m_impl.coeffRef(inputIndices[1]) = values[packetSize-1];
      for (int i = 1; i < packetSize-1; ++i) {
        this->coeffRef(index+i) = values[i];
      }
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType& coeffRef(const array<Index, NumDims>& coords)
  {
    array<Index, NumDims> inputCoords;
    for (int i = 0; i < NumDims; ++i) {
      inputCoords = coords[i] + this->m_offsets[i];
    }
    return this->m_impl.coeffRef(inputCoords);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void writeBlock(
      const TensorBlock& block) {
    this->m_impl.writeBlock(
        TensorBlock(this->srcCoeff(block.first_coeff_index()),
                    block.block_sizes(),
                    block.block_strides(),
                    this->m_inputStrides,
                    const_cast<ScalarNonConst*>(block.data())));

  }
};


} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_MORPHING_H
