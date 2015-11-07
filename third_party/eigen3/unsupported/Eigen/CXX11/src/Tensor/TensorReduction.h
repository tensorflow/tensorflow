// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_H
#define EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_H

namespace Eigen {

/** \class TensorReduction
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Tensor reduction class.
  *
  */

namespace internal {
template<typename Op, typename Dims, typename XprType>
struct traits<TensorReductionOp<Op, Dims, XprType> >
 : traits<XprType>
{
  typedef typename traits<XprType>::Scalar Scalar;
  typedef typename traits<XprType>::StorageKind StorageKind;
  typedef typename traits<XprType>::Index Index;
  typedef typename XprType::Nested Nested;
};

template<typename Op, typename Dims, typename XprType>
struct eval<TensorReductionOp<Op, Dims, XprType>, Eigen::Dense>
{
  typedef const TensorReductionOp<Op, Dims, XprType>& type;
};

template<typename Op, typename Dims, typename XprType>
struct nested<TensorReductionOp<Op, Dims, XprType>, 1, typename eval<TensorReductionOp<Op, Dims, XprType> >::type>
{
  typedef TensorReductionOp<Op, Dims, XprType> type;
};



template <typename InputDims, typename OutputDims, typename ReducedDims> EIGEN_DEVICE_FUNC
static void partition_dims(const InputDims& input_dims,
                           const array<bool, internal::array_size<InputDims>::value>& reduced,
                           OutputDims* output_dims, ReducedDims* reduced_dims) {
  const int NumInputDims = internal::array_size<InputDims>::value;
  int outputIndex = 0;
  int reduceIndex = 0;
  for (int i = 0; i < NumInputDims; ++i) {
    if (OutputDims::count == 0 || reduced[i]) {
      (*reduced_dims)[reduceIndex] = input_dims[i];
      ++reduceIndex;
    } else {
      (*output_dims)[outputIndex] = input_dims[i];
      ++outputIndex;
    }
  }
}



template <typename ReducedDims, int NumTensorDims, int Layout>
struct are_inner_most_dims {
  static const bool value = false;
};
template <typename ReducedDims, int NumTensorDims, int Layout>
struct preserve_inner_most_dims {
  static const bool value = false;
};

#if defined(EIGEN_HAS_CONSTEXPR) && defined(EIGEN_HAS_VARIADIC_TEMPLATES)
// The use of the tmp1, tmp2, tmp3 intermediate variables is needed for nvcc 7
// to compile the code below. NVidia is working on a fix.
template <typename ReducedDims, int NumTensorDims>
struct are_inner_most_dims<ReducedDims, NumTensorDims, ColMajor>{
  static const bool tmp1 = indices_statically_known_to_increase<ReducedDims>()();
  static const bool tmp2 = index_statically_eq<ReducedDims>()(0, 0);
  static const bool tmp3 = index_statically_eq<ReducedDims>()(array_size<ReducedDims>::value-1, array_size<ReducedDims>::value-1);
  static const bool value = tmp1 & tmp2 & tmp3;
};
template <typename ReducedDims, int NumTensorDims>
struct are_inner_most_dims<ReducedDims, NumTensorDims, RowMajor>{
  static const bool tmp1 = indices_statically_known_to_increase<ReducedDims>()();
  static const bool tmp2 = index_statically_eq<ReducedDims>()(0, NumTensorDims - array_size<ReducedDims>::value);
  static const bool tmp3 = index_statically_eq<ReducedDims>()(array_size<ReducedDims>::value - 1, NumTensorDims - 1);
  static const bool value = tmp1 & tmp2 & tmp3;

};
template <typename ReducedDims, int NumTensorDims>
struct preserve_inner_most_dims<ReducedDims, NumTensorDims, ColMajor>{
  static const bool tmp1 = indices_statically_known_to_increase<ReducedDims>()();
  static const bool tmp2 = index_statically_gt<ReducedDims>()(0, 0);
  static const bool value = tmp1 & tmp2;

};
template <typename ReducedDims, int NumTensorDims>
struct preserve_inner_most_dims<ReducedDims, NumTensorDims, RowMajor>{
  static const bool tmp1 = indices_statically_known_to_increase<ReducedDims>()();
  static const bool tmp2 = index_statically_lt<ReducedDims>()(array_size<ReducedDims>::value - 1, NumTensorDims - 1);
  static const bool value = tmp1 & tmp2;
};
#endif


template <int DimIndex, typename Self, typename Op>
struct GenericDimReducer {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const Self& self, typename Self::Index firstIndex, Op& reducer, typename Self::CoeffReturnType* accum) {
    EIGEN_STATIC_ASSERT(DimIndex >= 0, YOU_MADE_A_PROGRAMMING_MISTAKE);
    for (int j = 0; j < self.m_reducedDims[DimIndex]; ++j) {
      const typename Self::Index input = firstIndex + j * self.m_reducedStrides[DimIndex];
      GenericDimReducer<DimIndex-1, Self, Op>::reduce(self, input, reducer, accum);
    }
  }
};
template <typename Self, typename Op>
struct GenericDimReducer<-1, Self, Op> {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const Self& self, typename Self::Index firstIndex, Op& reducer, typename Self::CoeffReturnType* accum) {
    reducer.reduce(self.m_impl.coeff(firstIndex), accum);
  }
};

template <typename Self, typename Op, bool Vectorizable = (Self::InputPacketAccess & Op::PacketAccess)>
struct InnerMostDimReducer {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE typename Self::CoeffReturnType reduce(const Self& self, typename Self::Index firstIndex, typename Self::Index numValuesToReduce, Op& reducer) {
    typename Self::CoeffReturnType accum = reducer.initialize();
    for (typename Self::Index j = 0; j < numValuesToReduce; ++j) {
      reducer.reduce(self.m_impl.coeff(firstIndex + j), &accum);
    }
    return reducer.finalize(accum);
  }
};

template <typename Self, typename Op>
struct InnerMostDimReducer<Self, Op, true> {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE typename Self::CoeffReturnType reduce(const Self& self, typename Self::Index firstIndex, typename Self::Index numValuesToReduce, Op& reducer) {
    const int packetSize = internal::unpacket_traits<typename Self::PacketReturnType>::size;
    const typename Self::Index VectorizedSize = (numValuesToReduce / packetSize) * packetSize;
    typename Self::PacketReturnType p = reducer.template initializePacket<typename Self::PacketReturnType>();
    for (typename Self::Index j = 0; j < VectorizedSize; j += packetSize) {
      reducer.reducePacket(self.m_impl.template packet<Unaligned>(firstIndex + j), &p);
    }
    typename Self::CoeffReturnType accum = reducer.initialize();
    for (typename Self::Index j = VectorizedSize; j < numValuesToReduce; ++j) {
      reducer.reduce(self.m_impl.coeff(firstIndex + j), &accum);
    }
    return reducer.finalizeBoth(accum, p);
  }
};

template <int DimIndex, typename Self, typename Op, bool vectorizable = (Self::InputPacketAccess & Op::PacketAccess)>
struct InnerMostDimPreserver {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const Self& self, typename Self::Index firstIndex, Op& reducer, typename Self::PacketReturnType* accum) {
    eigen_assert(false && "should never be called");
  }
};

template <int DimIndex, typename Self, typename Op>
struct InnerMostDimPreserver<DimIndex, Self, Op, true> {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const Self& self, typename Self::Index firstIndex, Op& reducer, typename Self::PacketReturnType* accum) {
    EIGEN_STATIC_ASSERT(DimIndex >= 0, YOU_MADE_A_PROGRAMMING_MISTAKE);
    for (typename Self::Index j = 0; j < self.m_reducedDims[DimIndex]; ++j) {
      const typename Self::Index input = firstIndex + j * self.m_reducedStrides[DimIndex];
      InnerMostDimPreserver<DimIndex-1, Self, Op>::reduce(self, input, reducer, accum);
    }
  }
};

template <typename Self, typename Op>
struct InnerMostDimPreserver<-1, Self, Op, true> {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const Self& self, typename Self::Index firstIndex, Op& reducer, typename Self::PacketReturnType* accum) {
    reducer.reducePacket(self.m_impl.template packet<Unaligned>(firstIndex), accum);
  }
};

// Default full reducer
template <typename Self, typename Op, typename Device, bool Vectorizable = (Self::InputPacketAccess & Op::PacketAccess)>
struct FullReducer {
  static const bool HasOptimizedImplementation = false;

  static EIGEN_DEVICE_FUNC void run(const Self& self, Op& reducer, const Device&, typename Self::CoeffReturnType* output) {
    const typename Self::Index num_coeffs = array_prod(self.m_impl.dimensions());
    *output = InnerMostDimReducer<Self, Op>::reduce(self, 0, num_coeffs, reducer);
  }
};


#ifdef EIGEN_USE_THREADS
// Multithreaded full reducers
template <typename Eval, typename Op, bool Vectorizable = (Eval::InputPacketAccess & Op::PacketAccess)>
struct FullReducerShard {
  static void run(const Eval& eval, typename Eval::Index firstIndex, typename Eval::Index numValuesToReduce, Op& reducer, FullReducerShard* shard) {

    shard->saccum = reducer.initialize();
    for (typename Eval::Index j = 0; j < numValuesToReduce; ++j) {
      reducer.reduce(eval.m_impl.coeff(firstIndex + j), &shard->saccum);
    }
  }

  typename Eval::CoeffReturnType saccum;
};

template <typename Eval, typename Op>
struct FullReducerShard<Eval, Op, true> {
  static void run(const Eval& eval, typename Eval::Index firstIndex, typename Eval::Index numValuesToReduce, Op& reducer, FullReducerShard* shard) {

    const int packetSize = internal::unpacket_traits<typename Eval::PacketReturnType>::size;
    const typename Eval::Index VectorizedSize = (numValuesToReduce / packetSize) * packetSize;

    shard->paccum = reducer.template initializePacket<typename Eval::PacketReturnType>();
    for (typename Eval::Index j = 0; j < VectorizedSize; j += packetSize) {
      reducer.reducePacket(eval.m_impl.template packet<Unaligned>(firstIndex + j), &shard->paccum);
    }
    shard->saccum = reducer.initialize();
    for (typename Eval::Index j = VectorizedSize; j < numValuesToReduce; ++j) {
      reducer.reduce(eval.m_impl.coeff(firstIndex + j), &shard->saccum);
    }
  }

  typename Eval::PacketReturnType paccum;
  typename Eval::CoeffReturnType saccum;
};


template <typename Self, typename Op>
struct FullReducer<Self, Op, ThreadPoolDevice, false> {
  static const bool HasOptimizedImplementation = !Op::IsStateful;

  // launch one reducer per thread and accumulate the result.
  static void run(const Self& self, Op& reducer, const ThreadPoolDevice& device, typename Self::CoeffReturnType* output) {
    typedef typename Self::Index Index;
    const Index num_coeffs = array_prod(self.m_impl.dimensions());
    const Index blocksize = std::floor<Index>(static_cast<float>(num_coeffs)/device.numThreads());
    const Index numblocks = blocksize > 0 ? num_coeffs / blocksize : 0;
    eigen_assert(num_coeffs >= numblocks * blocksize);

    FixedSizeVector<Notification*> results(numblocks);
    FixedSizeVector<FullReducerShard<Self, Op, false> > shards(numblocks, FullReducerShard<Self, Op, false>());
    for (Index i = 0; i < numblocks; ++i) {
      results.push_back(device.enqueue(&FullReducerShard<Self, Op, false>::run, self, i*blocksize, blocksize, reducer, &shards[i]));
    }

    FullReducerShard<Self, Op, false> finalShard;
    if (numblocks * blocksize < num_coeffs) {
      FullReducerShard<Self, Op, false>::run(self, numblocks * blocksize, num_coeffs - numblocks * blocksize, reducer, &finalShard);
    } else {
      finalShard.saccum = reducer.initialize();
    }

    for (Index i = 0; i < numblocks; ++i) {
      wait_until_ready(results[i]);
      delete results[i];
    }

    for (Index i = 0; i < numblocks; ++i) {
      reducer.reduce(shards[i].saccum, &finalShard.saccum);
    }
    *output = reducer.finalize(finalShard.saccum);
  }
};

template <typename Self, typename Op>
struct FullReducer<Self, Op, ThreadPoolDevice, true> {
  static const bool HasOptimizedImplementation = !Op::IsStateful;

  // launch one reducer per thread and accumulate the result.
  static void run(const Self& self, Op& reducer, const ThreadPoolDevice& device, typename Self::CoeffReturnType* output) {
    typedef typename Self::Index Index;
    const Index num_coeffs = array_prod(self.m_impl.dimensions());
    const Index blocksize = std::floor<Index>(static_cast<float>(num_coeffs)/device.numThreads());
    const Index numblocks = blocksize > 0 ? num_coeffs / blocksize : 0;
    eigen_assert(num_coeffs >= numblocks * blocksize);

    FixedSizeVector<Notification*> results(numblocks);
    FixedSizeVector<FullReducerShard<Self, Op, true> > shards(numblocks, FullReducerShard<Self, Op, true>());
    for (Index i = 0; i < numblocks; ++i) {
      results.push_back(device.enqueue(&FullReducerShard<Self, Op, true>::run, self, i*blocksize, blocksize, reducer, &shards[i]));
    }

    FullReducerShard<Self, Op, true> finalShard;
    if (numblocks * blocksize < num_coeffs) {
      FullReducerShard<Self, Op, true>::run(self, numblocks * blocksize, num_coeffs - numblocks * blocksize, reducer, &finalShard);
    } else {
      finalShard.paccum = reducer.template initializePacket<typename Self::PacketReturnType>();
      finalShard.saccum = reducer.initialize();
    }

    for (Index i = 0; i < numblocks; ++i) {
      wait_until_ready(results[i]);
      delete results[i];
    }

    for (Index i = 0; i < numblocks; ++i) {
      reducer.reducePacket(shards[i].paccum, &finalShard.paccum);
      reducer.reduce(shards[i].saccum, &finalShard.saccum);
    }

    *output = reducer.finalizeBoth(finalShard.saccum, finalShard.paccum);
  }
};
#endif


#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
// Full reducers for GPU, don't vectorize for now

// Reducer function that enables multiple cuda thread to safely accumulate at the same
// output address. It basically reads the current value of the output variable, and
// attempts to update it with the new value. If in the meantime another cuda thread
// updated the content of the output address it will try again.
template <typename T, typename R>
__device__ EIGEN_ALWAYS_INLINE void atomicReduce(T* output, T accum, R& reducer) {
#if __CUDA_ARCH__ >= 300
  if (sizeof(T) == 4)
  {
    unsigned int oldval = *reinterpret_cast<unsigned int*>(output);
    unsigned int newval = oldval;
    reducer.reduce(accum, reinterpret_cast<T*>(&newval));
    if (newval == oldval) {
      return;
    }
    unsigned int readback;
    while ((readback = atomicCAS((unsigned int*)output, oldval, newval)) != oldval) {
      oldval = readback;
      newval = oldval;
      reducer.reduce(accum, reinterpret_cast<T*>(&newval));
      if (newval == oldval) {
        return;
      }
    }
  }
  else if (sizeof(T) == 8) {
    unsigned long long oldval = *reinterpret_cast<unsigned long long*>(output);
    unsigned long long newval = oldval;
    reducer.reduce(accum, reinterpret_cast<T*>(&newval));
    if (newval == oldval) {
      return;
    }
    unsigned long long readback;
    while ((readback = atomicCAS((unsigned long long*)output, oldval, newval)) != oldval) {
      oldval = readback;
      newval = oldval;
      reducer.reduce(accum, reinterpret_cast<T*>(&newval));
      if (newval == oldval) {
        return;
      }
    }
  }
  else {
    assert(0 && "Wordsize not supported");
  }
#else
  assert(0 && "Shouldn't be called on unsupported device");
#endif
}

template <typename T>
__device__ inline void atomicReduce(T* output, T accum, SumReducer<T>&) {
#if __CUDA_ARCH__ >= 300
  atomicAdd(output, accum);
#else
  assert(0 && "Shouldn't be called on unsupported device");
#endif
}

template <int BlockSize, int NumPerThread, typename Self,
          typename Reducer, typename Index>
__global__ void FullReductionKernel(Reducer reducer, const Self input, Index num_coeffs,
                                    typename Self::CoeffReturnType* output) {
  const Index first_index = blockIdx.x * BlockSize * NumPerThread + threadIdx.x;

  if (first_index == 0) {
    *output = reducer.initialize();
  }

  typename Self::CoeffReturnType accum = reducer.initialize();
  for (Index i = 0; i < NumPerThread; ++i) {
    const Index index = first_index + i * BlockSize;
    if (index >= num_coeffs) {
      break;
    }
    typename Self::CoeffReturnType val = input.m_impl.coeff(index);
    reducer.reduce(val, &accum);
  }

  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    reducer.reduce(__shfl_down(accum, offset), &accum);
  }

  if ((threadIdx.x & (warpSize - 1)) == 0) {
    atomicReduce(output, accum, reducer);
  }
}


template <typename Self, typename Op, bool Vectorizable>
struct FullReducer<Self, Op, GpuDevice, Vectorizable> {
  // Unfortunately nvidia doesn't support well exotic types such as complex,
  // so reduce the scope of the optimized version of the code to the simple case
  // of floats.
  static const bool HasOptimizedImplementation = !Op::IsStateful &&
                                                 internal::is_same<typename Self::CoeffReturnType, float>::value;

  template <typename OutputType>
  static void run(const Self& self, Op& reducer, const GpuDevice& device, OutputType* output) {
    assert(false && "Should only be called on floats");
  }

  static void run(const Self& self, Op& reducer, const GpuDevice& device, float* output) {
    typedef typename Self::Index Index;

    const Index num_coeffs = array_prod(self.m_impl.dimensions());
    const int block_size = 256;
    const int num_per_thread = 128;
    const int num_blocks = std::ceil(static_cast<float>(num_coeffs) / (block_size * num_per_thread));
    LAUNCH_CUDA_KERNEL((FullReductionKernel<block_size, num_per_thread>),
                       num_blocks, block_size, 0, device, reducer, self, num_coeffs, output);
  }
};

#endif


template <typename Self, typename Op,
          bool Vectorizable = (Self::InputPacketAccess & Op::PacketAccess)>
class BlockReducer {
 public:
  typedef typename Self::Index Index;
  typedef typename Self::Scalar Scalar;
  typedef typename Self::CoeffReturnType CoeffReturnType;
  typedef typename Self::PacketReturnType PacketReturnType;
  explicit BlockReducer(const Op& reducer) : op_(reducer) {
    accum_ = op_.initialize();
  }
  void Reduce(Index index, Index num_values_to_reduce, Scalar* data) {
    for (Index i = 0; i < num_values_to_reduce; ++i) {
      op_.reduce(data[index + i], &accum_);
    }
  }
  CoeffReturnType Finalize() {
    return op_.finalize(accum_);
  }
  PacketReturnType FinalizePacket() {
    // TODO(andydavis) This function should not be called for Scalar
    // reductions: clean this up or add an assert here.
    return PacketReturnType();
  }

 private:
  CoeffReturnType accum_;
  Op op_;
};

template <typename Self, typename Op>
class BlockReducer<Self, Op, true> {
 public:
  typedef typename Self::Index Index;
  typedef typename Self::Scalar Scalar;
  typedef typename Self::CoeffReturnType CoeffReturnType;
  typedef typename Self::PacketReturnType PacketReturnType;
  explicit BlockReducer(const Op& reducer) : op_(reducer) {
    vaccum_ = op_.template initializePacket<PacketReturnType>();
    accum_ = op_.initialize();
  }
  void Reduce(Index index, Index num_values_to_reduce, Scalar* data) {
    const int packet_size = internal::unpacket_traits<PacketReturnType>::size;
    const Index vectorized_size = (num_values_to_reduce / packet_size) *
        packet_size;
    for (Index i = 0; i < vectorized_size; i += packet_size) {
      op_.reducePacket(internal::ploadt<PacketReturnType, Unaligned>(
          &data[index + i]), &vaccum_);
    }
    for (Index i = vectorized_size; i < num_values_to_reduce; ++i) {
      op_.reduce(data[index + i], &accum_);
    }
  }
  CoeffReturnType Finalize() {
    return op_.finalizeBoth(accum_, vaccum_);
  }
  PacketReturnType FinalizePacket() {
    return op_.finalizePacket(vaccum_);
  }

 private:
  PacketReturnType vaccum_;
  CoeffReturnType accum_;
  Op op_;
};

}  // end namespace internal


template <typename Op, typename Dims, typename XprType>
class TensorReductionOp : public TensorBase<TensorReductionOp<Op, Dims, XprType>, ReadOnlyAccessors> {
  public:
    typedef typename Eigen::internal::traits<TensorReductionOp>::Scalar Scalar;
    typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
    typedef typename internal::remove_const<typename XprType::CoeffReturnType>::type CoeffReturnType;
    typedef typename Eigen::internal::nested<TensorReductionOp>::type Nested;
    typedef typename Eigen::internal::traits<TensorReductionOp>::StorageKind StorageKind;
    typedef typename Eigen::internal::traits<TensorReductionOp>::Index Index;

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    TensorReductionOp(const XprType& expr, const Dims& dims) : m_expr(expr), m_dims(dims)
    { }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    TensorReductionOp(const XprType& expr, const Dims& dims, const Op& reducer) : m_expr(expr), m_dims(dims), m_reducer(reducer)
    { }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const XprType& expression() const { return m_expr; }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const Dims& dims() const { return m_dims; }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const Op& reducer() const { return m_reducer; }

  protected:
    typename XprType::Nested m_expr;
    const Dims m_dims;
    const Op m_reducer;
};


// Eval as rvalue
template<typename Op, typename Dims, typename ArgType, typename Device>
struct TensorEvaluator<const TensorReductionOp<Op, Dims, ArgType>, Device>
{
  typedef TensorReductionOp<Op, Dims, ArgType> XprType;
  typedef typename XprType::Index Index;
  typedef typename TensorEvaluator<ArgType, Device>::Dimensions InputDimensions;
  static const int NumInputDims = internal::array_size<InputDimensions>::value;
  static const int NumReducedDims = internal::array_size<Dims>::value;
  EIGEN_STATIC_ASSERT(NumInputDims >= NumReducedDims, YOU_MADE_A_PROGRAMMING_MISTAKE)
  static const int NumOutputDims = NumInputDims - NumReducedDims;
  typedef DSizes<Index, NumOutputDims> Dimensions;
  typedef typename XprType::Scalar Scalar;
  typedef typename internal::remove_const<Scalar>::type ScalarNonConst;
  typedef TensorEvaluator<const TensorReductionOp<Op, Dims, ArgType>, Device> Self;
  static const bool InputPacketAccess = TensorEvaluator<ArgType, Device>::PacketAccess;

  enum {
    IsAligned = false,
    PacketAccess = Self::InputPacketAccess && Op::PacketAccess,
    BlockAccess = TensorEvaluator<ArgType, Device>::BlockAccess,
    Layout = TensorEvaluator<ArgType, Device>::Layout,
    CoordAccess = false,  // to be implemented
  };

  typedef typename internal::TensorBlock<Index, ScalarNonConst, NumOutputDims,
                                         Layout> OutputTensorBlock;
  typedef typename internal::TensorBlock<Index, ScalarNonConst, NumInputDims,
                                         Layout> InputTensorBlock;

  static const bool ReducingInnerMostDims = internal::are_inner_most_dims<Dims, NumInputDims, Layout>::value;
  static const bool PreservingInnerMostDims = internal::preserve_inner_most_dims<Dims, NumInputDims, Layout>::value;
  static const bool RunningFullReduction = (NumInputDims==NumReducedDims);

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device)
      : m_impl(op.expression(), device), m_reducer(op.reducer()), m_result(NULL), m_device(device)
  {
    EIGEN_STATIC_ASSERT((!ReducingInnerMostDims | !PreservingInnerMostDims | (NumReducedDims == NumInputDims)),
                        YOU_MADE_A_PROGRAMMING_MISTAKE);
    for (int i = 0; i < NumInputDims; ++i) {
      m_reduced_dim[i] = false;
    }
    for (int i = 0; i < NumReducedDims; ++i) {
      eigen_assert(op.dims()[i] >= 0);
      eigen_assert(op.dims()[i] < NumInputDims);
      m_reduced_dim[op.dims()[i]] = true;
    }

    const typename TensorEvaluator<ArgType, Device>::Dimensions& input_dims = m_impl.dimensions();
    internal::partition_dims(input_dims, m_reduced_dim, &m_dimensions, &m_reducedDims);

    // Precompute output strides.
    if (NumOutputDims > 0) {
      if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
        m_outputStrides[0] = 1;
        for (int i = 1; i < NumOutputDims; ++i) {
          m_outputStrides[i] = m_outputStrides[i - 1] * m_dimensions[i - 1];
          m_fastOutputStrides[i] = internal::TensorIntDivisor<Index>(m_outputStrides[i]);
        }
      } else {
        m_outputStrides[NumOutputDims - 1] = 1;
        for (int i = NumOutputDims - 2; i >= 0; --i) {
          m_outputStrides[i] = m_outputStrides[i + 1] * m_dimensions[i + 1];
          m_fastOutputStrides[i] = internal::TensorIntDivisor<Index>(m_outputStrides[i]);
        }
      }
    }

    // Precompute input strides.
    if (NumInputDims > 0) {
      if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
        m_inputStrides[0] = 1;
        for (int i = 1; i < NumInputDims; ++i) {
          m_inputStrides[i] = m_inputStrides[i-1] * input_dims[i-1];
        }
      } else {
        m_inputStrides[NumInputDims - 1] = 1;
        for (int i = NumInputDims - 2; i >= 0; --i) {
          m_inputStrides[i] = m_inputStrides[i + 1] * input_dims[i + 1];
        }
      }
    }

    int outputIndex = 0;
    int reduceIndex = 0;
    for (int i = 0; i < NumInputDims; ++i) {
      if (m_reduced_dim[i]) {
        m_reducedStrides[reduceIndex] = m_inputStrides[i];
        ++reduceIndex;
      } else {
        m_preservedStrides[outputIndex] = m_inputStrides[i];
        m_output_to_input_dim_map[outputIndex] = i;
        ++outputIndex;
      }
    }

    m_numValuesToReduce
        = NumOutputDims == 0 ? internal::array_prod(input_dims)
        : (static_cast<int>(Layout) == static_cast<int>(ColMajor))
            ? m_preservedStrides[0] : m_preservedStrides[NumOutputDims - 1];

    m_block_total_size_max = numext::maxi(static_cast<std::size_t>(1),
                                        device.lastLevelCacheSize() /
                                        sizeof(Scalar));
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_dimensions; }

  typedef typename internal::remove_const<typename XprType::CoeffReturnType>::type CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, Device>::type PacketReturnType;

  EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(CoeffReturnType* data) {
    m_impl.evalSubExprsIfNeeded(NULL);

    // Use the FullReducer if possible.
    if (RunningFullReduction && internal::FullReducer<Self, Op, Device>::HasOptimizedImplementation &&
        ((RunningOnGPU && (m_device.majorDeviceVersion() >= 3)) ||
         (internal::array_prod(m_impl.dimensions()) > 1024 * 1024))) {

      bool need_assign = false;
      if (!data) {
        m_result = static_cast<CoeffReturnType*>(m_device.allocate(sizeof(CoeffReturnType)));
        data = m_result;
        need_assign = true;
      }

      Op reducer(m_reducer);
      internal::FullReducer<Self, Op, Device>::run(*this, reducer, m_device, data);
      return need_assign;
    }

    return true;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
    m_impl.cleanup();

    if (m_result) {
      m_device.deallocate(m_result);
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const
  {
    if (RunningFullReduction && m_result) {
      return *m_result;
    }
    Op reducer(m_reducer);
    if (ReducingInnerMostDims) {
      return internal::InnerMostDimReducer<Self, Op>::reduce(*this, firstInput(index),
                                                             m_numValuesToReduce, reducer);
    } else {
      typename Self::CoeffReturnType accum = reducer.initialize();
      internal::GenericDimReducer<NumReducedDims-1, Self, Op>::reduce(*this, firstInput(index), reducer, &accum);
      return reducer.finalize(accum);
    }
  }

  // TODO(bsteiner): provide a more efficient implementation.
  template<int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType packet(Index index) const
  {
    const int packetSize = internal::unpacket_traits<PacketReturnType>::size;
    EIGEN_STATIC_ASSERT(packetSize > 1, YOU_MADE_A_PROGRAMMING_MISTAKE)
    eigen_assert(index + packetSize - 1 < dimensions().TotalSize());

    EIGEN_ALIGN_DEFAULT typename internal::remove_const<CoeffReturnType>::type values[packetSize];
    if (ReducingInnerMostDims) {
      const Index num_values_to_reduce = m_numValuesToReduce;
      const Index firstIndex = firstInput(index);
      for (Index i = 0; i < packetSize; ++i) {
        Op reducer(m_reducer);
        values[i] = internal::InnerMostDimReducer<Self, Op>::reduce(*this, firstIndex + i * num_values_to_reduce,
                                                                    num_values_to_reduce, reducer);
      }
    } else if (PreservingInnerMostDims) {
      const Index firstIndex = firstInput(index);
      const int innermost_dim = (static_cast<int>(Layout) == static_cast<int>(ColMajor)) ? 0 : NumOutputDims - 1;
      // TBD: extend this the the n innermost dimensions that we preserve.
      if (((firstIndex % m_dimensions[innermost_dim]) + packetSize - 1) < m_dimensions[innermost_dim]) {
        Op reducer(m_reducer);
        typename Self::PacketReturnType accum = reducer.template initializePacket<typename Self::PacketReturnType>();
        internal::InnerMostDimPreserver<NumReducedDims-1, Self, Op>::reduce(*this, firstIndex, reducer, &accum);
        return reducer.finalizePacket(accum);
      } else {
        for (int i = 0; i < packetSize; ++i) {
          values[i] = coeff(index + i);
        }
      }
    } else {
      for (int i = 0; i < packetSize; ++i) {
        values[i] = coeff(index + i);
      }
    }
    PacketReturnType rslt = internal::pload<PacketReturnType>(values);
    return rslt;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void getResourceRequirements(
      std::vector<internal::TensorOpResourceRequirements>* resources) const {
    resources->push_back(internal::TensorOpResourceRequirements(
        internal::kSkewedInnerDims, m_block_total_size_max));
    m_impl.getResourceRequirements(resources);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void block(
      OutputTensorBlock* output_block) const {
    // Special case full reductions to avoid input block copy below.
    if (NumInputDims == NumReducedDims) {
      eigen_assert(output_block->first_coeff_index() == 0);
      eigen_assert(output_block->block_sizes().TotalSize() == 1);
      Op reducer(m_reducer);
      output_block->data()[0] = internal::InnerMostDimReducer<Self, Op>::reduce(
          *this, 0, m_numValuesToReduce, reducer);
      return;
    }

    // Calculate input tensor 'slice' required to reduce output block coeffs.
    DSizes<Index, NumInputDims> input_slice_sizes(m_impl.dimensions());
    for (int i = 0; i < NumOutputDims; ++i) {
      // Clip preserved input dimensions by output block size.
      input_slice_sizes[m_output_to_input_dim_map[i]] =
          output_block->block_sizes()[i];
    }

    // Shard input tensor slice into blocks (because it could be large if we
    // need to reduce along several dimensions to calculate required output
    // coefficients).
    const Index max_coeff_count =
        numext::mini(((m_device.firstLevelCacheSize()) / sizeof(Scalar)),
                   input_slice_sizes.TotalSize());

    // Calculate max output shard size needed to keep working set of reducers
    // in L1, while leaving enough space for reducer overhead and 'packet_size'
    // reductions.
    DSizes<Index, NumInputDims> target_input_block_sizes;
    CalculateTargetInputBlockShape(max_coeff_count, input_slice_sizes,
                                   &target_input_block_sizes);
    // Calculate indices for first preserved dimension.
    const Index first_preserved_dim_output_index =
        static_cast<int>(Layout) == static_cast<int>(ColMajor) ?
        0 : NumOutputDims - 1;
    const Index first_preserved_dim_input_index = m_output_to_input_dim_map[
        first_preserved_dim_output_index];
    const bool inner_most_dim_preserved = first_preserved_dim_input_index ==
        (static_cast<int>(Layout) == static_cast<int>(ColMajor) ? 0 :
         NumInputDims - 1) | PreservingInnerMostDims;

    // Calculate output block inner/outer dimension sizes.
    const Index output_block_inner_dim_size = output_block->block_sizes()[
        first_preserved_dim_output_index];
    const Index output_block_outer_dim_size =
        output_block->block_sizes().TotalSize() / output_block_inner_dim_size;
    // Calculate shard size for first preserved dimension.
    const Index output_shard_size = target_input_block_sizes[
        first_preserved_dim_input_index];
    const Index num_output_shards =
        (output_block_inner_dim_size + output_shard_size - 1) /
        output_shard_size;

    // Initialize 'tensor_slice_offsets' from input coords of output index.
    DSizes<Index, NumInputDims> tensor_slice_offsets;
    GetInputCoordsForOutputIndex(output_block->first_coeff_index(),
                                 &tensor_slice_offsets);

    // Store tensor slice offset in first preserved dimension to be used
    // to update tensor slice extents in loop below.
    const Index first_preserved_dim_offset_start = tensor_slice_offsets[
        first_preserved_dim_input_index];

    array<BlockIteratorState, NumOutputDims> block_iter_state;

    // Initialize state used to iterate through output coefficients
    // and update 'tensor_slice_offsets' in outer preserved dims.
    for (int i = 0; i < NumOutputDims - 1; ++i) {
      const int dim = static_cast<int>(Layout) == static_cast<int>(ColMajor)
          ? i + 1 : NumOutputDims - i - 2;
      block_iter_state[i].input_dim = m_output_to_input_dim_map[dim];
      block_iter_state[i].output_size = output_block->block_sizes()[dim];
      block_iter_state[i].output_count = 0;
    }

    // Allocate input block memory.
    ScalarNonConst* input_block_data = static_cast<ScalarNonConst*>(
        m_device.allocate(max_coeff_count * sizeof(Scalar)));
    // Allocate reducer memory.
    const bool packet_reductions_enabled = (Self::InputPacketAccess &
                                            Op::PacketAccess);
    const Index packet_size = internal::unpacket_traits<PacketReturnType>::size;
    const Index num_reducers =
        (inner_most_dim_preserved && packet_reductions_enabled) ?
        (output_shard_size / packet_size + output_shard_size % packet_size +
         packet_size) : output_shard_size;
    typedef internal::BlockReducer<Self, Op> BlockReducer;
    BlockReducer* reducers = static_cast<BlockReducer*>(
        m_device.allocate(num_reducers * sizeof(BlockReducer)));

    InputDimensions input_tensor_dims(m_impl.dimensions());
    for (Index output_outer_index = 0;
         output_outer_index < output_block_outer_dim_size;
         ++output_outer_index) {
      for (Index output_shard_index = 0;
           output_shard_index < num_output_shards;
           ++output_shard_index) {
        // Initialize 'tensor_slice_extents' for this output shard.
        DSizes<Index, NumInputDims> tensor_slice_extents(input_slice_sizes);
        for (int i = 0; i < NumInputDims; ++i) {
          if (i == first_preserved_dim_input_index) {
            // Clip first preserved dim size to output shard size.
            tensor_slice_extents[i] = numext::mini(
                output_shard_size,
                input_slice_sizes[i] - (tensor_slice_offsets[i] -
                                        first_preserved_dim_offset_start));

          } else if (!m_reduced_dim[i]) {
            // Clip outer preserved dims to size 1, so that we reduce a
            // contiguous set of output coefficients.
            tensor_slice_extents[i] = 1;
          }
        }

        // Intialize output coefficient reducers.
        for (int i = 0; i < num_reducers; ++i) {
          new (&reducers[i]) BlockReducer(m_reducer);
        }

        typedef internal::TensorSliceBlockMapper<
          Index, ScalarNonConst, NumInputDims, Layout> TensorSliceBlockMapper;

        // TODO(andydavis) Consider removing 'input_block_stride_order' if we
        // find that scattered reads are not worth supporting in
        // TensorSliceBlockMapper.
        TensorSliceBlockMapper block_mapper(
            input_tensor_dims, tensor_slice_offsets, tensor_slice_extents,
            target_input_block_sizes, DimensionList<Index, NumInputDims>());

        const Index num_outputs_to_update = tensor_slice_extents[
            first_preserved_dim_input_index];
        const Index preserved_dim_vector_reducer_count =
            (inner_most_dim_preserved && packet_reductions_enabled) ?
            num_outputs_to_update / packet_size: 0;
        const Index preserved_dim_vector_coeff_count =
            inner_most_dim_preserved ? preserved_dim_vector_reducer_count *
            packet_size : 0;
        const Index preserved_dim_reducer_limit =
            (inner_most_dim_preserved && packet_reductions_enabled) ?
          (preserved_dim_vector_reducer_count +
           num_outputs_to_update % packet_size) : num_outputs_to_update;

        const Index total_block_count = block_mapper.total_block_count();
        for (Index b = 0; b < total_block_count; ++b) {
          InputTensorBlock input_block = block_mapper.GetBlockForIndex(
              b, input_block_data);
          // Read.
          m_impl.block(&input_block);

          Index num_values_to_reduce = 1;
          for (Index i = 0; i < NumInputDims; ++i) {
            if (m_reduced_dim[i]) {
              num_values_to_reduce *= input_block.block_sizes()[i];
            }
          }
          // Reduce.
          if (inner_most_dim_preserved) {
            const Index input_outer_dim_size =
                input_block.block_sizes().TotalSize() / num_outputs_to_update;
            for (Index input_outer_dim_index = 0;
                 input_outer_dim_index < input_outer_dim_size;
                 ++input_outer_dim_index) {
              const Index input_outer_dim_base = input_outer_dim_index *
                  num_outputs_to_update;
              for (Index i = 0; i < preserved_dim_vector_reducer_count; ++i) {
                reducers[i].Reduce(input_outer_dim_base + i * packet_size,
                                   packet_size, input_block.data());
              }
              const Index scalar_reducer_base = input_outer_dim_base +
                  preserved_dim_vector_coeff_count;
              for (Index i = preserved_dim_vector_reducer_count;
                   i < preserved_dim_reducer_limit; ++i) {
                reducers[i].Reduce(scalar_reducer_base + i -
                                   preserved_dim_vector_reducer_count,
                                   1,
                                   input_block.data());
              }
            }
          } else {
            for (Index i = 0; i < num_outputs_to_update; ++i) {
              reducers[i].Reduce(i * num_values_to_reduce,
                                 num_values_to_reduce,
                                 input_block.data());
            }
          }
        }

        // Finalize all reducers for this output shard.
        const Index output_base_index =
            output_outer_index * output_block_inner_dim_size +
            output_shard_index * output_shard_size;
        if (inner_most_dim_preserved) {
          EIGEN_ALIGN_DEFAULT CoeffReturnType values[packet_size];
          for (Index i = 0; i < preserved_dim_vector_reducer_count; ++i) {
            const Index reducer_base = output_base_index + i * packet_size;
            internal::pstore<CoeffReturnType, PacketReturnType>(
                values, reducers[i].FinalizePacket());
            for (Index j = 0; j < packet_size; ++j) {
              output_block->data()[reducer_base + j] = values[j];
            }
          }
          const Index scalar_reducer_base = output_base_index +
              preserved_dim_vector_coeff_count;

          for (Index i = preserved_dim_vector_reducer_count;
               i < preserved_dim_reducer_limit; ++i) {
            output_block->data()[
                scalar_reducer_base + i - preserved_dim_vector_reducer_count] =
                reducers[i].Finalize();
          }
        } else {
          for (int i = 0; i < num_outputs_to_update; ++i) {
            output_block->data()[output_base_index + i] =
                reducers[i].Finalize();
          }
        }

        // Update 'tensor_slice_offsets' by num outputs for this output shard.
        tensor_slice_offsets[first_preserved_dim_input_index] +=
            num_outputs_to_update;
      }
      // Update slice offset for inner preserved dim.
      tensor_slice_offsets[first_preserved_dim_input_index] -=
          output_block_inner_dim_size;
      // Update slice offsets for remaining output dims.
      for (int i = 0; i < NumOutputDims - 1; ++i) {
        BlockIteratorState& b = block_iter_state[i];
        if (++b.output_count < b.output_size) {
          ++tensor_slice_offsets[b.input_dim];
          break;
        }
        b.output_count = 0;
        tensor_slice_offsets[b.input_dim] -= b.output_size - 1;
      }
    }

    // Free memory.
    m_device.deallocate(input_block_data);
    m_device.deallocate(reducers);
  }

  EIGEN_DEVICE_FUNC Scalar* data() const { return NULL; }

  private:
  template <int, typename, typename> friend struct internal::GenericDimReducer;
  template <typename, typename, bool> friend struct internal::InnerMostDimReducer;
  template <int, typename, typename, bool> friend struct internal::InnerMostDimPreserver;
  template <typename S, typename O, typename D, bool V> friend struct internal::FullReducer;
#ifdef EIGEN_USE_THREADS
  template <typename S, typename O, bool V> friend struct internal::FullReducerShard;
#endif
#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
  template <int B, int N, typename S, typename R, typename I> friend void internal::FullReductionKernel(R, const S, I, typename S::CoeffReturnType*);
#endif

  struct BlockIteratorState {
    Index input_dim;
    Index output_size;
    Index output_count;
  };

  // Returns the Index in the input tensor of the first value that needs to be
  // used to compute the reduction at output index "index".
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index firstInput(Index index) const {
    if (ReducingInnerMostDims) {
      return index * m_numValuesToReduce;
    }
    Index startInput = 0;
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      for (int i = NumOutputDims - 1; i > 0; --i) {
        // This is index_i in the output tensor.
        const Index idx = index / m_fastOutputStrides[i];
        startInput += idx * m_preservedStrides[i];
        index -= idx * m_outputStrides[i];
      }
    } else {
      for (int i = 0; i < NumOutputDims - 1; ++i) {
        // This is index_i in the output tensor.
        const Index idx = index / m_fastOutputStrides[i];
        startInput += idx * m_preservedStrides[i];
        index -= idx * m_outputStrides[i];
      }
    }
    if (PreservingInnerMostDims) {
      eigen_assert(m_numValuesToReduce == 1);
      startInput += index;
    } else {
      startInput += index * m_numValuesToReduce;
    }
    return startInput;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void GetInputCoordsForOutputIndex(
      Index index,
      DSizes<Index, NumInputDims>* coords) const {
    for (int i = 0; i < NumInputDims; ++i) {
      (*coords)[i] = 0;
    }
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      for (int i = NumOutputDims - 1; i > 0; --i) {
        const Index idx = index / m_fastOutputStrides[i];
        (*coords)[m_output_to_input_dim_map[i]] = idx;
        index -= idx * m_outputStrides[i];
      }
      (*coords)[m_output_to_input_dim_map[0]] = index;
    } else {
      for (int i = 0; i < NumOutputDims - 1; ++i) {
        const Index idx = index / m_fastOutputStrides[i];
        (*coords)[m_output_to_input_dim_map[i]] = idx;
        index -= idx * m_outputStrides[i];
      }
      (*coords)[m_output_to_input_dim_map[NumOutputDims-1]] = index;
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void CalculateTargetInputBlockShape(
      const Index max_coeff_count,
      const DSizes<Index, NumInputDims>& input_slice_sizes,
      DSizes<Index, NumInputDims>* target_input_block_sizes) const {
    typedef typename internal::packet_traits<Scalar>::type Packet;
    const Index packet_size = internal::unpacket_traits<Packet>::size;
    typedef internal::BlockReducer<Self, Op> BlockReducer;
    // TODO(andydavis) Compute reducer overhead correctly for the case where
    // we are preserving the inner most dimension, and a single reducer
    // reduces a packet's worth of output coefficients.
    const Index reducer_overhead = sizeof(BlockReducer) / sizeof(Scalar);

    Index coeff_to_allocate = max_coeff_count;
    bool first_preserved_dim_allocated = false;
    bool first_reduced_dim_allocated = false;
    for (int i = 0; i < NumInputDims; ++i) {
      const int dim = static_cast<int>(Layout) == static_cast<int>(ColMajor)
          ? i : NumInputDims - i - 1;
      (*target_input_block_sizes)[dim] = 1;
      if (m_reduced_dim[dim]) {
        // TODO(andydavis) Consider allocating to multiple reduced dimensions.
        // Watch out for cases where reduced dimensions are not contiguous,
        // which induces scattered reads.
        if (!first_reduced_dim_allocated) {
          (*target_input_block_sizes)[dim] = numext::mini(input_slice_sizes[dim],
                                                        coeff_to_allocate);
          coeff_to_allocate /= (*target_input_block_sizes)[dim];
          first_reduced_dim_allocated = true;
        }
      } else if (!first_preserved_dim_allocated) {
        // TODO(andydavis) Include output block size in this L1 working set
        // calculation.
        const Index allocated = max_coeff_count - coeff_to_allocate;
        const Index alloc_size = numext::maxi(static_cast<Index>(1),
                                            coeff_to_allocate /
                                            reducer_overhead);
        (*target_input_block_sizes)[dim] = numext::mini(input_slice_sizes[dim],
                                                      alloc_size);
        coeff_to_allocate = numext::maxi(
            static_cast<Index>(1),
            coeff_to_allocate / ((*target_input_block_sizes)[dim] *
                                 reducer_overhead));
        first_preserved_dim_allocated = true;
      }
    }
  }

  // Bitmap indicating if an input dimension is reduced or not.
  array<bool, NumInputDims> m_reduced_dim;
  // Dimensions of the output of the operation.
  Dimensions m_dimensions;
  // Precomputed strides for the input tensor.
  array<Index, NumInputDims> m_inputStrides;
  // Precomputed strides for the output tensor.
  array<Index, NumOutputDims> m_outputStrides;
  array<internal::TensorIntDivisor<Index>, NumOutputDims> m_fastOutputStrides;
  // Subset of strides of the input tensor for the non-reduced dimensions.
  // Indexed by output dimensions.
  array<Index, NumOutputDims> m_preservedStrides;
  // Map from output to input dimension index.
  array<Index, NumOutputDims> m_output_to_input_dim_map;
  // How many values go into each reduction
  Index m_numValuesToReduce;

  // Subset of strides of the input tensor for the reduced dimensions.
  // Indexed by reduced dimensions.
  array<Index, NumReducedDims> m_reducedStrides;
  // Size of the input dimensions that are reduced.
  // Indexed by reduced dimensions.
  array<Index, NumReducedDims> m_reducedDims;

  // Evaluator for the input expression.
  TensorEvaluator<ArgType, Device> m_impl;

  // Operation to apply for computing the reduction.
  Op m_reducer;

  // For full reductions
#ifdef EIGEN_USE_GPU
  static const bool RunningOnGPU = internal::is_same<Device, Eigen::GpuDevice>::value;
#else
  static const bool RunningOnGPU = false;
#endif
  CoeffReturnType* m_result;
  std::size_t m_block_total_size_max;

  const Device& m_device;
};

} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_H
