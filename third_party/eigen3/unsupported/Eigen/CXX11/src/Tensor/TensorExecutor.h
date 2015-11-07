// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_EXECUTOR_H
#define EIGEN_CXX11_TENSOR_TENSOR_EXECUTOR_H

namespace Eigen {

/** \class TensorExecutor
  * \ingroup CXX11_Tensor_Module
  *
  * \brief The tensor executor class.
  *
  * This class is responsible for launch the evaluation of the expression on
  * the specified computing device.
  */
namespace internal {

// Default strategy: the expression is evaluated with a single cpu thread.
template <typename Expression, typename Device,
          bool Vectorizable, bool Tileable>
class TensorExecutor {
 public:
  typedef typename Expression::Index Index;
  EIGEN_DEVICE_FUNC static inline void run(const Expression& expr, const Device& device = Device())
  {
    TensorEvaluator<Expression, Device> evaluator(expr, device);
    const bool needs_assign = evaluator.evalSubExprsIfNeeded(NULL);
    if (needs_assign)
    {
      const Index size = array_prod(evaluator.dimensions());
      for (Index i = 0; i < size; ++i) {
        evaluator.evalScalar(i);
      }
    }
    evaluator.cleanup();
  }
};

template <typename Expression>
class TensorExecutor<Expression, DefaultDevice, true, false> {
 public:
  typedef typename Expression::Index Index;
  EIGEN_DEVICE_FUNC
  static inline void run(const Expression& expr, const DefaultDevice& device = DefaultDevice())
  {
    TensorEvaluator<Expression, DefaultDevice> evaluator(expr, device);
    const bool needs_assign = evaluator.evalSubExprsIfNeeded(NULL);
    if (needs_assign)
    {
      const Index size = array_prod(evaluator.dimensions());
      const int PacketSize = unpacket_traits<typename TensorEvaluator<Expression, DefaultDevice>::PacketReturnType>::size;

      // Manually unroll this loop since compilers don't do it.
      const Index UnrolledSize = (size / (4 * PacketSize)) * 4 * PacketSize;
      for (Index i = 0; i < UnrolledSize; i += 4*PacketSize) {
        evaluator.evalPacket(i);
        evaluator.evalPacket(i+PacketSize);
        evaluator.evalPacket(i+2*PacketSize);
        evaluator.evalPacket(i+3*PacketSize);
      }
      const Index VectorizedSize = (size / PacketSize) * PacketSize;
      for (Index i = UnrolledSize; i < VectorizedSize; i += PacketSize) {
        evaluator.evalPacket(i);
      }
      for (Index i = VectorizedSize; i < size; ++i) {
        evaluator.evalScalar(i);
      }
    }
    evaluator.cleanup();
  }
};

template <typename Expression, bool Vectorizable>
class TensorExecutor<Expression, DefaultDevice, Vectorizable, true> {
 public:
  typedef typename Expression::Index Index;
  EIGEN_DEVICE_FUNC
  static inline void run(const Expression& expr,
                         const DefaultDevice& device = DefaultDevice()) {
    typedef TensorEvaluator<Expression, DefaultDevice> Evaluator;
    typedef typename traits<Expression>::Scalar Scalar;
    typedef typename traits<Expression>::Index Index;
    const std::size_t NumDims = traits<Expression>::NumDimensions;

    typedef TensorBlockMapper<Index,
                              typename internal::remove_const<Scalar>::type,
                              NumDims, Evaluator::Layout> TensorBlockMapper;
    typedef TensorBlock<Index, typename internal::remove_const<Scalar>::type,
                        NumDims, Evaluator::Layout> TensorBlock;

    Evaluator evaluator(expr, device);
    std::size_t total_size = array_prod(evaluator.dimensions());
    std::size_t cache_size = device.firstLevelCacheSize() / sizeof(Scalar);
    if (total_size < cache_size) {
      // TODO(andydavis) Reduce block management overhead for small tensors.
      internal::TensorExecutor<Expression, DefaultDevice, Vectorizable,
                               false>::run(expr, device);
      return;
    }

    const bool needs_assign = evaluator.evalSubExprsIfNeeded(NULL);
    if (needs_assign) {
      // Size tensor blocks to fit in cache (or requested target block size).
      size_t block_total_size = numext::mini(cache_size, total_size);
      TensorBlockShapeType block_shape = kUniformAllDims;
      // Query expression tree for desired block size/shape.
      std::vector<internal::TensorOpResourceRequirements> resources;
      evaluator.getResourceRequirements(&resources);
      if (!resources.empty()) {
        // TODO(andydavis) Implement different policies (i.e. revert to a
        // default policy if block shapes/sizes conflict).
        block_shape = resources[0].block_shape;
        block_total_size = resources[0].block_total_size;
      }

      TensorBlockMapper block_mapper(evaluator.dimensions(),
                                     block_shape,
                                     block_total_size);

      Scalar* data = static_cast<Scalar*>(device.allocate(
          block_total_size * sizeof(Scalar)));

      const Index total_block_count = block_mapper.total_block_count();
      for (Index i = 0; i < total_block_count; ++i) {
        TensorBlock block = block_mapper.GetBlockForIndex(i, data);
        evaluator.evalBlock(&block);
      }
      device.deallocate(data);
    }
    evaluator.cleanup();
  }
};

// Multicore strategy: the index space is partitioned and each partition is executed on a single core
#ifdef EIGEN_USE_THREADS
template <typename Evaluator, typename Index, bool Vectorizable>
struct EvalRange {
  static void run(Evaluator evaluator, const Index first, const Index last) {
    eigen_assert(last > first);
    for (Index i = first; i < last; ++i) {
      evaluator.evalScalar(i);
    }
  }
};

template <typename Evaluator, typename Index>
struct EvalRange<Evaluator, Index, true> {
  static void run(Evaluator evaluator, const Index first, const Index last) {
    eigen_assert(last > first);

    Index i = first;
    static const int PacketSize = unpacket_traits<typename Evaluator::PacketReturnType>::size;
    if (last - first >= PacketSize) {
      eigen_assert(first % PacketSize == 0);
      Index lastPacket = last - (last % PacketSize);
      for (; i < lastPacket; i += PacketSize) {
        evaluator.evalPacket(i);
      }
    }

    for (; i < last; ++i) {
      evaluator.evalScalar(i);
    }
  }
};

template <typename Expression, bool Vectorizable, bool Tileable>
class TensorExecutor<Expression, ThreadPoolDevice, Vectorizable, Tileable> {
 public:
  typedef typename Expression::Index Index;
  static inline void run(const Expression& expr, const ThreadPoolDevice& device)
  {
    if (device.numThreads() <= 1) {
      DefaultDevice dd;
      TensorExecutor<Expression, DefaultDevice, Vectorizable, Tileable>::run(expr, dd);
      return;
    }

    typedef TensorEvaluator<Expression, ThreadPoolDevice> Evaluator;
    Evaluator evaluator(expr, device);
    const bool needs_assign = evaluator.evalSubExprsIfNeeded(NULL);
    if (needs_assign)
    {
      const Index size = array_prod(evaluator.dimensions());

      static const Index PacketSize = Vectorizable ? unpacket_traits<typename Evaluator::PacketReturnType>::size : 1;
      Index blocksz = std::ceil<Index>(static_cast<float>(size)/device.numThreads()) + PacketSize - 1;
      const Index blocksize = numext::maxi<Index>(PacketSize, (blocksz - (blocksz % PacketSize)));
      const Index numblocks = size / blocksize;

      Index i = 0;
      FixedSizeVector<Notification*> results(numblocks);
      for (int i = 0; i < numblocks; ++i) {
        results.push_back(device.enqueue(&EvalRange<Evaluator, Index, Vectorizable>::run, evaluator, i*blocksize, (i+1)*blocksize));
      }

      if (numblocks * blocksize < size) {
        EvalRange<Evaluator, Index, Vectorizable>::run(evaluator, numblocks * blocksize, size);
      }

      for (int i = 0; i < numblocks; ++i) {
        wait_until_ready(results[i]);
        delete results[i];
      }
    }
    evaluator.cleanup();
  }
};

template <typename Index, typename Scalar>
struct BlockRange {
  BlockRange(Index s, Index l, Scalar* d)
      : index_start(s), index_limit(l), data(d) {}
  const Index index_start;
  const Index index_limit;
  Scalar* data;
};

template <typename Evaluator, typename Index, typename Scalar,
          std::size_t NumDims>
struct EvalBlockRange {
  typedef TensorBlockMapper<Index, Scalar, NumDims, Evaluator::Layout>
      BlockMapper;

  static void run(Evaluator evaluator, const BlockMapper& block_mapper,
                  BlockRange<Index, Scalar> block_range) {
    typedef TensorBlock<Index, Scalar, NumDims, Evaluator::Layout>
        TensorBlock;
    eigen_assert(block_range.index_limit > block_range.index_start);

    for (Index i = block_range.index_start; i < block_range.index_limit; ++i) {
      TensorBlock block = block_mapper.GetBlockForIndex(i, block_range.data);
      evaluator.evalBlock(&block);
    }
  }
};

template <typename Expression, bool Vectorizable>
class TensorExecutor<Expression, ThreadPoolDevice, Vectorizable, true> {
 public:
  typedef typename Expression::Index Index;
  static inline void run(const Expression& expr,
                         const ThreadPoolDevice& device) {
    typedef TensorEvaluator<Expression, ThreadPoolDevice> Evaluator;
    typedef typename internal::remove_const<
        typename traits<Expression>::Scalar>::type Scalar;
    typedef typename traits<Expression>::Index Index;
    static const std::size_t NumDims = traits<Expression>::NumDimensions;
    typedef TensorBlockMapper<Index, Scalar, NumDims, Evaluator::Layout>
        TensorBlockMapper;
    typedef TensorBlock<Index, Scalar, NumDims, Evaluator::Layout>
        TensorBlock;
    typedef BlockRange<Index, Scalar> BlockRange;

    Evaluator evaluator(expr, device);
    std::size_t total_size = array_prod(evaluator.dimensions());
    std::size_t cache_size = device.firstLevelCacheSize() / sizeof(Scalar);
    if (total_size < cache_size || device.numThreads() <= 1) {
      // TODO(andydavis) Reduce block management overhead for small tensors.
      DefaultDevice dd;
      internal::TensorExecutor<Expression, DefaultDevice, Vectorizable, false>::run(expr, dd);
      return;
    }
    const bool needs_assign = evaluator.evalSubExprsIfNeeded(NULL);
    if (needs_assign) {
      TensorBlockShapeType block_shape = kUniformAllDims;
      size_t block_total_size = 0;
      // Query expression tree for desired block size/shape.
      std::vector<internal::TensorOpResourceRequirements> resources;
      evaluator.getResourceRequirements(&resources);
      if (!resources.empty()) {
        // TODO(andydavis) Implement different shape/size policies.
        block_shape = resources[0].block_shape;
        block_total_size = resources[0].block_total_size;
      }

      // Divide the tensor coefficients across the number of threads, subject
      // to min/max block size constraints.
      const size_t min_block_size =
          device.firstLevelCacheSize() / sizeof(Scalar);
      const size_t max_block_size = block_total_size > 0 ? block_total_size :
          device.lastLevelCacheSize() / sizeof(Scalar);
      const size_t target_block_size = numext::maxi(
          min_block_size,
          numext::mini(static_cast<size_t>(array_prod(evaluator.dimensions())) / device.numThreads(),
                       max_block_size));

      TensorBlockMapper block_mapper(evaluator.dimensions(),
                                     block_shape,
                                     target_block_size);

      const Index block_partition_size =
          (block_mapper.total_block_count() + device.numThreads() - 1) /
          device.numThreads();
      const Index block_partition_count =
          (block_mapper.total_block_count() + block_partition_size - 1) /
          block_partition_size;

      if (block_partition_count == 1) {
        // Avoid thread hop if no parallelism is possible.
        Scalar* data = static_cast<Scalar*>(
            device.allocate(target_block_size * sizeof(Scalar)));
        EvalBlockRange<Evaluator, Index, Scalar, NumDims>::run(
            evaluator, block_mapper,
            BlockRange(0, block_mapper.total_block_count(), data));
        device.deallocate(data);
      } else {
        // Multi-threaded case.
        struct ThreadState {
          Notification* done;
          Scalar* data;
        };
        FixedSizeVector<ThreadState> thread_state(block_partition_count,
                                                  ThreadState());

        // Dispatch threads.
        for (int i = 0; i < block_partition_count; ++i) {
          thread_state[i].data = static_cast<Scalar*>(
              device.allocate(target_block_size * sizeof(Scalar)));
          thread_state[i].done = device.enqueue(
              &EvalBlockRange<Evaluator, Index, Scalar, NumDims>::run,
              evaluator, block_mapper,
              BlockRange(i * block_partition_size,
                         numext::mini((i + 1) * block_partition_size,
                                    block_mapper.total_block_count()),
                         thread_state[i].data));
        }

        // Join threads.
        for (int i = 0; i < block_partition_count; ++i) {
          wait_until_ready(thread_state[i].done);
          delete thread_state[i].done;
          device.deallocate(thread_state[i].data);
        }
      }
    }
    evaluator.cleanup();
  }
};

#endif


// GPU: the evaluation of the expression is offloaded to a GPU.
#if defined(EIGEN_USE_GPU)

template <typename Expression, bool Tileable>
class TensorExecutor<Expression, GpuDevice, false, Tileable> {
 public:
  typedef typename Expression::Index Index;
  static void run(const Expression& expr, const GpuDevice& device);
};

template <typename Expression, bool Tileable>
class TensorExecutor<Expression, GpuDevice, true, Tileable> {
 public:
  typedef typename Expression::Index Index;
  static void run(const Expression& expr, const GpuDevice& device);
};

#if defined(__CUDACC__)
template <typename Evaluator, typename Index>
__global__ void
__launch_bounds__(1024)
 EigenMetaKernel_NonVectorizable(Evaluator memcopied_eval, Index size) {

  const Index first_index = blockIdx.x * blockDim.x + threadIdx.x;
  const Index step_size = blockDim.x * gridDim.x;

  // Cuda memcopies the kernel arguments. That's fine for POD, but for more
  // complex types such as evaluators we should really conform to the C++
  // standard and call a proper copy constructor.
  Evaluator eval(memcopied_eval);

  // Use the scalar path
  for (Index i = first_index; i < size; i += step_size) {
    eval.evalScalar(i);
  }
}

template <typename Evaluator, typename Index>
__global__ void
__launch_bounds__(1024)
 EigenMetaKernel_Vectorizable(Evaluator memcopied_eval, Index size) {

  const Index first_index = blockIdx.x * blockDim.x + threadIdx.x;
  const Index step_size = blockDim.x * gridDim.x;

  // Cuda memcopies the kernel arguments. That's fine for POD, but for more
  // complex types such as evaluators we should really conform to the C++
  // standard and call a proper copy constructor.
  Evaluator eval(memcopied_eval);

  // Use the vector path
  const Index PacketSize = unpacket_traits<typename Evaluator::PacketReturnType>::size;
  const Index vectorized_step_size = step_size * PacketSize;
  const Index vectorized_size = (size / PacketSize) * PacketSize;
  for (Index i = first_index * PacketSize; i < vectorized_size;
       i += vectorized_step_size) {
    eval.evalPacket(i);
  }
  for (Index i = vectorized_size + first_index; i < size; i += step_size) {
    eval.evalScalar(i);
  }
}

/*static*/
template <typename Expression, bool Tileable>
inline void TensorExecutor<Expression, GpuDevice, false, Tileable>::run(
    const Expression& expr, const GpuDevice& device) {
  TensorEvaluator<Expression, GpuDevice> evaluator(expr, device);
  const bool needs_assign = evaluator.evalSubExprsIfNeeded(NULL);
  if (needs_assign) {
    const int num_blocks = device.getNumCudaMultiProcessors() *
                           device.maxCudaThreadsPerMultiProcessor() /
                           device.maxCudaThreadsPerBlock();
    const int block_size = device.maxCudaThreadsPerBlock();
    const Index size = array_prod(evaluator.dimensions());
    LAUNCH_CUDA_KERNEL(
        (EigenMetaKernel_NonVectorizable<TensorEvaluator<Expression, GpuDevice>,
                                         Index>),
        num_blocks, block_size, 0, device, evaluator, size);
  }
  evaluator.cleanup();
}

/*static*/
template <typename Expression, bool Tileable>
inline void TensorExecutor<Expression, GpuDevice, true, Tileable>::run(
    const Expression& expr, const GpuDevice& device) {
  TensorEvaluator<Expression, GpuDevice> evaluator(expr, device);
  const bool needs_assign = evaluator.evalSubExprsIfNeeded(NULL);
  if (needs_assign) {
    const int num_blocks = device.getNumCudaMultiProcessors() *
                           device.maxCudaThreadsPerMultiProcessor() /
                           device.maxCudaThreadsPerBlock();
    const int block_size = device.maxCudaThreadsPerBlock();
    const Index size = array_prod(evaluator.dimensions());
    LAUNCH_CUDA_KERNEL(
        (EigenMetaKernel_Vectorizable<TensorEvaluator<Expression, GpuDevice>,
                                      Index>),
        num_blocks, block_size, 0, device, evaluator, size);
  }
  evaluator.cleanup();
}

#endif  // __CUDACC__
#endif  // EIGEN_USE_GPU

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_EXECUTOR_H
