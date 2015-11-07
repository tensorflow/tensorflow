#ifndef EIGEN_CXX11_TENSOR_TENSOR_BLOCK_H
#define EIGEN_CXX11_TENSOR_TENSOR_BLOCK_H

namespace Eigen {

/** \class TensorBlock
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Tensor block class.
  *
  * This class represents a tensor block specified by the index of the
  * first block coefficient, and the size of the block in each dimension.
  *
  */

namespace internal {

template <typename Index, typename Scalar, std::size_t NumDims, int Layout>
class TensorBlock {
 public:
  typedef DSizes<Index, NumDims> Dimensions;

  TensorBlock(const Index first_coeff_index,
              const Dimensions& block_sizes,
              const Dimensions& block_strides,
              const Dimensions& tensor_strides,
              Scalar* data)
      : m_first_coeff_index(first_coeff_index),
        m_block_sizes(block_sizes),
        m_block_strides(block_strides),
        m_tensor_strides(tensor_strides),
        m_data(data) {}

  Index first_coeff_index() const { return m_first_coeff_index; }

  const Dimensions& block_sizes() const { return m_block_sizes; }

  const Dimensions& block_strides() const { return m_block_strides; }

  const Dimensions& tensor_strides() const { return m_tensor_strides; }

  Scalar* data() { return m_data; }

  const Scalar* data() const { return m_data; }

 private:
  Index m_first_coeff_index;
  Dimensions m_block_sizes;
  Dimensions m_block_strides;
  Dimensions m_tensor_strides;
  Scalar* m_data;  // Not owned.
};

template <typename Index, typename Scalar, bool Vectorizable>
struct TensorBlockCopyOp {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void Run(
      const Index num_coeff_to_copy, const Index dst_index,
      const Index dst_stride, Scalar* EIGEN_RESTRICT dst_data, const Index src_index,
      const Index src_stride, const Scalar* EIGEN_RESTRICT src_data) {
    for (Index i = 0; i < num_coeff_to_copy; ++i) {
      dst_data[dst_index + i * dst_stride] =
          src_data[src_index + i * src_stride];
    }
  }
};

// NOTE: Benchmarks run on an implementation of this that broke each of the
// loops in these conditionals into it's own template specialization (to
// avoid conditionals in the caller's loop) did not show an improvement.
template <typename Index, typename Scalar>
struct TensorBlockCopyOp<Index, Scalar, true> {
  typedef typename packet_traits<Scalar>::type Packet;
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void Run(
      const Index num_coeff_to_copy, const Index dst_index,
      const Index dst_stride, Scalar* EIGEN_RESTRICT dst_data,
      const Index src_index, const Index src_stride,
      const Scalar* EIGEN_RESTRICT src_data) {
    if (src_stride == 1) {
      const Index packet_size = internal::unpacket_traits<Packet>::size;
      const Index vectorized_size =
          (num_coeff_to_copy / packet_size) * packet_size;
      if (dst_stride == 1) {
        // LINEAR
        for (Index i = 0; i < vectorized_size; i += packet_size) {
          Packet p = internal::ploadt<Packet, Unaligned>(
              src_data + src_index + i);
          internal::pstoret<Scalar, Packet, Unaligned>(
              dst_data + dst_index + i, p);
        }
        for (Index i = vectorized_size; i < num_coeff_to_copy; ++i) {
          dst_data[dst_index + i] = src_data[src_index + i];
        }
      } else {
        // SCATTER
        for (Index i = 0; i < vectorized_size; i += packet_size) {
          Packet p = internal::ploadt<Packet, Unaligned>(
              src_data + src_index + i);
          internal::pscatter<Scalar, Packet>(
              dst_data + dst_index + i * dst_stride, p, dst_stride);
        }
        for (Index i = vectorized_size; i < num_coeff_to_copy; ++i) {
          dst_data[dst_index + i * dst_stride] = src_data[src_index + i];
        }
      }
    } else {
      if (dst_stride == 1) {
        // GATHER
        const Index packet_size = internal::unpacket_traits<Packet>::size;
        const Index vectorized_size =
            (num_coeff_to_copy / packet_size) * packet_size;
        for (Index i = 0; i < vectorized_size; i += packet_size) {
          Packet p = internal::pgather<Scalar, Packet>(
              src_data + src_index + i * src_stride, src_stride);
          internal::pstoret<Scalar, Packet, Unaligned>(
              dst_data + dst_index + i, p);
        }
        for (Index i = vectorized_size; i < num_coeff_to_copy; ++i) {
          dst_data[dst_index + i] = src_data[src_index + i * src_stride];
        }
      } else {
        // RANDOM
        for (Index i = 0; i < num_coeff_to_copy; ++i) {
          dst_data[dst_index + i * dst_stride] =
              src_data[src_index + i * src_stride];
        }
      }
    }
  }
};

/** \class TensorBlockIO
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Tensor block IO class.
  *
  * This class is responsible for copying data between a tensor and a tensor
  * block.
  *
  */
template <typename Index, typename Scalar, std::size_t NumDims, int Layout,
          bool Vectorizable, bool BlockRead>
class TensorBlockIO {
 public:
  typedef typename internal::TensorBlock<Index, Scalar, NumDims, Layout>
    TensorBlock;
  typedef typename internal::TensorBlockCopyOp<Index, Scalar, Vectorizable>
    TensorBlockCopyOp;

 protected:
  struct BlockIteratorState {
    Index input_stride;
    Index output_stride;
    Index input_span;
    Index output_span;
    Index size;
    Index count;
  };

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void Copy(
      const TensorBlock& block, Index first_coeff_index,
      const array<Index, NumDims>& tensor_to_block_dim_map,
      const array<Index, NumDims>& tensor_strides, const Scalar* src_data,
      Scalar* dst_data) {
    // Calculate strides and dimensions.
    const Index block_dim_for_tensor_stride1_dim =
        NumDims == 0 ? 1 :
        tensor_to_block_dim_map[static_cast<int>(Layout) ==
                                        static_cast<int>(ColMajor)
                                    ? 0
                                    : NumDims - 1];
    const size_t block_inner_dim_size =
        NumDims == 0 ? 1 :
        block.block_sizes()[block_dim_for_tensor_stride1_dim];
    const size_t block_outer_dim_size =
        NumDims == 0 ? 1 :
        block.block_sizes().TotalSize() / block_inner_dim_size;

    Index inputIndex;
    Index outputIndex;
    Index input_stride;
    Index output_stride;

    // Setup strides to read/write along the tensor's stride1 dimension.
    if (BlockRead) {
      inputIndex = first_coeff_index;
      outputIndex = 0;
      input_stride = 1;
      output_stride = NumDims == 0 ? 1
          : block.block_strides()[block_dim_for_tensor_stride1_dim];
    } else {
      inputIndex = 0;
      outputIndex = first_coeff_index;
      input_stride = NumDims == 0 ? 1
          : block.block_strides()[block_dim_for_tensor_stride1_dim];
      output_stride = 1;
    }

    const std::size_t at_least_1_dim = NumDims <= 1 ? 1 : NumDims - 1;
    array<BlockIteratorState, at_least_1_dim> block_iter_state;

    // Initialize block iterator state.
    for (int i = 0; i < static_cast<int>(NumDims) - 1; ++i) {
      const int dim = static_cast<int>(Layout) == static_cast<int>(ColMajor)
                          ? i + 1
                          : NumDims - i - 2;
      block_iter_state[i].size =
          block.block_sizes()[tensor_to_block_dim_map[dim]];
      if (BlockRead) {
        block_iter_state[i].input_stride = tensor_strides[dim];
        block_iter_state[i].output_stride =
            block.block_strides()[tensor_to_block_dim_map[dim]];
      } else {
        block_iter_state[i].input_stride =
            block.block_strides()[tensor_to_block_dim_map[dim]];
        block_iter_state[i].output_stride = tensor_strides[dim];
      }
      block_iter_state[i].input_span =
          block_iter_state[i].input_stride * (block_iter_state[i].size - 1);
      block_iter_state[i].output_span =
          block_iter_state[i].output_stride * (block_iter_state[i].size - 1);
      block_iter_state[i].count = 0;
    }

    // Iterate copying data from src to dst.
    for (Index i = 0; i < block_outer_dim_size; ++i) {
      TensorBlockCopyOp::Run(block_inner_dim_size, outputIndex, output_stride,
                             dst_data, inputIndex, input_stride, src_data);
      // Update index.
      for (int i = 0; i < static_cast<int>(NumDims) - 1; ++i) {
        if (++block_iter_state[i].count < block_iter_state[i].size) {
          inputIndex += block_iter_state[i].input_stride;
          outputIndex += block_iter_state[i].output_stride;
          break;
        }
        block_iter_state[i].count = 0;
        inputIndex -= block_iter_state[i].input_span;
        outputIndex -= block_iter_state[i].output_span;
      }
    }
  }
};

/** \class TensorBlockReader
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Tensor block reader class.
  *
  * This class is responsible for reading a tensor block.
  *
  */

template <typename Index, typename Scalar, std::size_t NumDims, int Layout,
          bool Vectorizable>
class TensorBlockReader : public TensorBlockIO<Index, Scalar, NumDims,
                                               Layout, Vectorizable, true> {
 public:
  typedef typename internal::TensorBlock<Index, Scalar, NumDims, Layout>
      TensorBlock;
  typedef TensorBlockIO<Index, Scalar, NumDims, Layout, Vectorizable, true>
      Base;

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void Run(
      TensorBlock* block, const Scalar* src_data) {
    array<Index, NumDims> tensor_to_block_dim_map;
    for (int i = 0; i < NumDims; ++i) {
      tensor_to_block_dim_map[i] = i;
    }
    Base::Copy(*block, block->first_coeff_index(), tensor_to_block_dim_map,
               block->tensor_strides(), src_data, block->data());
  }

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void Run(
      TensorBlock* block, Index first_coeff_index,
      const array<Index, NumDims>& tensor_to_block_dim_map,
      const array<Index, NumDims>& tensor_strides, const Scalar* src_data) {
    Base::Copy(*block, first_coeff_index, tensor_to_block_dim_map,
               tensor_strides, src_data, block->data());
  }
};

/** \class TensorBlockWriter
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Tensor block writer class.
  *
  * This class is responsible for writing a tensor block.
  *
  */

template <typename Index, typename Scalar, std::size_t NumDims, int Layout,
          bool Vectorizable>
class TensorBlockWriter : public TensorBlockIO<Index, Scalar, NumDims,
                                               Layout, Vectorizable, false> {
 public:
  typedef typename internal::TensorBlock<Index, Scalar, NumDims, Layout>
      TensorBlock;
  typedef TensorBlockIO<Index, Scalar, NumDims, Layout, Vectorizable, false>
      Base;

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void Run(
      const TensorBlock& block, Scalar* dst_data) {
    array<Index, NumDims> tensor_to_block_dim_map;
    for (int i = 0; i < NumDims; ++i) {
      tensor_to_block_dim_map[i] = i;
    }
    Base::Copy(block, block.first_coeff_index(), tensor_to_block_dim_map,
               block.tensor_strides(), block.data(), dst_data);
  }

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void Run(
      const TensorBlock& block, Index first_coeff_index,
      const array<Index, NumDims>& tensor_to_block_dim_map,
      const array<Index, NumDims>& tensor_strides, Scalar* dst_data) {
    Base::Copy(block, first_coeff_index, tensor_to_block_dim_map,
               tensor_strides, block.data(), dst_data);
  }
};

enum TensorBlockShapeType {
  kUniformAllDims,
  kSkewedInnerDims,
};

struct TensorOpResourceRequirements {
  TensorBlockShapeType block_shape;
  std::size_t block_total_size;
  // TODO(andydavis) Add 'target_num_threads' to support communication of
  // thread-resource requirements. This will allow ops deep in the
  // expression tree (like reductions) to communicate resources
  // requirements based on local state (like the total number of reductions
  // to be computed).
  TensorOpResourceRequirements(internal::TensorBlockShapeType shape,
                               const std::size_t size)
      : block_shape(shape), block_total_size(size) {}
};

/** \class TensorBlockMapper
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Tensor block mapper class.
  *
  * This class is responsible for iterating over the blocks of a tensor.
  *
  */

template <typename Index, typename Scalar, std::size_t NumDims, int Layout>
class TensorBlockMapper {
 public:
  typedef typename internal::TensorBlock<Index, Scalar, NumDims, Layout>
      TensorBlock;

  TensorBlockMapper(const Eigen::DSizes<Index, NumDims>& dims,
                    const TensorBlockShapeType block_shape,
                    const size_t max_coeff_count)
      : m_dimensions(dims), m_block_dim_sizes(dims), m_total_block_count(1) {
    if (m_block_dim_sizes.TotalSize() > max_coeff_count) {
      if (block_shape == kUniformAllDims) {
        // Tensor will not fit within 'max_coeff_count' budget: calculate tensor
        // block dimension sizes based on "square" dimension size target.
        const size_t dim_size_target =
            std::pow(static_cast<float>(max_coeff_count),
                     1.0 / static_cast<float>(m_block_dim_sizes.rank()));
        for (size_t i = 0; i < m_block_dim_sizes.rank(); ++i) {
          // TODO(andydavis) Adjust the inner most 'm_block_dim_size' to make it
          // a multiple of the packet size. Note that reducing 'm_block_dim_size'
          // in this manner can increase the number of blocks, and so will
          // amplify any per-block overhead.
          m_block_dim_sizes[i] =
              numext::mini(dim_size_target, static_cast<size_t>(m_dimensions[i]));
        }
        // Add any un-allocated coefficients to inner dimension(s).
        Index total_size = m_block_dim_sizes.TotalSize();
        for (int i = 0; i < NumDims; ++i) {
          const int dim = static_cast<int>(Layout) == static_cast<int>(ColMajor)
              ? i : NumDims - i - 1;
          if (m_block_dim_sizes[dim] < m_dimensions[dim]) {
            const Index total_size_other_dims = total_size /
                m_block_dim_sizes[dim];
            const Index alloc_avail = max_coeff_count / total_size_other_dims;
            if (alloc_avail == m_block_dim_sizes[dim]) {
              // Insufficient excess coefficients to allocate.
              break;
            }
            m_block_dim_sizes[dim] = numext::mini(m_dimensions[dim], alloc_avail);
            total_size = total_size_other_dims * m_block_dim_sizes[dim];
          }
        }
      } else {
        eigen_assert(block_shape == kSkewedInnerDims);
        Index coeff_to_allocate = max_coeff_count;
        for (int i = 0; i < NumDims; ++i) {
          const int dim = static_cast<int>(Layout) == static_cast<int>(ColMajor)
              ? i : NumDims - i - 1;
          m_block_dim_sizes[dim] = numext::mini(coeff_to_allocate,
                                                m_dimensions[dim]);
          coeff_to_allocate /= numext::maxi(static_cast<Index>(1),
                                            m_block_dim_sizes[dim]);
        }
      }
    }

    // Calculate block counts by dimension and total block count.
    DSizes<Index, NumDims> block_count;
    for (size_t i = 0; i < block_count.rank(); ++i) {
      block_count[i] =
          (m_dimensions[i] + m_block_dim_sizes[i] - 1) / m_block_dim_sizes[i];
    }
    m_total_block_count = array_prod(block_count);

    // Calculate block strides (used for enumerating blocks).
    if (NumDims > 0) {
      if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
        m_block_strides[0] = 1;
        m_tensor_strides[0] = 1;
        for (int i = 1; i < NumDims; ++i) {
          m_block_strides[i] = m_block_strides[i - 1] * block_count[i - 1];
          m_tensor_strides[i] = m_tensor_strides[i - 1] * m_dimensions[i - 1];
        }
      } else {
        m_block_strides[NumDims - 1] = 1;
        m_tensor_strides[NumDims - 1] = 1;
        for (int i = NumDims - 2; i >= 0; --i) {
          m_block_strides[i] = m_block_strides[i + 1] * block_count[i + 1];
          m_tensor_strides[i] = m_tensor_strides[i + 1] * m_dimensions[i + 1];
        }
      }
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorBlock
  GetBlockForIndex(Index block_index, Scalar* data) const {
    Index first_coeff_index = 0;
    DSizes<Index, NumDims> coords;
    DSizes<Index, NumDims> sizes;
    DSizes<Index, NumDims> strides;
    if (NumDims > 0) {
      if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
        for (int i = NumDims - 1; i > 0; --i) {
          const Index idx = block_index / m_block_strides[i];
          coords[i] = idx * m_block_dim_sizes[i];
          sizes[i] =
              numext::mini((m_dimensions[i] - coords[i]), m_block_dim_sizes[i]);
          block_index -= idx * m_block_strides[i];
          first_coeff_index += coords[i] * m_tensor_strides[i];
        }
        coords[0] = block_index * m_block_dim_sizes[0];
        sizes[0] =
            numext::mini((m_dimensions[0] - coords[0]), m_block_dim_sizes[0]);
        first_coeff_index += coords[0] * m_tensor_strides[0];

        strides[0] = 1;
        for (int i = 1; i < NumDims; ++i) {
          strides[i] = strides[i - 1] * sizes[i - 1];
        }
      } else {
        for (int i = 0; i < NumDims - 1; ++i) {
          const Index idx = block_index / m_block_strides[i];
          coords[i] = idx * m_block_dim_sizes[i];
          sizes[i] =
              numext::mini((m_dimensions[i] - coords[i]), m_block_dim_sizes[i]);
          block_index -= idx * m_block_strides[i];
          first_coeff_index += coords[i] * m_tensor_strides[i];
        }
        coords[NumDims - 1] = block_index * m_block_dim_sizes[NumDims - 1];
        sizes[NumDims - 1] =
            numext::mini((m_dimensions[NumDims - 1] - coords[NumDims - 1]),
                       m_block_dim_sizes[NumDims - 1]);
        first_coeff_index += coords[NumDims - 1] * m_tensor_strides[NumDims - 1];

        strides[NumDims - 1] = 1;
        for (int i = NumDims - 2; i >= 0; --i) {
          strides[i] = strides[i + 1] * sizes[i + 1];
        }
      }
    }

    return TensorBlock(first_coeff_index, sizes, strides, m_tensor_strides,
                       data);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index total_block_count() const {
    return m_total_block_count;
  }

 private:
  DSizes<Index, NumDims> m_dimensions;
  DSizes<Index, NumDims> m_block_dim_sizes;
  DSizes<Index, NumDims> m_block_strides;
  DSizes<Index, NumDims> m_tensor_strides;
  Index m_total_block_count;
};

/** \class TensorSliceBlockMapper
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Tensor slice block mapper class.
  *
  * This class is responsible for iterating over the blocks of
  * a slice of a tensor. Supports shuffling of the block strides
  * for callers that want to reduce strides for dimensions to be
  * processed together.
  *
  */

template <typename Index, typename Scalar, std::size_t NumDims, int Layout>
class TensorSliceBlockMapper {
 public:
  typedef typename internal::TensorBlock<Index, Scalar, NumDims, Layout>
      TensorBlock;
  typedef DSizes<Index, NumDims> Dimensions;

  TensorSliceBlockMapper(const Dimensions& tensor_dims,
                         const Dimensions& tensor_slice_offsets,
                         const Dimensions& tensor_slice_extents,
                         const Dimensions& block_dim_sizes,
                         const Dimensions& block_stride_order)
      : m_tensor_dimensions(tensor_dims),
        m_tensor_slice_offsets(tensor_slice_offsets),
        m_tensor_slice_extents(tensor_slice_extents),
        m_block_dim_sizes(block_dim_sizes),
        m_block_stride_order(block_stride_order),
        m_total_block_count(1) {
    // Calculate block counts by dimension and total block count.
    DSizes<Index, NumDims> block_count;
    for (size_t i = 0; i < block_count.rank(); ++i) {
      block_count[i] = (m_tensor_slice_extents[i] + m_block_dim_sizes[i] - 1) /
          m_block_dim_sizes[i];
    }
    m_total_block_count = array_prod(block_count);

    // Calculate block strides (used for enumerating blocks).
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      m_block_strides[0] = 1;
      m_tensor_strides[0] = 1;
      for (int i = 1; i < NumDims; ++i) {
        m_block_strides[i] = m_block_strides[i - 1] * block_count[i - 1];
        m_tensor_strides[i] = m_tensor_strides[i - 1] *
            m_tensor_dimensions[i - 1];
      }
    } else {
      m_block_strides[NumDims - 1] = 1;
      m_tensor_strides[NumDims - 1] = 1;
      for (int i = NumDims - 2; i >= 0; --i) {
        m_block_strides[i] = m_block_strides[i + 1] * block_count[i + 1];
        m_tensor_strides[i] = m_tensor_strides[i + 1] *
            m_tensor_dimensions[i + 1];
      }
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorBlock
  GetBlockForIndex(Index block_index, Scalar* data) const {
    Index first_coeff_index = 0;
    DSizes<Index, NumDims> coords;
    DSizes<Index, NumDims> sizes;
    DSizes<Index, NumDims> strides;
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      for (int i = NumDims - 1; i > 0; --i) {
        const Index idx = block_index / m_block_strides[i];
        coords[i] = m_tensor_slice_offsets[i] + idx * m_block_dim_sizes[i];
        sizes[i] = numext::mini(m_tensor_slice_offsets[i] + m_tensor_slice_extents[i] - coords[i],
                                m_block_dim_sizes[i]);
        block_index -= idx * m_block_strides[i];
        first_coeff_index += coords[i] * m_tensor_strides[i];
      }
      coords[0] = m_tensor_slice_offsets[0] +
          block_index * m_block_dim_sizes[0];
      sizes[0] = numext::mini(m_tensor_slice_offsets[0] + m_tensor_slice_extents[0] - coords[0],
                                m_block_dim_sizes[0]);
      first_coeff_index += coords[0] * m_tensor_strides[0];

      Index prev_dim = m_block_stride_order[0];
      strides[prev_dim] = 1;
      for (int i = 1; i < NumDims; ++i) {
        const Index curr_dim = m_block_stride_order[i];
        strides[curr_dim] = strides[prev_dim] * sizes[prev_dim];
        prev_dim = curr_dim;
      }
    } else {
      for (int i = 0; i < static_cast<int>(NumDims) - 1; ++i) {
        const Index idx = block_index / m_block_strides[i];
        coords[i] = m_tensor_slice_offsets[i] + idx * m_block_dim_sizes[i];
        sizes[i] = numext::mini(m_tensor_slice_offsets[i] + m_tensor_slice_extents[i] - coords[i],
                                m_block_dim_sizes[i]);
        block_index -= idx * m_block_strides[i];
        first_coeff_index += coords[i] * m_tensor_strides[i];
      }
      coords[NumDims - 1] = m_tensor_slice_offsets[NumDims - 1] +
          block_index * m_block_dim_sizes[NumDims - 1];
      sizes[NumDims - 1] = numext::mini(
          m_tensor_slice_offsets[NumDims - 1] + m_tensor_slice_extents[NumDims - 1] - coords[NumDims - 1],
          m_block_dim_sizes[NumDims - 1]);
      first_coeff_index += coords[NumDims - 1] * m_tensor_strides[NumDims - 1];

      Index prev_dim = m_block_stride_order[NumDims - 1];
      strides[prev_dim] = 1;
      for (int i = NumDims - 2; i >= 0; --i) {
        const Index curr_dim = m_block_stride_order[i];
        strides[curr_dim] = strides[prev_dim] * sizes[prev_dim];
        prev_dim = curr_dim;
      }
    }

    return TensorBlock(first_coeff_index, sizes, strides, m_tensor_strides,
                       data);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index total_block_count() const {
    return m_total_block_count;
  }

 private:
  Dimensions m_tensor_dimensions;
  Dimensions m_tensor_slice_offsets;
  Dimensions m_tensor_slice_extents;
  Dimensions m_tensor_strides;
  Dimensions m_block_dim_sizes;
  Dimensions m_block_stride_order;
  Dimensions m_block_strides;
  Index m_total_block_count;
};

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_CXX11_TENSOR_TENSOR_BLOCK_H
