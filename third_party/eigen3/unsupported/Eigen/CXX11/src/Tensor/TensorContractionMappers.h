// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Eric Martin <eric@ericmart.in>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_MAPPERS_H
#define EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_MAPPERS_H

// NOTE: The file has strong column major bias/assumptions, which is pointed out
// in comments. As of right now, this code will only work the column major packing
// routines.

/*
 * A tensor contraction can be represented by a matrix multiplication. We don't
 * want to actually reshape the tensor into a matrix (because this involves a
 * full copy of the tensor), so the reshaping operation is implicit in a sense.
 * This means we need a collection of methods take a matrix index and return
 * the element of the tensor that would be at that index if we were to actually
 * reshape the matrix. This file consists of these methods.
 */

namespace Eigen {
namespace internal {

enum {
  Rhs = 0,
  Lhs = 1,
};

/*
 * Used to lookup the tensor index when working with the left and right
 * arguments to a tensor contraction.
 */
template<typename Scalar, typename Index, int side,
         typename Tensor,
         typename nocontract_t, typename contract_t,
         size_t packet_size, bool inner_dim_contiguous>
class SimpleTensorContractionMapper {
  public:
  EIGEN_DEVICE_FUNC
  SimpleTensorContractionMapper(const Tensor& tensor,
                              const nocontract_t& nocontract_strides,
                              const nocontract_t& ij_strides,
                              const contract_t& contract_strides,
                              const contract_t& k_strides) :
      m_tensor(tensor),
      m_nocontract_strides(nocontract_strides),
      m_ij_strides(ij_strides),
      m_contract_strides(contract_strides),
      m_k_strides(k_strides) { }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE void prefetch(int i) { }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Scalar operator()(Index row) const {
    // column major assumption
    return operator()(row, 0);
  }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Scalar operator()(Index row, Index col) const {
    return m_tensor.coeff(computeIndex(row, col));
  }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Index computeIndex(Index row, Index col) const {
    const bool left = (side == Lhs);
    Index nocontract_val = left ? row : col;
    Index linidx = 0;
    for (int i = array_size<nocontract_t>::value - 1; i > 0; i--) {
      const Index idx = nocontract_val / m_ij_strides[i];
      linidx += idx * m_nocontract_strides[i];
      nocontract_val -= idx * m_ij_strides[i];
    }
    if (array_size<typename Tensor::Dimensions>::value > array_size<contract_t>::value) {
      if (side == Lhs && inner_dim_contiguous) {
        eigen_assert(m_nocontract_strides[0] == 1);
        linidx += nocontract_val;
      } else {
        linidx += nocontract_val * m_nocontract_strides[0];
      }
    }

    Index contract_val = left ? col : row;
    for (int i = array_size<contract_t>::value - 1; i > 0; i--) {
      const Index idx = contract_val / m_k_strides[i];
      linidx += idx * m_contract_strides[i];
      contract_val -= idx * m_k_strides[i];
    }
    EIGEN_STATIC_ASSERT(array_size<contract_t>::value > 0, YOU_MADE_A_PROGRAMMING_MISTAKE);
    if (side == Rhs && inner_dim_contiguous) {
      eigen_assert(m_contract_strides[0] == 1);
      linidx += contract_val;
    } else {
      linidx += contract_val * m_contract_strides[0];
    }

    return linidx;
  }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE IndexPair<Index> computeIndexPair(Index row, Index col, const Index distance) const {
    const bool left = (side == Lhs);
    Index nocontract_val[2] = {left ? row : col, left ? row + distance : col};
    Index linidx[2] = {0, 0};
    for (int i = array_size<nocontract_t>::value - 1; i > 0; i--) {
      const Index idx0 = nocontract_val[0] / m_ij_strides[i];
      const Index idx1 = nocontract_val[1] / m_ij_strides[i];
      linidx[0] += idx0 * m_nocontract_strides[i];
      linidx[1] += idx1 * m_nocontract_strides[i];
      nocontract_val[0] -= idx0 * m_ij_strides[i];
      nocontract_val[1] -= idx1 * m_ij_strides[i];
    }
    if (array_size<typename Tensor::Dimensions>::value > array_size<contract_t>::value) {
      if (side == Lhs && inner_dim_contiguous) {
        eigen_assert(m_nocontract_strides[0] == 1);
        linidx[0] += nocontract_val[0];
        linidx[1] += nocontract_val[1];
      } else {
        linidx[0] += nocontract_val[0] * m_nocontract_strides[0];
        linidx[1] += nocontract_val[1] * m_nocontract_strides[0];
      }
    }

    Index contract_val[2] = {left ? col : row, left ? col : row + distance};
    for (int i = array_size<contract_t>::value - 1; i > 0; i--) {
      const Index idx0 = contract_val[0] / m_k_strides[i];
      const Index idx1 = contract_val[1] / m_k_strides[i];
      linidx[0] += idx0 * m_contract_strides[i];
      linidx[1] += idx1 * m_contract_strides[i];
      contract_val[0] -= idx0 * m_k_strides[i];
      contract_val[1] -= idx1 * m_k_strides[i];
    }
    EIGEN_STATIC_ASSERT(array_size<contract_t>::value > 0, YOU_MADE_A_PROGRAMMING_MISTAKE);
    if (side == Rhs && inner_dim_contiguous) {
      eigen_assert(m_contract_strides[0] == 1);
      linidx[0] += contract_val[0];
      linidx[1] += contract_val[1];
    } else {
      linidx[0] += contract_val[0] * m_contract_strides[0];
      linidx[1] += contract_val[1] * m_contract_strides[0];
    }
    return IndexPair<Index>(linidx[0], linidx[1]);
  }

  Index firstAligned(Index size) const {
    return size;
  }
  Index stride() const {
    return 1;
  }

 protected:
  const Tensor m_tensor;
  const nocontract_t m_nocontract_strides;
  const nocontract_t m_ij_strides;
  const contract_t m_contract_strides;
  const contract_t m_k_strides;
};



template<typename Scalar, typename Index, int side,
         typename Tensor,
         typename nocontract_t, typename contract_t,
         size_t packet_size, bool inner_dim_contiguous,
         bool inner_dim_reordered, int Alignment>
  class BaseTensorContractionMapper : public SimpleTensorContractionMapper<Scalar, Index, side, Tensor, nocontract_t, contract_t, packet_size, inner_dim_contiguous>
{
 public:
  typedef SimpleTensorContractionMapper<Scalar, Index, side, Tensor, nocontract_t, contract_t, packet_size, inner_dim_contiguous> ParentMapper;

  EIGEN_DEVICE_FUNC
  BaseTensorContractionMapper(const Tensor& tensor,
                              const nocontract_t& nocontract_strides,
                              const nocontract_t& ij_strides,
                              const contract_t& contract_strides,
                              const contract_t& k_strides) :
  ParentMapper(tensor, nocontract_strides, ij_strides, contract_strides, k_strides) { }

  typedef typename packet_traits<Scalar>::type Packet;
  typedef typename packet_traits<Scalar>::half HalfPacket;

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Packet loadPacket(Index i, Index j) const {
    // whole method makes column major assumption

    // don't need to add offsets for now (because operator handles that)
    // current code assumes packet size must be a multiple of 2
    EIGEN_STATIC_ASSERT(packet_size % 2 == 0, YOU_MADE_A_PROGRAMMING_MISTAKE);

    if (Tensor::PacketAccess && inner_dim_contiguous && !inner_dim_reordered) {
      const Index index = this->computeIndex(i, j);
      eigen_assert(this->computeIndex(i+packet_size-1, j) == index + packet_size-1);
      return this->m_tensor.template packet<Alignment>(index);
    }

    const IndexPair<Index> indexPair = this->computeIndexPair(i, j, packet_size - 1);
    const Index first = indexPair.first;
    const Index last = indexPair.second;

    // We can always do optimized packet reads from left hand side right now, because
    // the vertical matrix dimension on the left hand side is never contracting.
    // On the right hand side we need to check if the contracting dimensions may have
    // been shuffled first.
    if (Tensor::PacketAccess &&
        (side == Lhs || internal::array_size<contract_t>::value <= 1 || !inner_dim_reordered) &&
        (last - first) == (packet_size - 1)) {

      return this->m_tensor.template packet<Alignment>(first);
    }

    EIGEN_ALIGN_DEFAULT Scalar data[packet_size];

    data[0] = this->m_tensor.coeff(first);
    for (Index k = 1; k < packet_size - 1; k += 2) {
      const IndexPair<Index> internal_pair = this->computeIndexPair(i + k, j, 1);
      data[k] = this->m_tensor.coeff(internal_pair.first);
      data[k + 1] = this->m_tensor.coeff(internal_pair.second);
    }
    data[packet_size - 1] = this->m_tensor.coeff(last);

    return pload<Packet>(data);
  }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE HalfPacket loadHalfPacket(Index i, Index j) const {
    // whole method makes column major assumption

    // don't need to add offsets for now (because operator handles that)
    const Index half_packet_size = unpacket_traits<HalfPacket>::size;
    if (half_packet_size == packet_size) {
      return loadPacket(i, j);
    }
    EIGEN_ALIGN_DEFAULT Scalar data[half_packet_size];
    for (Index k = 0; k < half_packet_size; k++) {
      data[k] = operator()(i + k, j);
    }
    return pload<HalfPacket>(data);
  }
};


template<typename Scalar, typename Index, int side,
         typename Tensor,
         typename nocontract_t, typename contract_t,
         bool inner_dim_contiguous,
         bool inner_dim_reordered, int Alignment>
class BaseTensorContractionMapper<Scalar, Index, side, Tensor, nocontract_t, contract_t, 1, inner_dim_contiguous, inner_dim_reordered, Alignment> : public SimpleTensorContractionMapper<Scalar, Index, side, Tensor, nocontract_t, contract_t, 1, inner_dim_contiguous>
{
 public:
  typedef SimpleTensorContractionMapper<Scalar, Index, side, Tensor, nocontract_t, contract_t, 1, inner_dim_contiguous> ParentMapper;

  EIGEN_DEVICE_FUNC
  BaseTensorContractionMapper(const Tensor& tensor,
                              const nocontract_t& nocontract_strides,
                              const nocontract_t& ij_strides,
                              const contract_t& contract_strides,
                              const contract_t& k_strides) :
  ParentMapper(tensor, nocontract_strides, ij_strides, contract_strides, k_strides) { }

  typedef typename packet_traits<Scalar>::type Packet;
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Packet loadPacket(Index i, Index j) const {
    EIGEN_ALIGN_DEFAULT Scalar data[1];
    data[0] = this->m_tensor.coeff(this->computeIndex(i, j));
    return pload<typename packet_traits<Scalar>::type>(data);
  }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Packet loadHalfPacket(Index i, Index j) const {
    return loadPacket(i, j);
  }
};

template<typename Scalar, typename Index, int side,
         typename Tensor,
         typename nocontract_t, typename contract_t,
         size_t packet_size,
         bool inner_dim_contiguous, bool inner_dim_reordered, int Alignment>
class TensorContractionInputMapper;

template<typename Scalar, typename Index, int side,
         typename Tensor,
         typename nocontract_t, typename contract_t,
         size_t packet_size,
         bool inner_dim_contiguous, bool inner_dim_reordered, int Alignment>
class TensorContractionSubMapper {
 public:
  typedef typename packet_traits<Scalar>::type Packet;
  typedef typename packet_traits<Scalar>::half HalfPacket;

  typedef TensorContractionInputMapper<Scalar, Index, side, Tensor, nocontract_t, contract_t, packet_size, inner_dim_contiguous, inner_dim_reordered, Alignment> ParentMapper;
  typedef TensorContractionSubMapper<Scalar, Index, side, Tensor, nocontract_t, contract_t, packet_size, inner_dim_contiguous, inner_dim_reordered, Alignment> Self;
  typedef Self LinearMapper;

  EIGEN_DEVICE_FUNC TensorContractionSubMapper(const ParentMapper& base_mapper, Index vert_offset, Index horiz_offset)
      : m_base_mapper(base_mapper), m_vert_offset(vert_offset), m_horiz_offset(horiz_offset) { }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Scalar operator()(Index i) const {
    return m_base_mapper(i + m_vert_offset, m_horiz_offset);
  }
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Scalar operator()(Index i, Index j) const {
    return m_base_mapper(i + m_vert_offset, j + m_horiz_offset);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet loadPacket(Index i) const {
    return m_base_mapper.loadPacket(i + m_vert_offset, m_horiz_offset);
  }
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet loadPacket(Index i, Index j) const {
   return m_base_mapper.loadPacket(i + m_vert_offset, j + m_horiz_offset);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE HalfPacket loadHalfPacket(Index i) const {
    return m_base_mapper.loadHalfPacket(i + m_vert_offset, m_horiz_offset);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void storePacket(Index i, Packet p) const {
    m_base_mapper.storePacket(i + m_vert_offset, m_horiz_offset, p);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE LinearMapper getLinearMapper(Index i, Index j) const {
    return LinearMapper(m_base_mapper, i + m_vert_offset, j + m_horiz_offset);
  }

  template <typename PacketT, int AlignmentType>
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE PacketT load(Index i) const {
    EIGEN_STATIC_ASSERT((internal::is_same<PacketT, Packet>::value), YOU_MADE_A_PROGRAMMING_MISTAKE);
    EIGEN_STATIC_ASSERT((AlignmentType == Aligned || Alignment == Unaligned), YOU_MADE_A_PROGRAMMING_MISTAKE);
    return loadPacket(i);
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC bool aligned(Index i) const {
    return false;
  }

 private:
  const ParentMapper& m_base_mapper;
  const Index m_vert_offset;
  const Index m_horiz_offset;
};


template<typename Scalar, typename Index, int side,
         typename Tensor,
         typename nocontract_t, typename contract_t,
         size_t packet_size,
         bool inner_dim_contiguous, bool inner_dim_reordered, int Alignment>
class TensorContractionInputMapper
  : public BaseTensorContractionMapper<Scalar, Index, side, Tensor, nocontract_t, contract_t, packet_size, inner_dim_contiguous, inner_dim_reordered, Alignment> {

 public:
  typedef BaseTensorContractionMapper<Scalar, Index, side, Tensor, nocontract_t, contract_t, packet_size, inner_dim_contiguous, inner_dim_reordered, Alignment> Base;
  typedef TensorContractionSubMapper<Scalar, Index, side, Tensor, nocontract_t, contract_t, packet_size, inner_dim_contiguous, inner_dim_reordered, Alignment> SubMapper;
  typedef SubMapper VectorMapper;

  EIGEN_DEVICE_FUNC TensorContractionInputMapper(const Tensor& tensor,
                               const nocontract_t& nocontract_strides,
                               const nocontract_t& ij_strides,
                               const contract_t& contract_strides,
                               const contract_t& k_strides)
      : Base(tensor, nocontract_strides, ij_strides, contract_strides, k_strides) { }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE SubMapper getSubMapper(Index i, Index j) const {
    return SubMapper(*this, i, j);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE VectorMapper getVectorMapper(Index i, Index j) const {
    return VectorMapper(*this, i, j);
  }
};


} // end namespace internal
} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_MAPPERS_H
