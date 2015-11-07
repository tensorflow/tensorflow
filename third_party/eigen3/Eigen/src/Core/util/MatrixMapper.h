// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Eric Martin <eric@ericmart.in>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MATRIXMAPPER_H
#define EIGEN_MATRIXMAPPER_H

// To support both matrices and tensors, we need a way to abstractly access an
// element of a matrix (where the matrix might be an implicitly flattened
// tensor). This file abstracts the logic needed to access elements in a row
// major or column major matrix.

namespace Eigen {

namespace internal {

template<typename Scalar, typename Index>
class BlasVectorMapper {
  public:
  EIGEN_ALWAYS_INLINE BlasVectorMapper(Scalar *data) : m_data(data) {}

  EIGEN_ALWAYS_INLINE Scalar operator()(Index i) const {
    return m_data[i];
  }
  template <typename Packet, int AlignmentType>
  EIGEN_ALWAYS_INLINE Packet load(Index i) const {
    return ploadt<Packet, AlignmentType>(m_data + i);
  }

  template <typename Packet>
  bool aligned(Index i) const {
    return (size_t(m_data+i)%sizeof(Packet))==0;
  }

  protected:
  Scalar* m_data;
};

// We need a fast way to iterate down columns (if column major) that doesn't
// involves performing a multiplication for each lookup.
template<typename Scalar, typename Index, int AlignmentType>
class BlasLinearMapper {
  public:
  typedef typename packet_traits<Scalar>::type Packet;
  typedef typename packet_traits<Scalar>::half HalfPacket;

  EIGEN_ALWAYS_INLINE BlasLinearMapper(Scalar *data) : m_data(data) {}

  EIGEN_ALWAYS_INLINE void prefetch(int i) const {
    internal::prefetch(&operator()(i));
  }

  EIGEN_ALWAYS_INLINE Scalar& operator()(Index i) const {
    return m_data[i];
  }

  EIGEN_ALWAYS_INLINE Packet loadPacket(Index i) const {
    return ploadt<Packet, AlignmentType>(m_data + i);
  }

  EIGEN_ALWAYS_INLINE HalfPacket loadHalfPacket(Index i) const {
    return ploadt<HalfPacket, AlignmentType>(m_data + i);
  }

  EIGEN_ALWAYS_INLINE void storePacket(Index i, Packet p) const {
    pstoret<Scalar, Packet, AlignmentType>(m_data + i, p);
  }

  protected:
  Scalar* m_data;
};

// This mapper allows access into matrix by coordinates i and j.
template<typename Scalar, typename Index, int StorageOrder, int AlignmentType = Unaligned>
class blas_data_mapper {
  public:
  typedef typename packet_traits<Scalar>::type Packet;
  typedef typename packet_traits<Scalar>::half HalfPacket;

  typedef BlasLinearMapper<Scalar, Index, AlignmentType> LinearMapper;
  typedef BlasVectorMapper<Scalar, Index> VectorMapper;

  EIGEN_ALWAYS_INLINE blas_data_mapper(Scalar* data, Index stride) : m_data(data), m_stride(stride) {}

  EIGEN_ALWAYS_INLINE blas_data_mapper<Scalar, Index, StorageOrder, AlignmentType>
  getSubMapper(Index i, Index j) const {
    return blas_data_mapper<Scalar, Index, StorageOrder, AlignmentType>(&operator()(i, j), m_stride);
  }

  EIGEN_ALWAYS_INLINE LinearMapper getLinearMapper(Index i, Index j) const {
    return LinearMapper(&operator()(i, j));
  }

  EIGEN_ALWAYS_INLINE VectorMapper getVectorMapper(Index i, Index j) const {
    return VectorMapper(&operator()(i, j));
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Scalar& operator()(Index i, Index j) const {
    return m_data[StorageOrder==RowMajor ? j + i*m_stride : i + j*m_stride];
  }

  EIGEN_ALWAYS_INLINE Packet loadPacket(Index i, Index j) const {
    return ploadt<Packet, AlignmentType>(&operator()(i, j));
  }

  EIGEN_ALWAYS_INLINE HalfPacket loadHalfPacket(Index i, Index j) const {
    return ploadt<HalfPacket, AlignmentType>(&operator()(i, j));
  }

  template<typename SubPacket>
  EIGEN_ALWAYS_INLINE void scatterPacket(Index i, Index j, SubPacket p) const {
    pscatter<Scalar, SubPacket>(&operator()(i, j), p, m_stride);
  }

  template<typename SubPacket>
  EIGEN_ALWAYS_INLINE SubPacket gatherPacket(Index i, Index j) const {
    return pgather<Scalar, SubPacket>(&operator()(i, j), m_stride);
  }

  const Index stride() const { return m_stride; }

  Index firstAligned(Index size) const {
    if (size_t(m_data)%sizeof(Scalar)) {
      return -1;
    }
    return internal::first_aligned(m_data, size);
  }

  protected:
  Scalar* EIGEN_RESTRICT m_data;
  const Index m_stride;
};

// This is just a convienent way to work with
// blas_data_mapper<const Scalar, Index, StorageOrder>
template<typename Scalar, typename Index, int StorageOrder>
class const_blas_data_mapper : public blas_data_mapper<const Scalar, Index, StorageOrder> {
  public:
  EIGEN_ALWAYS_INLINE const_blas_data_mapper(const Scalar *data, Index stride) : blas_data_mapper<const Scalar, Index, StorageOrder>(data, stride) {}

  EIGEN_ALWAYS_INLINE const_blas_data_mapper<Scalar, Index, StorageOrder> getSubMapper(Index i, Index j) const {
    return const_blas_data_mapper<Scalar, Index, StorageOrder>(&(this->operator()(i, j)), this->m_stride);
  }
};

} // end namespace internal
} // end namespace eigen

#endif //EIGEN_MATRIXMAPPER_H
