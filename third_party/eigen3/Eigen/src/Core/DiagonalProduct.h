// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2007-2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_DIAGONALPRODUCT_H
#define EIGEN_DIAGONALPRODUCT_H

namespace Eigen { 

namespace internal {
template<typename MatrixType, typename DiagonalType, int ProductOrder>
struct traits<DiagonalProduct<MatrixType, DiagonalType, ProductOrder> >
 : traits<MatrixType>
{
  typedef typename scalar_product_traits<typename MatrixType::Scalar, typename DiagonalType::Scalar>::ReturnType Scalar;
  enum {
    RowsAtCompileTime = MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = MatrixType::ColsAtCompileTime,
    MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime,

    _StorageOrder = MatrixType::Flags & RowMajorBit ? RowMajor : ColMajor,
    _ScalarAccessOnDiag =  !((int(_StorageOrder) == ColMajor && int(ProductOrder) == OnTheLeft)
                          ||(int(_StorageOrder) == RowMajor && int(ProductOrder) == OnTheRight)),
    _SameTypes = is_same<typename MatrixType::Scalar, typename DiagonalType::Scalar>::value,
    // FIXME currently we need same types, but in the future the next rule should be the one
    //_Vectorizable = bool(int(MatrixType::Flags)&PacketAccessBit) && ((!_PacketOnDiag) || (_SameTypes && bool(int(DiagonalType::DiagonalVectorType::Flags)&PacketAccessBit))),
    _Vectorizable = bool(int(MatrixType::Flags)&PacketAccessBit) && _SameTypes && (_ScalarAccessOnDiag || (bool(int(DiagonalType::DiagonalVectorType::Flags)&PacketAccessBit))),
    _LinearAccessMask = (RowsAtCompileTime==1 || ColsAtCompileTime==1) ? LinearAccessBit : 0,

    Flags = ((HereditaryBits|_LinearAccessMask) & (unsigned int)(MatrixType::Flags)) | (_Vectorizable ? PacketAccessBit : 0) | AlignedBit,//(int(MatrixType::Flags)&int(DiagonalType::DiagonalVectorType::Flags)&AlignedBit),
    CoeffReadCost = NumTraits<Scalar>::MulCost + MatrixType::CoeffReadCost + DiagonalType::DiagonalVectorType::CoeffReadCost
  };
};
}

template<typename MatrixType, typename DiagonalType, int ProductOrder>
class DiagonalProduct : internal::no_assignment_operator,
                        public MatrixBase<DiagonalProduct<MatrixType, DiagonalType, ProductOrder> >
{
  public:

    typedef MatrixBase<DiagonalProduct> Base;
    EIGEN_DENSE_PUBLIC_INTERFACE(DiagonalProduct)

    inline DiagonalProduct(const MatrixType& matrix, const DiagonalType& diagonal)
      : m_matrix(matrix), m_diagonal(diagonal)
    {
      eigen_assert(diagonal.diagonal().size() == (ProductOrder == OnTheLeft ? matrix.rows() : matrix.cols()));
    }

    EIGEN_STRONG_INLINE Index rows() const { return m_matrix.rows(); }
    EIGEN_STRONG_INLINE Index cols() const { return m_matrix.cols(); }

    EIGEN_STRONG_INLINE const Scalar coeff(Index row, Index col) const
    {
      return m_diagonal.diagonal().coeff(ProductOrder == OnTheLeft ? row : col) * m_matrix.coeff(row, col);
    }
    
    EIGEN_STRONG_INLINE const Scalar coeff(Index idx) const
    {
      enum {
        StorageOrder = int(MatrixType::Flags) & RowMajorBit ? RowMajor : ColMajor
      };
      return coeff(int(StorageOrder)==ColMajor?idx:0,int(StorageOrder)==ColMajor?0:idx);
    }

    template<int LoadMode>
    EIGEN_STRONG_INLINE PacketScalar packet(Index row, Index col) const
    {
      enum {
        StorageOrder = Flags & RowMajorBit ? RowMajor : ColMajor
      };
      const Index indexInDiagonalVector = ProductOrder == OnTheLeft ? row : col;
      return packet_impl<LoadMode>(row,col,indexInDiagonalVector,typename internal::conditional<
        ((int(StorageOrder) == RowMajor && int(ProductOrder) == OnTheLeft)
       ||(int(StorageOrder) == ColMajor && int(ProductOrder) == OnTheRight)), internal::true_type, internal::false_type>::type());
    }
    
    template<int LoadMode>
    EIGEN_STRONG_INLINE PacketScalar packet(Index idx) const
    {
      enum {
        StorageOrder = int(MatrixType::Flags) & RowMajorBit ? RowMajor : ColMajor
      };
      return packet<LoadMode>(int(StorageOrder)==ColMajor?idx:0,int(StorageOrder)==ColMajor?0:idx);
    }

  protected:
    template<int LoadMode>
    EIGEN_STRONG_INLINE PacketScalar packet_impl(Index row, Index col, Index id, internal::true_type) const
    {
      return internal::pmul(m_matrix.template packet<LoadMode>(row, col),
                     internal::pset1<PacketScalar>(m_diagonal.diagonal().coeff(id)));
    }

    template<int LoadMode>
    EIGEN_STRONG_INLINE PacketScalar packet_impl(Index row, Index col, Index id, internal::false_type) const
    {
      enum {
        InnerSize = (MatrixType::Flags & RowMajorBit) ? MatrixType::ColsAtCompileTime : MatrixType::RowsAtCompileTime,
        DiagonalVectorPacketLoadMode = (LoadMode == Aligned && (((InnerSize%16) == 0) || (int(DiagonalType::DiagonalVectorType::Flags)&AlignedBit)==AlignedBit) ? Aligned : Unaligned)
      };
      return internal::pmul(m_matrix.template packet<LoadMode>(row, col),
                     m_diagonal.diagonal().template packet<DiagonalVectorPacketLoadMode>(id));
    }

    typename MatrixType::Nested m_matrix;
    typename DiagonalType::Nested m_diagonal;
};

/** \returns the diagonal matrix product of \c *this by the diagonal matrix \a diagonal.
  */
template<typename Derived>
template<typename DiagonalDerived>
inline const DiagonalProduct<Derived, DiagonalDerived, OnTheRight>
MatrixBase<Derived>::operator*(const DiagonalBase<DiagonalDerived> &a_diagonal) const
{
  return DiagonalProduct<Derived, DiagonalDerived, OnTheRight>(derived(), a_diagonal.derived());
}

} // end namespace Eigen

#endif // EIGEN_DIAGONALPRODUCT_H
