// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSEPRODUCT_H
#define EIGEN_SPARSEPRODUCT_H

namespace Eigen { 

template<typename Lhs, typename Rhs>
struct SparseSparseProductReturnType
{
  typedef typename internal::traits<Lhs>::Scalar Scalar;
  typedef typename internal::traits<Lhs>::Index Index;
  enum {
    LhsRowMajor = internal::traits<Lhs>::Flags & RowMajorBit,
    RhsRowMajor = internal::traits<Rhs>::Flags & RowMajorBit,
    TransposeRhs = (!LhsRowMajor) && RhsRowMajor,
    TransposeLhs = LhsRowMajor && (!RhsRowMajor)
  };

  typedef typename internal::conditional<TransposeLhs,
    SparseMatrix<Scalar,0,Index>,
    typename internal::nested<Lhs,Rhs::RowsAtCompileTime>::type>::type LhsNested;

  typedef typename internal::conditional<TransposeRhs,
    SparseMatrix<Scalar,0,Index>,
    typename internal::nested<Rhs,Lhs::RowsAtCompileTime>::type>::type RhsNested;

  typedef SparseSparseProduct<LhsNested, RhsNested> Type;
};

namespace internal {
template<typename LhsNested, typename RhsNested>
struct traits<SparseSparseProduct<LhsNested, RhsNested> >
{
  typedef MatrixXpr XprKind;
  // clean the nested types:
  typedef typename remove_all<LhsNested>::type _LhsNested;
  typedef typename remove_all<RhsNested>::type _RhsNested;
  typedef typename _LhsNested::Scalar Scalar;
  typedef typename promote_index_type<typename traits<_LhsNested>::Index,
                                         typename traits<_RhsNested>::Index>::type Index;

  enum {
    LhsCoeffReadCost = _LhsNested::CoeffReadCost,
    RhsCoeffReadCost = _RhsNested::CoeffReadCost,
    LhsFlags = _LhsNested::Flags,
    RhsFlags = _RhsNested::Flags,

    RowsAtCompileTime    = _LhsNested::RowsAtCompileTime,
    ColsAtCompileTime    = _RhsNested::ColsAtCompileTime,
    MaxRowsAtCompileTime = _LhsNested::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = _RhsNested::MaxColsAtCompileTime,

    InnerSize = EIGEN_SIZE_MIN_PREFER_FIXED(_LhsNested::ColsAtCompileTime, _RhsNested::RowsAtCompileTime),

    EvalToRowMajor = (RhsFlags & LhsFlags & RowMajorBit),

    RemovedBits = ~(EvalToRowMajor ? 0 : RowMajorBit),

    Flags = (int(LhsFlags | RhsFlags) & HereditaryBits & RemovedBits)
          | EvalBeforeAssigningBit
          | EvalBeforeNestingBit,

    CoeffReadCost = Dynamic
  };

  typedef Sparse StorageKind;
};

} // end namespace internal

template<typename LhsNested, typename RhsNested>
class SparseSparseProduct : internal::no_assignment_operator,
  public SparseMatrixBase<SparseSparseProduct<LhsNested, RhsNested> >
{
  public:

    typedef SparseMatrixBase<SparseSparseProduct> Base;
    EIGEN_DENSE_PUBLIC_INTERFACE(SparseSparseProduct)

  private:

    typedef typename internal::traits<SparseSparseProduct>::_LhsNested _LhsNested;
    typedef typename internal::traits<SparseSparseProduct>::_RhsNested _RhsNested;

  public:

    template<typename Lhs, typename Rhs>
    EIGEN_STRONG_INLINE SparseSparseProduct(const Lhs& lhs, const Rhs& rhs)
      : m_lhs(lhs), m_rhs(rhs), m_tolerance(0), m_conservative(true)
    {
      init();
    }

    template<typename Lhs, typename Rhs>
    EIGEN_STRONG_INLINE SparseSparseProduct(const Lhs& lhs, const Rhs& rhs, const RealScalar& tolerance)
      : m_lhs(lhs), m_rhs(rhs), m_tolerance(tolerance), m_conservative(false)
    {
      init();
    }

    SparseSparseProduct pruned(const Scalar& reference = 0, const RealScalar& epsilon = NumTraits<RealScalar>::dummy_precision()) const
    {
      using std::abs;
      return SparseSparseProduct(m_lhs,m_rhs,abs(reference)*epsilon);
    }

    template<typename Dest>
    void evalTo(Dest& result) const
    {
      if(m_conservative)
        internal::conservative_sparse_sparse_product_selector<_LhsNested, _RhsNested, Dest>::run(lhs(),rhs(),result);
      else
        internal::sparse_sparse_product_with_pruning_selector<_LhsNested, _RhsNested, Dest>::run(lhs(),rhs(),result,m_tolerance);
    }

    EIGEN_STRONG_INLINE Index rows() const { return m_lhs.rows(); }
    EIGEN_STRONG_INLINE Index cols() const { return m_rhs.cols(); }

    EIGEN_STRONG_INLINE const _LhsNested& lhs() const { return m_lhs; }
    EIGEN_STRONG_INLINE const _RhsNested& rhs() const { return m_rhs; }

  protected:
    void init()
    {
      eigen_assert(m_lhs.cols() == m_rhs.rows());

      enum {
        ProductIsValid = _LhsNested::ColsAtCompileTime==Dynamic
                      || _RhsNested::RowsAtCompileTime==Dynamic
                      || int(_LhsNested::ColsAtCompileTime)==int(_RhsNested::RowsAtCompileTime),
        AreVectors = _LhsNested::IsVectorAtCompileTime && _RhsNested::IsVectorAtCompileTime,
        SameSizes = EIGEN_PREDICATE_SAME_MATRIX_SIZE(_LhsNested,_RhsNested)
      };
      // note to the lost user:
      //    * for a dot product use: v1.dot(v2)
      //    * for a coeff-wise product use: v1.cwise()*v2
      EIGEN_STATIC_ASSERT(ProductIsValid || !(AreVectors && SameSizes),
        INVALID_VECTOR_VECTOR_PRODUCT__IF_YOU_WANTED_A_DOT_OR_COEFF_WISE_PRODUCT_YOU_MUST_USE_THE_EXPLICIT_FUNCTIONS)
      EIGEN_STATIC_ASSERT(ProductIsValid || !(SameSizes && !AreVectors),
        INVALID_MATRIX_PRODUCT__IF_YOU_WANTED_A_COEFF_WISE_PRODUCT_YOU_MUST_USE_THE_EXPLICIT_FUNCTION)
      EIGEN_STATIC_ASSERT(ProductIsValid || SameSizes, INVALID_MATRIX_PRODUCT)
    }

    LhsNested m_lhs;
    RhsNested m_rhs;
    RealScalar m_tolerance;
    bool m_conservative;
};

// sparse = sparse * sparse
template<typename Derived>
template<typename Lhs, typename Rhs>
inline Derived& SparseMatrixBase<Derived>::operator=(const SparseSparseProduct<Lhs,Rhs>& product)
{
  product.evalTo(derived());
  return derived();
}

/** \returns an expression of the product of two sparse matrices.
  * By default a conservative product preserving the symbolic non zeros is performed.
  * The automatic pruning of the small values can be achieved by calling the pruned() function
  * in which case a totally different product algorithm is employed:
  * \code
  * C = (A*B).pruned();             // supress numerical zeros (exact)
  * C = (A*B).pruned(ref);
  * C = (A*B).pruned(ref,epsilon);
  * \endcode
  * where \c ref is a meaningful non zero reference value.
  * */
template<typename Derived>
template<typename OtherDerived>
inline const typename SparseSparseProductReturnType<Derived,OtherDerived>::Type
SparseMatrixBase<Derived>::operator*(const SparseMatrixBase<OtherDerived> &other) const
{
  return typename SparseSparseProductReturnType<Derived,OtherDerived>::Type(derived(), other.derived());
}

} // end namespace Eigen

#endif // EIGEN_SPARSEPRODUCT_H
