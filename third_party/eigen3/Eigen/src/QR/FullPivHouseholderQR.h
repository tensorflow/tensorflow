// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_FULLPIVOTINGHOUSEHOLDERQR_H
#define EIGEN_FULLPIVOTINGHOUSEHOLDERQR_H

namespace Eigen { 

namespace internal {

template<typename MatrixType> struct FullPivHouseholderQRMatrixQReturnType;

template<typename MatrixType>
struct traits<FullPivHouseholderQRMatrixQReturnType<MatrixType> >
{
  typedef typename MatrixType::PlainObject ReturnType;
};

}

/** \ingroup QR_Module
  *
  * \class FullPivHouseholderQR
  *
  * \brief Householder rank-revealing QR decomposition of a matrix with full pivoting
  *
  * \param MatrixType the type of the matrix of which we are computing the QR decomposition
  *
  * This class performs a rank-revealing QR decomposition of a matrix \b A into matrices \b P, \b P', \b Q and \b R
  * such that 
  * \f[
  *  \mathbf{P} \, \mathbf{A} \, \mathbf{P}' = \mathbf{Q} \, \mathbf{R}
  * \f]
  * by using Householder transformations. Here, \b P and \b P' are permutation matrices, \b Q a unitary matrix 
  * and \b R an upper triangular matrix.
  *
  * This decomposition performs a very prudent full pivoting in order to be rank-revealing and achieve optimal
  * numerical stability. The trade-off is that it is slower than HouseholderQR and ColPivHouseholderQR.
  *
  * \sa MatrixBase::fullPivHouseholderQr()
  */
template<typename _MatrixType> class FullPivHouseholderQR
{
  public:

    typedef _MatrixType MatrixType;
    enum {
      RowsAtCompileTime = MatrixType::RowsAtCompileTime,
      ColsAtCompileTime = MatrixType::ColsAtCompileTime,
      Options = MatrixType::Options,
      MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
    };
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::RealScalar RealScalar;
    typedef typename MatrixType::Index Index;
    typedef internal::FullPivHouseholderQRMatrixQReturnType<MatrixType> MatrixQReturnType;
    typedef typename internal::plain_diag_type<MatrixType>::type HCoeffsType;
    typedef Matrix<Index, 1,
                   EIGEN_SIZE_MIN_PREFER_DYNAMIC(ColsAtCompileTime,RowsAtCompileTime), RowMajor, 1,
                   EIGEN_SIZE_MIN_PREFER_FIXED(MaxColsAtCompileTime,MaxRowsAtCompileTime)> IntDiagSizeVectorType;
    typedef PermutationMatrix<ColsAtCompileTime, MaxColsAtCompileTime> PermutationType;
    typedef typename internal::plain_row_type<MatrixType>::type RowVectorType;
    typedef typename internal::plain_col_type<MatrixType>::type ColVectorType;

    /** \brief Default Constructor.
      *
      * The default constructor is useful in cases in which the user intends to
      * perform decompositions via FullPivHouseholderQR::compute(const MatrixType&).
      */
    FullPivHouseholderQR()
      : m_qr(),
        m_hCoeffs(),
        m_rows_transpositions(),
        m_cols_transpositions(),
        m_cols_permutation(),
        m_temp(),
        m_isInitialized(false),
        m_usePrescribedThreshold(false) {}

    /** \brief Default Constructor with memory preallocation
      *
      * Like the default constructor but with preallocation of the internal data
      * according to the specified problem \a size.
      * \sa FullPivHouseholderQR()
      */
    FullPivHouseholderQR(Index rows, Index cols)
      : m_qr(rows, cols),
        m_hCoeffs((std::min)(rows,cols)),
        m_rows_transpositions((std::min)(rows,cols)),
        m_cols_transpositions((std::min)(rows,cols)),
        m_cols_permutation(cols),
        m_temp(cols),
        m_isInitialized(false),
        m_usePrescribedThreshold(false) {}

    /** \brief Constructs a QR factorization from a given matrix
      *
      * This constructor computes the QR factorization of the matrix \a matrix by calling
      * the method compute(). It is a short cut for:
      * 
      * \code
      * FullPivHouseholderQR<MatrixType> qr(matrix.rows(), matrix.cols());
      * qr.compute(matrix);
      * \endcode
      * 
      * \sa compute()
      */
    FullPivHouseholderQR(const MatrixType& matrix)
      : m_qr(matrix.rows(), matrix.cols()),
        m_hCoeffs((std::min)(matrix.rows(), matrix.cols())),
        m_rows_transpositions((std::min)(matrix.rows(), matrix.cols())),
        m_cols_transpositions((std::min)(matrix.rows(), matrix.cols())),
        m_cols_permutation(matrix.cols()),
        m_temp(matrix.cols()),
        m_isInitialized(false),
        m_usePrescribedThreshold(false)
    {
      compute(matrix);
    }

    /** This method finds a solution x to the equation Ax=b, where A is the matrix of which
      * \c *this is the QR decomposition.
      *
      * \param b the right-hand-side of the equation to solve.
      *
      * \returns the exact or least-square solution if the rank is greater or equal to the number of columns of A,
      * and an arbitrary solution otherwise.
      *
      * \note The case where b is a matrix is not yet implemented. Also, this
      *       code is space inefficient.
      *
      * \note_about_checking_solutions
      *
      * \note_about_arbitrary_choice_of_solution
      *
      * Example: \include FullPivHouseholderQR_solve.cpp
      * Output: \verbinclude FullPivHouseholderQR_solve.out
      */
    template<typename Rhs>
    inline const internal::solve_retval<FullPivHouseholderQR, Rhs>
    solve(const MatrixBase<Rhs>& b) const
    {
      eigen_assert(m_isInitialized && "FullPivHouseholderQR is not initialized.");
      return internal::solve_retval<FullPivHouseholderQR, Rhs>(*this, b.derived());
    }

    /** \returns Expression object representing the matrix Q
      */
    MatrixQReturnType matrixQ(void) const;

    /** \returns a reference to the matrix where the Householder QR decomposition is stored
      */
    const MatrixType& matrixQR() const
    {
      eigen_assert(m_isInitialized && "FullPivHouseholderQR is not initialized.");
      return m_qr;
    }

    FullPivHouseholderQR& compute(const MatrixType& matrix);

    /** \returns a const reference to the column permutation matrix */
    const PermutationType& colsPermutation() const
    {
      eigen_assert(m_isInitialized && "FullPivHouseholderQR is not initialized.");
      return m_cols_permutation;
    }

    /** \returns a const reference to the vector of indices representing the rows transpositions */
    const IntDiagSizeVectorType& rowsTranspositions() const
    {
      eigen_assert(m_isInitialized && "FullPivHouseholderQR is not initialized.");
      return m_rows_transpositions;
    }

    /** \returns the absolute value of the determinant of the matrix of which
      * *this is the QR decomposition. It has only linear complexity
      * (that is, O(n) where n is the dimension of the square matrix)
      * as the QR decomposition has already been computed.
      *
      * \note This is only for square matrices.
      *
      * \warning a determinant can be very big or small, so for matrices
      * of large enough dimension, there is a risk of overflow/underflow.
      * One way to work around that is to use logAbsDeterminant() instead.
      *
      * \sa logAbsDeterminant(), MatrixBase::determinant()
      */
    typename MatrixType::RealScalar absDeterminant() const;

    /** \returns the natural log of the absolute value of the determinant of the matrix of which
      * *this is the QR decomposition. It has only linear complexity
      * (that is, O(n) where n is the dimension of the square matrix)
      * as the QR decomposition has already been computed.
      *
      * \note This is only for square matrices.
      *
      * \note This method is useful to work around the risk of overflow/underflow that's inherent
      * to determinant computation.
      *
      * \sa absDeterminant(), MatrixBase::determinant()
      */
    typename MatrixType::RealScalar logAbsDeterminant() const;

    /** \returns the rank of the matrix of which *this is the QR decomposition.
      *
      * \note This method has to determine which pivots should be considered nonzero.
      *       For that, it uses the threshold value that you can control by calling
      *       setThreshold(const RealScalar&).
      */
    inline Index rank() const
    {
      using std::abs;
      eigen_assert(m_isInitialized && "FullPivHouseholderQR is not initialized.");
      RealScalar premultiplied_threshold = abs(m_maxpivot) * threshold();
      Index result = 0;
      for(Index i = 0; i < m_nonzero_pivots; ++i)
        result += (abs(m_qr.coeff(i,i)) > premultiplied_threshold);
      return result;
    }

    /** \returns the dimension of the kernel of the matrix of which *this is the QR decomposition.
      *
      * \note This method has to determine which pivots should be considered nonzero.
      *       For that, it uses the threshold value that you can control by calling
      *       setThreshold(const RealScalar&).
      */
    inline Index dimensionOfKernel() const
    {
      eigen_assert(m_isInitialized && "FullPivHouseholderQR is not initialized.");
      return cols() - rank();
    }

    /** \returns true if the matrix of which *this is the QR decomposition represents an injective
      *          linear map, i.e. has trivial kernel; false otherwise.
      *
      * \note This method has to determine which pivots should be considered nonzero.
      *       For that, it uses the threshold value that you can control by calling
      *       setThreshold(const RealScalar&).
      */
    inline bool isInjective() const
    {
      eigen_assert(m_isInitialized && "FullPivHouseholderQR is not initialized.");
      return rank() == cols();
    }

    /** \returns true if the matrix of which *this is the QR decomposition represents a surjective
      *          linear map; false otherwise.
      *
      * \note This method has to determine which pivots should be considered nonzero.
      *       For that, it uses the threshold value that you can control by calling
      *       setThreshold(const RealScalar&).
      */
    inline bool isSurjective() const
    {
      eigen_assert(m_isInitialized && "FullPivHouseholderQR is not initialized.");
      return rank() == rows();
    }

    /** \returns true if the matrix of which *this is the QR decomposition is invertible.
      *
      * \note This method has to determine which pivots should be considered nonzero.
      *       For that, it uses the threshold value that you can control by calling
      *       setThreshold(const RealScalar&).
      */
    inline bool isInvertible() const
    {
      eigen_assert(m_isInitialized && "FullPivHouseholderQR is not initialized.");
      return isInjective() && isSurjective();
    }

    /** \returns the inverse of the matrix of which *this is the QR decomposition.
      *
      * \note If this matrix is not invertible, the returned matrix has undefined coefficients.
      *       Use isInvertible() to first determine whether this matrix is invertible.
      */    inline const
    internal::solve_retval<FullPivHouseholderQR, typename MatrixType::IdentityReturnType>
    inverse() const
    {
      eigen_assert(m_isInitialized && "FullPivHouseholderQR is not initialized.");
      return internal::solve_retval<FullPivHouseholderQR,typename MatrixType::IdentityReturnType>
               (*this, MatrixType::Identity(m_qr.rows(), m_qr.cols()));
    }

    inline Index rows() const { return m_qr.rows(); }
    inline Index cols() const { return m_qr.cols(); }
    
    /** \returns a const reference to the vector of Householder coefficients used to represent the factor \c Q.
      * 
      * For advanced uses only.
      */
    const HCoeffsType& hCoeffs() const { return m_hCoeffs; }

    /** Allows to prescribe a threshold to be used by certain methods, such as rank(),
      * who need to determine when pivots are to be considered nonzero. This is not used for the
      * QR decomposition itself.
      *
      * When it needs to get the threshold value, Eigen calls threshold(). By default, this
      * uses a formula to automatically determine a reasonable threshold.
      * Once you have called the present method setThreshold(const RealScalar&),
      * your value is used instead.
      *
      * \param threshold The new value to use as the threshold.
      *
      * A pivot will be considered nonzero if its absolute value is strictly greater than
      *  \f$ \vert pivot \vert \leqslant threshold \times \vert maxpivot \vert \f$
      * where maxpivot is the biggest pivot.
      *
      * If you want to come back to the default behavior, call setThreshold(Default_t)
      */
    FullPivHouseholderQR& setThreshold(const RealScalar& threshold)
    {
      m_usePrescribedThreshold = true;
      m_prescribedThreshold = threshold;
      return *this;
    }

    /** Allows to come back to the default behavior, letting Eigen use its default formula for
      * determining the threshold.
      *
      * You should pass the special object Eigen::Default as parameter here.
      * \code qr.setThreshold(Eigen::Default); \endcode
      *
      * See the documentation of setThreshold(const RealScalar&).
      */
    FullPivHouseholderQR& setThreshold(Default_t)
    {
      m_usePrescribedThreshold = false;
      return *this;
    }

    /** Returns the threshold that will be used by certain methods such as rank().
      *
      * See the documentation of setThreshold(const RealScalar&).
      */
    RealScalar threshold() const
    {
      eigen_assert(m_isInitialized || m_usePrescribedThreshold);
      return m_usePrescribedThreshold ? m_prescribedThreshold
      // this formula comes from experimenting (see "LU precision tuning" thread on the list)
      // and turns out to be identical to Higham's formula used already in LDLt.
                                      : NumTraits<Scalar>::epsilon() * RealScalar(m_qr.diagonalSize());
    }

    /** \returns the number of nonzero pivots in the QR decomposition.
      * Here nonzero is meant in the exact sense, not in a fuzzy sense.
      * So that notion isn't really intrinsically interesting, but it is
      * still useful when implementing algorithms.
      *
      * \sa rank()
      */
    inline Index nonzeroPivots() const
    {
      eigen_assert(m_isInitialized && "LU is not initialized.");
      return m_nonzero_pivots;
    }

    /** \returns the absolute value of the biggest pivot, i.e. the biggest
      *          diagonal coefficient of U.
      */
    RealScalar maxPivot() const { return m_maxpivot; }

  protected:
    MatrixType m_qr;
    HCoeffsType m_hCoeffs;
    IntDiagSizeVectorType m_rows_transpositions;
    IntDiagSizeVectorType m_cols_transpositions;
    PermutationType m_cols_permutation;
    RowVectorType m_temp;
    bool m_isInitialized, m_usePrescribedThreshold;
    RealScalar m_prescribedThreshold, m_maxpivot;
    Index m_nonzero_pivots;
    RealScalar m_precision;
    Index m_det_pq;
};

template<typename MatrixType>
typename MatrixType::RealScalar FullPivHouseholderQR<MatrixType>::absDeterminant() const
{
  using std::abs;
  eigen_assert(m_isInitialized && "FullPivHouseholderQR is not initialized.");
  eigen_assert(m_qr.rows() == m_qr.cols() && "You can't take the determinant of a non-square matrix!");
  return abs(m_qr.diagonal().prod());
}

template<typename MatrixType>
typename MatrixType::RealScalar FullPivHouseholderQR<MatrixType>::logAbsDeterminant() const
{
  eigen_assert(m_isInitialized && "FullPivHouseholderQR is not initialized.");
  eigen_assert(m_qr.rows() == m_qr.cols() && "You can't take the determinant of a non-square matrix!");
  return m_qr.diagonal().cwiseAbs().array().log().sum();
}

/** Performs the QR factorization of the given matrix \a matrix. The result of
  * the factorization is stored into \c *this, and a reference to \c *this
  * is returned.
  *
  * \sa class FullPivHouseholderQR, FullPivHouseholderQR(const MatrixType&)
  */
template<typename MatrixType>
FullPivHouseholderQR<MatrixType>& FullPivHouseholderQR<MatrixType>::compute(const MatrixType& matrix)
{
  using std::abs;
  Index rows = matrix.rows();
  Index cols = matrix.cols();
  Index size = (std::min)(rows,cols);

  m_qr = matrix;
  m_hCoeffs.resize(size);

  m_temp.resize(cols);

  m_precision = NumTraits<Scalar>::epsilon() * RealScalar(size);

  m_rows_transpositions.resize(size);
  m_cols_transpositions.resize(size);
  Index number_of_transpositions = 0;

  RealScalar biggest(0);

  m_nonzero_pivots = size; // the generic case is that in which all pivots are nonzero (invertible case)
  m_maxpivot = RealScalar(0);

  for (Index k = 0; k < size; ++k)
  {
    Index row_of_biggest_in_corner, col_of_biggest_in_corner;
    RealScalar biggest_in_corner;

    biggest_in_corner = m_qr.bottomRightCorner(rows-k, cols-k)
                            .cwiseAbs()
                            .maxCoeff(&row_of_biggest_in_corner, &col_of_biggest_in_corner);
    row_of_biggest_in_corner += k;
    col_of_biggest_in_corner += k;
    if(k==0) biggest = biggest_in_corner;

    // if the corner is negligible, then we have less than full rank, and we can finish early
    if(internal::isMuchSmallerThan(biggest_in_corner, biggest, m_precision))
    {
      m_nonzero_pivots = k;
      for(Index i = k; i < size; i++)
      {
        m_rows_transpositions.coeffRef(i) = i;
        m_cols_transpositions.coeffRef(i) = i;
        m_hCoeffs.coeffRef(i) = Scalar(0);
      }
      break;
    }

    m_rows_transpositions.coeffRef(k) = row_of_biggest_in_corner;
    m_cols_transpositions.coeffRef(k) = col_of_biggest_in_corner;
    if(k != row_of_biggest_in_corner) {
      m_qr.row(k).tail(cols-k).swap(m_qr.row(row_of_biggest_in_corner).tail(cols-k));
      ++number_of_transpositions;
    }
    if(k != col_of_biggest_in_corner) {
      m_qr.col(k).swap(m_qr.col(col_of_biggest_in_corner));
      ++number_of_transpositions;
    }

    RealScalar beta;
    m_qr.col(k).tail(rows-k).makeHouseholderInPlace(m_hCoeffs.coeffRef(k), beta);
    m_qr.coeffRef(k,k) = beta;

    // remember the maximum absolute value of diagonal coefficients
    if(abs(beta) > m_maxpivot) m_maxpivot = abs(beta);

    m_qr.bottomRightCorner(rows-k, cols-k-1)
        .applyHouseholderOnTheLeft(m_qr.col(k).tail(rows-k-1), m_hCoeffs.coeffRef(k), &m_temp.coeffRef(k+1));
  }

  m_cols_permutation.setIdentity(cols);
  for(Index k = 0; k < size; ++k)
    m_cols_permutation.applyTranspositionOnTheRight(k, m_cols_transpositions.coeff(k));

  m_det_pq = (number_of_transpositions%2) ? -1 : 1;
  m_isInitialized = true;

  return *this;
}

namespace internal {

template<typename _MatrixType, typename Rhs>
struct solve_retval<FullPivHouseholderQR<_MatrixType>, Rhs>
  : solve_retval_base<FullPivHouseholderQR<_MatrixType>, Rhs>
{
  EIGEN_MAKE_SOLVE_HELPERS(FullPivHouseholderQR<_MatrixType>,Rhs)

  template<typename Dest> void evalTo(Dest& dst) const
  {
    const Index rows = dec().rows(), cols = dec().cols();
    eigen_assert(rhs().rows() == rows);

    // FIXME introduce nonzeroPivots() and use it here. and more generally,
    // make the same improvements in this dec as in FullPivLU.
    if(dec().rank()==0)
    {
      dst.setZero();
      return;
    }

    typename Rhs::PlainObject c(rhs());

    Matrix<Scalar,1,Rhs::ColsAtCompileTime> temp(rhs().cols());
    for (Index k = 0; k < dec().rank(); ++k)
    {
      Index remainingSize = rows-k;
      c.row(k).swap(c.row(dec().rowsTranspositions().coeff(k)));
      c.bottomRightCorner(remainingSize, rhs().cols())
       .applyHouseholderOnTheLeft(dec().matrixQR().col(k).tail(remainingSize-1),
                                  dec().hCoeffs().coeff(k), &temp.coeffRef(0));
    }

    dec().matrixQR()
       .topLeftCorner(dec().rank(), dec().rank())
       .template triangularView<Upper>()
       .solveInPlace(c.topRows(dec().rank()));

    for(Index i = 0; i < dec().rank(); ++i) dst.row(dec().colsPermutation().indices().coeff(i)) = c.row(i);
    for(Index i = dec().rank(); i < cols; ++i) dst.row(dec().colsPermutation().indices().coeff(i)).setZero();
  }
};

/** \ingroup QR_Module
  *
  * \brief Expression type for return value of FullPivHouseholderQR::matrixQ()
  *
  * \tparam MatrixType type of underlying dense matrix
  */
template<typename MatrixType> struct FullPivHouseholderQRMatrixQReturnType
  : public ReturnByValue<FullPivHouseholderQRMatrixQReturnType<MatrixType> >
{
public:
  typedef typename MatrixType::Index Index;
  typedef typename FullPivHouseholderQR<MatrixType>::IntDiagSizeVectorType IntDiagSizeVectorType;
  typedef typename internal::plain_diag_type<MatrixType>::type HCoeffsType;
  typedef Matrix<typename MatrixType::Scalar, 1, MatrixType::RowsAtCompileTime, RowMajor, 1,
                 MatrixType::MaxRowsAtCompileTime> WorkVectorType;

  FullPivHouseholderQRMatrixQReturnType(const MatrixType&       qr,
                                        const HCoeffsType&      hCoeffs,
                                        const IntDiagSizeVectorType& rowsTranspositions)
    : m_qr(qr),
      m_hCoeffs(hCoeffs),
      m_rowsTranspositions(rowsTranspositions)
      {}

  template <typename ResultType>
  void evalTo(ResultType& result) const
  {
    const Index rows = m_qr.rows();
    WorkVectorType workspace(rows);
    evalTo(result, workspace);
  }

  template <typename ResultType>
  void evalTo(ResultType& result, WorkVectorType& workspace) const
  {
    using numext::conj;
    // compute the product H'_0 H'_1 ... H'_n-1,
    // where H_k is the k-th Householder transformation I - h_k v_k v_k'
    // and v_k is the k-th Householder vector [1,m_qr(k+1,k), m_qr(k+2,k), ...]
    const Index rows = m_qr.rows();
    const Index cols = m_qr.cols();
    const Index size = (std::min)(rows, cols);
    workspace.resize(rows);
    result.setIdentity(rows, rows);
    for (Index k = size-1; k >= 0; k--)
    {
      result.block(k, k, rows-k, rows-k)
            .applyHouseholderOnTheLeft(m_qr.col(k).tail(rows-k-1), conj(m_hCoeffs.coeff(k)), &workspace.coeffRef(k));
      result.row(k).swap(result.row(m_rowsTranspositions.coeff(k)));
    }
  }

    Index rows() const { return m_qr.rows(); }
    Index cols() const { return m_qr.rows(); }

protected:
  typename MatrixType::Nested m_qr;
  typename HCoeffsType::Nested m_hCoeffs;
  typename IntDiagSizeVectorType::Nested m_rowsTranspositions;
};

} // end namespace internal

template<typename MatrixType>
inline typename FullPivHouseholderQR<MatrixType>::MatrixQReturnType FullPivHouseholderQR<MatrixType>::matrixQ() const
{
  eigen_assert(m_isInitialized && "FullPivHouseholderQR is not initialized.");
  return MatrixQReturnType(m_qr, m_hCoeffs, m_rows_transpositions);
}

#ifndef __CUDACC__
/** \return the full-pivoting Householder QR decomposition of \c *this.
  *
  * \sa class FullPivHouseholderQR
  */
template<typename Derived>
const FullPivHouseholderQR<typename MatrixBase<Derived>::PlainObject>
MatrixBase<Derived>::fullPivHouseholderQr() const
{
  return FullPivHouseholderQR<PlainObject>(eval());
}
#endif // __CUDACC__

} // end namespace Eigen

#endif // EIGEN_FULLPIVOTINGHOUSEHOLDERQR_H
