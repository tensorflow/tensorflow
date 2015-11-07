// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_COLPIVOTINGHOUSEHOLDERQR_H
#define EIGEN_COLPIVOTINGHOUSEHOLDERQR_H

namespace Eigen { 

/** \ingroup QR_Module
  *
  * \class ColPivHouseholderQR
  *
  * \brief Householder rank-revealing QR decomposition of a matrix with column-pivoting
  *
  * \param MatrixType the type of the matrix of which we are computing the QR decomposition
  *
  * This class performs a rank-revealing QR decomposition of a matrix \b A into matrices \b P, \b Q and \b R
  * such that 
  * \f[
  *  \mathbf{A} \, \mathbf{P} = \mathbf{Q} \, \mathbf{R}
  * \f]
  * by using Householder transformations. Here, \b P is a permutation matrix, \b Q a unitary matrix and \b R an 
  * upper triangular matrix.
  *
  * This decomposition performs column pivoting in order to be rank-revealing and improve
  * numerical stability. It is slower than HouseholderQR, and faster than FullPivHouseholderQR.
  *
  * \sa MatrixBase::colPivHouseholderQr()
  */
template<typename _MatrixType> class ColPivHouseholderQR
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
    typedef Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime, Options, MaxRowsAtCompileTime, MaxRowsAtCompileTime> MatrixQType;
    typedef typename internal::plain_diag_type<MatrixType>::type HCoeffsType;
    typedef PermutationMatrix<ColsAtCompileTime, MaxColsAtCompileTime> PermutationType;
    typedef typename internal::plain_row_type<MatrixType, Index>::type IntRowVectorType;
    typedef typename internal::plain_row_type<MatrixType>::type RowVectorType;
    typedef typename internal::plain_row_type<MatrixType, RealScalar>::type RealRowVectorType;
    typedef HouseholderSequence<MatrixType,typename internal::remove_all<typename HCoeffsType::ConjugateReturnType>::type> HouseholderSequenceType;
    
  private:
    
    typedef typename PermutationType::Index PermIndexType;
    
  public:

    /**
    * \brief Default Constructor.
    *
    * The default constructor is useful in cases in which the user intends to
    * perform decompositions via ColPivHouseholderQR::compute(const MatrixType&).
    */
    ColPivHouseholderQR()
      : m_qr(),
        m_hCoeffs(),
        m_colsPermutation(),
        m_colsTranspositions(),
        m_temp(),
        m_colSqNorms(),
        m_isInitialized(false),
        m_usePrescribedThreshold(false) {}

    /** \brief Default Constructor with memory preallocation
      *
      * Like the default constructor but with preallocation of the internal data
      * according to the specified problem \a size.
      * \sa ColPivHouseholderQR()
      */
    ColPivHouseholderQR(Index rows, Index cols)
      : m_qr(rows, cols),
        m_hCoeffs((std::min)(rows,cols)),
        m_colsPermutation(PermIndexType(cols)),
        m_colsTranspositions(cols),
        m_temp(cols),
        m_colSqNorms(cols),
        m_isInitialized(false),
        m_usePrescribedThreshold(false) {}

    /** \brief Constructs a QR factorization from a given matrix
      *
      * This constructor computes the QR factorization of the matrix \a matrix by calling
      * the method compute(). It is a short cut for:
      * 
      * \code
      * ColPivHouseholderQR<MatrixType> qr(matrix.rows(), matrix.cols());
      * qr.compute(matrix);
      * \endcode
      * 
      * \sa compute()
      */
    ColPivHouseholderQR(const MatrixType& matrix)
      : m_qr(matrix.rows(), matrix.cols()),
        m_hCoeffs((std::min)(matrix.rows(),matrix.cols())),
        m_colsPermutation(PermIndexType(matrix.cols())),
        m_colsTranspositions(matrix.cols()),
        m_temp(matrix.cols()),
        m_colSqNorms(matrix.cols()),
        m_isInitialized(false),
        m_usePrescribedThreshold(false)
    {
      compute(matrix);
    }

    /** This method finds a solution x to the equation Ax=b, where A is the matrix of which
      * *this is the QR decomposition, if any exists.
      *
      * \param b the right-hand-side of the equation to solve.
      *
      * \returns a solution.
      *
      * \note The case where b is a matrix is not yet implemented. Also, this
      *       code is space inefficient.
      *
      * \note_about_checking_solutions
      *
      * \note_about_arbitrary_choice_of_solution
      *
      * Example: \include ColPivHouseholderQR_solve.cpp
      * Output: \verbinclude ColPivHouseholderQR_solve.out
      */
    template<typename Rhs>
    inline const internal::solve_retval<ColPivHouseholderQR, Rhs>
    solve(const MatrixBase<Rhs>& b) const
    {
      eigen_assert(m_isInitialized && "ColPivHouseholderQR is not initialized.");
      return internal::solve_retval<ColPivHouseholderQR, Rhs>(*this, b.derived());
    }

    HouseholderSequenceType householderQ(void) const;
    HouseholderSequenceType matrixQ(void) const
    {
      return householderQ(); 
    }

    /** \returns a reference to the matrix where the Householder QR decomposition is stored
      */
    const MatrixType& matrixQR() const
    {
      eigen_assert(m_isInitialized && "ColPivHouseholderQR is not initialized.");
      return m_qr;
    }
    
    /** \returns a reference to the matrix where the result Householder QR is stored 
     * \warning The strict lower part of this matrix contains internal values. 
     * Only the upper triangular part should be referenced. To get it, use
     * \code matrixR().template triangularView<Upper>() \endcode
     * For rank-deficient matrices, use 
     * \code 
     * matrixR().topLeftCorner(rank(), rank()).template triangularView<Upper>() 
     * \endcode
     */
    const MatrixType& matrixR() const
    {
      eigen_assert(m_isInitialized && "ColPivHouseholderQR is not initialized.");
      return m_qr;
    }
    
    ColPivHouseholderQR& compute(const MatrixType& matrix);

    /** \returns a const reference to the column permutation matrix */
    const PermutationType& colsPermutation() const
    {
      eigen_assert(m_isInitialized && "ColPivHouseholderQR is not initialized.");
      return m_colsPermutation;
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
      eigen_assert(m_isInitialized && "ColPivHouseholderQR is not initialized.");
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
      eigen_assert(m_isInitialized && "ColPivHouseholderQR is not initialized.");
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
      eigen_assert(m_isInitialized && "ColPivHouseholderQR is not initialized.");
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
      eigen_assert(m_isInitialized && "ColPivHouseholderQR is not initialized.");
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
      eigen_assert(m_isInitialized && "ColPivHouseholderQR is not initialized.");
      return isInjective() && isSurjective();
    }

    /** \returns the inverse of the matrix of which *this is the QR decomposition.
      *
      * \note If this matrix is not invertible, the returned matrix has undefined coefficients.
      *       Use isInvertible() to first determine whether this matrix is invertible.
      */
    inline const
    internal::solve_retval<ColPivHouseholderQR, typename MatrixType::IdentityReturnType>
    inverse() const
    {
      eigen_assert(m_isInitialized && "ColPivHouseholderQR is not initialized.");
      return internal::solve_retval<ColPivHouseholderQR,typename MatrixType::IdentityReturnType>
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
    ColPivHouseholderQR& setThreshold(const RealScalar& threshold)
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
    ColPivHouseholderQR& setThreshold(Default_t)
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
      eigen_assert(m_isInitialized && "ColPivHouseholderQR is not initialized.");
      return m_nonzero_pivots;
    }

    /** \returns the absolute value of the biggest pivot, i.e. the biggest
      *          diagonal coefficient of R.
      */
    RealScalar maxPivot() const { return m_maxpivot; }
    
    /** \brief Reports whether the QR factorization was succesful.
      *
      * \note This function always returns \c Success. It is provided for compatibility 
      * with other factorization routines.
      * \returns \c Success 
      */
    ComputationInfo info() const
    {
      eigen_assert(m_isInitialized && "Decomposition is not initialized.");
      return Success;
    }

  protected:
    MatrixType m_qr;
    HCoeffsType m_hCoeffs;
    PermutationType m_colsPermutation;
    IntRowVectorType m_colsTranspositions;
    RowVectorType m_temp;
    RealRowVectorType m_colSqNorms;
    bool m_isInitialized, m_usePrescribedThreshold;
    RealScalar m_prescribedThreshold, m_maxpivot;
    Index m_nonzero_pivots;
    Index m_det_pq;
};

template<typename MatrixType>
typename MatrixType::RealScalar ColPivHouseholderQR<MatrixType>::absDeterminant() const
{
  using std::abs;
  eigen_assert(m_isInitialized && "ColPivHouseholderQR is not initialized.");
  eigen_assert(m_qr.rows() == m_qr.cols() && "You can't take the determinant of a non-square matrix!");
  return abs(m_qr.diagonal().prod());
}

template<typename MatrixType>
typename MatrixType::RealScalar ColPivHouseholderQR<MatrixType>::logAbsDeterminant() const
{
  eigen_assert(m_isInitialized && "ColPivHouseholderQR is not initialized.");
  eigen_assert(m_qr.rows() == m_qr.cols() && "You can't take the determinant of a non-square matrix!");
  return m_qr.diagonal().cwiseAbs().array().log().sum();
}

/** Performs the QR factorization of the given matrix \a matrix. The result of
  * the factorization is stored into \c *this, and a reference to \c *this
  * is returned.
  *
  * \sa class ColPivHouseholderQR, ColPivHouseholderQR(const MatrixType&)
  */
template<typename MatrixType>
ColPivHouseholderQR<MatrixType>& ColPivHouseholderQR<MatrixType>::compute(const MatrixType& matrix)
{
  using std::abs;
  Index rows = matrix.rows();
  Index cols = matrix.cols();
  Index size = matrix.diagonalSize();
  
  // the column permutation is stored as int indices, so just to be sure:
  eigen_assert(cols<=NumTraits<int>::highest());

  m_qr = matrix;
  m_hCoeffs.resize(size);

  m_temp.resize(cols);

  m_colsTranspositions.resize(matrix.cols());
  Index number_of_transpositions = 0;

  m_colSqNorms.resize(cols);
  for(Index k = 0; k < cols; ++k)
    m_colSqNorms.coeffRef(k) = m_qr.col(k).squaredNorm();

  RealScalar threshold_helper = m_colSqNorms.maxCoeff() * numext::abs2(NumTraits<Scalar>::epsilon()) / RealScalar(rows);

  m_nonzero_pivots = size; // the generic case is that in which all pivots are nonzero (invertible case)
  m_maxpivot = RealScalar(0);

  for(Index k = 0; k < size; ++k)
  {
    // first, we look up in our table m_colSqNorms which column has the biggest squared norm
    Index biggest_col_index;
    RealScalar biggest_col_sq_norm = m_colSqNorms.tail(cols-k).maxCoeff(&biggest_col_index);
    biggest_col_index += k;

    // since our table m_colSqNorms accumulates imprecision at every step, we must now recompute
    // the actual squared norm of the selected column.
    // Note that not doing so does result in solve() sometimes returning inf/nan values
    // when running the unit test with 1000 repetitions.
    biggest_col_sq_norm = m_qr.col(biggest_col_index).tail(rows-k).squaredNorm();

    // we store that back into our table: it can't hurt to correct our table.
    m_colSqNorms.coeffRef(biggest_col_index) = biggest_col_sq_norm;

    // if the current biggest column is smaller than epsilon times the initial biggest column,
    // terminate to avoid generating nan/inf values.
    // Note that here, if we test instead for "biggest == 0", we get a failure every 1000 (or so)
    // repetitions of the unit test, with the result of solve() filled with large values of the order
    // of 1/(size*epsilon).
    if(biggest_col_sq_norm < threshold_helper * RealScalar(rows-k))
    {
      m_nonzero_pivots = k;
      m_hCoeffs.tail(size-k).setZero();
      m_qr.bottomRightCorner(rows-k,cols-k)
          .template triangularView<StrictlyLower>()
          .setZero();
      break;
    }

    // apply the transposition to the columns
    m_colsTranspositions.coeffRef(k) = biggest_col_index;
    if(k != biggest_col_index) {
      m_qr.col(k).swap(m_qr.col(biggest_col_index));
      std::swap(m_colSqNorms.coeffRef(k), m_colSqNorms.coeffRef(biggest_col_index));
      ++number_of_transpositions;
    }

    // generate the householder vector, store it below the diagonal
    RealScalar beta;
    m_qr.col(k).tail(rows-k).makeHouseholderInPlace(m_hCoeffs.coeffRef(k), beta);

    // apply the householder transformation to the diagonal coefficient
    m_qr.coeffRef(k,k) = beta;

    // remember the maximum absolute value of diagonal coefficients
    if(abs(beta) > m_maxpivot) m_maxpivot = abs(beta);

    // apply the householder transformation
    m_qr.bottomRightCorner(rows-k, cols-k-1)
        .applyHouseholderOnTheLeft(m_qr.col(k).tail(rows-k-1), m_hCoeffs.coeffRef(k), &m_temp.coeffRef(k+1));

    // update our table of squared norms of the columns
    m_colSqNorms.tail(cols-k-1) -= m_qr.row(k).tail(cols-k-1).cwiseAbs2();
  }

  m_colsPermutation.setIdentity(PermIndexType(cols));
  for(PermIndexType k = 0; k < m_nonzero_pivots; ++k)
    m_colsPermutation.applyTranspositionOnTheRight(k, PermIndexType(m_colsTranspositions.coeff(k)));

  m_det_pq = (number_of_transpositions%2) ? -1 : 1;
  m_isInitialized = true;

  return *this;
}

namespace internal {

template<typename _MatrixType, typename Rhs>
struct solve_retval<ColPivHouseholderQR<_MatrixType>, Rhs>
  : solve_retval_base<ColPivHouseholderQR<_MatrixType>, Rhs>
{
  EIGEN_MAKE_SOLVE_HELPERS(ColPivHouseholderQR<_MatrixType>,Rhs)

  template<typename Dest> void evalTo(Dest& dst) const
  {
    eigen_assert(rhs().rows() == dec().rows());

    const Index cols = dec().cols(),
				nonzero_pivots = dec().nonzeroPivots();

    if(nonzero_pivots == 0)
    {
      dst.setZero();
      return;
    }

    typename Rhs::PlainObject c(rhs());

    // Note that the matrix Q = H_0^* H_1^*... so its inverse is Q^* = (H_0 H_1 ...)^T
    c.applyOnTheLeft(householderSequence(dec().matrixQR(), dec().hCoeffs())
                     .setLength(dec().nonzeroPivots())
		     .transpose()
      );

    dec().matrixR()
       .topLeftCorner(nonzero_pivots, nonzero_pivots)
       .template triangularView<Upper>()
       .solveInPlace(c.topRows(nonzero_pivots));

    for(Index i = 0; i < nonzero_pivots; ++i) dst.row(dec().colsPermutation().indices().coeff(i)) = c.row(i);
    for(Index i = nonzero_pivots; i < cols; ++i) dst.row(dec().colsPermutation().indices().coeff(i)).setZero();
  }
};

} // end namespace internal

/** \returns the matrix Q as a sequence of householder transformations */
template<typename MatrixType>
typename ColPivHouseholderQR<MatrixType>::HouseholderSequenceType ColPivHouseholderQR<MatrixType>
  ::householderQ() const
{
  eigen_assert(m_isInitialized && "ColPivHouseholderQR is not initialized.");
  return HouseholderSequenceType(m_qr, m_hCoeffs.conjugate()).setLength(m_nonzero_pivots);
}

#ifndef __CUDACC__
/** \return the column-pivoting Householder QR decomposition of \c *this.
  *
  * \sa class ColPivHouseholderQR
  */
template<typename Derived>
const ColPivHouseholderQR<typename MatrixBase<Derived>::PlainObject>
MatrixBase<Derived>::colPivHouseholderQr() const
{
  return ColPivHouseholderQR<PlainObject>(eval());
}
#endif // __CUDACC__

} // end namespace Eigen

#endif // EIGEN_COLPIVOTINGHOUSEHOLDERQR_H
