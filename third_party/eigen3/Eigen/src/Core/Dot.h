// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008, 2010 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_DOT_H
#define EIGEN_DOT_H

namespace Eigen { 

namespace internal {

// helper function for dot(). The problem is that if we put that in the body of dot(), then upon calling dot
// with mismatched types, the compiler emits errors about failing to instantiate cwiseProduct BEFORE
// looking at the static assertions. Thus this is a trick to get better compile errors.
template<typename T, typename U,
// the NeedToTranspose condition here is taken straight from Assign.h
         bool NeedToTranspose = T::IsVectorAtCompileTime
                && U::IsVectorAtCompileTime
                && ((int(T::RowsAtCompileTime) == 1 && int(U::ColsAtCompileTime) == 1)
                      |  // FIXME | instead of || to please GCC 4.4.0 stupid warning "suggest parentheses around &&".
                         // revert to || as soon as not needed anymore.
                    (int(T::ColsAtCompileTime) == 1 && int(U::RowsAtCompileTime) == 1))
>
struct dot_nocheck
{
  typedef typename scalar_product_traits<typename traits<T>::Scalar,typename traits<U>::Scalar>::ReturnType ResScalar;
  EIGEN_DEVICE_FUNC
  static inline ResScalar run(const MatrixBase<T>& a, const MatrixBase<U>& b)
  {
    return a.template binaryExpr<scalar_conj_product_op<typename traits<T>::Scalar,typename traits<U>::Scalar> >(b).sum();
  }
};

template<typename T, typename U>
struct dot_nocheck<T, U, true>
{
  typedef typename scalar_product_traits<typename traits<T>::Scalar,typename traits<U>::Scalar>::ReturnType ResScalar;
  EIGEN_DEVICE_FUNC
  static inline ResScalar run(const MatrixBase<T>& a, const MatrixBase<U>& b)
  {
    return a.transpose().template binaryExpr<scalar_conj_product_op<typename traits<T>::Scalar,typename traits<U>::Scalar> >(b).sum();
  }
};

} // end namespace internal

/** \returns the dot product of *this with other.
  *
  * \only_for_vectors
  *
  * \note If the scalar type is complex numbers, then this function returns the hermitian
  * (sesquilinear) dot product, conjugate-linear in the first variable and linear in the
  * second variable.
  *
  * \sa squaredNorm(), norm()
  */
template<typename Derived>
template<typename OtherDerived>
EIGEN_DEVICE_FUNC
typename internal::scalar_product_traits<typename internal::traits<Derived>::Scalar,typename internal::traits<OtherDerived>::Scalar>::ReturnType
MatrixBase<Derived>::dot(const MatrixBase<OtherDerived>& other) const
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(OtherDerived)
  EIGEN_STATIC_ASSERT_SAME_VECTOR_SIZE(Derived,OtherDerived)
  typedef internal::scalar_conj_product_op<Scalar,typename OtherDerived::Scalar> func;
  EIGEN_CHECK_BINARY_COMPATIBILIY(func,Scalar,typename OtherDerived::Scalar);

  eigen_assert(size() == other.size());

  return internal::dot_nocheck<Derived,OtherDerived>::run(*this, other);
}

#ifdef EIGEN2_SUPPORT
/** \returns the dot product of *this with other, with the Eigen2 convention that the dot product is linear in the first variable
  * (conjugating the second variable). Of course this only makes a difference in the complex case.
  *
  * This method is only available in EIGEN2_SUPPORT mode.
  *
  * \only_for_vectors
  *
  * \sa dot()
  */
template<typename Derived>
template<typename OtherDerived>
typename internal::traits<Derived>::Scalar
MatrixBase<Derived>::eigen2_dot(const MatrixBase<OtherDerived>& other) const
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(OtherDerived)
  EIGEN_STATIC_ASSERT_SAME_VECTOR_SIZE(Derived,OtherDerived)
  EIGEN_STATIC_ASSERT((internal::is_same<Scalar, typename OtherDerived::Scalar>::value),
    YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY)

  eigen_assert(size() == other.size());

  return internal::dot_nocheck<OtherDerived,Derived>::run(other,*this);
}
#endif


//---------- implementation of L2 norm and related functions ----------

/** \returns, for vectors, the squared \em l2 norm of \c *this, and for matrices the Frobenius norm.
  * In both cases, it consists in the sum of the square of all the matrix entries.
  * For vectors, this is also equals to the dot product of \c *this with itself.
  *
  * \sa dot(), norm()
  */
template<typename Derived>
EIGEN_STRONG_INLINE typename NumTraits<typename internal::traits<Derived>::Scalar>::Real MatrixBase<Derived>::squaredNorm() const
{
  return numext::real((*this).cwiseAbs2().sum());
}

/** \returns, for vectors, the \em l2 norm of \c *this, and for matrices the Frobenius norm.
  * In both cases, it consists in the square root of the sum of the square of all the matrix entries.
  * For vectors, this is also equals to the square root of the dot product of \c *this with itself.
  *
  * \sa dot(), squaredNorm()
  */
template<typename Derived>
inline typename NumTraits<typename internal::traits<Derived>::Scalar>::Real MatrixBase<Derived>::norm() const
{
  using std::sqrt;
  return sqrt(squaredNorm());
}

/** \returns an expression of the quotient of *this by its own norm.
  *
  * \only_for_vectors
  *
  * \sa norm(), normalize()
  */
template<typename Derived>
inline const typename MatrixBase<Derived>::PlainObject
MatrixBase<Derived>::normalized() const
{
  typedef typename internal::nested<Derived>::type Nested;
  typedef typename internal::remove_reference<Nested>::type _Nested;
  _Nested n(derived());
  return n / n.norm();
}

/** Normalizes the vector, i.e. divides it by its own norm.
  *
  * \only_for_vectors
  *
  * \sa norm(), normalized()
  */
template<typename Derived>
inline void MatrixBase<Derived>::normalize()
{
  *this /= norm();
}

//---------- implementation of other norms ----------

namespace internal {

template<typename Derived, int p>
struct lpNorm_selector
{
  typedef typename NumTraits<typename traits<Derived>::Scalar>::Real RealScalar;
  EIGEN_DEVICE_FUNC
  static inline RealScalar run(const MatrixBase<Derived>& m)
  {
    using std::pow;
    return pow(m.cwiseAbs().array().pow(p).sum(), RealScalar(1)/p);
  }
};

template<typename Derived>
struct lpNorm_selector<Derived, 1>
{
  EIGEN_DEVICE_FUNC
  static inline typename NumTraits<typename traits<Derived>::Scalar>::Real run(const MatrixBase<Derived>& m)
  {
    return m.cwiseAbs().sum();
  }
};

template<typename Derived>
struct lpNorm_selector<Derived, 2>
{
  EIGEN_DEVICE_FUNC
  static inline typename NumTraits<typename traits<Derived>::Scalar>::Real run(const MatrixBase<Derived>& m)
  {
    return m.norm();
  }
};

template<typename Derived>
struct lpNorm_selector<Derived, Infinity>
{
  EIGEN_DEVICE_FUNC
  static inline typename NumTraits<typename traits<Derived>::Scalar>::Real run(const MatrixBase<Derived>& m)
  {
    return m.cwiseAbs().maxCoeff();
  }
};

} // end namespace internal

/** \returns the \f$ \ell^p \f$ norm of *this, that is, returns the p-th root of the sum of the p-th powers of the absolute values
  *          of the coefficients of *this. If \a p is the special value \a Eigen::Infinity, this function returns the \f$ \ell^\infty \f$
  *          norm, that is the maximum of the absolute values of the coefficients of *this.
  *
  * \sa norm()
  */
template<typename Derived>
template<int p>
inline typename NumTraits<typename internal::traits<Derived>::Scalar>::Real
MatrixBase<Derived>::lpNorm() const
{
  return internal::lpNorm_selector<Derived, p>::run(*this);
}

//---------- implementation of isOrthogonal / isUnitary ----------

/** \returns true if *this is approximately orthogonal to \a other,
  *          within the precision given by \a prec.
  *
  * Example: \include MatrixBase_isOrthogonal.cpp
  * Output: \verbinclude MatrixBase_isOrthogonal.out
  */
template<typename Derived>
template<typename OtherDerived>
bool MatrixBase<Derived>::isOrthogonal
(const MatrixBase<OtherDerived>& other, const RealScalar& prec) const
{
  typename internal::nested<Derived,2>::type nested(derived());
  typename internal::nested<OtherDerived,2>::type otherNested(other.derived());
  return numext::abs2(nested.dot(otherNested)) <= prec * prec * nested.squaredNorm() * otherNested.squaredNorm();
}

/** \returns true if *this is approximately an unitary matrix,
  *          within the precision given by \a prec. In the case where the \a Scalar
  *          type is real numbers, a unitary matrix is an orthogonal matrix, whence the name.
  *
  * \note This can be used to check whether a family of vectors forms an orthonormal basis.
  *       Indeed, \c m.isUnitary() returns true if and only if the columns (equivalently, the rows) of m form an
  *       orthonormal basis.
  *
  * Example: \include MatrixBase_isUnitary.cpp
  * Output: \verbinclude MatrixBase_isUnitary.out
  */
template<typename Derived>
bool MatrixBase<Derived>::isUnitary(const RealScalar& prec) const
{
  typename Derived::Nested nested(derived());
  for(Index i = 0; i < cols(); ++i)
  {
    if(!internal::isApprox(nested.col(i).squaredNorm(), static_cast<RealScalar>(1), prec))
      return false;
    for(Index j = 0; j < i; ++j)
      if(!internal::isMuchSmallerThan(nested.col(i).dot(nested.col(j)), static_cast<Scalar>(1), prec))
        return false;
  }
  return true;
}

} // end namespace Eigen

#endif // EIGEN_DOT_H
