// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// This file is a base class plugin containing common coefficient wise functions.

#ifndef EIGEN_PARSED_BY_DOXYGEN

/** \internal Represents a scalar multiple of an expression */
typedef CwiseUnaryOp<internal::scalar_multiple_op<Scalar>, const Derived> ScalarMultipleReturnType;
/** \internal Represents a quotient of an expression by a scalar*/
typedef CwiseUnaryOp<internal::scalar_quotient1_op<Scalar>, const Derived> ScalarQuotient1ReturnType;
/** \internal the return type of conjugate() */
typedef typename internal::conditional<NumTraits<Scalar>::IsComplex,
                    const CwiseUnaryOp<internal::scalar_conjugate_op<Scalar>, const Derived>,
                    const Derived&
                  >::type ConjugateReturnType;
/** \internal the return type of real() const */
typedef typename internal::conditional<NumTraits<Scalar>::IsComplex,
                    const CwiseUnaryOp<internal::scalar_real_op<Scalar>, const Derived>,
                    const Derived&
                  >::type RealReturnType;
/** \internal the return type of real() */
typedef typename internal::conditional<NumTraits<Scalar>::IsComplex,
                    CwiseUnaryView<internal::scalar_real_ref_op<Scalar>, Derived>,
                    Derived&
                  >::type NonConstRealReturnType;
/** \internal the return type of imag() const */
typedef CwiseUnaryOp<internal::scalar_imag_op<Scalar>, const Derived> ImagReturnType;
/** \internal the return type of imag() */
typedef CwiseUnaryView<internal::scalar_imag_ref_op<Scalar>, Derived> NonConstImagReturnType;

#endif // not EIGEN_PARSED_BY_DOXYGEN

/** \returns an expression of the opposite of \c *this
  */
EIGEN_DEVICE_FUNC
inline const CwiseUnaryOp<internal::scalar_opposite_op<typename internal::traits<Derived>::Scalar>, const Derived>
operator-() const { return derived(); }


/** \returns an expression of \c *this scaled by the scalar factor \a scalar */
EIGEN_DEVICE_FUNC
inline const ScalarMultipleReturnType
operator*(const Scalar& scalar) const
{
  return CwiseUnaryOp<internal::scalar_multiple_op<Scalar>, const Derived>
    (derived(), internal::scalar_multiple_op<Scalar>(scalar));
}

#ifdef EIGEN_PARSED_BY_DOXYGEN
const ScalarMultipleReturnType operator*(const RealScalar& scalar) const;
#endif

/** \returns an expression of \c *this divided by the scalar value \a scalar */
EIGEN_DEVICE_FUNC
inline const CwiseUnaryOp<internal::scalar_quotient1_op<typename internal::traits<Derived>::Scalar>, const Derived>
operator/(const Scalar& scalar) const
{
  return CwiseUnaryOp<internal::scalar_quotient1_op<Scalar>, const Derived>
    (derived(), internal::scalar_quotient1_op<Scalar>(scalar));
}

/** Overloaded for efficient real matrix times complex scalar value */
EIGEN_DEVICE_FUNC
inline const CwiseUnaryOp<internal::scalar_multiple2_op<Scalar,std::complex<Scalar> >, const Derived>
operator*(const std::complex<Scalar>& scalar) const
{
  return CwiseUnaryOp<internal::scalar_multiple2_op<Scalar,std::complex<Scalar> >, const Derived>
    (*static_cast<const Derived*>(this), internal::scalar_multiple2_op<Scalar,std::complex<Scalar> >(scalar));
}

EIGEN_DEVICE_FUNC
inline friend const ScalarMultipleReturnType
operator*(const Scalar& scalar, const StorageBaseType& matrix)
{ return matrix*scalar; }

EIGEN_DEVICE_FUNC
inline friend const CwiseUnaryOp<internal::scalar_multiple2_op<Scalar,std::complex<Scalar> >, const Derived>
operator*(const std::complex<Scalar>& scalar, const StorageBaseType& matrix)
{ return matrix*scalar; }

/** \returns an expression of *this with the \a Scalar type casted to
  * \a NewScalar.
  *
  * The template parameter \a NewScalar is the type we are casting the scalars to.
  *
  * \sa class CwiseUnaryOp
  */
template<typename NewType>
EIGEN_DEVICE_FUNC
typename internal::cast_return_type<Derived,const CwiseUnaryOp<internal::scalar_cast_op<typename internal::traits<Derived>::Scalar, NewType>, const Derived> >::type
cast() const
{
  return derived();
}

/** \returns an expression of *this with the \a Scalar type converted to
  * \a NewScalar using the custom conversion functor \a ConvertOp.
  *
  * The template parameter \a NewType is the type we are casting the scalars to.
  * The template parameter \a ConvertOp is the conversion functor.
  *
  * \sa class CwiseUnaryOp
  */
template<typename NewType, typename ConvertOp>
typename internal::cast_return_type<Derived,const CwiseUnaryOp<internal::scalar_convert_op<typename internal::traits<Derived>::Scalar, NewType, ConvertOp>, const Derived> >::type
convert() const
{
  return derived();
}

/** \returns an expression of the complex conjugate of \c *this.
  *
  * \sa adjoint() */
EIGEN_DEVICE_FUNC
inline ConjugateReturnType
conjugate() const
{
  return ConjugateReturnType(derived());
}

/** \returns a read-only expression of the real part of \c *this.
  *
  * \sa imag() */
EIGEN_DEVICE_FUNC
inline RealReturnType
real() const { return derived(); }

/** \returns an read-only expression of the imaginary part of \c *this.
  *
  * \sa real() */
EIGEN_DEVICE_FUNC
inline const ImagReturnType
imag() const { return derived(); }

/** \brief Apply a unary operator coefficient-wise
  * \param[in]  func  Functor implementing the unary operator
  * \tparam  CustomUnaryOp Type of \a func  
  * \returns An expression of a custom coefficient-wise unary operator \a func of *this
  *
  * The function \c ptr_fun() from the C++ standard library can be used to make functors out of normal functions.
  *
  * Example:
  * \include class_CwiseUnaryOp_ptrfun.cpp
  * Output: \verbinclude class_CwiseUnaryOp_ptrfun.out
  *
  * Genuine functors allow for more possibilities, for instance it may contain a state.
  *
  * Example:
  * \include class_CwiseUnaryOp.cpp
  * Output: \verbinclude class_CwiseUnaryOp.out
  *
  * \sa class CwiseUnaryOp, class CwiseBinaryOp
  */
template<typename CustomUnaryOp>
EIGEN_DEVICE_FUNC
inline const CwiseUnaryOp<CustomUnaryOp, const Derived>
unaryExpr(const CustomUnaryOp& func = CustomUnaryOp()) const
{
  return CwiseUnaryOp<CustomUnaryOp, const Derived>(derived(), func);
}

/** \returns an expression of a custom coefficient-wise unary operator \a func of *this
  *
  * The template parameter \a CustomUnaryOp is the type of the functor
  * of the custom unary operator.
  *
  * Example:
  * \include class_CwiseUnaryOp.cpp
  * Output: \verbinclude class_CwiseUnaryOp.out
  *
  * \sa class CwiseUnaryOp, class CwiseBinaryOp
  */
template<typename CustomViewOp>
EIGEN_DEVICE_FUNC
inline const CwiseUnaryView<CustomViewOp, const Derived>
unaryViewExpr(const CustomViewOp& func = CustomViewOp()) const
{
  return CwiseUnaryView<CustomViewOp, const Derived>(derived(), func);
}

/** \returns a non const expression of the real part of \c *this.
  *
  * \sa imag() */
EIGEN_DEVICE_FUNC
inline NonConstRealReturnType
real() { return derived(); }

/** \returns a non const expression of the imaginary part of \c *this.
  *
  * \sa real() */
EIGEN_DEVICE_FUNC
inline NonConstImagReturnType
imag() { return derived(); }
