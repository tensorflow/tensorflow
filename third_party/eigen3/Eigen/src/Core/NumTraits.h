// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_NUMTRAITS_H
#define EIGEN_NUMTRAITS_H

namespace Eigen {

/** \class NumTraits
  * \ingroup Core_Module
  *
  * \brief Holds information about the various numeric (i.e. scalar) types allowed by Eigen.
  *
  * \param T the numeric type at hand
  *
  * This class stores enums, typedefs and static methods giving information about a numeric type.
  *
  * The provided data consists of:
  * \li A typedef \a Real, giving the "real part" type of \a T. If \a T is already real,
  *     then \a Real is just a typedef to \a T. If \a T is \c std::complex<U> then \a Real
  *     is a typedef to \a U.
  * \li A typedef \a NonInteger, giving the type that should be used for operations producing non-integral values,
  *     such as quotients, square roots, etc. If \a T is a floating-point type, then this typedef just gives
  *     \a T again. Note however that many Eigen functions such as internal::sqrt simply refuse to
  *     take integers. Outside of a few cases, Eigen doesn't do automatic type promotion. Thus, this typedef is
  *     only intended as a helper for code that needs to explicitly promote types.
  * \li A typedef \a Nested giving the type to use to nest a value inside of the expression tree. If you don't know what
  *     this means, just use \a T here.
  * \li An enum value \a IsComplex. It is equal to 1 if \a T is a \c std::complex
  *     type, and to 0 otherwise.
  * \li An enum value \a IsInteger. It is equal to \c 1 if \a T is an integer type such as \c int,
  *     and to \c 0 otherwise.
  * \li Enum values ReadCost, AddCost and MulCost representing a rough estimate of the number of CPU cycles needed
  *     to by move / add / mul instructions respectively, assuming the data is already stored in CPU registers.
  *     Stay vague here. No need to do architecture-specific stuff.
  * \li An enum value \a IsSigned. It is equal to \c 1 if \a T is a signed type and to 0 if \a T is unsigned.
  * \li An enum value \a RequireInitialization. It is equal to \c 1 if the constructor of the numeric type \a T must
  *     be called, and to 0 if it is safe not to call it. Default is 0 if \a T is an arithmetic type, and 1 otherwise.
  * \li An epsilon() function which, unlike std::numeric_limits::epsilon(), returns a \a Real instead of a \a T.
  * \li A dummy_precision() function returning a weak epsilon value. It is mainly used as a default
  *     value by the fuzzy comparison operators.
  * \li highest() and lowest() functions returning the highest and lowest possible values respectively.
  */

template<typename T> struct GenericNumTraits
{
  enum {
    IsInteger = std::numeric_limits<T>::is_integer,
    IsSigned = std::numeric_limits<T>::is_signed,
    IsComplex = 0,
    RequireInitialization = internal::is_arithmetic<T>::value ? 0 : 1,
    ReadCost = 1,
    AddCost = 1,
    MulCost = 1
  };

  typedef T Real;
  typedef typename internal::conditional<
                     IsInteger,
                     typename internal::conditional<sizeof(T)<=2, float, double>::type,
                     T
                   >::type NonInteger;
  typedef T Nested;

  EIGEN_DEVICE_FUNC
  static inline Real epsilon()
  {
#if defined(__CUDA_ARCH__) && !defined(__GCUDACC__)
    return internal::device::numeric_limits<T>::epsilon();
#else
    return std::numeric_limits<T>::epsilon();
#endif
  }
  EIGEN_DEVICE_FUNC
  static inline Real dummy_precision()
  {
    // make sure to override this for floating-point types
    return Real(0);
  }

  EIGEN_DEVICE_FUNC
  static inline T highest() {
#if defined(__CUDA_ARCH__) && !defined(__GCUDACC__)
    return internal::device::numeric_limits<T>::max();
#else
    return (std::numeric_limits<T>::max)();
#endif
  }

  EIGEN_DEVICE_FUNC
  static inline T lowest()  {
#if defined(__CUDA_ARCH__) && !defined(__GCUDACC__)
    return internal::device::numeric_limits<T>::lowest();
#else
    return IsInteger ? (std::numeric_limits<T>::min)() : (-(std::numeric_limits<T>::max)());
#endif
  }
  
#ifdef EIGEN2_SUPPORT
  enum {
    HasFloatingPoint = !IsInteger
  };
  typedef NonInteger FloatingPoint;
#endif
};

template<typename T> struct NumTraits : GenericNumTraits<T>
{};

template<> struct NumTraits<float>
  : GenericNumTraits<float>
{
  EIGEN_DEVICE_FUNC
  static inline float dummy_precision() { return 1e-5f; }
};

template<> struct NumTraits<double> : GenericNumTraits<double>
{
  EIGEN_DEVICE_FUNC
  static inline double dummy_precision() { return 1e-12; }
};

template<> struct NumTraits<long double>
  : GenericNumTraits<long double>
{
  static inline long double dummy_precision() { return 1e-15l; }
};

template<typename _Real> struct NumTraits<std::complex<_Real> >
  : GenericNumTraits<std::complex<_Real> >
{
  typedef _Real Real;
  enum {
    IsComplex = 1,
    RequireInitialization = NumTraits<_Real>::RequireInitialization,
    ReadCost = 2 * NumTraits<_Real>::ReadCost,
    AddCost = 2 * NumTraits<Real>::AddCost,
    MulCost = 4 * NumTraits<Real>::MulCost + 2 * NumTraits<Real>::AddCost
  };

  static inline Real epsilon() { return NumTraits<Real>::epsilon(); }
  static inline Real dummy_precision() { return NumTraits<Real>::dummy_precision(); }
};

template<typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
struct NumTraits<Array<Scalar, Rows, Cols, Options, MaxRows, MaxCols> >
{
  typedef Array<Scalar, Rows, Cols, Options, MaxRows, MaxCols> ArrayType;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Array<RealScalar, Rows, Cols, Options, MaxRows, MaxCols> Real;
  typedef typename NumTraits<Scalar>::NonInteger NonIntegerScalar;
  typedef Array<NonIntegerScalar, Rows, Cols, Options, MaxRows, MaxCols> NonInteger;
  typedef ArrayType & Nested;
  
  enum {
    IsComplex = NumTraits<Scalar>::IsComplex,
    IsInteger = NumTraits<Scalar>::IsInteger,
    IsSigned  = NumTraits<Scalar>::IsSigned,
    RequireInitialization = 1,
    ReadCost = ArrayType::SizeAtCompileTime==Dynamic ? Dynamic : ArrayType::SizeAtCompileTime * NumTraits<Scalar>::ReadCost,
    AddCost  = ArrayType::SizeAtCompileTime==Dynamic ? Dynamic : ArrayType::SizeAtCompileTime * NumTraits<Scalar>::AddCost,
    MulCost  = ArrayType::SizeAtCompileTime==Dynamic ? Dynamic : ArrayType::SizeAtCompileTime * NumTraits<Scalar>::MulCost
  };
  
  static inline RealScalar epsilon() { return NumTraits<RealScalar>::epsilon(); }
  static inline RealScalar dummy_precision() { return NumTraits<RealScalar>::dummy_precision(); }
};

} // end namespace Eigen

#endif // EIGEN_NUMTRAITS_H
