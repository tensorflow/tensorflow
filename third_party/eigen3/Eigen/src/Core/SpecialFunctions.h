// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2015 Eugene Brevdo <ebrevdo@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPECIALFUNCTIONS_H
#define EIGEN_SPECIALFUNCTIONS_H

namespace Eigen {

namespace internal {

template <typename Scalar>
EIGEN_STRONG_INLINE Scalar __lgamma(Scalar x) {
  EIGEN_STATIC_ASSERT((internal::is_same<Scalar, Scalar>::value == false),
                      THIS_TYPE_IS_NOT_SUPPORTED);
}

template <> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float __lgamma<float>(float x) { return lgammaf(x); }
template <> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double __lgamma<double>(double x) { return lgamma(x); }

template <typename Scalar>
EIGEN_STRONG_INLINE Scalar __erf(Scalar x) {
  EIGEN_STATIC_ASSERT((internal::is_same<Scalar, Scalar>::value == false),
                      THIS_TYPE_IS_NOT_SUPPORTED);
}

template <> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float __erf<float>(float x) { return erff(x); }
template <> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double __erf<double>(double x) { return erf(x); }

template <typename Scalar>
EIGEN_STRONG_INLINE Scalar __erfc(Scalar x) {
  EIGEN_STATIC_ASSERT((internal::is_same<Scalar, Scalar>::value == false),
                      THIS_TYPE_IS_NOT_SUPPORTED);
}

template <> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float __erfc<float>(float x) { return erfcf(x); }
template <> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double __erfc<double>(double x) { return erfc(x); }

}  // end namespace internal

/****************************************************************************
* Implementations                                                           *
****************************************************************************/

namespace internal {

/****************************************************************************
* Implementation of lgamma                                                  *
****************************************************************************/

template<typename Scalar>
struct lgamma_impl
{
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(const Scalar& x)
  {
    return __lgamma<Scalar>(x);
  }
};

template<typename Scalar>
struct lgamma_retval
{
  typedef Scalar type;
};

/****************************************************************************
* Implementation of erf                                                  *
****************************************************************************/

template<typename Scalar>
struct erf_impl
{
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(const Scalar& x)
  {
    return __erf<Scalar>(x);
  }
};

template<typename Scalar>
struct erf_retval
{
  typedef Scalar type;
};

/****************************************************************************
* Implementation of erfc                                                  *
****************************************************************************/

template<typename Scalar>
struct erfc_impl
{
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(const Scalar& x)
  {
    return __erfc<Scalar>(x);
  }
};

template<typename Scalar>
struct erfc_retval
{
  typedef Scalar type;
};

}  // end namespace internal

namespace numext {

template<typename Scalar>
EIGEN_DEVICE_FUNC
inline EIGEN_MATHFUNC_RETVAL(lgamma, Scalar) lgamma(const Scalar& x)
{
  return EIGEN_MATHFUNC_IMPL(lgamma, Scalar)::run(x);
}

template<typename Scalar>
EIGEN_DEVICE_FUNC
inline EIGEN_MATHFUNC_RETVAL(erf, Scalar) erf(const Scalar& x)
{
  return EIGEN_MATHFUNC_IMPL(erf, Scalar)::run(x);
}

template<typename Scalar>
EIGEN_DEVICE_FUNC
inline EIGEN_MATHFUNC_RETVAL(erfc, Scalar) erfc(const Scalar& x)
{
  return EIGEN_MATHFUNC_IMPL(erfc, Scalar)::run(x);
}

}  // end namespace numext

}  // end namespace Eigen

#endif  // EIGEN_SPECIALFUNCTIONS_H
