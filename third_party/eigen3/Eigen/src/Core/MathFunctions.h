// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MATHFUNCTIONS_H
#define EIGEN_MATHFUNCTIONS_H

// source: http://www.geom.uiuc.edu/~huberty/math5337/groupe/digits.html
#define EIGEN_PI 3.141592653589793238462643383279502884197169399375105820974944592307816406

namespace Eigen {

// On WINCE, std::abs is defined for int only, so let's defined our own overloads:
// This issue has been confirmed with MSVC 2008 only, but the issue might exist for more recent versions too.
#if EIGEN_OS_WINCE && EIGEN_COMP_MSVC && EIGEN_COMP_MSVC<=1500
long        abs(long        x) { return (labs(x));  }
double      abs(double      x) { return (fabs(x));  }
float       abs(float       x) { return (fabsf(x)); }
long double abs(long double x) { return (fabsl(x)); }
#endif
  
namespace internal {

/** \internal \struct global_math_functions_filtering_base
  *
  * What it does:
  * Defines a typedef 'type' as follows:
  * - if type T has a member typedef Eigen_BaseClassForSpecializationOfGlobalMathFuncImpl, then
  *   global_math_functions_filtering_base<T>::type is a typedef for it.
  * - otherwise, global_math_functions_filtering_base<T>::type is a typedef for T.
  *
  * How it's used:
  * To allow to defined the global math functions (like sin...) in certain cases, like the Array expressions.
  * When you do sin(array1+array2), the object array1+array2 has a complicated expression type, all what you want to know
  * is that it inherits ArrayBase. So we implement a partial specialization of sin_impl for ArrayBase<Derived>.
  * So we must make sure to use sin_impl<ArrayBase<Derived> > and not sin_impl<Derived>, otherwise our partial specialization
  * won't be used. How does sin know that? That's exactly what global_math_functions_filtering_base tells it.
  *
  * How it's implemented:
  * SFINAE in the style of enable_if. Highly susceptible of breaking compilers. With GCC, it sure does work, but if you replace
  * the typename dummy by an integer template parameter, it doesn't work anymore!
  */

template<typename T, typename dummy = void>
struct global_math_functions_filtering_base
{
  typedef T type;
};

template<typename T> struct always_void { typedef void type; };

template<typename T>
struct global_math_functions_filtering_base
  <T,
   typename always_void<typename T::Eigen_BaseClassForSpecializationOfGlobalMathFuncImpl>::type
  >
{
  typedef typename T::Eigen_BaseClassForSpecializationOfGlobalMathFuncImpl type;
};

#define EIGEN_MATHFUNC_IMPL(func, scalar) Eigen::internal::func##_impl<typename Eigen::internal::global_math_functions_filtering_base<scalar>::type>
#define EIGEN_MATHFUNC_RETVAL(func, scalar) typename Eigen::internal::func##_retval<typename Eigen::internal::global_math_functions_filtering_base<scalar>::type>::type

/****************************************************************************
* Implementation of real                                                 *
****************************************************************************/

template<typename Scalar, bool IsComplex = NumTraits<Scalar>::IsComplex>
struct real_default_impl
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  EIGEN_DEVICE_FUNC
  static inline RealScalar run(const Scalar& x)
  {
    return x;
  }
};

template<typename Scalar>
struct real_default_impl<Scalar,true>
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  EIGEN_DEVICE_FUNC
  static inline RealScalar run(const Scalar& x)
  {
    using std::real;
    return real(x);
  }
};

template<typename Scalar> struct real_impl : real_default_impl<Scalar> {};

template<typename Scalar>
struct real_retval
{
  typedef typename NumTraits<Scalar>::Real type;
};

/****************************************************************************
* Implementation of imag                                                 *
****************************************************************************/

template<typename Scalar, bool IsComplex = NumTraits<Scalar>::IsComplex>
struct imag_default_impl
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  EIGEN_DEVICE_FUNC
  static inline RealScalar run(const Scalar&)
  {
    return RealScalar(0);
  }
};

template<typename Scalar>
struct imag_default_impl<Scalar,true>
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  EIGEN_DEVICE_FUNC
  static inline RealScalar run(const Scalar& x)
  {
    using std::imag;
    return imag(x);
  }
};

template<typename Scalar> struct imag_impl : imag_default_impl<Scalar> {};

template<typename Scalar>
struct imag_retval
{
  typedef typename NumTraits<Scalar>::Real type;
};

/****************************************************************************
* Implementation of real_ref                                             *
****************************************************************************/

template<typename Scalar>
struct real_ref_impl
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  EIGEN_DEVICE_FUNC
  static inline RealScalar& run(Scalar& x)
  {
    return reinterpret_cast<RealScalar*>(&x)[0];
  }
  EIGEN_DEVICE_FUNC
  static inline const RealScalar& run(const Scalar& x)
  {
    return reinterpret_cast<const RealScalar*>(&x)[0];
  }
};

template<typename Scalar>
struct real_ref_retval
{
  typedef typename NumTraits<Scalar>::Real & type;
};

/****************************************************************************
* Implementation of imag_ref                                             *
****************************************************************************/

template<typename Scalar, bool IsComplex>
struct imag_ref_default_impl
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  EIGEN_DEVICE_FUNC
  static inline RealScalar& run(Scalar& x)
  {
    return reinterpret_cast<RealScalar*>(&x)[1];
  }
  EIGEN_DEVICE_FUNC
  static inline const RealScalar& run(const Scalar& x)
  {
    return reinterpret_cast<RealScalar*>(&x)[1];
  }
};

template<typename Scalar>
struct imag_ref_default_impl<Scalar, false>
{
  EIGEN_DEVICE_FUNC
  static inline Scalar run(Scalar&)
  {
    return Scalar(0);
  }
  EIGEN_DEVICE_FUNC
  static inline const Scalar run(const Scalar&)
  {
    return Scalar(0);
  }
};

template<typename Scalar>
struct imag_ref_impl : imag_ref_default_impl<Scalar, NumTraits<Scalar>::IsComplex> {};

template<typename Scalar>
struct imag_ref_retval
{
  typedef typename NumTraits<Scalar>::Real & type;
};

/****************************************************************************
* Implementation of conj                                                 *
****************************************************************************/

template<typename Scalar, bool IsComplex = NumTraits<Scalar>::IsComplex>
struct conj_impl
{
  EIGEN_DEVICE_FUNC
  static inline Scalar run(const Scalar& x)
  {
    return x;
  }
};

template<typename Scalar>
struct conj_impl<Scalar,true>
{
  EIGEN_DEVICE_FUNC
  static inline Scalar run(const Scalar& x)
  {
    using std::conj;
    return conj(x);
  }
};

template<typename Scalar>
struct conj_retval
{
  typedef Scalar type;
};

/****************************************************************************
* Implementation of abs2                                                 *
****************************************************************************/

template<typename Scalar>
struct abs2_impl
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  EIGEN_DEVICE_FUNC
  static inline RealScalar run(const Scalar& x)
  {
    return x*x;
  }
};

template<typename RealScalar>
struct abs2_impl<std::complex<RealScalar> >
{
  EIGEN_DEVICE_FUNC
  static inline RealScalar run(const std::complex<RealScalar>& x)
  {
    return real(x)*real(x) + imag(x)*imag(x);
  }
};

template<typename Scalar>
struct abs2_retval
{
  typedef typename NumTraits<Scalar>::Real type;
};

/****************************************************************************
* Implementation of norm1                                                *
****************************************************************************/

template<typename Scalar, bool IsComplex>
struct norm1_default_impl
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  EIGEN_DEVICE_FUNC
  static inline RealScalar run(const Scalar& x)
  {
    using std::abs;
    return abs(real(x)) + abs(imag(x));
  }
};

template<typename Scalar>
struct norm1_default_impl<Scalar, false>
{
  EIGEN_DEVICE_FUNC
  static inline Scalar run(const Scalar& x)
  {
    using std::abs;
    return abs(x);
  }
};

template<typename Scalar>
struct norm1_impl : norm1_default_impl<Scalar, NumTraits<Scalar>::IsComplex> {};

template<typename Scalar>
struct norm1_retval
{
  typedef typename NumTraits<Scalar>::Real type;
};

/****************************************************************************
* Implementation of hypot                                                *
****************************************************************************/

template<typename Scalar>
struct hypot_impl
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  static inline RealScalar run(const Scalar& x, const Scalar& y)
  {
    using std::abs;
    using std::sqrt;
    RealScalar _x = abs(x);
    RealScalar _y = abs(y);
    Scalar p, qp;
    if(_x>_y)
    {
      p = _x;
      qp = _y / p;
    }
    else
    {
      p = _y;
      qp = _x / p;
    }
    if(p==RealScalar(0)) return RealScalar(0);
    return p * sqrt(RealScalar(1) + qp*qp);
  }
};

template<typename Scalar>
struct hypot_retval
{
  typedef typename NumTraits<Scalar>::Real type;
};

/****************************************************************************
* Implementation of cast                                                 *
****************************************************************************/

template<typename OldType, typename NewType>
struct cast_impl
{
  EIGEN_DEVICE_FUNC static inline NewType run(const OldType& x)
  {
    return static_cast<NewType>(x);
  }
};

// here, for once, we're plainly returning NewType: we don't want cast to do weird things.

template<typename OldType, typename NewType>
EIGEN_DEVICE_FUNC inline NewType cast(const OldType& x)
{
  return cast_impl<OldType, NewType>::run(x);
}

/****************************************************************************
* Implementation of atanh2                                                *
****************************************************************************/

template<typename Scalar>
struct atanh2_impl
{
  static inline Scalar run(const Scalar& x, const Scalar& r)
  {
    EIGEN_STATIC_ASSERT_NON_INTEGER(Scalar)
    using std::abs;
    using std::log;
    using std::sqrt;
    Scalar z = x / r;
    if (r == 0 || abs(z) > sqrt(NumTraits<Scalar>::epsilon()))
      return log((r + x) / (r - x)) / 2;
    else
      return z + z*z*z / 3;
  }
};

template<typename RealScalar>
struct atanh2_impl<std::complex<RealScalar> >
{
  typedef std::complex<RealScalar> Scalar;
  static inline Scalar run(const Scalar& x, const Scalar& r)
  {
    using std::log;
    using std::norm;
    using std::sqrt;
    Scalar z = x / r;
    if (r == Scalar(0) || norm(z) > NumTraits<RealScalar>::epsilon())
      return RealScalar(0.5) * log((r + x) / (r - x));
    else
      return z + z*z*z / RealScalar(3);
  }
};

template<typename Scalar>
struct atanh2_retval
{
  typedef Scalar type;
};

/****************************************************************************
* Implementation of round                                                   *
****************************************************************************/

#if EIGEN_HAS_CXX11_MATH
  template<typename Scalar>
  struct round_impl {
    static inline Scalar run(const Scalar& x)
    {
      EIGEN_STATIC_ASSERT((!NumTraits<Scalar>::IsComplex), NUMERIC_TYPE_MUST_BE_REAL)
      using std::round;
      return round(x);
    }
  };
#else
  template<typename Scalar>
  struct round_impl
  {
    static inline Scalar run(const Scalar& x)
    {
      EIGEN_STATIC_ASSERT((!NumTraits<Scalar>::IsComplex), NUMERIC_TYPE_MUST_BE_REAL)
      using std::floor;
      using std::ceil;
      return (x > 0.0) ? floor(x + 0.5) : ceil(x - 0.5);
    }
  };
#endif

template<typename Scalar>
struct round_retval
{
  typedef Scalar type;
};

/****************************************************************************
* Implementation of arg                                                     *
****************************************************************************/

#if EIGEN_HAS_CXX11_MATH
  template<typename Scalar>
  struct arg_impl {
    static inline Scalar run(const Scalar& x)
    {
      using std::arg;
      return arg(x);
    }
  };
#else
  template<typename Scalar, bool IsComplex = NumTraits<Scalar>::IsComplex>
  struct arg_default_impl
  {
    typedef typename NumTraits<Scalar>::Real RealScalar;
    EIGEN_DEVICE_FUNC
    static inline RealScalar run(const Scalar& x)
    {
      return (x < 0.0) ? EIGEN_PI : 0.0; }
  };

  template<typename Scalar>
  struct arg_default_impl<Scalar,true>
  {
    typedef typename NumTraits<Scalar>::Real RealScalar;
    EIGEN_DEVICE_FUNC
    static inline RealScalar run(const Scalar& x)
    {
      using std::arg;
      return arg(x);
    }
  };

  template<typename Scalar> struct arg_impl : arg_default_impl<Scalar> {};
#endif

template<typename Scalar>
struct arg_retval
{
  typedef typename NumTraits<Scalar>::Real type;
};

/****************************************************************************
* Implementation of log1p                                                   *
****************************************************************************/
template<typename Scalar, bool isComplex = NumTraits<Scalar>::IsComplex >
struct log1p_impl
{
  static inline Scalar run(const Scalar& x)
  {
    EIGEN_STATIC_ASSERT_NON_INTEGER(Scalar)
    typedef typename NumTraits<Scalar>::Real RealScalar;
    using std::log;
    Scalar x1p = RealScalar(1) + x;
    return ( x1p == Scalar(1) ) ? x : x * ( log(x1p) / (x1p - RealScalar(1)) );
  }
};

#if EIGEN_HAS_CXX11_MATH
template<typename Scalar>
struct log1p_impl<Scalar, false> {
  static inline Scalar run(const Scalar& x)
  {
    EIGEN_STATIC_ASSERT_NON_INTEGER(Scalar)
    using std::log1p;
    return log1p(x);
  }
};
#endif

template<typename Scalar>
struct log1p_retval
{
  typedef Scalar type;
};

/****************************************************************************
* Implementation of pow                                                  *
****************************************************************************/

template<typename Scalar, bool IsInteger>
struct pow_default_impl
{
  typedef Scalar retval;
  static inline Scalar run(const Scalar& x, const Scalar& y)
  {
    using std::pow;
    return pow(x, y);
  }
};

template<typename Scalar>
struct pow_default_impl<Scalar, true>
{
  static inline Scalar run(Scalar x, Scalar y)
  {
    Scalar res(1);
    eigen_assert(!NumTraits<Scalar>::IsSigned || y >= 0);
    if(y & 1) res *= x;
    y >>= 1;
    while(y)
    {
      x *= x;
      if(y&1) res *= x;
      y >>= 1;
    }
    return res;
  }
};

template<typename Scalar>
struct pow_impl : pow_default_impl<Scalar, NumTraits<Scalar>::IsInteger> {};

template<typename Scalar>
struct pow_retval
{
  typedef Scalar type;
};

/****************************************************************************
* Implementation of random                                               *
****************************************************************************/

template<typename Scalar,
         bool IsComplex,
         bool IsInteger>
struct random_default_impl {};

template<typename Scalar>
struct random_impl : random_default_impl<Scalar, NumTraits<Scalar>::IsComplex, NumTraits<Scalar>::IsInteger> {};

template<typename Scalar>
struct random_retval
{
  typedef Scalar type;
};

template<typename Scalar> inline EIGEN_MATHFUNC_RETVAL(random, Scalar) random(const Scalar& x, const Scalar& y);
template<typename Scalar> inline EIGEN_MATHFUNC_RETVAL(random, Scalar) random();

template<typename Scalar>
struct random_default_impl<Scalar, false, false>
{
  static inline Scalar run(const Scalar& x, const Scalar& y)
  {
    return x + (y-x) * Scalar(std::rand()) / Scalar(RAND_MAX);
  }
  static inline Scalar run()
  {
    return run(Scalar(NumTraits<Scalar>::IsSigned ? -1 : 0), Scalar(1));
  }
};

enum {
  meta_floor_log2_terminate,
  meta_floor_log2_move_up,
  meta_floor_log2_move_down,
  meta_floor_log2_bogus
};

template<unsigned int n, int lower, int upper> struct meta_floor_log2_selector
{
  enum { middle = (lower + upper) / 2,
         value = (upper <= lower + 1) ? int(meta_floor_log2_terminate)
               : (n < (1 << middle)) ? int(meta_floor_log2_move_down)
               : (n==0) ? int(meta_floor_log2_bogus)
               : int(meta_floor_log2_move_up)
  };
};

template<unsigned int n,
         int lower = 0,
         int upper = sizeof(unsigned int) * CHAR_BIT - 1,
         int selector = meta_floor_log2_selector<n, lower, upper>::value>
struct meta_floor_log2 {};

template<unsigned int n, int lower, int upper>
struct meta_floor_log2<n, lower, upper, meta_floor_log2_move_down>
{
  enum { value = meta_floor_log2<n, lower, meta_floor_log2_selector<n, lower, upper>::middle>::value };
};

template<unsigned int n, int lower, int upper>
struct meta_floor_log2<n, lower, upper, meta_floor_log2_move_up>
{
  enum { value = meta_floor_log2<n, meta_floor_log2_selector<n, lower, upper>::middle, upper>::value };
};

template<unsigned int n, int lower, int upper>
struct meta_floor_log2<n, lower, upper, meta_floor_log2_terminate>
{
  enum { value = (n >= ((unsigned int)(1) << (lower+1))) ? lower+1 : lower };
};

template<unsigned int n, int lower, int upper>
struct meta_floor_log2<n, lower, upper, meta_floor_log2_bogus>
{
  // no value, error at compile time
};

template<typename Scalar>
struct random_default_impl<Scalar, false, true>
{
  static inline Scalar run(const Scalar& x, const Scalar& y)
  {
    typedef typename conditional<NumTraits<Scalar>::IsSigned,std::ptrdiff_t,std::size_t>::type ScalarX;
    if(y<x)
      return x;
    std::size_t range = ScalarX(y)-ScalarX(x);
    std::size_t offset = 0;
    // rejection sampling
    std::size_t divisor    = (range+RAND_MAX-1)/(range+1);
    std::size_t multiplier = (range+RAND_MAX-1)/std::size_t(RAND_MAX);

    do {
      offset = ( (std::size_t(std::rand()) * multiplier) / divisor );
    } while (offset > range);

    return Scalar(ScalarX(x) + offset);
  }

  static inline Scalar run()
  {
#ifdef EIGEN_MAKING_DOCS
    return run(Scalar(NumTraits<Scalar>::IsSigned ? -10 : 0), Scalar(10));
#else
    enum { rand_bits = meta_floor_log2<(unsigned int)(RAND_MAX)+1>::value,
           scalar_bits = sizeof(Scalar) * CHAR_BIT,
           shift = EIGEN_PLAIN_ENUM_MAX(0, int(rand_bits) - int(scalar_bits)),
           offset = NumTraits<Scalar>::IsSigned ? (1 << (EIGEN_PLAIN_ENUM_MIN(rand_bits,scalar_bits)-1)) : 0
    };
    return Scalar((std::rand() >> shift) - offset);
#endif
  }
};

template<typename Scalar>
struct random_default_impl<Scalar, true, false>
{
  static inline Scalar run(const Scalar& x, const Scalar& y)
  {
    return Scalar(random(real(x), real(y)),
                  random(imag(x), imag(y)));
  }
  static inline Scalar run()
  {
    typedef typename NumTraits<Scalar>::Real RealScalar;
    return Scalar(random<RealScalar>(), random<RealScalar>());
  }
};

template<typename Scalar>
inline EIGEN_MATHFUNC_RETVAL(random, Scalar) random(const Scalar& x, const Scalar& y)
{
  return EIGEN_MATHFUNC_IMPL(random, Scalar)::run(x, y);
}

template<typename Scalar>
inline EIGEN_MATHFUNC_RETVAL(random, Scalar) random()
{
  return EIGEN_MATHFUNC_IMPL(random, Scalar)::run();
}

} // end namespace internal

/****************************************************************************
* Generic math functions                                                    *
****************************************************************************/

namespace numext {

#ifndef __CUDA_ARCH__
template<typename T>
EIGEN_DEVICE_FUNC
EIGEN_ALWAYS_INLINE T mini(const T& x, const T& y)
{
  EIGEN_USING_STD_MATH(min);
  return min EIGEN_NOT_A_MACRO (x,y);
}

template<typename T>
EIGEN_DEVICE_FUNC
EIGEN_ALWAYS_INLINE T maxi(const T& x, const T& y)
{
  EIGEN_USING_STD_MATH(max);
  return max EIGEN_NOT_A_MACRO (x,y);
}
#else
template<typename T>
EIGEN_DEVICE_FUNC
EIGEN_ALWAYS_INLINE T mini(const T& x, const T& y)
{
  return y < x ? y : x;
}
template<>
EIGEN_DEVICE_FUNC
EIGEN_ALWAYS_INLINE float mini(const float& x, const float& y)
{
  return fmin(x, y);
}
template<typename T>
EIGEN_DEVICE_FUNC
EIGEN_ALWAYS_INLINE T maxi(const T& x, const T& y)
{
  return x < y ? y : x;
}
template<>
EIGEN_DEVICE_FUNC
EIGEN_ALWAYS_INLINE float maxi(const float& x, const float& y)
{
  return fmax(x, y);
}
#endif

template<typename Scalar>
EIGEN_DEVICE_FUNC
inline EIGEN_MATHFUNC_RETVAL(real, Scalar) real(const Scalar& x)
{
  return EIGEN_MATHFUNC_IMPL(real, Scalar)::run(x);
}

template<typename Scalar>
EIGEN_DEVICE_FUNC
inline typename internal::add_const_on_value_type< EIGEN_MATHFUNC_RETVAL(real_ref, Scalar) >::type real_ref(const Scalar& x)
{
  return internal::real_ref_impl<Scalar>::run(x);
}

template<typename Scalar>
EIGEN_DEVICE_FUNC
inline EIGEN_MATHFUNC_RETVAL(real_ref, Scalar) real_ref(Scalar& x)
{
  return EIGEN_MATHFUNC_IMPL(real_ref, Scalar)::run(x);
}

template<typename Scalar>
EIGEN_DEVICE_FUNC
inline EIGEN_MATHFUNC_RETVAL(imag, Scalar) imag(const Scalar& x)
{
  return EIGEN_MATHFUNC_IMPL(imag, Scalar)::run(x);
}

template<typename Scalar>
EIGEN_DEVICE_FUNC
inline EIGEN_MATHFUNC_RETVAL(arg, Scalar) arg(const Scalar& x)
{
  return EIGEN_MATHFUNC_IMPL(arg, Scalar)::run(x);
}

template<typename Scalar>
EIGEN_DEVICE_FUNC
inline typename internal::add_const_on_value_type< EIGEN_MATHFUNC_RETVAL(imag_ref, Scalar) >::type imag_ref(const Scalar& x)
{
  return internal::imag_ref_impl<Scalar>::run(x);
}

template<typename Scalar>
EIGEN_DEVICE_FUNC
inline EIGEN_MATHFUNC_RETVAL(imag_ref, Scalar) imag_ref(Scalar& x)
{
  return EIGEN_MATHFUNC_IMPL(imag_ref, Scalar)::run(x);
}

template<typename Scalar>
EIGEN_DEVICE_FUNC
inline EIGEN_MATHFUNC_RETVAL(conj, Scalar) conj(const Scalar& x)
{
  return EIGEN_MATHFUNC_IMPL(conj, Scalar)::run(x);
}

template<typename Scalar>
EIGEN_DEVICE_FUNC
inline EIGEN_MATHFUNC_RETVAL(abs2, Scalar) abs2(const Scalar& x)
{
  return EIGEN_MATHFUNC_IMPL(abs2, Scalar)::run(x);
}

template<typename Scalar>
EIGEN_DEVICE_FUNC
inline EIGEN_MATHFUNC_RETVAL(norm1, Scalar) norm1(const Scalar& x)
{
  return EIGEN_MATHFUNC_IMPL(norm1, Scalar)::run(x);
}

template<typename Scalar>
EIGEN_DEVICE_FUNC
inline EIGEN_MATHFUNC_RETVAL(hypot, Scalar) hypot(const Scalar& x, const Scalar& y)
{
  return EIGEN_MATHFUNC_IMPL(hypot, Scalar)::run(x, y);
}

template<typename Scalar>
EIGEN_DEVICE_FUNC
inline EIGEN_MATHFUNC_RETVAL(log1p, Scalar) log1p(const Scalar& x)
{
  return EIGEN_MATHFUNC_IMPL(log1p, Scalar)::run(x);
}

template<typename Scalar>
EIGEN_DEVICE_FUNC
inline EIGEN_MATHFUNC_RETVAL(atanh2, Scalar) atanh2(const Scalar& x, const Scalar& y)
{
  return EIGEN_MATHFUNC_IMPL(atanh2, Scalar)::run(x, y);
}

template<typename Scalar>
EIGEN_DEVICE_FUNC
inline EIGEN_MATHFUNC_RETVAL(pow, Scalar) pow(const Scalar& x, const Scalar& y)
{
  return EIGEN_MATHFUNC_IMPL(pow, Scalar)::run(x, y);
}

template<typename T>
EIGEN_DEVICE_FUNC
bool (isfinite)(const T& x)
{
  #if EIGEN_HAS_CXX11_MATH
    using std::isfinite;
    return isfinite(x);
  #else
    return x<NumTraits<T>::highest() && x>NumTraits<T>::lowest();
  #endif
}

template<typename T>
EIGEN_DEVICE_FUNC
bool (isfinite)(const std::complex<T>& x)
{
  return numext::isfinite(numext::real(x)) && numext::isfinite(numext::imag(x));
}

template<typename T>
EIGEN_DEVICE_FUNC
bool (isnan)(const T& x)
{
  #if EIGEN_HAS_CXX11_MATH
    using std::isnan;
    return isnan(x);
  #else
    return x != x;
  #endif
}

template<typename T>
EIGEN_DEVICE_FUNC
bool (isnan)(const std::complex<T>& x)
{
  return numext::isnan(numext::real(x)) || numext::isnan(numext::imag(x));
}

template<typename T>
EIGEN_DEVICE_FUNC
bool (isinf)(const T& x)
{
  #if EIGEN_HAS_CXX11_MATH
    using std::isinf;
    return isinf(x);
  #else
    return x>NumTraits<T>::highest() || x<NumTraits<T>::lowest();
  #endif
}

template<typename T>
EIGEN_DEVICE_FUNC
bool (isinf)(const std::complex<T>& x)
{
  return (numext::isinf(numext::real(x)) || numext::isinf(numext::imag(x))) && (!numext::isnan(x));
}

template<typename Scalar>
EIGEN_DEVICE_FUNC
inline EIGEN_MATHFUNC_RETVAL(round, Scalar) round(const Scalar& x)
{
  return EIGEN_MATHFUNC_IMPL(round, Scalar)::run(x);
}

template<typename T>
EIGEN_DEVICE_FUNC
T (floor)(const T& x)
{
  using std::floor;
  return floor(x);
}

template<typename T>
EIGEN_DEVICE_FUNC
T (ceil)(const T& x)
{
  using std::ceil;
  return ceil(x);
}

// Log base 2 for 32 bits positive integers.
// Conveniently returns 0 for x==0.
inline int log2(int x)
{
  eigen_assert(x>=0);
  unsigned int v(x);
  static const int table[32] = { 0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30, 8, 12, 20, 28, 15, 17, 24, 7, 19, 27, 23, 6, 26, 5, 4, 31 };
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return table[(v * 0x07C4ACDDU) >> 27];
}

} // end namespace numext

namespace internal {

/****************************************************************************
* Implementation of fuzzy comparisons                                       *
****************************************************************************/

template<typename Scalar,
         bool IsComplex,
         bool IsInteger>
struct scalar_fuzzy_default_impl {};

template<typename Scalar>
struct scalar_fuzzy_default_impl<Scalar, false, false>
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  template<typename OtherScalar> EIGEN_DEVICE_FUNC
  static inline bool isMuchSmallerThan(const Scalar& x, const OtherScalar& y, const RealScalar& prec)
  {
    using std::abs;
    return abs(x) <= abs(y) * prec;
  }
  EIGEN_DEVICE_FUNC
  static inline bool isApprox(const Scalar& x, const Scalar& y, const RealScalar& prec)
  {
    using std::abs;
    return abs(x - y) <= numext::mini(abs(x), abs(y)) * prec;
  }
  EIGEN_DEVICE_FUNC
  static inline bool isApproxOrLessThan(const Scalar& x, const Scalar& y, const RealScalar& prec)
  {
    return x <= y || isApprox(x, y, prec);
  }
};

template<typename Scalar>
struct scalar_fuzzy_default_impl<Scalar, false, true>
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  template<typename OtherScalar> EIGEN_DEVICE_FUNC
  static inline bool isMuchSmallerThan(const Scalar& x, const Scalar&, const RealScalar&)
  {
    return x == Scalar(0);
  }
  EIGEN_DEVICE_FUNC
  static inline bool isApprox(const Scalar& x, const Scalar& y, const RealScalar&)
  {
    return x == y;
  }
  EIGEN_DEVICE_FUNC
  static inline bool isApproxOrLessThan(const Scalar& x, const Scalar& y, const RealScalar&)
  {
    return x <= y;
  }
};

template<typename Scalar>
struct scalar_fuzzy_default_impl<Scalar, true, false>
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  template<typename OtherScalar>
  static inline bool isMuchSmallerThan(const Scalar& x, const OtherScalar& y, const RealScalar& prec)
  {
    return numext::abs2(x) <= numext::abs2(y) * prec * prec;
  }
  static inline bool isApprox(const Scalar& x, const Scalar& y, const RealScalar& prec)
  {
    return numext::abs2(x - y) <= numext::mini(numext::abs2(x), numext::abs2(y)) * prec * prec;
  }
};

template<typename Scalar>
struct scalar_fuzzy_impl : scalar_fuzzy_default_impl<Scalar, NumTraits<Scalar>::IsComplex, NumTraits<Scalar>::IsInteger> {};

template<typename Scalar, typename OtherScalar> EIGEN_DEVICE_FUNC
inline bool isMuchSmallerThan(const Scalar& x, const OtherScalar& y,
                                   typename NumTraits<Scalar>::Real precision = NumTraits<Scalar>::dummy_precision())
{
  return scalar_fuzzy_impl<Scalar>::template isMuchSmallerThan<OtherScalar>(x, y, precision);
}

template<typename Scalar> EIGEN_DEVICE_FUNC
inline bool isApprox(const Scalar& x, const Scalar& y,
                          typename NumTraits<Scalar>::Real precision = NumTraits<Scalar>::dummy_precision())
{
  return scalar_fuzzy_impl<Scalar>::isApprox(x, y, precision);
}

template<typename Scalar> EIGEN_DEVICE_FUNC
inline bool isApproxOrLessThan(const Scalar& x, const Scalar& y,
                                    typename NumTraits<Scalar>::Real precision = NumTraits<Scalar>::dummy_precision())
{
  return scalar_fuzzy_impl<Scalar>::isApproxOrLessThan(x, y, precision);
}

/******************************************
***  The special case of the  bool type ***
******************************************/

template<> struct random_impl<bool>
{
  static inline bool run()
  {
    return random<int>(0,1)==0 ? false : true;
  }
};

template<> struct scalar_fuzzy_impl<bool>
{
  typedef bool RealScalar;
  
  template<typename OtherScalar> EIGEN_DEVICE_FUNC
  static inline bool isMuchSmallerThan(const bool& x, const bool&, const bool&)
  {
    return !x;
  }
  
  EIGEN_DEVICE_FUNC
  static inline bool isApprox(bool x, bool y, bool)
  {
    return x == y;
  }

  EIGEN_DEVICE_FUNC
  static inline bool isApproxOrLessThan(const bool& x, const bool& y, const bool&)
  {
    return (!x) || y;
  }
  
};

  
} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_MATHFUNCTIONS_H
