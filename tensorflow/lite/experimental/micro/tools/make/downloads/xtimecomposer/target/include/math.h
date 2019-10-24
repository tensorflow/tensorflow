#ifndef  _MATH_H_

#define  _MATH_H_

#include <sys/reent.h>
#include <machine/ieeefp.h>
#include "_ansi.h"

_BEGIN_STD_C

union __dmath
{
  __ULong i[2];
  double d;
};

union __fmath
{
  __ULong i[1];
  float f;
};

#if defined(_HAVE_LONG_DOUBLE)
#error "xcore does not have a 128bit long double"
union __ldmath
{
  __ULong i[4];
  _LONG_DOUBLE ld;
};
#endif

/* Natural log of 2 */
#define _M_LOG2_E        0.693147180559945309417

#if defined(__GNUC__) && \
  ( (__GNUC__ >= 4) || \
    ( (__GNUC__ >= 3) && defined(__GNUC_MINOR__) && (__GNUC_MINOR__ >= 3) ) )

 /* gcc >= 3.3 implicitly defines builtins for HUGE_VALx values.  */

# ifndef HUGE_VAL
#  define HUGE_VAL (__builtin_huge_val())
# endif

# ifndef HUGE_VALF
#  define HUGE_VALF (__builtin_huge_valf())
# endif

# ifndef HUGE_VALL
#  if defined(_HAVE_LONG_DOUBLE)
#   define HUGE_VALL (__builtin_huge_vall())
#  else
#   define HUGE_VALL HUGE_VAL
#  endif
# endif

# ifndef INFINITY
#  define INFINITY (__builtin_inff())
# endif

# ifndef NAN
#  define NAN (__builtin_nanf(""))
# endif

#else /* !gcc >= 3.3  */

 /* No builtins.  Use floating-point unions instead.  Declare as an array
    without bounds so no matter what small data support a port and/or
    library has, the reference will be via the general method for accessing
    globals. */

 #ifndef HUGE_VAL
  extern __IMPORT const union __dmath __infinity[];
  #define HUGE_VAL (__infinity[0].d)
 #endif

 #ifndef HUGE_VALF
  extern __IMPORT const union __fmath __infinityf[];
  #define HUGE_VALF (__infinityf[0].f)
 #endif

 #ifndef HUGE_VALL
 # if defined(_HAVE_LONG_DOUBLE)
    extern __IMPORT const union __ldmath __infinityld[];
    #define HUGE_VALL (__infinityld[0].ld)
 # else
    #define HUGE_VALL HUGE_VAL
 # endif
 #endif

#endif /* !gcc >= 3.3  */

/* Reentrant ANSI C functions.  */

#ifndef __math_68881
extern double atan _PARAMS((double));
extern double cos _PARAMS((double));
extern double sin _PARAMS((double));
extern double tan _PARAMS((double));
extern double tanh _PARAMS((double));
extern double frexp _PARAMS((double, int *));
extern double modf _PARAMS((double, double *));
extern double ceil _PARAMS((double));
extern double fabs _PARAMS((double));
extern double floor _PARAMS((double));
#endif /* ! defined (__math_68881) */

/* Non reentrant ANSI C functions.  */

#ifndef __math_6881
extern double acos _PARAMS((double));
extern double asin _PARAMS((double));
extern double atan2 _PARAMS((double, double));
extern double cosh _PARAMS((double));
extern double sinh _PARAMS((double));
extern double exp _PARAMS((double));
extern double ldexp _PARAMS((double, int));
extern double log _PARAMS((double));
extern double log10 _PARAMS((double));
extern double pow _PARAMS((double, double));
extern double sqrt _PARAMS((double));
extern double fmod _PARAMS((double, double));
#endif /* ! defined (__math_68881) */

#if !defined(__STRICT_ANSI__) || defined(__cplusplus) || __STDC_VERSION__ >= 199901L

/* ISO C99 types and macros. */

#ifndef FLT_EVAL_METHOD
#define FLT_EVAL_METHOD 0
typedef float float_t;
typedef double double_t;
#endif /* FLT_EVAL_METHOD */

#define FP_NAN         0
#define FP_INFINITE    1
#define FP_ZERO        2
#define FP_SUBNORMAL   3
#define FP_NORMAL      4

#ifndef FP_ILOGB0
# define FP_ILOGB0 (-INT_MAX)
#endif
#ifndef FP_ILOGBNAN
# define FP_ILOGBNAN INT_MAX
#endif

#ifndef MATH_ERRNO
# define MATH_ERRNO 1
#endif
#ifndef MATH_ERREXCEPT
# define MATH_ERREXCEPT 2
#endif
#ifndef math_errhandling
# define math_errhandling MATH_ERRNO
#endif

extern int __isinff (float x);
extern int __isinfd (double x);
extern int __isnanf (float x);
extern int __isnand (double x);
extern int __fpclassifyf (float x);
extern int __fpclassifyd (double x);
extern int __signbitf (float x);
extern int __signbitd (double x);

#if !defined(__XC__)
#define fpclassify(x) \
          (__extension__ ({__typeof__(x) __x = (x); \
                           (sizeof (__x) == sizeof (float))  ? __fpclassifyf(__x) : __fpclassifyd(__x);}))

#ifndef isfinite
#define isfinite(y) \
          (__extension__ ({__typeof__(y) __y = (y); \
                           fpclassify(__y) != FP_INFINITE && fpclassify(__y) != FP_NAN;}))
#endif

/* Note: isinf and isnan were once functions in newlib that took double
 *       arguments.  C99 specifies that these names are reserved for macros
 *       supporting multiple floating point types.  Thus, they are
 *       now defined as macros.  Implementations of the old functions
 *       taking double arguments still exist for compatibility purposes.  */
#ifndef isinf
#define isinf(x) \
          (__extension__ ({__typeof__(x) __x = (x); \
                           (sizeof (__x) == sizeof (float))  ? __isinff(__x) : __isinfd(__x);}))
#endif

#ifndef isnan
#define isnan(x) \
          (__extension__ ({__typeof__(x) __x = (x); \
                           (sizeof (__x) == sizeof (float))  ? __isnanf(__x) : __isnand(__x);}))
#endif

#define isnormal(y) (fpclassify(y) == FP_NORMAL)
#define signbit(x) \
          (__extension__ ({__typeof__(x) __x = (x); \
                           (sizeof(__x) == sizeof(float)) ? __signbitf(__x) : __signbitd(__x);}))

#define isgreater(x,y) \
          (__extension__ ({__typeof__(x) __x = (x); __typeof__(y) __y = (y); \
                           !isunordered(__x,__y) && (__x > __y);}))
#define isgreaterequal(x,y) \
          (__extension__ ({__typeof__(x) __x = (x); __typeof__(y) __y = (y); \
                           !isunordered(__x,__y) && (__x >= __y);}))
#define isless(x,y) \
          (__extension__ ({__typeof__(x) __x = (x); __typeof__(y) __y = (y); \
                           !isunordered(__x,__y) && (__x < __y);}))
#define islessequal(x,y) \
          (__extension__ ({__typeof__(x) __x = (x); __typeof__(y) __y = (y); \
                           !isunordered(__x,__y) && (__x <= __y);}))
#define islessgreater(x,y) \
          (__extension__ ({__typeof__(x) __x = (x); __typeof__(y) __y = (y); \
                           !isunordered(__x,__y) && (__x < __y || __x > __y);}))

#define isunordered(a,b) \
          (__extension__ ({__typeof__(a) __a = (a); __typeof__(b) __b = (b); \
                           fpclassify(__a) == FP_NAN || fpclassify(__b) == FP_NAN;}))

#endif /*  !defined(__XC__) */

/* Non ANSI double precision functions.  */

extern double nan _PARAMS((const char *));
extern int finite _PARAMS((double));
extern double copysign _PARAMS((double, double));
extern double logb _PARAMS((double));
extern int ilogb _PARAMS((double));

extern double asinh _PARAMS((double));
extern double cbrt _PARAMS((double));
extern double nextafter _PARAMS((double, double));
extern double rint _PARAMS((double));
extern double scalbn _PARAMS((double, int));

extern double exp2 _PARAMS((double));
extern double scalbln _PARAMS((double, long int));
extern double tgamma _PARAMS((double));
extern double nearbyint _PARAMS((double));
extern long int lrint _PARAMS((double));
extern long long int llrint _PARAMS((double));
extern double round _PARAMS((double));
extern long int lround _PARAMS((double));
extern long long int llround _PARAMS((double));
extern double trunc _PARAMS((double));
extern double remquo _PARAMS((double, double, int *));
extern double fdim _PARAMS((double, double));
extern double fmax _PARAMS((double, double));
extern double fmin _PARAMS((double, double));
extern double fma _PARAMS((double, double, double));

#ifndef __math_68881
extern double log1p _PARAMS((double));
extern double expm1 _PARAMS((double));
#endif /* ! defined (__math_68881) */

extern double acosh _PARAMS((double));
extern double atanh _PARAMS((double));
extern double remainder _PARAMS((double, double));
extern double gamma _PARAMS((double));
extern double lgamma _PARAMS((double));
extern double erf _PARAMS((double));
extern double erfc _PARAMS((double));

#ifndef __math_68881
extern double hypot _PARAMS((double, double));
#endif

/* Single precision versions of ANSI functions.  */

extern float atanf _PARAMS((float));
extern float cosf _PARAMS((float));
extern float sinf _PARAMS((float));
extern float tanf _PARAMS((float));
extern float tanhf _PARAMS((float));
extern float frexpf _PARAMS((float, int *));
extern float modff _PARAMS((float, float *));
extern float ceilf _PARAMS((float));
extern float fabsf _PARAMS((float));
extern float floorf _PARAMS((float));

extern float acosf _PARAMS((float));
extern float asinf _PARAMS((float));
extern float atan2f _PARAMS((float, float));
extern float coshf _PARAMS((float));
extern float sinhf _PARAMS((float));
extern float expf _PARAMS((float));
extern float ldexpf _PARAMS((float, int));
extern float logf _PARAMS((float));
extern float log10f _PARAMS((float));
extern float powf _PARAMS((float, float));
extern float sqrtf _PARAMS((float));
extern float fmodf _PARAMS((float, float));

/* Other single precision functions.  */

extern float exp2f _PARAMS((float));
extern float scalblnf _PARAMS((float, long int));
extern float tgammaf _PARAMS((float));
extern float nearbyintf _PARAMS((float));
extern long int lrintf _PARAMS((float));
extern long long llrintf _PARAMS((float));
extern float roundf _PARAMS((float));
extern long int lroundf _PARAMS((float));
extern long long int llroundf _PARAMS((float));
extern float truncf _PARAMS((float));
extern float remquof _PARAMS((float, float, int *));
extern float fdimf _PARAMS((float, float));
extern float fmaxf _PARAMS((float, float));
extern float fminf _PARAMS((float, float));
extern float fmaf _PARAMS((float, float, float));

extern float nanf _PARAMS((const char *));
extern int isnanf _PARAMS((float));
extern int isinff _PARAMS((float));
extern int finitef _PARAMS((float));
extern float copysignf _PARAMS((float, float));
extern float logbf _PARAMS((float));
extern int ilogbf _PARAMS((float));

extern float asinhf _PARAMS((float));
extern float cbrtf _PARAMS((float));
extern float nextafterf _PARAMS((float, float));
extern float rintf _PARAMS((float));
extern float scalbnf _PARAMS((float, int));
extern float log1pf _PARAMS((float));
extern float expm1f _PARAMS((float));

extern float acoshf _PARAMS((float));
extern float atanhf _PARAMS((float));
extern float remainderf _PARAMS((float, float));
extern float gammaf _PARAMS((float));
extern float lgammaf _PARAMS((float));
extern float erff _PARAMS((float));
extern float erfcf _PARAMS((float));
extern float hypotf _PARAMS((float, float));


#ifdef _ELIDABLE_INLINE
# error "_ELIDABLE_INLINE already defined"
#endif
#if __GNUC__ && !__GNUC_STDC_INLINE__
# define _ELIDABLE_INLINE extern inline
#else
# define _ELIDABLE_INLINE inline
#endif
/* Reentrant ANSI C functions.  */
_ELIDABLE_INLINE long double atanl (long double x)
    {return atan(x);};
_ELIDABLE_INLINE long double cosl (long double x)
    {return cos(x);};
_ELIDABLE_INLINE long double sinl (long double x)
    {return sin(x);};
_ELIDABLE_INLINE long double tanl (long double x)
    {return tan(x);};
_ELIDABLE_INLINE long double tanhl (long double x)
    {return tanh(x);};
_ELIDABLE_INLINE long double frexpl (long double x, int *y)
    {return frexp(x,y);};
_ELIDABLE_INLINE long double modfl (long double x, long double *y)
    {return modf(x,(double*)y);};
_ELIDABLE_INLINE long double ceill (long double x)
    {return ceil(x);};
_ELIDABLE_INLINE long double fabsl (long double x)
    {return fabs(x);};
_ELIDABLE_INLINE long double floorl (long double x)
    {return floor(x);};
_ELIDABLE_INLINE long double log1pl (long double x)
    {return log1p(x);};
_ELIDABLE_INLINE long double expm1l (long double x)
    {return expm1(x);};
/* Non reentrant ANSI C functions.  */
_ELIDABLE_INLINE long double acosl (long double x)
    {return acos(x);};
_ELIDABLE_INLINE long double asinl (long double x)
    {return asin(x);};
_ELIDABLE_INLINE long double atan2l (long double x, long double y)
    {return atan2(x,y);};
_ELIDABLE_INLINE long double coshl (long double x)
    {return cosh(x);};
_ELIDABLE_INLINE long double sinhl (long double x)
    {return sinh(x);};
_ELIDABLE_INLINE long double expl (long double x)
    {return exp(x);};
_ELIDABLE_INLINE long double ldexpl (long double x, int y)
    {return ldexp(x,y);};
_ELIDABLE_INLINE long double logl (long double x)
    {return log(x);};
_ELIDABLE_INLINE long double log10l (long double x)
    {return log10(x);};
_ELIDABLE_INLINE double log2 (double x)
    {return log (x) / _M_LOG2_E;};
_ELIDABLE_INLINE float log2f (float x)
    {return logf (x) / (float) _M_LOG2_E;};
_ELIDABLE_INLINE long double log2l (long double x)
    {return log (x) / _M_LOG2_E;};
_ELIDABLE_INLINE long double powl (long double x, long double y)
    {return pow(x,y);};
_ELIDABLE_INLINE long double sqrtl (long double x)
    {return sqrt(x);};
_ELIDABLE_INLINE long double fmodl (long double x, long double y)
    {return fmod(x,y);};
_ELIDABLE_INLINE long double hypotl (long double x, long double y)
    {return hypot(x,y);};
_ELIDABLE_INLINE long double copysignl (long double x, long double y)
    {return copysign(x,y);};
_ELIDABLE_INLINE long double nanl (const char *x)
    {return nan(x);};
_ELIDABLE_INLINE int ilogbl (long double x)
    {return ilogb(x);};
_ELIDABLE_INLINE long double logbl (long double x)
    {return logb(x);};
_ELIDABLE_INLINE long double asinhl (long double x)
    {return asinh(x);};
_ELIDABLE_INLINE long double cbrtl (long double x)
    {return cbrt(x);};
_ELIDABLE_INLINE long double nextafterl (long double x, long double y)
    {return nextafter(x,y);};
_ELIDABLE_INLINE double nexttoward (double x, long double y)
    {return nextafter(x,(double)y);};
_ELIDABLE_INLINE float nexttowardf (float x, long double y)
    {return nextafterf(x,(float)y);};
_ELIDABLE_INLINE long double nexttowardl (long double x, long double y)
    {return nextafterl(x,y);};
_ELIDABLE_INLINE long double rintl (long double x)
    {return rint(x);};
_ELIDABLE_INLINE long double scalbnl (long double x, int y)
    {return scalbn(x,y);};
_ELIDABLE_INLINE long double exp2l (long double x)
    {return exp2(x);};
_ELIDABLE_INLINE long double scalblnl (long double x, long int y)
    {return scalbln(x,y);};
_ELIDABLE_INLINE long double tgammal (long double x)
    {return tgamma(x);};
_ELIDABLE_INLINE long double nearbyintl (long double x)
    {return nearbyint(x);};
_ELIDABLE_INLINE long int lrintl (long double x)
    {return lrint(x);};
_ELIDABLE_INLINE long long int llrintl (long double x)
    {return llrint(x);};
_ELIDABLE_INLINE long double roundl (long double x)
    {return round(x);};
_ELIDABLE_INLINE long int lroundl (long double x)
    {return lround(x);};
_ELIDABLE_INLINE long long int llroundl (long double x)
    {return llround(x);};
_ELIDABLE_INLINE long double truncl (long double x)
    {return truncl(x);};
_ELIDABLE_INLINE long double remquol (long double x, long double y, int *z)
    {return remquo(x,y,z);};
_ELIDABLE_INLINE long double fdiml (long double x, long double y)
    {return fdim(x,y);};
_ELIDABLE_INLINE long double fmaxl (long double x, long double y)
    {return fmax(x,y);};
_ELIDABLE_INLINE long double fminl (long double x, long double y)
    {return fmin(x,y);};
_ELIDABLE_INLINE long double fmal (long double x, long double y, long double z)
    {return fma(x,y,z);};
_ELIDABLE_INLINE long double acoshl (long double x)
    {return acosh(x);};
_ELIDABLE_INLINE long double atanhl (long double x)
    {return atanh(x);};
_ELIDABLE_INLINE long double remainderl (long double x, long double y)
    {return remainder(x,y);};
_ELIDABLE_INLINE long double lgammal (long double x)
    {return lgamma(x);};
_ELIDABLE_INLINE long double erfl (long double x)
    {return erf(x);};
_ELIDABLE_INLINE long double erfcl (long double x)
    {return erfc(x);};
#undef _ELIDABLE_INLINE

#endif /* !defined (__STRICT_ANSI__) || defined(__cplusplus) || __STDC_VERSION__ >= 199901L */

#if !defined (__STRICT_ANSI__) || defined(__cplusplus)

extern double cabs();
extern double drem _PARAMS((double, double));
extern void sincos _PARAMS((double, double *, double *));
extern double gamma_r _PARAMS((double, int *));
extern double lgamma_r _PARAMS((double, int *));

extern double y0 _PARAMS((double));
extern double y1 _PARAMS((double));
extern double yn _PARAMS((int, double));
extern double j0 _PARAMS((double));
extern double j1 _PARAMS((double));
extern double jn _PARAMS((int, double));

extern float cabsf();
extern float dremf _PARAMS((float, float));
extern void sincosf _PARAMS((float, float *, float *));
extern float gammaf_r _PARAMS((float, int *));
extern float lgammaf_r _PARAMS((float, int *));

extern float y0f _PARAMS((float));
extern float y1f _PARAMS((float));
extern float ynf _PARAMS((int, float));
extern float j0f _PARAMS((float));
extern float j1f _PARAMS((float));
extern float jnf _PARAMS((int, float));

/* GNU extensions */
# ifndef exp10
extern double exp10 _PARAMS((double));
# endif
# ifndef pow10
extern double pow10 _PARAMS((double));
# endif
# ifndef exp10f
extern float exp10f _PARAMS((float));
# endif
# ifndef pow10f
extern float pow10f _PARAMS((float));
# endif

#endif /* !defined (__STRICT_ANSI__) || defined(__cplusplus) */

#ifndef __STRICT_ANSI__

#ifndef __XC__
/* The gamma functions use a global variable, signgam.  */
#define signgam (*__signgam())
extern int *__signgam _PARAMS((void));
#endif

/* The exception structure passed to the matherr routine.  */
/* We have a problem when using C++ since `exception' is a reserved
   name in C++.  */
#ifdef __cplusplus
struct __exception
#else
struct exception
#endif
{
  int type;
  char *name;
  double arg1;
  double arg2;
  double retval;
  int err;
};

#ifdef __cplusplus
extern int matherr _PARAMS((struct __exception *e));
#else
extern int matherr _PARAMS((struct exception *e));
#endif

/* Values for the type field of struct exception.  */

#define DOMAIN 1
#define SING 2
#define OVERFLOW 3
#define UNDERFLOW 4
#define TLOSS 5
#define PLOSS 6

/* Useful constants.  */

#define MAXFLOAT	3.40282347e+38F

#define M_E		2.7182818284590452354
#define M_LOG2E		1.4426950408889634074
#define M_LOG10E	0.43429448190325182765
#define M_LN2		0.69314718055994530942
#define M_LN10		2.30258509299404568402
#define M_PI		3.14159265358979323846
#define M_TWOPI         (M_PI * 2.0)
#define M_PI_2		1.57079632679489661923
#define M_PI_4		0.78539816339744830962
#define M_3PI_4		2.3561944901923448370E0
#define M_SQRTPI        1.77245385090551602792981
#define M_1_PI		0.31830988618379067154
#define M_2_PI		0.63661977236758134308
#define M_2_SQRTPI	1.12837916709551257390
#define M_SQRT2		1.41421356237309504880
#define M_SQRT1_2	0.70710678118654752440
#define M_LN2LO         1.9082149292705877000E-10
#define M_LN2HI         6.9314718036912381649E-1
#define M_SQRT3	1.73205080756887719000
#define M_IVLN10        0.43429448190325182765 /* 1 / log(10) */
#define M_LOG2_E        _M_LOG2_E
#define M_INVLN2        1.4426950408889633870E0  /* 1 / log(2) */


/* Global control over fdlibm error handling.  */

enum __fdlibm_version
{
  __fdlibm_ieee = -1,
  __fdlibm_svid,
  __fdlibm_xopen,
  __fdlibm_posix
};

#define _LIB_VERSION_TYPE enum __fdlibm_version
#define _LIB_VERSION __fdlib_version

extern __IMPORT _LIB_VERSION_TYPE _LIB_VERSION;

#define _IEEE_  __fdlibm_ieee
#define _SVID_  __fdlibm_svid
#define _XOPEN_ __fdlibm_xopen
#define _POSIX_ __fdlibm_posix

#endif /* ! defined (__STRICT_ANSI__) */

_END_STD_C

#ifdef __FAST_MATH__
#include <machine/fastmath.h>
#endif

#endif /* _MATH_H_ */
