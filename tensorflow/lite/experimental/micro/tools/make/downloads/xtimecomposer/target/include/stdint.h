/*
 * Copyright (c) 2004, 2005 by
 * Ralf Corsepius, Ulm/Germany. All rights reserved.
 *
 * Permission to use, copy, modify, and distribute this software
 * is freely granted, provided that this notice is preserved.
 */

/*
 * @todo - Add support for wint_t types.
 */

#ifndef _STDINT_H
#define _STDINT_H

#if defined(__cplusplus) || defined(__XC__)
extern "C" {
#endif

#if defined(__GNUC__) && \
  ( (__GNUC__ >= 4) || \
    ( (__GNUC__ >= 3) && defined(__GNUC_MINOR__) && (__GNUC_MINOR__ > 2) ) )
/* gcc > 3.2 implicitly defines the values we are interested */
#define __STDINT_EXP(x) __##x##__
#else
#define __STDINT_EXP(x) x
#include <limits.h>
#endif

/* Check if "long long" is 64bit wide */
/* Modern GCCs provide __LONG_LONG_MAX__, SUSv3 wants LLONG_MAX */
#if ( defined(__LONG_LONG_MAX__) && (__LONG_LONG_MAX__ > 0x7fffffff) ) \
  || ( defined(LLONG_MAX) && (LLONG_MAX > 0x7fffffff) )
#define __have_longlong64 1
#endif

/* Check if "long" is 64bit or 32bit wide */
#if __STDINT_EXP(LONG_MAX) > 0x7fffffff
#define __have_long64 1
#elif __STDINT_EXP(LONG_MAX) == 0x7fffffff && !defined(__SPU__)
#define __have_long32 1
#endif

#if __STDINT_EXP(SCHAR_MAX) == 0x7f
typedef signed char int8_t ;
typedef unsigned char uint8_t ;
#define __int8_t_defined 1
#endif

#if __int8_t_defined
typedef signed char int_least8_t;
typedef unsigned char uint_least8_t;
#define __int_least8_t_defined 1
#endif

#if __STDINT_EXP(SHRT_MAX) == 0x7fff
typedef signed short int16_t;
typedef unsigned short uint16_t;
#define __int16_t_defined 1
#elif __STDINT_EXP(INT_MAX) == 0x7fff
typedef signed int int16_t;
typedef unsigned int uint16_t;
#define __int16_t_defined 1
#elif __STDINT_EXP(SCHAR_MAX) == 0x7fff
typedef signed char int16_t;
typedef unsigned char uint16_t;
#define __int16_t_defined 1
#endif

#if __int16_t_defined
typedef int16_t   	int_least16_t;
typedef uint16_t 	uint_least16_t;
#define __int_least16_t_defined 1

#if !__int_least8_t_defined
typedef int16_t	   	int_least8_t;
typedef uint16_t  	uint_least8_t;
#define __int_least8_t_defined 1
#endif
#endif

#if __have_long32
typedef signed long int32_t;
typedef unsigned long uint32_t;
#define __int32_t_defined 1
#elif __STDINT_EXP(INT_MAX) == 0x7fffffffL
typedef signed int int32_t;
typedef unsigned int uint32_t;
#define __int32_t_defined 1
#elif __STDINT_EXP(SHRT_MAX) == 0x7fffffffL
typedef signed short int32_t;
typedef unsigned short uint32_t;
#define __int32_t_defined 1
#elif __STDINT_EXP(SCHAR_MAX) == 0x7fffffffL
typedef signed char int32_t;
typedef unsigned char uint32_t;
#define __int32_t_defined 1
#endif

#if __int32_t_defined
typedef int32_t   	int_least32_t;
typedef uint32_t 	uint_least32_t;
#define __int_least32_t_defined 1

#if !__int_least8_t_defined
typedef int32_t	   	int_least8_t;
typedef uint32_t  	uint_least8_t;
#define __int_least8_t_defined 1
#endif

#if !__int_least16_t_defined
typedef int32_t	   	int_least16_t;
typedef uint32_t  	uint_least16_t;
#define __int_least16_t_defined 1
#endif
#endif

#if __have_long64
typedef signed long int64_t;
typedef unsigned long uint64_t;
#define __int64_t_defined 1
#elif __have_longlong64
typedef signed long long int64_t;
typedef unsigned long long uint64_t;
#define __int64_t_defined 1
#elif  __STDINT_EXP(INT_MAX) > 0x7fffffff
typedef signed int int64_t;
typedef unsigned int uint64_t;
#define __int64_t_defined 1
#endif

#if __int64_t_defined
typedef int64_t   	int_least64_t;
typedef uint64_t 	uint_least64_t;
#define __int_least64_t_defined 1

#if !__int_least8_t_defined
typedef int64_t	   	int_least8_t;
typedef uint64_t  	uint_least8_t;
#define __int_least8_t_defined 1
#endif

#if !__int_least16_t_defined
typedef int64_t	   	int_least16_t;
typedef uint64_t  	uint_least16_t;
#define __int_least16_t_defined 1
#endif

#if !__int_least32_t_defined
typedef int64_t	   	int_least32_t;
typedef uint64_t  	uint_least32_t;
#define __int_least32_t_defined 1
#endif
#endif

/*
 * Fastest minimum-width integer types
 *
 * Assume int to be the fastest type for all types with a width 
 * less than __INT_MAX__ rsp. INT_MAX
 */
#if __STDINT_EXP(INT_MAX) >= 0x7f
  typedef signed int int_fast8_t;
  typedef unsigned int uint_fast8_t;
#define __int_fast8_t_defined 1
#endif

#if __STDINT_EXP(INT_MAX) >= 0x7fff
  typedef signed int int_fast16_t;
  typedef unsigned int uint_fast16_t;
#define __int_fast16_t_defined 1
#endif

#if __STDINT_EXP(INT_MAX) >= 0x7fffffff
  typedef signed int int_fast32_t;
  typedef unsigned int uint_fast32_t;
#define __int_fast32_t_defined 1
#endif

#if __STDINT_EXP(INT_MAX) > 0x7fffffff
  typedef signed int int_fast64_t;
  typedef unsigned int uint_fast64_t;
#define __int_fast64_t_defined 1
#endif

/*
 * Fall back to [u]int_least<N>_t for [u]int_fast<N>_t types
 * not having been defined, yet.
 * Leave undefined, if [u]int_least<N>_t should not be available.
 */
#if !__int_fast8_t_defined
#if __int_least8_t_defined
  typedef int_least8_t int_fast8_t;
  typedef uint_least8_t uint_fast8_t;
#define __int_fast8_t_defined 1
#endif
#endif

#if !__int_fast16_t_defined
#if __int_least16_t_defined
  typedef int_least16_t int_fast16_t;
  typedef uint_least16_t uint_fast16_t;
#define __int_fast16_t_defined 1
#endif
#endif

#if !__int_fast32_t_defined
#if __int_least32_t_defined
  typedef int_least32_t int_fast32_t;
  typedef uint_least32_t uint_fast32_t;
#define __int_fast32_t_defined 1
#endif
#endif

#if !__int_fast64_t_defined
#if __int_least64_t_defined
  typedef int_least64_t int_fast64_t;
  typedef uint_least64_t uint_fast64_t;
#define __int_fast64_t_defined 1
#endif
#endif

/* Greatest-width integer types */
/* Modern GCCs provide __INTMAX_TYPE__ */
#if defined(__INTMAX_TYPE__)
  typedef __INTMAX_TYPE__ intmax_t;
#elif __have_longlong64
  typedef signed long long intmax_t;
#else
  typedef signed long intmax_t;
#endif

/* Modern GCCs provide __UINTMAX_TYPE__ */
#if defined(__UINTMAX_TYPE__)
  typedef __UINTMAX_TYPE__ uintmax_t;
#elif __have_longlong64
  typedef unsigned long long uintmax_t;
#else
  typedef unsigned long uintmax_t;
#endif

/*
 * GCC doesn't provide an appropriate macro for [u]intptr_t
 * For now, use __PTRDIFF_TYPE__
 */
#if defined(__PTRDIFF_TYPE__)
typedef signed __PTRDIFF_TYPE__ intptr_t;
typedef unsigned __PTRDIFF_TYPE__ uintptr_t;
#define INTPTR_MAX PTRDIFF_MAX
#define INTPTR_MIN PTRDIFF_MIN
#ifdef __UINTPTR_MAX__
#define UINTPTR_MAX __UINTPTR_MAX__
#else
#define UINTPTR_MAX (2UL * PTRDIFF_MAX + 1)
#endif
#else
/*
 * Fallback to hardcoded values, 
 * should be valid on cpu's with 32bit int/32bit void*
 */
typedef signed long intptr_t;
typedef unsigned long uintptr_t;
#define INTPTR_MAX __STDINT_EXP(LONG_MAX)
#define INTPTR_MIN (-__STDINT_EXP(LONG_MAX) - 1)
#define UINTPTR_MAX (__STDINT_EXP(LONG_MAX) * 2UL + 1)
#endif

/* Limits of Specified-Width Integer Types */

#if __int8_t_defined
#define INT8_MIN 	-128
#define INT8_MAX 	 127
#define UINT8_MAX 	 255
#endif

#if __int_least8_t_defined
#define INT_LEAST8_MIN 	-128
#define INT_LEAST8_MAX 	 127
#define UINT_LEAST8_MAX	 255
#else
#error required type int_least8_t missing
#endif

#if __int16_t_defined
#define INT16_MIN 	-32768
#define INT16_MAX 	 32767
#define UINT16_MAX 	 65535
#endif

#if __int_least16_t_defined
#define INT_LEAST16_MIN	-32768
#define INT_LEAST16_MAX	 32767
#define UINT_LEAST16_MAX 65535
#else
#error required type int_least16_t missing
#endif

#if __int32_t_defined
#define INT32_MIN 	 (-2147483647-1)
#define INT32_MAX 	 2147483647
#define UINT32_MAX       4294967295U
#endif

#if __int_least32_t_defined
#define INT_LEAST32_MIN  (-2147483647-1)
#define INT_LEAST32_MAX  2147483647
#define UINT_LEAST32_MAX 4294967295U
#else
#error required type int_least32_t missing
#endif

#if __int64_t_defined
#if __have_long64
#define INT64_MIN 	(-9223372036854775807L-1L)
#define INT64_MAX 	 9223372036854775807L
#define UINT64_MAX 	18446744073709551615U
#elif __have_longlong64
#define INT64_MIN 	(-9223372036854775807LL-1LL)
#define INT64_MAX 	 9223372036854775807LL
#define UINT64_MAX 	18446744073709551615ULL
#endif
#endif

#if __int_least64_t_defined
#if __have_long64
#define INT_LEAST64_MIN  (-9223372036854775807L-1L)
#define INT_LEAST64_MAX  9223372036854775807L
#define UINT_LEAST64_MAX 18446744073709551615U
#elif __have_longlong64
#define INT_LEAST64_MIN  (-9223372036854775807LL-1LL)
#define INT_LEAST64_MAX  9223372036854775807LL
#define UINT_LEAST64_MAX 18446744073709551615ULL
#endif
#endif

#if __int_fast8_t_defined
#if __STDINT_EXP(INT_MAX) >= 0x7f
#define INT_FAST8_MIN	(-__STDINT_EXP(INT_MAX) - 1)
#define INT_FAST8_MAX	__STDINT_EXP(INT_MAX)
#define UINT_FAST8_MAX	(__STDINT_EXP(INT_MAX) * 2U + 1U)
#else
#define INT_FAST8_MIN	INT8_MIN
#define INT_FAST8_MAX	INT8_MAX
#define UINT_FAST8_MAX	UINT8_MAX
#endif
#endif /* __int_fast8_t_defined */

#if __int_fast16_t_defined
#if __STDINT_EXP(INT_MAX) >= 0x7fff
#define INT_FAST16_MIN	(-__STDINT_EXP(INT_MAX) - 1)
#define INT_FAST16_MAX	__STDINT_EXP(INT_MAX)
#define UINT_FAST16_MAX	(__STDINT_EXP(INT_MAX) * 2U + 1U)
#else
#define INT_FAST16_MIN	INT16_MIN
#define INT_FAST16_MAX	INT16_MAX
#define UINT_FAST16_MAX	UINT16_MAX
#endif
#endif /* __int_fast16_t_defined */

#if __int_fast32_t_defined
#if __STDINT_EXP(INT_MAX) >= 0x7fffffff
#define INT_FAST32_MIN	(-__STDINT_EXP(INT_MAX) - 1)
#define INT_FAST32_MAX	__STDINT_EXP(INT_MAX)
#define UINT_FAST32_MAX	(__STDINT_EXP(INT_MAX) * 2U + 1U)
#else
#define INT_FAST32_MIN	INT32_MIN
#define INT_FAST32_MAX	INT32_MAX
#define UINT_FAST32_MAX	UINT32_MAX
#endif
#endif /* __int_fast32_t_defined */

#if __int_fast64_t_defined
#if __STDINT_EXP(INT_MAX) > 0x7fffffff
#define INT_FAST64_MIN	(-__STDINT_EXP(INT_MAX) - 1)
#define INT_FAST64_MAX	__STDINT_EXP(INT_MAX)
#define UINT_FAST64_MAX	(__STDINT_EXP(INT_MAX) * 2U + 1U)
#else
#define INT_FAST64_MIN	INT64_MIN
#define INT_FAST64_MAX	INT64_MAX
#define UINT_FAST64_MAX	UINT64_MAX
#endif
#endif /* __int_fast64_t_defined */

#ifdef __INTMAX_MAX__
#define INTMAX_MAX __INTMAX_MAX__
#define INTMAX_MIN (-INTMAX_MAX - 1)
#elif defined(__INTMAX_TYPE__)
/* All relevant GCC versions prefer long to long long for intmax_t.  */
#define INTMAX_MAX INT64_MAX
#define INTMAX_MIN INT64_MIN
#endif

#ifdef __UINTMAX_MAX__
#define UINTMAX_MAX __UINTMAX_MAX__
#elif defined(__UINTMAX_TYPE__)
/* All relevant GCC versions prefer long to long long for intmax_t.  */
#define UINTMAX_MAX UINT64_MAX
#endif

/* This must match size_t in stddef.h, currently long unsigned int */
#define SIZE_MAX (__STDINT_EXP(LONG_MAX) * 2UL + 1)

/* This must match sig_atomic_t in <signal.h> (currently int) */
#define SIG_ATOMIC_MIN (-__STDINT_EXP(INT_MAX) - 1)
#define SIG_ATOMIC_MAX __STDINT_EXP(INT_MAX)

/* This must match ptrdiff_t  in <stddef.h> (currently long int) */
#define PTRDIFF_MIN (-__STDINT_EXP(LONG_MAX) - 1L)
#define PTRDIFF_MAX __STDINT_EXP(LONG_MAX)

/* This must match the definition in <wchar.h> */
#ifndef WCHAR_MIN
#define WCHAR_MIN 0
#endif
#ifndef WCHAR_MAX
#ifdef __WCHAR_MAX__
#define WCHAR_MAX __WCHAR_MAX__
#else
#define WCHAR_MAX 0x7fffffffu
#endif
#endif

/* wint_t is unsigned int on almost all GCC targets.  */
#ifdef __WINT_MAX__
#define WINT_MAX __WINT_MAX__
#else
#define WINT_MAX (__STDINT_EXP(INT_MAX) * 2U + 1U)
#endif
#ifdef __WINT_MIN__
#define WINT_MIN __WINT_MIN__
#else
#define WINT_MIN 0U
#endif

/** Macros for minimum-width integer constant expressions */
#define INT8_C(x)	x
#define UINT8_C(x)	x##U

#define INT16_C(x)	x
#define UINT16_C(x)	x##U

#if __have_long32
#define INT32_C(x)	x##L
#define UINT32_C(x)	x##UL
#else
#define INT32_C(x)	x
#define UINT32_C(x)	x##U
#endif

#if __int64_t_defined
#if __have_longlong64
#define INT64_C(x)	x##LL
#define UINT64_C(x)	x##ULL
#else
#define INT64_C(x)	x##L
#define UINT64_C(x)	x##UL
#endif
#endif

/** Macros for greatest-width integer constant expression */
#if __have_longlong64
#define INTMAX_C(x)	x##LL
#define UINTMAX_C(x)	x##ULL
#else
#define INTMAX_C(x)	x##L
#define UINTMAX_C(x)	x##UL
#endif


#if defined(__cplusplus) || defined(__XC__)
}
#endif

#endif /* _STDINT_H */
