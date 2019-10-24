#ifndef _LIBC_LIMITS_H_
# define _LIBC_LIMITS_H_	1

#include <newlib.h>

# ifdef _MB_LEN_MAX
#  define MB_LEN_MAX	_MB_LEN_MAX
# else
#  define MB_LEN_MAX    1
# endif

/* Maximum number of positional arguments, if _WANT_IO_POS_ARGS.  */
# ifndef NL_ARGMAX
#  define NL_ARGMAX 32
# endif

/* if do not have #include_next support, then we
   have to define the limits here. */
# if !defined __GNUC__ || __GNUC__ < 2

#  ifndef _LIMITS_H
#   define _LIMITS_H	1

#   include <sys/config.h>

/* Number of bits in a `char'.  */
#   undef CHAR_BIT
#   define CHAR_BIT 8

/* Minimum and maximum values a `signed char' can hold.  */
#   undef SCHAR_MIN
#   define SCHAR_MIN (-128)
#   undef SCHAR_MAX
#   define SCHAR_MAX 127

/* Maximum value an `unsigned char' can hold.  (Minimum is 0).  */
#   undef UCHAR_MAX
#   define UCHAR_MAX 255

/* Minimum and maximum values a `char' can hold.  */
#   ifdef __CHAR_UNSIGNED__
#    undef CHAR_MIN
#    define CHAR_MIN 0
#    undef CHAR_MAX
#    define CHAR_MAX 255
#   else
#    undef CHAR_MIN
#    define CHAR_MIN (-128)
#    undef CHAR_MAX
#    define CHAR_MAX 127
#   endif

/* Minimum and maximum values a `signed short int' can hold.  */
#   undef SHRT_MIN
/* For the sake of 16 bit hosts, we may not use -32768 */
#   define SHRT_MIN (-32767-1)
#   undef SHRT_MAX
#   define SHRT_MAX 32767

/* Maximum value an `unsigned short int' can hold.  (Minimum is 0).  */
#   undef USHRT_MAX
#   define USHRT_MAX 65535

/* Minimum and maximum values a `signed int' can hold.  */
#   ifndef __INT_MAX__
#    define __INT_MAX__ 2147483647
#   endif
#   undef INT_MIN
#   define INT_MIN (-INT_MAX-1)
#   undef INT_MAX
#   define INT_MAX __INT_MAX__

/* Maximum value an `unsigned int' can hold.  (Minimum is 0).  */
#   undef UINT_MAX
#   define UINT_MAX (INT_MAX * 2U + 1)

/* Minimum and maximum values a `signed long int' can hold.
   (Same as `int').  */
#   ifndef __LONG_MAX__
#    if defined (__alpha__) || (defined (__sparc__) && defined(__arch64__)) || defined (__sparcv9)
#     define __LONG_MAX__ 9223372036854775807L
#    else
#     define __LONG_MAX__ 2147483647L
#    endif /* __alpha__ || sparc64 */
#   endif
#   undef LONG_MIN
#   define LONG_MIN (-LONG_MAX-1)
#   undef LONG_MAX
#   define LONG_MAX __LONG_MAX__

/* Maximum value an `unsigned long int' can hold.  (Minimum is 0).  */
#   undef ULONG_MAX
#   define ULONG_MAX (LONG_MAX * 2UL + 1)

#   if !defined(__LONG_LONG_MAX__)
#    define __LONG_LONG_MAX__ 9223372036854775807LL
#   endif

#   if (defined (__STDC_VERSION__) && __STDC_VERSION__ >= 199901L) || defined(__XC__)
/* Minimum and maximum values a `signed long long int' can hold.  */
#    undef LLONG_MIN
#    define LLONG_MIN (-LLONG_MAX-1)
#    undef LLONG_MAX
#    define LLONG_MAX __LONG_LONG_MAX__

/* Maximum value an `unsigned long long int' can hold.  (Minimum is 0).  */
#    undef ULLONG_MAX
#    define ULLONG_MAX (LLONG_MAX * 2ULL + 1)
#   endif

#  if defined (__GNU_LIBRARY__) ? defined (__USE_GNU) : \
                                  !defined (__STRICT_ANSI__)
/* Minimum and maximum values a `signed long long int' can hold.  */
#    undef LONG_LONG_MIN
#    define LONG_LONG_MIN (-LONG_LONG_MAX-1)
#    undef LONG_LONG_MAX
#    define LONG_LONG_MAX __LONG_LONG_MAX__

/* Maximum value an `unsigned long long int' can hold.  (Minimum is 0).  */
#    undef ULONG_LONG_MAX
#    define ULONG_LONG_MAX (LONG_LONG_MAX * 2ULL + 1)
#   endif

#  endif /* _LIMITS_H  */
# endif	 /* GCC 2.  */

#endif	 /* !_LIBC_LIMITS_H_ */

#if defined __GNUC__ && !defined _GCC_LIMITS_H_
/* `_GCC_LIMITS_H_' is what GCC's file defines.  */
# include_next <limits.h>
#endif /* __GNUC__ && !_GCC_LIMITS_H_ */
