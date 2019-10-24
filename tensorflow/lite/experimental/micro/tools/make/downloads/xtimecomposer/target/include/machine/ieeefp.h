#ifndef __IEEE_BIG_ENDIAN
#ifndef __IEEE_LITTLE_ENDIAN

/* This file can define macros to choose variations of the IEEE float
   format:

   _FLT_LARGEST_EXPONENT_IS_NORMAL

	Defined if the float format uses the largest exponent for finite
	numbers rather than NaN and infinity representations.  Such a
	format cannot represent NaNs or infinities at all, but it's FLT_MAX
	is twice the IEEE value.

   _FLT_NO_DENORMALS

	Defined if the float format does not support IEEE denormals.  Every
	float with a zero exponent is taken to be a zero representation.
 
   ??? At the moment, there are no equivalent macros above for doubles and
   the macros are not fully supported by --enable-newlib-hw-fp.

   __IEEE_BIG_ENDIAN

        Defined if the float format is big endian.  This is mutually exclusive
        with __IEEE_LITTLE_ENDIAN.

   __IEEE_LITTLE_ENDIAN
 
        Defined if the float format is little endian.  This is mutually exclusive
        with __IEEE_BIG_ENDIAN.

   Note that one of __IEEE_BIG_ENDIAN or __IEEE_LITTLE_ENDIAN must be specified for a
   platform or error will occur.

   __IEEE_BYTES_LITTLE_ENDIAN

        This flag is used in conjunction with __IEEE_BIG_ENDIAN to describe a situation 
	whereby multiple words of an IEEE floating point are in big endian order, but the
	words themselves are little endian with respect to the bytes.

   _DOUBLE_IS_32BITS 

        This is used on platforms that support double by using the 32-bit IEEE
        float type.

   _FLOAT_ARG

        This represents what type a float arg is passed as.  It is used when the type is
        not promoted to double.
	
*/

#if (defined(__arm__) || defined(__thumb__)) && !defined(__MAVERICK__)
/* ARM traditionally used big-endian words; and within those words the
   byte ordering was big or little endian depending upon the target.
   Modern floating-point formats are naturally ordered; in this case
   __VFP_FP__ will be defined, even if soft-float.  */
#ifdef __VFP_FP__
# ifdef __ARMEL__
#  define __IEEE_LITTLE_ENDIAN
# else
#  define __IEEE_BIG_ENDIAN
# endif
#else
# define __IEEE_BIG_ENDIAN
# ifdef __ARMEL__
#  define __IEEE_BYTES_LITTLE_ENDIAN
# endif
#endif
#endif

#ifdef __hppa__
#define __IEEE_BIG_ENDIAN
#endif

#ifdef __SPU__
#define __IEEE_BIG_ENDIAN

#define isfinite(y) \
          (__extension__ ({__typeof__(y) __y = (y); \
                           (sizeof (__y) == sizeof (float))  ? (1) : \
                           fpclassify(__y) != FP_INFINITE && fpclassify(__y) != FP_NAN;}))
#define isinf(x) \
          (__extension__ ({__typeof__(x) __x = (x); \
                           (sizeof (__x) == sizeof (float))  ? (0) : __isinfd(__x);}))
#define isnan(x) \
          (__extension__ ({__typeof__(x) __x = (x); \
                           (sizeof (__x) == sizeof (float))  ? (0) : __isnand(__x);}))

/*
 * Macros for use in ieeefp.h. We can't just define the real ones here
 * (like those above) as we have name space issues when this is *not*
 * included via generic the ieeefp.h.
 */
#define __ieeefp_isnanf(x)	0
#define __ieeefp_isinff(x)	0
#define __ieeefp_finitef(x)	1
#endif

#ifdef __sparc__
#ifdef __LITTLE_ENDIAN_DATA__
#define __IEEE_LITTLE_ENDIAN
#else
#define __IEEE_BIG_ENDIAN
#endif
#endif

#if defined(__m68k__) || defined(__mc68000__)
#define __IEEE_BIG_ENDIAN
#endif

#if defined(__mc68hc11__) || defined(__mc68hc12__) || defined(__mc68hc1x__)
#define __IEEE_BIG_ENDIAN
#ifdef __HAVE_SHORT_DOUBLE__
# define _DOUBLE_IS_32BITS
#endif
#endif

#if defined (__H8300__) || defined (__H8300H__) || defined (__H8300S__) || defined (__H8500__) || defined (__H8300SX__)
#define __IEEE_BIG_ENDIAN
#define _FLOAT_ARG float
#define _DOUBLE_IS_32BITS
#endif

#ifdef __sh__
#ifdef __LITTLE_ENDIAN__
#define __IEEE_LITTLE_ENDIAN
#else
#define __IEEE_BIG_ENDIAN
#endif
#if defined(__SH2E__) || defined(__SH3E__) || defined(__SH4_SINGLE_ONLY__) || defined(__SH2A_SINGLE_ONLY__)
#define _DOUBLE_IS_32BITS
#endif
#endif

#ifdef _AM29K
#define __IEEE_BIG_ENDIAN
#endif

#ifdef _WIN32
#define __IEEE_LITTLE_ENDIAN
#endif

#ifdef __i386__
#define __IEEE_LITTLE_ENDIAN
#endif

#ifdef __i960__
#define __IEEE_LITTLE_ENDIAN
#endif

#ifdef __M32R__
#define __IEEE_BIG_ENDIAN
#endif

#if defined(_C4x) || defined(_C3x)
#define __IEEE_BIG_ENDIAN
#define _DOUBLE_IS_32BITS
#endif

#ifdef __TIC80__
#define __IEEE_LITTLE_ENDIAN
#endif

#ifdef __MIPSEL__
#define __IEEE_LITTLE_ENDIAN
#endif
#ifdef __MIPSEB__
#define __IEEE_BIG_ENDIAN
#endif

#ifdef __MMIX__
#define __IEEE_BIG_ENDIAN
#endif

#ifdef __D30V__
#define __IEEE_BIG_ENDIAN
#endif

/* necv70 was __IEEE_LITTLE_ENDIAN. */

#ifdef __W65__
#define __IEEE_LITTLE_ENDIAN
#define _DOUBLE_IS_32BITS
#endif

#if defined(__Z8001__) || defined(__Z8002__)
#define __IEEE_BIG_ENDIAN
#endif

#ifdef __m88k__
#define __IEEE_BIG_ENDIAN
#endif

#ifdef __mn10300__
#define __IEEE_LITTLE_ENDIAN
#endif

#ifdef __mn10200__
#define __IEEE_LITTLE_ENDIAN
#define _DOUBLE_IS_32BITS
#endif

#ifdef __v800
#define __IEEE_LITTLE_ENDIAN
#endif

#ifdef __v850
#define __IEEE_LITTLE_ENDIAN
#endif

#ifdef __D10V__
#define __IEEE_BIG_ENDIAN
#if __DOUBLE__ == 32
#define _DOUBLE_IS_32BITS
#endif
#endif

#ifdef __PPC__
#if (defined(_BIG_ENDIAN) && _BIG_ENDIAN) || (defined(_AIX) && _AIX)
#define __IEEE_BIG_ENDIAN
#else
#if (defined(_LITTLE_ENDIAN) && _LITTLE_ENDIAN) || (defined(__sun__) && __sun__) || (defined(_WIN32) && _WIN32)
#define __IEEE_LITTLE_ENDIAN
#endif
#endif
#endif

#if defined(__xcore__)
#define __IEEE_LITTLE_ENDIAN
#endif

#ifdef __xstormy16__
#define __IEEE_LITTLE_ENDIAN
#endif

#ifdef __arc__
#ifdef __big_endian__
#define __IEEE_BIG_ENDIAN
#else
#define __IEEE_LITTLE_ENDIAN
#endif
#endif

#ifdef __CRX__
#define __IEEE_LITTLE_ENDIAN
#endif

#ifdef __fr30__
#define __IEEE_BIG_ENDIAN
#endif

#ifdef __mcore__
#define __IEEE_BIG_ENDIAN
#endif

#ifdef __mt__
#define __IEEE_BIG_ENDIAN
#endif

#ifdef __frv__
#define __IEEE_BIG_ENDIAN
#endif

#ifdef __ia64__
#ifdef __BIG_ENDIAN__
#define __IEEE_BIG_ENDIAN
#else
#define __IEEE_LITTLE_ENDIAN
#endif
#endif

#ifdef __AVR__
#define __IEEE_LITTLE_ENDIAN
#define _DOUBLE_IS_32BITS
#endif

#if defined(__or32__) || defined(__or1k__) || defined(__or16__)
#define __IEEE_BIG_ENDIAN
#endif

#ifdef __IP2K__
#define __IEEE_BIG_ENDIAN
#define __SMALL_BITFIELDS
#define _DOUBLE_IS_32BITS
#endif

#ifdef __iq2000__
#define __IEEE_BIG_ENDIAN
#endif

#ifdef __MAVERICK__
#ifdef __ARMEL__
#  define __IEEE_LITTLE_ENDIAN
#else  /* must be __ARMEB__ */
#  define __IEEE_BIG_ENDIAN
#endif /* __ARMEL__ */
#endif /* __MAVERICK__ */

#ifdef __m32c__
#define __IEEE_LITTLE_ENDIAN
#define __SMALL_BITFIELDS
#endif

#ifdef __CRIS__
#define __IEEE_LITTLE_ENDIAN
#endif

#ifdef __BFIN__
#define __IEEE_LITTLE_ENDIAN
#endif

#ifdef __x86_64__
#define __IEEE_LITTLE_ENDIAN
#endif

#ifdef __mep__
#ifdef __LITTLE_ENDIAN__
#define __IEEE_LITTLE_ENDIAN
#else
#define __IEEE_BIG_ENDIAN
#endif
#endif

#ifndef __IEEE_BIG_ENDIAN
#ifndef __IEEE_LITTLE_ENDIAN
#error Endianess not declared!!
#endif /* not __IEEE_LITTLE_ENDIAN */
#endif /* not __IEEE_BIG_ENDIAN */

#endif /* not __IEEE_LITTLE_ENDIAN */
#endif /* not __IEEE_BIG_ENDIAN */

