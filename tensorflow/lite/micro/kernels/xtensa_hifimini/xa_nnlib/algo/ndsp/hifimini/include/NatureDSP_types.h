/*******************************************************************************
* Copyright (c) 2018-2020 Cadence Design Systems, Inc.
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to use this Software with Cadence processor cores only and
* not with any other processors and platforms, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

******************************************************************************/
/* ------------------------------------------------------------------------ */
/* Copyright (c) 2018 by Cadence Design Systems, Inc. ALL RIGHTS RESERVED.  */
/* These coded instructions, statements, and computer programs ("Cadence    */
/* Libraries") are the copyrighted works of Cadence Design Systems Inc.	    */
/* Cadence IP is licensed for use with Cadence processor cores only and     */
/* must not be used for any other processors and platforms. Your use of the */
/* Cadence Libraries is subject to the terms of the license agreement you   */
/* have entered into with Cadence Design Systems, or a sublicense granted   */
/* to you by a direct Cadence licensee.                                     */
/* ------------------------------------------------------------------------ */
/*  IntegrIT, Ltd.   www.integrIT.com, info@integrIT.com                    */
/*                                                                          */
/* DSP Library                                                              */
/*                                                                          */
/* This library contains copyrighted materials, trade secrets and other     */
/* proprietary information of IntegrIT, Ltd. This software is licensed for  */
/* use with Cadence processor cores only and must not be used for any other */
/* processors and platforms. The license to use these sources was given to  */
/* Cadence, Inc. under Terms and Condition of a Software License Agreement  */
/* between Cadence, Inc. and IntegrIT, Ltd.                                 */
/* ------------------------------------------------------------------------ */
/*          Copyright (C) 2015-2018 IntegrIT, Limited.                      */
/*                      All Rights Reserved.                                */
/* ------------------------------------------------------------------------ */
#ifndef __NATUREDSPTYPES_H__
#define __NATUREDSPTYPES_H__

#include <stddef.h>

#ifndef COMPILER_ANSI
// ----------------------------------------------------------
//             Compilers autodetection
// ----------------------------------------------------------
#define ___UNKNOWN_COMPILER_YET
#ifdef __ICC
  #define COMPILER_INTEL /* Intel C/C++ */
  #undef ___UNKNOWN_COMPILER_YET
#endif

#ifdef ___UNKNOWN_COMPILER_YET
 #ifdef _MSC_VER

#ifdef _ARM_
  #define COMPILER_CEARM9E /* Microsoft Visual C++,ARM9E */
#else
  #define COMPILER_MSVC /* Microsoft Visual C++ */
#endif

  #undef ___UNKNOWN_COMPILER_YET
 #endif
#endif

#ifdef ___UNKNOWN_COMPILER_YET
 #ifdef _QC
  #define COMPILER_MSQC /* Microsoft Quick C */
  #undef ___UNKNOWN_COMPILER_YET
 #endif
#endif


#ifdef ___UNKNOWN_COMPILER_YET
 #ifdef __BORLANDC__
  #define COMPILER_BORLAND /* Some Borland compiler */
  #undef ___UNKNOWN_COMPILER_YET
 #endif
#endif

#ifdef ___UNKNOWN_COMPILER_YET
 #ifdef __IBMC__
  #define COMPILER_IBM	/* IBM Visual Age for C++ */
  #undef ___UNKNOWN_COMPILER_YET
 #endif
#endif

#ifdef ___UNKNOWN_COMPILER_YET
 #ifdef __SC__
  #ifndef COMPILER_SYMANTEC
   #define COMPILER_SYMANTEC	/* Symantec C++ */
   #undef ___UNKNOWN_COMPILER_YET
  #endif
 #endif
#endif

#ifdef ___UNKNOWN_COMPILER_YET
 #ifdef __ZTC__
  #define COMPILER_ZORTECH	/* Zortech C/C++ 3.x */
  #undef ___UNKNOWN_COMPILER_YET
 #endif
#endif

#ifdef ___UNKNOWN_COMPILER_YET
 #ifdef __WATCOMC__
  #define COMPILER_WATCOM	/* Watcom C/C++ */
  #undef ___UNKNOWN_COMPILER_YET
 #endif
#endif

#ifdef ___UNKNOWN_COMPILER_YET
 #ifdef __CC_ARM
  #define COMPILER_ARM	/* ARM C/C++ */
  #undef ___UNKNOWN_COMPILER_YET
 #endif
#endif

#ifdef ___UNKNOWN_COMPILER_YET
 #ifdef _TMS320C6X
   #if defined (_TMS320C6400)
   #define COMPILER_C64
   #undef ___UNKNOWN_COMPILER_YET
   #endif
   #if defined   (_TMS320C6400_PLUS)
   #define COMPILER_C64PLUS
   #undef ___UNKNOWN_COMPILER_YET
   #endif
 #endif
#endif

#ifdef ___UNKNOWN_COMPILER_YET
 #ifdef __TMS320C55X__
  #define COMPILER_C55
  #undef ___UNKNOWN_COMPILER_YET
 #endif
#endif

#ifdef ___UNKNOWN_COMPILER_YET
 #ifdef __ADSPBLACKFIN__
  #define COMPILER_ADSP_BLACKFIN
  #undef ___UNKNOWN_COMPILER_YET
 #endif
#endif

#ifdef ___UNKNOWN_COMPILER_YET
 #ifdef __XCC__
   #define COMPILER_XTENSA
   #undef ___UNKNOWN_COMPILER_YET
 #endif
#endif

#ifdef ___UNKNOWN_COMPILER_YET
 #ifdef __GNUC__
   #ifdef __arm__
     #ifndef COMPILER_GNU_ARM
     #endif
     #define COMPILER_GNUARM /* GNU C/C++ compiler*/
   #else
     // GNU GCC x86 compiler
     #ifndef COMPILER_GNU
     #endif
     #define COMPILER_GNU /* GNU C/C++ */
   #endif
   #undef ___UNKNOWN_COMPILER_YET
 #endif
#endif

#ifdef ___UNKNOWN_COMPILER_YET
  #error  Unknown compiler
#endif


#endif //#ifndef COMPILER_ANSI

// ----------------------------------------------------------
//             Common types
// ----------------------------------------------------------
#if defined (COMPILER_GNU) | defined (COMPILER_GNUARM) | defined (COMPILER_XTENSA)
/*
  typedef signed char   int8_t;
  typedef unsigned char uint8_t;
*/
  #include <inttypes.h>
#elif defined (COMPILER_C64)
	#include <stdint.h>
#elif defined (COMPILER_C55)
	#include <stdint.h>
    typedef signed char   int8_t;
    typedef unsigned char uint8_t;
#elif defined(COMPILER_ADSP_BLACKFIN)
  typedef signed char   int8_t;
  typedef unsigned char uint8_t;
  typedef unsigned long  uint32_t;
  typedef unsigned short uint16_t;
  typedef long           int32_t;
  typedef short          int16_t;
  typedef long long          int64_t;
  typedef unsigned long long uint64_t;
  typedef uint32_t  uintptr_t;
#else
  typedef signed char   int8_t;
  typedef unsigned char uint8_t;
  typedef unsigned long  uint32_t;
  typedef unsigned short uint16_t;
  typedef long           int32_t;
  typedef short          int16_t;
  typedef __int64          int64_t;
  typedef unsigned __int64 uint64_t;
#endif



#if defined(COMPILER_CEARM9E)
typedef uint32_t uintptr_t;
#endif

#if defined(COMPILER_ARM)
typedef uint32_t uintptr_t;
#endif

typedef struct tagComplex16_t
{
  int16_t re;
  int16_t im;
} complex16_t;

typedef struct tagComplex32_t
{
  int32_t re;
  int32_t im;
} complex32_t;

// ----------------------------------------------------------
//	      Language-dependent definitions
// ----------------------------------------------------------
#ifdef __cplusplus

  #undef  extern_C
  #define extern_C extern "C"

#else

  #undef extern_C
  #define extern_C

  #ifndef false
  #define false 0
  #endif
  #ifndef true
  #define true 1
  #endif

#endif

/*    Assertion support                   */
#if !defined(_ASSERT)
  #include <assert.h>
  #if defined(_DEBUG) && defined(COMPILER_MSVC)
    #define ASSERT(x)  assert(x)
  #else

//#undef ASSERT
#ifndef ASSERT
	#define ASSERT(_ignore)  ((void)0)
#endif

  #endif  /* _DEBUG */
#else  /* ASSERT*/
  #define ASSERT(exp) \
	{\
  		extern void ExternalAssertHandler(void *, void *, unsigned);\
		(void)( (exp) || (ExternalAssertHandler(#exp, __FILE__, __LINE__), 0) );\
	}
#endif  /* ASSERT */


/*** Inline methods definition ***/
#undef inline_
#if (defined COMPILER_MSVC)||(defined COMPILER_CEARM9E)
  #define inline_ __inline
#elif defined (COMPILER_ADSP_BLACKFIN)
  #define inline_ inline
#elif defined(COMPILER_ANSI)
  #define inline_
#elif (defined COMPILER_GNU)||(defined COMPILER_GNUARM)||(defined COMPILER_ARM)
  #define inline_ static inline
#else
  #define inline_ static inline
#endif


#ifndef MAX_INT16
#define MAX_INT16 ((int16_t)0x7FFF)
#endif
#ifndef MIN_INT16
#define MIN_INT16 ((int16_t)0x8000)
#endif
#ifndef MAX_INT32
#define MAX_INT32 ((int32_t)0x7FFFFFFFL)
#endif
#ifndef MIN_INT32
#define MIN_INT32 ((int32_t)0x80000000L)
#endif
#ifndef MIN_INT64
#define MIN_INT64 ((int64_t)0x8000000000000000LL)
#endif
#ifndef MAX_INT64
#define MAX_INT64 ((int64_t)0x7fffffffffffffffLL)
#endif

// size of variables in bytes
#ifdef COMPILER_C55
  #define SIZEOF_BYTE(x)  (sizeof(x)<<1)
#else
  #define SIZEOF_BYTE(x)  sizeof(x)
#endif

//---------------------------------------
// special keywords definition
// restrict  keyword means that the memory
//           is addressed exclusively via
//	     this pointer
// onchip    keyword means that the memory
//           is on-chip and can not be
//           accessed via external bus
//---------------------------------------
#if   defined (COMPILER_C55)
  #define NASSERT _nassert
#elif defined (COMPILER_C64)
  #define onchip
  #define NASSERT _nassert
#elif defined (COMPILER_ADSP_BLACKFIN)
  #define onchip
  #define NASSERT(x) __builtin_assert(x)
#elif defined (COMPILER_GNUARM)
  #define onchip
  #define NASSERT(x) {(void)__builtin_expect((x)!=0,1);}
  #define restrict __restrict
#elif defined (COMPILER_GNU)
  #define onchip
  #define NASSERT(x)
  #define restrict __restrict
#elif defined (COMPILER_CEARM9E)
  #define onchip
  #define NASSERT(x)
  #define restrict
#elif defined (COMPILER_XTENSA)
  #define restrict __restrict
  #define onchip
  #define NASSERT(x) {(void)__builtin_expect((x)!=0,1);}
#else
  #define restrict
  #define onchip
  #define NASSERT ASSERT
#endif
#if defined (COMPILER_ADSP_BLACKFIN)
#define NASSERT_ALIGN(addr,align) __builtin_aligned(addr, align)
#else
#define NASSERT_ALIGN(addr,align) NASSERT(((uintptr_t)addr)%align==0)
#endif
#define NASSERT_ALIGN2(addr) NASSERT_ALIGN(addr,2)
#define NASSERT_ALIGN4(addr) NASSERT_ALIGN(addr,4)
#define NASSERT_ALIGN8(addr) NASSERT_ALIGN(addr,8)
#define NASSERT_ALIGN16(addr) NASSERT_ALIGN(addr,16)

/* ----------------------------------------------------------
             Common types
 ----------------------------------------------------------*/
#if defined (COMPILER_GNU) | defined (COMPILER_GNUARM) | defined (COMPILER_XTENSA)
  #include <inttypes.h>
#else
  typedef signed char        int8_t;
  typedef unsigned char      uint8_t;
  typedef unsigned long      uint32_t;
  typedef unsigned short     uint16_t;
  typedef long               int32_t;
  typedef short              int16_t;
  typedef __int64            int64_t;
  typedef unsigned __int64   uint64_t;
#endif

typedef int16_t float16_t;
typedef float   float32_t;
typedef double  float64_t;
typedef int16_t fract16;
typedef int32_t fract32;

typedef int32_t f24;
#ifdef COMPILER_MSVC
    typedef __int64 f48;
    typedef __int64 i56;
#else
    typedef long long f48;
    typedef long long i56;
#endif

typedef union tag_complex_fract16
{
    struct
    {
        fract16 re, im;
    }s;
    uint32_t a; /* just for 32-bit alignment */
}
complex_fract16;

typedef union tag_complex_fract32
{
    struct
    {
        fract32 re, im;
    }s;
    uint64_t a;/* just for 64-bit alignment */
}
complex_fract32;

#if defined(COMPILER_MSVC)
/* Note: Visual Studio does not support C99 compatible complex types yet */
typedef union tag_complex_float
{
    struct
    {
        float32_t re, im;
    }s;
    uint64_t a;/* just for 64-bit alignment */
}
complex_float;
typedef union tag_complex_double
{
    struct
    {
      float64_t re, im;
    }s;
    uint64_t a[2];/* only 64-bit alignment under Visual Studio :(( */
}
complex_double;

inline_ float32_t crealf(complex_float x) { return x.s.re; }
inline_ float32_t cimagf(complex_float x) { return x.s.im; }
inline_ float64_t creal (complex_double x) { return x.s.re; }
inline_ float64_t cimag (complex_double x) { return x.s.im; }
#else
/* C99 compatible type */
#include <complex.h>
#define complex_float  __complex__ float
#define complex_double __complex__ double
#endif

/* complex half-precision datatype */
typedef union tag_complex_half
{
    struct
    {
        float16_t re, im;
    }s;
    uint32_t a;/* just for 32-bit alignment */
}
complex_half;

inline_ float16_t crealh(complex_half x) { return x.s.re; }
inline_ float16_t cimagh(complex_half x) { return x.s.im; }

/* ----------------------------------------------------------
 Redefinition layer for math.h from the standard library.
 ----------------------------------------------------------*/

/*
 xcc compiler may replace invocations of selected math functions (fabs,
 sqrt, etc) with special code sequences. This option is globally disabled
 through -fno-builtin to avoid name conflicts with LIBDSP, but built-in
 functions still may be used by means of BUILTIN_MATH macro, e.g.
 y = BUILTIN_MATH(fabsf)( x ).
*/

#ifdef COMPILER_XTENSA
#define BUILTIN_MATH(fxn)  __builtin_ ## fxn
#else
#define BUILTIN_MATH(fxn)  fxn
#endif

#if 1
#define STDLIB_MATH(fxn) fxn
#define LIBDSP_MATH(fxn) fxn
#else
/*
 LIBDSP redefines a number of functions from the standard math.h needed by
 the reference code. To avoid name conflicts, the redefinition layer adds
 _libdsp suffix to LIBDSP function names, and provides the STDLIB_MATH() macro
 for calling standard library functions, e.g. STD_MATH(cos)(x).
*/

/* support of xclib */
#ifdef acosf
#undef acosf
#endif
#ifdef acos
#undef acos
#endif
#ifdef asinf
#undef asinf
#endif
#ifdef asin
#undef asin
#endif
#ifdef atanf
#undef atanf
#endif
#ifdef atan
#undef atan
#endif
#ifdef atan2f
#undef atan2f
#endif
#ifdef atan2
#undef atan2
#endif
#ifdef cosf
#undef cosf
#endif
#ifdef cos
#undef cos
#endif
#ifdef sinf
#undef sinf
#endif
#ifdef sin
#undef sin
#endif
#ifdef tanf
#undef tanf
#endif
#ifdef tan
#undef tan
#endif
#ifdef coshf
#undef coshf
#endif
#ifdef cosh
#undef cosh
#endif
#ifdef sinhf
#undef sinhf
#endif
#ifdef sinh
#undef sinh
#endif
#ifdef tanhf
#undef tanhf
#endif
#ifdef tanh
#undef tanh
#endif
#ifdef floorf
#undef floorf
#endif
#ifdef floor
#undef floor
#endif
#ifdef ceilf
#undef ceilf
#endif
#ifdef ceil
#undef ceil
#endif
#ifdef fmodf
#undef fmodf
#endif
#ifdef fmod
#undef fmod
#endif
#ifdef ldexpf
#undef ldexpf
#endif
#ifdef ldexp
#undef ldexp
#endif
#ifdef logf
#undef logf
#endif
#ifdef log
#undef log
#endif
#ifdef log10f
#undef log10f
#endif
#ifdef log10
#undef log10
#endif
#ifdef expf
#undef expf
#endif
#ifdef exp
#undef exp
#endif
#ifdef powf
#undef powf
#endif
#ifdef pow
#undef pow
#endif
#ifdef copysignf
#undef copysignf
#endif
#ifdef conjf
#undef conjf
#endif

#define STDLIB_MATH(fxn) fxn ## _stdlib
#define LIBDSP_MATH(fxn) fxn ## _libdsp

#define acosf        LIBDSP_MATH(acosf)
#define acos         LIBDSP_MATH(acos)
#define asinf        LIBDSP_MATH(asinf)
#define asin         LIBDSP_MATH(asin)
#define atanf        LIBDSP_MATH(atanf)
#define atan         LIBDSP_MATH(atan)
#define atan2f       LIBDSP_MATH(atan2f)
#define atan2        LIBDSP_MATH(atan2)
#define cosf         LIBDSP_MATH(cosf)
#define cos          LIBDSP_MATH(cos)
#define sinf         LIBDSP_MATH(sinf)
#define sin          LIBDSP_MATH(sin)
#define tanf         LIBDSP_MATH(tanf)
#define tan          LIBDSP_MATH(tan)
#define coshf        LIBDSP_MATH(coshf)
#define cosh         LIBDSP_MATH(cosh)
#define sinhf        LIBDSP_MATH(sinhf)
#define sinh         LIBDSP_MATH(sinh)
#define tanhf        LIBDSP_MATH(tanhf)
#define tanh         LIBDSP_MATH(tanh)
#define floorf       LIBDSP_MATH(floorf)
#define floor        LIBDSP_MATH(floor)
#define ceilf        LIBDSP_MATH(ceilf)
#define ceil         LIBDSP_MATH(ceil)
#define fmodf        LIBDSP_MATH(fmodf)
#define fmod         LIBDSP_MATH(fmod)
#define ldexpf       LIBDSP_MATH(ldexpf)
#define ldexp        LIBDSP_MATH(ldexp)
#define logf         LIBDSP_MATH(logf)
#define log          LIBDSP_MATH(log)
#define log10f       LIBDSP_MATH(log10f)
#define log10        LIBDSP_MATH(log10)
#define expf         LIBDSP_MATH(expf)
#define exp          LIBDSP_MATH(exp)
#define powf         LIBDSP_MATH(powf)
#define pow          LIBDSP_MATH(pow)
#define copysignf    LIBDSP_MATH(copysignf)
#define conjf        LIBDSP_MATH(conjf)

#ifdef __cplusplus
extern "C" {
#endif

float32_t STDLIB_MATH(acosf) ( float32_t x );
float64_t STDLIB_MATH(acos) ( float64_t x );
float32_t STDLIB_MATH(asinf) ( float32_t  x );
float64_t STDLIB_MATH(asin) ( float64_t  x );
float32_t STDLIB_MATH(atanf) ( float32_t x );
float64_t STDLIB_MATH(atan) ( float64_t x );
float32_t STDLIB_MATH(atan2f) ( float32_t y, float32_t x );
float64_t STDLIB_MATH(atan2) ( float64_t y, float64_t x );
float32_t STDLIB_MATH(cosf) ( float32_t x );
float64_t STDLIB_MATH(cos) ( float64_t x );
float32_t STDLIB_MATH(sinf) ( float32_t x );
float64_t STDLIB_MATH(sin) ( float64_t x );
float32_t STDLIB_MATH(tanf) ( float32_t x );
float64_t STDLIB_MATH(tan) ( float64_t x );
float32_t STDLIB_MATH(coshf) ( float32_t x );
float64_t STDLIB_MATH(cosh) ( float64_t x );
float32_t STDLIB_MATH(sinhf) ( float32_t x );
float64_t STDLIB_MATH(sinh) ( float64_t x );
float32_t STDLIB_MATH(tanhf) ( float32_t x );
float64_t STDLIB_MATH(tanh) ( float64_t x );
float32_t STDLIB_MATH(floorf) ( float32_t x );
float64_t STDLIB_MATH(floor) ( float64_t x );
float32_t STDLIB_MATH(ceilf) ( float32_t x );
float64_t STDLIB_MATH(ceil) ( float64_t x );
float32_t STDLIB_MATH(fmodf) ( float32_t x, float32_t y );
float64_t STDLIB_MATH(fmod) ( float64_t x, float64_t y );
float32_t STDLIB_MATH(ldexpf) ( float32_t x, int n );
float64_t STDLIB_MATH(ldexp) ( float64_t x, int n );
float32_t STDLIB_MATH(logf) ( float32_t x );
float64_t STDLIB_MATH(log) ( float64_t x );
float32_t STDLIB_MATH(log10f) ( float32_t x );
float64_t STDLIB_MATH(log10) ( float64_t x );
float32_t STDLIB_MATH(expf) ( float32_t x );
float64_t STDLIB_MATH(exp) ( float64_t x );
float32_t STDLIB_MATH(powf) ( float32_t x, float32_t y );
float64_t STDLIB_MATH(pow) ( float64_t x, float64_t y );
float32_t STDLIB_MATH(copysignf) ( float32_t x, float32_t y );
float32_t STDLIB_MATH(conjf) (complex_float x);

#ifdef __cplusplus
};
#endif
#endif

/*    union data type for writing float32_t/float64_t constants in a bitexact form */
union ufloat32uint32 {  uint32_t  u;  float32_t f; };
union ufloat64uint64 {  uint64_t  u;  float64_t f; };

#if defined(__RENAMING__)
#include "__renaming__.h"
#endif

#endif //__NATUREDSPTYPES_H__

