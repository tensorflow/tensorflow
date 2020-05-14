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
#ifndef __COMMON_H__
#define __COMMON_H__

#include <assert.h>

#include "NatureDSP_types.h"
#ifdef __RENAMING__
#include "__renaming__.h"
#endif

#if defined COMPILER_XTENSA
  #include <xtensa/config/core-isa.h>
  #include <xtensa/tie/xt_core.h>
  #include <xtensa/tie/xt_misc.h>
  #include <xtensa/tie/xt_hifi2.h>
#if XCHAL_HAVE_HIFI4_VFPU
  #include <xtensa/tie/xt_FP.h>
#endif
#else
#if defined COMPILER_MSVC
  #pragma warning( disable :4800 4244)
#endif
/* the code below causes inclusion of file "cstub-"XTENSA_CORE".h" */
#define PPCAT_NX(A, B) A-B
#define PPCAT(A, B) PPCAT_NX(A, B)      /* Concatenate preprocessor tokens A and B after macro-expanding them. */
#define STRINGIZE_NX(A) #A              /* Turn A into a string literal without expanding macro definitions */
#define STRINGIZE(A) STRINGIZE_NX(A)    /*  Turn A into a string literal after macro-expanding it. */
//#include STRINGIZE(PPCAT(cstub,XTENSA_CORE).h)
//#include STRINGIZE(PPCAT(PPCAT(cstub,XTENSA_CORE),c.h))
#include "xtensa/tie/xt_hifi2.h"
#include "xtensa/config/core-isa.h"
#endif

//-----------------------------------------------------
// C99 pragma wrapper
//-----------------------------------------------------

#ifdef COMPILER_XTENSA
#define __Pragma(a) _Pragma(a)
#else
#define __Pragma(a)
#endif

#define IS_ALIGN(p) ((((int)(p))&0x7) == 0)

#ifdef _MSC_VER
    #define ALIGN(x)    _declspec(align(x))
#else
    #define ALIGN(x)    __attribute__((aligned(x)))
#endif

#define INV_TBL_BITS 7
extern const int32_t tab_invQ30[128];

#if XCHAL_HAVE_NSA
  #define NSA(n) XT_NSA(n)
#else
  inline_ int32_t NSA(int32_t n)
  {
    ae_q56s t;
    if (!n) return 31;
    t = AE_CVTQ48A32S(n);
    return AE_NSAQ56S(t)-8;
  }
#endif

// special XCC type casting of pointers
#ifdef __cplusplus
#define castxcc(type_,ptr)  (ptr)
#else
#define castxcc(type_,ptr)  (type_ *)(ptr)
#endif

// return 64-bit data converting from ae_int64
#ifdef __cplusplus
#define return_int64(x) return x;
#else
#define return_int64(x) {  union {ae_int64  ai;int64_t   i; } r; r.ai = x;  return r.i; }
#endif

#if  defined (__cplusplus) || defined(COMPILER_XTENSA)

#else
#error sorry, C compiler is not supported excluding the XCC
#endif


#ifdef COMPILER_MSVC
#define MSC_ALIGNED ALIGN(8)
#define GCC_ALIGNED
#else
#define MSC_ALIGNED
#define GCC_ALIGNED ALIGN(8)
#endif

typedef const int32_t * cint32_ptr;
typedef const uint64_t * cuint64_ptr;
typedef const short * cint16_ptr;

//-----------------------------------------------------
// API annotation helper macro
//-----------------------------------------------------

#ifdef COMPILER_XTENSA
#define ANNOTATE_ATTR __attribute__ ((section (".rodata.NatureDSP_Signal_annotation")))
#else
#define ANNOTATE_ATTR
#endif

#define ANNOTATE_FUN_REF(fun)   NatureDSP_Signal_annotation_##fun

#ifdef __cplusplus
#define ANNOTATE_FUN(fun,text) \
  extern "C" const char ANNOTATE_ATTR ANNOTATE_FUN_REF(fun)[] = (text)
#else
#define ANNOTATE_FUN(fun,text) \
  const char ANNOTATE_ATTR ANNOTATE_FUN_REF(fun)[] = (text)
#endif
//-----------------------------------------------------
// Conditionalization support
//-----------------------------------------------------
/* place DISCARD_FUN(retval_type,name) instead of function definition for functions
   to be discarded from the executable
   THIS WORKS only for external library functions declared as extern "C" and
   not supported for internal references without "C" qualifier!
*/
#ifdef COMPILER_MSVC
#pragma section( "$DISCARDED_FUNCTIONS" , execute, discard )
#pragma section( "$$$$$$$$$$" , execute, discard )
#define DISCARD_FUN(retval_type,name,arglist) __pragma (alloc_text( "$DISCARDED_FUNCTIONS",name))\
__pragma(section( "$DISCARDED_FUNCTIONS" , execute, discard ))\
__pragma (warning(push))\
__pragma (warning( disable : 4026 4716))\
retval_type name arglist {}\
__pragma (warning(pop))
#endif

#if defined (COMPILER_GNU)
#define F_UNDERSCORE " "
#define DISCARD_FUN(retval_type,name,arglist)    \
__asm__                        \
(                              \
".section unused_section\n"    \
".globl " F_UNDERSCORE STRINGIZE(name) "\n" \
".type "F_UNDERSCORE STRINGIZE(name)", @function \n"\
F_UNDERSCORE STRINGIZE(name) ":\n"          \
".text"                        \
);
#endif

#if defined(COMPILER_XTENSA)
#define DISCARD_FUN_FOR_NONVOID_RETURN(retval_type,name,arglist) \
__attribute__ ((section ("/DISCARD/"))) \
retval_type name arglist \
{ return (retval_type) 0; }

#define DISCARD_FUN(retval_type,name,arglist) \
__attribute__ ((section ("/DISCARD/"))) \
retval_type name arglist \
{  }
#endif

#ifdef __cplusplus
#define externC extern "C"
#else
#define externC extern
#endif

/* maximum size (in bytes) allocated storage on stack by temporary arrays inside library functions */
#define MAX_ALLOCA_SZ 512

#endif
