/*************************************************************************************\
* Copyright (C) CEVA(R) Inc. All rights reserved *
* *
* This information embodies materials and concepts, which are proprietary and *
* confidential to CEVA Inc., and is made available solely pursuant to the terms
*
* of a written license agreement, or NDA, or another written agreement, as *
* applicable ("CEVA Agreement"), with CEVA Inc. or any of its subsidiaries
("CEVA").  *
* *
* This information can be used only with the written permission from CEVA, in *
* accordance with the terms and conditions stipulated in the CEVA Agreement,
under    *
* which the information has been supplied and solely as expressly permitted for
*
* the purpose specified in the CEVA Agreement. *
* *
* This information is made available exclusively to licensees or parties that
have    *
* received express written authorization from CEVA to download or receive the *
* information and have agreed to the terms and conditions of the CEVA Agreement.
*
* *
* IF YOU HAVE NOT RECEIVED SUCH EXPRESS AUTHORIZATION AND AGREED TO THE CEVA *
* AGREEMENT, YOU MAY NOT DOWNLOAD, INSTALL OR USE THIS INFORMATION. *
* *
* The information contained in this document is subject to change without notice
*
* and does not represent a commitment on any part of CEVA. Unless specifically *
* agreed otherwise in the CEVA Agreement, CEVA make no warranty of any kind with
*
* regard to this material, including, but not limited to implied warranties of *
* merchantability and fitness for a particular purpose whether arising out of
law,    *
* custom, conduct or otherwise. *
* *
* While the information contained herein is assumed to be accurate, CEVA assumes
no   *
* responsibility for any errors or omissions contained herein, and assumes no *
* liability for special, direct, indirect or consequential damage, losses,
costs,     *
* charges, claims, demands, fees or expenses, of any nature or kind, which are *
* incurred in connection with the furnishing, performance or use of this
material.    *
* *
* This document contains proprietary information, which is protected by U.S. and
*
* international copyright laws. All rights reserved. No part of this document
may     *
* be reproduced, photocopied, or translated into another language without the
prior   *
* written consent of CEVA. *
\*************************************************************************************/
/*
  ===========================================================================
   File: BASOP32.H                                       v.2.1 - March 2006
  ===========================================================================

            ITU-T STL  BASIC OPERATORS

            GLOBAL FUNCTION PROTOTYPES

   History:
   26.Jan.00   v1.0     Incorporated to the STL from updated G.723.1/G.729
                        basic operator library (based on basic_op.h) and
                        G.723.1's basop.h.
   05.Jul.00    v1.1    Added 32-bit shiftless mult/mac/msub operators

   03 Nov 04   v2.0     Incorporation of new 32-bit / 40-bit / control
                        operators for the ITU-T Standard Tool Library as
                        described in Geneva, 20-30 January 2004 WP 3/16 Q10/16
                        TD 11 document and subsequent discussions on the
                        wp3audio@yahoogroups.com email reflector.
                        norm_s()      weight reduced from 15 to 1.
                        norm_l()      weight reduced from 30 to 1.
                        L_abs()       weight reduced from  2 to 1.
                        L_add()       weight reduced from  2 to 1.
                        L_negate()    weight reduced from  2 to 1.
                        L_shl()       weight reduced from  2 to 1.
                        L_shr()       weight reduced from  2 to 1.
                        L_sub()       weight reduced from  2 to 1.
                        mac_r()       weight reduced from  2 to 1.
                        msu_r()       weight reduced from  2 to 1.
                        mult_r()      weight reduced from  2 to 1.
                        L_deposit_h() weight reduced from  2 to 1.
                        L_deposit_l() weight reduced from  2 to 1.
   15 Nov 04   v2.0     L_mls() weight of 5.
                                                div_l() weight of 32.
                                                i_mult() weight of 3.

  ============================================================================
*/

#ifndef _CEVA_G729AB_BASIC_OP_H
#define _CEVA_G729AB_BASIC_OP_H
typedef int I32;
typedef short I16;
// typedef int		I32;

#ifndef MAX_32
#define MAX_32 (I32)0x7fffffffL
#endif

#ifndef MIN_32
#define MIN_32 (I32)0x80000000L
#endif

#ifdef CEVA_SensPro
static inline I32 L_mpy_ll_sat_rnd(I32 inp1, I32 inp2);

static I32 L_shl(I32 L_var1, I16 var2);
static I32 L_shr(I32 L_var1, I16 var2);

static I32 L_shr(I32 L_var1, I16 var2) {
  I32 L_var_out;

  if (var2 < 0) {
    if (var2 < -32) var2 = -32;
    var2 = -var2;
    L_var_out = L_shl(L_var1, var2);
  } else {
    if (var2 >= 31) {
      L_var_out = (L_var1 < 0L) ? -1 : 0;
    } else {
      if (L_var1 < 0) {
        L_var_out = ~((~L_var1) >> var2);
      } else {
        L_var_out = L_var1 >> var2;
      }
    }
  }
  return (L_var_out);
}
static I32 L_shl(I32 L_var1, I16 var2) {
  I32 L_var_out = 0L;

  if (var2 <= 0) {
    if (var2 < -32) var2 = -32;
    var2 = -var2;
    L_var_out = L_shr(L_var1, var2);
  } else {
    for (; var2 > 0; var2--) {
      if (L_var1 > (I32)0X3fffffffL) {
        //                Overflow = 1;
        L_var_out = MAX_32;
        break;
      } else {
        if (L_var1 < (I32)0xc0000000L) {
          //                    Overflow = 1;
          L_var_out = MIN_32;
          break;
        }
      }
      L_var1 *= 2;
      L_var_out = L_var1;
    }
  }
  return (L_var_out);
}
static inline I32 L_mpy_ll_sat_rnd(I32 inp1, I32 inp2) {
  I32 out;
  out = (I32)(((((long long)inp1 * inp2) + 0x040000000LL)) >> 31);
  if ((inp1 == 0x80000000) & (inp2 == 0x80000000)) out = 0x7fffffff;
  return (out);
}

#else
#ifndef WIN32
#include <vec-c.h>
#endif

///////////////////////////////////////////////////////////////////////

typedef short Word16;
typedef unsigned short UWord16;
typedef int Word32;
typedef unsigned UWord32;
typedef long long Word40;
typedef unsigned long long UWord40;
typedef int Flag;

#if defined __cevabx1__ || defined __cevabx2__
#define psl1 _psl1
#define sat _sat
#define psat _psat
#define rnd _rnd
#define sat16 _sat16
#define noflag _noflag
#define lo _lo
#define zero _zero
#endif /*def __cevabx1__ || def __cevabx2__*/

#if defined __cevaxc16__
#define psl1 _psl1
#define sat _sat
#define psat _psat
#define rnd _rnd
#define sat16 _sat16
#define noflag _noflag
#define lo _lo
#define zero _zero
#endif /*def __cevaxc16__*/
// For Bringup only
#if defined __cevaxm8__
#define psl1 _psl1
#define sat _sat
#define psat _psat
#define rnd _rnd
#define sat16 _sat16
#define noflag _noflag
#define lo _lo
#define zero _zero
#endif /*def __cevaxm8__*/

#ifdef __cevaxm6__
#define _extract extract
#define _shiftl shiftl
#define _shiftlui shiftlui
#define _shiftl shiftl
#define _mpyslimitslf mpyslimitslf
#define _limslimitslf limslimitslf
#define _negslimitslf negslimitslf
#define _negslimitslfs negslimitslfs
#define _shiftlslimitslf shiftlslimitslf
#define _subslimitslfs subslimitslfs
#define _subslimitslf subslimitslf
#define _shiftrslimitslf shiftrslimitslf
#define _macslimitslf macslimitslf
#define _msuslimitslf msuslimitslf
#define _addslimitslfs addslimitslfs
#define _addslimitslf addslimitslf
#define add _add
#define sub _sub
#endif

typedef long long I64;
typedef unsigned long long U64;

typedef short int16;
typedef unsigned int U32;
typedef unsigned short U16;

#ifdef WIN32
typedef struct {
  int lo;
  int hi;
} int2;
typedef struct {
  short lo;
  short hi;
} short2;

#endif

#ifndef MAX_8
#define MAX_8 (char)0x7f
#endif

#ifndef MIN_8
#define MIN_8 (char)0x80
#endif

#ifndef MAX_16
#define MAX_16 (int16)0x7fff
#endif

#ifndef MIN_16
#define MIN_16 (int16)0x8000
#endif

#define MAX_40 (Word40)0x7FFFFFFFFFLL
#define MIN_40 (Word40)0xFFFFFF8000000000LL

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

#ifdef WIN32
float FPMPY(float a, float b);
float FPMAC(float a, float b, float c);
float FPMSU(float a, float b, float c);
#endif

extern Flag get_overflow(void);
extern void clr_overflow(void);
extern void set_overflow(void);

extern Flag get_carry(void);
extern void clr_carry(void);
extern void set_carry(void);

#ifndef WIN32
static inline __attribute__((always_inline)) short2 min_short2(short2 inA,
                                                               short2 inB);
static inline __attribute__((always_inline)) short2 max_short2(short2 inA,
                                                               short2 inB);
static inline __attribute__((always_inline)) int lim_sat8(int inA);
static inline __attribute__((always_inline)) short2 sub_short2(short2 inA,
                                                               short2 inB);
static inline __attribute__((always_inline)) int2 sub_int2(int2 inA, int2 inB);
static inline __attribute__((always_inline)) int2 shiftl_int2(int2 inA,
                                                              int inB);
static inline __attribute__((always_inline)) int2 shiftr_int2_rnd(int2 inA,
                                                                  int inB);
static inline __attribute__((always_inline)) int2 mpy_int2_rnd_sat(int2 inA,
                                                                   int2 inB);
static inline __attribute__((always_inline)) int2 mac3_4_short2(
    short2 inA, short2 inB, short2 inC, short2 inD, int2 in_acc);
static inline __attribute__((always_inline)) int mac3_2_short2(short2 inA,
                                                               short2 inB,
                                                               int inC);
static inline __attribute__((always_inline)) int mac(short inA, short inB,
                                                     int inC);
static inline __attribute__((always_inline)) I32 L_mpy_ll_sat_rnd(I32 inp1,
                                                                  I32 inp2);

static inline __attribute__((always_inline)) short2 min_short2(short2 inA,
                                                               short2 inB) {
  return _min(inA, inB);
}
static inline __attribute__((always_inline)) short2 max_short2(short2 inA,
                                                               short2 inB) {
  return _max(inA, inB);
}
static inline __attribute__((always_inline)) int lim_sat8(int inA) {
  return _lim(_sat8, inA);
}
static inline __attribute__((always_inline)) short2 sub_short2(short2 inA,
                                                               short2 inB) {
  return _sub(inA, inB);
}
static inline __attribute__((always_inline)) int2 sub_int2(int2 inA, int2 inB) {
  return _sub(inA, inB);
}
static inline __attribute__((always_inline)) int2 shiftl_int2(int2 inA,
                                                              int inB) {
  return _shiftl(inA, inB);
}
static inline __attribute__((always_inline)) int2 shiftr_int2_rnd(int2 inA,
                                                                  int inB) {
  return _shiftr(_rnd, inA, inB);
}
static inline __attribute__((always_inline)) int2 mpy_int2_rnd_sat(int2 inA,
                                                                   int2 inB) {
  return _mpyps(_rnd, _sat, inA, inB, 31);
}
static inline __attribute__((always_inline)) int2 mac3_4_short2(
    short2 inA, short2 inB, short2 inC, short2 inD, int2 inE) {
  return _mac3(inA, inB, inC, inD, inE);
}
static inline __attribute__((always_inline)) int mac3_2_short2(short2 inA,
                                                               short2 inB,
                                                               int inE) {
  return _mac3(inA, inB, inE);
}
static inline __attribute__((always_inline)) int mac(short inA, short inB,
                                                     int inE) {
  return _mac(inA, inB, inE);
}
static inline __attribute__((always_inline)) I32 L_mpy_ll_sat_rnd(I32 inp1,
                                                                  I32 inp2) {
  return _mpyps(_rnd, _sat, inp1, inp2, 0x1f);
}

static inline __attribute__((always_inline)) Word32 L_shl(Word32 L_v1,
                                                          Word16 v2);
static inline __attribute__((always_inline)) Word32 L_shr(Word32 L_v1,
                                                          Word16 v2);

static inline __attribute__((always_inline)) Word32 L_shl(Word32 L_v1,
                                                          Word16 v2) {
  return _shiftl(sat, L_v1, v2);
}
static inline __attribute__((always_inline)) Word32 L_shr(Word32 L_v1,
                                                          Word16 v2) {
  return _shiftr(sat, L_v1, v2);
}

#else

static inline short2 min_short2(short2 inA, short2 inB);
static inline short2 max_short2(short2 inA, short2 inB);
static inline int lim_sat8(int inA);
static inline short2 sub_short2(short2 inA, short2 inB);
static inline int2 sub_int2(int2 inA, int2 inB);
static inline int2 shiftl_int2(int2 inA, int inB);
static inline int2 shiftr_int2_rnd(int2 inA, int inB);
static inline int2 mpy_int2_rnd_sat(int2 inA, int2 inB);
static inline int2 mac3_4_short2(short2 inA, short2 inB, short2 inC, short2 inD,
                                 int2 inE);
static inline int mac3_2_short2(short2 inA, short2 inB, int inE);
static inline int mac(short inA, short inB, int inE);
static inline I32 L_mpy_ll_sat_rnd(I32 inp1, I32 inp2);

static inline short2 min_short2(short2 inA, short2 inB) {
  short2 ret_val;
  ret_val.hi = ((inA.hi) < (inB.hi) ? (inA.hi) : (inB.hi));
  ret_val.lo = ((inA.lo) < (inB.lo) ? (inA.lo) : (inB.lo));
  return ret_val;
}
static inline short2 max_short2(short2 inA, short2 inB) {
  short2 ret_val;
  ret_val.hi = ((inA.hi) > (inB.hi) ? (inA.hi) : (inB.hi));
  ret_val.lo = ((inA.lo) > (inB.lo) ? (inA.lo) : (inB.lo));
  return ret_val;
}
static inline int lim_sat8(int inA) {
  int ret_val;
  ret_val = (int32_t)MIN_8 > inA ? (int32_t)MIN_8 : inA;
  ret_val = (int32_t)MAX_8 < ret_val ? (int32_t)MAX_8 : ret_val;
  return ret_val;
}
static inline short2 sub_short2(short2 inA, short2 inB) {
  short2 ret_val;
  ret_val.hi = inA.hi - inB.hi;
  ret_val.lo = inA.lo - inB.lo;
  return ret_val;
}
static inline int2 sub_int2(int2 inA, int2 inB) {
  int2 ret_val;
  ret_val.hi = inA.hi - inB.hi;
  ret_val.lo = inA.lo - inB.lo;
  return ret_val;
}
static inline int2 shiftl_int2(int2 inA, int inB) {
  int2 ret_val;
  ret_val.hi = inA.hi << inB;
  ret_val.lo = inA.lo << inB;
  return ret_val;
}
static inline int2 shiftr_int2_rnd(int2 inA, int inB) {
  int2 ret_val;
  const int mask = ((1ll << inB) - 1);
  const int zero = (0);
  const int one = (1);
  int2 remainder;
  remainder.lo = inA.lo & mask;
  remainder.hi = inA.hi & mask;
  int2 threshold;
  threshold.lo = (mask >> 1) + ((inA.lo < zero) & one);
  threshold.lo = (mask >> 1) + ((inA.hi < zero) & one);
  ret_val.lo = ((inA.lo >> inB) + ((remainder.lo > threshold.lo) & one));
  ret_val.hi = ((inA.hi >> inB) + ((remainder.hi > threshold.hi) & one));
  return ret_val;
}
#ifndef INT32_MIN
#define INT32_MIN (-2147483647 - 1)  // minimum (signed) int value
#endif
#ifndef INT32_MAX
#define INT32_MAX 2147483647  // maximum (signed) int value
#endif
static inline int2 mpy_int2_rnd_sat(int2 inA, int2 inB) {
  int2 ret_val;
  int64_t a_64 = (int64_t)inA.lo * inB.lo;
  int64_t b_64 = (int64_t)inA.hi * inB.hi;
  int32_t nudge = (1 << 30);
  ret_val.lo = (int32_t)((a_64 + nudge) >> 31);
  ret_val.hi = (int32_t)((b_64 + nudge) >> 31);

  if (ret_val.lo > INT32_MAX) {
    ret_val.lo = INT32_MAX;
  }
  if (ret_val.lo < INT32_MIN) {
    ret_val.lo = INT32_MIN;
  }
  if (ret_val.hi > INT32_MAX) {
    ret_val.hi = INT32_MAX;
  }
  if (ret_val.hi < INT32_MIN) {
    ret_val.hi = INT32_MIN;
  }

  return ret_val;
}
static inline int2 mac3_4_short2(short2 inA, short2 inB, short2 inC, short2 inD,
                                 int2 inE) {
  int2 out_acc;
  long long acc_lo = (long long)inE.lo + (long long)inA.lo * inB.lo +
                     (long long)inC.lo * inD.lo;
  long long acc_hi = (long long)inE.hi + (long long)inA.hi * inB.hi +
                     (long long)inC.hi * inD.hi;
  out_acc.lo = (int)acc_lo;
  out_acc.hi = (int)acc_hi;
  return out_acc;
}
static inline int mac3_2_short2(short2 inA, short2 inB, int inE) {
  return ((inA.lo * inB.lo) + (inA.hi * inB.hi) + inE);
}
static inline int mac(short inA, short inB, int inE) {
  return (inA * inB + inE);
}
static inline I32 L_mpy_ll_sat_rnd(I32 inp1, I32 inp2) {
  I32 out;
  out = (I32)(((((long long)inp1 * inp2) + 0x040000000LL)) >> 31);
  if ((inp1 == 0x80000000) & (inp2 == 0x80000000)) out = 0x7fffffff;
  return (out);
}

static I32 L_shl(I32 L_var1, I16 var2);
static I32 L_shr(I32 L_var1, I16 var2);

static I32 L_shr(I32 L_var1, I16 var2) {
  I32 L_var_out;

  if (var2 < 0) {
    if (var2 < -32) var2 = -32;
    var2 = -var2;
    L_var_out = L_shl(L_var1, var2);
  } else {
    if (var2 >= 31) {
      L_var_out = (L_var1 < 0L) ? -1 : 0;
    } else {
      if (L_var1 < 0) {
        L_var_out = ~((~L_var1) >> var2);
      } else {
        L_var_out = L_var1 >> var2;
      }
    }
  }
  return (L_var_out);
}
static I32 L_shl(I32 L_var1, I16 var2) {
  I32 L_var_out = 0L;

  if (var2 <= 0) {
    if (var2 < -32) var2 = -32;
    var2 = -var2;
    L_var_out = L_shr(L_var1, var2);
  } else {
    for (; var2 > 0; var2--) {
      if (L_var1 > (I32)0X3fffffffL) {
        //                Overflow = 1;
        L_var_out = MAX_32;
        break;
      } else {
        if (L_var1 < (I32)0xc0000000L) {
          //                    Overflow = 1;
          L_var_out = MIN_32;
          break;
        }
      }
      L_var1 *= 2;
      L_var_out = L_var1;
    }
  }
  return (L_var_out);
}

#endif

//////////////////////////////////////////
#if defined __cevabx1__ || defined __cevabx2__
#undef psl1
#undef sat
#undef psat
#undef rnd
#undef sat16
#undef noflag
#undef lo
#undef zero
#endif /*def __cevabx1__ || def __cevabx2__*/

#if defined __cevaxc16__
#undef psl1
#undef sat
#undef psat
#undef rnd
#undef sat16
#undef noflag
#undef lo
#undef zero
#endif /*def __cevaxc16__*/
// For Bringup only
#if defined __cevaxm8__
#undef psl1
#undef sat
#undef psat
#undef rnd
#undef sat16
#undef noflag
#undef lo
#undef zero
#endif /*def __cevaxm8__*/

#ifdef __cevaxm6__
#undef _extract
#undef _shiftl
#undef _shiftlui
#undef _shiftl
#undef _mpyslimitslf
#undef _limslimitslf
#undef _negslimitslf
#undef _negslimitslfs
#undef _shiftlslimitslf
#undef _subslimitslfs
#undef _subslimitslf
#undef _shiftrslimitslf
#undef _macslimitslf
#undef _msuslimitslf
#undef _addslimitslfs
#undef _addslimitslf
#endif

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif

//////////////////////////////////////////////////////////////////////
#endif /* ifndef _CEVA_G729AB_BASIC_OP_H */
