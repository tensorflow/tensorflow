/******************************************************************************
 * Copyright (c) 2013, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/******************************************************************************
 *
 * Code and text by Sean Baxter, NVIDIA Research
 * See http://nvlabs.github.io/moderngpu for repository and documentation.
 *
 ******************************************************************************/

#pragma once

#include <functional>
#include <iterator>
#include <cfloat>
#include <typeinfo>
#include <vector>
#include <list>
#include <map>
#include <algorithm>
#include <cassert>
#include <memory>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#ifndef MGPU_MIN
#define MGPU_MIN(x, y) (((x) <= (y)) ? (x) : (y))
#define MGPU_MAX(x, y) (((x) >= (y)) ? (x) : (y))
#define MGPU_MAX0(x) (((x) >= 0) ? (x) : 0)
#define MGPU_ABS(x) (((x) >= 0) ? (x) : (-x))

#define MGPU_DIV_UP(x, y) (((x) + (y) - 1) / (y))
#define MGPU_DIV_ROUND(x, y) (((x) + (y) / 2) / (y))
#define MGPU_ROUND_UP(x, y) ((y) * MGPU_DIV_UP(x, y))
#define MGPU_SHIFT_DIV_UP(x, y) (((x) + ((1<< (y)) - 1))>> y)
#define MGPU_ROUND_UP_POW2(x, y) (((x) + (y) - 1) & ~((y) - 1))
#define MGPU_ROUND_DOWN_POW2(x, y) ((x) & ~((y) - 1))
#define MGPU_IS_POW_2(x) (0 == ((x) & ((x) - 1)))

#endif // MGPU_MIN

namespace mgpu {


typedef unsigned char byte;

typedef unsigned int uint;
typedef signed short int16;

typedef unsigned short ushort;
typedef unsigned short uint16;

typedef long long int64;
typedef unsigned long long uint64;

// IsPow2<X>::value is true if X is a power of 2.
template<int X> struct sIsPow2 {
	enum { value = 0 == (X & (X - 1)) };
};

// Finds the base-2 logarithm of X. value is -1 if X is not a power of 2.
template<int X, bool roundUp = true> struct sLogPow2 { 
	enum { extra = sIsPow2<X>::value ? 0 : (roundUp ? 1 : 0) };
	enum { inner = sLogPow2<X / 2>::inner + 1 };
	enum { value = inner + extra };
};
template<bool roundUp> struct sLogPow2<0, roundUp> {
	enum { inner = 0 };
	enum { value = 0 };
};
template<bool roundUp> struct sLogPow2<1, roundUp> { 
	enum { inner = 0 };
	enum { value = 0 };
};

template<int X, int Y>
struct sDivUp {
	enum { value = (X + Y - 1) / Y };
};

template<int count, int levels> struct sDiv2RoundUp {
	enum { value = sDiv2RoundUp<sDivUp<count, 2>::value, levels - 1>::value };
};
template<int count> struct sDiv2RoundUp<count, 0> {
	enum { value = count };
};

template<int X, int Y>
struct sDivSafe {
	enum { value = X / Y };
};
template<int X>
struct sDivSafe<X, 0> {
	enum { value = 0 };
};

template<int X, int Y>
struct sRoundUp {
	enum { rem = X % Y };
	enum { value = X + (rem ? (Y - rem) : 0) };
};

template<int X, int Y>
struct sRoundDown {
	enum { rem = X % Y };
	enum { value = X - rem };
};

// IntegerDiv is a template for avoiding divisions by zero in template 
// evaluation. Templates always evaluate both b and c in an expression like
// a ? b : c, and will error if either rhs contains an illegal expression,
// even if the ternary is explictly designed to guard against that.
template<int X, int Y>
struct sIntegerDiv {
	enum { value = X / (Y ? Y : (X + 1)) };
};

template<int X, int Y>
struct sMax {
	enum { value = (X >= Y) ? X : Y };
};
template<int X, int Y>
struct sMin {
	enum { value = (X <= Y) ? X : Y };
};

template<int X>
struct sAbs {
	enum { value = (X >= 0) ? X : -X };
};


// Finds the number of powers of 2 in the prime factorization of X.
template<int X, int LSB = 1 & X> struct sNumFactorsOf2 {
	enum { shifted = X >> 1 };
	enum { value = 1 + sNumFactorsOf2<shifted>::value };
};
template<int X> struct sNumFactorsOf2<X, 1> {
	enum { value = 0 };
};

// Returns the divisor for a conflict-free transpose.
template<int X, int NumBanks = 32> struct sBankConflictDivisor {
	enum { value = 
		(1 & X) ? 0 : 
		(sIsPow2<X>::value ? NumBanks :
		(1<< sNumFactorsOf2<X>::value)) }; 
	enum { log_value = sLogPow2<value>::value };
};

template<int NT, int X, int NumBanks = 32> struct sConflictFreeStorage {
	enum { count = NT * X };
	enum { divisor = sBankConflictDivisor<X, NumBanks>::value };
	enum { padding = sDivSafe<count, divisor>::value };
	enum { value = count + padding };
};

} // namespace mgpu
