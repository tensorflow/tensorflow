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

#if __CUDA_ARCH__ == 100
	#error "COMPUTE CAPABILITY 1.0 NOT SUPPORTED BY MPGU. TRY 2.0!"
#endif 

#include <climits>
#include "../util/static.h"

#ifdef _MSC_VER
#define INLINESYMBOL __forceinline__
#else
#define INLINESYMBOL inline
#endif

namespace mgpu {

#define MGPU_HOST __host__ INLINESYMBOL
#define MGPU_DEVICE __device__ INLINESYMBOL
#define MGPU_HOST_DEVICE __host__ __device__ INLINESYMBOL

const int WARP_SIZE = 32;
const int LOG_WARP_SIZE = 5;

////////////////////////////////////////////////////////////////////////////////
// Device-side comparison operators

template<typename T>
struct less : public std::binary_function<T, T, bool> {
	MGPU_HOST_DEVICE bool operator()(T a, T b) { return a < b; }
};
template<typename T>
struct less_equal : public std::binary_function<T, T, bool> {
	MGPU_HOST_DEVICE bool operator()(T a, T b) { return a <= b; }
};
template<typename T>
struct greater : public std::binary_function<T, T, bool> {
	MGPU_HOST_DEVICE bool operator()(T a, T b) { return a > b; }
};
template<typename T>
struct greater_equal : public std::binary_function<T, T, bool> {
	MGPU_HOST_DEVICE bool operator()(T a, T b) { return a >= b; }
};
template<typename T>
struct equal_to : public std::binary_function<T, T, bool> {
	MGPU_HOST_DEVICE bool operator()(T a, T b) { return a == b; }
};
template<typename T>
struct not_equal_to : public std::binary_function<T, T, bool> {
	MGPU_HOST_DEVICE bool operator()(T a, T b) { return a != b; }
};

////////////////////////////////////////////////////////////////////////////////
// Device-side arithmetic operators

template<typename T>
struct plus : public std::binary_function<T, T, T> {
	MGPU_HOST_DEVICE T operator()(T a, T b) { return a + b; }
};

template<typename T>
struct minus : public std::binary_function<T, T, T> {
	MGPU_HOST_DEVICE T operator()(T a, T b) { return a - b; }
};

template<typename T>
struct multiplies : public std::binary_function<T, T, T> {
	MGPU_HOST_DEVICE T operator()(T a, T b) { return a * b; }
};

template<typename T>
struct modulus : public std::binary_function<T, T, T> {
	MGPU_HOST_DEVICE T operator()(T a, T b) { return a % b; }
};

template<typename T>
struct bit_or : public std::binary_function<T, T, T> {
	MGPU_HOST_DEVICE T operator()(T a, T b) { return a | b; }
};

template<typename T>
struct bit_and : public std::binary_function<T, T, T> {
	MGPU_HOST_DEVICE T operator()(T a, T b) { return a & b; }
};

template<typename T>
struct bit_xor : public std::binary_function<T, T, T> {
	MGPU_HOST_DEVICE T operator()(T a, T b) { return a ^ b; }
};

template<typename T>
struct maximum : public std::binary_function<T, T, T> {
	MGPU_HOST_DEVICE T operator()(T a, T b) { return max(a, b); }
};

template<typename T>
struct minimum : public std::binary_function<T, T, T> {
	MGPU_HOST_DEVICE T operator()(T a, T b) { return min(a, b); }
};

////////////////////////////////////////////////////////////////////////////////

template<typename T>
MGPU_HOST_DEVICE void swap(T& a, T& b) {
	T c = a;
	a = b;
	b = c;
}

template<typename T>
struct DevicePair {
	T x, y;
};

template<typename T>
MGPU_HOST_DEVICE DevicePair<T> MakeDevicePair(T x, T y) {
	DevicePair<T> p = { x, y };
	return p;
}

template<typename T> struct numeric_limits;
template<> struct numeric_limits<int> {
	MGPU_HOST_DEVICE static int min() { return INT_MIN; }
	MGPU_HOST_DEVICE static int max() { return INT_MAX; }
	MGPU_HOST_DEVICE static int lowest() { return INT_MIN; }
	MGPU_HOST_DEVICE static int AddIdent() { return 0; }
	MGPU_HOST_DEVICE static int MulIdent() { return 1; }
};
template<> struct numeric_limits<long long> {
	MGPU_HOST_DEVICE static long long min() { return LLONG_MIN; }
	MGPU_HOST_DEVICE static long long max() { return LLONG_MAX; }
	MGPU_HOST_DEVICE static long long lowest() { return LLONG_MIN; }
	MGPU_HOST_DEVICE static long long AddIdent() { return 0; }
	MGPU_HOST_DEVICE static long long MulIdent() { return 1; }
};
template<> struct numeric_limits<uint> {
	MGPU_HOST_DEVICE static uint min() { return 0; }
	MGPU_HOST_DEVICE static uint max() { return UINT_MAX; }
	MGPU_HOST_DEVICE static uint lowest() { return 0; }
	MGPU_HOST_DEVICE static uint AddIdent() { return 0; }
	MGPU_HOST_DEVICE static uint MulIdent() { return 1; }
};
template<> struct numeric_limits<unsigned long long> {
	MGPU_HOST_DEVICE static unsigned long long min() { return 0; }
	MGPU_HOST_DEVICE static unsigned long long max() { return ULLONG_MAX; }
	MGPU_HOST_DEVICE static unsigned long long lowest() { return 0; }
	MGPU_HOST_DEVICE static unsigned long long AddIdent() { return 0; }
	MGPU_HOST_DEVICE static unsigned long long MulIdent() { return 1; }
};
template<> struct numeric_limits<float> {
	MGPU_HOST_DEVICE static float min() { return FLT_MIN; }
	MGPU_HOST_DEVICE static float max() { return FLT_MAX; }
	MGPU_HOST_DEVICE static float lowest() { return -FLT_MAX; }
	MGPU_HOST_DEVICE static float AddIdent() { return 0; }
	MGPU_HOST_DEVICE static float MulIdent() { return 1; }
};
template<> struct numeric_limits<double> {
	MGPU_HOST_DEVICE static double min() { return DBL_MIN; }
	MGPU_HOST_DEVICE static double max() { return DBL_MAX; }
	MGPU_HOST_DEVICE static double lowest() { return -DBL_MAX; }
	MGPU_HOST_DEVICE static double AddIdent() { return 0; }
	MGPU_HOST_DEVICE static double MulIdent() { return 1; }
};


MGPU_HOST_DEVICE int2 operator+(int2 a, int2 b) {
	return make_int2(a.x + b.x, a.y + b.y); 
}
MGPU_HOST_DEVICE int2& operator+=(int2& a, int2 b) {
	a = a + b;
	return a;
}
MGPU_HOST_DEVICE int2 operator*(int2 a, int2 b) {
	return make_int2(a.x * b.x, a.y * b.y);
}
MGPU_HOST_DEVICE int2& operator*=(int2& a, int2 b) {
	a = a * b;
	return a;
}

template<typename T>
MGPU_HOST_DEVICE T max(T a, T b) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 100)
	return std::max(a, b);
#else
	return (a < b) ? b : a;
#endif
}
template<typename T>
MGPU_HOST_DEVICE T min(T a, T b) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 100)
	return std::min(a, b);
#else
	return (b < a) ? b : a;
#endif
}

MGPU_HOST_DEVICE int2 max(int2 a, int2 b) {
	return make_int2(max(a.x, b.x), max(a.y, b.y));
}

MGPU_HOST_DEVICE int2 min(int2 a, int2 b) {
	return make_int2(min(a.x, b.x), min(a.y, b.y));
}

template<> struct numeric_limits<int2> {
	MGPU_HOST_DEVICE static int2 min() { return make_int2(INT_MIN, INT_MIN); }
	MGPU_HOST_DEVICE static int2 max() { return make_int2(INT_MAX, INT_MAX); }
	MGPU_HOST_DEVICE static int2 lowest() { 
		return make_int2(INT_MIN, INT_MIN); 
	}
	MGPU_HOST_DEVICE static int2 AddIdent() { return make_int2(0, 0); }
	MGPU_HOST_DEVICE static int2 MulIdent() { return make_int2(1, 1); }
};

template<typename T>
class constant_iterator : public std::iterator_traits<const T*> {
public:
	MGPU_HOST_DEVICE constant_iterator(T value) : _value(value) { }

	MGPU_HOST_DEVICE T operator[](ptrdiff_t i) const { 
		return _value;
	}
	MGPU_HOST_DEVICE T operator*() const {
		return _value;
	}
	MGPU_HOST_DEVICE constant_iterator operator+(ptrdiff_t diff) const {
		return constant_iterator(_value);
	}
	MGPU_HOST_DEVICE constant_iterator operator-(ptrdiff_t diff) const {
		return constant_iterator(_value);
	}
	MGPU_HOST_DEVICE constant_iterator& operator+=(ptrdiff_t diff) {
		return *this;
	}
	MGPU_HOST_DEVICE constant_iterator& operator-=(ptrdiff_t diff) {
		return *this;
	}
private:
	T _value;
};

template<typename T>
class counting_iterator : public std::iterator_traits<const T*> {
public:
	MGPU_HOST_DEVICE counting_iterator(T value) : _value(value) { }

	MGPU_HOST_DEVICE T operator[](ptrdiff_t i) { 
		return _value + i;
	}
	MGPU_HOST_DEVICE T operator*() {
		return _value;
	}
	MGPU_HOST_DEVICE counting_iterator operator+(ptrdiff_t diff) {
		return counting_iterator(_value + diff);
	}
	MGPU_HOST_DEVICE counting_iterator operator-(ptrdiff_t diff) {
		return counting_iterator(_value - diff);
	}
	MGPU_HOST_DEVICE counting_iterator& operator+=(ptrdiff_t diff) {
		_value += diff;
		return *this;
	}
	MGPU_HOST_DEVICE counting_iterator& operator-=(ptrdiff_t diff) {
		_value -= diff;
		return *this;
	}
private:
	T _value;
};

template<typename T>
class step_iterator : public std::iterator_traits<const T*> {
public:
	MGPU_HOST_DEVICE step_iterator(T base, T step) :
		_base(base), _step(step), _offset(0) { }

	MGPU_HOST_DEVICE T operator[](ptrdiff_t i) { 
		return _base + (_offset + i) * _step; 
	}
	MGPU_HOST_DEVICE T operator*() { 
		return _base + _offset * _step; 
	} 
	MGPU_HOST_DEVICE step_iterator operator+(ptrdiff_t diff) {
		step_iterator it = *this;
		it._offset += diff;
		return it;
	}
	MGPU_HOST_DEVICE step_iterator operator-(ptrdiff_t diff) {
		step_iterator it = *this;
		it._offset -= diff;
		return it;
	}
	MGPU_HOST_DEVICE step_iterator& operator+=(ptrdiff_t diff) { 
		_offset += diff;
		return *this;
	}
	MGPU_HOST_DEVICE step_iterator& operator-=(ptrdiff_t diff) { 
		_offset -= diff;
		return *this;
	}
private:
	ptrdiff_t _offset;
	T _base, _step;	
};

} // namespace mgpu


template<typename T>
MGPU_HOST_DEVICE mgpu::counting_iterator<T> operator+(ptrdiff_t diff,
	mgpu::counting_iterator<T> it) {
	return it + diff;
}
template<typename T>
MGPU_HOST_DEVICE mgpu::counting_iterator<T> operator-(ptrdiff_t diff,
	mgpu::counting_iterator<T> it) {
	return it + (-diff);
}
template<typename T>
MGPU_HOST_DEVICE mgpu::step_iterator<T> operator+(ptrdiff_t diff, 
	mgpu::step_iterator<T> it) {
	return it + diff;
}
template<typename T>
MGPU_HOST_DEVICE mgpu::step_iterator<T> operator-(ptrdiff_t diff, 
	mgpu::step_iterator<T> it) {
	return it + (-diff);
}
