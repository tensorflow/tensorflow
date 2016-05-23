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

#include "deviceutil.cuh"

namespace mgpu {

////////////////////////////////////////////////////////////////////////////////
// Odd-even transposition sorting network. Sorts keys and values in-place in
// register.
// http://en.wikipedia.org/wiki/Odd%E2%80%93even_sort

// CUDA Compiler does not currently unroll these loops correctly. Write using
// template loop unrolling.
/*
template<int VT, typename T, typename V, typename Comp>
MGPU_DEVICE void OddEvenTransposeSort(T* keys, V* values, Comp comp) {
	#pragma unroll
	for(int level = 0; level < VT; ++level) {

		#pragma unroll
		for(int i = 1 & level; i < VT - 1; i += 2) {
			if(comp(keys[i + 1], keys[i])) {
				mgpu::swap(keys[i], keys[i + 1]);
				mgpu::swap(values[i], values[i + 1]);
			}
		}
	}
}*/

template<int I, int VT>
struct OddEvenTransposeSortT {
	// Sort segments marked by head flags. If the head flag between i and i + 1
	// is set (so that (2<< i) & flags is true), the values belong to different
	// segments and are not swapped.
	template<typename K, typename V, typename Comp>
	static MGPU_DEVICE void Sort(K* keys, V* values, int flags, Comp comp) {
		#pragma unroll
		for(int i = 1 & I; i < VT - 1; i += 2)
			if((0 == ((2<< i) & flags)) && comp(keys[i + 1], keys[i])) {
				mgpu::swap(keys[i], keys[i + 1]);
				mgpu::swap(values[i], values[i + 1]);
			}
		OddEvenTransposeSortT<I + 1, VT>::Sort(keys, values, flags, comp);
	}
};
template<int I> struct OddEvenTransposeSortT<I, I> {
	template<typename K, typename V, typename Comp>
	static MGPU_DEVICE void Sort(K* keys, V* values, int flags, Comp comp) { }
};

template<int VT, typename K, typename V, typename Comp>
MGPU_DEVICE void OddEvenTransposeSort(K* keys, V* values, Comp comp) {
	OddEvenTransposeSortT<0, VT>::Sort(keys, values, 0, comp);
}
template<int VT, typename K, typename V, typename Comp>
MGPU_DEVICE void OddEvenTransposeSortFlags(K* keys, V* values, int flags,
	Comp comp) {
	OddEvenTransposeSortT<0, VT>::Sort(keys, values, flags, comp);
}

////////////////////////////////////////////////////////////////////////////////
// Batcher Odd-Even Mergesort network
// Unstable but executes much faster than the transposition sort.
// http://en.wikipedia.org/wiki/Batcher_odd%E2%80%93even_mergesort

template<int Width, int Low, int Count>
struct OddEvenMergesortT {
	template<typename K, typename V, typename Comp>
	MGPU_DEVICE static void CompareAndSwap(K* keys, V* values, int flags,
		int a, int b, Comp comp) {
		if(b < Count) {
			// Mask the bits between a and b. Any head flags in this interval
			// means the keys are in different segments and must not be swapped.
			const int Mask = ((2<< b) - 1) ^ ((2<< a) - 1);
			if(!(Mask & flags) && comp(keys[b], keys[a])) {
				mgpu::swap(keys[b], keys[a]);
				mgpu::swap(values[b], values[a]);
			}
		}
	}

	template<int R, int Low2, bool Recurse = 2 * R < Width>
	struct OddEvenMerge {
		template<typename K, typename V, typename Comp>
		MGPU_DEVICE static void Merge(K* keys, V* values, int flags,
			Comp comp) {
			// Compare and swap
			const int M = 2 * R;
			OddEvenMerge<M, Low2>::Merge(keys, values, flags, comp);
			OddEvenMerge<M, Low2 + R>::Merge(keys, values, flags, comp);

			#pragma unroll
			for(int i = Low2 + R; i + R < Low2 + Width; i += M)
				CompareAndSwap(keys, values, flags, i, i + R, comp);
		}
	};
	template<int R, int Low2>
	struct OddEvenMerge<R, Low2, false> {
		template<typename K, typename V, typename Comp>
		MGPU_DEVICE static void Merge(K* keys, V* values, int flags,
			Comp comp) {
			CompareAndSwap(keys, values, flags, Low2, Low2 + R, comp);
		}
	};

	template<typename K, typename V, typename Comp>
	MGPU_DEVICE static void Sort(K* keys, V* values, int flags,
		Comp comp) {

		const int M = Width / 2;
		OddEvenMergesortT<M, Low, Count>::Sort(keys, values, flags, comp);
		OddEvenMergesortT<M, Low + M, Count>::Sort(keys, values, flags, comp);
		OddEvenMerge<1, Low>::Merge(keys, values, flags, comp);
	}
};
template<int Low, int Count> struct OddEvenMergesortT<1, Low, Count> {
	template<typename K, typename V, typename Comp>
	MGPU_DEVICE static void Sort(K* keys, V* values, int flags,
		Comp comp) { }
};

template<int VT, typename K, typename V, typename Comp>
MGPU_DEVICE void OddEvenMergesort(K* keys, V* values, Comp comp) {
	const int Width = 1<< sLogPow2<VT, true>::value;
	OddEvenMergesortT<Width, 0, VT>::Sort(keys, values, 0, comp);
}
template<int VT, typename K, typename V, typename Comp>
MGPU_DEVICE void OddEvenMergesortFlags(K* keys, V* values, int flags,
	Comp comp) {
	const int Width = 1<< sLogPow2<VT, true>::value;
	OddEvenMergesortT<Width, 0, VT>::Sort(keys, values, flags, comp);
}

} // namespace mgpu
