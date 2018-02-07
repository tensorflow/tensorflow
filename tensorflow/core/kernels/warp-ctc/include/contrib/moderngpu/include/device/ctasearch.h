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

#include "deviceutil.h"
#include "../mgpudevice.h"

namespace mgpu {

template<MgpuBounds Bounds, typename IntT, typename It, typename T,
	typename Comp>
MGPU_HOST_DEVICE void BinarySearchIt(It data, int& begin, int& end, T key,
	int shift, Comp comp) {

	IntT scale = (1<< shift) - 1;
	int mid = (int)((begin + scale * end)>> shift);

	T key2 = data[mid];
	bool pred = (MgpuBoundsUpper == Bounds) ?
		!comp(key, key2) :
		comp(key2, key);
	if(pred) begin = mid + 1;
	else end = mid;
}

template<MgpuBounds Bounds, typename IntT, typename T, typename It,
	typename Comp>
MGPU_HOST_DEVICE int BiasedBinarySearch(It data, int count, T key, int levels,
	Comp comp) {

	int begin = 0;
	int end = count;

	if(levels >= 4 && begin < end)
		BinarySearchIt<Bounds, IntT>(data, begin, end, key, 9, comp);
	if(levels >= 3 && begin < end)
		BinarySearchIt<Bounds, IntT>(data, begin, end, key, 7, comp);
	if(levels >= 2 && begin < end)
		BinarySearchIt<Bounds, IntT>(data, begin, end, key, 5, comp);
	if(levels >= 1 && begin < end)
		BinarySearchIt<Bounds, IntT>(data, begin, end, key, 4, comp);

	while(begin < end)
		BinarySearchIt<Bounds, int>(data, begin, end, key, 1, comp);
	return begin;
}

template<MgpuBounds Bounds, typename T, typename It, typename Comp>
MGPU_HOST_DEVICE int BinarySearch(It data, int count, T key, Comp comp) {
	int begin = 0;
	int end = count;
	while(begin < end)
		BinarySearchIt<Bounds, int>(data, begin, end, key, 1, comp);
	return begin;
}

////////////////////////////////////////////////////////////////////////////////
// MergePath search

template<MgpuBounds Bounds, typename It1, typename It2, typename Comp>
MGPU_HOST_DEVICE int MergePath(It1 a, int aCount, It2 b, int bCount, int diag,
	Comp comp) {

	typedef typename std::iterator_traits<It1>::value_type T;
	int begin = max(0, diag - bCount);
	int end = min(diag, aCount);

	while(begin < end) {
		int mid = (begin + end)>> 1;
		T aKey = a[mid];
		T bKey = b[diag - 1 - mid];
		bool pred = (MgpuBoundsUpper == Bounds) ?
			comp(aKey, bKey) :
			!comp(bKey, aKey);
		if(pred) begin = mid + 1;
		else end = mid;
	}
	return begin;
}


////////////////////////////////////////////////////////////////////////////////
// SegmentedMergePath search

template<typename InputIt, typename Comp>
MGPU_HOST_DEVICE int SegmentedMergePath(InputIt keys, int aOffset, int aCount,
	int bOffset, int bCount, int leftEnd, int rightStart, int diag, Comp comp) {

	// leftEnd and rightStart are defined from the origin, and diag is defined
	// from aOffset.
	// We only need to run a Merge Path search if the diagonal intersects the
	// segment that strides the left and right halves (i.e. is between leftEnd
	// and rightStart).
	if(aOffset + diag <= leftEnd) return diag;
	if(aOffset + diag >= rightStart) return aCount;

	bCount = min(bCount, rightStart - bOffset);
	int begin = max(max(leftEnd - aOffset, 0), diag - bCount);
	int end = min(diag, aCount);

	while(begin < end) {
		int mid = (begin + end)>> 1;
		int ai = aOffset + mid;
		int bi = bOffset + diag - 1 - mid;

		bool pred = !comp(keys[bi], keys[ai]);
		if(pred) begin = mid + 1;
		else end = mid;
	}
	return begin;
}

////////////////////////////////////////////////////////////////////////////////
// BalancedPath search

template<bool Duplicates, typename IntT, typename InputIt1, typename InputIt2,
	typename Comp>
MGPU_HOST_DEVICE int2 BalancedPath(InputIt1 a, int aCount, InputIt2 b,
	int bCount, int diag, int levels, Comp comp) {

	typedef typename std::iterator_traits<InputIt1>::value_type T;

	int p = MergePath<MgpuBoundsLower>(a, aCount, b, bCount, diag, comp);
	int aIndex = p;
	int bIndex = diag - p;

	bool star = false;
	if(bIndex < bCount) {
		if(Duplicates) {
			T x = b[bIndex];

			// Search for the beginning of the duplicate run in both A and B.
			// Because
			int aStart = BiasedBinarySearch<MgpuBoundsLower, IntT>(a, aIndex, x,
				levels, comp);
			int bStart = BiasedBinarySearch<MgpuBoundsLower, IntT>(b, bIndex, x,
				levels, comp);

			// The distance between the merge path and the lower_bound is the
			// 'run'. We add up the a- and b- runs and evenly distribute them to
			// get a stairstep path.
			int aRun = aIndex - aStart;
			int bRun = bIndex - bStart;
			int xCount = aRun + bRun;

			// Attempt to advance b and regress a.
			int bAdvance = max(xCount>> 1, bRun);
			int bEnd = min(bCount, bStart + bAdvance + 1);
			int bRunEnd = BinarySearch<MgpuBoundsUpper>(b + bIndex,
				bEnd - bIndex, x, comp) + bIndex;
			bRun = bRunEnd - bStart;

			bAdvance = min(bAdvance, bRun);
			int aAdvance = xCount - bAdvance;

			bool roundUp = (aAdvance == bAdvance + 1) && (bAdvance < bRun);
			aIndex = aStart + aAdvance;

			if(roundUp) star = true;
		} else {
			if(aIndex && aCount) {
				T aKey = a[aIndex - 1];
				T bKey = b[bIndex];

				// If the last consumed element in A (aIndex - 1) is the same as
				// the next element in B (bIndex), we're sitting at a starred
				// partition.
				if(!comp(aKey, bKey)) star = true;
			}
		}
	}
	return make_int2(aIndex, star);
}

} // namespace mgpu
