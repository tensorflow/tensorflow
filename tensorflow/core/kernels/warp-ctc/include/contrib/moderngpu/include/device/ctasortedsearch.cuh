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

#include "../mgpudevice.cuh"
#include "ctasearch.cuh"

namespace mgpu {


////////////////////////////////////////////////////////////////////////////////
// DeviceSerialSearch

template<int VT, MgpuBounds Bounds, bool RangeCheck, bool IndexA, bool MatchA,
	bool IndexB, bool MatchB, typename T, typename Comp>
MGPU_DEVICE int3 DeviceSerialSearch(const T* keys_shared, int aBegin,
	int aEnd, int bBegin, int bEnd, int aOffset, int bOffset, int* indices,
	Comp comp) {

	const int FlagA = IndexA ? 0x80000000 : 1;
	const int FlagB = IndexB ? 0x80000000 : 1;

	T aKey = keys_shared[aBegin];
	T bKey = keys_shared[bBegin];
	T aPrev, bPrev;
	if(aBegin > 0) aPrev = keys_shared[aBegin - 1];
	if(bBegin > 0) bPrev = keys_shared[bBegin - 1];
	int decisions = 0;
	int matchCountA = 0;
	int matchCountB = 0;

	#pragma unroll
	for(int i = 0; i < VT; ++i) {
		bool p;
		if(RangeCheck && aBegin >= aEnd) p = false;
		else if(RangeCheck && bBegin >= bEnd) p = true;
		else p = (MgpuBoundsUpper == Bounds) ?
			comp(aKey, bKey) :
			!comp(bKey, aKey);

		if(p) {
			// aKey is smaller than bKey, so it is inserted before bKey.
			// Save bKey's index (bBegin + first) as the result of the search
			// and advance to the next needle in A.
			bool match = false;
			if(MatchA) {
				// Test if there is an element in B that matches aKey.
				if(MgpuBoundsUpper == Bounds) {
					// Upper Bound: We're inserting aKey after bKey. If there
					// is a match for aKey it must be bPrev. Check that bPrev
					// is in range and equal to aKey.
					// The predicate test result !comp(aKey, bPrev) was
					// established on the previous A-advancing iteration (it
					// failed the comp(aKey, bKey) test to get us to this
					// point). Check the other half of the equality condition
					// with a second comparison.
					bool inRange = !RangeCheck || (bBegin > aEnd);
					match = inRange && !comp(bPrev, aKey);
				} else {
					// Lower Bound: We're inserting aKey before bKey. If there
					// is a match for aKey, it must be bKey. Check that bKey
					// is in range and equal to aKey.
					// The predicate test !comp(bKey, aKey) has established one
					// half of the equality condition. We establish the other
					// half with a second comparison.
					bool inRange = !RangeCheck || (bBegin < bEnd);
					match = inRange && !comp(aKey, bKey);
				}
			}

			int index = 0;
		 	if(IndexA) index = bOffset + bBegin;
			if(match) index |= FlagA;
			if(IndexA || MatchA) indices[i] = index;
			matchCountA += match;

			// Mark the decision bit to indicate that this iteration has
			// progressed A (the needles).
			decisions |= 1<< i;
			aPrev = aKey;
			aKey = keys_shared[++aBegin];
		} else {
			// aKey is larger than bKey, so it is inserted after bKey (but we
			// don't know where yet). Advance the B index to the next element in
			// the haystack to continue the search for the current needle.
			bool match = false;
			if(MatchB) {
				if(MgpuBoundsUpper == Bounds) {
					// Upper Bound: aKey is not smaller than bKey. We advance to
					// the next haystack element in B. If there is a match in A
					// for bKey it must be aKey. By entering this branch we've
					// verified that !comp(aKey, bKey). Making the reciprocal
					// comparison !comp(bKey, aKey) establishes aKey == bKey.
					bool inRange = !RangeCheck ||
						((bBegin < bEnd) && (aBegin < aEnd));
					match = inRange && !comp(bKey, aKey);
				} else {
					// Lower Bound: bKey is smaller than aKey. We advance to the
					// next element in B. If there is a match for bKey, it must
					// be aPrev. The previous A-advancing iteration proved that
					// !comp(bKey, aPrev). We test !comp(aPrev, bKey) for the
					// other half of the equality condition.
					bool inRange = !RangeCheck ||
						((bBegin < bEnd) && (aBegin > 0));
					match = inRange && !comp(aPrev, bKey);
				}
			}

			int index = 0;
			if(IndexB) index = aOffset + aBegin;
			if(match) index |= FlagB;
			if(IndexB || MatchB) indices[i] = index;
			matchCountB += match;

			// Keep the decision bit cleared to indicate that this iteration
			// has progressed B (the haystack).
			bPrev = bKey;
			bKey = keys_shared[++bBegin];
		}
	}
	return make_int3(decisions, matchCountA, matchCountB);
}

////////////////////////////////////////////////////////////////////////////////
// CTASortedSearch
// Take keys in shared memory and return indices and b-match flags in shared
// memory.
// NOTE: This function doesn't do any strided-to-thread order transposes so
// using an even number of values per thread will incur no additional bank
// conflicts.

template<int NT, int VT, MgpuBounds Bounds, bool IndexA, bool MatchA,
	bool IndexB, bool MatchB, typename T, typename Comp>
MGPU_DEVICE int2 CTASortedSearch(T* keys_shared, int aStart, int aCount,
	int aEnd, int a0, int bStart, int bCount, int bEnd, int b0, bool extended,
	int tid, int* indices_shared, Comp comp) {

	// Run a merge path to find the start of the serial search for each thread.
	int diag = VT * tid;
	int mp = MergePath<Bounds>(keys_shared + aStart, aCount,
		keys_shared + bStart, bCount, diag, comp);
	int a0tid = mp;
	int b0tid = diag - mp;

	// Serial search into register.
	int3 results;
	int indices[VT];
	if(extended)
		results = DeviceSerialSearch<VT, Bounds, false, IndexA, MatchA, IndexB,
			MatchB>(keys_shared, a0tid + aStart, aEnd, b0tid + bStart, bEnd,
			a0 - aStart, b0 - bStart, indices, comp);
	else
		results = DeviceSerialSearch<VT, Bounds, true, IndexA, MatchA, IndexB,
			MatchB>(keys_shared, a0tid + aStart, aEnd, b0tid + bStart, bEnd,
			a0 - aStart, b0 - bStart, indices, comp);
	__syncthreads();

	// Compact the indices into shared memory. Use the decision bits (set is A,
	// cleared is B) to select the destination.
	int decisions = results.x;
	b0tid += aCount;
	#pragma unroll
	for(int i = 0; i < VT; ++i) {
		if((1<< i) & decisions) {
			if(IndexA || MatchA) indices_shared[a0tid++] = indices[i];
		} else {
			if(IndexB || MatchB) indices_shared[b0tid++] = indices[i];
		}
	}
	__syncthreads();

	// Return the match counts for A and B keys.
	return make_int2(results.y, results.z);
}

} // namespace mgpu
