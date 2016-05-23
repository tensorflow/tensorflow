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

#include "ctasearch.cuh"
#include "loadstore.cuh"

namespace mgpu {

////////////////////////////////////////////////////////////////////////////////
// DeviceLoadBalancingSearch
// Upper Bound search from A (needles) into B (haystack). The A values are
// natural numbers from aBegin to aEnd. bFirst is the index of the B value at
// bBegin in shared memory.

template<int VT, bool RangeCheck>
MGPU_DEVICE void DeviceSerialLoadBalanceSearch(const int* b_shared, int aBegin,
	int aEnd, int bFirst, int bBegin, int bEnd, int* a_shared) {

	int bKey = b_shared[bBegin];

	#pragma unroll
	for(int i = 0; i < VT; ++i) {
		bool p;
		if(RangeCheck)
			p = (aBegin < aEnd) && ((bBegin >= bEnd) || (aBegin < bKey));
		else
			p = aBegin < bKey;

		if(p)
			// Advance A (the needle).
			a_shared[aBegin++] = bFirst + bBegin;
		else
			// Advance B (the haystack).
			bKey = b_shared[++bBegin];
	}
}

////////////////////////////////////////////////////////////////////////////////
// CTALoadBalance
// Computes upper_bound(counting_iterator<int>(first), b_global) - 1.

// Unlike most other CTA* functions, CTALoadBalance loads from global memory.
// This returns the loaded B elements at the beginning or end of shared memory
// depending on the aFirst argument.

// CTALoadBalance requires NT * VT + 2 slots of shared memory.
template<int NT, int VT, typename InputIt>
MGPU_DEVICE int4 CTALoadBalance(int destCount, InputIt b_global,
	int sourceCount, int block, int tid, const int* mp_global,
	int* indices_shared, bool loadPrecedingB) {

	int4 range = ComputeMergeRange(destCount, sourceCount, block, 0, NT * VT,
		mp_global);

	int a0 = range.x;
	int a1 = range.y;
	int b0 = range.z;
	int b1 = range.w;
	if(!b0) loadPrecedingB = false;

	// Load one trailing term from B. If we're already at the end, fill the
	// end of the buffer with destCount.
	int aCount = a1 - a0;
	int bCount = b1 - b0;
	int extended = b1 < sourceCount;
	int loadCount = bCount + extended;
	int fillCount = NT * VT + 1 - loadCount - aCount;

	int* a_shared = indices_shared;
	int* b_shared = indices_shared + aCount + (int)loadPrecedingB;

	// Load the B values.
//	DeviceMemToMemLoop<NT>(bCount + extended + (int)loadPrecedingB,
//		b_global + b0 - (int)loadPrecedingB, tid,
//		b_shared - (int)loadPrecedingB);

	for(int i = tid - (int)loadPrecedingB; i < bCount + extended; i += NT)
		b_shared[i] = b_global[b0 + i];

	// Fill the end of the array with destCount.
	for(int i = tid + extended; i < fillCount; i += NT)
		b_shared[bCount + i] = destCount;
	__syncthreads();

	// Run a merge path to find the start of the serial merge for each thread.
	int diag = VT * tid;
	int mp = MergePath<MgpuBoundsUpper>(mgpu::counting_iterator<int>(a0),
		aCount, b_shared, bCount, diag, mgpu::less<int>());

	int a0tid = a0 + mp;
	int b0tid = diag - mp;

	// Subtract 1 from b0 because we want to return upper_bound - 1.
	DeviceSerialLoadBalanceSearch<VT, false>(b_shared, a0tid, a1, b0 - 1,
		b0tid, bCount, a_shared - a0);
	__syncthreads();

	b0 -= (int)loadPrecedingB;
	return make_int4(a0, a1, b0, b1);
}


} // namespace mgpu
