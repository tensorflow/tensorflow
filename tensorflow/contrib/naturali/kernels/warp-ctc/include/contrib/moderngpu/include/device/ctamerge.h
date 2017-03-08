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

#include "ctasearch.h"
#include "loadstore.h"
#include "sortnetwork.h"

namespace mgpu {

////////////////////////////////////////////////////////////////////////////////
// SerialMerge

template<int VT, bool RangeCheck, typename T, typename Comp>
MGPU_DEVICE void SerialMerge(const T* keys_shared, int aBegin, int aEnd,
	int bBegin, int bEnd, T* results, int* indices, Comp comp) {

	T aKey = keys_shared[aBegin];
	T bKey = keys_shared[bBegin];

	#pragma unroll
	for(int i = 0; i < VT; ++i) {
		bool p;
		if(RangeCheck)
			p = (bBegin >= bEnd) || ((aBegin < aEnd) && !comp(bKey, aKey));
		else
			p = !comp(bKey, aKey);

		results[i] = p ? aKey : bKey;
		indices[i] = p ? aBegin : bBegin - !RangeCheck;

		if(p) aKey = keys_shared[++aBegin];
		else bKey = keys_shared[++bBegin];
	}
	__syncthreads();
}

////////////////////////////////////////////////////////////////////////////////
// FindMergeFrame and FindMergesortInterval help mergesort (both CTA and global
// merge pass levels) locate lists within the single source array.

// Returns (offset of a, offset of b, length of list).
MGPU_HOST_DEVICE int3 FindMergesortFrame(int coop, int block, int nv) {
	// coop is the number of CTAs or threads cooperating to merge two lists into
	// one. We round block down to the first CTA's ID that is working on this
	// merge.
	int start = ~(coop - 1) & block;
	int size = nv * (coop>> 1);
	return make_int3(nv * start, nv * start + size, size);
}

// Returns (a0, a1, b0, b1) into mergesort input lists between mp0 and mp1.
MGPU_HOST_DEVICE int4 FindMergesortInterval(int3 frame, int coop, int block,
	int nv, int count, int mp0, int mp1) {

	// Locate diag from the start of the A sublist.
	int diag = nv * block - frame.x;
	int a0 = frame.x + mp0;
	int a1 = min(count, frame.x + mp1);
	int b0 = min(count, frame.y + diag - mp0);
	int b1 = min(count, frame.y + diag + nv - mp1);

	// The end partition of the last block for each merge operation is computed
	// and stored as the begin partition for the subsequent merge. i.e. it is
	// the same partition but in the wrong coordinate system, so its 0 when it
	// should be listSize. Correct that by checking if this is the last block
	// in this merge operation.
	if(coop - 1 == ((coop - 1) & block)) {
		a1 = min(count, frame.x + frame.z);
		b1 = min(count, frame.y + frame.z);
	}
	return make_int4(a0, a1, b0, b1);
}

////////////////////////////////////////////////////////////////////////////////
// ComputeMergeRange

MGPU_HOST_DEVICE int4 ComputeMergeRange(int aCount, int bCount, int block,
	int coop, int NV, const int* mp_global) {

	// Load the merge paths computed by the partitioning kernel.
	int mp0 = mp_global[block];
	int mp1 = mp_global[block + 1];
	int gid = NV * block;

	// Compute the ranges of the sources in global memory.
	int4 range;
	if(coop) {
		int3 frame = FindMergesortFrame(coop, block, NV);
		range = FindMergesortInterval(frame, coop, block, NV, aCount, mp0,
			mp1);
	} else {
		range.x = mp0;											// a0
		range.y = mp1;											// a1
		range.z = gid - range.x;								// b0
		range.w = min(aCount + bCount, gid + NV) - range.y;		// b1
	}
	return range;
}

////////////////////////////////////////////////////////////////////////////////
// CTA mergesort support

template<int NT, int VT, typename T, typename Comp>
MGPU_DEVICE void CTABlocksortPass(T* keys_shared, int tid, int count,
	int coop, T* keys, int* indices, Comp comp) {

	int list = ~(coop - 1) & tid;
	int diag = min(count, VT * ((coop - 1) & tid));
	int start = VT * list;
	int a0 = min(count, start);
	int b0 = min(count, start + VT * (coop / 2));
	int b1 = min(count, start + VT * coop);

	int p = MergePath<MgpuBoundsLower>(keys_shared + a0, b0 - a0,
		keys_shared + b0, b1 - b0, diag, comp);

	SerialMerge<VT, true>(keys_shared, a0 + p, b0, b0 + diag - p, b1, keys,
		indices, comp);
}

template<int NT, int VT, bool HasValues, typename KeyType, typename ValType,
	typename Comp>
MGPU_DEVICE void CTABlocksortLoop(ValType threadValues[VT],
	KeyType* keys_shared, ValType* values_shared, int tid, int count,
	Comp comp) {

	#pragma unroll
	for(int coop = 2; coop <= NT; coop *= 2) {
		int indices[VT];
		KeyType keys[VT];
		CTABlocksortPass<NT, VT>(keys_shared, tid, count, coop, keys,
			indices, comp);

		if(HasValues) {
			// Exchange the values through shared memory.
			DeviceThreadToShared<VT>(threadValues, tid, values_shared);
			DeviceGather<NT, VT>(NT * VT, values_shared, indices, tid,
				threadValues);
		}

		// Store results in shared memory in sorted order.
		DeviceThreadToShared<VT>(keys, tid, keys_shared);
	}
}

////////////////////////////////////////////////////////////////////////////////
// CTAMergesort
// Caller provides the keys in shared memory. This functions sorts the first
// count elements.

template<int NT, int VT, bool Stable, bool HasValues, typename KeyType,
	typename ValType, typename Comp>
MGPU_DEVICE void CTAMergesort(KeyType threadKeys[VT], ValType threadValues[VT],
	KeyType* keys_shared, ValType* values_shared, int count, int tid,
	Comp comp) {

	// Stable sort the keys in the thread.
	if(VT * tid < count) {
		if(Stable)
			OddEvenTransposeSort<VT>(threadKeys, threadValues, comp);
		else
			OddEvenMergesort<VT>(threadKeys, threadValues, comp);
	}

	// Store the locally sorted keys into shared memory.
	DeviceThreadToShared<VT>(threadKeys, tid, keys_shared);

	// Recursively merge lists until the entire CTA is sorted.
	CTABlocksortLoop<NT, VT, HasValues>(threadValues, keys_shared,
		values_shared, tid, count, comp);
}

template<int NT, int VT, bool Stable, typename KeyType, typename Comp>
MGPU_DEVICE void CTAMergesortKeys(KeyType threadKeys[VT],
	KeyType* keys_shared, int count, int tid, Comp comp) {

	int valuesTemp[VT];
	CTAMergesort<NT, VT, Stable, false>(threadKeys, valuesTemp, keys_shared,
		(int*)keys_shared, count, tid, comp);
}

template<int NT, int VT, bool Stable, typename KeyType, typename ValType,
	typename Comp>
MGPU_DEVICE void CTAMergesortPairs(KeyType threadKeys[VT],
	ValType threadValues[VT], KeyType* keys_shared, ValType* values_shared,
	int count, int tid, Comp comp) {

	CTAMergesort<NT, VT, Stable, true>(threadKeys, threadValues, keys_shared,
		values_shared, count, tid, comp);
}

////////////////////////////////////////////////////////////////////////////////
// DeviceMergeKeysIndices

template<int NT, int VT, bool LoadExtended, typename It1, typename It2,
	typename T, typename Comp>
MGPU_DEVICE void DeviceMergeKeysIndices(It1 a_global, int aCount, It2 b_global,
	int bCount, int4 range, int tid, T* keys_shared, T* results, int* indices,
	Comp comp) {

	int a0 = range.x;
	int a1 = range.y;
	int b0 = range.z;
	int b1 = range.w;

	if(LoadExtended) {
		bool extended = (a1 < aCount) && (b1 < bCount);
		aCount = a1 - a0;
		bCount = b1 - b0;
		int aCount2 = aCount + (int)extended;
		int bCount2 = bCount + (int)extended;

		// Load one element past the end of each input to avoid having to use
		// range checking in the merge loop.
		DeviceLoad2ToShared<NT, VT, VT + 1>(a_global + a0, aCount2,
			b_global + b0, bCount2, tid, keys_shared);

		// Run a Merge Path search for each thread's starting point.
		int diag = VT * tid;
		int mp = MergePath<MgpuBoundsLower>(keys_shared, aCount,
			keys_shared + aCount2, bCount, diag, comp);

		// Compute the ranges of the sources in shared memory.
		int a0tid = mp;
		int b0tid = aCount2 + diag - mp;
		if(extended) {
			SerialMerge<VT, false>(keys_shared, a0tid, 0, b0tid, 0, results,
				indices, comp);
		} else {
			int a1tid = aCount;
			int b1tid = aCount2 + bCount;
			SerialMerge<VT, true>(keys_shared, a0tid, a1tid, b0tid, b1tid,
				results, indices, comp);
		}
	} else {
		// Use the input intervals from the ranges between the merge path
		// intersections.
		aCount = a1 - a0;
		bCount = b1 - b0;

		// Load the data into shared memory.
		DeviceLoad2ToShared<NT, VT, VT>(a_global + a0, aCount, b_global + b0,
			bCount, tid, keys_shared);

		// Run a merge path to find the start of the serial merge for each
		// thread.
		int diag = VT * tid;
		int mp = MergePath<MgpuBoundsLower>(keys_shared, aCount,
			keys_shared + aCount, bCount, diag, comp);

		// Compute the ranges of the sources in shared memory.
		int a0tid = mp;
		int a1tid = aCount;
		int b0tid = aCount + diag - mp;
		int b1tid = aCount + bCount;

		// Serial merge into register.
		SerialMerge<VT, true>(keys_shared, a0tid, a1tid, b0tid, b1tid, results,
			indices, comp);
	}
}

////////////////////////////////////////////////////////////////////////////////
// DeviceMerge
// Merge pairs from global memory into global memory. Useful factorization to
// enable calling from merge, mergesort, and locality sort.

template<int NT, int VT, bool HasValues, bool LoadExtended, typename KeysIt1,
	typename KeysIt2, typename KeysIt3, typename ValsIt1, typename ValsIt2,
	typename KeyType, typename ValsIt3, typename Comp>
MGPU_DEVICE void DeviceMerge(KeysIt1 aKeys_global, ValsIt1 aVals_global,
	int aCount, KeysIt2 bKeys_global, ValsIt2 bVals_global, int bCount,
	int tid, int block, int4 range, KeyType* keys_shared, int* indices_shared,
	KeysIt3 keys_global, ValsIt3 vals_global, Comp comp) {

	KeyType results[VT];
	int indices[VT];
	DeviceMergeKeysIndices<NT, VT, LoadExtended>(aKeys_global, aCount,
		bKeys_global, bCount, range, tid, keys_shared, results, indices, comp);

	// Store merge results back to shared memory.
	DeviceThreadToShared<VT>(results, tid, keys_shared);

	// Store merged keys to global memory.
	aCount = range.y - range.x;
	bCount = range.w - range.z;
	DeviceSharedToGlobal<NT, VT>(aCount + bCount, keys_shared, tid,
		keys_global + NT * VT * block);

	// Copy the values.
	if(HasValues) {
		DeviceThreadToShared<VT>(indices, tid, indices_shared);

		DeviceTransferMergeValuesShared<NT, VT>(aCount + bCount,
			aVals_global + range.x, bVals_global + range.z, aCount,
			indices_shared, tid, vals_global + NT * VT * block);
	}
}

} // namespace mgpu
