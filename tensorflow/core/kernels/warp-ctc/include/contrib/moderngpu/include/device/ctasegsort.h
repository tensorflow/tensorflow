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

#include "ctascan.h"
#include "ctasearch.h"
#include "loadstore.h"
#include "sortnetwork.h"

namespace mgpu {

template<int VT, typename T, typename Comp>
MGPU_DEVICE void SegmentedSerialMerge(const T* keys_shared, int aBegin,
	int aEnd, int bBegin, int bEnd, T results[VT], int indices[VT],
	int leftEnd, int rightStart, Comp comp, bool sync = true) {

	bEnd = min(rightStart, bEnd);
	T aKey = keys_shared[aBegin];
	T bKey = keys_shared[bBegin];

	#pragma unroll
	for(int i = 0; i < VT; ++i) {
		bool p;

		// If A has run out of inputs, emit B.
		if(aBegin >= aEnd)
			p = false;
		else if(bBegin >= bEnd || aBegin < leftEnd)
			// B has hit the end of the middle segment.
			// Emit A if A has inputs remaining in the middle segment.
			p = true;
		else
			// Emit the smaller element in the middle segment.
			p = !comp(bKey, aKey);

		results[i] = p ? aKey : bKey;
		indices[i] = p ? aBegin : bBegin;
		if(p) aKey = keys_shared[++aBegin];
		else bKey = keys_shared[++bBegin];
	}
	if(sync) { __syncthreads(); }
}

////////////////////////////////////////////////////////////////////////////////
// CTASegsortPass

template<int NT, int VT, typename T, typename Comp>
MGPU_DEVICE void CTASegsortPass(T* keys_shared, int* ranges_shared, int tid,
	int pass, T results[VT], int indices[VT], int2& activeRange, Comp comp) {

	// Locate the intervals of the input lists.
	int3 frame = FindMergesortFrame(2<< pass, tid, VT);
	int a0 = frame.x;
	int b0 = frame.y;
	int listLen = frame.z;
	int list = tid>> pass;
	int listParity = 1 & list;
	int diag = VT * tid - frame.x;

	// Fetch the active range for the list this thread's list is merging with.
	int siblingRange = ranges_shared[1 ^ list];
	int siblingStart = 0x0000ffff & siblingRange;
	int siblingEnd = siblingRange>> 16;

	// Create a new active range for the merge.
	int leftEnd = listParity ? siblingEnd : activeRange.y;
	int rightStart = listParity ? activeRange.x : siblingStart;
	activeRange.x = min(activeRange.x, siblingStart);
	activeRange.y = max(activeRange.y, siblingEnd);

	int p = SegmentedMergePath(keys_shared, a0, listLen, b0, listLen, leftEnd,
		rightStart, diag, comp);

	int a0tid = a0 + p;
	int b0tid = b0 + diag - p;
	SegmentedSerialMerge<VT>(keys_shared, a0tid, b0, b0tid, b0 + listLen,
		results, indices, leftEnd, rightStart, comp);

	// Store the ranges to shared memory.
	if(0 == diag)
		ranges_shared[list>> 1] =
			(int)bfi(activeRange.y, activeRange.x, 16, 16);
}

////////////////////////////////////////////////////////////////////////////////
// CTASegsortLoop

template<int NT, int VT, bool HasValues, typename KeyType, typename ValType,
	typename Comp>
MGPU_DEVICE int2 CTASegsortLoop(KeyType threadKeys[VT],
	ValType threadValues[VT], KeyType* keys_shared, ValType* values_shared,
	int* ranges_shared, int tid, int2 activeRange, Comp comp) {

	const int NumPasses = sLogPow2<NT>::value;
	#pragma unroll
	for(int pass = 0; pass < NumPasses; ++pass) {
		int indices[VT];
		CTASegsortPass<NT, VT>(keys_shared, ranges_shared, tid, pass,
			threadKeys, indices, activeRange, comp);

		if(HasValues) {
			// Exchange values through shared memory.
			DeviceThreadToShared<VT>(threadValues, tid, values_shared);
			DeviceGather<NT, VT>(NT * VT, values_shared, indices, tid,
				threadValues);
		}

		// Store results in shared memory in sorted order.
		DeviceThreadToShared<VT>(threadKeys, tid, keys_shared);
	}
	return activeRange;
}

////////////////////////////////////////////////////////////////////////////////
// CTASegsort
// Pass keys and values in register. On return, values are returned in register
// and keys returned in shared memory.

template<int NT, int VT, bool Stable, bool HasValues, typename KeyType,
	typename ValType, typename Comp>
MGPU_DEVICE int2 CTASegsort(KeyType threadKeys[VT], ValType threadValues[VT],
	int tid, int headFlags, KeyType* keys_shared, ValType* values_shared,
	int* ranges_shared, Comp comp) {

	if(Stable)
		// Odd-even transpose sort.
		OddEvenTransposeSortFlags<VT>(threadKeys, threadValues, headFlags,
			comp);
	else
		// Batcher's odd-even mergesort.
		OddEvenMergesortFlags<VT>(threadKeys, threadValues, headFlags, comp);

	// Record the first and last occurrence of head flags in this segment.
	int blockEnd = 31 - clz(headFlags);
	if(-1 != blockEnd) blockEnd += VT * tid;

	int blockStart = ffs(headFlags);
	blockStart = blockStart ? (VT * tid - 1 + blockStart) : (NT * VT);

	ranges_shared[tid] = (int)bfi(blockEnd, blockStart, 16, 16);

	// Store back to shared mem. The values are in VT-length sorted lists.
	// These are merged recursively.
	DeviceThreadToShared<VT>(threadKeys, tid, keys_shared);

	int2 activeRange = CTASegsortLoop<NT, VT, HasValues>(threadKeys,
		threadValues, keys_shared, values_shared, ranges_shared, tid,
		make_int2(blockStart, blockEnd), comp);
	return activeRange;
}


template<int NT, int VT, bool Stable, typename KeyType, typename Comp>
MGPU_DEVICE int2 CTASegsortKeys(KeyType threadKeys[VT], int tid, int headFlags,
	KeyType* keys_shared, int* ranges_shared, Comp comp) {

	int valuesTemp[VT];
	return CTASegsort<NT, VT, Stable, false>(threadKeys, valuesTemp, tid,
		headFlags, keys_shared, (int*)keys_shared, ranges_shared, comp);
}

template<int NT, int VT, bool Stable, typename KeyType, typename ValType,
	typename Comp>
MGPU_DEVICE int2 CTASegsortPairs(KeyType threadKeys[VT],
	ValType threadValues[VT], int tid, int headFlags, KeyType* keys_shared,
	ValType* values_shared, int* ranges_shared, Comp comp) {

	return CTASegsort<NT, VT, Stable, true>(threadKeys, threadValues, tid,
		headFlags, keys_shared, values_shared, ranges_shared, comp);
}

////////////////////////////////////////////////////////////////////////////////
// DeviceSegBlocksort
// Load keys and values from global memory, sort in shared memory, and store
// back to global memory. Store the left-most and right-most encountered
// headflag locations to ranges_global to prepare for the next pass.
// This function is factored out of the blocksort kernel to allow easier
// customization of that kernel - we have two implementations currently:
// sort over indices and sort over bitfield.

template<int NT, int VT, bool Stable, bool HasValues, typename InputIt1,
	typename InputIt2, typename KeyType, typename ValType, typename OutputIt1,
	typename OutputIt2, typename Comp>
MGPU_DEVICE void DeviceSegBlocksort(InputIt1 keys_global,
	InputIt2 values_global, int count2, KeyType* keys_shared,
	ValType* values_shared, int* ranges_shared, int headFlags, int tid,
	int block, OutputIt1 keysDest_global, OutputIt2 valsDest_global,
	int* ranges_global, Comp comp) {

	// Load keys into register in thread order.
	int gid = NT * VT * block;
	KeyType threadKeys[VT];
	DeviceGlobalToShared<NT, VT>(count2, keys_global + gid, tid, keys_shared);
	DeviceSharedToThread<VT>(keys_shared, tid, threadKeys);

	// Load the values from global memory and into register in thread order.
	ValType threadValues[VT];
	if(HasValues) {
		DeviceGlobalToShared<NT, VT>(count2, values_global + gid, tid,
			values_shared);
		DeviceSharedToThread<VT>(values_shared, tid, threadValues);
	}

	// Run the CTA segmented blocksort.
	int2 activeRange = CTASegsort<NT, VT, Stable, HasValues>(threadKeys,
		threadValues, tid, headFlags, keys_shared, values_shared, ranges_shared,
		comp);

	// Store the keys to global memory.
	DeviceSharedToGlobal<NT, VT>(count2, keys_shared, tid,
		 keysDest_global + gid);

	if(HasValues) {
		// Store the values to global memory.xk b
		DeviceThreadToShared<VT>(threadValues, tid, values_shared);
		DeviceSharedToGlobal<NT, VT>(count2, values_shared, tid,
			valsDest_global + gid, false);
	}

	// Store the 16-bit packed ranges. These are used by all merge kernels and
	// the first level of global segmented merge path partitioning.
	if(!tid)
		ranges_global[block] = bfi(activeRange.y, activeRange.x, 16, 16);
}

////////////////////////////////////////////////////////////////////////////////
// DeviceIndicesToHeadFlags
// Load indices from an array and cooperatively turn into a head flag bitfield
// for each thread.

template<int NT, int VT>
MGPU_DEVICE int DeviceIndicesToHeadFlags(const int* indices_global,
	const int* partitions_global, int tid, int block, int count2,
	int* words_shared, byte* flags_shared) {

	const int FlagWordsPerThread = MGPU_DIV_UP(VT, 4);
	int gid = NT * VT * block;
	int p0 = partitions_global[block];
	int p1 = partitions_global[block + 1];

	int headFlags = 0;
	if(p1 > p0 || count2 < NT * VT) {

		// Clear the flag bytes, then loop through the indices and poke in flag
		// values.
		#pragma unroll
		for(int i = 0; i < FlagWordsPerThread; ++i)
			words_shared[NT * i + tid] = 0;
		__syncthreads();

		for(int index = p0 + tid; index < p1; index += NT) {
			int headFlag = indices_global[index];
			flags_shared[headFlag - gid] = 1;
		}
		__syncthreads();

		// Combine all the head flags for this thread.
		int first = VT * tid;
		int offset = first / 4;
		int prev = words_shared[offset];
		int mask = 0x3210 + 0x1111 * (3 & first);
		#pragma unroll
		for(int i = 0; i < FlagWordsPerThread; ++i) {
			// Gather the next four flags.
			int next = words_shared[offset + 1 + i];
			int x = prmt(prev, next, mask);
			prev = next;

			// Set the head flag bits.
			if(0x00000001 & x) headFlags |= 1<< (4 * i);
			if(0x00000100 & x) headFlags |= 1<< (4 * i + 1);
			if(0x00010000 & x) headFlags |= 1<< (4 * i + 2);
			if(0x01000000 & x) headFlags |= 1<< (4 * i + 3);
		}
		__syncthreads();

		// Set head flags for out-of-range keys.
		int outOfRange = min(VT, first + VT - count2);
		if(outOfRange > 0)
			headFlags = bfi(0xffffffff, headFlags, VT - outOfRange, outOfRange);

		// Clear head flags above VT.
		headFlags &= (1<< VT) - 1;
	}
	return headFlags;
}

////////////////////////////////////////////////////////////////////////////////
// SegSortSupport

struct SegSortSupport {
	int* ranges_global;
	int2* ranges2_global;

	int4* mergeList_global;
	int* copyList_global;
	int2* queueCounters_global;
	int2* nextCounters_global;

	byte* copyStatus_global;
};

////////////////////////////////////////////////////////////////////////////////
// DeviceSegSortMerge

template<int NT, int VT, bool HasValues, typename KeyType, typename ValueType,
	typename Comp>
MGPU_DEVICE void DeviceSegSortMerge(const KeyType* keys_global,
	const ValueType* values_global, int2 segmentRange, int tid,
	int block, int4 range, int pass, KeyType* keys_shared,
	int* indices_shared, KeyType* keysDest_global, ValueType* valsDest_global,
	Comp comp) {

	const int NV = NT * VT;
	int gid = NV * block;

	// Load the local compressed segment indices.
	int a0 = range.x;
	int aCount = range.y - range.x;
	int b0 = range.z;
	int bCount = range.w - range.z;

	DeviceLoad2ToShared<NT, VT, VT>(keys_global + a0, aCount, keys_global + b0,
		bCount, tid, keys_shared);

	////////////////////////////////////////////////////////////////////////////
	// Run a merge path to find the starting point for each thread to merge.
	// If the entire warp fits into the already-sorted segments, we can skip
	// sorting it and leave its keys in shared memory. Doing this on the warp
	// level rather than thread level (also legal) gives slightly better
	// performance.

	int segStart = segmentRange.x;
	int segEnd = segmentRange.y;
	int listParity = 1 & (block>> pass);

	int warpOffset = VT * (~31 & tid);
	bool sortWarp = listParity ?
		// The spliced segment is to the left (segStart).
		(warpOffset < segStart) :
		// The spliced segment is to the right (segEnd).
		(warpOffset + 32 * VT > segEnd);

	KeyType threadKeys[VT];
	int indices[VT];
	if(sortWarp) {
		int diag = VT * tid;
		int mp = SegmentedMergePath(keys_shared, 0, aCount, aCount, bCount,
			listParity ? 0 : segEnd, listParity ? segStart : NV, diag, comp);
		int a0tid = mp;
		int a1tid = aCount;
		int b0tid = aCount + diag - mp;
		int b1tid = aCount + bCount;

		// Serial merge into register. All threads in the CTA so we hoist the
		// check for list parity outside the function call to simplify the
		// logic. Unlike in the blocksort, this does not cause warp divergence.
		SegmentedSerialMerge<VT>(keys_shared, a0tid, a1tid, b0tid, b1tid,
			threadKeys, indices, listParity ? 0 : segEnd,
			listParity ? segStart : NV, comp, false);
	}
	__syncthreads();

	// Store sorted data in register back to shared memory. Then copy to global.
	if(sortWarp)
		DeviceThreadToShared<VT>(threadKeys, tid, keys_shared, false);
	__syncthreads();

	DeviceSharedToGlobal<NT, VT>(aCount + bCount, keys_shared, tid,
		keysDest_global + gid);

	////////////////////////////////////////////////////////////////////////////
	// Use the merge indices to gather values from global memory. Store directly
	// to valsDest_global.

	if(HasValues) {
		// Transpose the gather indices to help coalesce loads.
		if(sortWarp)
			DeviceThreadToShared<VT>(indices, tid, indices_shared, false);
		else {
			#pragma unroll
			for(int i = 0; i < VT; ++i)
				indices_shared[VT * tid + i] = VT * tid + i;
		}
		__syncthreads();

		DeviceTransferMergeValuesShared<NT, VT>(aCount + bCount,
			values_global + a0,  values_global + b0, aCount, indices_shared,
			tid, valsDest_global + NV * block);
	}
}

////////////////////////////////////////////////////////////////////////////////
// DeviceSegSortCopy

template<int NT, int VT, bool HasValues, typename KeyType, typename ValueType>
MGPU_DEVICE void DeviceSegSortCopy(const KeyType* keys_global,
	const ValueType* values_global, int tid, int block, int count,
	KeyType* keysDest_global, ValueType* valsDest_global) {

	int gid = NT * VT * block;
	int count2 = min(NT * VT, count - gid);

	DeviceGlobalToGlobal<NT, VT>(count2, keys_global + gid, tid,
		keysDest_global + gid);
	if(HasValues)
		DeviceGlobalToGlobal<NT, VT>(count2, values_global + gid, tid,
			valsDest_global + gid);
}

} // namespace mgpu
