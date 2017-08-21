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

namespace mgpu {

////////////////////////////////////////////////////////////////////////////////
// SerialSetIntersection
// Emit A if A and B are in range and equal.

template<int VT, bool RangeCheck, typename T, typename Comp>
MGPU_DEVICE int SerialSetIntersection(const T* data, int aBegin, int aEnd,
	int bBegin, int bEnd, int end, T* results, int* indices, Comp comp) {

	const int MinIterations = VT / 2;
	int commit = 0;

	#pragma unroll
	for(int i = 0; i < VT; ++i) {
		bool test = RangeCheck ?
			((aBegin + bBegin < end) && (aBegin < aEnd) && (bBegin < bEnd)) :
			(i < MinIterations || (aBegin + bBegin < end));

		if(test) {
			T aKey = data[aBegin];
			T bKey = data[bBegin];

			bool pA = comp(aKey, bKey);
			bool pB = comp(bKey, aKey);

			// The outputs must come from A by definition of set interection.
			results[i] = aKey;
			indices[i] = aBegin;

			if(!pB) ++aBegin;
			if(!pA) ++bBegin;
			if(pA == pB) commit |= 1<< i;
		}
	}
	return commit;
}

////////////////////////////////////////////////////////////////////////////////
// SerialSetUnion
// Emit A if A <= B. Emit B if B < A.

template<int VT, bool RangeCheck, typename T, typename Comp>
MGPU_DEVICE int SerialSetUnion(const T* data, int aBegin, int aEnd,
	int bBegin, int bEnd, int end, T* results, int* indices, Comp comp) {

	const int MinIterations = VT / 2;
	int commit = 0;

	#pragma unroll
	for(int i = 0; i < VT; ++i) {
		bool test = RangeCheck ?
			(aBegin + bBegin < end) :
			(i < MinIterations || (aBegin + bBegin < end));

		if(test) {
			T aKey = data[aBegin];
			T bKey = data[bBegin];

			bool pA = false, pB = false;
			if(RangeCheck && aBegin >= aEnd)
				pB = true;
			else if(RangeCheck && bBegin >= bEnd)
				pA = true;
			else {
				// Both are in range.
				pA = comp(aKey, bKey);
				pB = comp(bKey, aKey);
			}

			// Output A in case of a tie, so check if b < a.
			results[i] = pB ? bKey : aKey;
			indices[i] = pB ? bBegin : aBegin;
			if(!pB) ++aBegin;
			if(!pA) ++bBegin;
			commit |= 1<< i;
		}
	}
	return commit;
}

////////////////////////////////////////////////////////////////////////////////
// SerialSetDifference
// Emit A if A < B.

template<int VT, bool RangeCheck, typename T, typename Comp>
MGPU_DEVICE int SerialSetDifference(const T* data, int aBegin, int aEnd,
	int bBegin, int bEnd, int end, T* results, int* indices, Comp comp) {

	const int MinIterations = VT / 2;
	int commit = 0;

	#pragma unroll
	for(int i = 0; i < VT; ++i) {
		bool test = RangeCheck ?
			(aBegin + bBegin < end) :
			(i < MinIterations || (aBegin + bBegin < end));
		if(test) {
			T aKey = data[aBegin];
			T bKey = data[bBegin];

			bool pA = false, pB = false;
			if(RangeCheck && aBegin >= aEnd)
				pB = true;
			else if(RangeCheck && bBegin >= bEnd)
				pA = true;
			else {
				pA = comp(aKey, bKey);
				pB = comp(bKey, aKey);
			}

			// The outputs must come from A by definition of set difference.
			results[i] = aKey;
			indices[i] = aBegin;
			if(!pB) ++aBegin;
			if(!pA) ++bBegin;
			if(pA) commit |= 1<< i;
		}
	}
	return commit;
}

////////////////////////////////////////////////////////////////////////////////
// SerialSetSymDiff
// Emit A if A < B and emit B if B < A.

template<int VT, bool RangeCheck, typename T, typename Comp>
MGPU_DEVICE int SerialSetSymDiff(const T* data, int aBegin, int aEnd,
	int bBegin, int bEnd, int end, T* results, int* indices, Comp comp) {

	const int MinIterations = VT / 2;
	int commit = 0;

	#pragma unroll
	for(int i = 0; i < VT; ++i) {
		bool test = RangeCheck ?
			(aBegin + bBegin < end) :
			(i < MinIterations || (aBegin + bBegin < end));
		if(test) {
			T aKey = data[aBegin];
			T bKey = data[bBegin];

			bool pA = false, pB = false;
			if(RangeCheck && (bBegin >= bEnd))
				pA = true;
			else if(RangeCheck && (aBegin >= aEnd))
				pB = true;
			else {
				pA = comp(aKey, bKey);
				pB = comp(bKey, aKey);
			}

			results[i] = pA ? aKey : bKey;
			indices[i] = pA ? aBegin : bBegin;
			if(!pA) ++bBegin;
			if(!pB) ++aBegin;
			if(pA != pB) commit |= 1<< i;
		}
	}
	return commit;
}

////////////////////////////////////////////////////////////////////////////////
// SerialSetOp
// Uses the MgpuSetOp enum to statically select one of the four serial ops
// above.

template<int VT, bool RangeCheck, MgpuSetOp Op, typename T, typename Comp>
MGPU_DEVICE int SerialSetOp(const T* data, int aBegin, int aEnd,
	int bBegin, int bEnd, int star, T* results, int* indices, Comp comp) {

	int end = aBegin + bBegin + VT - star;
	if(RangeCheck) end = min(end, aEnd + bEnd);
	int commit;
	switch(Op) {
		case MgpuSetOpIntersection:
			commit = SerialSetIntersection<VT, RangeCheck>(data, aBegin,
				aEnd, bBegin, bEnd, end, results, indices, comp);
			break;
		case MgpuSetOpUnion:
			commit = SerialSetUnion<VT, RangeCheck>(data, aBegin, aEnd,
				bBegin, bEnd, end, results, indices, comp);
			break;
		case MgpuSetOpDiff:
			commit = SerialSetDifference<VT, RangeCheck>(data, aBegin, aEnd,
				bBegin, bEnd, end, results, indices, comp);
			break;
		case MgpuSetOpSymDiff:
			commit = SerialSetSymDiff<VT, RangeCheck>(data, aBegin, aEnd,
				bBegin, bEnd, end, results, indices, comp);
			break;
	}
	__syncthreads();
	return commit;
}

} // namespace mgpu
