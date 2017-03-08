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

#include "../mgpudevice.h"
#include "deviceutil.h"
#include "intrinsics.h"

namespace mgpu {

////////////////////////////////////////////////////////////////////////////////
// Cooperative load functions.

template<int NT, int VT, typename InputIt, typename T>
MGPU_DEVICE void DeviceSharedToReg(InputIt data, int tid, T* reg,
	bool sync) {

	#pragma unroll
	for(int i = 0; i < VT; ++i)
		reg[i] = data[NT * i + tid];

	if(sync) __syncthreads();
}

template<int NT, int VT, typename InputIt, typename T>
MGPU_DEVICE void DeviceGlobalToRegPred(int count, InputIt data, int tid,
	T* reg, bool sync) {

	// TODO: Attempt to issue 4 loads at a time.
	#pragma unroll
	for(int i = 0; i < VT; ++i) {
		int index = NT * i + tid;
		if(index < count) reg[i] = data[index];
	}
	if(sync) __syncthreads();
}

template<int NT, int VT, typename InputIt, typename T>
MGPU_DEVICE void DeviceGlobalToReg(int count, InputIt data, int tid,
	T* reg, bool sync) {

	if(count >= NT * VT) {
		#pragma unroll
		for(int i = 0; i < VT; ++i)
			reg[i] = data[NT * i + tid];
	} else
		DeviceGlobalToRegPred<NT, VT>(count, data, tid, reg, false);
	if(sync) __syncthreads();
}
template<int NT, int VT0, int VT1, typename InputIt, typename T>
MGPU_DEVICE void DeviceGlobalToReg2(int count, InputIt data, int tid,
	T* reg, bool sync) {

	DeviceGlobalToReg<NT, VT0>(count, data, tid, reg, false);
	#pragma unroll
	for(int i = VT0; i < VT1; ++i) {
		int index = NT * i + tid;
		if(index < count) reg[i] = data[index];
	}
	if(sync) __syncthreads();
}

template<int NT, int VT, typename InputIt, typename T>
MGPU_DEVICE void DeviceGlobalToRegDefault(int count, InputIt data, int tid,
	T* reg, T init, bool sync) {

	if(count >= NT * VT) {
		#pragma unroll
		for(int i = 0; i < VT; ++i)
			reg[i] = data[NT * i + tid];
	} else {
		#pragma unroll
		for(int i = 0; i < VT; ++i) {
			int index = NT * i + tid;
			reg[i] = init;
			if(index < count) reg[i] = data[index];
		}
	}
	if(sync) __syncthreads();
}
template<int NT, int VT0, int VT1, typename InputIt, typename T>
MGPU_DEVICE void DeviceGlobalToRegDefault2(int count, InputIt data, int tid,
	T* reg, T init, bool sync) {

	DeviceGlobalToRegDefault<NT, VT0>(count, data, tid, reg, init, false);
	#pragma unroll
	for(int i = VT0; i < VT1; ++i) {
		int index = NT * i + tid;
		reg[i] = init;
		if(index < count) reg[i] = data[index];
	}
	if(sync) __syncthreads();
}

////////////////////////////////////////////////////////////////////////////////

template<int NT, int VT, typename InputIt, typename T>
MGPU_DEVICE void DeviceGlobalToThread(int count, InputIt data, int tid,
	T* reg) {

	data += VT * tid;
	if(count >= NT * VT) {
		#pragma unroll
		for(int i = 0; i < VT; ++i)
			reg[i] = ldg(data + i);
	} else {
		count -= VT * tid;
		#pragma unroll
		for(int i = 0; i < VT; ++i)
			if(i < count) reg[i] = ldg(data + i);
	}
}

template<int NT, int VT, typename InputIt, typename T>
MGPU_DEVICE void DeviceGlobalToThreadDefault(int count, InputIt data, int tid,
	T* reg, T init) {

	data += VT * tid;
	if(count >= NT * VT) {
		#pragma unroll
		for(int i = 0; i < VT; ++i)
			reg[i] = ldg(data + i);
	} else {
		count -= VT * tid;
		#pragma unroll
		for(int i = 0; i < VT; ++i)
			reg[i] = (i < count) ? ldg(data + i) : init;
	}
}


////////////////////////////////////////////////////////////////////////////////
// Cooperative store functions.

template<int NT, int VT, typename OutputIt, typename T>
MGPU_DEVICE void DeviceRegToShared(const T* reg, int tid,
	OutputIt dest, bool sync) {

	typedef typename std::iterator_traits<OutputIt>::value_type T2;
	#pragma unroll
	for(int i = 0; i < VT; ++i)
		dest[NT * i + tid] = (T2)reg[i];

	if(sync) __syncthreads();
}

template<int NT, int VT, typename OutputIt, typename T>
MGPU_DEVICE void DeviceRegToGlobal(int count, const T* reg, int tid,
	OutputIt dest, bool sync) {

	#pragma unroll
	for(int i = 0; i < VT; ++i) {
		int index = NT * i + tid;
		if(index < count)
			dest[index] = reg[i];
	}
	if(sync) __syncthreads();
}

////////////////////////////////////////////////////////////////////////////////
// DeviceMemToMemLoop
// Transfer from shared memory to global, or global to shared, for transfers
// that are smaller than NT * VT in the average case. The goal is to reduce
// unnecessary comparison logic.

template<int NT, int VT, typename InputIt, typename OutputIt>
MGPU_DEVICE void DeviceMemToMem4(int count, InputIt source, int tid,
	OutputIt dest, bool sync) {

	typedef typename std::iterator_traits<InputIt>::value_type T;

	T x[VT];
	const int Count = (VT < 4) ? VT : 4;
	if(count >= NT * VT) {
		#pragma unroll
		for(int i = 0; i < Count; ++i)
			x[i] = source[NT * i + tid];
		#pragma unroll
		for(int i = 0; i < Count; ++i)
			dest[NT * i + tid] = x[i];
	} else {
		#pragma unroll
		for(int i = 0; i < Count; ++i) {
			int index = NT * i + tid;
			if(index < count)
				x[i] = source[NT * i + tid];
		}
		#pragma unroll
		for(int i = 0; i < Count; ++i) {
			int index = NT * i + tid;
			if(index < count)
				dest[index] = x[i];
		}
	}
	if(sync) __syncthreads();
}
template<int NT, typename InputIt, typename OutputIt>
MGPU_DEVICE void DeviceMemToMemLoop(int count, InputIt source, int tid,
	OutputIt dest, bool sync) {

	for(int i = 0; i < count; i += 4 * NT)
		DeviceMemToMem4<NT, 4>(count - i, source + i, tid, dest + i,
			false);
	if(sync) __syncthreads();
}


////////////////////////////////////////////////////////////////////////////////
// Functions to copy between shared and global memory where the average case is
// to transfer NT * VT elements.

template<int NT, int VT, typename T, typename OutputIt>
MGPU_DEVICE void DeviceSharedToGlobal(int count, const T* source, int tid,
	OutputIt dest, bool sync) {

	typedef typename std::iterator_traits<OutputIt>::value_type T2;
	#pragma unroll
	for(int i = 0; i < VT; ++i) {
		int index = NT * i + tid;
		if(index < count) dest[index] = (T2)source[index];
	}
	if(sync) __syncthreads();
}

template<int NT, int VT, typename InputIt, typename T>
MGPU_DEVICE void DeviceGlobalToShared(int count, InputIt source, int tid,
	T* dest, bool sync) {

	T reg[VT];
	DeviceGlobalToReg<NT, VT>(count, source, tid, reg, false);
	DeviceRegToShared<NT, VT>(reg, tid, dest, sync);
}

template<int NT, int VT0, int VT1, typename InputIt, typename T>
MGPU_DEVICE void DeviceGlobalToShared2(int count, InputIt source, int tid,
	T* dest, bool sync) {

	T reg[VT1];
	DeviceGlobalToReg2<NT, VT0, VT1>(count, source, tid, reg, false);
	DeviceRegToShared<NT, VT1>(reg, tid, dest, sync);
}


template<int NT, int VT, typename InputIt, typename T>
MGPU_DEVICE void DeviceGlobalToSharedDefault(int count, InputIt source, int tid,
	T* dest, T init, bool sync) {

	T reg[VT];
	DeviceGlobalToRegDefault<NT, VT>(count, source, tid, reg, init, false);
	DeviceRegToShared<NT, VT>(reg, tid, dest, sync);
}

template<int NT, int VT0, int VT1, typename InputIt, typename T>
MGPU_DEVICE void DeviceGlobalToSharedDefault2(int count, InputIt data, int tid,
	T* dest, T init, bool sync) {

	T reg[VT1];
	DeviceGlobalToRegDefault2<NT, VT0, VT1>(count, data, tid, reg, init, false);
	DeviceRegToShared<NT, VT1>(reg, tid, dest, sync);
}


////////////////////////////////////////////////////////////////////////////////

template<int NT, int VT, typename InputIt, typename T>
MGPU_DEVICE void DeviceGlobalToSharedLoop(int count, InputIt source, int tid,
	T* dest, bool sync) {

	const int Granularity = MGPU_MIN(VT, 3);
	DeviceGlobalToShared<NT, Granularity>(count, source, tid, dest, false);

	int offset = Granularity * NT;
	if(count > offset)
		DeviceGlobalToShared<NT, VT - Granularity>(count - offset,
			source + offset, tid, dest + offset, false);

	if(sync) __syncthreads();

	/*
	source += tid;
	while(count > 0) {
		T reg[Granularity];
		#pragma unroll
		for(int i = 0; i < Granularity; ++i) {
			int index = NT * i + tid;
			if(index < count)
				reg[i] = source[NT * i];
		}
		DeviceRegToShared<NT, Granularity>(reg, tid, dest, false);
		source += Granularity * NT;
		dest += Granularity * NT;
		count -= Granularity * NT;
	}
	if(sync) __syncthreads();*/
}

template<int NT, int VT, typename InputIt, typename OutputIt>
MGPU_DEVICE void DeviceGlobalToGlobal(int count, InputIt source, int tid,
	OutputIt dest, bool sync) {

	typedef typename std::iterator_traits<OutputIt>::value_type T;
	T values[VT];
	DeviceGlobalToReg<NT, VT>(count, source, tid, values, false);
	DeviceRegToGlobal<NT, VT>(count, values, tid, dest, sync);
}

////////////////////////////////////////////////////////////////////////////////
// Transponse VT elements in NT threads (x) into thread-order registers (y)
// using only NT * VT / 2 elements of shared memory.

//This function definitely has a bug, don't use!!! fix TODO(erich)
template<int NT, int VT, typename T>
MGPU_DEVICE void HalfSmemTranspose(const T* x, int tid, T* shared, T* y) {
    printf("HalfSmemTranspose has a bug, use WAR SmemTranpose or find bug before using in production");
	// Transpose the first half values (tid < NT / 2)
	#pragma unroll
	for(int i = 0; i <= VT / 2; ++i)
		if(i < VT / 2 || tid < NT / 2)
			shared[NT * i + tid] = x[i];
	__syncthreads();

	if(tid < NT / 2) {
		#pragma unroll
		for(int i = 0; i < VT; ++i)
			y[i] = shared[VT * tid + i];
	}
	__syncthreads();

	// Transpose the second half values (tid >= NT / 2)
	#pragma unroll
	for(int i = VT / 2; i < VT; ++i)
		if(i > VT / 2 || tid >= NT / 2)
			shared[NT * i - NT * VT / 2 + tid] = x[i];
	__syncthreads();

	if(tid >= NT / 2) {
		#pragma unroll
		for(int i = 0; i < VT; ++i)
			y[i] = shared[VT * tid + i - NT * VT / 2];
	}
	__syncthreads();
}

////////////////////////////////////////////////////////////////////////////////
// Gather/scatter functions

template<int NT, int VT, typename InputIt, typename T>
MGPU_DEVICE void DeviceGather(int count, InputIt data, int indices[VT],
	int tid, T* reg, bool sync) {

	if(count >= NT * VT) {
		#pragma unroll
		for(int i = 0; i < VT; ++i)
			reg[i] = data[indices[i]];
	} else {
		#pragma unroll
		for(int i = 0; i < VT; ++i) {
			int index = NT * i + tid;
			if(index < count)
				reg[i] = data[indices[i]];
		}
	}
	if(sync) __syncthreads();
}

template<int NT, int VT, typename InputIt, typename T>
MGPU_DEVICE void DeviceGatherDefault(int count, InputIt data, int indices[VT],
	int tid, T* reg, T identity, bool sync) {

	if(count >= NT * VT) {
		#pragma unroll
		for(int i = 0; i < VT; ++i)
			reg[i] = data[indices[i]];
	} else {
		#pragma unroll
		for(int i = 0; i < VT; ++i) {
			int index = NT * i + tid;
			reg[i] = (index < count) ? data[indices[i]] : identity;
		}
	}
	if(sync) __syncthreads();
}

template<int NT, int VT, typename T, typename OutputIt>
MGPU_DEVICE void DeviceScatter(int count, const T* reg, int tid,
	int indices[VT], OutputIt data, bool sync) {

	if(count >= NT * VT) {
		#pragma unroll
		for(int i = 0; i < VT; ++i)
			data[indices[i]] = reg[i];
	} else {
		#pragma unroll
		for(int i = 0; i < VT; ++i) {
			int index = NT * i + tid;
			if(index < count)
				data[indices[i]] = reg[i];
		}
	}
	if(sync) __syncthreads();
}

////////////////////////////////////////////////////////////////////////////////
// Cooperative transpose functions (strided to thread order)

template<int VT, typename T>
MGPU_DEVICE void DeviceThreadToShared(const T* threadReg, int tid, T* shared,
	bool sync) {

	if(1 & VT) {
		// Odd grain size. Store as type T.
		#pragma unroll
		for(int i = 0; i < VT; ++i)
			shared[VT * tid + i] = threadReg[i];
	} else {
		// Even grain size. Store as DevicePair<T>. This lets us exploit the
		// 8-byte shared memory mode on Kepler.
		DevicePair<T>* dest = (DevicePair<T>*)(shared + VT * tid);
		#pragma unroll
		for(int i = 0; i < VT / 2; ++i)
			dest[i] = MakeDevicePair(threadReg[2 * i], threadReg[2 * i + 1]);
	}
	if(sync) __syncthreads();
}

template<int VT, typename T>
MGPU_DEVICE void DeviceSharedToThread(const T* shared, int tid, T* threadReg,
	bool sync) {

	if(1 & VT) {
		#pragma unroll
		for(int i = 0; i < VT; ++i)
			threadReg[i] = shared[VT * tid + i];
	} else {
		const DevicePair<T>* source = (const DevicePair<T>*)(shared + VT * tid);
		#pragma unroll
		for(int i = 0; i < VT / 2; ++i) {
			DevicePair<T> p = source[i];
			threadReg[2 * i] = p.x;
			threadReg[2 * i + 1] = p.y;
		}
	}
	if(sync) __syncthreads();
}

////////////////////////////////////////////////////////////////////////////////
// DeviceLoad2 - load from pointers of the same type. Optimize for a single LD
// statement.

template<int NT, int VT0, int VT1, typename T>
MGPU_DEVICE void DeviceLoad2ToReg(const T* a_global, int aCount,
	const T* b_global, int bCount, int tid, T* reg, bool sync) {

	int b0 = b_global - a_global - aCount;
	int total = aCount + bCount;
	if(total >= NT * VT0) {
		#pragma unroll
		for(int i = 0; i < VT0; ++i) {
			int index = NT * i + tid;
			reg[i] = a_global[index + ((index >= aCount) ? b0 : 0)];
		}
	} else {
		#pragma unroll
		for(int i = 0; i < VT0; ++i) {
			int index = NT * i + tid;
			if(index < total)
				reg[i] = a_global[index + ((index >= aCount) ? b0 : 0)];
		}
	}
	#pragma unroll
	for(int i = VT0; i < VT1; ++i) {
		int index = NT * i + tid;
		if(index < total)
			reg[i] = a_global[index + ((index >= aCount) ? b0 : 0)];
	}
}

template<int NT, int VT0, int VT1, typename T>
MGPU_DEVICE void DeviceLoad2ToShared(const T* a_global, int aCount,
	const T* b_global, int bCount, int tid, T* shared, bool sync) {

	T reg[VT1];
	DeviceLoad2ToReg<NT, VT0, VT1>(a_global, aCount, b_global, bCount, tid,
		reg, false);
	DeviceRegToShared<NT, VT1>(reg, tid, shared, sync);
}

////////////////////////////////////////////////////////////////////////////////
// DeviceLoad2 - load from pointers of different types. Uses two LD statements.

template<int NT, int VT0, int VT1, typename InputIt1, typename InputIt2,
	typename T>
MGPU_DEVICE void DeviceLoad2ToReg(InputIt1 a_global, int aCount,
	InputIt2 b_global, int bCount, int tid, T* reg, bool sync)  {

	b_global -= aCount;
	int total = aCount + bCount;
	if(total >= NT * VT0) {
		#pragma unroll
		for(int i = 0; i < VT0; ++i) {
			int index = NT * i + tid;
			if(index < aCount) reg[i] = a_global[index];
			else reg[i] = b_global[index];
		}
	} else {
		#pragma unroll
		for(int i = 0; i < VT0; ++i) {
			int index = NT * i + tid;
			if(index < aCount) reg[i] = a_global[index];
			else if(index < total) reg[i] = b_global[index];
		}
	}
	#pragma unroll
	for(int i = VT0; i < VT1; ++i) {
		int index = NT * i + tid;
		if(index < aCount) reg[i] = a_global[index];
		else if(index < total) reg[i] = b_global[index];
	}
}

template<int NT, int VT0, int VT1, typename InputIt1, typename InputIt2,
	typename T>
MGPU_DEVICE void DeviceLoad2ToShared(InputIt1 a_global, int aCount,
	InputIt2 b_global, int bCount, int tid, T* shared, bool sync) {

	T reg[VT1];
	DeviceLoad2ToReg<NT, VT0, VT1>(a_global, aCount, b_global, bCount, tid,
		reg, false);
	DeviceRegToShared<NT, VT1>(reg, tid, shared, sync);
}


////////////////////////////////////////////////////////////////////////////////
// DeviceGatherGlobalToGlobal

template<int NT, int VT, typename InputIt, typename OutputIt>
MGPU_DEVICE void DeviceGatherGlobalToGlobal(int count, InputIt data_global,
	const int* indices_shared, int tid, OutputIt dest_global, bool sync) {

	typedef typename std::iterator_traits<InputIt>::value_type ValType;
	ValType values[VT];

	#pragma unroll
	for(int i = 0; i < VT; ++i) {
		int index = NT * i + tid;
		if(index < count) {
			int gather = indices_shared[index];
			values[i] = data_global[gather];
		}
	}
	if(sync) __syncthreads();
	DeviceRegToGlobal<NT, VT>(count, values, tid, dest_global, false);
}

////////////////////////////////////////////////////////////////////////////////
// DeviceTransferMergeValues
// Gather in a merge-like value from two input arrays and store to a single
// output. Like DeviceGatherGlobalToGlobal, but for two arrays at once.

template<int NT, int VT, typename InputIt1, typename InputIt2,
	typename T>
MGPU_DEVICE void DeviceTransferMergeValuesReg(int count, InputIt1 a_global,
	InputIt2 b_global, int bStart, const int* indices, int tid,
	T* reg, bool sync) {

	b_global -= bStart;
	if(count >= NT * VT) {
		#pragma unroll
		for(int i = 0; i < VT; ++i) {
			reg[i] = (indices[i] < bStart) ? a_global[indices[i]] :
				b_global[indices[i]];
		}
	} else {
		#pragma unroll
		for(int i = 0; i < VT; ++i) {
			int index = NT * i + tid;
			if(index < count)
				reg[i] = (indices[i] < bStart) ? a_global[indices[i]] :
					b_global[indices[i]];
		}
	}
	if(sync) __syncthreads();
}

template<int NT, int VT, typename InputIt1, typename InputIt2,
	typename OutputIt>
MGPU_DEVICE void DeviceTransferMergeValuesShared(int count, InputIt1 a_global,
	InputIt2 b_global, int bStart, const int* indices_shared, int tid,
	OutputIt dest_global, bool sync) {

	int indices[VT];
	DeviceSharedToReg<NT, VT>(indices_shared, tid, indices);

	typedef typename std::iterator_traits<InputIt1>::value_type ValType;
	ValType reg[VT];
	DeviceTransferMergeValuesReg<NT, VT>(count, a_global, b_global, bStart,
		indices, tid, reg, sync);
	DeviceRegToGlobal<NT, VT>(count, reg, tid, dest_global, sync);
}

template<int NT, int VT, typename T>
MGPU_DEVICE void DeviceTransferMergeValuesReg(int count, const T* a_global,
	const T* b_global, int bStart, const int* indices, int tid, T* reg,
	bool sync) {

	int bOffset = (int)(b_global - a_global - bStart);

	if(count >= NT * VT) {
		#pragma unroll
		for(int i = 0; i < VT; ++i) {
			int gather = indices[i];
			if(gather >= bStart) gather += bOffset;
			reg[i] = a_global[gather];
		}
	} else {
		#pragma unroll
		for(int i = 0; i < VT; ++i) {
			int index = NT * i + tid;
			int gather = indices[i];
			if(gather >= bStart) gather += bOffset;
			if(index < count)
				reg[i] = a_global[gather];
		}
	}
	if(sync) __syncthreads();
}

template<int NT, int VT, typename T, typename OutputIt>
MGPU_DEVICE void DeviceTransferMergeValuesShared(int count, const T* a_global,
	const T* b_global, int bStart, const int* indices_shared, int tid,
	OutputIt dest_global, bool sync) {

	int indices[VT];
	DeviceSharedToReg<NT, VT>(indices_shared, tid, indices);

	T reg[VT];
	DeviceTransferMergeValuesReg<NT, VT>(count, a_global, b_global, bStart,
		indices, tid, reg, sync);
	DeviceRegToGlobal<NT, VT>(count, reg, tid, dest_global, sync);
}

} // namespace mgpu
