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

#include "mgpuenums.h"
#include "device/deviceutil.h"

namespace mgpu {

////////////////////////////////////////////////////////////////////////////////
// device/loadstore.cuh

// For 0 <= i < VT:
//		index = NT * i + tid;
//		reg[i] = data[index];
// Synchronize after load.
template<int NT, int VT, typename InputIt, typename T>
MGPU_DEVICE void DeviceSharedToReg(InputIt data, int tid, T* reg,
	bool sync = true);

// For 0 <= i < VT:
//		index = NT * i + tid;
//		if(index < count) reg[i] = data[index];
// No synchronize after load.
template<int NT, int VT, typename InputIt, typename T>
MGPU_DEVICE void DeviceGlobalToReg(int count, InputIt data, int tid,
	T* reg, bool sync = false);

template<int NT, int VT, typename InputIt, typename T>
MGPU_DEVICE void DeviceGlobalToRegDefault(int count, InputIt data, int tid,
	T* reg, T init, bool sync = false);

// For 0 <= i < VT:
//		index = NT * i + tid;
//		if(index < count) reg[i] = data[index];
// No synchronize after load.
template<int NT, int VT0, int VT1, typename InputIt, typename T>
MGPU_DEVICE void DeviceGlobalToReg(int count, InputIt data, int tid,
	T* reg, bool sync = false);

// For 0 <= i < VT:
//		index = NT * i + tid;
//		if(index < count) reg[i] = data[index];
// No synchronize after load.
template<int NT, int VT0, int VT1, typename InputIt, typename T>
MGPU_DEVICE void DeviceGlobalToRegDefault2(int count, InputIt data, int tid,
	T* reg, T init, bool sync = false);

// For 0 <= i < VT:
//		index = NT * i + tid;
//		if(index < count) reg[i] = data[index];
// No synchronize after load.
// No optimized code path for count < NV (smaller generated code).
template<int NT, int VT, typename InputIt, typename T>
MGPU_DEVICE void DeviceGlobalToRegLoop(int count, InputIt data, int tid,
	T* reg, bool sync = false);


// For 0 <= i < VT:
//		index = VT * tid + i.
//		if(index < count) reg[i] = data[index];
// No synchronize after load.
template<int NT, int VT, typename InputIt, typename T>
MGPU_DEVICE void DeviceGlobalToThread(int count, InputIt data, int tid,
	T* reg);

template<int NT, int VT, typename InputIt, typename T>
MGPU_DEVICE void DeviceGlobalToThreadDefault(int count, InputIt data, int tid,
	T* reg, T init);

// For 0 <= i < VT:
//		index = NT * i + tid;
//		if(index < count) data[index] = reg[i];
// Synchronize after load.
template<int NT, int VT, typename OutputIt, typename T>
MGPU_DEVICE void DeviceRegToShared(const T* reg, int tid, OutputIt dest,
	bool sync = true);

// For 0 <= i < VT:
//		index = NT * i + tid;
//		if(index < count) data[index] = reg[i];
// No synchronize after load.
template<int NT, int VT, typename OutputIt, typename T>
MGPU_DEVICE void DeviceRegToGlobal(int count, const T* reg, int tid,
	OutputIt dest, bool sync = false);

// For 0 <= index < count:
//		dest[index] = source[index];
// This function is intended to replace DeviceGlobalToShared in cases where
// count is much less than NT * VT.
template<int NT, typename InputIt, typename OutputIt>
MGPU_DEVICE void DeviceMemToMemLoop(int count, InputIt source, int tid,
	OutputIt dest, bool sync = true);

// For 0 <= index < count:
//		dest[index] = source[index];
// Synchronize after store.
template<int NT, int VT, typename T, typename OutputIt>
MGPU_DEVICE void DeviceSharedToGlobal(int count, const T* source, int tid,
	OutputIt dest, bool sync = true);

// For 0 <= index < count:
//		dest[index] = source[index];
// Synchronize after store.
template<int NT, int VT, typename InputIt, typename T>
MGPU_DEVICE void DeviceGlobalToShared(int count, InputIt source, int tid,
	T* dest, bool sync = true);

template<int NT, int VT0, int VT1, typename InputIt, typename T>
MGPU_DEVICE void DeviceGlobalToShared2(int count, InputIt source, int tid,
	T* dest, bool sync = true);

// For 0 <= index < count:
//		dest[index] = source[index];
// Synchronize after store.
// No optimized code path for count < NV (smaller generated code).
template<int NT, int VT, typename InputIt, typename T>
MGPU_DEVICE void DeviceGlobalToSharedLoop(int count, InputIt source, int tid,
	T* dest, bool sync = true);

template<int NT, int VT, typename InputIt, typename T>
MGPU_DEVICE void DeviceGlobalToSharedDefault(int count, InputIt source, int tid,
	T* dest, T init, bool sync = true);

template<int NT, int VT0, int VT1, typename InputIt, typename T>
MGPU_DEVICE void DeviceGlobalToSharedDefault2(int count, InputIt source,
	int tid, T* dest, T init, bool sync = true);

// For 0 <= index < count:
//		dest[index] = source[index];
// No synchronize.
template<int NT, int VT, typename InputIt, typename OutputIt>
MGPU_DEVICE void DeviceGlobalToGlobal(int count, InputIt source, int tid,
	OutputIt dest, bool sync = false);

// Transponse VT elements in NT threads (x) into thread-order registers (y)
// using only NT * VT / 2 elements of shared memory.
template<int NT, int VT, typename T>
MGPU_DEVICE void HalfSmemTranspose(const T* x, int tid, T* shared, T* y);

// For 0 <= i < VT:
//		index = NT * i + tid;
//		if(index < count)
//			gather = indices[index];
//			reg[i] = data[gather];
// Synchronize after load.
template<int NT, int VT, typename InputIt, typename T>
MGPU_DEVICE void DeviceGather(int count, InputIt data, int indices[VT],
	int tid, T* reg, bool sync = true);

template<int NT, int VT, typename InputIt, typename T>
MGPU_DEVICE void DeviceGatherDefault(int count, InputIt data, int indices[VT],
	int tid, T* reg, T identity, bool sync = true);

// For 0 <= i < VT:
//		index = NT * i + tid;
//		if(index < count)
//			scatter = indices[index];
//			data[scatter] = reg[i];
// Synchronize after store.
template<int NT, int VT, typename T, typename OutputIt>
MGPU_DEVICE void DeviceScatter(int count, const T* reg, int tid,
	int indices[VT], OutputIt data, bool sync = true);

// For 0 <= i < VT:
//		shared[VT * tid + i] = threadReg[i];
// Synchronize after store.
// Note this function moves data in THREAD ORDER.
// (DeviceRegToShared moves data in STRIDED ORDER).
template<int VT, typename T>
MGPU_DEVICE void DeviceThreadToShared(const T* threadReg, int tid, T* shared,
	bool sync = true);

// For 0 <= i < VT:
//		threadReg[i] = shared[VT * tid + i];
// Synchronize after load.
// Note this function moves data in THREAD ORDER.
// (DeviceSharedToReg moves data in STRIDED ORDER).
template<int VT, typename T>
MGPU_DEVICE void DeviceSharedToThread(const T* shared, int tid, T* threadReg,
	bool sync = true);

// For 0 <= index < aCount:
//		shared[index] = a_global[index];
// For 0 <= index < bCount:
//		shared[aCount + index] = b_global[index];
// VT0 is the lower-bound for predication-free execution:
//		If count >= NT * VT0, a predication-free branch is taken.
// VT1 is the upper-bound for loads:
//		NT * VT1 must >= aCount + bCount.

template<int NT, int VT0, int VT1, typename T>
MGPU_DEVICE void DeviceLoad2ToReg(const T* a_global, int aCount,
	const T* b_global, int bCount, int tid, T* reg, bool sync = false);

template<int NT, int VT0, int VT1, typename T>
MGPU_DEVICE void DeviceLoad2ToShared(const T* a_global, int aCount,
	const T* b_global, int bCount, int tid, T* shared, bool sync = true);

template<int NT, int VT0, int VT1, typename InputIt1, typename InputIt2,
	typename T>
MGPU_DEVICE void DeviceLoad2ToReg(InputIt1 a_global, int aCount,
	InputIt2 b_global, int bCount, int tid, T* reg, bool sync = false);

template<int NT, int VT0, int VT1, typename InputIt1, typename InputIt2,
	typename T>
MGPU_DEVICE void DeviceLoad2ToShared(InputIt1 a_global, int aCount,
	InputIt2 b_global, int bCount, int tid, T* shared, bool sync = true);

// For 0 <= i < VT
//		index = NT * i + tid;
//		if(index < count)
//			gather = indices_shared[index];
//			dest_global[index] = data_global[gather];
// Synchronize after load.
template<int NT, int VT, typename InputIt, typename OutputIt>
MGPU_DEVICE void DeviceGatherGlobalToGlobal(int count, InputIt data_global,
	const int* indices_shared, int tid, OutputIt dest_global,
	bool sync = true);

// For 0 <= i < VT
//		index = NT * i + tid
//		if(index < count)
//			gather = indices[index];
//			if(gather < aCount) data = a_global[gather];
//			else data = b_global[gather - aCount];
//			dest_global[index] = data;
// Synchronize after load.
template<int NT, int VT, typename InputIt1, typename InputIt2,
	typename T>
MGPU_DEVICE void DeviceTransferMergeValuesReg(int count, InputIt1 a_global,
	InputIt2 b_global, int bStart, const int* indices, int tid,
	T* reg, bool sync = false);

template<int NT, int VT, typename InputIt1, typename InputIt2,
	typename OutputIt>
MGPU_DEVICE void DeviceTransferMergeValuesShared(int count, InputIt1 a_global,
	InputIt2 b_global, int bStart, const int* indices_shared, int tid,
	OutputIt dest_global, bool sync = true);

template<int NT, int VT, typename T>
MGPU_DEVICE void DeviceTransferMergeValuesReg(int count, const T* a_global,
	const T* b_global, int bStart, const int* indices, int tid,
	T* reg, bool sync = false);

template<int NT, int VT, typename T, typename OutputIt>
MGPU_DEVICE void DeviceTransferMergeValuesShared(int count, const T* a_global,
	const T* b_global, int bStart, const int* indices_shared, int tid,
	OutputIt dest_global, bool sync = true);



} // namespace mgpu


#include "device/loadstore.h"
#include "device/ctasegscan.h"
