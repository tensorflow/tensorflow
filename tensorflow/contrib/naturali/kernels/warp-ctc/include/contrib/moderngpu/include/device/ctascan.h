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

#include "../mgpuenums.h"
#include "deviceutil.h"
#include "intrinsics.h"

namespace mgpu {

////////////////////////////////////////////////////////////////////////////////
// CTAReduce

template<int NT, typename Op = mgpu::plus<int> >
struct CTAReduce {
	typedef typename Op::first_argument_type T;
	enum { Size = NT, Capacity = NT };
	struct Storage { T shared[Capacity]; };

	MGPU_DEVICE static T Reduce(int tid, T x, Storage& storage, Op op = Op()) {
		storage.shared[tid] = x;
		__syncthreads();

		// Fold the data in half with each pass.
		#pragma unroll
		for(int destCount = NT / 2; destCount >= 1; destCount /= 2) {
			if(tid < destCount) {
				// Read from the right half and store to the left half.
				x = op(x, storage.shared[destCount + tid]);
				storage.shared[tid] = x;
			}
			__syncthreads();
		}
		T total = storage.shared[0];
		__syncthreads();
		return total;
	}
};

#if __CUDA_ARCH__ >= 300

template<int NT>
struct CTAReduce<NT, mgpu::plus<int> > {
	typedef mgpu::plus<int> Op;
	typedef int T;
	enum { Size = NT, Capacity = WARP_SIZE };
	struct Storage { int shared[Capacity]; };

	MGPU_DEVICE static int Reduce(int tid, int x, Storage& storage,
		Op op = Op()) {

		const int NumSections = WARP_SIZE;
		const int SecSize = NT / NumSections;
		int lane = (SecSize - 1) & tid;
		int sec = tid / SecSize;

		// In the first phase, threads cooperatively find the reduction within
		// their segment. The segments are SecSize threads (NT / WARP_SIZE)
		// wide.
		#pragma unroll
		for(int offset = 1; offset < SecSize; offset *= 2)
			x = shfl_add(x, offset, SecSize);

		// The last thread in each segment stores the local reduction to shared
		// memory.
		if(SecSize - 1 == lane) storage.shared[sec] = x;
		__syncthreads();

		// Reduce the totals of each input segment. The spine is WARP_SIZE
		// threads wide.
		if(tid < NumSections) {
			x = storage.shared[tid];
			#pragma unroll
			for(int offset = 1; offset < NumSections; offset *= 2)
				x = shfl_add(x, offset, NumSections);
			storage.shared[tid] = x;
		}
		__syncthreads();

		int reduction = storage.shared[NumSections - 1];
		__syncthreads();

		return reduction;
	}
};

template<int NT>
struct CTAReduce<NT, mgpu::maximum<int> > {
	typedef mgpu::maximum<int> Op;
	enum { Size = NT, Capacity = WARP_SIZE };
	struct Storage { int shared[Capacity]; };

	MGPU_DEVICE static int Reduce(int tid, int x, Storage& storage,
		Op op = Op()) {

		const int NumSections = WARP_SIZE;
		const int SecSize = NT / NumSections;
		int lane = (SecSize - 1) & tid;
		int sec = tid / SecSize;

		#pragma unroll
		for(int offset = 1; offset < SecSize; offset *= 2)
			x = shfl_max(x, offset, SecSize);

		if(SecSize - 1 == lane) storage.shared[sec] = x;
		__syncthreads();

		if(tid < NumSections) {
			x = storage.shared[tid];
			#pragma unroll
			for(int offset = 1; offset < NumSections; offset *= 2)
				x = shfl_max(x, offset, NumSections);
			storage.shared[tid] = x;
		}
		__syncthreads();

		int reduction = storage.shared[NumSections - 1];
		__syncthreads();

		return reduction;
	}
};

#endif // __CUDA_ARCH__ >= 300

////////////////////////////////////////////////////////////////////////////////
// CTAScan

template<int NT, typename Op = mgpu::plus<int> >
struct CTAScan {
	typedef typename Op::result_type T;
	enum { Size = NT, Capacity = 2 * NT + 1 };
	struct Storage { T shared[Capacity]; };

	MGPU_DEVICE static T Scan(int tid, T x, Storage& storage, T* total,
		MgpuScanType type = MgpuScanTypeExc, T identity = (T)0, Op op = Op()) {

		storage.shared[tid] = x;
		int first = 0;
		__syncthreads();

		#pragma unroll
		for(int offset = 1; offset < NT; offset += offset) {
			if(tid >= offset)
				x = op(storage.shared[first + tid - offset], x);
			first = NT - first;
			storage.shared[first + tid] = x;
			__syncthreads();
		}
		*total = storage.shared[first + NT - 1];

		if(MgpuScanTypeExc == type)
			x = tid ? storage.shared[first + tid - 1] : identity;

		__syncthreads();
		return x;
	}
	MGPU_DEVICE static T Scan(int tid, T x, Storage& storage) {
		T total;
		return Scan(tid, x, storage, &total, MgpuScanTypeExc, (T)0, Op());
	}
};

////////////////////////////////////////////////////////////////////////////////
// Special partial specialization for CTAScan<NT, ScanOpAdd> on Kepler.
// This uses the shfl intrinsic to reduce scan latency.

#if __CUDA_ARCH__ >= 300

template<int NT>
struct CTAScan<NT, mgpu::plus<int> > {
	typedef mgpu::plus<int> Op;
	enum { Size = NT, NumSegments = WARP_SIZE, SegSize = NT / NumSegments };
	enum { Capacity = NumSegments + 1 };
	struct Storage { int shared[Capacity + 1]; };

	MGPU_DEVICE static int Scan(int tid, int x, Storage& storage, int* total,
		MgpuScanType type = MgpuScanTypeExc, int identity = 0, Op op = Op()) {

		// Define WARP_SIZE segments that are NT / WARP_SIZE large.
		// Each warp makes log(SegSize) shfl_add calls.
		// The spine makes log(WARP_SIZE) shfl_add calls.
		int lane = (SegSize - 1) & tid;
		int segment = tid / SegSize;

		// Scan each segment using shfl_add.
		int scan = x;
		#pragma unroll
		for(int offset = 1; offset < SegSize; offset *= 2)
			scan = shfl_add(scan, offset, SegSize);

		// Store the reduction (last element) of each segment into storage.
		if(SegSize - 1 == lane) storage.shared[segment] = scan;
		__syncthreads();

		// Warp 0 does a full shfl warp scan on the partials. The total is
		// stored to shared[NumSegments]. (NumSegments = WARP_SIZE)
		if(tid < NumSegments) {
			int y = storage.shared[tid];
			int scan = y;
			#pragma unroll
			for(int offset = 1; offset < NumSegments; offset *= 2)
				scan = shfl_add(scan, offset, NumSegments);
			storage.shared[tid] = scan - y;
			if(NumSegments - 1 == tid) storage.shared[NumSegments] = scan;
		}
		__syncthreads();

		// Add the scanned partials back in and convert to exclusive scan.
		scan += storage.shared[segment];
		if(MgpuScanTypeExc == type) {
			scan -= x;
			if(identity && !tid) scan = identity;
		}
		*total = storage.shared[NumSegments];
		__syncthreads();

		return scan;
	}
	MGPU_DEVICE static int Scan(int tid, int x, Storage& storage) {
		int total;
		return Scan(tid, x, storage, &total, MgpuScanTypeExc, 0);
	}
};

#endif // __CUDA_ARCH__ >= 300

////////////////////////////////////////////////////////////////////////////////
// CTABinaryScan

template<int NT>
MGPU_DEVICE int CTABinaryScan(int tid, bool x, int* shared, int* total) {
	const int NumWarps = NT / WARP_SIZE;
	int warp = tid / WARP_SIZE;
	int lane = (WARP_SIZE - 1);

	// Store the bit totals for each warp.
	uint bits = __ballot(x);
	shared[warp] = popc(bits);
	__syncthreads();

#if __CUDA_ARCH__ >= 300
	if(tid < NumWarps) {
		int x = shared[tid];
		int scan = x;
		#pragma unroll
		for(int offset = 1; offset < NumWarps; offset *= 2)
			scan = shfl_add(scan, offset, NumWarps);
		shared[tid] = scan - x;
	}
	__syncthreads();

#else
	// Thread 0 scans warp totals.
	if(!tid) {
		int scan = 0;
		#pragma unroll
		for(int i = 0; i < NumWarps; ++i) {
			int y = shared[i];
			shared[i] = scan;
			scan += y;
		}
		shared[NumWarps] = scan;
	}
	__syncthreads();

#endif // __CUDA_ARCH__ >= 300

	// Add the warp scan back into the partials.
	int scan = shared[warp] + __popc(bfe(bits, 0, lane));
	*total = shared[NumWarps];
	__syncthreads();
	return scan;
}

} // namespace mgpu
