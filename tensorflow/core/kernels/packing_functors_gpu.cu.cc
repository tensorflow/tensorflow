/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/packing_functors.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
__global__ void PackedSequenceAlignmentKernel(
	const int64 batch_size, 
	T const * const sequence_lengths, 
	T* const alignments,
	T* const batch_sizes) {		
		if(blockIdx.x== 0 && threadIdx.x ==0){
			T idx = 0;
			T pos = batch_size-1;
			const T maxlen = sequence_lengths[0];
			T current_len = sequence_lengths[pos];
			for(T i=0; i<maxlen; i++){
				while(current_len <= i){
					pos--;
					current_len = sequence_lengths[pos];
				}
				alignments[i] = idx;
				idx += pos+1;
				batch_sizes[i] =pos+1;
			}
		}
	}
	
template <typename T, typename Index>
__global__ void PackSequenceKernel(
	const int64 total_count,
	const int64 batch_size,
	const int64 dim,
	T const * const sequence, 
	Index const * const  alignments, 
	Index const * const batch_sizes, 
	T* const packed) {		
		CUDA_1D_KERNEL_LOOP(i, total_count) {
			const int64 x = i / (batch_size*dim);
			const int64 y = (i / dim) % batch_size;
			const int64 z = i % dim;
			
			const Index current_batch_size = batch_sizes[x];
			if(y < current_batch_size){
				const Index current_aligment = alignments[x];
				const int64 target = (current_aligment*dim) + (y*dim) + z;
				packed[target] = sequence[i];
			}
		}
	}

template <typename T, typename Index>
__global__ void UnpackSequenceKernel(
	const int64 total_count,
	const int64 batch_size,
	const int64 dim,
	T const * const packed, 
	Index const * const alignments, 
	Index const * const batch_sizes, 
	T* const sequence) {		
		CUDA_1D_KERNEL_LOOP(i, total_count) {
			const int64 x = i / (batch_size*dim);
			const int64 y = (i / dim) % batch_size;
			const int64 z = i % dim;
			
			const Index current_batch_size = batch_sizes[x];
			if(y < current_batch_size){
				const Index current_aligment = alignments[x];
				const int64 source = (current_aligment*dim) + (y*dim) + z;
				sequence[i] = packed[source];
			}else{
				sequence[i] = 0;
			}
		}
	}

	
namespace functor {

template <typename T>
struct PackedSequenceAlignmentFunctor<GPUDevice, T> {
  Status operator()(
	const GPUDevice& d, 
   typename TTypes<T>::ConstFlat Tsequence_lengths,
   typename TTypes<T>::Flat Talignments,
   typename TTypes<T>::Flat Tbatch_sizes){     
	PackedSequenceAlignmentKernel<T>
        <<<1, 1, 0, d.stream()>>>(
            Tsequence_lengths.dimension(0),
			Tsequence_lengths.data(), 
			Talignments.data(), 
			Tbatch_sizes.data());
    return Status::OK();
  }
};

template <typename T, typename Index>
struct PackSequenceFunctor<GPUDevice, T, Index> {
  Status operator()(
	const GPUDevice& d, 
   typename TTypes<T,3>::ConstTensor Tsequence,
   typename TTypes<Index>::ConstFlat Talignments,
   typename TTypes<Index>::ConstFlat Tbatch_sizes,
   typename TTypes<T,2>::Tensor Tpacked
   ){     
    const int total_count = Tsequence.size();
    const int64 batch_size = Tsequence.dimension(1);
    const int64 dim = Tsequence.dimension(2);
    CudaLaunchConfig config = GetCudaLaunchConfig(total_count, d);
    PackSequenceKernel<T, Index> 
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
            total_count,
			batch_size,
            dim,
			Tsequence.data(), 
			Talignments.data(), 
			Tbatch_sizes.data(), 
			Tpacked.data());
    return Status::OK();
  }
};

template <typename T, typename Index>
struct UnpackSequenceFunctor<GPUDevice, T, Index> {
  Status operator()(
	const GPUDevice& d, 
   typename TTypes<T,2>::ConstTensor Tpacked,
   typename TTypes<Index>::ConstFlat Talignments,
   typename TTypes<Index>::ConstFlat Tbatch_sizes,
   typename TTypes<T,3>::Tensor Tsequence
   ){
	const int total_count = Tsequence.size();
    const int64 batch_size = Tsequence.dimension(1);
    const int64 dim = Tsequence.dimension(2);
	CudaLaunchConfig config = GetCudaLaunchConfig(total_count, d);
    UnpackSequenceKernel<T, Index> 
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
            total_count,
			batch_size,
			dim,
			Tpacked.data(), 
			Talignments.data(), 
			Tbatch_sizes.data(),
			Tsequence.data());
    return Status::OK();
  }
};

}  // namespace functor


#define DEFINE_GPU_SPECS_T(T) \
  template struct functor::PackedSequenceAlignmentFunctor<GPUDevice, T>;

DEFINE_GPU_SPECS_T(int8)
DEFINE_GPU_SPECS_T(int16)
DEFINE_GPU_SPECS_T(int32)
DEFINE_GPU_SPECS_T(int64)

#undef DEFINE_GPU_SPECS_T


#define DEFINE_GPU_SPECS_T_Index(T, Index) \
  template struct functor::PackSequenceFunctor<GPUDevice, T, Index>;
#define DEFINE_GPU_SPECS_T(T) \
  DEFINE_GPU_SPECS_T_Index(T, int32) \
  DEFINE_GPU_SPECS_T_Index(T, int64)

DEFINE_GPU_SPECS_T(int32)
DEFINE_GPU_SPECS_T(int64)
DEFINE_GPU_SPECS_T(float)
DEFINE_GPU_SPECS_T(double)

#undef DEFINE_GPU_SPECS_T
#undef DEFINE_GPU_SPECS_T_Index

#define DEFINE_GPU_SPECS_T_Index(T, Index) \
  template struct functor::UnpackSequenceFunctor<GPUDevice, T, Index>;
#define DEFINE_GPU_SPECS_T(T) \
  DEFINE_GPU_SPECS_T_Index(T, int32) \
  DEFINE_GPU_SPECS_T_Index(T, int64)

DEFINE_GPU_SPECS_T(int32)
DEFINE_GPU_SPECS_T(int64)
DEFINE_GPU_SPECS_T(float)
DEFINE_GPU_SPECS_T(double)

#undef DEFINE_GPU_SPECS_T
#undef DEFINE_GPU_SPECS_T_Index


}  // namespace tensorflow

#endif  // GOOGLE_CUDA
