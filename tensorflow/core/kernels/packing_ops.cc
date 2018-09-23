/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/state_ops.cc.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/packing_functors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/util.h"

#ifdef TENSORFLOW_USE_SYCL
#include "tensorflow/core/common_runtime/sycl/sycl_util.h"
#endif  // TENSORFLOW_USE_SYCL

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;


Status ExtractPackedSequenceAlignmentInfo(OpKernelContext* context,
							   const Tensor** sequence_lengths) {
	TF_RETURN_IF_ERROR(context->input("sequence_lengths", sequence_lengths));
	return Status::OK();
}

template <typename T>
class PackedSequenceAlignmentOp<GPUDevice, T> : public OpKernel {
 public:
  explicit PackedSequenceAlignmentOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
	const Tensor* sequence_lengths;
    OP_REQUIRES_OK(context, ExtractPackedSequenceAlignmentInfo(context, &sequence_lengths));
	
	const T max_sequence_length = sequence_lengths->vec<T>()(0);
    Tensor* alignments_t = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, {max_sequence_length}, &alignments_t));
    Tensor* batch_sizes_t = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, {max_sequence_length}, &batch_sizes_t));
	
	auto func = functor::PackedSequenceAlignmentFunctor<GPUDevice, T>();
	OP_REQUIRES_OK(context, func(
		context->eigen_device<GPUDevice>(),
		sequence_lengths->flat<T>(),
		alignments_t->flat<T>(),
		batch_sizes_t->flat<T>()));
	}
};

#define REGISTER_GPU(T)                                    \
  REGISTER_KERNEL_BUILDER(Name("PackedSequenceAlignment")       \
                              .Device(DEVICE_GPU) \
							  .HostMemory("sequence_lengths")          \
                              .TypeConstraint<T>("T"),      \
                          PackedSequenceAlignmentOp<GPUDevice, T>);
REGISTER_GPU(int8)					  
REGISTER_GPU(int16)					  
REGISTER_GPU(int32)					  
REGISTER_GPU(int64)					  
#undef REGISTER_GPU


Status ExtractSequenceGatherScatterIndicesInfo(OpKernelContext* context,
							    const Tensor** total_length, const Tensor** sequence_lengths, const Tensor** batch_order) {
	TF_RETURN_IF_ERROR(context->input("total_length", total_length));
	TF_RETURN_IF_ERROR(context->input("sequence_lengths", sequence_lengths));
	TF_RETURN_IF_ERROR(context->input("batch_order", batch_order));
	return Status::OK();
}

template <typename T>
class SequenceGatherScatterIndicesOp<GPUDevice, T> : public OpKernel {
 public:
  explicit SequenceGatherScatterIndicesOp(OpKernelConstruction* context)
      : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("time_major", &time_major_));
	}

  void Compute(OpKernelContext* context) override {
	const Tensor* total_length;
	const Tensor* sequence_lengths;
	const Tensor* batch_order;
    OP_REQUIRES_OK(context, ExtractSequenceGatherScatterIndicesInfo(context, &total_length, &sequence_lengths, &batch_order));
	
	const T actual_total_length = total_length->scalar<T>()(0);
    Tensor* gather_scatter_indices = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, {actual_total_length, 2}, &gather_scatter_indices));
	
	auto func = functor::SequenceGatherScatterIndicesFunctor<GPUDevice, T>();

	OP_REQUIRES_OK(context, func(
		context->eigen_device<GPUDevice>(),
		sequence_lengths->flat<T>(),
		batch_order->flat<T>(),
		gather_scatter_indices->flat<T>(),
		time_major_));
	}
  private:
    bool time_major_;
};

#define REGISTER_GPU(T)                                    \
  REGISTER_KERNEL_BUILDER(Name("SequenceGatherScatterIndices")       \
                              .Device(DEVICE_GPU) \
							  .HostMemory("total_length")          \
                              .TypeConstraint<T>("T"),      \
                          SequenceGatherScatterIndicesOp<GPUDevice, T>);
REGISTER_GPU(int8)					  
REGISTER_GPU(int16)					  
REGISTER_GPU(int32)					  
REGISTER_GPU(int64)					  
#undef REGISTER_GPU

Status ExtractPackSequenceInfo(OpKernelContext* context,
							   const Tensor** sequence,
							   const Tensor** alignments,
							   const Tensor** batch_sizes) {
	TF_RETURN_IF_ERROR(context->input("sequence", sequence));
	TF_RETURN_IF_ERROR(context->input("alignments", alignments));
	TF_RETURN_IF_ERROR(context->input("batch_sizes", batch_sizes));
	return Status::OK();
}


template <typename T, typename Index>
class PackSequenceOp<GPUDevice, T, Index> : public OpKernel {
 public:
  explicit PackSequenceOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
	const Tensor* sequence;
	const Tensor* alignments;
	const Tensor* batch_sizes;
    OP_REQUIRES_OK(context, ExtractPackSequenceInfo(
		context, 
		&sequence,
		&alignments,
		&batch_sizes));
	
	auto sequence_length = alignments->dim_size(0);
	auto dim = sequence->dim_size(2);
	
	Index output_length = (
		alignments->vec<Index>()(sequence_length-1)
		+ batch_sizes->vec<Index>()(sequence_length-1));
	
    Tensor* packed = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, {output_length, dim}, &packed));

	
	auto func = functor::PackSequenceFunctor<GPUDevice, T, Index>();
	OP_REQUIRES_OK(context, func(
		context->eigen_device<GPUDevice>(),
		sequence->tensor<T,3>(),
		alignments->flat<Index>(),
		batch_sizes->flat<Index>(),
		packed->tensor<T,2>()));
	}
};

#define REGISTER_GPU_T_Index(T, Index)                                    \
  REGISTER_KERNEL_BUILDER(Name("PackSequence")       \
                              .Device(DEVICE_GPU)          \
							  .HostMemory("alignments")  \
							  .HostMemory("batch_sizes")  \
                              .TypeConstraint<T>("T")      \
                              .TypeConstraint<Index>("Index"),      \
                          PackSequenceOp<GPUDevice, T, Index>);
#define REGISTER_GPU_T(T) \
	REGISTER_GPU_T_Index(T, int32) \
	REGISTER_GPU_T_Index(T, int64)
						  
REGISTER_GPU_T(int32)					  
REGISTER_GPU_T(int64)	  
REGISTER_GPU_T(float)					  
REGISTER_GPU_T(double)
				  
#undef REGISTER_GPU_T				  
#undef REGISTER_GPU_T_Index


Status ExtractUnpackSequenceInfo(OpKernelContext* context,
							   const Tensor** packed,
							   const Tensor** alignments,
							   const Tensor** batch_sizes) {
	TF_RETURN_IF_ERROR(context->input("packed", packed));
	TF_RETURN_IF_ERROR(context->input("alignments", alignments));
	TF_RETURN_IF_ERROR(context->input("batch_sizes", batch_sizes));
	return Status::OK();
}


template <typename T, typename Index>
class UnpackSequenceOp<GPUDevice, T, Index> : public OpKernel {
 public:
  explicit UnpackSequenceOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
	const Tensor* packed;
	const Tensor* alignments;
	const Tensor* batch_sizes;
    OP_REQUIRES_OK(context, ExtractUnpackSequenceInfo(
		context, 
		&packed,
		&alignments,
		&batch_sizes));
	
	auto sequence_length = alignments->dim_size(0);
	auto batch_size = batch_sizes->vec<Index>()(0);
	auto dim = packed->dim_size(1);
	
    Tensor* sequence = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, {sequence_length, batch_size, dim}, &sequence));
	
	auto func = functor::UnpackSequenceFunctor<GPUDevice, T, Index>();
	OP_REQUIRES_OK(context, func(
		context->eigen_device<GPUDevice>(),
		packed->tensor<T,2>(),
		alignments->flat<Index>(),
		batch_sizes->flat<Index>(),
		sequence->tensor<T,3>()));
	}
};

#define REGISTER_GPU_T_Index(T, Index)                                    \
  REGISTER_KERNEL_BUILDER(Name("UnpackSequence")       \
                              .Device(DEVICE_GPU)          \
							  .HostMemory("alignments")  \
							  .HostMemory("batch_sizes")  \
                              .TypeConstraint<T>("T")      \
                              .TypeConstraint<Index>("Index"),      \
                          UnpackSequenceOp<GPUDevice, T, Index>);
#define REGISTER_GPU_T(T) \
	REGISTER_GPU_T_Index(T, int32) \
	REGISTER_GPU_T_Index(T, int64)
						  
REGISTER_GPU_T(int32)					  
REGISTER_GPU_T(int64)	  
REGISTER_GPU_T(float)					  
REGISTER_GPU_T(double)
				  
#undef REGISTER_GPU_T				  
#undef REGISTER_GPU_T_Index

}  // namespace tensorflow
