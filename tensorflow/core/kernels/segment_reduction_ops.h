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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_SEGMENT_REDUCTION_OPS_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_SEGMENT_REDUCTION_OPS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {

class OpKernelContext;

namespace functor {
// BaseFunctor for definition of UnsorteSegmentReductionOp
// for usage without templates.
template <typename Device, typename T, typename Index>
struct UnsortedSegmentBaseFunctor{
  virtual ~UnsortedSegmentBaseFunctor(){}
  virtual void operator()(OpKernelContext* ctx, const Device& d,
                  const Index output_rows, const TensorShape& segment_ids_shape,
                  typename TTypes<Index>::ConstFlat segment_ids,
                  const Index data_size, const T* data,
                  typename TTypes<T, 2>::Tensor output){};
};

// Functor for UnsortedSegmentSumOp.
// 'output_rows': the number of output segments (unique segment ids in
//                'segment_ids').
// 'segment_ids_shape': shape of 'segment_ids' tensor.
// 'segment_ids': unsorted map from input to output segment ids at which to
//                perform segment sum operation.
// 'data_size': size of input data tensor.
// 'data': input data tensor.
// 'output': output reshaped to {output_rows, output.size/output_rows}
template <typename Device, typename T, typename Index>
struct UnsortedSegmentSumFunctor: public UnsortedSegmentBaseFunctor<Device, T, Index> {
  void operator()(OpKernelContext* ctx, const Device& d,
                  const Index output_rows, const TensorShape& segment_ids_shape,
                  typename TTypes<Index>::ConstFlat segment_ids,
                  const Index data_size, const T* data,
                  typename TTypes<T, 2>::Tensor output);
};

// Functor for UnsortedSegmentMaxOp.
// 'output_rows': the number of output segments (unique segment ids in
//                'segment_ids').
// 'segment_ids_shape': shape of 'segment_ids' tensor.
// 'segment_ids': unsorted map from input to output segment ids at which to
//                perform segment sum operation.
// 'data_size': size of input data tensor.
// 'data': input data tensor.
// 'output': output reshaped to {output_rows, output.size/output_rows}
template <typename Device, typename T, typename Index>
struct UnsortedSegmentMaxFunctor: public UnsortedSegmentBaseFunctor<Device, T, Index> {
  void operator()(OpKernelContext* ctx, const Device& d,
                  const Index output_rows, const TensorShape& segment_ids_shape,
                  typename TTypes<Index>::ConstFlat segment_ids,
                  const Index data_size, const T* data,
                  typename TTypes<T, 2>::Tensor output);
};
}  // namespace functor
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_SEGMENT_REDUCTION_OPS_H_
