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

#ifndef TENSORFLOW_CORE_KERNELS_PADDING_FIFO_QUEUE_H_
#define TENSORFLOW_CORE_KERNELS_PADDING_FIFO_QUEUE_H_

#include <deque>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/fifo_queue.h"
#include "tensorflow/core/kernels/typed_queue.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class PaddingFIFOQueue : public FIFOQueue {
 public:
  PaddingFIFOQueue(int32 capacity, const DataTypeVector& component_dtypes,
                   const std::vector<PartialTensorShape>& component_shapes,
                   const string& name);

  Status Initialize() override;

  // Implementations of QueueInterface methods --------------------------------

  void TryDequeueMany(int num_elements, OpKernelContext* ctx,
                      bool allow_small_batch,
                      CallbackWithTuple callback) override;
  Status MatchesNodeDef(const NodeDef& node_def) override;

 protected:
  Status ValidateManyTuple(const Tuple& tuple) override;
  Status ValidateTuple(const Tuple& tuple) override;
  Status CompatibleNodeDefShapes(const NodeDef& node_def) const;

  // Convert a list of PartialTensorShape to a list of
  // TensorShape.
  // Any unknown dimension sizes are converted to 0.
  // REQUIRED: All the input shapes have well defined rank.
  static std::vector<TensorShape> ConvertShapesPartialDimensionsToZero(
      const gtl::ArraySlice<PartialTensorShape>& partial_shapes);

  // Sets the values in the given element to zero.
  static Status SetElementZero(Tensor* element);

  // Copies element into the index^th slice (in the first dimension)
  // of parent.  Allows for the parent's slice to have a larger size
  // than the element, and copies the element into the upper left hand
  // corner of the slice.
  static Status CopyElementToLargerSlice(const Tensor& element, Tensor* parent,
                                         int index);

  std::vector<PartialTensorShape> partial_shapes_;

 private:
  ~PaddingFIFOQueue() override {}

  static Status GetElementComponent(const PaddingFIFOQueue::Tuple& tuple,
                                    int component, OpKernelContext* ctx,
                                    Tensor* out_tensor);

  static Status IsSameSizeExceptZerosInFirst(const TensorShape& first,
                                             const TensorShape& second);

  TF_DISALLOW_COPY_AND_ASSIGN(PaddingFIFOQueue);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_PADDING_FIFO_QUEUE_H_
