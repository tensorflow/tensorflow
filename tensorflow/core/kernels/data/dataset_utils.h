/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_DATASET_UTILS_H_
#define TENSORFLOW_CORE_KERNELS_DATA_DATASET_UTILS_H_

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/captured_function.h"

namespace tensorflow {
namespace data {

// This method is used to determine whether we can short-circuit the evaluation
// of the user-defined function `func`. Short-circuting is possible if every
// function output corresponds to one of its inputs (e.g. `f(x) = x`, `f(x,y) =
// (y,x)`, or `f(x) = (x,x)`).
//
// If short-circuiting is possible, the method stores the mapping from output
// indices to input indices in `indices`. Otherwise, `indices` will be empty.
//
// Returns non-ok status if analysis of the function fails.
//
// TODO(jsimsa): Extend this to support constants as well.
Status ComputeShortCircuitIndices(OpKernelContext* ctx,
                                  const NameAttrList& func,
                                  std::vector<int>* indices);

// Given a vector that maps output indices to input indices, return a vector
// that identifies for which output indices can we move the input (assuming
// output indices are processed left to right).
std::vector<bool> ComputeMoveVector(const std::vector<int>& indices);

Status MakeIteratorFromInputElement(
    IteratorContext* ctx, const std::vector<Tensor>& input_element,
    int64 thread_index, const InstantiatedCapturedFunction& inst_captured_func,
    StringPiece prefix, std::unique_ptr<IteratorBase>* out_iterator);

// Returns Status::OK() if `expected` and `received` types match,
// errors::InvalidArgument otherwise.
Status VerifyTypesMatch(const DataTypeVector& expected,
                        const DataTypeVector& received);

// Returns Status::OK() if `expected` and `received` shapes are compatible,
// errors::InvalidArgument otherwise.
Status VerifyShapesCompatible(const std::vector<PartialTensorShape>& expected,
                              const std::vector<PartialTensorShape>& received);

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_DATASET_UTILS_H_
