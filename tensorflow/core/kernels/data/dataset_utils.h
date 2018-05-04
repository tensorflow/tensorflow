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

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/captured_function.h"
#include "tensorflow/core/kernels/data/dataset.h"

namespace tensorflow {

namespace dataset {

Status MakeIteratorFromInputElement(
    IteratorContext* ctx, const std::vector<Tensor>& input_element,
    int64 thread_index, CapturedFunction* captured_func, StringPiece prefix,
    std::unique_ptr<IteratorBase>* out_iterator);

IteratorContext MakeIteratorContext(OpKernelContext* ctx);

}  // namespace dataset

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_DATASET_UTILS_H_
