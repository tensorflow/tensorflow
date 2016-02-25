/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_KERNELS_SAVE_RESTORE_TENSOR_H_
#define TENSORFLOW_KERNELS_SAVE_RESTORE_TENSOR_H_

#include "tensorflow/core/util/tensor_slice_reader.h"
#include "tensorflow/core/util/tensor_slice_writer.h"

namespace tensorflow {

class OpKernelContext;

// Save input tensors in *context to a writer built from builder_func().
// context must have the following inputs:
//  0: a single element string tensor that contains the file name.
//  1: names for the remaining tensors
// If save_slices is true:
//  2: shape and slice specifications.
//  rest: tensors to save
void SaveTensors(
    OpKernelContext* context,
    checkpoint::TensorSliceWriter::CreateBuilderFunction builder_func,
    bool save_slices);

// Reads a tensor from the reader built from open_func() and produces it as
// context->output(0).  "preferred_shard" is the same the TensorSliceReader
// preferred_shard parameter.
//
// context must have the following inputs:
//  0: a single element string tensor that contains the file name.
//  1: a single element string tensor that names the output to be restored.
// If restore_slice is true:
//  2: shape and slice specification of the tensor to restore.
void RestoreTensor(OpKernelContext* context,
                   checkpoint::TensorSliceReader::OpenTableFunction open_func,
                   int preferred_shard, bool restore_slice);

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_SAVE_RESTORE_TENSOR_H_
