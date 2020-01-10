#ifndef TENSORFLOW_KERNELS_IO_H_
#define TENSORFLOW_KERNELS_IO_H_

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

#endif  // TENSORFLOW_KERNELS_IO_H_
