/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/sparse_indices_to_ragged_row_splits_op.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_solvers.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

namespace functor {

namespace {

template <typename IndexType>
__global__ void SparseIndicesToRaggedRowSplitsKernel(
        const int num_nonzero, // total number of nonzero values in tensor
        const bool validate_ragged_right, // enable validation of input tensor format
        const IndexType* indices_flat_2d, // array of length 2*num_nonzero, sparse indices
        const IndexType* dense_shape,
        IndexType* row_splits, // array of length num_rows + 1, output row splits
        int32_t* invalid_flag // single bool, will be set to 1 if input tensor is invalid
        ) {

    *invalid_flag = 0; // extremely simplistic way of zeroing out this value
    __syncthreads();

    auto num_rows = dense_shape[0];
    auto num_cols = dense_shape[1];

    int kernel_index = blockIdx.x*blockDim.x + threadIdx.x;
    int kernel_count = blockDim.x*gridDim.x;
    int kernel_dim = (kernel_count == 1) ? num_nonzero : ((num_nonzero / (kernel_count - 1)) + 1);
    if (kernel_dim == 0) {kernel_dim = 1;}
    int start_index = kernel_dim*kernel_index;
    if (start_index >= num_nonzero) { // in case of very weird dimensions
        return;
    }
    int end_index = kernel_dim*(kernel_index + 1);
    if (end_index > num_nonzero) {end_index = num_nonzero;}
    IndexType prev_row = (kernel_index == 0) ? -1 : indices_flat_2d[(start_index - 1)*2];
    IndexType prev_col = -1;

    IndexType n = start_index;
    if (validate_ragged_right && (n > 0)) {
        // if starting in the middle of the row, set the previous column idx for comparison
        prev_col = indices_flat_2d[2*n-1];
    }
    for (; n < end_index; ++n) { // for each pair of value + indices
        IndexType curr_row = indices_flat_2d[2*n];
        if (validate_ragged_right) {
            // if another kernel invocation found an issue with input tensor, quit
            if (*invalid_flag == 1){
                return;
            }
            // at the end of a row, check that row idx increased and is not outside size
            // (to ensure indices are in order)
            if (curr_row != prev_row) {
                if ((curr_row < prev_row) || (curr_row >= num_rows)) {
                    GpuAtomicMax(invalid_flag, 1);
                    return;
                }
                prev_col = -1;
            }
            // within a row, check that column increased by one
            // (to ensure tensor is ragged-right and indices are in order)
            IndexType curr_col = indices_flat_2d[2*n+1];
            if ((curr_col != prev_col + 1) || (curr_col >= num_cols)) {
                GpuAtomicMax(invalid_flag, 1);
                return;
            } else {
                prev_col = curr_col;
            }
        }
        // fill in row splits; loop used to fill all if row is empty
        for (IndexType r = prev_row; r < curr_row; ++r) {
            row_splits[r+1] = n;
        }
        prev_row = curr_row;
    }
    // if at end of tensor, fill in final row split + splits for any training empty rows
    if ((start_index < num_nonzero) && (end_index == num_nonzero)) {
        for (IndexType r = prev_row; r < num_rows; ++r) {
            row_splits[r+1] = n;
        }
    }
}

} // namespace

template <typename IndexType>
struct SparseIndicesToRaggedRowSplitsFunctor<GPUDevice, IndexType> {
    Status operator()(
            OpKernelContext* context,
            //const GPUDevice& d,
            int num_nonzero, // total number of nonzero values in tensor
            bool validate_ragged_right,
            const IndexType* indices_flat_2d, // array of length 2*num_nonzero
            const IndexType* dense_shape,
            int32_t* invalid_flag
            ){
        // copy number of rows to host in order to allocate row_splits tensor with correct size
        ScratchSpace<IndexType> dense_row_count(context, 1, /*on_host=*/true);
        cudaMemcpy(dense_row_count.mutable_data(), dense_shape, sizeof(*dense_row_count.data()), cudaMemcpyDeviceToHost);
        IndexType num_rows = *dense_row_count.data();
        Tensor* output;
        TF_RETURN_IF_ERROR(context, context->allocate_output("row_splits", TensorShape({num_rows + 1}), &output));
        IndexType* row_splits = output->flat<IndexType>().data();

        int splits = num_rows / 2;
        if (splits > (num_nonzero / 10)) {
            splits = (num_nonzero / 10);
        }
        if (splits < 1) {
            splits = 1;
        }

        const GPUDevice& d = context->eigen_gpu_device();
        GpuLaunchConfig config = GetGpuLaunchConfig(splits, d);
        int block_count = config.block_count;
        int thread_per_block = config.thread_per_block;

        TF_CHECK_OK(GpuLaunchKernel(
                    SparseIndicesToRaggedRowSplitsKernel<IndexType>,
                    block_count, thread_per_block, 0, d.stream(),
                    num_nonzero,
                    validate_ragged_right,
                    indices_flat_2d,
                    dense_shape,
                    row_splits,
                    invalid_flag
        ));
        return Status::OK();
    }
};

template struct SparseIndicesToRaggedRowSplitsFunctor<GPUDevice, int32>;
template struct SparseIndicesToRaggedRowSplitsFunctor<GPUDevice, int64>;

} // namespace functor

} // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
