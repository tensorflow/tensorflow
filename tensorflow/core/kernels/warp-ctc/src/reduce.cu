// Includes, system
// #include <stdio.h>
// #include <stdlib.h>

// Includes, cuda
// #include <cuda_runtime.h>
// #include <cublas_v2.h>

// Includes, cuda helper functions
// #include <helper_cuda.h>

// For the functors
#include "tensorflow/core/kernels/warp-ctc/include/detail/ctc_helper.h"
#include "tensorflow/core/kernels/warp-ctc/include/ctc.h"

const int warp_size = 32;

template<int NT, typename T, typename Rop>
struct CTAReduce;

template<int NT, typename T, typename Rop>
struct CTAReduce {
    enum { Size = NT, Capacity = NT };
    struct Storage { T shared[Capacity]; };

    __device__ static T reduce(int tid, T x, Storage& storage, int count, Rop g) {
        T* s = storage.shared;
        s[tid] = x;
        __syncthreads();

        // Fold the data in half with each pass.
#pragma unroll
        for(int offset = NT / 2; offset >= warp_size; offset /= 2) {
            if(tid + offset < count && tid < offset) {
                // Read from the right half and store to the left half.
                x = g(x, s[offset + tid]);
                s[tid] = x;
            }
            __syncthreads();
        }

        T shuff;
        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
            shuff = __shfl_down(x, offset);
            if (tid + offset < count && tid < offset)
                x = g(x, shuff);
        }
        return x;
    }
};

template <int NT, typename Iop, typename Rop, typename T>
__global__ void reduce_rows(Iop f, Rop g, const T* input, T* output,
                            int num_rows, int num_cols) {

    typedef CTAReduce<NT, T, Rop> R;
    __shared__ typename R::Storage storage;

    int tid = threadIdx.x;
    int idx = tid;
    int col = blockIdx.x;
    T curr;

    // Each block works on a column
    if (idx < num_rows)
        curr = f(input[idx + col*num_rows]);
    idx += NT;


    while (idx < num_rows) {
        curr += f(input[idx + col*num_rows]);
        idx += NT;
    }

    // Sum thread-totals over the CTA.
    curr = R::reduce(tid, curr, storage, num_rows, g);

    // Store result in out
    if (tid == 0)
        output[col] = curr;
}

template <int NT, typename Iop, typename Rop, typename T>
__global__ void reduce_cols(Iop f, Rop g, const T* input, T* output,
                            int num_rows, int num_cols) {

    __shared__ T s[NT];

    int warps_per_block = NT / warp_size;
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = threadIdx.y;
    T curr;

    if (row < num_rows && col < num_cols) {
        curr = f(input[row + col*num_rows]);
        col += blockDim.y;
        while (col < num_cols) {
            curr = g(curr, f(input[row + col*num_rows]));
            col += blockDim.y;
        }
    }
    s[threadIdx.x * warps_per_block + threadIdx.y] = curr;
    __syncthreads();

    // Reduce
    if (threadIdx.y == 0 && row < num_rows) {
#pragma unroll
        for (int i = 1; i < warps_per_block && i < num_cols; ++i)
            curr = g(curr, s[i + threadIdx.x * warps_per_block]);
        output[row] = curr;
    }
}

struct ReduceHelper {

    template<typename T, typename Iof, typename Rof>
    static void impl(Iof f, Rof g, const T* input, T* output, int num_rows, int num_cols, bool axis, cudaStream_t stream) {

        int grid_size;

        if (axis) {
            grid_size = num_cols;
            reduce_rows<128><<<grid_size, 128, 0, stream>>>
               (f, g, input, output, num_rows, num_cols);

        } else {
            dim3 tpb(warp_size, 128 / warp_size);
            grid_size = (num_cols + warp_size - 1)/warp_size;
            reduce_cols<128><<<grid_size, tpb, 0, stream>>>
                (f, g, input, output, num_rows, num_cols);

        }
    }
};


template<typename T, typename Iof, typename  Rof>
ctcStatus_t reduce(Iof f, Rof g, const T* input, T* output, int rows, int cols, bool axis, cudaStream_t stream) {
    ReduceHelper::impl(f, g, input, output, rows, cols, axis, stream);
    cudaStreamSynchronize(stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return CTC_STATUS_EXECUTION_FAILED;

    return CTC_STATUS_SUCCESS;
}

ctcStatus_t reduce_negate(const float *input, float *output, int rows, int cols, bool axis, cudaStream_t stream) {
    return reduce(ctc_helper::negate<float>(), ctc_helper::add<float>(), input, output, rows, cols, axis, stream);
}

ctcStatus_t reduce_exp(const float *input, float *output, int rows, int cols, bool axis, cudaStream_t stream) {
    return reduce(ctc_helper::exponential<float>(), ctc_helper::add<float>(), input, output, rows, cols, axis, stream);
}

ctcStatus_t reduce_max(const float *input, float *output, int rows, int cols, bool axis, cudaStream_t stream) {
    return reduce(ctc_helper::identity<float>(), ctc_helper::maximum<float>(),input, output, rows, cols, axis, stream);
}
