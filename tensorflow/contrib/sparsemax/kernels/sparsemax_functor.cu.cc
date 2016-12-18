/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");

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

#include "sparsemax_functor.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

#include <cmath>
#include <type_traits>
#include <math_constants.h>

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename T>
__global__ void even_sort_kernel(T *sorted,
                                 const int num_rows,
                                 const int num_cols) {
  const int col_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_index = blockIdx.y;

  T* sorted_row = &sorted[row_index * num_cols];

  if (!(col_index & 1) && col_index < (num_cols - 1)) {
      if (sorted_row[col_index] < sorted_row[col_index + 1]) {
        T temp = sorted_row[col_index];
        sorted_row[col_index] = sorted_row[col_index + 1];
        sorted_row[col_index + 1] = temp;
      }
  }
}

template <typename T>
__global__ void odd_sort_kernel(T *sorted,
                                const int num_rows,
                                const int num_cols) {
  const int col_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_index = blockIdx.y;

  T* sorted_row = &sorted[row_index * num_cols];

  if ((col_index & 1) && col_index < (num_cols - 1)) {
      if (sorted_row[col_index] < sorted_row[col_index + 1]) {
        T temp = sorted_row[col_index];
        sorted_row[col_index] = sorted_row[col_index + 1];
        sorted_row[col_index + 1] = temp;
      }
  }
}

template <typename T>
void odd_even_sort(typename TTypes<T>::Matrix sorted,
                   const int num_rows,
                   const int num_cols) {
  // calculate paramization constants
  const int col_threads_per_block = 256;
  const int col_blocks = static_cast<int>(std::ceil(
   static_cast<double>(num_cols) / static_cast<double>(col_threads_per_block)
  ));

  dim3 threads_per_block(col_threads_per_block, 1, 1);
  dim3 blocks(col_blocks, num_rows, 1);

  // calculate number of odd-even iterations
  const int iterations = static_cast<int>(std::ceil(
   static_cast<double>(num_cols) / 2.0
  ));

  // Launch the even and odd kernels separately to get a global syncronization.
  // This is a very naive approach, as global syncronization is expensive.
  for(int i = 0; i < iterations; i++) {
    // launch even kernel
    even_sort_kernel<T><<<blocks, threads_per_block>>>(
      sorted.data(), num_rows, num_cols
    );

    // launch odd kernel
    odd_sort_kernel<T><<<blocks, threads_per_block>>>(
      sorted.data(), num_rows, num_cols
    );
  }
}

template <typename T>
__global__ void support_threshold_kernel(const T* cumsum,
                                         const int num_rows,
                                         const int num_cols,
                                         T* sorted) {
  const int col_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_index = blockIdx.y;

  const T* cumsum_row = &cumsum[row_index * num_cols];
  T* sorted_row = &sorted[row_index * num_cols];

  const T one = static_cast<T>(1);

  if (col_index < num_cols) {
    const T k = static_cast<T>(col_index + 1);
    sorted_row[col_index] = static_cast<T>(
      one + k * sorted_row[col_index] > cumsum_row[col_index]
    );
  }
}

template <typename T>
void support_threshold(typename TTypes<T>::Matrix cumsum,
                       const int num_rows,
                       const int num_cols,
                       typename TTypes<T>::Matrix sorted) {
   // calculate paramization constants
   const int col_threads_per_block = 256;
   const int col_blocks = static_cast<int>(std::ceil(
    static_cast<double>(num_cols) / static_cast<double>(col_threads_per_block)
   ));

   dim3 threads_per_block(col_threads_per_block, 1, 1);
   dim3 blocks(col_blocks, num_rows, 1);

   // launch kernel
   support_threshold_kernel<T><<<blocks, threads_per_block>>>(
     cumsum.data(), num_rows, num_cols, sorted.data()
   );
}

template <typename T>
__global__ void calculate_tau_kernel(const T* cumsum,
                                     const int num_rows,
                                     const int num_cols,
                                     T* support) {
  const int row_index = blockIdx.x * blockDim.x + threadIdx.x;

  const T one = static_cast<T>(1);

  if (row_index < num_rows) {
    const int support_index = static_cast<int>(support[row_index]) - 1;
    const T cumsum_value = cumsum[row_index * num_cols + support_index];
    support[row_index] = (cumsum_value - one) / support[row_index];
  }
}

template <typename T>
void calculate_tau(typename TTypes<T>::Matrix cumsum,
                   const int num_rows,
                   const int num_cols,
                   typename TTypes<T>::Matrix support) {
   // calculate paramization constants
   const int threads_per_block = 256;
   const int blocks = static_cast<int>(std::ceil(
    static_cast<double>(num_rows) / static_cast<double>(threads_per_block)
   ));

   // launch kernel
   calculate_tau_kernel<T><<<blocks, threads_per_block>>>(
     cumsum.data(), num_rows, num_cols, support.data()
   );
}

template <typename T>
__global__ void calculate_probability_kernel(const T* input,
                                              const T* mean,
                                              const T* tau,
                                              const int num_rows,
                                              const int num_cols,
                                              T* output) {
  const int col_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_index = blockIdx.y;

  T zero = static_cast<T>(0);

  if (col_index < num_cols) {
    const int flat_index = row_index * num_cols + col_index;
    output[flat_index] = max(
      (input[flat_index] - mean[row_index]) - tau[row_index],
      zero
    );
  }
}

template <typename T>
void calculate_probability(typename TTypes<T>::ConstMatrix input,
                            typename TTypes<T>::Vec mean,
                            typename TTypes<T>::Matrix tau,
                            const int num_rows,
                            const int num_cols,
                            typename TTypes<T>::Matrix output) {
  // calculate paramization constants
  const int col_threads_per_block = 256;
  const int col_blocks = static_cast<int>(std::ceil(
   static_cast<double>(num_cols) / static_cast<double>(col_threads_per_block)
  ));

  dim3 threads_per_block(col_threads_per_block, 1, 1);
  dim3 blocks(col_blocks, num_rows, 1);

  // launch kernel
  calculate_probability_kernel<T><<<blocks, threads_per_block>>>(
    input.data(), mean.data(), tau.data(), num_rows, num_cols, output.data()
  );
}

template <typename T>
struct Sparsemax<GPUDevice, T> {
  void operator()(const GPUDevice& d,
                  typename TTypes<T>::ConstMatrix input,
                  typename TTypes<T>::Vec temp_vec,
                  typename TTypes<T>::Matrix temp_mat,
                  typename TTypes<T>::Matrix output) {
    // NOTE: This GPU implementation uses the naive sorting/breakpoint based
    // algorithm. But on the GPU, sequental algorithms wont work thus the
    // naive algorithm is used.

    const int kBatchDim = 0;
    const int kClassDim = 1;

    const int batch_size = input.dimension(kBatchDim);
    const int num_classes = input.dimension(kClassDim);

// These arrays are used to reduce along the class dimension, and broadcast
// the resulting value to all classes.
#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::DSizes<int, 1> along_class(kClassDim);
    Eigen::DSizes<int, 2> batch_by_one(batch_size, 1);
    Eigen::DSizes<int, 2> one_by_class(1, num_classes);
#else
    Eigen::IndexList<Eigen::type2index<kClassDim> > along_class;
    Eigen::IndexList<Eigen::type2index<1> > depth_dim;
    Eigen::IndexList<int, Eigen::type2index<1> > batch_by_one;
    batch_by_one.set(0, batch_size);
    Eigen::IndexList<Eigen::type2index<1>, int> one_by_class;
    one_by_class.set(1, num_classes);
#endif

    // sparsemax, is like softmax, invarient to adding a constant,
    // so for numerical stability the mean is substracted.
    temp_vec.device(d) = input.mean(along_class);
    temp_mat.device(d) = (input -
                          temp_vec.reshape(batch_by_one)
                                  .broadcast(one_by_class));

    // NOTE: this odd even sort implementation will only be efficient for when
    // the observation has a small dimentionality, such each row vectors fits
    // within the L1 cache. It will work for larger vectors, but then it will
    // use the global memory.
    odd_even_sort<T>(temp_mat, batch_size, num_classes);

    // Cumsum the sorted matrix along axis 1.
    // Put results in output as the the sorted and cumsum needs to be used
    // together.
    Eigen::internal::SumReducer<T> reducer;
    output.device(d) = temp_mat.scan(1, reducer, false);

    // Calculate threshold used in support calculation.
    // Replace sorted matrix, with the threshold booleans.
    support_threshold<T>(output, batch_size, num_classes, temp_mat);

    // Sum each row, to get the support index.
    // This will reuse the temporary matrix, which is larger than required,
    // but the results will just be stored as if it was a flat vector.
    temp_mat.device(d) = temp_mat.sum(along_class);

    // Calculate tau
    // Overwrites temp results with tau(z), again this just uses temp
    // as a flat vector.
    calculate_tau<T>(output, batch_size, num_classes, temp_mat);

    // Calculate probability
    // Use temp and input, and put results in output
    calculate_probability<T>(input, temp_vec, temp_mat,
                              batch_size, num_classes, output);
  }
};

template struct Sparsemax<GPUDevice, Eigen::half>;
template struct Sparsemax<GPUDevice, float>;
template struct Sparsemax<GPUDevice, double>;

}  // namespace functor
}  // namespace tensorflow

#endif
