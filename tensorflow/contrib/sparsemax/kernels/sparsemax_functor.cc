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

#include "sparsemax_functor.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <algorithm>

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

#define UNUSED(x) (void)(x)

template <typename T>
struct Sparsemax<CPUDevice, T> {
  void operator()(const CPUDevice& d,
                  typename TTypes<T>::ConstMatrix input,
                  typename TTypes<T>::Vec temp_vec_unused, //only used for GPU
                  typename TTypes<T>::Matrix temp_mat_unused, //only used for GPU
                  typename TTypes<T>::Matrix output) {
    // NOTE: This CPU implementation uses the naive sorting/breakpoint based
    // algorithm. On the CPU, sequental algorithms works just fine, thus there
    // are much better alternatives.

    UNUSED(d);
    UNUSED(temp_vec_unused);
    UNUSED(temp_mat_unused);

    // define integers {0, 1} in matching template type
    T zero = static_cast<T>(0);
    T one = static_cast<T>(1);
    // get input size
    const int num_rows = input.dimension(0); // batch_size
    const int num_cols = input.dimension(1);

    // create temporary vector used for sorting
    std::vector<T> sorted_temp(num_cols);
    // calculate sparsemax for each row
    for (int r = 0; r < num_rows; r++) {

      // sparsemax, is like softmax, invarient to adding a constant,
      // so for numerical stability the mean is substracted.
      // First calculate the mean, using a numerically stable algorithm.
      double mean_double = 0;
      double mean_obs = 1;
      for (int c = 0; c < num_cols; c++) {
        double x = static_cast<double>(input(r, c));
        mean_double += (x - mean_double) / mean_obs;
        mean_obs += 1;
      }
      T mean = static_cast<T>(mean_double);

      // copy input to temporary vector for sorting and simultaneously
      // substract the mean for numerical stability.
      for (int c = 0; c < num_cols; c++) {
        sorted_temp[c] = input(r, c) - mean;
      }

      // sort vector
      std::sort(sorted_temp.begin(), sorted_temp.end(), std::greater<T>());

      // calculate k(z), the sorted support index
      T cumsum = zero; // cumsum use for finding support k
      T support = zero; // k
      T cumsum_support = zero; // cumsum for support i <= k
      for (int c = 0; c < num_cols; c++) {
        const T k = static_cast<T>(c) + one; // the 1-indexed index

        cumsum += sorted_temp[c];
        if (one + k * sorted_temp[c] > cumsum) {
          support = k;
          cumsum_support = cumsum;
        } else {
          // All the remaining cases will be false, thus we break to save
          // computation time.
          break;
        }
      }

      // calculate tau(z)
      const T tau = (cumsum_support - one) / support;

      // calculate sparse probability and copy it to the output
      for (int c = 0; c < num_cols; c++) {
        output(r, c) = std::max((input(r, c) - mean) - tau, zero);
      }
    }
  }
};

template struct Sparsemax<CPUDevice, Eigen::half>;
template struct Sparsemax<CPUDevice, float>;
template struct Sparsemax<CPUDevice, double>;

}  // namespace functor
}  // namespace tensorflow
