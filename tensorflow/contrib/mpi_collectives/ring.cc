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

#ifdef TENSORFLOW_USE_MPI

#define EIGEN_USE_THREADS

#include "tensorflow/contrib/mpi_collectives/ring.h"

namespace tensorflow {
namespace contrib {
namespace mpi {

using CPUDevice = Eigen::ThreadPoolDevice;

extern template MPI_Datatype MPIType<float>();
extern template MPI_Datatype MPIType<int>();
extern template MPI_Datatype MPIType<long long>();
extern template DataType TensorFlowDataType<float>();
extern template DataType TensorFlowDataType<int>();
extern template DataType TensorFlowDataType<long long>();

// Generate all necessary specializations for RingAllreduce.
template Status RingAllreduce<CPUDevice, int>(OpKernelContext*, const Tensor*,
                                              Tensor*, Tensor*);
template Status RingAllreduce<CPUDevice, long long>(OpKernelContext*,
                                                    const Tensor*, Tensor*,
                                                    Tensor*);
template Status RingAllreduce<CPUDevice, float>(OpKernelContext*, const Tensor*,
                                                Tensor*, Tensor*);

// Generate all necessary specializations for RingAllgather.
template Status RingAllgather<CPUDevice, int>(OpKernelContext*, const Tensor*,
                                              const std::vector<size_t>&,
                                              Tensor*);
template Status RingAllgather<CPUDevice, long long>(OpKernelContext*,
                                                    const Tensor*,
                                                    const std::vector<size_t>&,
                                                    Tensor*);
template Status RingAllgather<CPUDevice, float>(OpKernelContext*, const Tensor*,
                                                const std::vector<size_t>&,
                                                Tensor*);

// Copy data on a CPU using a straight-forward memcpy.
template <>
void CopyTensorData<CPUDevice>(void* dst, void* src, size_t size) {
  std::memcpy(dst, src, size);
};

// Accumulate values on a CPU.
#define GENERATE_ACCUMULATE(type)                                    \
  template <>                                                        \
  void AccumulateTensorData<CPUDevice, type>(type * dst, type * src, \
                                             size_t size) {          \
    for (unsigned int i = 0; i < size; i++) {                        \
      dst[i] += src[i];                                              \
    }                                                                \
  };
GENERATE_ACCUMULATE(int);
GENERATE_ACCUMULATE(long long);
GENERATE_ACCUMULATE(float);
#undef GENERATE_ACCUMULATE

}  // namespace mpi
}  // namespace contrib
}  // namespace tensorflow

#endif  // TENSORFLOW_USE_MPI
