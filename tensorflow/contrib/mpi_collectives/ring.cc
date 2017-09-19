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
template Status RingAllreduce<CPUDevice, int>(
    OpKernelContext*, const Tensor*, Tensor*, Tensor*);
template Status RingAllreduce<CPUDevice, long long>(
    OpKernelContext*, const Tensor*, Tensor*, Tensor*);
template Status RingAllreduce<CPUDevice, float>(
    OpKernelContext*, const Tensor*, Tensor*, Tensor*);

// Generate all necessary specializations for RingAllgather.
template Status RingAllgather<CPUDevice, int>(
    OpKernelContext*, const Tensor*, const std::vector<size_t>&, Tensor*);
template Status RingAllgather<CPUDevice, long long>(
    OpKernelContext*, const Tensor*, const std::vector<size_t>&, Tensor*);
template Status RingAllgather<CPUDevice, float>(
    OpKernelContext*, const Tensor*, const std::vector<size_t>&, Tensor*);

// Copy data on a CPU using a straight-forward memcpy.
template<> void CopyTensorData<CPUDevice>(void* dst, void* src, size_t size) {
    std::memcpy(dst, src, size);
};

// Accumulate values on a CPU.
#define GENERATE_ACCUMULATE(type)                                  \
template<> void AccumulateTensorData<CPUDevice, type>(             \
        type* dst, type* src, size_t size) {                       \
    for (unsigned int i = 0; i < size; i++) {                      \
        dst[i] += src[i];                                          \
    }                                                              \
};
GENERATE_ACCUMULATE(int);
GENERATE_ACCUMULATE(long long);
GENERATE_ACCUMULATE(float);
#undef GENERATE_ACCUMULATE

}
}
}

#endif // TENSORFLOW_USE_MPI
