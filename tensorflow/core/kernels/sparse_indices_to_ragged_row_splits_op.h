
#ifndef GEN_ROW_SPLITS_H_
#define GEN_ROW_SPLITS_H_

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"



using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

#if GOOGLE_CUDA
template <typename Device, typename IndexType>
struct SparseIndicesToRaggedRowSplitsFunctor {
    tensorflow::Status operator()(
        tensorflow::OpKernelContext* context,
        const Device& d,
        int num_nonzero,
        bool validate_ragged_right,
        const IndexType* indices_flat_2d,
        const IndexType* dense_shape,
        int32_t* invalid_flag
    );
};

template <typename IndexType>
struct SparseIndicesToRaggedRowSplitsFunctor<GPUDevice, IndexType> {
    tensorflow::Status operator()(
        tensorflow::OpKernelContext* context,
        const GPUDevice& d,
        int num_nonzero,
        bool validate_ragged_right,
        const IndexType* indices_flat_2d,
        const IndexType* dense_shape,
        int32_t* invalid_flag
    );
};

#endif
#endif

#endif


