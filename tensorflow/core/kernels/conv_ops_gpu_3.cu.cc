#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <algorithm>

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

// A simple array that contains data that can be passed between CPU and GPU.
template <typename T, int IndexCount>
struct Array {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const T& operator[](int index) const {
    return data[index];
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T& operator[](int index) {
    return data[index];
  }
  int data[IndexCount];
};

// A dimension type with compile-time known size.
template <int IndexCount>
struct Dimension : Array<int, IndexCount> {};

// An index type with compile-time known size.
template <int IndexCount>
struct Index : Array<int, IndexCount> {};

// A helper function that converts a tensor index into a flat array index.
template <int IndexCount>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int TensorIndexToFlat(
    const Index<IndexCount>& index, const Dimension<IndexCount>& dims) {
  int flat_index = index[0];
  for (int i = 1; i < IndexCount; i++) {
    flat_index = flat_index * dims[i] + index[i];
  }
  return flat_index;
}

// A helper function that converts a flat arrary index into a tensor index.
template <int IndexCount>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index<IndexCount> FlatToTensorIndex(
    int index, const Dimension<IndexCount>& dims) {
  Index<IndexCount> tensor_index;
  for (int i = IndexCount - 1; i >= 0; i--) {
    tensor_index[i] = index % dims[i];
    index /= dims[i];
  }
  return tensor_index;
}

// A Cuda custom kernel that swaps dimension-0 and dimension-2 of a 3D tensor.
template <typename T>
__global__ void SwapDimension0And2InTensor3(int nthreads, const T* input,
                                            Dimension<3> input_dims,
                                            T* output) {
  Dimension<3> output_dims;
  output_dims[0] = input_dims[2];
  output_dims[1] = input_dims[1];
  output_dims[2] = input_dims[0];

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int output_index = index;

    Index<3> output_tensor_index = FlatToTensorIndex(output_index, output_dims);

    Index<3> input_tensor_index;
    input_tensor_index[0] = output_tensor_index[2];
    input_tensor_index[1] = output_tensor_index[1];
    input_tensor_index[2] = output_tensor_index[0];

    int input_index = TensorIndexToFlat(input_tensor_index, input_dims);

    output[output_index] = input[input_index];
  }
}

// A Cuda custom kernel that swaps dimension-1 and dimension-2 of a 3D tensor.
template <typename T>
__global__ void SwapDimension1And2InTensor3(int nthreads, const T* input,
                                            Dimension<3> input_dims,
                                            T* output) {
  Dimension<3> output_dims;
  output_dims[0] = input_dims[0];
  output_dims[1] = input_dims[2];
  output_dims[2] = input_dims[1];

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int output_index = index;
    Index<3> output_tensor_index = FlatToTensorIndex(output_index, output_dims);

    Index<3> input_tensor_index;
    input_tensor_index[0] = output_tensor_index[0];
    input_tensor_index[1] = output_tensor_index[2];
    input_tensor_index[2] = output_tensor_index[1];

    int input_index = TensorIndexToFlat(input_tensor_index, input_dims);

    output[output_index] = input[input_index];
  }
}

// A Cuda custom kernel that converst input to output, given proper padding on
// the left and the top. The padded value is zero.
template <typename T>
__global__ void PadInputCustomKernel(int nthreads, const T* input,
                                     Dimension<4> input_dims, T* output,
                                     Dimension<4> output_dims,
                                     int padding_rows_left,
                                     int padding_cols_left) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int output_index = index;
    Index<4> output_tensor_index = FlatToTensorIndex(output_index, output_dims);

    Index<4> input_tensor_index;
    input_tensor_index[0] = output_tensor_index[0];
    input_tensor_index[1] = output_tensor_index[1] - padding_rows_left;
    input_tensor_index[2] = output_tensor_index[2] - padding_cols_left;
    input_tensor_index[3] = output_tensor_index[3];

    if (input_tensor_index[1] >= 0 && input_tensor_index[1] < input_dims[1] &&
        input_tensor_index[2] >= 0 && input_tensor_index[2] < input_dims[2]) {
      int input_index = TensorIndexToFlat(input_tensor_index, input_dims);
      output[output_index] = input[input_index];
    } else {
      output[output_index] = 0;
    }
  }
}

// A GPU helper function that converts TensorFlow filter format to Cudnn filter
// format.
template <typename T>
struct TransformFilter<GPUDevice, T, int> {
  typedef GPUDevice Device;
  void operator()(const Device& d, typename TTypes<T, 4, int>::ConstTensor in,
                  typename TTypes<T, 4, int>::Tensor out) {
    Dimension<3> combined_dims;
    combined_dims[0] = in.dimension(0) * in.dimension(1);
    combined_dims[1] = in.dimension(2);
    combined_dims[2] = in.dimension(3);
    CudaLaunchConfig config = GetCudaLaunchConfig(out.size(), d);
    SwapDimension0And2InTensor3<
        T><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        config.virtual_thread_count, in.data(), combined_dims, out.data());
  }
};

// Converts Cudnn filter format back to TensorFlow filter format.
template <typename T>
struct ReverseTransformFilter<GPUDevice, T> {
  typedef GPUDevice Device;
  void operator()(const Device& d, typename TTypes<T, 4>::ConstTensor in,
                  typename TTypes<T, 4>::Tensor out) {
    Dimension<3> combined_dims;
    combined_dims[0] = in.dimension(0);
    combined_dims[1] = in.dimension(1);
    combined_dims[2] = in.dimension(2) * in.dimension(3);
    CudaLaunchConfig config = GetCudaLaunchConfig(out.size(), d);
    SwapDimension0And2InTensor3<
        T><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        config.virtual_thread_count, in.data(), combined_dims, out.data());
  }
};

// A GPU helper function that converts input tensor to a larger output tensor,
// given proper padding values. The padded value is zero.
template <typename T>
struct PadInput<GPUDevice, T, int> {
  typedef GPUDevice Device;
  void operator()(const Device& d, typename TTypes<T, 4, int>::ConstTensor in,
                  int padding_rows_left, int padding_rows_right,
                  int padding_cols_left, int padding_cols_right,
                  typename TTypes<T, 4, int>::Tensor out) {
    CudaLaunchConfig config = GetCudaLaunchConfig(out.size(), d);
    Dimension<4> input_dims;
    for (int i = 0; i < 4; i++) {
      input_dims[i] = in.dimension(i);
    }
    Dimension<4> output_dims;
    for (int i = 0; i < 4; i++) {
      output_dims[i] = out.dimension(i);
    }

    PadInputCustomKernel<
        T><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        config.virtual_thread_count, in.data(), input_dims, out.data(),
        output_dims, padding_rows_left, padding_cols_left);
  }
};

// A GPU helper functor that converts NHWC TensorFlow data format to
// NCHW format that is accepted by Cudnn.
template <typename T>
struct NHWCToNCHW<GPUDevice, T> {
  typedef GPUDevice Device;
  void operator()(const Device& d, typename TTypes<T, 4>::ConstTensor in,
                  typename TTypes<T, 4>::Tensor out) {
    Dimension<3> combined_dims;
    combined_dims[0] = in.dimension(0);
    combined_dims[1] = in.dimension(1) * in.dimension(2);
    combined_dims[2] = in.dimension(3);
    CudaLaunchConfig config = GetCudaLaunchConfig(out.size(), d);
    SwapDimension1And2InTensor3<
        T><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        config.virtual_thread_count, in.data(), combined_dims, out.data());
  }
};

// A GPU helper functor that converts NCHW Cudnn data format to NHWC TensorFlow
// Format.
template <typename T>
struct NCHWToNHWC<GPUDevice, T> {
  typedef GPUDevice Device;
  void operator()(const Device& d, typename TTypes<T, 4>::ConstTensor in,
                  typename TTypes<T, 4>::Tensor out) {
    Dimension<3> combined_dims;
    combined_dims[0] = in.dimension(0);
    combined_dims[1] = in.dimension(1);
    combined_dims[2] = in.dimension(2) * in.dimension(3);
    CudaLaunchConfig config = GetCudaLaunchConfig(out.size(), d);
    SwapDimension1And2InTensor3<
        T><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        config.virtual_thread_count, in.data(), combined_dims, out.data());
  }
};

}  // namespace functor

template struct functor::ShuffleAndReverse<GPUDevice, float, 4, int>;

template struct functor::ShuffleAndReverse<GPUDevice, float, 4,
                                           Eigen::DenseIndex>;

template struct functor::TransformFilter<GPUDevice, float, int>;

template struct functor::ReverseTransformFilter<GPUDevice, float>;

template struct functor::PadInput<GPUDevice, float, int>;

template struct functor::TransformDepth<GPUDevice, float, int>;

template struct functor::NHWCToNCHW<GPUDevice, float>;

template struct functor::NCHWToNHWC<GPUDevice, float>;

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
