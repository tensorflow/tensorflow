// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/avgpooling_op.h"

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/pooling_ops_common.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "third_party/eigen3/unsupported/Eigen/CXX11/NeuralNetworks"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/maxpooling_op_gpu.h"
#include "tensorflow/core/kernels/pooling_ops_common_gpu.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class AvgPoolingOp : public UnaryOp<T> {
 public:
  explicit AvgPoolingOp(OpKernelConstruction* context) : UnaryOp<T>(context) {
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument(
                    "Sliding window ksize field must "
                    "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument(
                    "Sliding window stride field must "
                    "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
    PoolParameters params{context, ksize_, stride_, padding_,
                          tensor_in.shape()};
    if (!context->status().ok()) {
      return;
    }
    OP_REQUIRES(context, params.depth_window == 1,
                errors::Unimplemented(
                    "Non-spatial pooling is not "
                    "yet supported. Volunteers? :)"));

    // For avgpooling, tensor_in should have 4 dimensions.
    OP_REQUIRES(context, tensor_in.dims() == 4,
                errors::InvalidArgument("tensor_in must be 4-dimensional"));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, params.forward_output_shape(), &output));

    if (std::is_same<Device, GPUDevice>::value) {
      Eigen::PaddingType pt = BrainPadding2EigenPadding(padding_);
      functor::SpatialAvgPooling<Device, T>()(
          context->eigen_device<Device>(), output->tensor<T, 4>(),
          tensor_in.tensor<T, 4>(), params.window_rows, params.window_cols,
          params.row_stride, params.col_stride, pt);
    } else {
      SpatialAvgPool<Device, T>(context, output, tensor_in, params, padding_);
    }
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
};

REGISTER_KERNEL_BUILDER(Name("AvgPool")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        AvgPoolingOp<CPUDevice, float>);

#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                      \
  template <>                                                    \
  void SpatialAvgPooling<GPUDevice, T>::operator()(              \
      const GPUDevice& d, typename TTypes<T, 4>::Tensor output,  \
      typename TTypes<T, 4>::ConstTensor input, int window_rows, \
      int window_cols, int row_stride, int col_stride,           \
      const Eigen::PaddingType& padding);                        \
  extern template struct SpatialAvgPooling<GPUDevice, T>;

DECLARE_GPU_SPEC(float);
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNEL_BUILDER(Name("AvgPool")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T"),
                        AvgPoolingOp<GPUDevice, float>);
#endif  // GOOGLE_CUDA

// The operation to compute AvgPool gradients.
// It takes two inputs:
//   - The original input tensor shape
//   - Backprop tensor for output
// It produces one output: backprop tensor for input.
template <typename Device, class T>
class AvgPoolingGradOp : public OpKernel {
 public:
  explicit AvgPoolingGradOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument(
                    "Sliding window ksize field must "
                    "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument(
                    "Sliding window strides field must "
                    "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in_shape = context->input(0);
    const Tensor& out_backprop = context->input(1);
    // For avgpooling, tensor_in_shape should have 1 dimension, and 4 elements.
    OP_REQUIRES(context, tensor_in_shape.dims() == 1 &&
                             tensor_in_shape.NumElements() == 4,
                errors::InvalidArgument(
                    "out_backprop must be 1-dimensional and 4 "
                    "elements"));
    // For avgpooling, out_backprop should have 4 dimensions.
    OP_REQUIRES(context, out_backprop.dims() == 4,
                errors::InvalidArgument("out_backprop must be 4-dimensional"));
    const int64 out_backprop_batch = out_backprop.dim_size(0);
    const int64 out_backprop_rows = out_backprop.dim_size(1);
    const int64 out_backprop_cols = out_backprop.dim_size(2);
    const int64 out_backprop_depth = out_backprop.dim_size(3);

    TensorShape output_shape;
    auto shape_vec = tensor_in_shape.vec<int32>();
    for (int64 i = 0; i < tensor_in_shape.NumElements(); ++i) {
      output_shape.AddDim(shape_vec(i));
    }
    const int64 in_rows = output_shape.dim_size(1);
    const int64 in_cols = output_shape.dim_size(2);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    output->flat<T>().setZero();

    const int window_rows = ksize_[1];
    const int window_cols = ksize_[2];
    const int depth_window = ksize_[3];

    const int row_stride = stride_[1];
    const int col_stride = stride_[2];

    // We (will) use different code for spatial pooling and
    // non-spatial pooling.
    //
    // Spatial pooling is when depth_window = 1
    OP_REQUIRES(context, depth_window == 1,
                errors::Unimplemented(
                    "Non-spatial pooling is not "
                    "yet supported. Volunteers? :)"));

    int out_height, out_width, pad_rows, pad_cols;
    OP_REQUIRES_OK(
        context, Get2dOutputSize(in_rows, in_cols, window_rows, window_cols,
                                 row_stride, col_stride, padding_, &out_height,
                                 &out_width, &pad_rows, &pad_cols));

    const T* out_backprop_ptr = out_backprop.flat<T>().data();
    T* input_backprop_ptr = output->flat<T>().data();

    for (int64 b = 0; b < out_backprop_batch; ++b) {
      for (int64 r = 0; r < out_backprop_rows; ++r) {
        // Calculates row broadcast size.  For SAME padding, current
        // index could be in the padding area, and r*row_stride +
        // window_rows could be beyond the input tensor's boundary. In
        // such cases, change the starting index and reduce the
        // broadcast size.
        int rindex, rsize;
        OP_REQUIRES_OK(context,
                       GetBroadcastSize(r, in_rows, window_rows, row_stride,
                                        pad_rows, &rindex, &rsize));
        for (int64 c = 0; c < out_backprop_cols; ++c) {
          // Calculates col broadcast size.  For SAME padding, current
          // index could be in the padding area, and c*col_stride +
          // window_cols could be beyond the input tensor's boundary. In
          // such cases, change the starting index and reduce the
          // broadcast size.
          int cindex, csize;
          OP_REQUIRES_OK(context,
                         GetBroadcastSize(c, in_cols, window_cols, col_stride,
                                          pad_cols, &cindex, &csize));

          T divide_coeff = 1.0 / (rsize * csize);
          int64 output_index =
              (b * out_backprop_rows + r) * out_backprop_cols + c;
          for (int64 r_dst = rindex; r_dst < rindex + rsize; ++r_dst) {
            for (int64 c_dst = cindex; c_dst < cindex + csize; ++c_dst) {
              int64 input_index = (b * in_rows + r_dst) * in_cols + c_dst;
              const T* output_offset =
                  out_backprop_ptr + output_index * out_backprop_depth;
              T* input_offset =
                  input_backprop_ptr + input_index * out_backprop_depth;
              for (int64 d = 0; d < out_backprop_depth; ++d) {
                *input_offset += *output_offset * divide_coeff;
                ++output_offset;
                ++input_offset;
              }
            }
          }
        }
      }
    }
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
};

REGISTER_KERNEL_BUILDER(Name("AvgPoolGrad")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T")
                            .HostMemory("orig_input_shape"),
                        AvgPoolingGradOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("AvgPoolGrad")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<double>("T")
                            .HostMemory("orig_input_shape"),
                        AvgPoolingGradOp<CPUDevice, double>);

#if GOOGLE_CUDA

// A CUDNN based AvgPoolingGrad implementation. It includes the padding as the
// candidates for the pooling operation.
template <class T>
class AvgPoolingGradOp<GPUDevice, T> : public OpKernel {
 public:
  typedef GPUDevice Device;

  explicit AvgPoolingGradOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in_shape = context->input(0);
    const Tensor& out_backprop = context->input(1);
    // For avgpooling, tensor_in_shape should have 1 dimension, and 4 elements.
    OP_REQUIRES(
        context,
        tensor_in_shape.dims() == 1 && tensor_in_shape.NumElements() == 4,
        errors::InvalidArgument("out_backprop must be 1-dimensional and 4 "
                                "elements"));
    // For avgpooling, out_backprop should have 4 dimensions.
    OP_REQUIRES(context, out_backprop.dims() == 4,
                errors::InvalidArgument("out_backprop must be 4-dimensional"));

    TensorShape output_shape;
    auto shape_vec = tensor_in_shape.vec<int32>();
    for (int64 i = 0; i < tensor_in_shape.NumElements(); ++i) {
      output_shape.AddDim(shape_vec(i));
    }

    DnnPoolingGradOp<T>::Compute(
        context, perftools::gputools::dnn::PoolingMode::kAverage, ksize_,
        stride_, padding_, nullptr, nullptr, out_backprop, output_shape);
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
};

REGISTER_KERNEL_BUILDER(Name("AvgPoolGrad")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T")
                            .HostMemory("orig_input_shape")
                            .Label("cudnn"),
                        AvgPoolingGradOp<GPUDevice, float>);

// A custom GPU kernel based AvgPoolingGrad implementation. It includes the
// padding as the candidates for the pooling operation.
template <class T>
class AvgPoolingGradOpCustomGPUKernel : public OpKernel {
 public:
  typedef GPUDevice Device;

  explicit AvgPoolingGradOpCustomGPUKernel(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in_shape = context->input(0);
    const Tensor& out_backprop = context->input(1);
    // For avgpooling, tensor_in_shape should have 1 dimension, and 4 elements.
    OP_REQUIRES(
        context,
        tensor_in_shape.dims() == 1 && tensor_in_shape.NumElements() == 4,
        errors::InvalidArgument("out_backprop must be 1-dimensional and 4 "
                                "elements"));
    // For avgpooling, out_backprop should have 4 dimensions.
    OP_REQUIRES(context, out_backprop.dims() == 4,
                errors::InvalidArgument("out_backprop must be 4-dimensional"));
    const int64 out_backprop_batch = out_backprop.dim_size(0);
    const int64 out_backprop_rows = out_backprop.dim_size(1);
    const int64 out_backprop_cols = out_backprop.dim_size(2);
    const int64 out_backprop_depth = out_backprop.dim_size(3);

    TensorShape output_shape;
    auto shape_vec = tensor_in_shape.vec<int32>();
    for (int64 i = 0; i < tensor_in_shape.NumElements(); ++i) {
      output_shape.AddDim(shape_vec(i));
    }
    const int64 in_rows = output_shape.dim_size(1);
    const int64 in_cols = output_shape.dim_size(2);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    const int window_rows = ksize_[1];
    const int window_cols = ksize_[2];
    const int depth_window = ksize_[3];

    const int row_stride = stride_[1];
    const int col_stride = stride_[2];

    // We (will) use different code for spatial pooling and
    // non-spatial pooling.
    //
    // Spatial pooling is when depth_window = 1
    OP_REQUIRES(context, depth_window == 1,
                errors::Unimplemented("Non-spatial pooling is not "
                                      "yet supported. Volunteers? :)"));

    int out_height, out_width, pad_rows, pad_cols;
    OP_REQUIRES_OK(
        context, Get2dOutputSize(in_rows, in_cols, window_rows, window_cols,
                                 row_stride, col_stride, padding_, &out_height,
                                 &out_width, &pad_rows, &pad_cols));

    RunAvePoolBackwardNHWC<T>(out_backprop.flat<T>().data(),  // top_diff
                              out_backprop_batch,             // num
                              in_rows,                        // height
                              in_cols,                        // width
                              out_backprop_depth,             // channels
                              out_backprop_rows,              // pooled_height
                              out_backprop_cols,              // pooled_width
                              window_rows,                    // kernel_h
                              window_cols,                    // kernel_w
                              row_stride,                     // stride_h
                              col_stride,                     // stride_w
                              pad_rows,                       // pad_t
                              pad_cols,                       // pad_l
                              output->flat<T>().data(),       // bottom_diff
                              context->eigen_gpu_device());   // d
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
};

REGISTER_KERNEL_BUILDER(Name("AvgPoolGrad")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T")
                            .HostMemory("orig_input_shape"),
                        AvgPoolingGradOpCustomGPUKernel<float>);

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
