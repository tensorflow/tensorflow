// See docs in ../ops/nn_ops.cc.

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/use_cudnn.h"

#if GOOGLE_CUDA
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/core/common_runtime/gpu_device_context.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// The operation to compute Conv2D gradients.
//
//
// To compute the gradients for Conv2D, we need three input tensors:
//    input, filter, and backprop for output.
// And we need to compute two backprops: one for input and one for filter. We
// compute them in two different kernels.

// Both backprops can be computed as straightforward conv2d.
//
// Consider a case where the input is 3x3 and the filter is 2x1:
//
// INPUT = [ A  B  C ]
//         [ D  E  F ]
//         [ G  H  I ]
//
// where each "A", "B", etc is batch x in_depth
//
// FILTER = [ X  Y ]
//
// where both "X" and "Y" are in_depth x out_depth
//
// With VALID padding, the output is 3x2:
//
// OUTPUT = [ a  b ]
//          [ c  d ]
//          [ e  f ]
//
// where each "a", "b", etc is batch x out_depth
//
// So we have:
//
//   a = A * X + B * Y
//   b = B * X + C * Y
//   c = D * X + E * Y
//   d = E * X + F * Y
//   e = G * X + H * Y
//   f = H * X + I * Y
//
// So when we have backprops for the outputs (we denote them by
// a', b', ... ):
//
// The backprops for the input are:
//
//   A' = a' * X^t
//   B' = a' * Y^t + b' * X^t
//   C' = b' * Y^t
//   ...
//
// This is essentially computing a 2d conv of
//
// INPUT = [ 0  a'  b'  0 ]
//         [ 0  c'  d'  0 ]
//         [ 0  e'  f'  0 ]
// and
//
// FILTER = [ Y^t X^t ]
//
// The backprops for the filter are:
//
//   X' = A^t * a' + B^t * b' + D^t * c' + E^t * d' + G^t * e' + H^t * f'
//   Y' = B^t * a' + C^t * b' + E^t + c' + F^t * d' + H^t * e' + I^t * f'
//
// This is essentially computing a 2d conv of
//
// INPUT = [ A^t  B^t  C^t ]
//         [ D^t  E^t  F^t ]
//         [ G^t  H^t  I^t ]
//
// and
//
// FILTER = [ a'  b' ]
//          [ c'  d' ]
//          [ e'  f' ]
//
//
//////////////////////////////////////////////////////////
//
// With stride more than one, it's a bit more complicated (we will need to
// create holes to the backprop).
//
// Consider the case where
//
// INPUT = [ A B C D E ]
//         [ F G H I J ]
//         [ K L M N O ]
// and
//
// FILTER = [ X Y Z ]
//
// with stride 2.
//
// The output will be
//
// OUTPUT = [ a b ]
//          [ c d ]
//
// where:
//
//   a = A * X + B * Y + C * Z
//   b = C * X + D * Y + E * Z
//   c = K * X + L * Y + M * Z
//   d = M * X + N * Y + O * Z
//
//
// To compute the backprop for INPUT, we need to convolve
//
// INPUT = [ 0  0  a' 0  b' 0  0 ]
//         [ 0  0  0  0  0  0  0 ]
//         [ 0  0  c' 0  d' 0  0 ]
//
// (notice the holes in INPUT)
//
// and
//
// FILTER = [ Z^t  Y^t  X^t ]
//
// with stride 1.
//
// To compute the backprop for FILTER, we need to convolve

//
// INPUT = [ A^t  B^t  C^t  D^t  E^t ]
//         [ F^t  G^t  H^t  I^t  J^t ]
//         [ K^t  L^t  M^t  N^t  O^t ]
// and
//
// FILTER = [ a' 0  b' ]
//          [ 0  0  0  ]
//          [ c' 0  d' ]
//
// (notice the holes in FILTER)
//
//
// with stride 1
//
//////////////////////////////////////////////////////////
//
//
// The case for SAME padding is in fact very similar to VALID -- we just
// need to pad the input tensor a bit when computing the filter_backprop.

// Common code between the two kernels: verifies that the dimensions all match
// and extract the padded rows and columns.
#define EXTRACT_AND_VERIFY_DIMENSIONS(label)                                   \
  const Tensor& out_backprop = context->input(2);                              \
  OP_REQUIRES(                                                                 \
      context, input_shape.dims() == 4,                                        \
      errors::InvalidArgument(label, ": input must be 4-dimensional"));        \
  OP_REQUIRES(                                                                 \
      context, filter_shape.dims() == 4,                                       \
      errors::InvalidArgument(label, ": filter must be 4-dimensional"));       \
  OP_REQUIRES(                                                                 \
      context, out_backprop.dims() == 4,                                       \
      errors::InvalidArgument(label, ": out_backprop must be 4-dimensional")); \
  const int64 batch = input_shape.dim_size(0);                                 \
  OP_REQUIRES(                                                                 \
      context, batch == out_backprop.dim_size(0),                              \
      errors::InvalidArgument(                                                 \
          label, ": input and out_backprop must have the same batch size"));   \
  const int64 input_rows = input_shape.dim_size(1);                            \
  const int64 input_cols = input_shape.dim_size(2);                            \
  const int64 filter_rows = filter_shape.dim_size(0);                          \
  const int64 filter_cols = filter_shape.dim_size(1);                          \
  const int64 output_rows = out_backprop.dim_size(1);                          \
  const int64 output_cols = out_backprop.dim_size(2);                          \
  const int64 in_depth = input_shape.dim_size(3);                              \
  OP_REQUIRES(context, in_depth == filter_shape.dim_size(2),                   \
              errors::InvalidArgument(                                         \
                  label, ": input and filter must have the same depth"));      \
  const int64 out_depth = filter_shape.dim_size(3);                            \
  OP_REQUIRES(                                                                 \
      context, out_depth == out_backprop.dim_size(3),                          \
      errors::InvalidArgument(                                                 \
          label, ": filter and out_backprop must have the same out_depth"));   \
  const auto stride = strides_[1];                                             \
  int out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;                  \
  if (filter_cols == filter_rows && filter_rows == 1 && stride == 1) {         \
    out_rows = input_rows;                                                     \
    out_cols = input_cols;                                                     \
  } else {                                                                     \
    OP_REQUIRES_OK(                                                            \
        context, Get2dOutputSize(input_rows, input_cols, filter_rows,          \
                                 filter_cols, stride, stride, padding_,        \
                                 &out_rows, &out_cols, &pad_rows, &pad_cols)); \
  }                                                                            \
  OP_REQUIRES(                                                                 \
      context, output_rows == out_rows,                                        \
      errors::InvalidArgument(                                                 \
          label, ": Number of rows of out_backprop doesn't match computed: ",  \
          "actual = ", output_rows, ", computed = ", out_rows));               \
  OP_REQUIRES(                                                                 \
      context, output_cols == out_cols,                                        \
      errors::InvalidArgument(                                                 \
          label, ": Number of cols of out_backprop doesn't match computed: ",  \
          "actual = ", output_cols, ", computed = ", out_cols));               \
  const auto expanded_out_rows = (output_rows - 1) * stride + 1;               \
  const auto expanded_out_cols = (output_cols - 1) * stride + 1;               \
  const auto padded_out_rows = input_rows + filter_rows - 1;                   \
  const auto padded_out_cols = input_cols + filter_cols - 1;                   \
  const auto top_pad_rows = filter_rows - 1 - pad_rows;                        \
  const auto left_pad_cols = filter_cols - 1 - pad_cols;                       \
  const auto bottom_pad_rows =                                                 \
      padded_out_rows - expanded_out_rows - top_pad_rows;                      \
  const auto right_pad_cols =                                                  \
      padded_out_cols - expanded_out_cols - left_pad_cols;                     \
  Eigen::DSizes<Eigen::DenseIndex, 4> strides{1, stride, stride, 1};           \
  VLOG(2) << "Conv2d: " << label                                               \
          << ": expanded_out_rows = " << expanded_out_rows                     \
          << ", expanded_out_cols = " << expanded_out_cols                     \
          << ", filter_rows = " << filter_rows                                 \
          << ", filter_cols = " << filter_cols                                 \
          << ", padded_out_rows = " << padded_out_rows                         \
          << ", padded_out_cols = " << padded_out_cols                         \
          << ", top_pad_rows = " << top_pad_rows                               \
          << ", left_pad_cols = " << left_pad_cols                             \
          << ", bottom_pad_rows = " << bottom_pad_rows                         \
          << ", right_pad_cols = " << right_pad_cols                           \
          << ", strides = " << strides[1]

namespace {
TensorShape VectorToShape(const TTypes<int32>::ConstVec& sizes) {
  TensorShape shape;

  using Index = TTypes<int32>::ConstVec::Index;
  const Index dims = sizes.size();
  for (Index i = 0; i < dims; ++i) {
    shape.AddDim(sizes(i));
  }

  return shape;
}
}  // namespace

// The fast versions using eigen computations directly. They are only enabled
// for CPU for now since nvcc times out when trying to compile them.
// TODO(yangke): enable them for GPUs when we have a faster compiler.

template <typename Device, class T>
class Conv2DFastBackpropInputOp : public OpKernel {
 public:
  explicit Conv2DFastBackpropInputOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(context, strides_[1] == strides_[2],
                errors::InvalidArgument(
                    "Current implementation only supports equal length "
                    "strides in the row and column dimensions."));
    OP_REQUIRES(
        context, (strides_[0] == 1 && strides_[3] == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_sizes = context->input(0);
    const Tensor& filter = context->input(1);
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(input_sizes.shape()),
        errors::InvalidArgument(
            "Conv2DBackpropInput: input_sizes input must be 1-dim, not ",
            input_sizes.dims()));
    TensorShape input_shape = VectorToShape(input_sizes.vec<int32>());
    const TensorShape& filter_shape = filter.shape();

    EXTRACT_AND_VERIFY_DIMENSIONS("Conv2DBackpropInput");
    Tensor* in_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_shape, &in_backprop));
    // Need to flip the input_rows and input_cols when passing to eigen.
    functor::SpatialConvolutionBackwardInput<Device, T>()(
        context->eigen_device<Device>(), in_backprop->tensor<T, 4>(),
        filter.tensor<T, 4>(), out_backprop.tensor<T, 4>(), input_cols,
        input_rows, stride);
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv2DFastBackpropInputOp);
};

// Based on implementation written by Yangqing Jia (jiayq).
template <typename Device, class T>
class Conv2DCustomBackpropInputOp : public OpKernel {
 public:
  explicit Conv2DCustomBackpropInputOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(context, strides_[1] == strides_[2],
                errors::InvalidArgument(
                    "Current implementation only supports equal length "
                    "strides in the row and column dimensions."));
    OP_REQUIRES(
        context, (strides_[0] == 1 && strides_[3] == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_sizes = context->input(0);
    const Tensor& filter = context->input(1);
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(input_sizes.shape()),
        errors::InvalidArgument(
            "Conv2DBackpropInput: input_sizes input must be 1-dim, not ",
            input_sizes.dims()));
    TensorShape input_shape = VectorToShape(input_sizes.vec<int32>());
    const TensorShape& filter_shape = filter.shape();

    EXTRACT_AND_VERIFY_DIMENSIONS("Conv2DBackpropInput");
    Tensor* in_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_shape, &in_backprop));

    // TODO(andydavis) Consider moving code shared with
    // Conv2DCustomBackpropFilterOp into a shared helper function.
    int pad_top;
    int pad_bottom;
    int pad_left;
    int pad_right;
    OP_REQUIRES_OK(
        context,
        Get2dOutputSizeVerbose(input_rows, input_cols, filter_rows, filter_cols,
                               stride, stride, padding_, &out_rows, &out_cols,
                               &pad_top, &pad_bottom, &pad_left, &pad_right));

    // The total dimension size of each kernel.
    const int filter_total_size = filter_rows * filter_cols * in_depth;
    // The output image size is the spatial size of the output.
    const int output_image_size = out_rows * out_cols;

    Tensor col_buffer;
    OP_REQUIRES_OK(
        context,
        context->allocate_temp(
            DataTypeToEnum<T>::value,
            TensorShape({output_image_size, filter_total_size}), &col_buffer));

    // The input offset corresponding to a single input image.
    const int input_offset = input_rows * input_cols * in_depth;
    // The output offset corresponding to a single output image.
    const int output_offset = out_rows * out_cols * out_depth;

    auto* filter_data = filter.template flat<T>().data();
    auto* col_buffer_data = col_buffer.template flat<T>().data();
    auto* out_backprop_data = out_backprop.template flat<T>().data();
    auto* input_backprop_data = in_backprop->template flat<T>().data();

    typedef Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>,
                             Eigen::Unaligned> TensorMap;
    typedef Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor>,
                             Eigen::Unaligned> ConstTensorMap;

    // Initialize contraction dims (we need to transpose 'B' below).
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> contract_dims;
    contract_dims[0].first = 1;
    contract_dims[0].second = 1;

    for (int image_id = 0; image_id < batch; ++image_id) {
      // Compute gradient into col_buffer.
      TensorMap C(col_buffer_data, output_image_size, filter_total_size);

      ConstTensorMap A(out_backprop_data + output_offset * image_id,
                       output_image_size, out_depth);
      ConstTensorMap B(filter_data, filter_total_size, out_depth);

      C.device(context->eigen_cpu_device()) = A.contract(B, contract_dims);

      Col2im<T>(col_buffer_data, in_depth, input_rows, input_cols, filter_rows,
                filter_cols, pad_top, pad_left, pad_bottom, pad_right, stride,
                stride, input_backprop_data);

      input_backprop_data += input_offset;
    }
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv2DCustomBackpropInputOp);
};

REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropInput")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        Conv2DCustomBackpropInputOp<CPUDevice, float>);

REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropInput")
                            .Device(DEVICE_CPU)
                            .Label("custom")
                            .TypeConstraint<float>("T"),
                        Conv2DCustomBackpropInputOp<CPUDevice, float>);

REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropInput")
                            .Device(DEVICE_CPU)
                            .Label("eigen_tensor")
                            .TypeConstraint<float>("T"),
                        Conv2DFastBackpropInputOp<CPUDevice, float>);

template <typename Device, class T>
class Conv2DFastBackpropFilterOp : public OpKernel {
 public:
  explicit Conv2DFastBackpropFilterOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(context, strides_[1] == strides_[2],
                errors::InvalidArgument(
                    "Current implementation only supports equal length "
                    "strides in the row and column dimensions."));
    OP_REQUIRES(
        context, (strides_[0] == 1 && strides_[3] == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& filter_sizes = context->input(1);
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(filter_sizes.shape()),
        errors::InvalidArgument(
            "Conv2DBackpropFilter: filter_sizes input must be 1-dim, not ",
            filter_sizes.dims()));
    const TensorShape& input_shape = input.shape();
    TensorShape filter_shape = VectorToShape(filter_sizes.vec<int32>());

    EXTRACT_AND_VERIFY_DIMENSIONS("Conv2DBackpropFilter");
    Tensor* filter_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, filter_shape, &filter_backprop));

    // Need to flip the filter_rows and filter_cols when passing to eigen.
    functor::SpatialConvolutionBackwardKernel<Device, T>()(
        context->eigen_device<Device>(), filter_backprop->tensor<T, 4>(),
        input.tensor<T, 4>(), out_backprop.tensor<T, 4>(), filter_cols,
        filter_rows, stride);
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv2DFastBackpropFilterOp);
};

// Based on implementation written by Yangqing Jia (jiayq).
template <typename Device, class T>
class Conv2DCustomBackpropFilterOp : public OpKernel {
 public:
  explicit Conv2DCustomBackpropFilterOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(context, strides_[1] == strides_[2],
                errors::InvalidArgument(
                    "Current implementation only supports equal length "
                    "strides in the row and column dimensions."));
    OP_REQUIRES(
        context, (strides_[0] == 1 && strides_[3] == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& filter_sizes = context->input(1);
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(filter_sizes.shape()),
        errors::InvalidArgument(
            "Conv2DCustomBackpropFilter: filter_sizes input must be 1-dim, "
            "not ",
            filter_sizes.dims()));
    const TensorShape& input_shape = input.shape();
    TensorShape filter_shape = VectorToShape(filter_sizes.vec<int32>());

    EXTRACT_AND_VERIFY_DIMENSIONS("Conv2DCustomBackpropFilter");
    Tensor* filter_backprop;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, filter_shape, &filter_backprop));

    int pad_top;
    int pad_bottom;
    int pad_left;
    int pad_right;
    OP_REQUIRES_OK(
        context,
        Get2dOutputSizeVerbose(input_rows, input_cols, filter_rows, filter_cols,
                               stride, stride, padding_, &out_rows, &out_cols,
                               &pad_top, &pad_bottom, &pad_left, &pad_right));

    // The total dimension size of each kernel.
    const int filter_total_size = filter_rows * filter_cols * in_depth;
    // The output image size is the spatial size of the output.
    const int output_image_size = out_rows * out_cols;

    // Shard 'batch' images into 'shard_size' groups of images to be fed
    // into the parallel matmul. Calculate 'shard_size' by dividing the L3 cache
    // size ('target_working_set_size') by the matmul size of an individual
    // image ('work_unit_size').

    // TODO(andydavis)
    // *) Get L3 cache size from device at runtime (30MB is from ivybridge).
    // *) Consider reducing 'target_working_set_size' if L3 is shared by
    //    other concurrently running tensorflow ops.
    const size_t target_working_set_size = (30LL << 20) / sizeof(T);

    const size_t size_A = output_image_size * filter_total_size;

    const size_t size_B = output_image_size * out_depth;

    const size_t size_C = filter_total_size * out_depth;

    const size_t work_unit_size = size_A + size_B + size_C;

    const size_t shard_size =
        (target_working_set_size + work_unit_size - 1) / work_unit_size;

    Tensor col_buffer;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(
                       DataTypeToEnum<T>::value,
                       TensorShape({static_cast<int64>(shard_size),
                                    static_cast<int64>(output_image_size),
                                    static_cast<int64>(filter_total_size)}),
                       &col_buffer));

    // The input offset corresponding to a single input image.
    const int input_offset = input_rows * input_cols * in_depth;
    // The output offset corresponding to a single output image.
    const int output_offset = out_rows * out_cols * out_depth;

    auto* input_data = input.template flat<T>().data();
    auto* col_buffer_data = col_buffer.template flat<T>().data();
    auto* out_backprop_data = out_backprop.template flat<T>().data();
    auto* filter_backprop_data = filter_backprop->template flat<T>().data();

    typedef Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>,
                             Eigen::Unaligned> TensorMap;
    typedef Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor>,
                             Eigen::Unaligned> ConstTensorMap;

    TensorMap C(filter_backprop_data, filter_total_size, out_depth);
    C.setZero();

    // Initialize contraction dims (we need to transpose 'A' below).
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> contract_dims;
    contract_dims[0].first = 0;
    contract_dims[0].second = 0;

    for (int image_id = 0; image_id < batch; image_id += shard_size) {
      const int shard_limit = std::min(static_cast<int>(shard_size),
                                       static_cast<int>(batch) - image_id);
      for (int shard_id = 0; shard_id < shard_limit; ++shard_id) {
        // TODO(andydavis) Parallelize this loop.
        // When we compute the gradient with respect to the filters, we need
        // to do im2col to allow gemm-type computation.
        Im2col<T>(input_data, in_depth, input_rows, input_cols, filter_rows,
                  filter_cols, pad_top, pad_left, pad_bottom, pad_right, stride,
                  stride, col_buffer_data + shard_id * size_A);

        input_data += input_offset;
      }

      ConstTensorMap A(col_buffer_data, output_image_size * shard_limit,
                       filter_total_size);
      ConstTensorMap B(out_backprop_data, output_image_size * shard_limit,
                       out_depth);

      // Gradient with respect to filter.
      C.device(context->eigen_cpu_device()) += A.contract(B, contract_dims);

      out_backprop_data += output_offset * shard_limit;
    }
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv2DCustomBackpropFilterOp);
};

REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropFilter")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        Conv2DCustomBackpropFilterOp<CPUDevice, float>);

REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropFilter")
                            .Device(DEVICE_CPU)
                            .Label("custom")
                            .TypeConstraint<float>("T"),
                        Conv2DCustomBackpropFilterOp<CPUDevice, float>);

REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropFilter")
                            .Device(DEVICE_CPU)
                            .Label("eigen_tensor")
                            .TypeConstraint<float>("T"),
                        Conv2DFastBackpropFilterOp<CPUDevice, float>);

// GPU definitions of both ops.
#if GOOGLE_CUDA
namespace {
template <typename T>
perftools::gputools::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory,
                                                    uint64 size) {
  perftools::gputools::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory),
                                                size * sizeof(T));
  perftools::gputools::DeviceMemory<T> typed(wrapped);
  return typed;
}
}  // namespace

// The slow version (but compiles for GPU)

// Backprop for input.
template <typename Device, class T>
class Conv2DSlowBackpropInputOp : public OpKernel {
 public:
  explicit Conv2DSlowBackpropInputOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(context, strides_[1] == strides_[2],
                errors::InvalidArgument(
                    "Current implementation only supports equal length "
                    "strides in the row and column dimensions."));
    OP_REQUIRES(
        context, (strides_[0] == 1 && strides_[3] == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("use_cudnn_on_gpu", &use_cudnn_));
    use_cudnn_ &= CanUseCudnn();
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_sizes = context->input(0);
    const Tensor& filter = context->input(1);
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(input_sizes.shape()),
        errors::InvalidArgument(
            "Conv2DBackpropInput: input_sizes input must be 1-dim, not ",
            input_sizes.dims()));
    TensorShape input_shape = VectorToShape(input_sizes.vec<int32>());
    const TensorShape& filter_shape = filter.shape();

    EXTRACT_AND_VERIFY_DIMENSIONS("Conv2DBackpropInput");
    Tensor* in_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_shape, &in_backprop));

    const int padding_rows =
        (output_rows - 1) * stride + filter_rows - input_rows;
    const int padding_cols =
        (output_cols - 1) * stride + filter_cols - input_cols;

    // TODO(keveman): cuDNN only supports equal padding on both sides, so only
    // calling it when that is true. Remove this check when (if?) cuDNN starts
    // supporting different padding.
    bool padding_compatible =
        (padding_rows % 2 == 0) && (padding_cols % 2 == 0);

    auto* stream = context->op_device_context<GPUDeviceContext>()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

    if (use_cudnn_ && padding_compatible) {
      if (filter_rows == 1 && filter_cols == 1 && stride == 1) {
        // 1x1 filter, so call cublas directly.
        const uint64 m = batch * input_rows * input_cols;
        const uint64 k = out_depth;
        const uint64 n = in_depth;

        auto a_ptr = AsDeviceMemory(out_backprop.template flat<T>().data(),
                                    out_backprop.template flat<T>().size());
        auto b_ptr = AsDeviceMemory(filter.template flat<T>().data(),
                                    filter.template flat<T>().size());
        auto c_ptr = AsDeviceMemory(in_backprop->template flat<T>().data(),
                                    in_backprop->template flat<T>().size());

        auto transpose = perftools::gputools::blas::Transpose::kTranspose;
        auto no_transpose = perftools::gputools::blas::Transpose::kNoTranspose;

        bool blas_launch_status =
            stream->ThenBlasGemm(transpose, no_transpose, n, m, k, 1.0f, b_ptr,
                                 k, a_ptr, k, 0.0f, &c_ptr, n)
                .ok();
        if (!blas_launch_status) {
          context->SetStatus(errors::Internal("Blas SGEMM launch failed : m=",
                                              m, ", n=", n, ", k=", k));
        }
        return;
      }

      perftools::gputools::dnn::BatchDescriptor input_desc;
      input_desc.set_count(batch)
          .set_height(input_rows)
          .set_width(input_cols)
          .set_feature_map_count(in_depth)
          .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);
      perftools::gputools::dnn::BatchDescriptor output_desc;
      output_desc.set_count(batch)
          .set_height(output_rows)
          .set_width(output_cols)
          .set_feature_map_count(out_depth)
          .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);
      perftools::gputools::dnn::FilterDescriptor filter_desc;
      filter_desc.set_input_filter_height(filter_rows)
          .set_input_filter_width(filter_cols)
          .set_input_feature_map_count(in_depth)
          .set_output_feature_map_count(out_depth);
      perftools::gputools::dnn::ConvolutionDescriptor conv_desc;
      conv_desc.set_vertical_filter_stride(stride)
          .set_horizontal_filter_stride(stride)
          .set_zero_padding_height(padding_rows / 2)
          .set_zero_padding_width(padding_cols / 2);

      // NOTE(keveman):
      // cuDNN only supports the following layouts :
      // Input  : B x D x R x C
      // Filter : OD x ID x R x C
      // Whereas, we have
      // Input  : B x R x C x D
      // Filter : R x C x ID x OD
      // TransformFilter performs (R x C x ID x OD) => (OD x ID x R x C)
      // The first TransformDepth performs
      // (B x R x C x D) => (B x D x R x C).
      // Since the tensor returned from cuDNN is B x D x R x C also,
      // the second TransformDepth performs
      // (B x D x R x C) => (B x R x C x D).
      Tensor transformed_filter;
      OP_REQUIRES_OK(
          context,
          context->allocate_temp(
              DataTypeToEnum<T>::value,
              TensorShape({out_depth, in_depth, filter_rows, filter_cols}),
              &transformed_filter));

      functor::TransformFilter<Device, T, int>()(
          context->eigen_device<Device>(), To32Bit(filter.tensor<T, 4>()),
          To32Bit(transformed_filter.tensor<T, 4>()));

      Tensor transformed_out_backprop;
      OP_REQUIRES_OK(
          context,
          context->allocate_temp(
              DataTypeToEnum<T>::value,
              TensorShape({batch, out_depth, output_rows, output_cols}),
              &transformed_out_backprop));

      functor::TransformDepth<Device, T, int>()(
          context->eigen_device<Device>(), To32Bit(out_backprop.tensor<T, 4>()),
          Eigen::DSizes<int, 4>(0, 3, 1, 2),
          To32Bit(transformed_out_backprop.tensor<T, 4>()));

      Tensor pre_transformed_in_backprop;
      OP_REQUIRES_OK(context,
                     context->allocate_temp(
                         DataTypeToEnum<T>::value,
                         TensorShape({batch, in_depth, input_rows, input_cols}),
                         &pre_transformed_in_backprop));

      auto out_backprop_ptr =
          AsDeviceMemory(transformed_out_backprop.template flat<T>().data(),
                         transformed_out_backprop.template flat<T>().size());
      auto filter_ptr =
          AsDeviceMemory(transformed_filter.template flat<T>().data(),
                         transformed_filter.template flat<T>().size());
      auto in_backprop_ptr =
          AsDeviceMemory(pre_transformed_in_backprop.template flat<T>().data(),
                         pre_transformed_in_backprop.template flat<T>().size());

      bool cudnn_launch_status =
          stream->ThenConvolveBackwardData(filter_desc, filter_ptr, output_desc,
                                           out_backprop_ptr, conv_desc,
                                           input_desc, &in_backprop_ptr)
              .ok();

      if (!cudnn_launch_status) {
        context->SetStatus(errors::Internal(
            "cuDNN Backward Data function launch failure : input shape(",
            input_shape.DebugString(), ") filter shape(",
            filter_shape.DebugString(), ")"));
      }

      auto toConstTensor = [](const Tensor& x) -> const Tensor { return x; };
      functor::TransformDepth<Device, T, int>()(
          context->eigen_device<Device>(),
          To32Bit(toConstTensor(pre_transformed_in_backprop)
                      .template tensor<T, 4>()),
          Eigen::DSizes<int, 4>(0, 2, 3, 1),
          To32Bit(in_backprop->tensor<T, 4>()));
    } else {
      // We fill out a padded out_backprop
      TensorShape padded_out_shape(
          {batch, padded_out_rows, padded_out_cols, out_depth});
      Tensor padded_output;
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::v(),
                                            padded_out_shape, &padded_output));

      Eigen::DSizes<Eigen::DenseIndex, 4> trivial_order{0, 1, 2, 3};
      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 4> pad_dims{
          {{0, 0},
           {top_pad_rows, bottom_pad_rows},
           {left_pad_cols, right_pad_cols},
           {0, 0}}};

      functor::InflatePadAndShuffle<Device, T, 4, Eigen::DenseIndex>()(
          context->eigen_device<Device>(), out_backprop.tensor<T, 4>(), strides,
          pad_dims, trivial_order, padded_output.tensor<T, 4>());
      const Tensor& padded_output_cref = padded_output;

      // We then need to fill a new "reverted" filter
      // We need to transpose the in_depth and out_depth for the filter and
      // inverse the rows and cols.
      TensorShape r_filter_shape(
          {filter_rows, filter_cols, out_depth, in_depth});
      Tensor r_filter;
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::v(),
                                            r_filter_shape, &r_filter));

      Eigen::DSizes<Eigen::DenseIndex, 4> filter_order{0, 1, 3, 2};
      Eigen::array<bool, 4> filter_rev_dims{true, true, false, false};
      functor::ShuffleAndReverse<Device, T, 4, Eigen::DenseIndex>()(
          context->eigen_device<Device>(), filter.tensor<T, 4>(), filter_order,
          filter_rev_dims, r_filter.tensor<T, 4>());
      const Tensor& r_filter_cref = r_filter;

      // Now we can call conv_2d directly.
      functor::SpatialConvolution<Device, T>()(
          context->eigen_device<Device>(), in_backprop->tensor<T, 4>(),
          padded_output_cref.tensor<T, 4>(), r_filter_cref.tensor<T, 4>(), 1,
          BrainPadding2EigenPadding(VALID));
    }
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;
  bool use_cudnn_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv2DSlowBackpropInputOp);
};

// Backprop for filter.
template <typename Device, class T>
class Conv2DSlowBackpropFilterOp : public OpKernel {
 public:
  explicit Conv2DSlowBackpropFilterOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(context, strides_[1] == strides_[2],
                errors::InvalidArgument(
                    "Current implementation only supports equal length "
                    "strides in the row and column dimensions."));
    OP_REQUIRES(
        context, (strides_[0] == 1 && strides_[3] == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("use_cudnn_on_gpu", &use_cudnn_));
    use_cudnn_ &= CanUseCudnn();
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& filter_sizes = context->input(1);
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(filter_sizes.shape()),
        errors::InvalidArgument(
            "Conv2DBackpropFilter: filter_sizes input must be 1-dim, not ",
            filter_sizes.dims()));
    const TensorShape& input_shape = input.shape();
    TensorShape filter_shape = VectorToShape(filter_sizes.vec<int32>());

    EXTRACT_AND_VERIFY_DIMENSIONS("Conv2DBackpropFilter");
    Tensor* filter_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, filter_shape, &filter_backprop));

    const int padding_rows =
        (output_rows - 1) * stride + filter_rows - input_rows;
    const int padding_cols =
        (output_cols - 1) * stride + filter_cols - input_cols;

    // TODO(zhengxq): cuDNN only supports equal padding on both sides, so only
    // calling it when that is true. Remove this check when (if?) cuDNN starts
    // supporting different padding.
    bool padding_compatible =
        (padding_rows % 2 == 0) && (padding_cols % 2 == 0);

    auto* stream = context->op_device_context<GPUDeviceContext>()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

    if (use_cudnn_ && padding_compatible) {
      if (filter_rows == 1 && filter_cols == 1 && stride == 1) {
        const uint64 m = in_depth;
        const uint64 k = batch * input_rows * input_cols;
        const uint64 n = out_depth;

        // The shape of output backprop is
        //   [batch, out_rows, out_cols, out_depth]
        //   From cublas's perspective, it is: n x k
        auto a_ptr = AsDeviceMemory(out_backprop.template flat<T>().data(),
                                    out_backprop.template flat<T>().size());

        // The shape of input is
        //   [batch, in_rows, in_cols, in_depth],
        //   From cublas's perspective, it is: m x k
        auto b_ptr = AsDeviceMemory(input.template flat<T>().data(),
                                    input.template flat<T>().size());

        // the shape of the filter backprop from the conv_2d should be
        //   [1, 1, in_depth, out_depth]
        //   From cublas's perspective, it is: n x m
        auto c_ptr = AsDeviceMemory(filter_backprop->template flat<T>().data(),
                                    filter_backprop->template flat<T>().size());

        bool blas_launch_status =
            stream->ThenBlasGemm(
                      perftools::gputools::blas::Transpose::kNoTranspose,
                      perftools::gputools::blas::Transpose::kTranspose, n, m, k,
                      1.0f, a_ptr, n, b_ptr, m, 0.0f, &c_ptr, n)
                .ok();
        if (!blas_launch_status) {
          context->SetStatus(errors::Internal("Blas SGEMM launch failed : m=",
                                              m, ", n=", n, ", k=", k));
        }
        return;
      }

      perftools::gputools::dnn::BatchDescriptor input_desc;
      input_desc.set_count(batch)
          .set_height(input_rows)
          .set_width(input_cols)
          .set_feature_map_count(in_depth)
          .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);
      perftools::gputools::dnn::BatchDescriptor output_desc;
      output_desc.set_count(batch)
          .set_height(output_rows)
          .set_width(output_cols)
          .set_feature_map_count(out_depth)
          .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);
      perftools::gputools::dnn::FilterDescriptor filter_desc;
      filter_desc.set_input_filter_height(filter_rows)
          .set_input_filter_width(filter_cols)
          .set_input_feature_map_count(in_depth)
          .set_output_feature_map_count(out_depth);
      perftools::gputools::dnn::ConvolutionDescriptor conv_desc;
      conv_desc.set_vertical_filter_stride(stride)
          .set_horizontal_filter_stride(stride)
          .set_zero_padding_height(padding_rows / 2)
          .set_zero_padding_width(padding_cols / 2);

      // NOTE(zhengxq):
      // cuDNN only supports the following layouts :
      // Input  : B x D x R x C
      // Filter : OD x ID x R x C
      // Whereas, we have
      // Input  : B x R x C x D
      // Filter : R x C x ID x OD
      // TransformFilter performs (R x C x ID x OD) => (OD x ID x R x C)
      // The first TransformDepth performs
      // (B x R x C x D) => (B x D x R x C).
      // Since the tensor returned from cuDNN is B x D x R x C also,
      // the second TransformDepth performs
      // (B x D x R x C) => (B x R x C x D).

      Tensor pre_transformed_filter_backprop;
      OP_REQUIRES_OK(
          context,
          context->allocate_temp(
              DataTypeToEnum<T>::value,
              TensorShape({out_depth, in_depth, filter_rows, filter_cols}),
              &pre_transformed_filter_backprop));

      Tensor transformed_out_backprop;
      OP_REQUIRES_OK(
          context,
          context->allocate_temp(
              DataTypeToEnum<T>::value,
              TensorShape({batch, out_depth, output_rows, output_cols}),
              &transformed_out_backprop));

      functor::TransformDepth<Device, T, int>()(
          context->eigen_device<Device>(), To32Bit(out_backprop.tensor<T, 4>()),
          Eigen::DSizes<int, 4>(0, 3, 1, 2),
          To32Bit(transformed_out_backprop.tensor<T, 4>()));

      Tensor transformed_input;
      OP_REQUIRES_OK(context,
                     context->allocate_temp(
                         DataTypeToEnum<T>::value,
                         TensorShape({batch, in_depth, input_rows, input_cols}),
                         &transformed_input));

      functor::TransformDepth<Device, T, int>()(
          context->eigen_device<Device>(), To32Bit(input.tensor<T, 4>()),
          Eigen::DSizes<int, 4>(0, 3, 1, 2),
          To32Bit(transformed_input.tensor<T, 4>()));

      auto out_backprop_ptr =
          AsDeviceMemory(transformed_out_backprop.template flat<T>().data(),
                         transformed_out_backprop.template flat<T>().size());
      auto filter_backprop_ptr = AsDeviceMemory(
          pre_transformed_filter_backprop.template flat<T>().data(),
          pre_transformed_filter_backprop.template flat<T>().size());
      auto input_ptr =
          AsDeviceMemory(transformed_input.template flat<T>().data(),
                         transformed_input.template flat<T>().size());

      bool cudnn_launch_status =
          stream->ThenConvolveBackwardFilter(input_desc, input_ptr, output_desc,
                                             out_backprop_ptr, conv_desc,
                                             filter_desc, &filter_backprop_ptr)
              .ok();

      if (!cudnn_launch_status) {
        context->SetStatus(errors::Internal(
            "cuDNN Backward Filter function launch failure : input shape(",
            input_shape.DebugString(), ") filter shape(",
            filter_shape.DebugString(), ")"));
      }

      auto toConstTensor = [](const Tensor& x) -> const Tensor { return x; };
      functor::TransformDepth<Device, T, int>()(
          context->eigen_device<Device>(),
          To32Bit(toConstTensor(pre_transformed_filter_backprop)
                      .template tensor<T, 4>()),
          Eigen::DSizes<int, 4>(2, 3, 1, 0),
          To32Bit(filter_backprop->tensor<T, 4>()));
    } else {
      // Fall back to the non-cudnn code path

      // For the backprop of the filter, we need to also transpose the
      // out_backprop.
      // The shape of backprop is
      //   [batch, out_rows, out_cols, out_depth]
      // And we need to change it to
      //   [out_depth, out_rows, out_cols, batch]
      Eigen::DSizes<Eigen::DenseIndex, 4> out_order{3, 1, 2, 0};
      TensorShape padded_out_shape(
          {out_depth, padded_out_rows, padded_out_cols, batch});
      Tensor padded_output;
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::v(),
                                            padded_out_shape, &padded_output));

      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 4> pad_dims{
          {{0, 0},
           {top_pad_rows, bottom_pad_rows},
           {left_pad_cols, right_pad_cols},
           {0, 0}}};
      functor::InflatePadAndShuffle<Device, T, 4, Eigen::DenseIndex>()(
          context->eigen_device<Device>(), out_backprop.tensor<T, 4>(), strides,
          pad_dims, out_order, padded_output.tensor<T, 4>());
      const Tensor& padded_output_cref = padded_output;

      // For the backprop of the filter, we need to transpose the input.
      // The shape of input is
      //   [batch, in_rows, in_cols, in_depth]
      // And we need to change it to
      //   [in_rows, in_cols, batch, in_depth]
      Eigen::DSizes<Eigen::DenseIndex, 4> in_order{1, 2, 0, 3};
      TensorShape in_shuffle_shape({input_rows, input_cols, batch, in_depth});
      Tensor in_shuffle;
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::v(),
                                            in_shuffle_shape, &in_shuffle));

      // No need for reversing this time.
      Eigen::array<bool, 4> trivial_dims{false, false, false, false};
      functor::ShuffleAndReverse<Device, T, 4, Eigen::DenseIndex>()(
          context->eigen_device<Device>(), input.tensor<T, 4>(), in_order,
          trivial_dims, in_shuffle.tensor<T, 4>());
      const Tensor& in_shuffle_cref = in_shuffle;

      // The output of the conv_2d would be
      //   [out_depth, filter_rows, filter_cols, in_depth]
      // and we need to shuffle it back to
      //   [filter_rows, filter_cols, in_depth, out_depth];
      // And we need to reverse the filter backprops
      // So we need to allocated (sigh) yet another piece of memory to hold the
      // ouptut.
      TensorShape filter_shuffle_shape(
          {out_depth, filter_rows, filter_cols, in_depth});
      Tensor filter_shuffle;
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::v(),
                                                     filter_shuffle_shape,
                                                     &filter_shuffle));

      functor::SpatialConvolution<Device, T>()(
          context->eigen_device<Device>(), filter_shuffle.tensor<T, 4>(),
          padded_output_cref.tensor<T, 4>(), in_shuffle_cref.tensor<T, 4>(), 1,
          BrainPadding2EigenPadding(VALID));

      // Now copy the filter_backprop back to the destination.
      Eigen::DSizes<Eigen::DenseIndex, 4> filter_order{1, 2, 3, 0};
      Eigen::array<bool, 4> filter_rev_dims{true, true, false, false};
      const Tensor& filter_shuffle_cref = filter_shuffle;
      functor::ShuffleAndReverse<Device, T, 4, Eigen::DenseIndex>()(
          context->eigen_device<Device>(), filter_shuffle_cref.tensor<T, 4>(),
          filter_order, filter_rev_dims, filter_backprop->tensor<T, 4>());
    }
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;
  bool use_cudnn_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv2DSlowBackpropFilterOp);
};

// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                  \
  template <>                                                                \
  void ShuffleAndReverse<GPUDevice, T, 4, Eigen::DenseIndex>::operator()(    \
      const GPUDevice& d,                                                    \
      typename TTypes<T, 4, Eigen::DenseIndex>::ConstTensor input,           \
      const Eigen::DSizes<Eigen::DenseIndex, 4>& order,                      \
      const Eigen::array<bool, 4>& reverse_dims,                             \
      typename TTypes<T, 4, Eigen::DenseIndex>::Tensor output);              \
  extern template struct ShuffleAndReverse<GPUDevice, T, 4,                  \
                                           Eigen::DenseIndex>;               \
  template <>                                                                \
  void InflatePadAndShuffle<GPUDevice, T, 4, Eigen::DenseIndex>::operator()( \
      const GPUDevice& d,                                                    \
      typename TTypes<T, 4, Eigen::DenseIndex>::ConstTensor input,           \
      const Eigen::DSizes<Eigen::DenseIndex, 4>& strides,                    \
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 4>& pad_dims,  \
      const Eigen::DSizes<Eigen::DenseIndex, 4>& order,                      \
      typename TTypes<T, 4, Eigen::DenseIndex>::Tensor output);              \
  extern template struct InflatePadAndShuffle<GPUDevice, T, 4,               \
                                              Eigen::DenseIndex>;            \
  template <>                                                                \
  void ShuffleAndReverse<GPUDevice, T, 4, int>::operator()(                  \
      const GPUDevice& d, typename TTypes<T, 4, int>::ConstTensor input,     \
      const Eigen::DSizes<int, 4>& order,                                    \
      const Eigen::array<bool, 4>& reverse_dims,                             \
      typename TTypes<T, 4, int>::Tensor output);                            \
  extern template struct ShuffleAndReverse<GPUDevice, T, 4, int>;            \
  template <>                                                                \
  void InflatePadAndShuffle<GPUDevice, T, 4, int>::operator()(               \
      const GPUDevice& d, typename TTypes<T, 4, int>::ConstTensor input,     \
      const Eigen::DSizes<int, 4>& strides,                                  \
      const Eigen::array<Eigen::IndexPair<int>, 4>& pad_dims,                \
      const Eigen::DSizes<int, 4>& order,                                    \
      typename TTypes<T, 4, int>::Tensor output);                            \
  extern template struct InflatePadAndShuffle<GPUDevice, T, 4, int>;         \
  template <>                                                                \
  void TransformFilter<GPUDevice, T, int>::operator()(                       \
      const GPUDevice& d, typename TTypes<T, 4, int>::ConstTensor in,        \
      typename TTypes<T, 4, int>::Tensor out);                               \
  extern template struct TransformFilter<GPUDevice, T, int>;                 \
  template <>                                                                \
  void TransformDepth<GPUDevice, T, int>::operator()(                        \
      const GPUDevice& d, typename TTypes<T, 4, int>::ConstTensor in,        \
      const Eigen::DSizes<int, 4>& shuffle,                                  \
      typename TTypes<T, 4, int>::Tensor out);                               \
  extern template struct TransformDepth<GPUDevice, T, int>;                  \
  template <>                                                                \
  void SpatialConvolution<GPUDevice, T>::operator()(                         \
      const GPUDevice& d, typename TTypes<T, 4>::Tensor output,              \
      typename TTypes<T, 4>::ConstTensor input,                              \
      typename TTypes<T, 4>::ConstTensor filter, int stride,                 \
      const Eigen::PaddingType& padding);                                    \
  extern template struct SpatialConvolution<GPUDevice, T>;                   \
  template <>                                                                \
  void SpatialConvolutionBackwardInput<GPUDevice, T>::operator()(            \
      const GPUDevice& d, typename TTypes<T, 4>::Tensor in_backprop,         \
      typename TTypes<T, 4>::ConstTensor filter,                             \
      typename TTypes<T, 4>::ConstTensor output_backprop, int input_rows,    \
      int input_cols, int stride);                                           \
  extern template struct SpatialConvolutionBackwardInput<GPUDevice, T>

DECLARE_GPU_SPEC(float);
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropInput")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T")
                            .HostMemory("input_sizes"),
                        Conv2DSlowBackpropInputOp<GPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropFilter")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T")
                            .HostMemory("filter_sizes"),
                        Conv2DSlowBackpropFilterOp<GPUDevice, float>);
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
