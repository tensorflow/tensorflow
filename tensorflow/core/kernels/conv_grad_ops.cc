/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/nn_ops.cc.

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/conv_grad_ops.h"

#include <algorithm>
#include <vector>

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"
#include "tensorflow/core/util/work_sharder.h"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

namespace {

// Returns in 'col_data', image patches in storage order (height, width, depth)
// extracted from image at 'input_data', which is required to be in storage
// order (batch, height, width, depth).
// Implementation written by Yangqing Jia (jiayq).
template <typename T>
void Im2col(const T* input_data, const int depth, const int height,
            const int width, const int filter_h, const int filter_w,
            const int pad_t, const int pad_l, const int pad_b, const int pad_r,
            const int stride_h, const int stride_w, T* col_data) {
  int height_col = (height + pad_t + pad_b - filter_h) / stride_h + 1;
  int width_col = (width + pad_l + pad_r - filter_w) / stride_w + 1;

  int h_pad = -pad_t;
  for (int h = 0; h < height_col; ++h) {
    int w_pad = -pad_l;
    for (int w = 0; w < width_col; ++w) {
      for (int ih = h_pad; ih < h_pad + filter_h; ++ih) {
        for (int iw = w_pad; iw < w_pad + filter_w; ++iw) {
          if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
            memcpy(col_data, input_data + (ih * width + iw) * depth,
                   sizeof(T) * depth);
          } else {
            // This should be simply padded with zero.
            memset(col_data, 0, sizeof(T) * depth);
          }
          col_data += depth;
        }
      }
      w_pad += stride_w;
    }
    h_pad += stride_h;
  }
}

// Returns in 'im_data' (assumes to be zero-initialized) image patch in storage
// order (height, width, depth), constructed from patches in 'col_data', which
// is required to be in storage order (out_height * out_width, filter_height,
// filter_width, in_depth).  Implementation by Yangqing Jia (jiayq).
template <typename T>
void Col2im(const T* col_data, const int depth, const int height,
            const int width, const int filter_h, const int filter_w,
            const int pad_t, const int pad_l, const int pad_b, const int pad_r,
            const int stride_h, const int stride_w, T* im_data) {
  int height_col = (height + pad_t + pad_b - filter_h) / stride_h + 1;
  int width_col = (width + pad_l + pad_r - filter_w) / stride_w + 1;
  int h_pad = -pad_t;
  for (int h = 0; h < height_col; ++h) {
    int w_pad = -pad_l;
    for (int w = 0; w < width_col; ++w) {
      T* im_patch_data = im_data + (h_pad * width + w_pad) * depth;
      for (int ih = h_pad; ih < h_pad + filter_h; ++ih) {
        for (int iw = w_pad; iw < w_pad + filter_w; ++iw) {
          if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
            // TODO(andydavis) Vectorize this loop (if compiler does not).
            for (int i = 0; i < depth; ++i) {
              im_patch_data[i] += col_data[i];
            }
          }
          im_patch_data += depth;
          col_data += depth;
        }
        // Jump over remaining number of depth.
        im_patch_data += depth * (width - filter_w);
      }
      w_pad += stride_w;
    }
    h_pad += stride_h;
  }
}

}  // namespace

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

static Status ConvBackpropExtractAndVerifyDimension(
    StringPiece label, const TensorShape& input_shape,
    const TensorShape& filter_shape, const TensorShape& output_shape,
    const std::vector<int32>& strides, Padding padding, int spatial_dim,
    int filter_spatial_dim, ConvBackpropSpatialDimension* dim) {
  dim->input_size = input_shape.dim_size(spatial_dim);
  dim->filter_size = filter_shape.dim_size(filter_spatial_dim);
  dim->output_size = output_shape.dim_size(spatial_dim);
  dim->stride = strides[spatial_dim];
  int64 out_size = 0, pad_size = 0;
  TF_RETURN_IF_ERROR(GetWindowedOutputSize(dim->input_size, dim->filter_size,
                                           dim->stride, padding, &out_size,
                                           &pad_size));
  if (dim->output_size != out_size) {
    return errors::InvalidArgument(
        label, ": Size of out_backprop doesn't match computed: ", "actual = ",
        dim->output_size, ", computed = ", out_size);
  }

  dim->expanded_output_size = (dim->output_size - 1) * dim->stride + 1;
  const auto padded_out_size = dim->input_size + dim->filter_size - 1;
  dim->pad_before = dim->filter_size - 1 - pad_size;
  dim->pad_after =
      padded_out_size - dim->expanded_output_size - dim->pad_before;
  VLOG(2) << label << ": expanded_out = " << dim->expanded_output_size
          << ", filter = " << dim->filter_size
          << ", padded_out = " << padded_out_size
          << ", pad_before = " << dim->pad_before
          << ", pad_after = " << dim->pad_after
          << ", strides = " << dim->stride;
  return Status::OK();
}

Status Conv2DBackpropComputeDimensions(
    StringPiece label, const TensorShape& input_shape,
    const TensorShape& filter_shape, const TensorShape& out_backprop_shape,
    const std::vector<int32>& strides, Padding padding,
    TensorFormat data_format, Conv2DBackpropDimensions* dims) {
  if (input_shape.dims() != 4) {
    return errors::InvalidArgument(label, ": input must be 4-dimensional");
  }
  if (filter_shape.dims() != 4) {
    return errors::InvalidArgument(label, ": filter must be 4-dimensional");
  }
  if (out_backprop_shape.dims() != 4) {
    errors::InvalidArgument(label, ": out_backprop must be 4-dimensional");
  }
  dims->batch_size = GetTensorDim(input_shape, data_format, 'N');
  if (dims->batch_size != GetTensorDim(out_backprop_shape, data_format, 'N')) {
    return errors::InvalidArgument(
        label, ": input and out_backprop must have the same batch size");
  }

  dims->in_depth = GetTensorDim(input_shape, data_format, 'C');
  if (dims->in_depth != filter_shape.dim_size(2)) {
    return errors::InvalidArgument(
        label, ": input and filter must have the same depth");
  }
  dims->out_depth = filter_shape.dim_size(3);
  if (dims->out_depth != GetTensorDim(out_backprop_shape, data_format, 'C')) {
    return errors::InvalidArgument(
        label, ": filter and out_backprop must have the same out_depth");
  }

  const int row_dim = GetTensorDimIndex(data_format, 'H');
  const int col_dim = GetTensorDimIndex(data_format, 'W');
  const int filter_row_dim = 0, filter_col_dim = 1;
  TF_RETURN_IF_ERROR(ConvBackpropExtractAndVerifyDimension(
      label, input_shape, filter_shape, out_backprop_shape, strides, padding,
      row_dim, filter_row_dim, &dims->rows));
  TF_RETURN_IF_ERROR(ConvBackpropExtractAndVerifyDimension(
      label, input_shape, filter_shape, out_backprop_shape, strides, padding,
      col_dim, filter_col_dim, &dims->cols));
  return Status::OK();
}

// The fast versions using eigen computations directly. They are only enabled
// for CPU for now since nvcc times out when trying to compile them.
// TODO(yangke): enable them for GPUs when we have a faster compiler.

template <typename Device, class T>
class Conv2DFastBackpropInputOp : public OpKernel {
 public:
  explicit Conv2DFastBackpropInputOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(context, data_format_ == FORMAT_NHWC,
                errors::InvalidArgument(
                    "Eigen Conv2DFastBackpropInputOp only supports NHWC."));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(
        context, (strides_[0] == 1 && strides_[3] == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_sizes = context->input(0);
    const Tensor& filter = context->input(1);
    const Tensor& out_backprop = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(input_sizes.shape()),
        errors::InvalidArgument(
            "Conv2DBackpropInput: input_sizes input must be 1-dim, not ",
            input_sizes.dims()));
    TensorShape input_shape;
    OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                input_sizes.vec<int32>(), &input_shape));

    Conv2DBackpropDimensions dims;
    OP_REQUIRES_OK(context, Conv2DBackpropComputeDimensions(
                                "Conv2DFastBackpropInput", input_shape,
                                filter.shape(), out_backprop.shape(), strides_,
                                padding_, data_format_, &dims));

    Tensor* in_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_shape, &in_backprop));
    functor::SpatialConvolutionBackwardInput<Device, T>()(
        context->eigen_device<Device>(), in_backprop->tensor<T, 4>(),
        filter.tensor<T, 4>(), out_backprop.tensor<T, 4>(),
        dims.rows.input_size, dims.cols.input_size, dims.rows.stride,
        dims.cols.stride);
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv2DFastBackpropInputOp);
};

// Based on implementation written by Yangqing Jia (jiayq).
template <typename Device, class T>
class Conv2DCustomBackpropInputOp : public OpKernel {
 public:
  explicit Conv2DCustomBackpropInputOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(context, data_format_ == FORMAT_NHWC,
                errors::InvalidArgument(
                    "Conv2DCustomBackpropInputOp only supports NHWC."));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(
        context, (strides_[0] == 1 && strides_[3] == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_sizes = context->input(0);
    const Tensor& filter = context->input(1);
    const Tensor& out_backprop = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(input_sizes.shape()),
        errors::InvalidArgument(
            "Conv2DBackpropInput: input_sizes input must be 1-dim, not ",
            input_sizes.dims()));
    TensorShape input_shape;
    OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                input_sizes.vec<int32>(), &input_shape));

    Conv2DBackpropDimensions dims;
    OP_REQUIRES_OK(context, Conv2DBackpropComputeDimensions(
                                "Conv2DCustomBackpropInput", input_shape,
                                filter.shape(), out_backprop.shape(), strides_,
                                padding_, data_format_, &dims));

    Tensor* in_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_shape, &in_backprop));

    // TODO(andydavis) Consider moving code shared with
    // Conv2DCustomBackpropFilterOp into a shared helper function.
    int64 pad_top, pad_bottom;
    int64 pad_left, pad_right;
    OP_REQUIRES_OK(context, GetWindowedOutputSizeVerbose(
                                dims.rows.input_size, dims.rows.filter_size,
                                dims.rows.stride, padding_,
                                &dims.rows.output_size, &pad_top, &pad_bottom));
    OP_REQUIRES_OK(context, GetWindowedOutputSizeVerbose(
                                dims.cols.input_size, dims.cols.filter_size,
                                dims.cols.stride, padding_,
                                &dims.cols.output_size, &pad_left, &pad_right));

    // The total dimension size of each kernel.
    const int filter_total_size =
        dims.rows.filter_size * dims.cols.filter_size * dims.in_depth;
    // The output image size is the spatial size of the output.
    const int output_image_size = dims.rows.output_size * dims.cols.output_size;

    // TODO(andydavis) Get L2/L3 cache sizes from device.
    const size_t l2_cache_size = 256LL << 10;
    const size_t l3_cache_size = 30LL << 20;

    // Use L3 cache size as target working set size.
    const size_t target_working_set_size = l3_cache_size / sizeof(T);

    // Calculate size of matrices involved in MatMul: C = A x B.
    const size_t size_A = output_image_size * dims.out_depth;

    const size_t size_B = filter_total_size * dims.out_depth;

    const size_t size_C = output_image_size * filter_total_size;

    const size_t work_unit_size = size_A + size_B + size_C;

    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());

    // Calculate per-thread work unit size.
    const size_t thread_work_unit_size =
        work_unit_size / worker_threads.num_threads;

    // Set minimum per-thread work unit size to size of L2 cache.
    const size_t min_thread_work_unit_size = l2_cache_size / sizeof(T);

    // Use parallel tensor contractions if there is no batching, or if the
    // minimum per-thread work unit size threshold has been exceeded.
    // Otherwise, revert to multiple single-threaded matmul ops running in
    // parallel to keep all threads busy.
    // TODO(andydavis) Explore alternatives to branching the code in this way
    // (i.e. run multiple, parallel tensor contractions in another thread pool).
    const bool use_parallel_contraction =
        dims.batch_size == 1 ||
        thread_work_unit_size >= min_thread_work_unit_size;

    const size_t shard_size =
        use_parallel_contraction
            ? 1
            : (target_working_set_size + work_unit_size - 1) / work_unit_size;

    Tensor col_buffer;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(
                       DataTypeToEnum<T>::value,
                       TensorShape({static_cast<int64>(shard_size),
                                    static_cast<int64>(output_image_size),
                                    static_cast<int64>(filter_total_size)}),
                       &col_buffer));

    // The input offset corresponding to a single input image.
    const int input_offset =
        dims.rows.input_size * dims.cols.input_size * dims.in_depth;
    // The output offset corresponding to a single output image.
    const int output_offset =
        dims.rows.output_size * dims.cols.output_size * dims.out_depth;

    const T* filter_data = filter.template flat<T>().data();
    T* col_buffer_data = col_buffer.template flat<T>().data();
    const T* out_backprop_data = out_backprop.template flat<T>().data();

    auto in_backprop_flat = in_backprop->template flat<T>();
    T* input_backprop_data = in_backprop_flat.data();
    in_backprop_flat.device(context->eigen_device<Device>()) =
        in_backprop_flat.constant(T(0));

    if (use_parallel_contraction) {
      typedef Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>,
                               Eigen::Unaligned>
          TensorMap;
      typedef Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor>,
                               Eigen::Unaligned>
          ConstTensorMap;

      // Initialize contraction dims (we need to transpose 'B' below).
      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> contract_dims;
      contract_dims[0].first = 1;
      contract_dims[0].second = 1;

      for (int image_id = 0; image_id < dims.batch_size; ++image_id) {
        // Compute gradient into col_buffer.
        TensorMap C(col_buffer_data, output_image_size, filter_total_size);

        ConstTensorMap A(out_backprop_data + output_offset * image_id,
                         output_image_size, dims.out_depth);
        ConstTensorMap B(filter_data, filter_total_size, dims.out_depth);

        C.device(context->eigen_cpu_device()) = A.contract(B, contract_dims);

        Col2im<T>(col_buffer_data, dims.in_depth, dims.rows.input_size,
                  dims.cols.input_size, dims.rows.filter_size,
                  dims.cols.filter_size, pad_top, pad_left, pad_bottom,
                  pad_right, dims.rows.stride, dims.cols.stride,
                  input_backprop_data);

        input_backprop_data += input_offset;
      }
    } else {
      typedef Eigen::Map<
          Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
          MatrixMap;
      typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
                                             Eigen::RowMajor>>
          ConstMatrixMap;

      for (int image_id = 0; image_id < dims.batch_size;
           image_id += shard_size) {
        const int shard_limit =
            std::min(static_cast<int>(shard_size),
                     static_cast<int>(dims.batch_size) - image_id);

        auto shard = [&dims, &pad_top, &pad_left, &pad_bottom, &pad_right,
                      &output_image_size, &filter_total_size,
                      &input_backprop_data, &col_buffer_data,
                      &out_backprop_data, &filter_data, &input_offset,
                      &output_offset, &size_C](int64 start, int64 limit) {
          for (int shard_id = start; shard_id < limit; ++shard_id) {
            T* im2col_buf = col_buffer_data + shard_id * size_C;
            T* input_data = input_backprop_data + shard_id * input_offset;
            const T* out_data = out_backprop_data + shard_id * output_offset;

            // Compute gradient into 'im2col_buf'.
            MatrixMap C(im2col_buf, output_image_size, filter_total_size);

            ConstMatrixMap A(out_data, output_image_size, dims.out_depth);
            ConstMatrixMap B(filter_data, filter_total_size, dims.out_depth);

            C.noalias() = A * B.transpose();

            Col2im<T>(im2col_buf, dims.in_depth, dims.rows.input_size,
                      dims.cols.input_size, dims.rows.filter_size,
                      dims.cols.filter_size, pad_top, pad_left, pad_bottom,
                      pad_right, dims.rows.stride, dims.cols.stride,
                      input_data);
          }
        };
        Shard(worker_threads.num_threads, worker_threads.workers, shard_limit,
              work_unit_size, shard);

        input_backprop_data += input_offset * shard_limit;
        out_backprop_data += output_offset * shard_limit;
      }
    }
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv2DCustomBackpropInputOp);
};

#define REGISTER_CPU_KERNELS(T)                                              \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("Conv2DBackpropInput").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      Conv2DCustomBackpropInputOp<CPUDevice, T>);                            \
  REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropInput")                        \
                              .Device(DEVICE_CPU)                            \
                              .Label("custom")                               \
                              .TypeConstraint<T>("T"),                       \
                          Conv2DCustomBackpropInputOp<CPUDevice, T>);        \
  REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropInput")                        \
                              .Device(DEVICE_CPU)                            \
                              .Label("eigen_tensor")                         \
                              .TypeConstraint<T>("T"),                       \
                          Conv2DFastBackpropInputOp<CPUDevice, T>);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS

template <typename Device, class T>
class Conv2DFastBackpropFilterOp : public OpKernel {
 public:
  explicit Conv2DFastBackpropFilterOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(context, data_format_ == FORMAT_NHWC,
                errors::InvalidArgument(
                    "Conv2DFastBackpropFilterOp only supports NHWC."));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(
        context, (strides_[0] == 1 && strides_[3] == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& filter_sizes = context->input(1);
    const Tensor& out_backprop = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(filter_sizes.shape()),
        errors::InvalidArgument(
            "Conv2DBackpropFilter: filter_sizes input must be 1-dim, not ",
            filter_sizes.dims()));
    TensorShape filter_shape;
    OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                filter_sizes.vec<int32>(), &filter_shape));

    Conv2DBackpropDimensions dims;
    OP_REQUIRES_OK(context, Conv2DBackpropComputeDimensions(
                                "Conv2DFastBackpropFilter", input.shape(),
                                filter_shape, out_backprop.shape(), strides_,
                                padding_, data_format_, &dims));

    Tensor* filter_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, filter_shape, &filter_backprop));

    functor::SpatialConvolutionBackwardKernel<Device, T>()(
        context->eigen_device<Device>(), filter_backprop->tensor<T, 4>(),
        input.tensor<T, 4>(), out_backprop.tensor<T, 4>(),
        dims.rows.filter_size, dims.cols.filter_size, dims.rows.stride,
        dims.cols.stride);
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv2DFastBackpropFilterOp);
};

// Based on implementation written by Yangqing Jia (jiayq).
template <typename Device, class T>
class Conv2DCustomBackpropFilterOp : public OpKernel {
 public:
  explicit Conv2DCustomBackpropFilterOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(context, data_format_ == FORMAT_NHWC,
                errors::InvalidArgument(
                    "Conv2DCustomBackpropFilterOp only supports NHWC."));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(
        context, (strides_[0] == 1 && strides_[3] == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& filter_sizes = context->input(1);
    const Tensor& out_backprop = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(filter_sizes.shape()),
        errors::InvalidArgument(
            "Conv2DCustomBackpropFilter: filter_sizes input must be 1-dim, "
            "not ",
            filter_sizes.dims()));
    TensorShape filter_shape;
    OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                filter_sizes.vec<int32>(), &filter_shape));

    Conv2DBackpropDimensions dims;
    OP_REQUIRES_OK(context, Conv2DBackpropComputeDimensions(
                                "Conv2DCustomBackpropFilter", input.shape(),
                                filter_shape, out_backprop.shape(), strides_,
                                padding_, data_format_, &dims));

    Tensor* filter_backprop;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, filter_shape, &filter_backprop));

    int64 pad_top, pad_bottom;
    int64 pad_left, pad_right;
    OP_REQUIRES_OK(context, GetWindowedOutputSizeVerbose(
                                dims.rows.input_size, dims.rows.filter_size,
                                dims.rows.stride, padding_,
                                &dims.rows.output_size, &pad_top, &pad_bottom));
    OP_REQUIRES_OK(context, GetWindowedOutputSizeVerbose(
                                dims.cols.input_size, dims.cols.filter_size,
                                dims.cols.stride, padding_,
                                &dims.cols.output_size, &pad_left, &pad_right));

    // The total dimension size of each kernel.
    const int filter_total_size =
        dims.rows.filter_size * dims.cols.filter_size * dims.in_depth;
    // The output image size is the spatial size of the output.
    const int output_image_size = dims.rows.output_size * dims.cols.output_size;

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

    const size_t size_B = output_image_size * dims.out_depth;

    const size_t size_C = filter_total_size * dims.out_depth;

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
    const int input_offset =
        dims.rows.input_size * dims.cols.input_size * dims.in_depth;
    // The output offset corresponding to a single output image.
    const int output_offset =
        dims.rows.output_size * dims.cols.output_size * dims.out_depth;

    const T* input_data = input.template flat<T>().data();
    T* col_buffer_data = col_buffer.template flat<T>().data();
    const T* out_backprop_data = out_backprop.template flat<T>().data();
    T* filter_backprop_data = filter_backprop->template flat<T>().data();

    typedef Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>,
                             Eigen::Unaligned>
        TensorMap;
    typedef Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor>,
                             Eigen::Unaligned>
        ConstTensorMap;

    TensorMap C(filter_backprop_data, filter_total_size, dims.out_depth);
    C.setZero();

    // Initialize contraction dims (we need to transpose 'A' below).
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> contract_dims;
    contract_dims[0].first = 0;
    contract_dims[0].second = 0;

    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());

    for (int image_id = 0; image_id < dims.batch_size; image_id += shard_size) {
      const int shard_limit =
          std::min(static_cast<int>(shard_size),
                   static_cast<int>(dims.batch_size) - image_id);

      auto shard = [&input_data, &col_buffer_data, &dims, &pad_top, &pad_left,
                    &pad_bottom, &pad_right, &input_offset,
                    &size_A](int64 start, int64 limit) {
        for (int shard_id = start; shard_id < limit; ++shard_id) {
          const T* input_data_shard = input_data + shard_id * input_offset;
          T* col_data_shard = col_buffer_data + shard_id * size_A;

          // When we compute the gradient with respect to the filters, we need
          // to do im2col to allow gemm-type computation.
          Im2col<T>(input_data_shard, dims.in_depth, dims.rows.input_size,
                    dims.cols.input_size, dims.rows.filter_size,
                    dims.cols.filter_size, pad_top, pad_left, pad_bottom,
                    pad_right, dims.rows.stride, dims.cols.stride,
                    col_data_shard);
        }
      };
      Shard(worker_threads.num_threads, worker_threads.workers, shard_limit,
            size_A, shard);

      ConstTensorMap A(col_buffer_data, output_image_size * shard_limit,
                       filter_total_size);
      ConstTensorMap B(out_backprop_data, output_image_size * shard_limit,
                       dims.out_depth);

      // Gradient with respect to filter.
      C.device(context->eigen_cpu_device()) += A.contract(B, contract_dims);

      input_data += input_offset * shard_limit;
      out_backprop_data += output_offset * shard_limit;
    }
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv2DCustomBackpropFilterOp);
};

#define REGISTER_CPU_KERNELS(T)                                               \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("Conv2DBackpropFilter").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      Conv2DCustomBackpropFilterOp<CPUDevice, T>);                            \
  REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropFilter")                        \
                              .Device(DEVICE_CPU)                             \
                              .Label("custom")                                \
                              .TypeConstraint<T>("T"),                        \
                          Conv2DCustomBackpropFilterOp<CPUDevice, T>);        \
  REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropFilter")                        \
                              .Device(DEVICE_CPU)                             \
                              .Label("eigen_tensor")                          \
                              .TypeConstraint<T>("T"),                        \
                          Conv2DFastBackpropFilterOp<CPUDevice, T>);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS

// GPU definitions of both ops.
#if GOOGLE_CUDA
// The slow version (but compiles for GPU)

// Backprop for input.
template <typename Device, class T>
class Conv2DSlowBackpropInputOp : public OpKernel {
 public:
  explicit Conv2DSlowBackpropInputOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    int stride_n = GetTensorDim(strides_, data_format_, 'N');
    int stride_c = GetTensorDim(strides_, data_format_, 'C');
    OP_REQUIRES(
        context, (stride_n == 1 && stride_c == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("use_cudnn_on_gpu", &use_cudnn_));
    use_cudnn_ &= CanUseCudnn();
    cudnn_use_autotune_ = CudnnUseAutotune();
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    using perftools::gputools::dnn::AlgorithmConfig;
    using perftools::gputools::dnn::AlgorithmType;
    using perftools::gputools::dnn::ProfileResult;
    using perftools::gputools::dnn::kDefaultAlgorithm;
    const Tensor& input_sizes = context->input(0);
    const Tensor& filter = context->input(1);
    const Tensor& out_backprop = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(input_sizes.shape()),
        errors::InvalidArgument(
            "Conv2DBackpropInput: input_sizes input must be 1-dim, not ",
            input_sizes.dims()));
    TensorShape input_shape;
    OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                input_sizes.vec<int32>(), &input_shape));
    const TensorShape& filter_shape = filter.shape();

    Conv2DBackpropDimensions dims;
    OP_REQUIRES_OK(context, Conv2DBackpropComputeDimensions(
                                "Conv2DSlowBackpropInput", input_shape,
                                filter_shape, out_backprop.shape(), strides_,
                                padding_, data_format_, &dims));

    Tensor* in_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_shape, &in_backprop));

    const int padding_rows =
        (padding_ == VALID)
            ? 0
            : std::max<int>(0, (dims.rows.output_size - 1) * dims.rows.stride +
                                   dims.rows.filter_size -
                                   dims.rows.input_size);
    const int padding_cols =
        (padding_ == VALID)
            ? 0
            : std::max<int>(0, (dims.cols.output_size - 1) * dims.cols.stride +
                                   dims.cols.filter_size -
                                   dims.cols.input_size);

    // TODO(keveman): cuDNN only supports equal padding on both sides, so only
    // calling it when that is true. Remove this check when (if?) cuDNN starts
    // supporting different padding.
    bool rows_odd = (padding_rows % 2 != 0);
    bool cols_odd = (padding_cols % 2 != 0);

    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

    if (!use_cudnn_) {
      context->SetStatus(errors::Unimplemented(
          "Conv2DBackpropInput for GPU is not currently supported "
          "without cudnn"));
      return;
    }

    if (dims.rows.filter_size == 1 && dims.cols.filter_size == 1 &&
        dims.rows.stride == 1 && dims.cols.stride == 1 &&
        data_format_ == FORMAT_NHWC) {
      // 1x1 filter, so call cublas directly.
      const uint64 m =
          dims.batch_size * dims.rows.input_size * dims.cols.input_size;
      const uint64 k = dims.out_depth;
      const uint64 n = dims.in_depth;

      auto a_ptr = AsDeviceMemory(out_backprop.template flat<T>().data(),
                                  out_backprop.template flat<T>().size());
      auto b_ptr = AsDeviceMemory(filter.template flat<T>().data(),
                                  filter.template flat<T>().size());
      auto c_ptr = AsDeviceMemory(in_backprop->template flat<T>().data(),
                                  in_backprop->template flat<T>().size());

      auto transpose = perftools::gputools::blas::Transpose::kTranspose;
      auto no_transpose = perftools::gputools::blas::Transpose::kNoTranspose;

      bool blas_launch_status =
          stream
              ->ThenBlasGemm(transpose, no_transpose, n, m, k, 1.0f, b_ptr, k,
                             a_ptr, k, 0.0f, &c_ptr, n)
              .ok();
      if (!blas_launch_status) {
        context->SetStatus(errors::Internal("Blas SGEMM launch failed : m=", m,
                                            ", n=", n, ", k=", k));
      }

      return;
    }

    TensorShape compatible_input_shape;
    if (rows_odd || cols_odd) {
      // If a padding dimension is odd, we have one more element on the right
      // side or the bottom side. This is unsupported in cudnn. Therefore,
      // we pad that extra element and make it compatible.
      compatible_input_shape = ShapeFromFormat(
          data_format_, dims.batch_size, dims.rows.input_size + rows_odd,
          dims.cols.input_size + cols_odd, dims.in_depth);
    } else {
      compatible_input_shape = input_shape;
    }

    CHECK(padding_rows >= 0 && padding_cols >= 0)
        << "Negative row or col paddings: (" << padding_rows << ", "
        << padding_cols << ")";
    perftools::gputools::dnn::BatchDescriptor input_desc;
    input_desc.set_count(dims.batch_size)
        .set_height(GetTensorDim(compatible_input_shape, data_format_, 'H'))
        .set_width(GetTensorDim(compatible_input_shape, data_format_, 'W'))
        .set_feature_map_count(dims.in_depth)
        .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);
    perftools::gputools::dnn::BatchDescriptor output_desc;
    output_desc.set_count(dims.batch_size)
        .set_height(dims.rows.output_size)
        .set_width(dims.cols.output_size)
        .set_feature_map_count(dims.out_depth)
        .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);
    perftools::gputools::dnn::FilterDescriptor filter_desc;
    filter_desc.set_input_filter_height(dims.rows.filter_size)
        .set_input_filter_width(dims.cols.filter_size)
        .set_input_feature_map_count(dims.in_depth)
        .set_output_feature_map_count(dims.out_depth);
    perftools::gputools::dnn::ConvolutionDescriptor conv_desc;
    conv_desc.set_vertical_filter_stride(dims.rows.stride)
        .set_horizontal_filter_stride(dims.cols.stride)
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
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataTypeToEnum<T>::value,
                                TensorShape({dims.out_depth, dims.in_depth,
                                             dims.rows.filter_size,
                                             dims.cols.filter_size}),
                                &transformed_filter));

    functor::TransformFilter<Device, T, int, 4>()(
        context->eigen_device<Device>(), To32Bit(filter.tensor<T, 4>()),
        To32Bit(transformed_filter.tensor<T, 4>()));

    Tensor transformed_out_backprop;
    if (data_format_ == FORMAT_NHWC) {
      TensorShape nchw_shape =
          ShapeFromFormat(FORMAT_NCHW, dims.batch_size, dims.rows.output_size,
                          dims.cols.output_size, dims.out_depth);
      if (dims.out_depth > 1) {
        OP_REQUIRES_OK(context, context->allocate_temp(
                                    DataTypeToEnum<T>::value, nchw_shape,
                                    &transformed_out_backprop));
        functor::NHWCToNCHW<Device, T, 4>()(
            context->eigen_device<Device>(), out_backprop.tensor<T, 4>(),
            transformed_out_backprop.tensor<T, 4>());
      } else {
        // If depth <= 1, then just reshape.
        CHECK(transformed_out_backprop.CopyFrom(out_backprop, nchw_shape));
      }
    } else {
      transformed_out_backprop = out_backprop;
    }

    Tensor pre_transformed_in_backprop;
    OP_REQUIRES_OK(
        context,
        context->allocate_temp(
            DataTypeToEnum<T>::value,
            ShapeFromFormat(
                FORMAT_NCHW,
                GetTensorDim(compatible_input_shape, data_format_, 'N'),
                GetTensorDim(compatible_input_shape, data_format_, 'H'),
                GetTensorDim(compatible_input_shape, data_format_, 'W'),
                GetTensorDim(compatible_input_shape, data_format_, 'C')),
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

    static int64 ConvolveBackwardDataScratchSize = GetCudnnWorkspaceLimit(
        "TF_CUDNN_WORKSPACE_LIMIT_IN_MB", 1LL << 32  // 4GB by default
        );
    CudnnScratchAllocator scratch_allocator(ConvolveBackwardDataScratchSize,
                                            context);
    int device_id = stream->parent()->device_ordinal();
    ConvParameters conv_parameters = {
        dims.batch_size,        // batch
        dims.in_depth,          // in_depths
        input_desc.height(),    // in_rows
        input_desc.width(),     // in_cols
        dims.out_depth,         // out_depths
        dims.rows.filter_size,  // filter_rows
        dims.cols.filter_size,  // filter_cols
        dims.rows.stride,       // stride_rows
        dims.cols.stride,       // stride_cols
        padding_rows,           // padding_rows
        padding_cols,           // padding_cols
        device_id,              // device_id
    };
    AlgorithmConfig algorithm_config;
    if (cudnn_use_autotune_ &&
        !autotune_results_.Find(conv_parameters, &algorithm_config)) {
      std::vector<AlgorithmType> algorithms;
      CHECK(stream->parent()->GetConvolveBackwardDataAlgorithms(&algorithms));
      ProfileResult best_result;
      ProfileResult best_result_no_scratch;
      for (auto profile_algorithm : algorithms) {
        // TODO(zhengxq): profile each algorithm multiple times to better
        // accuracy.
        CudnnScratchAllocator scratch_allocator(ConvolveBackwardDataScratchSize,
                                                context);
        ProfileResult profile_result;
        bool cudnn_launch_status =
            stream
                ->ThenConvolveBackwardDataWithAlgorithm(
                    filter_desc, filter_ptr, output_desc, out_backprop_ptr,
                    conv_desc, input_desc, &in_backprop_ptr, &scratch_allocator,
                    AlgorithmConfig(profile_algorithm), &profile_result)
                .ok();
        if (cudnn_launch_status) {
          if (profile_result.is_valid()) {
            if (profile_result.elapsed_time_in_ms() <
                best_result.elapsed_time_in_ms()) {
              best_result = profile_result;
            }
            if (scratch_allocator.TotalByteSize() == 0 &&
                profile_result.elapsed_time_in_ms() <
                    best_result_no_scratch.elapsed_time_in_ms()) {
              best_result_no_scratch = profile_result;
            }
          }
        }
      }
      OP_REQUIRES(context, best_result.is_valid() &&
                               best_result.algorithm() != kDefaultAlgorithm,
                  errors::NotFound("No algorithm worked!"));
      OP_REQUIRES(context,
                  best_result_no_scratch.is_valid() &&
                      best_result_no_scratch.algorithm() != kDefaultAlgorithm,
                  errors::NotFound("No algorithm without scratch worked!"));
      algorithm_config.set_algorithm(best_result.algorithm());
      algorithm_config.set_algorithm_no_scratch(
          best_result_no_scratch.algorithm());
      autotune_results_.Insert(conv_parameters, algorithm_config);
    }
    bool cudnn_launch_status =
        stream
            ->ThenConvolveBackwardDataWithAlgorithm(
                filter_desc, filter_ptr, output_desc, out_backprop_ptr,
                conv_desc, input_desc, &in_backprop_ptr, &scratch_allocator,
                algorithm_config, nullptr)
            .ok();

    if (!cudnn_launch_status) {
      context->SetStatus(errors::Internal(
          "cuDNN Backward Data function launch failure : input shape(",
          input_shape.DebugString(), ") filter shape(",
          filter_shape.DebugString(), ")"));
      return;
    }

    if (rows_odd || cols_odd) {
      Tensor in_backprop_remove_padding;
      OP_REQUIRES_OK(
          context,
          context->allocate_temp(
              DataTypeToEnum<T>::value,
              ShapeFromFormat(FORMAT_NCHW,
                              GetTensorDim(input_shape, data_format_, 'N'),
                              GetTensorDim(input_shape, data_format_, 'H'),
                              GetTensorDim(input_shape, data_format_, 'W'),
                              GetTensorDim(input_shape, data_format_, 'C')),
              &in_backprop_remove_padding));

      // Remove the padding for odd rows or cols.
      functor::PadInput<GPUDevice, T, int, 4>()(
          context->template eigen_device<GPUDevice>(),
          To32Bit(const_cast<const Tensor&>(pre_transformed_in_backprop)
                      .tensor<T, 4>()),
          {{0, 0}}, {{-rows_odd, -cols_odd}},
          To32Bit(in_backprop_remove_padding.tensor<T, 4>()), FORMAT_NCHW);

      pre_transformed_in_backprop = in_backprop_remove_padding;
    }

    if (data_format_ == FORMAT_NHWC) {
      auto toConstTensor = [](const Tensor& x) -> const Tensor { return x; };
      functor::NCHWToNHWC<Device, T, 4>()(
          context->eigen_device<Device>(),
          toConstTensor(pre_transformed_in_backprop).template tensor<T, 4>(),
          in_backprop->tensor<T, 4>());
    } else {
      *in_backprop = pre_transformed_in_backprop;
    }
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;
  bool use_cudnn_;
  TensorFormat data_format_;
  AutoTuneMap<ConvParameters, perftools::gputools::dnn::AlgorithmConfig>
      autotune_results_;
  bool cudnn_use_autotune_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv2DSlowBackpropInputOp);
};

// Backprop for filter.
template <typename Device, class T>
class Conv2DSlowBackpropFilterOp : public OpKernel {
 public:
  explicit Conv2DSlowBackpropFilterOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    int stride_n = GetTensorDim(strides_, data_format_, 'N');
    int stride_c = GetTensorDim(strides_, data_format_, 'C');
    OP_REQUIRES(
        context, (stride_n == 1 && stride_c == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("use_cudnn_on_gpu", &use_cudnn_));
    use_cudnn_ &= CanUseCudnn();
    cudnn_use_autotune_ = CudnnUseAutotune();
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    using perftools::gputools::dnn::AlgorithmConfig;
    using perftools::gputools::dnn::AlgorithmType;
    using perftools::gputools::dnn::ProfileResult;
    using perftools::gputools::dnn::kDefaultAlgorithm;
    const Tensor& input = context->input(0);
    const Tensor& filter_sizes = context->input(1);
    const Tensor& out_backprop = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(filter_sizes.shape()),
        errors::InvalidArgument(
            "Conv2DBackpropFilter: filter_sizes input must be 1-dim, not ",
            filter_sizes.dims()));
    const TensorShape& input_shape = input.shape();
    TensorShape filter_shape;
    OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                filter_sizes.vec<int32>(), &filter_shape));

    Conv2DBackpropDimensions dims;
    OP_REQUIRES_OK(context, Conv2DBackpropComputeDimensions(
                                "Conv2DSlowBackpropFilter", input.shape(),
                                filter_shape, out_backprop.shape(), strides_,
                                padding_, data_format_, &dims));

    Tensor* filter_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, filter_shape, &filter_backprop));

    const int padding_rows =
        (padding_ == VALID)
            ? 0
            : std::max<int>(0, (dims.rows.output_size - 1) * dims.rows.stride +
                                   dims.rows.filter_size -
                                   dims.rows.input_size);
    const int padding_cols =
        (padding_ == VALID)
            ? 0
            : std::max<int>(0, (dims.cols.output_size - 1) * dims.cols.stride +
                                   dims.cols.filter_size -
                                   dims.cols.input_size);

    // TODO(zhengxq): cuDNN only supports equal padding on both sides, so only
    // calling it when that is true. Remove this check when (if?) cuDNN starts
    // supporting different padding.
    bool rows_odd = (padding_rows % 2 != 0);
    bool cols_odd = (padding_cols % 2 != 0);

    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

    if (!use_cudnn_) {
      context->SetStatus(errors::Unimplemented(
          "Conv2DBackprop for GPU is not currently supported "
          "without cudnn"));
      return;
    }

    if (dims.rows.filter_size == 1 && dims.cols.filter_size == 1 &&
        dims.rows.stride == 1 && dims.cols.stride == 1 &&
        data_format_ == FORMAT_NHWC) {
      const uint64 m = dims.in_depth;
      const uint64 k =
          dims.batch_size * dims.rows.input_size * dims.cols.input_size;
      const uint64 n = dims.out_depth;

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
          stream
              ->ThenBlasGemm(perftools::gputools::blas::Transpose::kNoTranspose,
                             perftools::gputools::blas::Transpose::kTranspose,
                             n, m, k, 1.0f, a_ptr, n, b_ptr, m, 0.0f, &c_ptr, n)
              .ok();
      if (!blas_launch_status) {
        context->SetStatus(errors::Internal("Blas SGEMM launch failed : m=", m,
                                            ", n=", n, ", k=", k));
      }
      return;
    }

    Tensor compatible_input;
    if (rows_odd || cols_odd) {
      // If a padding dimension is odd, we have one more element on the right
      // side or the bottom side. This is unsupported in cudnn. Therefore,
      // we pad that extra element and make it compatible.
      OP_REQUIRES_OK(
          context,
          context->allocate_temp(
              DataTypeToEnum<T>::value,
              ShapeFromFormat(data_format_, dims.batch_size,
                              dims.rows.input_size + rows_odd,
                              dims.cols.input_size + cols_odd, dims.in_depth),
              &compatible_input));

      functor::PadInput<GPUDevice, T, int, 4>()(
          context->template eigen_device<GPUDevice>(),
          To32Bit(input.tensor<T, 4>()), {{0, 0}}, {{rows_odd, cols_odd}},
          To32Bit(compatible_input.tensor<T, 4>()), data_format_);
    } else {
      compatible_input = input;
    }

    CHECK(padding_rows >= 0 && padding_cols >= 0)
        << "Negative row or col paddings: (" << padding_rows << ", "
        << padding_cols << ")";
    perftools::gputools::dnn::BatchDescriptor input_desc;
    input_desc.set_count(dims.batch_size)
        .set_height(GetTensorDim(compatible_input, data_format_, 'H'))
        .set_width(GetTensorDim(compatible_input, data_format_, 'W'))
        .set_feature_map_count(dims.in_depth)
        .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);
    perftools::gputools::dnn::BatchDescriptor output_desc;
    output_desc.set_count(dims.batch_size)
        .set_height(dims.rows.output_size)
        .set_width(dims.cols.output_size)
        .set_feature_map_count(dims.out_depth)
        .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);
    perftools::gputools::dnn::FilterDescriptor filter_desc;
    filter_desc.set_input_filter_height(dims.rows.filter_size)
        .set_input_filter_width(dims.cols.filter_size)
        .set_input_feature_map_count(dims.in_depth)
        .set_output_feature_map_count(dims.out_depth);
    perftools::gputools::dnn::ConvolutionDescriptor conv_desc;
    conv_desc.set_vertical_filter_stride(dims.rows.stride)
        .set_horizontal_filter_stride(dims.cols.stride)
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
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataTypeToEnum<T>::value,
                                TensorShape({dims.out_depth, dims.in_depth,
                                             dims.rows.filter_size,
                                             dims.cols.filter_size}),
                                &pre_transformed_filter_backprop));

    Tensor transformed_out_backprop;
    if (data_format_ == FORMAT_NHWC) {
      TensorShape nchw_shape =
          ShapeFromFormat(FORMAT_NCHW, dims.batch_size, dims.rows.output_size,
                          dims.cols.output_size, dims.out_depth);
      if (dims.out_depth > 1) {
        OP_REQUIRES_OK(context, context->allocate_temp(
                                    DataTypeToEnum<T>::value, nchw_shape,
                                    &transformed_out_backprop));
        functor::NHWCToNCHW<Device, T, 4>()(
            context->eigen_device<Device>(), out_backprop.tensor<T, 4>(),
            transformed_out_backprop.tensor<T, 4>());
      } else {
        // If depth <= 1, just reshape.
        CHECK(transformed_out_backprop.CopyFrom(out_backprop, nchw_shape));
      }
    } else {
      transformed_out_backprop = out_backprop;
    }

    Tensor transformed_input;
    if (data_format_ == FORMAT_NHWC) {
      TensorShape nchw_shape = ShapeFromFormat(
          FORMAT_NCHW, GetTensorDim(compatible_input, data_format_, 'N'),
          GetTensorDim(compatible_input, data_format_, 'H'),
          GetTensorDim(compatible_input, data_format_, 'W'),
          GetTensorDim(compatible_input, data_format_, 'C'));
      if (nchw_shape.dim_size(1) > 1) {
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<T>::value,
                                              nchw_shape, &transformed_input));
        functor::NHWCToNCHW<Device, T, 4>()(
            context->eigen_device<Device>(),
            const_cast<const Tensor&>(compatible_input).tensor<T, 4>(),
            transformed_input.tensor<T, 4>());
      } else {
        // If depth <= 1, just reshape.
        CHECK(transformed_input.CopyFrom(compatible_input, nchw_shape));
      }
    } else {
      transformed_input = compatible_input;
    }

    auto out_backprop_ptr =
        AsDeviceMemory(transformed_out_backprop.template flat<T>().data(),
                       transformed_out_backprop.template flat<T>().size());
    auto filter_backprop_ptr = AsDeviceMemory(
        pre_transformed_filter_backprop.template flat<T>().data(),
        pre_transformed_filter_backprop.template flat<T>().size());
    auto input_ptr =
        AsDeviceMemory(transformed_input.template flat<T>().data(),
                       transformed_input.template flat<T>().size());

    static int64 ConvolveBackwardFilterScratchSize = GetCudnnWorkspaceLimit(
        "TF_CUDNN_WORKSPACE_LIMIT_IN_MB", 1LL << 32  // 4GB by default
        );
    int device_id = stream->parent()->device_ordinal();
    ConvParameters conv_parameters = {
        dims.batch_size,        // batch
        dims.in_depth,          // in_depths
        input_desc.height(),    // in_rows
        input_desc.width(),     // in_cols
        dims.out_depth,         // out_depths
        dims.rows.filter_size,  // filter_rows
        dims.cols.filter_size,  // filter_cols
        dims.rows.stride,       // stride_rows
        dims.cols.stride,       // stride_cols
        padding_rows,           // padding_rows
        padding_cols,           // padding_cols
        device_id,              // device_id
    };
    AlgorithmConfig algorithm_config;
    if (cudnn_use_autotune_ &&
        !autotune_results_.Find(conv_parameters, &algorithm_config)) {
      std::vector<AlgorithmType> algorithms;
      CHECK(stream->parent()->GetConvolveBackwardFilterAlgorithms(&algorithms));
      ProfileResult best_result;
      ProfileResult best_result_no_scratch;
      for (auto profile_algorithm : algorithms) {
        // TODO(zhengxq): profile each algorithm multiple times to better
        // accuracy.
        CudnnScratchAllocator scratch_allocator(
            ConvolveBackwardFilterScratchSize, context);
        ProfileResult profile_result;
        bool cudnn_launch_status =
            stream
                ->ThenConvolveBackwardFilterWithAlgorithm(
                    input_desc, input_ptr, output_desc, out_backprop_ptr,
                    conv_desc, filter_desc, &filter_backprop_ptr,
                    &scratch_allocator, AlgorithmConfig(profile_algorithm),
                    &profile_result)
                .ok();
        if (cudnn_launch_status) {
          if (profile_result.is_valid()) {
            if (profile_result.elapsed_time_in_ms() <
                best_result.elapsed_time_in_ms()) {
              best_result = profile_result;
            }
            if (scratch_allocator.TotalByteSize() == 0 &&
                profile_result.elapsed_time_in_ms() <
                    best_result_no_scratch.elapsed_time_in_ms()) {
              best_result_no_scratch = profile_result;
            }
          }
        }
      }
      OP_REQUIRES(context, best_result.is_valid() &&
                               best_result.algorithm() != kDefaultAlgorithm,
                  errors::NotFound("No algorithm worked!"));
      OP_REQUIRES(context,
                  best_result_no_scratch.is_valid() &&
                      best_result_no_scratch.algorithm() != kDefaultAlgorithm,
                  errors::NotFound("No algorithm without scratch worked!"));
      algorithm_config.set_algorithm(best_result.algorithm());
      algorithm_config.set_algorithm_no_scratch(
          best_result_no_scratch.algorithm());
      autotune_results_.Insert(conv_parameters, algorithm_config);
    }
    CudnnScratchAllocator scratch_allocator(ConvolveBackwardFilterScratchSize,
                                            context);
    bool cudnn_launch_status =
        stream
            ->ThenConvolveBackwardFilterWithAlgorithm(
                input_desc, input_ptr, output_desc, out_backprop_ptr, conv_desc,
                filter_desc, &filter_backprop_ptr, &scratch_allocator,
                algorithm_config, nullptr)
            .ok();

    if (!cudnn_launch_status) {
      context->SetStatus(errors::Internal(
          "cuDNN Backward Filter function launch failure : input shape(",
          input_shape.DebugString(), ") filter shape(",
          filter_shape.DebugString(), ")"));
      return;
    }

    auto toConstTensor = [](const Tensor& x) -> const Tensor { return x; };
    functor::ReverseTransformFilter<Device, T, 4>()(
        context->eigen_device<Device>(),
        toConstTensor(pre_transformed_filter_backprop).template tensor<T, 4>(),
        filter_backprop->tensor<T, 4>());
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;
  bool use_cudnn_;
  TensorFormat data_format_;
  AutoTuneMap<ConvParameters, perftools::gputools::dnn::AlgorithmConfig>
      autotune_results_;
  bool cudnn_use_autotune_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv2DSlowBackpropFilterOp);
};

// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                              \
  template <>                                                            \
  void ShuffleAndReverse<GPUDevice, T, 4, int>::operator()(              \
      const GPUDevice& d, typename TTypes<T, 4, int>::ConstTensor input, \
      const Eigen::DSizes<int, 4>& order,                                \
      const Eigen::array<bool, 4>& reverse_dims,                         \
      typename TTypes<T, 4, int>::Tensor output);                        \
  extern template struct ShuffleAndReverse<GPUDevice, T, 4, int>;        \
  template <>                                                            \
  void InflatePadAndShuffle<GPUDevice, T, 4, int>::operator()(           \
      const GPUDevice& d, typename TTypes<T, 4, int>::ConstTensor input, \
      const Eigen::DSizes<int, 4>& strides,                              \
      const Eigen::array<Eigen::IndexPair<int>, 4>& pad_dims,            \
      const Eigen::DSizes<int, 4>& order,                                \
      typename TTypes<T, 4, int>::Tensor output);                        \
  extern template struct InflatePadAndShuffle<GPUDevice, T, 4, int>;     \
  template <>                                                            \
  void TransformFilter<GPUDevice, T, int, 4>::operator()(                \
      const GPUDevice& d, typename TTypes<T, 4, int>::ConstTensor in,    \
      typename TTypes<T, 4, int>::Tensor out);                           \
  extern template struct TransformFilter<GPUDevice, T, int, 4>;          \
  template <>                                                            \
  void TransformDepth<GPUDevice, T, int>::operator()(                    \
      const GPUDevice& d, typename TTypes<T, 4, int>::ConstTensor in,    \
      const Eigen::DSizes<int, 4>& shuffle,                              \
      typename TTypes<T, 4, int>::Tensor out);                           \
  extern template struct TransformDepth<GPUDevice, T, int>;              \
  template <>                                                            \
  void PadInput<GPUDevice, T, int, 4>::operator()(                       \
      const GPUDevice& d, typename TTypes<T, 4, int>::ConstTensor in,    \
      const std::array<int, 2>& padding_left,                            \
      const std::array<int, 2>& padding_right,                           \
      typename TTypes<T, 4, int>::Tensor out, TensorFormat data_format); \
  extern template struct PadInput<GPUDevice, T, int, 4>;

DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(Eigen::half);
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
REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropInput")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<Eigen::half>("T")
                            .HostMemory("input_sizes"),
                        Conv2DSlowBackpropInputOp<GPUDevice, Eigen::half>);
REGISTER_KERNEL_BUILDER(Name("Conv2DBackpropFilter")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<Eigen::half>("T")
                            .HostMemory("filter_sizes"),
                        Conv2DSlowBackpropFilterOp<GPUDevice, Eigen::half>);
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
