/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/conv_3d.h"
#include "tensorflow/core/kernels/conv_grad_ops.h"
#include "tensorflow/core/kernels/conv_grad_shape_utils.h"
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"
#include "tensorflow/core/util/work_sharder.h"

#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
#include "tensorflow/core/kernels/eigen_contraction_kernel.h"
#endif

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/platform/stream_executor.h"
using stream_executor::dnn::DimIndex;
#include "tensorflow/core/protobuf/autotuning.pb.h"
#include "tensorflow/core/util/proto/proto_utils.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#if GOOGLE_CUDA
#include "tensorflow/stream_executor/gpu/asm_compiler.h"
#include "tensorflow/stream_executor/gpu/redzone_allocator.h"
#include "tensorflow/stream_executor/tf_allocator_adapter.h"
#endif  // GOOGLE_CUDA

namespace {

// TODO(ezhulenev): Split this file into conv_grad_filter_ops_3d.cc and
// conv_grad_input_ops_3d.cc.

// TODO(ezhulenev): Generalize Col2im and Im2col for 2-d and 3-d kernels.

// "Depth" is already used for the channel dimension, so for the third spatial
// dimension in this file we use "plane", although in NDHWC layout it's
// indicated with a "D".

// Returns in 'im_data' (assumed to be zero-initialized) image patch in storage
// order (planes, height, width, depth), constructed from patches in 'col_data',
// which is required to be in storage order (out_planes * out_height *
// out_width, filter_planes, filter_height, filter_width, in_depth).
//
// Based on 2-dimensional implementation written by Yangqing Jia (jiayq).
template <typename T>
void Col2im(const T* col_data, const int depth, const int planes,
            const int height, const int width, const int filter_p,
            const int filter_h, const int filter_w, const int pad_pt,
            const int pad_t, const int pad_l, const int pad_pb, const int pad_b,
            const int pad_r, const int stride_p, const int stride_h,
            const int stride_w, T* im_data) {
  const int planes_col = (planes + pad_pt + pad_pb - filter_p) / stride_p + 1;
  const int height_col = (height + pad_t + pad_b - filter_h) / stride_h + 1;
  const int width_col = (width + pad_l + pad_r - filter_w) / stride_w + 1;
  int p_pad = -pad_pt;
  for (int p = 0; p < planes_col; ++p) {
    int h_pad = -pad_t;
    for (int h = 0; h < height_col; ++h) {
      int w_pad = -pad_l;
      for (int w = 0; w < width_col; ++w) {
        T* im_patch_data =
            im_data + (p_pad * height * width + h_pad * width + w_pad) * depth;
        for (int ip = p_pad; ip < p_pad + filter_p; ++ip) {
          for (int ih = h_pad; ih < h_pad + filter_h; ++ih) {
            for (int iw = w_pad; iw < w_pad + filter_w; ++iw) {
              if (ip >= 0 && ip < planes && ih >= 0 && ih < height && iw >= 0 &&
                  iw < width) {
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
          // Jump over remaining number of (depth * width).
          im_patch_data += (depth * width) * (height - filter_h);
        }
        w_pad += stride_w;
      }
      h_pad += stride_h;
    }
    p_pad += stride_p;
  }
}

// Returns in 'col_data', image patches in storage order (planes, height, width,
// depth) extracted from image at 'input_data', which is required to be in
// storage order (batch, planes, height, width, depth).
//
// Based on 2-dimensional implementation written by Yangqing Jia (jiayq).
template <typename T>
void Im2col(const T* input_data, const int depth, const int planes,
            const int height, const int width, const int filter_p,
            const int filter_h, const int filter_w, const int pad_pt,
            const int pad_t, const int pad_l, const int pad_pb, const int pad_b,
            const int pad_r, const int stride_p, const int stride_h,
            const int stride_w, T* col_data) {
  const int planes_col = (planes + pad_pt + pad_pb - filter_p) / stride_p + 1;
  const int height_col = (height + pad_t + pad_b - filter_h) / stride_h + 1;
  const int width_col = (width + pad_l + pad_r - filter_w) / stride_w + 1;

  int p_pad = -pad_pt;
  for (int p = 0; p < planes_col; ++p) {
    int h_pad = -pad_t;
    for (int h = 0; h < height_col; ++h) {
      int w_pad = -pad_l;
      for (int w = 0; w < width_col; ++w) {
        for (int ip = p_pad; ip < p_pad + filter_p; ++ip) {
          for (int ih = h_pad; ih < h_pad + filter_h; ++ih) {
            for (int iw = w_pad; iw < w_pad + filter_w; ++iw) {
              if (ip >= 0 && ip < planes && ih >= 0 && ih < height && iw >= 0 &&
                  iw < width) {
                memcpy(col_data,
                       input_data +
                           (ip * height * width + ih * width + iw) * depth,
                       sizeof(T) * depth);
              } else {
                // This should be simply padded with zero.
                memset(col_data, 0, sizeof(T) * depth);
              }
              col_data += depth;
            }
          }
        }
        w_pad += stride_w;
      }
      h_pad += stride_h;
    }
    p_pad += stride_p;
  }
}

}  // namespace

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// Backprop for input that offloads computation to
// Eigen::CuboidConvolutionBackwardInput.
template <typename Device, class T>
class Conv3DBackpropInputOp : public OpKernel {
 public:
  explicit Conv3DBackpropInputOp(OpKernelConstruction* context)
      : OpKernel(context),
        data_format_(FORMAT_NHWC),
        takes_shape_(type_string().find("V2") != std::string::npos) {
    // data_format is only available in V2.
    if (takes_shape_) {
      string data_format;
      OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
      OP_REQUIRES(
          context, data_format_ == FORMAT_NHWC,
          errors::InvalidArgument(
              "Conv3DBackpropInputOpV2 only supports NDHWC on the CPU."));
    }

    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilation_));
    OP_REQUIRES(context, dilation_.size() == 5,
                errors::InvalidArgument("Dilation rates field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES(context,
                (GetTensorDim(dilation_, data_format_, 'C') == 1 &&
                 GetTensorDim(dilation_, data_format_, 'N') == 1),
                errors::InvalidArgument(
                    "Current implementation does not yet support "
                    "dilation rates in the batch and depth dimensions."));

    // TODO(yangzihao): Add CPU version of dilated conv 3D.
    OP_REQUIRES(context,
                (GetTensorDim(dilation_, data_format_, '0') == 1 &&
                 GetTensorDim(dilation_, data_format_, '1') == 1 &&
                 GetTensorDim(dilation_, data_format_, '2') == 1),
                errors::InvalidArgument(
                    "Current CPU implementation does not yet support "
                    "dilation rates larger than 1."));

    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 5,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES(
        context,
        (GetTensorDim(stride_, data_format_, 'C') == 1 &&
         GetTensorDim(stride_, data_format_, 'N') == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& filter = context->input(1);
    const TensorShape& filter_shape = filter.shape();

    const Tensor& out_backprop = context->input(2);
    const TensorShape& out_backprop_shape = out_backprop.shape();

    TensorShape input_shape;
    if (takes_shape_) {
      const Tensor& input_sizes = context->input(0);
      // MakeShape is able to handle both DT_INT32 and DT_INT64 for input_sizes.
      OP_REQUIRES_OK(context, MakeShape(input_sizes, &input_shape));
    } else {
      input_shape = context->input(0).shape();
    }

    ConvBackpropDimensions dims;
    OP_REQUIRES_OK(context, ConvBackpropComputeDimensions(
                                "Conv3DBackpropInputOp", /*num_spatial_dims=*/3,
                                input_shape, filter_shape, out_backprop_shape,
                                stride_, padding_, data_format_, &dims));

    Tensor* in_backprop;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_shape, &in_backprop));

    functor::CuboidConvolutionBackwardInput<Device, T>()(
        context->eigen_device<Device>(),
        in_backprop->tensor<T, 5>(),                     // input_backward
        filter.tensor<T, 5>(),                           // filter
        out_backprop.tensor<T, 5>(),                     // output_backward
        static_cast<int>(dims.spatial_dims[0].stride),   // stride_planes
        static_cast<int>(dims.spatial_dims[1].stride),   // stride_rows
        static_cast<int>(dims.spatial_dims[2].stride));  // stride_cols
  }

 private:
  std::vector<int32> dilation_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
  bool takes_shape_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv3DBackpropInputOp);
};

// Custom backprop for input that explicitly does the work sharding and calls
// Eigen only to multiply matrices.
template <typename Device, class T>
class Conv3DCustomBackpropInputOp : public OpKernel {
  // Limit the maximum size of allocated temporary buffer to
  // kMaxTempAllocationOverhead times the size of the input tensors (input,
  // filter, out_backprop). If the size of the temporary buffer exceeds this
  // limit, fallback on Eigen implementation.
  static constexpr int kMaxTempAllocationOverhead = 25;

 public:
  explicit Conv3DCustomBackpropInputOp(OpKernelConstruction* context)
      : OpKernel(context),
        data_format_(FORMAT_NHWC),
        takes_shape_(type_string().find("V2") != std::string::npos) {
    // data_format is only available in V2.
    if (takes_shape_) {
      string data_format;
      OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
      OP_REQUIRES(
          context, data_format_ == FORMAT_NHWC,
          errors::InvalidArgument(
              "Conv3DBackpropInputOpV2 only supports NDHWC on the CPU."));
    }

    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilation_));
    OP_REQUIRES(context, dilation_.size() == 5,
                errors::InvalidArgument("Dilation rates field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES(context,
                (GetTensorDim(dilation_, data_format_, 'C') == 1 &&
                 GetTensorDim(dilation_, data_format_, 'N') == 1),
                errors::InvalidArgument(
                    "Current implementation does not yet support "
                    "dilation rates in the batch and depth dimensions."));

    // TODO(yangzihao): Add CPU version of dilated conv 3D.
    OP_REQUIRES(context,
                (GetTensorDim(dilation_, data_format_, '0') == 1 &&
                 GetTensorDim(dilation_, data_format_, '1') == 1 &&
                 GetTensorDim(dilation_, data_format_, '2') == 1),
                errors::InvalidArgument(
                    "Current CPU implementation does not yet support "
                    "dilation rates larger than 1."));

    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 5,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES(
        context,
        (GetTensorDim(stride_, data_format_, 'C') == 1 &&
         GetTensorDim(stride_, data_format_, 'N') == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& filter = context->input(1);
    const TensorShape& filter_shape = filter.shape();

    const Tensor& out_backprop = context->input(2);
    const TensorShape& out_backprop_shape = out_backprop.shape();

    TensorShape input_shape;
    if (takes_shape_) {
      const Tensor& input_sizes = context->input(0);
      // MakeShape is able to handle both DT_INT32 and DT_INT64 for input_sizes.
      OP_REQUIRES_OK(context, MakeShape(input_sizes, &input_shape));
    } else {
      input_shape = context->input(0).shape();
    }

    ConvBackpropDimensions dims;
    OP_REQUIRES_OK(context, ConvBackpropComputeDimensions(
                                "Conv3DBackpropInputOp", /*num_spatial_dims=*/3,
                                input_shape, filter_shape, out_backprop_shape,
                                stride_, padding_, data_format_, &dims));

    Tensor* in_backprop;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_shape, &in_backprop));

    int64 top_pad_planes, bottom_pad_planes;
    int64 top_pad_rows, bottom_pad_rows;
    int64 left_pad_cols, right_pad_cols;

    OP_REQUIRES_OK(context, GetWindowedOutputSizeVerbose(
                                dims.spatial_dims[0].input_size,
                                dims.spatial_dims[0].filter_size,
                                dims.spatial_dims[0].stride, padding_,
                                &dims.spatial_dims[0].output_size,
                                &top_pad_planes, &bottom_pad_planes));
    OP_REQUIRES_OK(context, GetWindowedOutputSizeVerbose(
                                dims.spatial_dims[1].input_size,
                                dims.spatial_dims[1].filter_size,
                                dims.spatial_dims[1].stride, padding_,
                                &dims.spatial_dims[1].output_size,
                                &top_pad_rows, &bottom_pad_rows));
    OP_REQUIRES_OK(context, GetWindowedOutputSizeVerbose(
                                dims.spatial_dims[2].input_size,
                                dims.spatial_dims[2].filter_size,
                                dims.spatial_dims[2].stride, padding_,
                                &dims.spatial_dims[2].output_size,
                                &left_pad_cols, &right_pad_cols));

    // TODO(ezhulenev): Extract work size and shard estimation to shared
    // functions in conv_grad_ops, and update 2d convolution backprop.

    // The total dimension size of each kernel.
    const int64 filter_total_size =
        dims.spatial_dims[0].filter_size * dims.spatial_dims[1].filter_size *
        dims.spatial_dims[2].filter_size * dims.in_depth;

    // The output image size is the spatial size of the output.
    const int64 output_image_size = dims.spatial_dims[0].output_size *
                                    dims.spatial_dims[1].output_size *
                                    dims.spatial_dims[2].output_size;

    const auto cache_sizes = Eigen::internal::CacheSizes();
    const ptrdiff_t l3_cache_size = cache_sizes.m_l3;

    // Use L3 cache size as target working set size.
    const size_t target_working_set_size = l3_cache_size / sizeof(T);

    // Calculate size of matrices involved in MatMul: C = A x B.
    const int64 size_A = output_image_size * dims.out_depth;

    const int64 size_B = filter_total_size * dims.out_depth;

    const int64 size_C = output_image_size * filter_total_size;

    const int64 work_unit_size = size_A + size_B + size_C;

    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());

    // Use parallel tensor contractions if there is no batching.
    //
    // Compared to Conv2D code, this version is missing work size estimation. In
    // benchmarks I didn't find a case when it's beneficial to run parallel
    // contraction compared to sharding and matmuls.
    const bool use_parallel_contraction = dims.batch_size == 1;

    const size_t shard_size =
        use_parallel_contraction
            ? 1
            : (target_working_set_size + work_unit_size - 1) / work_unit_size;

    // Total number of elements in all the tensors used by this kernel.
    int64 total_tensor_elements = input_shape.num_elements() +
                                  filter_shape.num_elements() +
                                  out_backprop_shape.num_elements();

    // Shape of the temporary workspace buffer.
    TensorShape col_buffer_shape = {static_cast<int64>(shard_size),
                                    static_cast<int64>(output_image_size),
                                    static_cast<int64>(filter_total_size)};
    int64 col_buffer_elements = col_buffer_shape.num_elements();

    // If the temporary allocation overhead is too large, fallback on Eigen
    // implementation which requires much less memory.
    int64 col_buffer_overhead = col_buffer_elements / total_tensor_elements;
    if (col_buffer_overhead > kMaxTempAllocationOverhead) {
      VLOG(2) << "Fallback on Eigen implementation of Conv3DBackpropInputOp: "
                 "col_buffer_overhead="
              << col_buffer_overhead;

      functor::CuboidConvolutionBackwardInput<Device, T>()(
          context->eigen_device<Device>(),
          in_backprop->tensor<T, 5>(),                     // input_backward
          filter.tensor<T, 5>(),                           // filter
          out_backprop.tensor<T, 5>(),                     // output_backward
          static_cast<int>(dims.spatial_dims[0].stride),   // stride_planes
          static_cast<int>(dims.spatial_dims[1].stride),   // stride_rows
          static_cast<int>(dims.spatial_dims[2].stride));  // stride_cols

      return;
    }

    Tensor col_buffer;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<T>::value,
                                          col_buffer_shape, &col_buffer));

    // The input offset corresponding to a single input image.
    const int64 input_offset = dims.spatial_dims[0].input_size *
                               dims.spatial_dims[1].input_size *
                               dims.spatial_dims[2].input_size * dims.in_depth;

    // The output offset corresponding to a single output image.
    const int64 output_offset =
        dims.spatial_dims[0].output_size * dims.spatial_dims[1].output_size *
        dims.spatial_dims[2].output_size * dims.out_depth;

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

        Col2im<T>(col_buffer_data, dims.in_depth,
                  // Input spatial dimensions.
                  dims.spatial_dims[0].input_size,  // input planes
                  dims.spatial_dims[1].input_size,  // input rows
                  dims.spatial_dims[2].input_size,  // input cols
                  // Filter spatial dimensions.
                  dims.spatial_dims[0].filter_size,  // filter planes
                  dims.spatial_dims[1].filter_size,  // filter rows
                  dims.spatial_dims[2].filter_size,  // filter cols
                  // Spatial padding.
                  top_pad_planes, top_pad_rows, left_pad_cols,
                  bottom_pad_planes, bottom_pad_rows, right_pad_cols,
                  // Spatial striding.
                  dims.spatial_dims[0].stride,  // stride planes
                  dims.spatial_dims[1].stride,  // stride rows
                  dims.spatial_dims[2].stride,  // stride cols
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

        auto shard = [&dims, &top_pad_planes, &top_pad_rows, &left_pad_cols,
                      &bottom_pad_planes, &bottom_pad_rows, &right_pad_cols,
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

            Col2im<T>(im2col_buf, dims.in_depth,
                      // Input spatial dimensions.
                      dims.spatial_dims[0].input_size,  // input planes
                      dims.spatial_dims[1].input_size,  // input rows
                      dims.spatial_dims[2].input_size,  // input cols
                      // Filter spatial dimensions.
                      dims.spatial_dims[0].filter_size,  // filter planes
                      dims.spatial_dims[1].filter_size,  // filter rows
                      dims.spatial_dims[2].filter_size,  // filter cols
                      // Spatial padding.
                      top_pad_planes, top_pad_rows, left_pad_cols,
                      bottom_pad_planes, bottom_pad_rows, right_pad_cols,
                      // Spatial striding.
                      dims.spatial_dims[0].stride,  // stride planes
                      dims.spatial_dims[1].stride,  // stride rows
                      dims.spatial_dims[2].stride,  // stride cols
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
  std::vector<int32> dilation_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
  bool takes_shape_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv3DCustomBackpropInputOp);
};

// Custom backrop input kernel is 30% - 4x faster when compiled with AVX2 than
// default Eigen implementation (at the cost of ~2x-8x peak memory usage).

#define REGISTER_CPU_KERNEL(T)                                                 \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("Conv3DBackpropInput").Device(DEVICE_CPU).TypeConstraint<T>("T"),   \
      Conv3DCustomBackpropInputOp<CPUDevice, T>);                              \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("Conv3DBackpropInputV2").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      Conv3DCustomBackpropInputOp<CPUDevice, T>);                              \
  REGISTER_KERNEL_BUILDER(Name("Conv3DBackpropInput")                          \
                              .Device(DEVICE_CPU)                              \
                              .Label("custom")                                 \
                              .TypeConstraint<T>("T"),                         \
                          Conv3DCustomBackpropInputOp<CPUDevice, T>);          \
  REGISTER_KERNEL_BUILDER(Name("Conv3DBackpropInputV2")                        \
                              .Device(DEVICE_CPU)                              \
                              .Label("custom")                                 \
                              .TypeConstraint<T>("T"),                         \
                          Conv3DCustomBackpropInputOp<CPUDevice, T>);          \
  REGISTER_KERNEL_BUILDER(Name("Conv3DBackpropInput")                          \
                              .Device(DEVICE_CPU)                              \
                              .Label("eigen_tensor")                           \
                              .TypeConstraint<T>("T"),                         \
                          Conv3DBackpropInputOp<CPUDevice, T>);                \
  REGISTER_KERNEL_BUILDER(Name("Conv3DBackpropInputV2")                        \
                              .Device(DEVICE_CPU)                              \
                              .Label("eigen_tensor")                           \
                              .TypeConstraint<T>("T"),                         \
                          Conv3DBackpropInputOp<CPUDevice, T>);

TF_CALL_half(REGISTER_CPU_KERNEL);
TF_CALL_float(REGISTER_CPU_KERNEL);
TF_CALL_double(REGISTER_CPU_KERNEL);
#undef REGISTER_CPU_KERNEL

// Backprop for filter that offloads computation to
// Eigen::CuboidConvolutionBackwardFilter.
template <typename Device, class T>
class Conv3DBackpropFilterOp : public OpKernel {
 public:
  explicit Conv3DBackpropFilterOp(OpKernelConstruction* context)
      : OpKernel(context),
        data_format_(FORMAT_NHWC),
        takes_shape_(type_string().find("V2") != std::string::npos) {
    // data_format is only available in V2.
    if (takes_shape_) {
      string data_format;
      OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
      OP_REQUIRES(
          context, data_format_ == FORMAT_NHWC,
          errors::InvalidArgument(
              "Conv3DBackpropFilterOpV2 only supports NDHWC on the CPU."));
    }

    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilation_));
    OP_REQUIRES(context, dilation_.size() == 5,
                errors::InvalidArgument("Dilation rates field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES(context,
                (GetTensorDim(dilation_, data_format_, 'C') == 1 &&
                 GetTensorDim(dilation_, data_format_, 'N') == 1),
                errors::InvalidArgument(
                    "Current implementation does not yet support "
                    "dilation rates in the batch and depth dimensions."));

    // TODO(yangzihao): Add CPU version of dilated conv 3D.
    OP_REQUIRES(context,
                (GetTensorDim(dilation_, data_format_, '0') == 1 &&
                 GetTensorDim(dilation_, data_format_, '1') == 1 &&
                 GetTensorDim(dilation_, data_format_, '2') == 1),
                errors::InvalidArgument(
                    "Current CPU implementation does not yet support "
                    "dilation rates larger than 1."));

    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 5,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES(
        context,
        (GetTensorDim(stride_, data_format_, 'C') == 1 &&
         GetTensorDim(stride_, data_format_, 'N') == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const TensorShape& input_shape = input.shape();

    const Tensor& out_backprop = context->input(2);
    const TensorShape& out_backprop_shape = out_backprop.shape();

    TensorShape filter_shape;
    if (takes_shape_) {
      const Tensor& filter_sizes = context->input(1);
      OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                  filter_sizes.vec<int32>(), &filter_shape));
    } else {
      filter_shape = context->input(1).shape();
    }

    ConvBackpropDimensions dims;
    OP_REQUIRES_OK(context,
                   ConvBackpropComputeDimensions(
                       "Conv3DBackpropFilterOp", /*num_spatial_dims=*/3,
                       input_shape, filter_shape, out_backprop_shape, stride_,
                       padding_, data_format_, &dims));

    Tensor* filter_backprop;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, filter_shape, &filter_backprop));

    if (input_shape.num_elements() == 0) {
      filter_backprop->template flat<T>().setZero();
      return;
    }

    functor::CuboidConvolutionBackwardFilter<Device, T>()(
        context->eigen_device<Device>(),
        filter_backprop->tensor<T, 5>(),                 // filter_backward
        input.tensor<T, 5>(),                            // input
        out_backprop.tensor<T, 5>(),                     // output_backward
        static_cast<int>(dims.spatial_dims[0].stride),   // stride_planes
        static_cast<int>(dims.spatial_dims[1].stride),   // stride_rows
        static_cast<int>(dims.spatial_dims[2].stride));  // stride_cols
  }

 private:
  std::vector<int32> dilation_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
  bool takes_shape_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv3DBackpropFilterOp);
};

// Custom backprop for filter that explicitly does the work sharding and calls
// Eigen only to multiply matrices.
template <typename Device, class T>
class Conv3DCustomBackpropFilterOp : public OpKernel {
  // Limit the maximum size of allocated temporary buffer to
  // kMaxTempAllocationOverhead times the size of the input tensors (input,
  // filter, out_backprop). If the size of the temporary buffer exceeds this
  // limit, fallback on Eigen implementation.
  static constexpr int kMaxTempAllocationOverhead = 25;

 public:
  explicit Conv3DCustomBackpropFilterOp(OpKernelConstruction* context)
      : OpKernel(context),
        data_format_(FORMAT_NHWC),
        takes_shape_(type_string().find("V2") != std::string::npos) {
    // data_format is only available in V2.
    if (takes_shape_) {
      string data_format;
      OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
      OP_REQUIRES(
          context, data_format_ == FORMAT_NHWC,
          errors::InvalidArgument(
              "Conv3DBackpropFilterOpV2 only supports NDHWC on the CPU."));
    }

    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilation_));
    OP_REQUIRES(context, dilation_.size() == 5,
                errors::InvalidArgument("Dilation rates field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES(context,
                (GetTensorDim(dilation_, data_format_, 'C') == 1 &&
                 GetTensorDim(dilation_, data_format_, 'N') == 1),
                errors::InvalidArgument(
                    "Current implementation does not yet support "
                    "dilation rates in the batch and depth dimensions."));

    // TODO(yangzihao): Add CPU version of dilated conv 3D.
    OP_REQUIRES(context,
                (GetTensorDim(dilation_, data_format_, '0') == 1 &&
                 GetTensorDim(dilation_, data_format_, '1') == 1 &&
                 GetTensorDim(dilation_, data_format_, '2') == 1),
                errors::InvalidArgument(
                    "Current CPU implementation does not yet support "
                    "dilation rates larger than 1."));

    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 5,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES(
        context,
        (GetTensorDim(stride_, data_format_, 'C') == 1 &&
         GetTensorDim(stride_, data_format_, 'N') == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const TensorShape& input_shape = input.shape();

    const Tensor& out_backprop = context->input(2);
    const TensorShape& out_backprop_shape = out_backprop.shape();

    TensorShape filter_shape;
    if (takes_shape_) {
      const Tensor& filter_sizes = context->input(1);
      OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                  filter_sizes.vec<int32>(), &filter_shape));
    } else {
      filter_shape = context->input(1).shape();
    }

    ConvBackpropDimensions dims;
    OP_REQUIRES_OK(context,
                   ConvBackpropComputeDimensions(
                       "Conv3DBackpropFilterOp", /*num_spatial_dims=*/3,
                       input_shape, filter_shape, out_backprop_shape, stride_,
                       padding_, data_format_, &dims));

    Tensor* filter_backprop;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, filter_shape, &filter_backprop));

    if (input_shape.num_elements() == 0) {
      filter_backprop->template flat<T>().setZero();
      return;
    }

    int64 top_pad_planes, bottom_pad_planes;
    int64 top_pad_rows, bottom_pad_rows;
    int64 left_pad_cols, right_pad_cols;

    OP_REQUIRES_OK(context, GetWindowedOutputSizeVerbose(
                                dims.spatial_dims[0].input_size,
                                dims.spatial_dims[0].filter_size,
                                dims.spatial_dims[0].stride, padding_,
                                &dims.spatial_dims[0].output_size,
                                &top_pad_planes, &bottom_pad_planes));
    OP_REQUIRES_OK(context, GetWindowedOutputSizeVerbose(
                                dims.spatial_dims[1].input_size,
                                dims.spatial_dims[1].filter_size,
                                dims.spatial_dims[1].stride, padding_,
                                &dims.spatial_dims[1].output_size,
                                &top_pad_rows, &bottom_pad_rows));
    OP_REQUIRES_OK(context, GetWindowedOutputSizeVerbose(
                                dims.spatial_dims[2].input_size,
                                dims.spatial_dims[2].filter_size,
                                dims.spatial_dims[2].stride, padding_,
                                &dims.spatial_dims[2].output_size,
                                &left_pad_cols, &right_pad_cols));

    // TODO(ezhulenev): Extract work size and shard estimation to shared
    // functions in conv_grad_ops, and update 2d convolution backprop.

    // The total dimension size of each kernel.
    const int64 filter_total_size =
        dims.spatial_dims[0].filter_size * dims.spatial_dims[1].filter_size *
        dims.spatial_dims[2].filter_size * dims.in_depth;
    // The output image size is the spatial size of the output.
    const int64 output_image_size = dims.spatial_dims[0].output_size *
                                    dims.spatial_dims[1].output_size *
                                    dims.spatial_dims[2].output_size;

    // Shard 'batch' images (volumes) into 'shard_size' groups of images
    // (volumes) to be fed into the parallel matmul. Calculate 'shard_size' by
    // dividing the L3 cache size ('target_working_set_size') by the matmul size
    // of an individual image ('work_unit_size').

    const auto cache_sizes = Eigen::internal::CacheSizes();
    const ptrdiff_t l3_cache_size = cache_sizes.m_l3;

    // TODO(andydavis)
    // *) Consider reducing 'target_working_set_size' if L3 is shared by
    //    other concurrently running tensorflow ops.
    const size_t target_working_set_size = l3_cache_size / sizeof(T);

    const int64 size_A = output_image_size * filter_total_size;

    const int64 size_B = output_image_size * dims.out_depth;

    const int64 size_C = filter_total_size * dims.out_depth;

    const int64 work_unit_size = size_A + size_B + size_C;

    const size_t shard_size =
        (target_working_set_size + work_unit_size - 1) / work_unit_size;

    // Total number of elements in all the tensors used by this kernel.
    int64 total_tensor_elements = input_shape.num_elements() +
                                  filter_shape.num_elements() +
                                  out_backprop_shape.num_elements();

    // Shape of the temporary workspace buffer.
    TensorShape col_buffer_shape = {static_cast<int64>(shard_size),
                                    static_cast<int64>(output_image_size),
                                    static_cast<int64>(filter_total_size)};
    int64 col_buffer_elements = col_buffer_shape.num_elements();

    // If the temporary allocation overhead is too large, fallback on Eigen
    // implementation which requires much less memory.
    int64 col_buffer_overhead = col_buffer_elements / total_tensor_elements;
    if (col_buffer_overhead > kMaxTempAllocationOverhead) {
      VLOG(2) << "Fallback on Eigen implementation of Conv3DBackpropFilterOp: "
                 "col_buffer_overhead="
              << col_buffer_overhead;

      functor::CuboidConvolutionBackwardFilter<Device, T>()(
          context->eigen_device<Device>(),
          filter_backprop->tensor<T, 5>(),                 // filter_backward
          input.tensor<T, 5>(),                            // input
          out_backprop.tensor<T, 5>(),                     // output_backward
          static_cast<int>(dims.spatial_dims[0].stride),   // stride_planes
          static_cast<int>(dims.spatial_dims[1].stride),   // stride_rows
          static_cast<int>(dims.spatial_dims[2].stride));  // stride_cols

      return;
    }

    Tensor col_buffer;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<T>::value,
                                          col_buffer_shape, &col_buffer));

    // The input offset corresponding to a single input image.
    const int64 input_offset = dims.spatial_dims[0].input_size *
                               dims.spatial_dims[1].input_size *
                               dims.spatial_dims[2].input_size * dims.in_depth;
    // The output offset corresponding to a single output image.
    const int64 output_offset =
        dims.spatial_dims[0].output_size * dims.spatial_dims[1].output_size *
        dims.spatial_dims[2].output_size * dims.out_depth;

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

      auto shard = [&input_data, &col_buffer_data, &dims, &top_pad_planes,
                    &top_pad_rows, &left_pad_cols, &bottom_pad_planes,
                    &bottom_pad_rows, &right_pad_cols, &input_offset,
                    &size_A](int64 start, int64 limit) {
        for (int shard_id = start; shard_id < limit; ++shard_id) {
          const T* input_data_shard = input_data + shard_id * input_offset;
          T* col_data_shard = col_buffer_data + shard_id * size_A;

          // When we compute the gradient with respect to the filters, we need
          // to do im2col to allow gemm-type computation.
          Im2col<T>(input_data_shard, dims.in_depth,
                    // Input spatial dimensions.
                    dims.spatial_dims[0].input_size,  // input planes
                    dims.spatial_dims[1].input_size,  // input rows
                    dims.spatial_dims[2].input_size,  // input cols
                    // Filter spatial dimensions.
                    dims.spatial_dims[0].filter_size,  // filter planes
                    dims.spatial_dims[1].filter_size,  // filter rows
                    dims.spatial_dims[2].filter_size,  // filter cols
                    // Spatial padding.
                    top_pad_planes, top_pad_rows, left_pad_cols,
                    bottom_pad_planes, bottom_pad_rows, right_pad_cols,
                    // Spatial striding.
                    dims.spatial_dims[0].stride,  // stride planes
                    dims.spatial_dims[1].stride,  // stride rows
                    dims.spatial_dims[2].stride,  // stride cols
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
  std::vector<int32> dilation_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
  bool takes_shape_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv3DCustomBackpropFilterOp);
};

// Custom backrop input kernel is 30% - 4x faster when compiled with AVX2 than
// default Eigen implementation (at the cost of ~2x-8x peak memory usage).

#define REGISTER_CPU_KERNEL(T)                                                \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("Conv3DBackpropFilter").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      Conv3DCustomBackpropFilterOp<CPUDevice, T>);                            \
  REGISTER_KERNEL_BUILDER(Name("Conv3DBackpropFilterV2")                      \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<T>("T"),                        \
                          Conv3DCustomBackpropFilterOp<CPUDevice, T>);        \
  REGISTER_KERNEL_BUILDER(Name("Conv3DBackpropFilter")                        \
                              .Device(DEVICE_CPU)                             \
                              .Label("custom")                                \
                              .TypeConstraint<T>("T"),                        \
                          Conv3DCustomBackpropFilterOp<CPUDevice, T>);        \
  REGISTER_KERNEL_BUILDER(Name("Conv3DBackpropFilterV2")                      \
                              .Device(DEVICE_CPU)                             \
                              .Label("custom")                                \
                              .TypeConstraint<T>("T"),                        \
                          Conv3DCustomBackpropFilterOp<CPUDevice, T>);        \
  REGISTER_KERNEL_BUILDER(Name("Conv3DBackpropFilter")                        \
                              .Device(DEVICE_CPU)                             \
                              .Label("eigen_tensor")                          \
                              .TypeConstraint<T>("T"),                        \
                          Conv3DBackpropFilterOp<CPUDevice, T>);              \
  REGISTER_KERNEL_BUILDER(Name("Conv3DBackpropFilterV2")                      \
                              .Device(DEVICE_CPU)                             \
                              .Label("eigen_tensor")                          \
                              .TypeConstraint<T>("T"),                        \
                          Conv3DBackpropFilterOp<CPUDevice, T>);

TF_CALL_float(REGISTER_CPU_KERNEL);
TF_CALL_double(REGISTER_CPU_KERNEL);
#undef REGISTER_CPU_KERNEL

// WARNING: Eigen::half is not trivially copyable and can't be used in
// custom backprop filter kernel because of memcpy and memset in Im2col.
#define REGISTER_CPU_KERNEL(T)                                                \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("Conv3DBackpropFilter").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      Conv3DBackpropFilterOp<CPUDevice, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("Conv3DBackpropFilterV2")                      \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<T>("T"),                        \
                          Conv3DBackpropFilterOp<CPUDevice, T>);

TF_CALL_half(REGISTER_CPU_KERNEL);
#undef REGISTER_CPU_KERNEL

// GPU definitions of both ops.
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
// This ensures that the custom implementation is used instead of the default
// Eigen one (which is used for CPU).
namespace functor {
#define DECLARE_GPU_SPEC(T)                                           \
  template <>                                                         \
  void TransformFilter<GPUDevice, T, int, 5>::operator()(             \
      const GPUDevice& d, FilterTensorFormat dst_filter_format,       \
      typename TTypes<T, 5, int>::ConstTensor in,                     \
      typename TTypes<T, 5, int>::Tensor out);                        \
  template <>                                                         \
  void ReverseTransformFilter<GPUDevice, T, 5>::operator()(           \
      const GPUDevice& d, FilterTensorFormat src_filter_format,       \
      typename TTypes<T, 5>::ConstTensor in,                          \
      typename TTypes<T, 5>::Tensor out);                             \
  template <>                                                         \
  void PadInput<GPUDevice, T, int, 5>::operator()(                    \
      const GPUDevice& d, typename TTypes<T, 5, int>::ConstTensor in, \
      const std::array<int, 3>& padding_left,                         \
      const std::array<int, 3>& padding_right,                        \
      typename TTypes<T, 5, int>::Tensor out, TensorFormat format);

DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  // namespace functor

// A dummy type to group backward data autotune results together.
struct Conv3dBackwardDataAutoTuneGroup {
  static string name() { return "Conv3dBwdData"; }
};
typedef AutoTuneSingleton<Conv3dBackwardDataAutoTuneGroup, ConvParameters,
                          se::dnn::AlgorithmConfig>

    AutoTuneConv3dBwdData;
template <typename T>
class Conv3DBackpropInputOp<GPUDevice, T> : public OpKernel {
 public:
  explicit Conv3DBackpropInputOp(OpKernelConstruction* context)
      : OpKernel(context),
        data_format_(FORMAT_NHWC),
        takes_shape_(type_string().find("V2") != std::string::npos) {
    // data_format is only available in V2.
    if (takes_shape_) {
      string data_format;
      OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
    }
    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilation_));
    OP_REQUIRES(context, dilation_.size() == 5,
                errors::InvalidArgument("Dilation rates field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES(context,
                (GetTensorDim(dilation_, data_format_, 'C') == 1 &&
                 GetTensorDim(dilation_, data_format_, 'N') == 1),
                errors::InvalidArgument(
                    "Current implementation does not yet support "
                    "dilation rates in the batch and depth dimensions."));
    OP_REQUIRES(
        context,
        (GetTensorDim(dilation_, data_format_, '0') > 0 &&
         GetTensorDim(dilation_, data_format_, '1') > 0 &&
         GetTensorDim(dilation_, data_format_, '2') > 0),
        errors::InvalidArgument("Dilated rates should be larger than 0."));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 5,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES(
        context,
        (GetTensorDim(stride_, data_format_, 'C') == 1 &&
         GetTensorDim(stride_, data_format_, 'N') == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES(
        context,
        (GetTensorDim(stride_, data_format_, '0') > 0 &&
         GetTensorDim(stride_, data_format_, '1') > 0 &&
         GetTensorDim(stride_, data_format_, '2') > 0),
        errors::InvalidArgument("Spatial strides should be larger than 0."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    cudnn_use_autotune_ = CudnnUseAutotune();
  }
  void Compute(OpKernelContext* context) override {
    const Tensor& filter = context->input(1);
    const TensorShape& filter_shape = filter.shape();

    const Tensor& out_backprop = context->input(2);
    const TensorShape& out_backprop_shape = out_backprop.shape();

    TensorShape input_shape;
    if (takes_shape_) {
      const Tensor& input_sizes = context->input(0);
      OP_REQUIRES_OK(context, MakeShape(input_sizes, &input_shape));
    } else {
      input_shape = context->input(0).shape();
    }

    ConvBackpropDimensions dims;
    OP_REQUIRES_OK(context, ConvBackpropComputeDimensionsV2(
                                "Conv3DBackpropInputOp", /*num_spatial_dims=*/3,
                                input_shape, filter_shape, out_backprop_shape,
                                dilation_, stride_, padding_,
                                /*explicit_paddings=*/{}, data_format_, &dims));

    Tensor* in_backprop;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_shape, &in_backprop));

    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

    bool is_grouped_convolution = filter_shape.dim_size(3) != dims.in_depth;
    if (!is_grouped_convolution && dims.filter_size(0) == 1 &&
        dims.filter_size(1) == 1 && dims.filter_size(2) == 1 &&
        dims.dilation(0) == 1 && dims.dilation(1) == 1 &&
        dims.dilation(2) == 1 && dims.stride(0) == 1 && dims.stride(1) == 1 &&
        dims.stride(2) == 1 && data_format_ == FORMAT_NHWC) {
      const uint64 m = dims.batch_size * dims.input_size(0) *
                       dims.input_size(1) * dims.input_size(2);
      const uint64 k = dims.out_depth;
      const uint64 n = dims.in_depth;

      auto a_ptr = AsDeviceMemory(out_backprop.template flat<T>().data(),
                                  out_backprop.template flat<T>().size());
      auto b_ptr = AsDeviceMemory(filter.template flat<T>().data(),
                                  filter.template flat<T>().size());
      auto c_ptr = AsDeviceMemory(in_backprop->template flat<T>().data(),
                                  in_backprop->template flat<T>().size());

      auto transpose = se::blas::Transpose::kTranspose;
      auto no_transpose = se::blas::Transpose::kNoTranspose;

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
    } else if (!is_grouped_convolution &&
               dims.filter_size(0) == dims.input_size(0) &&
               dims.filter_size(1) == dims.input_size(1) &&
               dims.filter_size(2) == dims.input_size(2) &&
               padding_ == Padding::VALID && data_format_ == FORMAT_NHWC) {
      const uint64 m = dims.batch_size;
      const uint64 k = dims.out_depth;
      const uint64 n = dims.input_size(0) * dims.input_size(1) *
                       dims.input_size(2) * dims.in_depth;

      auto a_ptr = AsDeviceMemory(out_backprop.template flat<T>().data(),
                                  out_backprop.template flat<T>().size());
      auto b_ptr = AsDeviceMemory(filter.template flat<T>().data(),
                                  filter.template flat<T>().size());
      auto c_ptr = AsDeviceMemory(in_backprop->template flat<T>().data(),
                                  in_backprop->template flat<T>().size());

      auto transpose = se::blas::Transpose::kTranspose;
      auto no_transpose = se::blas::Transpose::kNoTranspose;

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

    int padding_planes = dims.SpatialPadding(padding_, 0);
    int padding_rows = dims.SpatialPadding(padding_, 1);
    int padding_cols = dims.SpatialPadding(padding_, 2);
    const bool planes_odd = (padding_planes % 2 != 0);
    const bool rows_odd = (padding_rows % 2 != 0);
    const bool cols_odd = (padding_cols % 2 != 0);

    TensorShape compatible_input_shape;
    if (rows_odd || cols_odd || planes_odd) {
      // cuDNN only supports the same amount of padding on both sides.
      compatible_input_shape = {
          dims.batch_size,
          dims.in_depth,
          dims.input_size(0) + planes_odd,
          dims.input_size(1) + rows_odd,
          dims.input_size(2) + cols_odd,
      };
    } else {
      compatible_input_shape = {dims.batch_size, dims.in_depth,
                                dims.input_size(0), dims.input_size(1),
                                dims.input_size(2)};
    }

    CHECK(padding_rows >= 0 && padding_cols >= 0 && padding_planes >= 0)
        << "Negative paddings: (" << padding_rows << ", " << padding_cols
        << ", " << padding_planes << ")";
    se::dnn::BatchDescriptor input_desc(3);
    input_desc.set_count(dims.batch_size)
        .set_spatial_dim(DimIndex::X, compatible_input_shape.dim_size(4))
        .set_spatial_dim(DimIndex::Y, compatible_input_shape.dim_size(3))
        .set_spatial_dim(DimIndex::Z, compatible_input_shape.dim_size(2))
        .set_feature_map_count(dims.in_depth)
        .set_layout(se::dnn::DataLayout::kBatchDepthYX);
    se::dnn::BatchDescriptor output_desc(3);
    output_desc.set_count(dims.batch_size)
        .set_spatial_dim(DimIndex::X, dims.output_size(2))
        .set_spatial_dim(DimIndex::Y, dims.output_size(1))
        .set_spatial_dim(DimIndex::Z, dims.output_size(0))
        .set_feature_map_count(dims.out_depth)
        .set_layout(se::dnn::DataLayout::kBatchDepthYX);
    se::dnn::FilterDescriptor filter_desc(3);
    filter_desc.set_spatial_dim(DimIndex::X, dims.filter_size(2))
        .set_spatial_dim(DimIndex::Y, dims.filter_size(1))
        .set_spatial_dim(DimIndex::Z, dims.filter_size(0))
        .set_input_feature_map_count(filter_shape.dim_size(3))
        .set_output_feature_map_count(filter_shape.dim_size(4));
    se::dnn::ConvolutionDescriptor conv_desc(3);
    conv_desc.set_dilation_rate(DimIndex::X, dims.dilation(2))
        .set_dilation_rate(DimIndex::Y, dims.dilation(1))
        .set_dilation_rate(DimIndex::Z, dims.dilation(0))
        .set_filter_stride(DimIndex::X, dims.stride(2))
        .set_filter_stride(DimIndex::Y, dims.stride(1))
        .set_filter_stride(DimIndex::Z, dims.stride(0))
        .set_zero_padding(DimIndex::X, padding_cols / 2)
        .set_zero_padding(DimIndex::Y, padding_rows / 2)
        .set_zero_padding(DimIndex::Z, padding_planes / 2)
        .set_group_count(dims.in_depth / filter_shape.dim_size(3));

    // Shape: out, in, z, y, x.
    Tensor transformed_filter;
    OP_REQUIRES_OK(
        context, context->allocate_temp(
                     DataTypeToEnum<T>::value,
                     TensorShape({filter_shape.dim_size(4),
                                  filter_shape.dim_size(3), dims.filter_size(0),
                                  dims.filter_size(1), dims.filter_size(2)}),
                     &transformed_filter));
    functor::TransformFilter<GPUDevice, T, int, 5>()(
        context->eigen_device<GPUDevice>(), FORMAT_OIHW,
        To32Bit(filter.tensor<T, 5>()),
        To32Bit(transformed_filter.tensor<T, 5>()));

    // Shape: batch, filters, z, y, x.
    Tensor transformed_out_backprop;
    if (data_format_ == FORMAT_NHWC) {
      TensorShape nchw_shape = {dims.batch_size, dims.out_depth,
                                dims.output_size(0), dims.output_size(1),
                                dims.output_size(2)};
      if (dims.out_depth > 1) {
        OP_REQUIRES_OK(context, context->allocate_temp(
                                    DataTypeToEnum<T>::value, nchw_shape,
                                    &transformed_out_backprop));
        functor::NHWCToNCHW<GPUDevice, T, 5>()(
            context->eigen_device<GPUDevice>(), out_backprop.tensor<T, 5>(),
            transformed_out_backprop.tensor<T, 5>());
      } else {
        CHECK(transformed_out_backprop.CopyFrom(out_backprop, nchw_shape));
      }
    } else {
      transformed_out_backprop = out_backprop;
    }
    // Shape: batch, filters, z, y, x.
    Tensor pre_transformed_in_backprop;
    OP_REQUIRES_OK(
        context,
        context->allocate_temp(DataTypeToEnum<T>::value, compatible_input_shape,
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

    static int64 ConvolveBackwardDataScratchSize = GetDnnWorkspaceLimit(
        "TF_CUDNN_WORKSPACE_LIMIT_IN_MB", 1LL << 32);  // 4GB by default

    const int device_id = stream->parent()->device_ordinal();
    DataType dtype = context->input(0).dtype();
    const ConvParameters conv_parameters = {
        dims.batch_size,
        dims.in_depth,
        {{dims.input_size(0), dims.input_size(1), dims.input_size(2)}},
        FORMAT_NCHW,
        dims.out_depth,
        {{dims.filter_size(0), dims.filter_size(1), dims.filter_size(2)}},
        {{dims.dilation(0), dims.dilation(1), dims.dilation(2)}},
        {{dims.stride(0), dims.stride(1), dims.stride(2)}},
        {{padding_planes, padding_rows, padding_cols}},
        dtype,
        device_id,
        conv_desc.group_count()};

    using se::dnn::AlgorithmConfig;
    using se::dnn::AlgorithmDesc;
    using se::dnn::ProfileResult;
    AlgorithmConfig algorithm_config;
    if (cudnn_use_autotune_ && !AutoTuneConv3dBwdData::GetInstance()->Find(
                                   conv_parameters, &algorithm_config)) {
#if GOOGLE_CUDA
      se::TfAllocatorAdapter tf_allocator_adapter(
          context->device()->GetAllocator({}), stream);
      se::RedzoneAllocator rz_allocator(stream, &tf_allocator_adapter,
                                        se::GpuAsmOpts());
      se::DeviceMemory<T> in_backprop_ptr_rz(
          WrapRedzoneBestEffort(&rz_allocator, in_backprop_ptr));
      std::vector<AlgorithmDesc> algorithms;
      CHECK(stream->parent()->GetConvolveBackwardDataAlgorithms(
          conv_parameters.ShouldIncludeWinogradNonfusedAlgo<T>(
              stream->parent()),
          &algorithms));
      ProfileResult best_result;
      ProfileResult best_result_no_scratch;
      std::vector<tensorflow::AutotuneResult> results;
      for (auto profile_algorithm : algorithms) {
        // TODO(zhengxq): profile each algorithm multiple times to better
        // accuracy.
        DnnScratchAllocator scratch_allocator(ConvolveBackwardDataScratchSize,
                                              context);
        se::RedzoneAllocator rz_scratch_allocator(
            stream, &tf_allocator_adapter, se::GpuAsmOpts(),
            /*memory_limit=*/ConvolveBackwardDataScratchSize);
        se::ScratchAllocator* allocator_used =
            !RedzoneCheckDisabled()
                ? static_cast<se::ScratchAllocator*>(&rz_scratch_allocator)
                : static_cast<se::ScratchAllocator*>(&scratch_allocator);
        ProfileResult profile_result;
        bool cudnn_launch_status =
            stream
                ->ThenConvolveBackwardDataWithAlgorithm(
                    filter_desc, filter_ptr, output_desc, out_backprop_ptr,
                    conv_desc, input_desc, &in_backprop_ptr_rz, allocator_used,
                    AlgorithmConfig(profile_algorithm), &profile_result)
                .ok();
        if (cudnn_launch_status) {
          if (profile_result.is_valid()) {
            results.emplace_back();
            auto& result = results.back();
            result.mutable_conv()->set_algorithm(profile_algorithm.algo_id());
            result.mutable_conv()->set_tensor_ops_enabled(
                profile_algorithm.tensor_ops_enabled());
            result.set_scratch_bytes(
                !RedzoneCheckDisabled()
                    ? rz_scratch_allocator
                          .TotalAllocatedBytesExcludingRedzones()
                    : scratch_allocator.TotalByteSize());
            *result.mutable_run_time() = proto_utils::ToDurationProto(
                absl::Milliseconds(profile_result.elapsed_time_in_ms()));

            if (profile_result.elapsed_time_in_ms() <
                best_result.elapsed_time_in_ms()) {
              best_result = profile_result;
            }
            if (scratch_allocator.TotalByteSize() == 0 &&
                profile_result.elapsed_time_in_ms() <
                    best_result_no_scratch.elapsed_time_in_ms()) {
              best_result_no_scratch = profile_result;
            }
            // TODO(george): they don't do results at all??
            CheckRedzones(rz_scratch_allocator, &result);
            CheckRedzones(rz_allocator, &result);
          }
        }
      }
#elif TENSORFLOW_USE_ROCM
      std::vector<ProfileResult> algorithms;
      CHECK(stream->parent()->GetMIOpenConvolveAlgorithms(
          se::dnn::ConvolutionKind::BACKWARD_DATA, stream,
          se::dnn::ToDataType<T>::value, input_desc, filter_desc, conv_desc,
          output_desc, &algorithms));
      ProfileResult best_result;
      ProfileResult best_result_no_scratch;
      std::vector<tensorflow::AutotuneResult> results;
      for (auto miopen_algorithm : algorithms) {
        auto profile_algorithm = miopen_algorithm.algorithm();
        DnnScratchAllocator scratch_allocator(ConvolveBackwardDataScratchSize,
                                              context);
        ProfileResult profile_result;
        bool miopen_launch_status =
            stream
                ->ThenConvolveBackwardDataWithAlgorithm(
                    filter_desc, filter_ptr, output_desc, out_backprop_ptr,
                    conv_desc, input_desc, &in_backprop_ptr, &scratch_allocator,
                    AlgorithmConfig(profile_algorithm), &profile_result)
                .ok();
        if (miopen_launch_status) {
          if (profile_result.is_valid()) {
            results.emplace_back();
            auto& result = results.back();
            result.mutable_conv()->set_algorithm(profile_algorithm.algo_id());
            result.mutable_conv()->set_tensor_ops_enabled(
                profile_algorithm.tensor_ops_enabled());
            result.set_scratch_bytes(scratch_allocator.TotalByteSize());
            *result.mutable_run_time() = proto_utils::ToDurationProto(
                absl::Milliseconds(profile_result.elapsed_time_in_ms()));

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
#endif
      LogConvAutotuneResults(se::dnn::ConvolutionKind::BACKWARD_DATA,
                             se::dnn::ToDataType<T>::value, in_backprop_ptr,
                             filter_ptr, out_backprop_ptr, input_desc,
                             filter_desc, output_desc, conv_desc,
                             stream->parent(), results);
      OP_REQUIRES(context,
                  best_result.is_valid() || best_result_no_scratch.is_valid(),
                  errors::NotFound("No algorithm worked!"));
      if (best_result.is_valid()) {
        algorithm_config.set_algorithm(best_result.algorithm());
      }
      if (best_result_no_scratch.is_valid()) {
        algorithm_config.set_algorithm_no_scratch(
            best_result_no_scratch.algorithm());
      }
      AutoTuneConv3dBwdData::GetInstance()->Insert(conv_parameters,
                                                   algorithm_config);
    }
    DnnScratchAllocator scratch_allocator(ConvolveBackwardDataScratchSize,
                                          context);
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
    }

    if (rows_odd || cols_odd || planes_odd) {
      Tensor in_backprop_remove_padding;
      OP_REQUIRES_OK(context,
                     context->allocate_temp(
                         DataTypeToEnum<T>::value,
                         {dims.batch_size, dims.in_depth, dims.input_size(0),
                          dims.input_size(1), dims.input_size(2)},
                         &in_backprop_remove_padding));

      // Remove the padding for odd spatial dimensions.
      functor::PadInput<GPUDevice, T, int, 5>()(
          context->eigen_device<GPUDevice>(),
          To32Bit(const_cast<const Tensor&>(pre_transformed_in_backprop)
                      .tensor<T, 5>()),
          {{0, 0, 0}}, {{-planes_odd, -rows_odd, -cols_odd}},
          To32Bit(in_backprop_remove_padding.tensor<T, 5>()), FORMAT_NCHW);

      pre_transformed_in_backprop = in_backprop_remove_padding;
    }

    if (data_format_ == FORMAT_NHWC) {
      auto toConstTensor = [](const Tensor& x) -> const Tensor { return x; };
      functor::NCHWToNHWC<GPUDevice, T, 5>()(
          context->eigen_device<GPUDevice>(),
          toConstTensor(pre_transformed_in_backprop).template tensor<T, 5>(),
          in_backprop->tensor<T, 5>());
    } else {
      *in_backprop = pre_transformed_in_backprop;
    }
  }

 private:
  std::vector<int32> dilation_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
  bool takes_shape_;
  bool cudnn_use_autotune_;
};

// A dummy type to group backward filter autotune results together.
struct Conv3dBackwardFilterAutoTuneGroup {
  static string name() { return "Conv3dBwdFilter"; }
};
typedef AutoTuneSingleton<Conv3dBackwardFilterAutoTuneGroup, ConvParameters,
                          se::dnn::AlgorithmConfig>
    AutoTuneConv3dBwdFilter;

template <typename T>
class Conv3DBackpropFilterOp<GPUDevice, T> : public OpKernel {
 public:
  explicit Conv3DBackpropFilterOp(OpKernelConstruction* context)
      : OpKernel(context),
        data_format_(FORMAT_NHWC),
        takes_shape_(type_string().find("V2") != std::string::npos) {
    // data_format is only available in V2.
    if (takes_shape_) {
      string data_format;
      OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
    }
    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilation_));
    OP_REQUIRES(context, dilation_.size() == 5,
                errors::InvalidArgument("Dilation rates field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES(context,
                (GetTensorDim(dilation_, data_format_, 'C') == 1 &&
                 GetTensorDim(dilation_, data_format_, 'N') == 1),
                errors::InvalidArgument(
                    "Current implementation does not yet support "
                    "dilation rates in the batch and depth dimensions."));
    OP_REQUIRES(
        context,
        (GetTensorDim(dilation_, data_format_, '0') > 0 &&
         GetTensorDim(dilation_, data_format_, '1') > 0 &&
         GetTensorDim(dilation_, data_format_, '2') > 0),
        errors::InvalidArgument("Dilated rates should be larger than 0."));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 5,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES(
        context,
        (GetTensorDim(stride_, data_format_, 'C') == 1 &&
         GetTensorDim(stride_, data_format_, 'N') == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES(
        context,
        (GetTensorDim(stride_, data_format_, '0') > 0 &&
         GetTensorDim(stride_, data_format_, '1') > 0 &&
         GetTensorDim(stride_, data_format_, '2') > 0),
        errors::InvalidArgument("Spatial strides should be larger than 0."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    cudnn_use_autotune_ = CudnnUseAutotune();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const TensorShape& input_shape = input.shape();

    const Tensor& out_backprop = context->input(2);
    const TensorShape& out_backprop_shape = out_backprop.shape();

    TensorShape filter_shape;
    if (takes_shape_) {
      const Tensor& filter_sizes = context->input(1);
      OP_REQUIRES_OK(context, MakeShape(filter_sizes, &filter_shape));
    } else {
      filter_shape = context->input(1).shape();
    }

    ConvBackpropDimensions dims;
    OP_REQUIRES_OK(
        context,
        ConvBackpropComputeDimensionsV2(
            "Conv3DBackpropFilterOp", /*num_spatial_dims=*/3, input_shape,
            filter_shape, out_backprop_shape, dilation_, stride_, padding_,
            /*explicit_paddings=*/{}, data_format_, &dims));

    Tensor* filter_backprop;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, filter_shape, &filter_backprop));

    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

    bool is_grouped_convolution = filter_shape.dim_size(3) != dims.in_depth;
    if (!is_grouped_convolution && dims.filter_size(1) == 1 &&
        dims.filter_size(2) == 1 && dims.filter_size(0) == 1 &&
        dims.dilation(2) == 1 && dims.dilation(1) == 1 &&
        dims.dilation(0) == 1 && dims.stride(2) == 1 && dims.stride(1) == 1 &&
        dims.stride(0) == 1 && data_format_ == FORMAT_NHWC) {
      const uint64 m = dims.in_depth;
      const uint64 k = dims.batch_size * dims.input_size(1) *
                       dims.input_size(2) * dims.input_size(0);
      const uint64 n = dims.out_depth;

      // The shape of output backprop is
      //   [batch, out_z, out_y, out_x, out_depth]
      // From cublas's perspective, it is: n x k
      auto a_ptr = AsDeviceMemory(out_backprop.template flat<T>().data(),
                                  out_backprop.template flat<T>().size());

      // The shape of input is:
      //   [batch, in_z, in_y, in_x, in_depth],
      // From cublas's perspective, it is: m x k
      auto b_ptr = AsDeviceMemory(input.template flat<T>().data(),
                                  input.template flat<T>().size());

      // The shape of the filter backprop is:
      //   [1, 1, 1, in_depth, out_depth]
      // From cublas's perspective, it is: n x m
      auto c_ptr = AsDeviceMemory(filter_backprop->template flat<T>().data(),
                                  filter_backprop->template flat<T>().size());

      bool blas_launch_status =
          stream
              ->ThenBlasGemm(se::blas::Transpose::kNoTranspose,
                             se::blas::Transpose::kTranspose, n, m, k, 1.0f,
                             a_ptr, n, b_ptr, m, 0.0f, &c_ptr, n)
              .ok();
      if (!blas_launch_status) {
        context->SetStatus(errors::Internal("Blas SGEMM launch failed : m=", m,
                                            ", n=", n, ", k=", k));
      }
      return;
    } else if (!is_grouped_convolution &&
               dims.filter_size(0) == dims.input_size(0) &&
               dims.filter_size(1) == dims.input_size(1) &&
               dims.filter_size(2) == dims.input_size(2) &&
               padding_ == Padding::VALID && data_format_ == FORMAT_NHWC) {
      const uint64 m = dims.input_size(0) * dims.input_size(1) *
                       dims.input_size(2) * dims.in_depth;
      const uint64 k = dims.batch_size;
      const uint64 n = dims.out_depth;

      auto a_ptr = AsDeviceMemory(input.template flat<T>().data(),
                                  input.template flat<T>().size());
      auto b_ptr = AsDeviceMemory(out_backprop.template flat<T>().data(),
                                  out_backprop.template flat<T>().size());
      auto c_ptr = AsDeviceMemory(filter_backprop->template flat<T>().data(),
                                  filter_backprop->template flat<T>().size());

      bool blas_launch_status =
          stream
              ->ThenBlasGemm(se::blas::Transpose::kNoTranspose,
                             se::blas::Transpose::kTranspose, n, m, k, 1.0f,
                             b_ptr, n, a_ptr, m, 0.0f, &c_ptr, n)
              .ok();
      if (!blas_launch_status) {
        context->SetStatus(errors::Internal("Blas SGEMM launch failed : m=", m,
                                            ", n=", n, ", k=", k));
      }
      return;
    }

    int padding_planes = dims.SpatialPadding(padding_, 0);
    int padding_rows = dims.SpatialPadding(padding_, 1);
    int padding_cols = dims.SpatialPadding(padding_, 2);
    const bool planes_odd = (padding_planes % 2 != 0);
    const bool rows_odd = (padding_rows % 2 != 0);
    const bool cols_odd = (padding_cols % 2 != 0);

    Tensor compatible_input;
    if (rows_odd || cols_odd || planes_odd) {
      OP_REQUIRES_OK(context,
                     context->allocate_temp(
                         DataTypeToEnum<T>::value,
                         ShapeFromFormat(data_format_, dims.batch_size,
                                         {{dims.input_size(0) + planes_odd,
                                           dims.input_size(1) + rows_odd,
                                           dims.input_size(2) + cols_odd}},
                                         dims.in_depth),
                         &compatible_input));
      functor::PadInput<GPUDevice, T, int, 5>()(
          context->template eigen_device<GPUDevice>(),
          To32Bit(input.tensor<T, 5>()), {{0, 0, 0}},
          {{planes_odd, rows_odd, cols_odd}},
          To32Bit(compatible_input.tensor<T, 5>()), data_format_);
    } else {
      compatible_input = input;
    }

    CHECK(padding_rows >= 0 && padding_cols >= 0 && padding_planes >= 0)
        << "Negative paddings: (" << padding_rows << ", " << padding_cols
        << ", " << padding_planes << ")";
    se::dnn::BatchDescriptor input_desc(3);
    input_desc.set_count(dims.batch_size)
        .set_spatial_dim(DimIndex::X,
                         GetTensorDim(compatible_input, data_format_, '2'))
        .set_spatial_dim(DimIndex::Y,
                         GetTensorDim(compatible_input, data_format_, '1'))
        .set_spatial_dim(DimIndex::Z,
                         GetTensorDim(compatible_input, data_format_, '0'))
        .set_feature_map_count(dims.in_depth)
        .set_layout(se::dnn::DataLayout::kBatchDepthYX);
    se::dnn::BatchDescriptor output_desc(3);
    output_desc.set_count(dims.batch_size)
        .set_spatial_dim(DimIndex::X, dims.output_size(2))
        .set_spatial_dim(DimIndex::Y, dims.output_size(1))
        .set_spatial_dim(DimIndex::Z, dims.output_size(0))
        .set_feature_map_count(dims.out_depth)
        .set_layout(se::dnn::DataLayout::kBatchDepthYX);
    se::dnn::FilterDescriptor filter_desc(3);
    filter_desc.set_spatial_dim(DimIndex::X, dims.filter_size(2))
        .set_spatial_dim(DimIndex::Y, dims.filter_size(1))
        .set_spatial_dim(DimIndex::Z, dims.filter_size(0))
        .set_input_feature_map_count(filter_shape.dim_size(3))
        .set_output_feature_map_count(filter_shape.dim_size(4));
    se::dnn::ConvolutionDescriptor conv_desc(3);
    conv_desc.set_dilation_rate(DimIndex::X, dims.dilation(2))
        .set_dilation_rate(DimIndex::Y, dims.dilation(1))
        .set_dilation_rate(DimIndex::Z, dims.dilation(0))
        .set_filter_stride(DimIndex::X, dims.stride(2))
        .set_filter_stride(DimIndex::Y, dims.stride(1))
        .set_filter_stride(DimIndex::Z, dims.stride(0))
        .set_zero_padding(DimIndex::X, padding_cols / 2)
        .set_zero_padding(DimIndex::Y, padding_rows / 2)
        .set_zero_padding(DimIndex::Z, padding_planes / 2)
        .set_group_count(dims.in_depth / filter_shape.dim_size(3));
    Tensor pre_transformed_filter_backprop;
    OP_REQUIRES_OK(
        context, context->allocate_temp(
                     DataTypeToEnum<T>::value,
                     TensorShape({filter_shape.dim_size(4),
                                  filter_shape.dim_size(3), dims.filter_size(0),
                                  dims.filter_size(1), dims.filter_size(2)}),
                     &pre_transformed_filter_backprop));

    Tensor transformed_out_backprop;
    if (data_format_ == FORMAT_NHWC) {
      TensorShape nchw_shape = {dims.batch_size, dims.out_depth,
                                dims.output_size(0), dims.output_size(1),
                                dims.output_size(2)};
      OP_REQUIRES_OK(
          context, context->allocate_temp(DataTypeToEnum<T>::value, nchw_shape,
                                          &transformed_out_backprop));
      if (dims.out_depth > 1) {
        functor::NHWCToNCHW<GPUDevice, T, 5>()(
            context->eigen_device<GPUDevice>(), out_backprop.tensor<T, 5>(),
            transformed_out_backprop.tensor<T, 5>());
      } else {
        CHECK(transformed_out_backprop.CopyFrom(out_backprop, nchw_shape));
      }
    } else {
      transformed_out_backprop = out_backprop;
    }
    Tensor transformed_input;
    if (data_format_ == FORMAT_NHWC) {
      TensorShape nchw_shape = {
          dims.batch_size, dims.in_depth, compatible_input.dim_size(1),
          compatible_input.dim_size(2), compatible_input.dim_size(3)};
      if (dims.in_depth > 1) {
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<T>::value,
                                              nchw_shape, &transformed_input));
        functor::NHWCToNCHW<GPUDevice, T, 5>()(
            context->eigen_device<GPUDevice>(),
            const_cast<const Tensor&>(compatible_input).tensor<T, 5>(),
            transformed_input.tensor<T, 5>());
      } else {
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

    static int64 ConvolveBackwardFilterScratchSize = GetDnnWorkspaceLimit(
        "TF_CUDNN_WORKSPACE_LIMIT_IN_MB", 1LL << 32);  // 4GB by default

    const int device_id = stream->parent()->device_ordinal();
    DataType dtype = input.dtype();
    const ConvParameters conv_parameters = {
        dims.batch_size,
        dims.in_depth,
        {{dims.input_size(0), dims.input_size(1), dims.input_size(2)}},
        FORMAT_NCHW,
        dims.out_depth,
        {{dims.filter_size(0), dims.filter_size(1), dims.filter_size(2)}},
        {{dims.dilation(0), dims.dilation(1), dims.dilation(2)}},
        {{dims.stride(0), dims.stride(1), dims.stride(2)}},
        {{padding_planes, padding_rows, padding_cols}},
        dtype,
        device_id,
        conv_desc.group_count()};

    using se::dnn::AlgorithmConfig;
    using se::dnn::AlgorithmDesc;
    using se::dnn::ProfileResult;
    AlgorithmConfig algorithm_config;
    if (cudnn_use_autotune_ && !AutoTuneConv3dBwdFilter::GetInstance()->Find(
                                   conv_parameters, &algorithm_config)) {
#if GOOGLE_CUDA
      std::vector<AlgorithmDesc> algorithms;
      CHECK(stream->parent()->GetConvolveBackwardFilterAlgorithms(
          conv_parameters.ShouldIncludeWinogradNonfusedAlgo<T>(
              stream->parent()),
          &algorithms));
      ProfileResult best_result;
      ProfileResult best_result_no_scratch;
      for (auto profile_algorithm : algorithms) {
        // TODO(zhengxq): profile each algorithm multiple times to better
        // accuracy.
        DnnScratchAllocator scratch_allocator(ConvolveBackwardFilterScratchSize,
                                              context);
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
#elif TENSORFLOW_USE_ROCM
      std::vector<ProfileResult> algorithms;
      CHECK(stream->parent()->GetMIOpenConvolveAlgorithms(
          se::dnn::ConvolutionKind::BACKWARD_FILTER, stream,
          se::dnn::ToDataType<T>::value, input_desc, filter_desc, conv_desc,
          output_desc, &algorithms));
      ProfileResult best_result;
      ProfileResult best_result_no_scratch;
      if (algorithms.size() == 1) {
        best_result = algorithms[0];
      } else {
        for (auto miopen_algorithm : algorithms) {
          auto profile_algorithm = miopen_algorithm.algorithm();
          DnnScratchAllocator scratch_allocator(
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
      }
#endif
      OP_REQUIRES(context,
                  best_result.is_valid() || best_result_no_scratch.is_valid(),
                  errors::NotFound("No algorithm worked!"));
      if (best_result.is_valid()) {
        algorithm_config.set_algorithm(best_result.algorithm());
      }
      if (best_result_no_scratch.is_valid()) {
        algorithm_config.set_algorithm_no_scratch(
            best_result_no_scratch.algorithm());
      }
      AutoTuneConv3dBwdFilter::GetInstance()->Insert(conv_parameters,
                                                     algorithm_config);
    }
    DnnScratchAllocator scratch_allocator(ConvolveBackwardFilterScratchSize,
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
    }

    auto toConstTensor = [](const Tensor& x) -> const Tensor { return x; };
    functor::ReverseTransformFilter<GPUDevice, T, 5>()(
        context->eigen_device<GPUDevice>(), /*src_filter_format=*/FORMAT_OIHW,
        toConstTensor(pre_transformed_filter_backprop).template tensor<T, 5>(),
        filter_backprop->tensor<T, 5>());
  }

 private:
  std::vector<int32> dilation_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
  bool takes_shape_;
  bool cudnn_use_autotune_;
};

#define REGISTER_GPU_KERNEL(T)                                                \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("Conv3DBackpropInput").Device(DEVICE_GPU).TypeConstraint<T>("T"),  \
      Conv3DBackpropInputOp<GPUDevice, T>);                                   \
  REGISTER_KERNEL_BUILDER(Name("Conv3DBackpropInputV2")                       \
                              .Device(DEVICE_GPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .HostMemory("input_sizes"),                     \
                          Conv3DBackpropInputOp<GPUDevice, T>);               \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("Conv3DBackpropFilter").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      Conv3DBackpropFilterOp<GPUDevice, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("Conv3DBackpropFilterV2")                      \
                              .Device(DEVICE_GPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .HostMemory("filter_sizes"),                    \
                          Conv3DBackpropFilterOp<GPUDevice, T>);
TF_CALL_half(REGISTER_GPU_KERNEL);
TF_CALL_float(REGISTER_GPU_KERNEL);
TF_CALL_double(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
