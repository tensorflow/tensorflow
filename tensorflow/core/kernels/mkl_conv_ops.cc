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
#ifdef INTEL_MKL

#include <string.h>
#include <map>
#include <vector>
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

#include "third_party/mkl/include/mkl_dnn.h"
#include "third_party/mkl/include/mkl_dnn_types.h"
#include "tensorflow/core/util/mkl_util.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T, bool biasEnabled>
class MklConv2DOp : public OpKernel {
 public:
  ~MklConv2DOp() {}

  explicit MklConv2DOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));

    const int64 stride_n = GetTensorDim(strides_, data_format_, 'N');
    const int64 stride_c = GetTensorDim(strides_, data_format_, 'C');
    OP_REQUIRES(
        context, stride_n == 1 && stride_c == 1,
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = MklGetInput(context, 0);
    GetMklShape(context, 0, &(mkl_params_.input_shape));
    bool input_in_mkl_format = mkl_params_.input_shape.IsMklTensor();

    const Tensor& filter = MklGetInput(context, 1);
    MklShape mkl_filter_shape;
    GetMklShape(context, 1, &mkl_filter_shape);
    CHECK(!mkl_filter_shape.IsMklTensor())
        << "Conv filter should not be in MKL Layout";

    if (biasEnabled) {
      const Tensor& bias = MklGetInput(context, 2);
      OP_REQUIRES(context, bias.dims() == 1,
                  errors::InvalidArgument("bias must be 1-dimensional: ",
                                          bias.shape().DebugString()));
    }

    if (!input_in_mkl_format) {
      OP_REQUIRES(context, input.dims() == 4,
                  errors::InvalidArgument("input must be 4-dimensional",
                                          input.shape().DebugString()));
    }

    OP_REQUIRES(context, filter.dims() == 4,
                errors::InvalidArgument("filter must be 4-dimensional: ",
                                        filter.shape().DebugString()));

    for (int i = 0; i < 3; i++) {
      OP_REQUIRES(
          context,
          FastBoundsCheck(filter.dim_size(i), std::numeric_limits<int>::max()),
          errors::InvalidArgument("filter too large"));
    }

    const int64 input_depth = input_in_mkl_format
                                  ? mkl_params_.input_shape.GetSizes()[2]
                                  : GetTensorDim(input, data_format_, 'C');
    OP_REQUIRES(context, input_depth == filter.dim_size(2),
                errors::InvalidArgument(
                    "input and filter must have the same depth: ", input_depth,
                    " vs ", filter.dim_size(2)));
    // The last dimension for filter is out_depth.
    const int out_depth = static_cast<int>(filter.dim_size(3));

    // The second dimension for input is rows/height.
    // The first dimension for filter is rows/height.
    const int64 input_rows_raw = input_in_mkl_format
                                     ? mkl_params_.input_shape.GetSizes()[1]
                                     : GetTensorDim(input, data_format_, 'H');
    OP_REQUIRES(
        context,
        FastBoundsCheck(input_rows_raw, std::numeric_limits<int>::max()),
        errors::InvalidArgument("Input rows too large"));
    const int input_rows = static_cast<int>(input_rows_raw);
    const int filter_rows = static_cast<int>(filter.dim_size(0));

    // The third dimension for input is columns/width.
    // The second dimension for filter is columns/width.
    const int64 input_cols_raw = input_in_mkl_format
                                     ? mkl_params_.input_shape.GetSizes()[0]
                                     : GetTensorDim(input, data_format_, 'W');
    OP_REQUIRES(
        context,
        FastBoundsCheck(input_cols_raw, std::numeric_limits<int>::max()),
        errors::InvalidArgument("Input cols too large"));
    const int input_cols = static_cast<int>(input_cols_raw);
    const int filter_cols = static_cast<int>(filter.dim_size(1));

    // The first dimension for input is batch.
    const int64 input_batch_raw = input_in_mkl_format
                                      ? mkl_params_.input_shape.GetSizes()[3]
                                      : GetTensorDim(input, data_format_, 'N');
    OP_REQUIRES(
        context,
        FastBoundsCheck(input_batch_raw, std::numeric_limits<int>::max()),
        errors::InvalidArgument("batch is too large"));
    const int batch = static_cast<int>(input_batch_raw);

    // For now we take the stride from the second and third dimensions only (we
    // do not support striding on the batch or depth dimension).
    const int stride_rows = GetTensorDim(strides_, data_format_, 'H');
    const int stride_cols = GetTensorDim(strides_, data_format_, 'W');

    int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_rows, filter_rows, stride_rows,
                                         padding_, &out_rows, &pad_rows));
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_cols, filter_cols, stride_cols,
                                         padding_, &out_cols, &pad_cols));
    TensorShape out_shape =
        ShapeFromFormat(data_format_, batch, out_rows, out_cols, out_depth);

    // Output tensor is of the following dimensions:
    // [ in_batch, out_rows, out_cols, out_depth ]
    Tensor* output = nullptr;

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      // TODO(jbobba): Verify correctness here
      //               Need semantics for Null MKL tensor
      return;
    }

    if (batch == 0) {
      // Nothing to do, allocate output tensor and return
      MklShape mkl_output_mkl_shape;
      mkl_output_mkl_shape.SetMklTensor(false);
      AllocateOutputSetMklshape(context, 0, &output, input.shape(),
                                mkl_output_mkl_shape);
      return;
    }

    // Create MKL convolution primitives
    mkl_params_.in_dims = input_in_mkl_format
                              ? mkl_params_.input_shape.GetDimension()
                              : input.dims();
    mkl_params_.filter_dims = filter.dims();
    mkl_params_.in_sizes[0] = static_cast<size_t>(input_cols);
    mkl_params_.in_sizes[1] = static_cast<size_t>(input_rows);
    mkl_params_.in_sizes[2] = static_cast<size_t>(input_depth);
    mkl_params_.in_sizes[3] = static_cast<size_t>(batch);
    mkl_params_.out_sizes[0] = static_cast<size_t>(out_cols);
    mkl_params_.out_sizes[1] = static_cast<size_t>(out_rows);
    mkl_params_.out_sizes[2] = static_cast<size_t>(out_depth);
    mkl_params_.out_sizes[3] = static_cast<size_t>(batch);
    mkl_params_.input_offset[0] = static_cast<int>(-pad_cols);
    mkl_params_.input_offset[1] = static_cast<int>(-pad_rows);
    mkl_params_.conv_stride[0] = static_cast<size_t>(stride_cols);
    mkl_params_.conv_stride[1] = static_cast<size_t>(stride_rows);

    GetStridesFromSizes(data_format_, mkl_params_.out_strides,
                        mkl_params_.out_sizes);
    GetStridesFromSizes(data_format_, mkl_params_.in_strides,
                        mkl_params_.in_sizes);

    // TF filter dimension order (out_depth, in_depth, cols, rows) ->
    // MKL filter dimension order (out_depth, in_depth, rows, cols)
    mkl_params_.filter_sizes[0] = filter.dim_size(1);  // cols
    mkl_params_.filter_sizes[1] = filter.dim_size(0);  // rows
    mkl_params_.filter_sizes[2] = filter.dim_size(2);  // in_depth
    mkl_params_.filter_sizes[3] = filter.dim_size(3);  // out_depth

    // TF filter layout - (rows, cols, in_depth, out_depth)
    mkl_params_.filter_strides[0] =
        filter.dim_size(2) * filter.dim_size(3);  // cols
    mkl_params_.filter_strides[1] =
        filter.dim_size(1) * filter.dim_size(2) * filter.dim_size(3);  // rows
    mkl_params_.filter_strides[2] = filter.dim_size(3);  // in_depth
    mkl_params_.filter_strides[3] = 1;                   // out_depth

    if (biasEnabled) {
      const Tensor& bias = MklGetInput(context, 2);
      mkl_params_.bias_sizes[0] = {static_cast<size_t>(bias.dim_size(0))};
      mkl_params_.bias_strides[0] = {1};
    }

    // Create Convolution Primitive
    if (biasEnabled) {
      CHECK_EQ(dnnConvolutionCreateForwardBias_F32(
                   &mkl_prim_convolution_fwd_, nullptr,
                   dnnAlgorithmConvolutionDirect, mkl_params_.in_dims,
                   mkl_params_.in_sizes, mkl_params_.out_sizes,
                   mkl_params_.filter_sizes, mkl_params_.conv_stride,
                   mkl_params_.input_offset, dnnBorderZeros),
               E_SUCCESS);
    } else {
      CHECK_EQ(dnnConvolutionCreateForward_F32(
                   &mkl_prim_convolution_fwd_, nullptr,
                   dnnAlgorithmConvolutionDirect, mkl_params_.in_dims,
                   mkl_params_.in_sizes, mkl_params_.out_sizes,
                   mkl_params_.filter_sizes, mkl_params_.conv_stride,
                   mkl_params_.input_offset, dnnBorderZeros),
               E_SUCCESS);
    }

    TensorShape mkl_output_tf_shape;
    MklShape mkl_output_mkl_shape;
    mkl_output_mkl_shape.SetMklTensor(true);
    mkl_output_mkl_shape.SetMklLayout(mkl_prim_convolution_fwd_,
                                      dnnResourceDst);
    mkl_output_mkl_shape.SetTfLayout(mkl_params_.in_dims, mkl_params_.out_sizes,
                                     mkl_params_.out_strides);
    mkl_output_tf_shape.AddDim(
        dnnLayoutGetMemorySize_F32(
            static_cast<dnnLayout_t>(mkl_output_mkl_shape.GetMklLayout())) /
        sizeof(T));
    AllocateOutputSetMklshape(context, 0, &output, mkl_output_tf_shape,
                              mkl_output_mkl_shape);
    mkl_conv_res_[dnnResourceDst] =
        static_cast<void*>(output->flat<T>().data());

    MklCreateInputLayouts(context);

    Tensor mkl_tmp_input_buf_tensor, mkl_tmp_filter_buf_tensor,
        mkl_tmp_bias_buf_tensor;  // Temp tensor used to allocate tmp
                                  // buffers
    MklPrepareConvolutionInputs(context, &mkl_tmp_input_buf_tensor,
                                &mkl_tmp_filter_buf_tensor,
                                &mkl_tmp_bias_buf_tensor);

    // Execute convolution
    CHECK_EQ(dnnExecute_F32(mkl_prim_convolution_fwd_, mkl_conv_res_),
             E_SUCCESS);

    MklCleanup();
  }

 private:
  typedef struct {
    int in_dims;
    size_t in_sizes[4];
    size_t in_strides[4];
    size_t out_sizes[4];
    size_t out_strides[4];
    int filter_dims;
    size_t filter_sizes[4];
    size_t filter_strides[4];
    size_t bias_sizes[1];
    size_t bias_strides[1];
    int input_offset[2];
    size_t conv_stride[2];
    MklShape input_shape;
  } MklConv2DOpParams;

  // Create MKL dnnLayout_t objects for tensors coming into the layer
  void MklCreateInputLayouts(OpKernelContext* context) {
    bool input_in_mkl_format = mkl_params_.input_shape.IsMklTensor();
    if (input_in_mkl_format) {
      mkl_lt_input_ =
          static_cast<dnnLayout_t>(mkl_params_.input_shape.GetCurLayout());
    } else {
      CHECK_EQ(
          dnnLayoutCreate_F32(&mkl_lt_input_, mkl_params_.in_dims,
                              mkl_params_.in_sizes, mkl_params_.in_strides),
          E_SUCCESS);
    }

    CHECK_EQ(dnnLayoutCreate_F32(&mkl_lt_filter_, mkl_params_.filter_dims,
                                 mkl_params_.filter_sizes,
                                 mkl_params_.filter_strides),
             E_SUCCESS);

    if (biasEnabled) {
      CHECK_EQ(dnnLayoutCreate_F32(&mkl_lt_bias_, 1, mkl_params_.bias_sizes,
                                   mkl_params_.bias_strides),
               E_SUCCESS);
    }
  }

  // Compare incoming tensor layouts with MKL preferred layouts and convert
  // data to the preferred layout if necessary
  void MklPrepareConvolutionInputs(OpKernelContext* context,
                                   Tensor* mkl_tmp_input_buf_tensor,
                                   Tensor* mkl_tmp_filter_buf_tensor,
                                   Tensor* mkl_tmp_bias_buf_tensor) {
    bool mkl_convert_input, mkl_convert_filter, mkl_convert_bias;
    dnnPrimitive_t mkl_prim_convert_filter, mkl_prim_convert_bias,
        mkl_prim_convert_input;
    dnnLayout_t mkl_lt_internal_filter, mkl_lt_internal_bias,
        mkl_lt_internal_input;
    void *mkl_buf_convert_input, *mkl_buf_convert_filter, *mkl_buf_convert_bias;
    mkl_prim_convert_filter = nullptr;
    mkl_prim_convert_bias = nullptr;
    mkl_prim_convert_input = nullptr;
    mkl_lt_internal_filter = nullptr;
    mkl_lt_internal_bias = nullptr;
    mkl_lt_internal_input = nullptr;
    mkl_buf_convert_input = nullptr;
    mkl_buf_convert_filter = nullptr;
    mkl_buf_convert_bias = nullptr;

    // Compare with internal layouts and convert if needed
    const Tensor& input = MklGetInput(context, 0);
    void* mkl_buf_input =
        const_cast<void*>(static_cast<const void*>(input.flat<T>().data()));
    CHECK_EQ(
        dnnLayoutCreateFromPrimitive_F32(
            &mkl_lt_internal_input, mkl_prim_convolution_fwd_, dnnResourceSrc),
        E_SUCCESS);
    mkl_convert_input =
        !dnnLayoutCompare_F32(mkl_lt_internal_input, mkl_lt_input_);
    if (mkl_convert_input) {
      CHECK_EQ(dnnConversionCreate_F32(&mkl_prim_convert_input, mkl_lt_input_,
                                       mkl_lt_internal_input),
               E_SUCCESS);
      AllocTmpBuffer(context, mkl_tmp_input_buf_tensor, mkl_lt_internal_input,
                     &mkl_buf_convert_input);
      CHECK_EQ(dnnConversionExecute_F32(mkl_prim_convert_input, mkl_buf_input,
                                        mkl_buf_convert_input),
               E_SUCCESS);
      dnnDelete_F32(mkl_prim_convert_input);
    }
    dnnLayoutDelete_F32(mkl_lt_internal_input);

    mkl_conv_res_[dnnResourceSrc] =
        (mkl_convert_input) ? mkl_buf_convert_input : mkl_buf_input;

    const Tensor& filter = MklGetInput(context, 1);
    void* mkl_buf_filter =
        const_cast<void*>(static_cast<const void*>(filter.flat<T>().data()));
    CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(&mkl_lt_internal_filter,
                                              mkl_prim_convolution_fwd_,
                                              dnnResourceFilter),
             E_SUCCESS);
    mkl_convert_filter =
        !dnnLayoutCompare_F32(mkl_lt_internal_filter, mkl_lt_filter_);
    if (mkl_convert_filter) {
      CHECK_EQ(dnnConversionCreate_F32(&mkl_prim_convert_filter, mkl_lt_filter_,
                                       mkl_lt_internal_filter),
               E_SUCCESS);
      AllocTmpBuffer(context, mkl_tmp_filter_buf_tensor, mkl_lt_internal_filter,
                     &mkl_buf_convert_filter);
      CHECK_EQ(dnnConversionExecute_F32(mkl_prim_convert_filter, mkl_buf_filter,
                                        mkl_buf_convert_filter),
               E_SUCCESS);
      dnnDelete_F32(mkl_prim_convert_filter);
    }
    dnnLayoutDelete_F32(mkl_lt_internal_filter);

    mkl_conv_res_[dnnResourceFilter] =
        (mkl_convert_filter) ? mkl_buf_convert_filter : mkl_buf_filter;

    if (biasEnabled) {
      const Tensor& bias = MklGetInput(context, 2);
      void* mkl_buf_bias =
          const_cast<void*>(static_cast<const void*>(bias.flat<T>().data()));
      CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(&mkl_lt_internal_bias,
                                                mkl_prim_convolution_fwd_,
                                                dnnResourceBias),
               E_SUCCESS);
      mkl_convert_bias =
          !dnnLayoutCompare_F32(mkl_lt_internal_bias, mkl_lt_bias_);
      if (mkl_convert_bias) {
        CHECK_EQ(dnnConversionCreate_F32(&mkl_prim_convert_bias, mkl_lt_bias_,
                                         mkl_lt_internal_bias),
                 E_SUCCESS);
        AllocTmpBuffer(context, mkl_tmp_bias_buf_tensor, mkl_lt_internal_bias,
                       &mkl_buf_convert_bias);
        CHECK_EQ(dnnConversionExecute_F32(mkl_prim_convert_bias, mkl_buf_bias,
                                          mkl_buf_convert_bias),
                 E_SUCCESS);
        dnnDelete_F32(mkl_prim_convert_bias);
      }
      dnnLayoutDelete_F32(mkl_lt_internal_bias);

      mkl_conv_res_[dnnResourceBias] =
          (mkl_convert_bias) ? mkl_buf_convert_bias : mkl_buf_bias;
    }
  }

  void MklCleanup() {
    bool input_in_mkl_format = mkl_params_.input_shape.IsMklTensor();
    dnnDelete_F32(mkl_prim_convolution_fwd_);
    if (!input_in_mkl_format) dnnLayoutDelete_F32(mkl_lt_input_);
    dnnLayoutDelete_F32(mkl_lt_filter_);
    if (biasEnabled) dnnLayoutDelete_F32(mkl_lt_bias_);
  }

  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;

  MklConv2DOpParams mkl_params_;
  dnnPrimitive_t mkl_prim_convolution_fwd_ = nullptr;
  void* mkl_conv_res_[dnnResourceNumber];
  dnnLayout_t mkl_lt_filter_ = nullptr, mkl_lt_bias_ = nullptr,
              mkl_lt_input_ = nullptr;
};

#define REGISTER_MKL_CPU(T)                                               \
  REGISTER_KERNEL_BUILDER(Name("MklConv2D")                               \
                              .Device(DEVICE_CPU)                         \
                              .TypeConstraint<T>("T")                     \
                              .Label(mkl_layer_registry::kMklLayerLabel), \
                          MklConv2DOp<CPUDevice, T, false>);              \
  REGISTER_KERNEL_BUILDER(Name("MklConv2DWithBias")                       \
                              .Device(DEVICE_CPU)                         \
                              .TypeConstraint<T>("T")                     \
                              .Label(mkl_layer_registry::kMklLayerLabel), \
                          MklConv2DOp<CPUDevice, T, true>);

TF_CALL_float(REGISTER_MKL_CPU);

}  // namespace tensorflow
#endif  // INTEL_MKL
