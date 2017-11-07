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
#include <string>

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/mkl_conv_ops.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

#include "tensorflow/core/util/mkl_util.h"
#include "mkl_dnn.h"
#include "mkl_dnn_types.h"

#ifdef INTEL_MKL_DNN
#include "mkldnn.hpp"

using mkldnn::stream;
using mkldnn::prop_kind;

using mkldnn::convolution_forward;
using mkldnn::convolution_direct;
#endif

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

// For now, MKL-ML is default. So making MKL-DNN not a default choice.
#ifndef INTEL_MKL_DNN

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
    MklConv2DOpContext mkl_context;
    const Tensor& input = MklGetInput(context, 0);
    GetMklShape(context, 0, &(mkl_context.input_shape));
    bool input_in_mkl_format = mkl_context.input_shape.IsMklTensor();

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
      OP_REQUIRES(context, FastBoundsCheck(filter.dim_size(i),
                                           std::numeric_limits<int>::max()),
                  errors::InvalidArgument("filter too large"));
    }

    const int64 input_depth =
        input_in_mkl_format ? GetMklTensorDim(mkl_context.input_shape, 'C')
                            : GetTensorDim(input, data_format_, 'C');
    OP_REQUIRES(
        context, input_depth == filter.dim_size(2),
        errors::InvalidArgument("input and filter must have the same depth: ",
                                input_depth, " vs ", filter.dim_size(2)));
    // The last dimension for filter is out_depth.
    const int out_depth = static_cast<int>(filter.dim_size(3));

    // The second dimension for input is rows/height.
    // The first dimension for filter is rows/height.
    const int64 input_rows_raw =
        input_in_mkl_format ? GetMklTensorDim(mkl_context.input_shape, 'H')
                            : GetTensorDim(input, data_format_, 'H');
    OP_REQUIRES(context, FastBoundsCheck(input_rows_raw,
                                         std::numeric_limits<int>::max()),
                errors::InvalidArgument("Input rows too large"));
    const int input_rows = static_cast<int>(input_rows_raw);
    const int filter_rows = static_cast<int>(filter.dim_size(0));

    // The third dimension for input is columns/width.
    // The second dimension for filter is columns/width.
    const int64 input_cols_raw =
        input_in_mkl_format ? GetMklTensorDim(mkl_context.input_shape, 'W')
                            : GetTensorDim(input, data_format_, 'W');
    OP_REQUIRES(context, FastBoundsCheck(input_cols_raw,
                                         std::numeric_limits<int>::max()),
                errors::InvalidArgument("Input cols too large"));
    const int input_cols = static_cast<int>(input_cols_raw);
    const int filter_cols = static_cast<int>(filter.dim_size(1));

    // The first dimension for input is batch.
    const int64 input_batch_raw =
        input_in_mkl_format ? GetMklTensorDim(mkl_context.input_shape, 'N')
                            : GetTensorDim(input, data_format_, 'N');
    OP_REQUIRES(context, FastBoundsCheck(input_batch_raw,
                                         std::numeric_limits<int>::max()),
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
      // Nothing to do, allocate output tensor and return
      MklShape mkl_output_mkl_shape;
      mkl_output_mkl_shape.SetMklTensor(false);
      AllocateOutputSetMklShape(context, 0, &output, input.shape(),
                                mkl_output_mkl_shape);
      return;
    }

    if (batch == 0) {
      // Nothing to do, allocate output tensor and return
      MklShape mkl_output_mkl_shape;
      mkl_output_mkl_shape.SetMklTensor(false);
      AllocateOutputSetMklShape(context, 0, &output, input.shape(),
                                mkl_output_mkl_shape);
      return;
    }

    // Create MKL convolution primitives
    mkl_context.in_dims = input_in_mkl_format
                              ? mkl_context.input_shape.GetDimension()
                              : input.dims();
    mkl_context.filter_dims = filter.dims();

    mkl_context.in_sizes[MklDims::W] = static_cast<size_t>(input_cols);
    mkl_context.in_sizes[MklDims::H] = static_cast<size_t>(input_rows);
    mkl_context.in_sizes[MklDims::C] = static_cast<size_t>(input_depth);
    mkl_context.in_sizes[MklDims::N] = static_cast<size_t>(batch);

    mkl_context.out_sizes[MklDims::W] = static_cast<size_t>(out_cols);
    mkl_context.out_sizes[MklDims::H] = static_cast<size_t>(out_rows);
    mkl_context.out_sizes[MklDims::C] = static_cast<size_t>(out_depth);
    mkl_context.out_sizes[MklDims::N] = static_cast<size_t>(batch);

    mkl_context.input_offset[0] = static_cast<int>(-pad_cols);
    mkl_context.input_offset[1] = static_cast<int>(-pad_rows);

    mkl_context.conv_stride[0] = static_cast<size_t>(stride_cols);
    mkl_context.conv_stride[1] = static_cast<size_t>(stride_rows);

    GetStridesFromSizes(data_format_, mkl_context.out_strides,
                        mkl_context.out_sizes);
    GetStridesFromSizes(data_format_, mkl_context.in_strides,
                        mkl_context.in_sizes);

    // TF filter dimension order (out_depth, in_depth, cols, rows) ->
    // MKL filter dimension order (out_depth, in_depth, rows, cols)
    mkl_context.filter_sizes[0] = filter.dim_size(1);  // cols
    mkl_context.filter_sizes[1] = filter.dim_size(0);  // rows
    mkl_context.filter_sizes[2] = filter.dim_size(2);  // in_depth
    mkl_context.filter_sizes[3] = filter.dim_size(3);  // out_depth

    // TF filter layout - (rows, cols, in_depth, out_depth)
    mkl_context.filter_strides[0] =
        filter.dim_size(2) * filter.dim_size(3);  // cols
    mkl_context.filter_strides[1] =
        filter.dim_size(1) * filter.dim_size(2) * filter.dim_size(3);  // rows
    mkl_context.filter_strides[2] = filter.dim_size(3);  // in_depth
    mkl_context.filter_strides[3] = 1;                   // out_depth

    if (biasEnabled) {
      const Tensor& bias = MklGetInput(context, 2);
      mkl_context.bias_sizes[0] = {static_cast<size_t>(bias.dim_size(0))};
      mkl_context.bias_strides[0] = {1};
    }

    // Create Convolution Primitive
    if (biasEnabled) {
      CHECK_EQ(
          dnnConvolutionCreateForwardBias_F32(
              &mkl_context.prim_fwd, nullptr, dnnAlgorithmConvolutionDirect,
              mkl_context.in_dims, mkl_context.in_sizes, mkl_context.out_sizes,
              mkl_context.filter_sizes, mkl_context.conv_stride,
              mkl_context.input_offset, dnnBorderZeros),
          E_SUCCESS);
    } else {
      CHECK_EQ(
          dnnConvolutionCreateForward_F32(
              &mkl_context.prim_fwd, nullptr, dnnAlgorithmConvolutionDirect,
              mkl_context.in_dims, mkl_context.in_sizes, mkl_context.out_sizes,
              mkl_context.filter_sizes, mkl_context.conv_stride,
              mkl_context.input_offset, dnnBorderZeros),
          E_SUCCESS);
    }

    TensorShape mkl_output_tf_shape;
    MklShape mkl_output_mkl_shape;
    mkl_output_mkl_shape.SetMklTensor(true);
    mkl_output_mkl_shape.SetMklLayout(mkl_context.prim_fwd, dnnResourceDst);
    mkl_output_mkl_shape.SetTfLayout(mkl_context.in_dims, mkl_context.out_sizes,
                                     mkl_context.out_strides);
    // MKL might change the dimension ordering
    // Create mapping to recover the original TF dimension order
    mkl_output_mkl_shape.SetTfDimOrder(mkl_context.in_dims, data_format_);

    mkl_output_tf_shape.AddDim(
        dnnLayoutGetMemorySize_F32(
            static_cast<dnnLayout_t>(mkl_output_mkl_shape.GetMklLayout())) /
        sizeof(T));
    AllocateOutputSetMklShape(context, 0, &output, mkl_output_tf_shape,
                              mkl_output_mkl_shape);
    // Filter output to be used in the backprop_input
    TensorShape mkl_filter_output_tf_shape;
    MklShape mkl_filter_output_mkl_shape;
    mkl_filter_output_mkl_shape.SetMklTensor(true);
    mkl_filter_output_mkl_shape.SetMklLayout(mkl_context.prim_fwd,
                                             dnnResourceFilter);

    size_t filter_sizes[4] = {static_cast<size_t>(filter.dim_size(0)),
                              static_cast<size_t>(filter.dim_size(1)),
                              static_cast<size_t>(filter.dim_size(2)),
                              static_cast<size_t>(filter.dim_size(3))};
    mkl_filter_output_mkl_shape.SetTfLayout(filter.dims(), filter_sizes,
                                            mkl_context.filter_strides);

    mkl_filter_output_mkl_shape.SetTfDimOrder(mkl_context.filter_dims,
                                              data_format_);
    mkl_filter_output_tf_shape.AddDim(
        dnnLayoutGetMemorySize_F32(static_cast<dnnLayout_t>(
            mkl_filter_output_mkl_shape.GetMklLayout())) /
        sizeof(T));
    AllocateOutputSetMklShape(context, 1, &mkl_context.output_filter,
                              mkl_filter_output_tf_shape,
                              mkl_filter_output_mkl_shape);

    mkl_context.conv_res[dnnResourceDst] =
        static_cast<void*>(output->flat<T>().data());

    mkl_context.MklCreateInputLayouts(context);

    // Temp tensor used to allocate tmp buffers
    Tensor mkl_tmp_input_buf_tensor, mkl_tmp_filter_buf_tensor,
        mkl_tmp_bias_buf_tensor;
    mkl_context.MklPrepareConvolutionInputs(context,
                                            &mkl_tmp_input_buf_tensor,
                                            &mkl_tmp_filter_buf_tensor,
                                            &mkl_tmp_bias_buf_tensor);

    // Execute convolution
    CHECK_EQ(dnnExecute_F32(mkl_context.prim_fwd, mkl_context.conv_res),
             E_SUCCESS);

    mkl_context.MklCleanup();
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
    dnnPrimitive_t prim_fwd;
    void* conv_res[dnnResourceNumber];
    dnnLayout_t lt_filter, lt_bias, lt_input;
    Tensor* output_filter = nullptr;

    // Create MKL dnnLayout_t objects for tensors coming into the layer
    void MklCreateInputLayouts(OpKernelContext* context) {
      bool input_in_mkl_format = input_shape.IsMklTensor();
      if (input_in_mkl_format) {
        lt_input = static_cast<dnnLayout_t>(input_shape.GetCurLayout());
      } else {
        CHECK_EQ(dnnLayoutCreate_F32(&lt_input, in_dims, in_sizes, in_strides),
                 E_SUCCESS);
      }

      CHECK_EQ(dnnLayoutCreate_F32(&lt_filter, filter_dims, filter_sizes,
                                   filter_strides),
               E_SUCCESS);

      if (biasEnabled) {
        CHECK_EQ(dnnLayoutCreate_F32(&lt_bias, 1, bias_sizes, bias_strides),
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
      void *mkl_buf_convert_input, *mkl_buf_convert_filter,
          *mkl_buf_convert_bias;
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
      CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(&mkl_lt_internal_input,
                                                prim_fwd, dnnResourceSrc),
               E_SUCCESS);
      mkl_convert_input =
          !dnnLayoutCompare_F32(mkl_lt_internal_input, lt_input);
      if (mkl_convert_input) {
        CHECK_EQ(dnnConversionCreate_F32(&mkl_prim_convert_input,
                 lt_input, mkl_lt_internal_input), E_SUCCESS);
        AllocTmpBuffer(context, mkl_tmp_input_buf_tensor, mkl_lt_internal_input,
                       &mkl_buf_convert_input);
        CHECK_EQ(dnnConversionExecute_F32(mkl_prim_convert_input, mkl_buf_input,
                                          mkl_buf_convert_input),
                 E_SUCCESS);
        dnnDelete_F32(mkl_prim_convert_input);
      }
      dnnLayoutDelete_F32(mkl_lt_internal_input);

      conv_res[dnnResourceSrc] =
          (mkl_convert_input) ? mkl_buf_convert_input : mkl_buf_input;

      const Tensor& filter = MklGetInput(context, 1);
      void* mkl_buf_filter =
          const_cast<void*>(static_cast<const void*>(filter.flat<T>().data()));
      CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(&mkl_lt_internal_filter,
                                                prim_fwd, dnnResourceFilter),
               E_SUCCESS);
      mkl_convert_filter =
          !dnnLayoutCompare_F32(mkl_lt_internal_filter, lt_filter);
      if (mkl_convert_filter) {
        CHECK_EQ(dnnConversionCreate_F32(&mkl_prim_convert_filter, lt_filter,
                                         mkl_lt_internal_filter),
                 E_SUCCESS);

        mkl_buf_convert_filter = const_cast<void*>(
            static_cast<const void*>(output_filter->flat<T>().data()));

        CHECK_EQ(
            dnnConversionExecute_F32(mkl_prim_convert_filter, mkl_buf_filter,
                                     mkl_buf_convert_filter),
            E_SUCCESS);
        dnnDelete_F32(mkl_prim_convert_filter);
      }
      dnnLayoutDelete_F32(mkl_lt_internal_filter);

      conv_res[dnnResourceFilter] =
          (mkl_convert_filter) ? mkl_buf_convert_filter : mkl_buf_filter;

      if (biasEnabled) {
        const Tensor& bias = MklGetInput(context, 2);
        void* mkl_buf_bias =
            const_cast<void*>(static_cast<const void*>(bias.flat<T>().data()));
        CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(&mkl_lt_internal_bias,
                                                  prim_fwd, dnnResourceBias),
                 E_SUCCESS);
        mkl_convert_bias = !dnnLayoutCompare_F32(mkl_lt_internal_bias, lt_bias);
        if (mkl_convert_bias) {
          CHECK_EQ(dnnConversionCreate_F32(&mkl_prim_convert_bias, lt_bias,
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

        conv_res[dnnResourceBias] =
            (mkl_convert_bias) ? mkl_buf_convert_bias : mkl_buf_bias;
      }
    }

    void MklCleanup() {
      bool input_in_mkl_format = input_shape.IsMklTensor();
      dnnDelete_F32(prim_fwd);
      if (!input_in_mkl_format) dnnLayoutDelete_F32(lt_input);
      dnnLayoutDelete_F32(lt_filter);
      if (biasEnabled) dnnLayoutDelete_F32(lt_bias);
    }
  } MklConv2DOpContext;

  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;
};

#else

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
    try {
      auto cpu_engine = engine(engine::cpu, 0);

      // Input tensors
      size_t src_idx = 0, filter_idx = 1;
      const Tensor& src_tensor = MklGetInput(context, src_idx);
      const Tensor& filter_tensor = MklGetInput(context, filter_idx);

      MklDnnData<T> src(&cpu_engine);
      MklDnnData<T> filter(&cpu_engine);
      MklDnnData<T> output(&cpu_engine);

      memory::dims src_dims, filter_dims, padding_l, padding_r, strides;
      memory::dims output_dims_tf_order, output_dims_mkl_order;

      // Get shapes of input tensors in MKL-DNN order
      MklDnnConvUtil conv_utl(context, strides_, padding_, data_format_);
      conv_utl.GetConvFwdSizesInMklOrder(src_tensor.shape(),
                                         filter_tensor.shape(),
                                         &src_dims, &filter_dims, &strides,
                                         &output_dims_tf_order,
                                         &output_dims_mkl_order, &padding_l,
                                         &padding_r);
      if (!context->status().ok()) return;

      // Check for corner case - if there is nothing to compute, return.
      TensorShape tf_output_shape({output_dims_tf_order[0],
                                output_dims_tf_order[1],
                                output_dims_tf_order[2],
                                output_dims_tf_order[3]});
      Tensor* output_tensor = nullptr;
      MklShape mkl_output_mkl_shape;
      mkl_output_mkl_shape.SetMklTensor(false);
      AllocateOutputSetMklShape(context, 0, &output_tensor, tf_output_shape,
                                mkl_output_mkl_shape);

      // Forward filter in TF format from input at index 1 to output at index 1.
      ForwardTfTensorInToOut(context, 1, 1);

      if (tf_output_shape.num_elements() == 0) {
        // TODO(jbobba): Verify correctness here
        //               Need semantics for Null MKL tensor
        return;
      }

      // Corner case to handle 0 batch size.
      if (output_dims_tf_order[0] == 0) {
        // Nothing to do, allocate output tensor and return
        // TODO(nhasabni): remove this code later once serialization
        // in MKL-DNN is supported.
        AllocateOutputSetMklShape(context, 0, &output_tensor,
                                  src_tensor.shape(), mkl_output_mkl_shape);
        return;
      } else {
        // Otherwise regular output tensor allocation
        // Allocate output tensor.
      }
      CHECK_NOTNULL(output_tensor);

      // Create memory for user data.
      // Describe how the inputs and outputs of Convolution look like. Also
      // specify buffers containing actual input and output data.
      // Although input shape (src_dims) required is in MKL-DNN order,
      // the layout is Tensorflow's layout (NHWC or NCHW depending on data
      // format).
      src.SetUsrMem(src_dims, TFDataFormatToMklDnnDataFormat(data_format_),
                    const_cast<void*>(static_cast<const void*>(
                    src_tensor.flat<T>().data())));
      // Although filter shape (filter_dims) required is in MKL-DNN order,
      // the layout is Tensorflow's layout (HWIO).
      filter.SetUsrMem(filter_dims, memory::format::hwio,
                       const_cast<void*>(static_cast<const void*>(
                       filter_tensor.flat<T>().data())));
      // Although output shape (output_dims) required is in MKL-DNN order,
      // layout is Tensorflow's layout (NHWC or NCHW depending on data format).
      output.SetUsrMem(output_dims_mkl_order,
                       TFDataFormatToMklDnnDataFormat(data_format_),
                       output_tensor->flat<T>().data());

      // Create memory descriptors for convolution data w/ no specified format.
      src.SetOpMemDesc(src_dims, memory::format::any);
      filter.SetOpMemDesc(filter_dims, memory::format::any);
      output.SetOpMemDesc(output_dims_mkl_order, memory::format::any);

      // If bias is enabled, then do the same steps as above for bias.
      if (biasEnabled) {
        MklDnnData<T> bias(&cpu_engine);
        memory::dims bias_size;
        conv_utl.GetBiasSizeInMklOrder(2 /* bias idx */, &bias_size);
        const Tensor& bias_tensor = MklGetInput(context, 2);
        bias.SetUsrMem(bias_size, memory::format::x,
                       const_cast<void*>(static_cast<const void*>(
                       bias_tensor.flat<T>().data())));
        bias.SetOpMemDesc(bias_size, memory::format::any);

        // Create convolution primitive with Bias.
        auto conv_desc = convolution_forward::desc(prop_kind::forward,
            convolution_direct, src.GetOpMemDesc(), filter.GetOpMemDesc(),
            bias.GetOpMemDesc(), output.GetOpMemDesc(), strides,
            padding_l, padding_r, TFPaddingToMklDnnPadding(padding_));

        auto conv_prim_desc = convolution_forward::primitive_desc(conv_desc,
                                                                cpu_engine);
        PrepareAndExecuteNet(conv_prim_desc, &src, &filter, &bias, &output);
      } else {
        // Create convolution primitive without Bias.
        auto conv_desc = convolution_forward::desc(prop_kind::forward,
            convolution_direct, src.GetOpMemDesc(), filter.GetOpMemDesc(),
            output.GetOpMemDesc(), strides, padding_l, padding_r,
            TFPaddingToMklDnnPadding(padding_));

        auto conv_prim_desc = convolution_forward::primitive_desc(conv_desc,
                                                                cpu_engine);
        PrepareAndExecuteNet(conv_prim_desc, &src, &filter, nullptr, &output);
      }
    } catch (mkldnn::error &e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                       ", message: " + std::string(e.message) +
                       ", in file " + std::string(__FILE__) + ":" +
                       std::to_string(__LINE__);
      OP_REQUIRES_OK(context,
        errors::Aborted("Operation received an exception:", error_msg));
    }
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;

  // Prepare and execute net - checks for input and output reorders.
  void PrepareAndExecuteNet(
                  const convolution_forward::primitive_desc& conv_prim_desc,
                  MklDnnData<T>* src, MklDnnData<T>* filter,
                  MklDnnData<T>* bias, MklDnnData<T>* output) {
    // Create reorders between user layout and MKL layout if it is needed and
    // add it to the net before convolution.
    std::vector<primitive> net;
    src->CheckReorderToOpMem(conv_prim_desc.src_primitive_desc(), &net);
    filter->CheckReorderToOpMem(conv_prim_desc.weights_primitive_desc(), &net);

    // Memory for output of convolution. Since we may need reorder on the
    // output side, we will prepare reorder primitive in case output
    // reorder to user memory is required.
    bool output_reorder_required = output->PrepareReorderToUserMemIfReq(
                                      conv_prim_desc.dst_primitive_desc());

    // Create convolution primitive and add it to net.
    if (bias) {
      CHECK_EQ(biasEnabled, true);
      net.push_back(convolution_forward(conv_prim_desc, src->GetOpMem(),
                                    filter->GetOpMem(), bias->GetOpMem(),
                                    output->GetOpMem()));
    } else {
      CHECK_EQ(biasEnabled, false);
      net.push_back(convolution_forward(conv_prim_desc, src->GetOpMem(),
                                    filter->GetOpMem(), output->GetOpMem()));
    }

    // Insert reorder primitive in the net for output reorder if reorder is
    // required.
    if (output_reorder_required) {
      output->InsertReorderToUserMem(&net);
    }

    // Handle output reorder
    stream(stream::kind::eager).submit(net).wait();
  }
};

#endif

#define REGISTER_MKL_CPU(T)                                         \
  REGISTER_KERNEL_BUILDER(Name("_MklConv2D")                        \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<T>("T")               \
                              .Label(mkl_op_registry::kMklOpLabel), \
                          MklConv2DOp<CPUDevice, T, false>);        \
  REGISTER_KERNEL_BUILDER(Name("_MklConv2DWithBias")                \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<T>("T")               \
                              .Label(mkl_op_registry::kMklOpLabel), \
                          MklConv2DOp<CPUDevice, T, true>);

TF_CALL_float(REGISTER_MKL_CPU);

}  // namespace tensorflow
#endif  // INTEL_MKL
