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

// See docs in ../ops/nn_ops.cc. This opkernel uses MKL library, create MKL
// layout and primitives, use MKL dnn primitives to compute convolution backward
// input

#ifdef INTEL_MKL

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS
#include <algorithm>
#include <vector>
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/conv_grad_ops.h"
#include "tensorflow/core/kernels/mkl_conv_ops.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"
#include "tensorflow/core/util/work_sharder.h"
#include "mkl_dnn.h"
#include "mkl_dnn_types.h"

#ifdef INTEL_MKL_DNN
#include "mkldnn.hpp"

using mkldnn::stream;
using mkldnn::prop_kind;

using mkldnn::convolution_forward;
using mkldnn::convolution_direct;
using mkldnn::convolution_backward_data;
#endif

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

#ifndef INTEL_MKL_DNN

template <typename Device, class T>
class MklConv2DCustomBackpropInputOp : public OpKernel {
 public:
  ~MklConv2DCustomBackpropInputOp() {}
  explicit MklConv2DCustomBackpropInputOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string dataformat;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &dataformat));
    OP_REQUIRES(context, FormatFromString(dataformat, &data_format),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides));
    int stride_n = GetTensorDim(strides, data_format, 'N');
    int stride_c = GetTensorDim(strides, data_format, 'C');
    OP_REQUIRES(
        context, (stride_n == 1 && stride_c == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));

    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding));
  }

  void Compute(OpKernelContext* context) override {
    MklConvBackInputOpContext mkl_context;
    const Tensor& input = MklGetInput(context, 0);
    const Tensor& filter = MklGetInput(context, 1);

    GetMklShape(context, 1, &(mkl_context.filter_shape));
    bool filter_in_mkl_format = mkl_context.filter_shape.IsMklTensor();

    const Tensor& out_backprop = MklGetInput(context, 2);
    GetMklShape(context, 2, &(mkl_context.outback_shape));
    bool outback_in_mkl_format = mkl_context.outback_shape.IsMklTensor();

    TensorShape input_shape, filter_shape, outback_shape;

    // Generate input shape.
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(input.shape()),
        errors::InvalidArgument(
            "Conv2DBackpropInput: input_sizes input must be 1-dim, not ",
            input.dims()));
    OP_REQUIRES_OK(
        context, TensorShapeUtils::MakeShape(input.vec<int32>(), &input_shape));

    // Generate shape for filter prop if input is in MKL format.
    if (filter_in_mkl_format) {
      OP_REQUIRES(context, mkl_context.filter_shape.GetDimension() == 4,
                  errors::InvalidArgument(
                      "Conv2DCustomBackpropInput: size must be 4-dim"));

      const int64* filter_sizes =
          (const int64*)mkl_context.filter_shape.GetSizes();
      const int64 filter_dims = mkl_context.filter_shape.GetDimension();

      OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                  filter_sizes, filter_dims, &filter_shape));
    } else {
      filter_shape = filter.shape();
    }

    // Generate shape for outback prop if input is in MKL format.
    if (outback_in_mkl_format) {
      OP_REQUIRES(context, mkl_context.outback_shape.GetDimension() == 4,
                  errors::InvalidArgument(
                      "Conv2DCustomBackpropInput: size must be 4-dim"));

      MklSizesToTFSizes(context, data_format, mkl_context.outback_shape,
                        &outback_shape);
    } else {
      outback_shape = out_backprop.shape();
    }

    ConvBackpropDimensions dims;
    OP_REQUIRES_OK(
        context,
        ConvBackpropComputeDimensions(
            "Conv2DCustomBackpropInput", /*num_spatial_dims=*/2, input_shape,
            filter_shape, outback_shape, strides, padding, data_format, &dims));

    int64 pad_top, pad_bottom;
    int64 pad_left, pad_right;
    OP_REQUIRES_OK(
        context,
        GetWindowedOutputSizeVerbose(
            dims.spatial_dims[0].input_size, dims.spatial_dims[0].filter_size,
            dims.spatial_dims[0].stride, padding,
            &dims.spatial_dims[0].output_size, &pad_top, &pad_bottom));
    OP_REQUIRES_OK(
        context,
        GetWindowedOutputSizeVerbose(
            dims.spatial_dims[1].input_size, dims.spatial_dims[1].filter_size,
            dims.spatial_dims[1].stride, padding,
            &dims.spatial_dims[1].output_size, &pad_left, &pad_right));

    mkl_context.in_dims = 4;

    mkl_context.in_sizes[0] =
        static_cast<size_t>(dims.spatial_dims[1].input_size);
    mkl_context.in_sizes[1] =
        static_cast<size_t>(dims.spatial_dims[0].input_size);
    mkl_context.in_sizes[2] = static_cast<size_t>(dims.in_depth);
    mkl_context.in_sizes[3] = static_cast<size_t>(dims.batch_size);

    mkl_context.out_sizes[0] =
        static_cast<size_t>(dims.spatial_dims[1].output_size);
    mkl_context.out_sizes[1] =
        static_cast<size_t>(dims.spatial_dims[0].output_size);
    mkl_context.out_sizes[2] = static_cast<size_t>(dims.out_depth);
    mkl_context.out_sizes[3] = static_cast<size_t>(dims.batch_size);

    mkl_context.input_offset[0] = static_cast<int>(-pad_left);
    mkl_context.input_offset[1] = static_cast<int>(-pad_top);

    mkl_context.conv_strides[0] =
        static_cast<size_t>(dims.spatial_dims[1].stride);
    mkl_context.conv_strides[1] =
        static_cast<size_t>(dims.spatial_dims[0].stride);

    GetStridesFromSizes(data_format, mkl_context.out_strides,
                        mkl_context.out_sizes);
    GetStridesFromSizes(data_format, mkl_context.in_strides,
                        mkl_context.in_sizes);

    mkl_context.filter_size[0] = dims.spatial_dims[1].filter_size;
    mkl_context.filter_size[1] = dims.spatial_dims[0].filter_size;
    mkl_context.filter_size[2] = dims.in_depth;
    mkl_context.filter_size[3] = dims.out_depth;

    mkl_context.filter_stride[0] =
        mkl_context.filter_size[2] * mkl_context.filter_size[3];
    mkl_context.filter_stride[1] = mkl_context.filter_size[2] *
                                   mkl_context.filter_size[0] *
                                   mkl_context.filter_size[3];
    mkl_context.filter_stride[2] = mkl_context.filter_size[3];
    mkl_context.filter_stride[3] = 1;

    CHECK_EQ(
        dnnConvolutionCreateBackwardData_F32(
            &mkl_context.prim_bwddata, NULL, dnnAlgorithmConvolutionDirect,
            mkl_context.in_dims, mkl_context.in_sizes, mkl_context.out_sizes,
            mkl_context.filter_size, mkl_context.conv_strides,
            mkl_context.input_offset, dnnBorderZeros),
        E_SUCCESS);

    // Allocate output tensor and shape
    TensorShape mkl_out_shape;
    MklShape mklOutputShape;
    mklOutputShape.SetMklTensor(true);
    mklOutputShape.SetMklLayout(mkl_context.prim_bwddata, dnnResourceDiffSrc);
    mklOutputShape.SetTfLayout(mkl_context.in_dims, mkl_context.in_sizes,
                               mkl_context.in_strides);
    // MKL might change the dimension ordering.
    // Create mapping to recover the original TF dimension order
    mklOutputShape.SetTfDimOrder(mkl_context.in_dims, data_format);

    Tensor* in_backprop = nullptr;
    mkl_out_shape.AddDim(dnnLayoutGetMemorySize_F32(static_cast<dnnLayout_t>(
                             mklOutputShape.GetMklLayout())) /
                         sizeof(T));
    AllocateOutputSetMklShape(context, 0, &in_backprop, mkl_out_shape,
                              mklOutputShape);

    mkl_context.conv_res[dnnResourceDiffSrc] =
        static_cast<void*>(const_cast<T*>(in_backprop->flat<T>().data()));

    mkl_context.MklCreateInputLayouts(context);
    Tensor mkl_tmp_outbackprop_buf_tensor, mkl_tmp_filter_buf_tensor;
    mkl_context.MklPrepareConvolutionInputs(
        context, &mkl_tmp_outbackprop_buf_tensor, &mkl_tmp_filter_buf_tensor);

    CHECK_EQ(dnnExecute_F32(mkl_context.prim_bwddata, mkl_context.conv_res),
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
    int input_offset[2];
    size_t filter_size[4];
    size_t filter_stride[4];
    size_t conv_strides[2];
    MklShape filter_shape, outback_shape;
    dnnPrimitive_t prim_bwddata;
    void* conv_res[dnnResourceNumber];
    dnnLayout_t lt_filter, lt_outbackprop;

    // Create MKL dnnLayout_t objects for tensors coming into the layer
    void MklCreateInputLayouts(OpKernelContext* context) {
      bool filter_in_mkl_format = filter_shape.IsMklTensor();
      bool outback_in_mkl_format = outback_shape.IsMklTensor();
      if (filter_in_mkl_format) {
        lt_filter = (dnnLayout_t)filter_shape.GetCurLayout();
      } else {
        CHECK_EQ(dnnLayoutCreate_F32(&lt_filter, in_dims, filter_size,
                                     filter_stride),
                 E_SUCCESS);
      }

      if (outback_in_mkl_format) {
        lt_outbackprop = (dnnLayout_t)outback_shape.GetCurLayout();
      } else {
        CHECK_EQ(dnnLayoutCreate_F32(&lt_outbackprop, in_dims, out_sizes,
                                     out_strides),
                 E_SUCCESS);
      }
    }

    // Compare incoming input tensor layouts with MKL preferred layouts and
    // convert data to the preferred layout if necessary
    void MklPrepareConvolutionInputs(OpKernelContext* context,
                                     Tensor* mkl_tmp_outbackprop_buf_tensor,
                                     Tensor* mkl_tmp_filter_buf_tensor) {
      dnnPrimitive_t mkl_convert_filter = nullptr,
                     mkl_convert_outbackprop = nullptr;
      void *mkl_filter_buf = nullptr, *mkl_outbackprop_buf = nullptr;
      dnnLayout_t mkl_lt_filter_internal = nullptr,
                  mkl_lt_outbackprop_internal = nullptr;
      CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(
                   &mkl_lt_filter_internal, prim_bwddata, dnnResourceFilter),
               E_SUCCESS);

      const Tensor& filter = MklGetInput(context, 1);

      CHECK_EQ(
          dnnLayoutCreateFromPrimitive_F32(&mkl_lt_outbackprop_internal,
                                           prim_bwddata, dnnResourceDiffDst),
          E_SUCCESS);
      if (!dnnLayoutCompare_F32(mkl_lt_filter_internal, lt_filter)) {
        // Create conversion primitive
        CHECK_EQ(dnnConversionCreate_F32(&mkl_convert_filter, lt_filter,
                                         mkl_lt_filter_internal),
                 E_SUCCESS);

        AllocTmpBuffer(context, mkl_tmp_filter_buf_tensor,
                       mkl_lt_filter_internal, &mkl_filter_buf);
        CHECK_EQ(
            dnnConversionExecute_F32(
                mkl_convert_filter,
                static_cast<void*>(const_cast<T*>(filter.flat<T>().data())),
                mkl_filter_buf),
            E_SUCCESS);

        // Assign filter buf to resources[] for convolution.
        conv_res[dnnResourceFilter] = mkl_filter_buf;
        dnnDelete_F32(mkl_convert_filter);
      } else {
        // If we do not need any layout conversion for filter, then
        // we directly assign input filter to resources[].
        conv_res[dnnResourceFilter] =
            static_cast<void*>(const_cast<T*>(filter.flat<T>().data()));
      }
      dnnLayoutDelete_F32(mkl_lt_filter_internal);
      const Tensor& out_backprop = MklGetInput(context, 2);
      // --
      // We do similar steps as above for outputbackprop.
      if (!dnnLayoutCompare_F32(mkl_lt_outbackprop_internal, lt_outbackprop)) {
        CHECK_EQ(
            dnnConversionCreate_F32(&mkl_convert_outbackprop, lt_outbackprop,
                                    mkl_lt_outbackprop_internal),
            E_SUCCESS);
        AllocTmpBuffer(context, mkl_tmp_outbackprop_buf_tensor,
                       mkl_lt_outbackprop_internal, &mkl_outbackprop_buf);

        CHECK_EQ(dnnConversionExecute_F32(mkl_convert_outbackprop,
                                          static_cast<void*>(const_cast<T*>(
                                              out_backprop.flat<T>().data())),
                                          mkl_outbackprop_buf),
                 E_SUCCESS);

        conv_res[dnnResourceDiffDst] = mkl_outbackprop_buf;
        dnnDelete_F32(mkl_convert_outbackprop);
      } else {
        conv_res[dnnResourceDiffDst] =
            static_cast<void*>(const_cast<T*>(out_backprop.flat<T>().data()));
      }
      dnnLayoutDelete_F32(mkl_lt_outbackprop_internal);
    }

    // Cleanup member layouts and primitives
    void MklCleanup() {
      bool filter_in_mkl_format = filter_shape.IsMklTensor();
      bool outback_in_mkl_format = outback_shape.IsMklTensor();
      if (!filter_in_mkl_format) dnnLayoutDelete_F32(lt_filter);
      if (!outback_in_mkl_format) dnnLayoutDelete_F32(lt_outbackprop);
      dnnDelete_F32(prim_bwddata);
    }
  } MklConvBackInputOpContext;

  std::vector<int32> strides;
  Padding padding;
  TensorFormat data_format;
};

#else

template <typename Device, class T>
class MklConv2DCustomBackpropInputOp : public OpKernel {
 public:
  ~MklConv2DCustomBackpropInputOp() {}
  explicit MklConv2DCustomBackpropInputOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format_str;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format_str));
    OP_REQUIRES(context, FormatFromString(data_format_str, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    int stride_n = GetTensorDim(strides_, data_format_, 'N');
    int stride_c = GetTensorDim(strides_, data_format_, 'C');
    OP_REQUIRES(
        context, (stride_n == 1 && stride_c == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));

    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    try {
      auto cpu_engine = engine(engine::cpu, 0);

      MklDnnData<T> filter(&cpu_engine);
      MklDnnData<T> outbackprop(&cpu_engine);
      MklDnnData<T> output(&cpu_engine);

      // Input tensors
      const Tensor& input_tensor = MklGetInput(context, 0);
      const Tensor& filter_tensor = MklGetInput(context, 1);
      const Tensor& obp_tensor = MklGetInput(context, 2);  // Outbackprop

      // Generate input shape.
      TensorShape input_shape;
      OP_REQUIRES(context, TensorShapeUtils::IsVector(input_tensor.shape()),
        errors::InvalidArgument(
              "Conv2DBackpropInput: input_sizes input must be 1-dim, not ",
              input_tensor.dims()));
      OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                        input_tensor.vec<int32>(), &input_shape));
      TensorShape filter_shape = filter_tensor.shape();
      TensorShape obp_shape = obp_tensor.shape();

      // By default, all dims are in MKL order. Only dims in TF order
      // are those with prefix tf_order.
      memory::dims obp_dims, fwd_input_dims, fwd_filter_dims;
      memory::dims padding_l, padding_r, strides, fwd_output_dims;
      memory::dims fwd_output_dims_tf_order;

      // Get forward convolution parameters.
      MklDnnConvUtil conv_utl(context, strides_, padding_, data_format_);
      conv_utl.GetConvFwdSizesInMklOrder(input_shape, filter_shape,
                                         &fwd_input_dims, &fwd_filter_dims,
                                         &strides,
                                         &fwd_output_dims_tf_order,
                                         &fwd_output_dims,
                                         &padding_l, &padding_r);
      if (!context->status().ok()) return;

      // Create Convolution forward descriptor since Convolution backward
      // API needs it. For that, we first need to create input, filter
      // and output memory descriptors.
      auto mkl_data_format = TFDataFormatToMklDnnDataFormat(data_format_);
      auto fwd_src_md = memory::desc(fwd_input_dims, MklDnnType<T>(),
                                     mkl_data_format);
      auto fwd_filter_md = memory::desc(fwd_filter_dims, MklDnnType<T>(),
                                        memory::format::hwio);
      auto fwd_out_md = memory::desc(fwd_output_dims, MklDnnType<T>(),
                                     mkl_data_format);
      auto fwd_desc = convolution_forward::desc(prop_kind::forward,
            convolution_direct, fwd_src_md, fwd_filter_md, fwd_out_md,
            strides, padding_l, padding_r, TFPaddingToMklDnnPadding(padding_));
      auto fwd_pd = convolution_forward::primitive_desc(fwd_desc, cpu_engine);

      // Allocate output tensor and shape
      // TODO(nhasabni): Update this when support for MKL layout is added.
      // Shape of output of Conv2DBackpropInput is same as 'input' of Conv2D.
      TensorShape tf_output_shape(input_shape);
      MklShape mkl_output_mkl_shape;
      mkl_output_mkl_shape.SetMklTensor(false);
      Tensor* output_tensor = nullptr;
      AllocateOutputSetMklShape(context, 0, &output_tensor, tf_output_shape,
                                mkl_output_mkl_shape);

      // Create memory for user data.
      // Describe how the inputs and outputs of Convolution look like. Also
      // specify buffers containing actual input and output data.
      // Although input shape required is in MKL-DNN order, the layout is
      // Tensorflow's layout (NHWC or NCHW depending on data format).
      // Although filter shape (filter_dims) required is in MKL-DNN order,
      // the layout is Tensorflow's layout (HWIO).
      // Shape of Conv2DBackpropInput's filter is same as that of Conv2D filter.
      filter.SetUsrMem(fwd_filter_dims, memory::format::hwio, &filter_tensor);
      // Outbackprop shape is NHWC or NCHW depending on data format. Since
      // GetInputSizeInMklOrder function returns size in that order we just use
      // use that function directly.
      conv_utl.GetInputSizeInMklOrder(obp_shape, &obp_dims);
      if (!context->status().ok()) return;
      outbackprop.SetUsrMem(obp_dims, mkl_data_format, &obp_tensor);
      // Although output shape required is in MKL-DNN order,
      // layout is Tensorflow's layout (NHWC or NCHW depending on data format).
      // Shape of output of Conv2DBackpropInput is same as shape of 'input'
      // of Conv2D.
      memory::dims bwd_output_dims = fwd_input_dims;
      output.SetUsrMem(bwd_output_dims, mkl_data_format, output_tensor);

      // Create memory descriptors for convolution data w/ no specified format.
      filter.SetOpMemDesc(fwd_filter_dims, memory::format::any);
      outbackprop.SetOpMemDesc(obp_dims, memory::format::any);
      output.SetOpMemDesc(bwd_output_dims, memory::format::any);

      // Create convolution backward data primitive.
      auto bwd_desc = convolution_backward_data::desc(convolution_direct,
                          output.GetOpMemDesc(), filter.GetOpMemDesc(),
                          outbackprop.GetOpMemDesc(), strides, padding_l,
                          padding_r, TFPaddingToMklDnnPadding(padding_));

      auto bwd_pd = convolution_backward_data::primitive_desc(bwd_desc,
                                                              cpu_engine,
                                                              fwd_pd);

      PrepareAndExecutePrimitive(bwd_pd, &filter, &outbackprop, &output);
    } catch (mkldnn::error &e) {
     string error_msg = "Status: " + std::to_string(e.status) +
                       ", message: " + string(e.message) +
                       ", in file " + string(__FILE__) + ":" +
                       std::to_string(__LINE__);
     OP_REQUIRES_OK(context, errors::Aborted("Operation received an exception:",
                                            error_msg));
    }
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;

  // Prepare and execute net - checks for input and output reorders.
  void PrepareAndExecutePrimitive(
                  const convolution_backward_data::primitive_desc& conv_pd,
                  MklDnnData<T>* filter, MklDnnData<T>* obp,
                  MklDnnData<T>* output) {
    // Create reorders between user layout and MKL layout if it is needed and
    // add it to the net before convolution.
    std::vector<primitive> net;
    filter->CheckReorderToOpMem(conv_pd.weights_primitive_desc(), &net);
    obp->CheckReorderToOpMem(conv_pd.diff_dst_primitive_desc(), &net);

    // Memory for output of convolution. Since we may need reorder on the
    // output side, we will prepare reorder primitive in case output
    // reorder to user memory is required.
    bool output_reorder_required = output->PrepareReorderToUserMemIfReq(
                                      conv_pd.diff_src_primitive_desc());

    net.push_back(convolution_backward_data(conv_pd, obp->GetOpMem(),
                                    filter->GetOpMem(), output->GetOpMem()));

    // Insert reorder primitive in the net for output reorder if reorder is
    // required.
    if (output_reorder_required) {
      output->InsertReorderToUserMem(&net);
    }

    // Handle output reorder
    stream(stream::kind::eager).submit(net).wait();
  }
};

#endif  // INTEL_MKL_DNN

#define REGISTER_MKL_CPU_KERNELS(T)                                 \
  REGISTER_KERNEL_BUILDER(Name("_MklConv2DBackpropInput")           \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<T>("T")               \
                              .Label(mkl_op_registry::kMklOpLabel), \
                          MklConv2DCustomBackpropInputOp<CPUDevice, T>);

TF_CALL_float(REGISTER_MKL_CPU_KERNELS);
#undef REGISTER_MKL_CPU_KERNELS

}  // namespace tensorflow
#endif  // INTEL_MKL
