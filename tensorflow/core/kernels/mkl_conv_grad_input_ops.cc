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
class MklConv2DCustomBackpropInputOp :
  public MklConv2DBackpropCommonOp<Device, T> {
 public:
  explicit MklConv2DCustomBackpropInputOp(OpKernelConstruction* context)
      : MklConv2DBackpropCommonOp<Device, T>(context) { }
  ~MklConv2DCustomBackpropInputOp() {}

 private:
  const int kInputIndex_Filter = 1,
            kInputIndex_InputSizes = 0,
            kInputIndex_OutBackProp = 2;
  void ValidateMklShapes(const MklDnnShape& input_mkl_shape,
                         const MklDnnShape& filter_mkl_shape,
                         const MklDnnShape& obp_mkl_shape) {
    // Tensor that feeds to 'Input' slot of BackpropInput is always just a shape
    // of the Tensor and never an actual tensor. So it will never be in MKL
    // layout.
    CHECK(!input_mkl_shape.IsMklTensor())
      << "Conv2DBackpropInput: input should not be in MKL Layout";
  }

  size_t GetInputTensorIndexWithSizes() { return kInputIndex_InputSizes; }

  TensorShape MakeInputTfShape(OpKernelContext* context,
                               const Tensor& input_tensor) {
    TensorShape input_tf_shape;
    CHECK_EQ(TensorShapeUtils::IsVector(input_tensor.shape()), true);
    CHECK_EQ(TensorShapeUtils::MakeShape(input_tensor.vec<int32>(),
                                         &input_tf_shape).ok(), true);
    return input_tf_shape;
  }

  TensorShape MakeFilterTfShape(OpKernelContext* context,
                                const Tensor& filter_tensor) {
    return GetTfShape(context, kInputIndex_Filter);
  }

  const memory::dims& GetOutputDims(const memory::dims& fwd_input_dims,
                                    const memory::dims& fwd_filter_dims) {
    // Output Shape of Conv2DBackpropInput is same as shape of Conv2D 'input'.
    return fwd_input_dims;
  }

  memory::format GetOutputFormat(const memory::format data_format) {
    // Output layout is Tensorflow's layout in data format order.
    return data_format;
  }

  void CreatePrimitive(OpKernelContext* context,
                       const engine& cpu_engine,
                       const convolution_forward::primitive_desc& conv_fwd_pd,
                       MklDnnData<T>* input, MklDnnData<T>* filter,
                       MklDnnData<T>* outbackprop, MklDnnData<T>* output,
                       Tensor** output_tensor,
                       const memory::dims& strides,
                       const memory::dims& padding_l,
                       const memory::dims& padding_r,
                       padding_kind padding,
                       const memory::dims& bwd_output_dims,
                       memory::format bwd_output_format) {
    CHECK_NOTNULL(context);
    CHECK_NOTNULL(input);
    CHECK_NOTNULL(filter);
    CHECK_NOTNULL(outbackprop);
    CHECK_NOTNULL(output);
    CHECK_NOTNULL(output_tensor);

    // Create convolution backward data primitive.
    auto bwd_desc = convolution_backward_data::desc(convolution_direct,
                      output->GetOpMemDesc(), filter->GetOpMemDesc(),
                      outbackprop->GetOpMemDesc(), strides, padding_l,
                      padding_r, padding);

    auto bwd_pd = convolution_backward_data::primitive_desc(bwd_desc,
                                                          cpu_engine,
                                                          conv_fwd_pd);


    // Allocate output tensor in TensorFlow and MKL layout.
    AllocateOutputTensor(context, bwd_pd, bwd_output_dims,
                         bwd_output_format, output_tensor);
    CHECK_NOTNULL(*output_tensor);
    // Set buffer handle using allocated output tensor.
    output->SetUsrMemDataHandle(*output_tensor);

    PrepareAndExecutePrimitive(bwd_pd, filter, outbackprop, output);
  }

  // Allocate output tensor.
  void AllocateOutputTensor(OpKernelContext* context,
                  const convolution_backward_data::primitive_desc& conv_pd,
                  const memory::dims& output_dims_mkl_order,
                  memory::format output_tf_format, Tensor** output_tensor) {
      CHECK_NOTNULL(output_tensor);

      // Output primitive descriptor for backward data is diff_src.
      auto dst_pd = conv_pd.diff_src_primitive_desc();

      // Allocate shape of Mkl tensor.
      MklDnnShape output_mkl_shape;
      output_mkl_shape.SetMklTensor(true);
      output_mkl_shape.SetMklLayout(&dst_pd);
      output_mkl_shape.SetElemType(MklDnnType<T>());
      output_mkl_shape.SetTfLayout(output_dims_mkl_order.size(),
                                   output_dims_mkl_order, output_tf_format);

      // Allocate shape of TF tensor.
      TensorShape output_tf_shape;
      output_tf_shape.AddDim(dst_pd.get_size() / sizeof(T));

      AllocateOutputSetMklShape(context, 0, output_tensor, output_tf_shape,
                                output_mkl_shape);
  }

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

    net.push_back(convolution_backward_data(conv_pd, obp->GetOpMem(),
                                    filter->GetOpMem(), output->GetOpMem()));

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
