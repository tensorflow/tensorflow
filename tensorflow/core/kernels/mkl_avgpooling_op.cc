/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifdef INTEL_MKL
#define EIGEN_USE_THREADS

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/mkl_util.h"

#include "tensorflow/core/kernels/mkl_pooling_ops_common.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class MklAvgPoolingOp : public OpKernel {
 public:
  explicit MklAvgPoolingOp(OpKernelConstruction* context) : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));

    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented("Pooling is not yet supported on the "
                                      "batch dimension."));
  }

  void Compute(OpKernelContext* context) override {
    MklAvgPoolingOpContext mkl_context;
    const Tensor& tensor_in = MklGetInput(context, 0);
    GetMklShape(context, 0, &mkl_context.input_shape);
    bool input_in_mkl_format = mkl_context.input_shape.IsMklTensor();

    if (!input_in_mkl_format)
      mkl_context.params.in_dim = tensor_in.dims();
    else
      mkl_context.params.in_dim = mkl_context.input_shape.GetDimension();

    MklPoolParameters pool_params;
    if (!input_in_mkl_format) {
      pool_params.Init(context, ksize_, stride_, padding_, data_format_,
                       tensor_in.shape());
    } else {
      pool_params.Init(context, ksize_, stride_, padding_, data_format_,
                       &mkl_context.input_shape);
    }

    // Extract the parameters for the op from the pooling specs
    ExtractMklOpParams(context, data_format_, pool_params, &mkl_context.params);

    Tensor mkl_tmp_input_buf_tensor_;
    mkl_context.MklCreateLayoutsAndPrimitives(context,
                                              &mkl_tmp_input_buf_tensor_);
    OP_REQUIRES_OK(context, context->status());

    Tensor workspace_tensor;
    void* workspace_buf;
    AllocTmpBuffer(context, &workspace_tensor, mkl_context.lt_workspace,
                   &workspace_buf);

    if (mkl_context.convert_input != nullptr) {
      if (input_in_mkl_format == false) {
        CHECK_EQ(
            dnnConversionExecute_F32(
                mkl_context.convert_input,
                static_cast<void*>(const_cast<T*>(tensor_in.flat<T>().data())),
                mkl_context.input_buf),
            E_SUCCESS);
        CHECK_EQ(dnnDelete_F32(mkl_context.convert_input), E_SUCCESS);
      } else {
        mkl_context.input_shape.GetConvertedFlatData(
            mkl_context.lt_prim_input,
            static_cast<void*>(const_cast<T*>(tensor_in.flat<T>().data())),
            mkl_context.input_buf);
      }
      mkl_context.pooling_res[dnnResourceSrc] = mkl_context.input_buf;
    } else {
      mkl_context.pooling_res[dnnResourceSrc] =
          static_cast<void*>(const_cast<T*>(tensor_in.flat<T>().data()));
    }

    // Declare output tensor and allocate memory
    Tensor* output = nullptr;
    TensorShape tensor_out_shape;
    MklShape mkl_out_shape;
    mkl_out_shape.SetMklTensor(true);
    mkl_out_shape.SetMklLayout(mkl_context.prim_pooling_fwd, dnnResourceDst);
    mkl_out_shape.SetTfLayout(mkl_context.params.in_dim,
                              mkl_context.params.out_sizes,
                              mkl_context.params.out_strides);
    mkl_out_shape.SetTfDimOrder(mkl_context.params.in_dim, data_format_);

    tensor_out_shape.AddDim(dnnLayoutGetMemorySize_F32(static_cast<dnnLayout_t>(
                                mkl_out_shape.GetMklLayout())) /
                            sizeof(T));

    AllocateOutputSetMklShape(context, 0, &output, tensor_out_shape,
                              mkl_out_shape);
    mkl_context.pooling_res[dnnResourceDst] =
        static_cast<void*>(output->flat<T>().data());

    mkl_context.pooling_res[dnnResourceWorkspace] = workspace_buf;

    CHECK_EQ(
        dnnExecute_F32(mkl_context.prim_pooling_fwd, mkl_context.pooling_res),
        E_SUCCESS);

    mkl_context.MklCleanup();
  }

 private:
  typedef struct {
    MklPoolingOpParams params;
    MklShape input_shape;
    dnnPrimitive_t prim_pooling_fwd = nullptr, convert_input = nullptr;
    dnnLayout_t lt_user_input = nullptr, lt_prim_input = nullptr,
                lt_workspace = nullptr;
    void* input_buf = nullptr;
    void* pooling_res[dnnResourceNumber];

    void MklCreateLayoutsAndPrimitives(OpKernelContext* context,
                                       Tensor* mkl_tmp_input_buf_tensor) {
      bool input_in_mkl_format = input_shape.IsMklTensor();

      if (!input_in_mkl_format) {
        CHECK_EQ(dnnLayoutCreate_F32(&lt_user_input, params.in_dim,
                                     params.in_sizes, params.in_strides),
                 E_SUCCESS);
      } else {
        lt_user_input = (dnnLayout_t)input_shape.GetCurLayout();
      }

      dnnAlgorithm_t algorithm = dnnAlgorithmPoolingAvg;
      dnnPrimitiveAttributes_t primAttr = nullptr;

      // Create DNN primitives
      CHECK_EQ(dnnPoolingCreateForward_F32(
                   &prim_pooling_fwd, primAttr, algorithm, lt_user_input,
                   params.kernel_size, params.kernel_stride, params.in_offset,
                   dnnBorderZerosAsymm),
               E_SUCCESS);

      CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(
                   &lt_prim_input, prim_pooling_fwd, dnnResourceSrc),
               E_SUCCESS);
      if (!dnnLayoutCompare_F32(lt_user_input, lt_prim_input)) {
        CHECK_EQ(dnnConversionCreate_F32(&convert_input, lt_user_input,
                                         lt_prim_input),
                 E_SUCCESS);

        AllocTmpBuffer(context, mkl_tmp_input_buf_tensor, lt_prim_input,
                       &input_buf);
      }

      CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(&lt_workspace, prim_pooling_fwd,
                                                dnnResourceWorkspace),
               E_SUCCESS);
    }

    void MklCleanup() {
      bool input_in_mkl_format = input_shape.IsMklTensor();
      if (!input_in_mkl_format) {
        CHECK_EQ(dnnLayoutDelete_F32(lt_user_input), E_SUCCESS);
      }

      CHECK_EQ(dnnDelete_F32(prim_pooling_fwd), E_SUCCESS);
      CHECK_EQ(dnnLayoutDelete_F32(lt_prim_input), E_SUCCESS);
    }
  } MklAvgPoolingOpContext;

  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};

//-----------------------------------------------------------------------------

template <class Device, class T>
class MklAvgPoolingGradOp : public OpKernel {
 public:
  explicit MklAvgPoolingGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;

    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
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
                errors::Unimplemented("Pooling is not yet supported on the "
                                      "batch dimension."));
  }

  void Compute(OpKernelContext* context) override {
    MklAvgPoolingGradOpContext mkl_context;
    const Tensor& tensor_in_shape = MklGetInput(context, 0);
    const Tensor& out_backprop = MklGetInput(context, 1);
    GetMklShape(context, 1, &mkl_context.out_backprop_shape);
    bool outbackprop_in_mkl_format =
        mkl_context.out_backprop_shape.IsMklTensor();

    TensorShape output_shape;
    auto shape_vec = tensor_in_shape.vec<int32>();
    for (int64 i = 0; i < tensor_in_shape.NumElements(); ++i) {
      output_shape.AddDim(shape_vec(i));
    }

    MklPoolParameters pool_params;
    pool_params.Init(context, ksize_, stride_, padding_, data_format_,
                     output_shape);

    if (outbackprop_in_mkl_format == false)
      mkl_context.params.in_dim = out_backprop.dims();
    else
      mkl_context.params.in_dim = mkl_context.out_backprop_shape.GetDimension();

    // Extract the parameters for the op from the pooling specs
    ExtractMklOpParams(context, data_format_, pool_params, &mkl_context.params);

    // Tensors needed to create temporary buffers
    Tensor outbackprop_buf_tensor;
    void* outbackprop_buf;
    mkl_context.MklCreateLayoutsAndPrimitives(context);
    OP_REQUIRES_OK(context, context->status());

    // Check if outbackprop layout requires conversion.
    if (!dnnLayoutCompare_F32(mkl_context.lt_user_outbackprop,
                              mkl_context.lt_prim_outbackprop)) {
      CHECK_EQ(dnnConversionCreate_F32(&mkl_context.convert_outbackprop,
                                       mkl_context.lt_user_outbackprop,
                                       mkl_context.lt_prim_outbackprop),
               E_SUCCESS);

      AllocTmpBuffer(context, &outbackprop_buf_tensor,
                     mkl_context.lt_prim_outbackprop, &outbackprop_buf);

      if (!outbackprop_in_mkl_format) {
        CHECK_EQ(dnnConversionExecute_F32(mkl_context.convert_outbackprop,
                                          static_cast<void*>(const_cast<T*>(
                                              out_backprop.flat<T>().data())),
                                          outbackprop_buf),
                 E_SUCCESS);
        CHECK_EQ(dnnDelete_F32(mkl_context.convert_outbackprop), E_SUCCESS);
      } else {
        mkl_context.out_backprop_shape.GetConvertedFlatData(
            mkl_context.lt_prim_outbackprop,
            static_cast<void*>(const_cast<T*>(out_backprop.flat<T>().data())),
            outbackprop_buf);
      }
      mkl_context.pooling_res[dnnResourceDiffDst] = outbackprop_buf;
    } else {
      mkl_context.pooling_res[dnnResourceDiffDst] =
          static_cast<void*>(const_cast<T*>(out_backprop.flat<T>().data()));
    }

    // Handle workspace requirements.
    Tensor workspace_buf_tensor;
    void* workspace_buf;
    AllocTmpBuffer(context, &workspace_buf_tensor, mkl_context.lt_workspace,
                   &workspace_buf);
    mkl_context.pooling_res[dnnResourceWorkspace] = workspace_buf;

    // Handle MKL output tensor setup.
    Tensor* output = nullptr;
    TensorShape tensor_out_shape;
    MklShape mkl_out_shape;
    mkl_out_shape.SetMklTensor(true);
    mkl_out_shape.SetMklLayout(mkl_context.prim_pooling_bwd,
                               dnnResourceDiffSrc);
    mkl_out_shape.SetTfLayout(mkl_context.params.in_dim,
                              mkl_context.params.in_sizes,
                              mkl_context.params.in_strides);
    mkl_out_shape.SetTfDimOrder(mkl_context.params.in_dim, data_format_);

    tensor_out_shape.AddDim(dnnLayoutGetMemorySize_F32(static_cast<dnnLayout_t>(
                                mkl_out_shape.GetMklLayout())) /
                            sizeof(T));

    AllocateOutputSetMklShape(context, 0, &output, tensor_out_shape,
                              mkl_out_shape);

    // Set output tensor.
    mkl_context.pooling_res[dnnResourceDiffSrc] =
        static_cast<void*>(output->flat<T>().data());

    // Execute primitive.
    CHECK_EQ(
        dnnExecute_F32(mkl_context.prim_pooling_bwd, mkl_context.pooling_res),
        E_SUCCESS);

    mkl_context.MklCleanup();
  }

 private:
  typedef struct {
    MklPoolingOpParams params;
    MklShape out_backprop_shape;
    dnnPrimitive_t prim_pooling_bwd = nullptr, convert_outbackprop = nullptr;
    void* pooling_res[dnnResourceNumber];
    dnnLayout_t lt_user_input = nullptr, lt_user_outbackprop = nullptr,
                lt_prim_outbackprop = nullptr, lt_workspace = nullptr;

    void MklCreateLayoutsAndPrimitives(OpKernelContext* context) {
      const Tensor& tensor_in_shape = MklGetInput(context, 0);
      const Tensor& out_backprop = MklGetInput(context, 1);
      bool outbackprop_in_mkl_format = out_backprop_shape.IsMklTensor();

      if (!outbackprop_in_mkl_format) {
        // For avgpooling, tensor_in_shape should have 1 dimension, and 4
        // elements.
        OP_REQUIRES(context, tensor_in_shape.dims() == 1 &&
                                 tensor_in_shape.NumElements() == 4,
                    errors::InvalidArgument("original input shape must be "
                                            "1-dimensional and 4 elements"));

        // For avgpooling, out_backprop should have 4 dimensions.
        OP_REQUIRES(context, out_backprop.dims() == 4,
                    errors::InvalidArgument("out_backprop must be "
                                            "4-dimensional"));
      } else {
        // Input in MKL format.
        // For avgpooling, out_backprop should have 4 dimensions.
        OP_REQUIRES(context, out_backprop_shape.GetDimension() == 4,
                    errors::InvalidArgument("out_backprop must be "
                                            "4-dimensional"));
      }

      // TODO(inteltf): Get outbackprop layout.
      // Do we need to create layout in every invocation?
      if (!outbackprop_in_mkl_format) {
        CHECK_EQ(dnnLayoutCreate_F32(&lt_user_outbackprop, params.in_dim,
                                     params.out_sizes, params.out_strides),
                 E_SUCCESS);
      } else {
        lt_user_outbackprop = (dnnLayout_t)out_backprop_shape.GetCurLayout();
      }

      // Create the backward primitive
      // Create DNN user layout
      CHECK_EQ(dnnLayoutCreate_F32(&lt_user_input, params.in_dim,
                                   params.in_sizes, params.in_strides),
               E_SUCCESS);

      // Create PoolingBackward primitive
      dnnAlgorithm_t algorithm = dnnAlgorithmPoolingAvg;
      dnnPrimitiveAttributes_t primAttr = nullptr;
      CHECK_EQ(dnnPoolingCreateBackward_F32(
                   &prim_pooling_bwd, primAttr, algorithm, lt_user_input,
                   params.kernel_size, params.kernel_stride, params.in_offset,
                   dnnBorderZerosAsymm),
               E_SUCCESS);

      // Create expected outbackprop layout from the primitive.
      CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(
                   &lt_prim_outbackprop, prim_pooling_bwd, dnnResourceDiffDst),
               E_SUCCESS);

      CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(&lt_workspace, prim_pooling_bwd,
                                                dnnResourceWorkspace),
               E_SUCCESS);
    }

    void MklCleanup() {
      bool outbackprop_in_mkl_format = out_backprop_shape.IsMklTensor();
      CHECK_EQ(dnnDelete_F32(prim_pooling_bwd), E_SUCCESS);
      CHECK_EQ(dnnLayoutDelete_F32(lt_user_input), E_SUCCESS);
      if (!outbackprop_in_mkl_format) {
        CHECK_EQ(dnnLayoutDelete_F32(lt_user_outbackprop), E_SUCCESS);
      }
      CHECK_EQ(dnnLayoutDelete_F32(lt_prim_outbackprop), E_SUCCESS);
      CHECK_EQ(dnnLayoutDelete_F32(lt_workspace), E_SUCCESS);
    }
  } MklAvgPoolingGradOpContext;

  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};

REGISTER_KERNEL_BUILDER(Name("_MklAvgPool")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T")
                            .Label(mkl_op_registry::kMklOpLabel),
                        MklAvgPoolingOp<CPUDevice, float>);

REGISTER_KERNEL_BUILDER(Name("_MklAvgPoolGrad")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T")
                            .Label(mkl_op_registry::kMklOpLabel),
                        MklAvgPoolingGradOp<CPUDevice, float>);

}  // namespace tensorflow
#endif  // INTEL_MKL
