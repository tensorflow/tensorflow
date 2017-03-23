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

#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/register_types.h"

#include "tensorflow/core/kernels/mkl_pooling_ops_common.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class MklAvgPoolingOp : public UnaryOp<T> {
 public:
  explicit MklAvgPoolingOp(OpKernelConstruction* context)
    : UnaryOp<T>(context) {
    pooling_fwd_ = nullptr;
    lt_user_input_fwd_ = nullptr;
    lt_input_prim_ = nullptr;
    convert_input_ = nullptr;

    input_buf_ = nullptr;
    workspace_ = nullptr;

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
    const Tensor& tensor_in = MklGetInput(context, 0);

    GetMklShape(context, 0, &mkl_input_shape_);
    bool input_in_mkl_format = mkl_input_shape_.IsMklTensor();

    if (!input_in_mkl_format)
      mkl_params_.in_dim = tensor_in.dims();
    else
      mkl_params_.in_dim = mkl_input_shape_.GetDimension();

    MklPoolParameters params;
    if (!input_in_mkl_format) {
      params.Init(context, ksize_, stride_, padding_, data_format_,
                  tensor_in.shape());
    } else {
      params.Init(context, ksize_, stride_, padding_, data_format_,
                  &mkl_input_shape_);
    }

    // Extract the parameters for the op from the pooling specs
    ExtractMklOpParams(context, data_format_, params, &mkl_params_);

    MklCreateLayoutsAndPrimitives(context);

    AllocTmpBuffer(context, &workspace_tensor_, lt_workspace_, &workspace_);

    if (convert_input_ != nullptr) {
      if (input_in_mkl_format == false) {
        CHECK_EQ(dnnConversionExecute_F32(convert_input_,
                                          static_cast<void*>(
                                            const_cast<T*>(
                                              tensor_in.flat<T>().data())),
                                          input_buf_),
                 E_SUCCESS);
        CHECK_EQ(dnnDelete_F32(convert_input_), E_SUCCESS);
      } else {
        mkl_input_shape_.GetConvertedFlatData(lt_input_prim_,
                                              static_cast<void*>(
                                                const_cast<T*>(
                                                  tensor_in.flat<T>().data())),
                                              input_buf_);
      }
      pooling_res_[dnnResourceSrc] = input_buf_;
    } else {
      pooling_res_[dnnResourceSrc] =
        static_cast<void*>(const_cast<T*>(tensor_in.flat<T>().data()));
    }

    // Declare output tensor and allocate memory
    Tensor* output = nullptr;
    TensorShape tensor_out_shape;
    MklShape mkl_out_shape;
    mkl_out_shape.SetMklTensor(true);
    mkl_out_shape.SetMklLayout(pooling_fwd_, dnnResourceDst);
    mkl_out_shape.SetTfLayout(mkl_params_.in_dim,
                              mkl_params_.out_sizes,
                              mkl_params_.out_strides);

    tensor_out_shape.AddDim(
      dnnLayoutGetMemorySize_F32(
        static_cast<dnnLayout_t>(mkl_out_shape.GetMklLayout())) / sizeof(T));

    AllocateOutputSetMklshape(context,
                              0,
                              &output,
                              tensor_out_shape,
                              mkl_out_shape);
    pooling_res_[dnnResourceDst] =
      static_cast<void*>(output->flat<T>().data());

    pooling_res_[dnnResourceWorkspace] = workspace_;

    CHECK_EQ(dnnExecute_F32(pooling_fwd_, pooling_res_), E_SUCCESS);

    MklCleanup();
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
  MklShape mkl_input_shape_;

  dnnPrimitive_t pooling_fwd_;
  dnnPrimitive_t convert_input_;
  dnnLayout_t lt_user_input_fwd_;
  dnnLayout_t lt_input_prim_;
  dnnLayout_t lt_workspace_;

  void* workspace_;
  void* input_buf_;
  void* pooling_res_[dnnResourceNumber];

  // Tensors needed to create temporary buffers
  Tensor input_buf_tensor_;
  Tensor workspace_tensor_;

  MklPoolingOpParams mkl_params_;

  void MklCreateLayoutsAndPrimitives(OpKernelContext* context) {
    bool input_in_mkl_format = mkl_input_shape_.IsMklTensor();

    if (!input_in_mkl_format) {
      CHECK_EQ(dnnLayoutCreate_F32(&lt_user_input_fwd_,
                                   mkl_params_.in_dim,
                                   mkl_params_.in_sizes,
                                   mkl_params_.in_strides),
               E_SUCCESS);
    } else {
      lt_user_input_fwd_ = (dnnLayout_t) mkl_input_shape_.GetCurLayout();
    }

    dnnAlgorithm_t algorithm = dnnAlgorithmPoolingAvg;
    dnnPrimitiveAttributes_t primAttr = nullptr;

    // Create DNN primitives
    CHECK_EQ(dnnPoolingCreateForward_F32(&pooling_fwd_,
                                         primAttr,
                                         algorithm,
                                         lt_user_input_fwd_,
                                         mkl_params_.kernel_size,
                                         mkl_params_.kernel_stride,
                                         mkl_params_.in_offset,
                                         dnnBorderZerosAsymm),
             E_SUCCESS);

    CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(&lt_input_prim_,
                                              pooling_fwd_,
                                              dnnResourceSrc),
             E_SUCCESS);
    if (!dnnLayoutCompare_F32(lt_user_input_fwd_, lt_input_prim_)) {
      CHECK_EQ(dnnConversionCreate_F32(&convert_input_,
                                       lt_user_input_fwd_,
                                       lt_input_prim_),
               E_SUCCESS);

      AllocTmpBuffer(context,
                     &input_buf_tensor_,
                     lt_input_prim_,
                     &input_buf_);
    }

    CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(&lt_workspace_,
                                              pooling_fwd_,
                                              dnnResourceWorkspace),
             E_SUCCESS);
  }

  void MklCleanup() {
    bool input_in_mkl_format = mkl_input_shape_.IsMklTensor();
    if (!input_in_mkl_format) {
      CHECK_EQ(dnnLayoutDelete_F32(lt_user_input_fwd_), E_SUCCESS);
      lt_user_input_fwd_ = nullptr;
    }

    CHECK_EQ(dnnDelete_F32(pooling_fwd_), E_SUCCESS);
    pooling_fwd_ = nullptr;

    CHECK_EQ(dnnLayoutDelete_F32(lt_input_prim_), E_SUCCESS);
    lt_input_prim_ = nullptr;
  }
};

//-----------------------------------------------------------------------------

template <class Device, class T>
class MklAvgPoolingGradOp : public OpKernel {
 public:
  explicit MklAvgPoolingGradOp(OpKernelConstruction* context)
    : OpKernel(context) {
    string data_format;

    pooling_bwd_ = nullptr;
    convert_outbackprop_ = nullptr;
    lt_user_input_bwd_ = nullptr;
    lt_outbackprop_user_ = nullptr;
    lt_outbackprop_prim_ = nullptr;
    lt_workspace_prim_ = nullptr;

    outbackprop_buf_ = nullptr;
    workspace_ = nullptr;

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
    mkl_params_.in_dim = 4;
  }

  void Compute(OpKernelContext* context) override {
    const Tensor &out_backprop = MklGetInput(context, 1);
    GetMklShape(context, 2, &mkl_out_backprop_shape);
    outbackprop_in_mkl_format_ = mkl_out_backprop_shape.IsMklTensor();

    MklCreateLayoutsAndPrimitives(context);

    // Check if outbackprop layout requires conversion.
    if (!dnnLayoutCompare_F32(lt_outbackprop_user_, lt_outbackprop_prim_)) {
      CHECK_EQ(dnnConversionCreate_F32(&convert_outbackprop_,
                                       lt_outbackprop_user_,
                                       lt_outbackprop_prim_),
               E_SUCCESS);

      AllocTmpBuffer(context,
                     &outbackprop_buf_tensor,
                     lt_outbackprop_prim_,
                     &outbackprop_buf_);

      if (!outbackprop_in_mkl_format_) {
        CHECK_EQ(dnnConversionExecute_F32(convert_outbackprop_,
                                          static_cast<void*>(const_cast<T*>(
                                            out_backprop.flat<T>().data())),
                                          outbackprop_buf_),
                 E_SUCCESS);
        CHECK_EQ(dnnDelete_F32(convert_outbackprop_), E_SUCCESS);
      } else {
        mkl_out_backprop_shape.
          GetConvertedFlatData(lt_outbackprop_prim_,
                               static_cast<void*>(const_cast<T*>(
                                 out_backprop.flat<T>().data())),
                               outbackprop_buf_);
      }
      pooling_res_[dnnResourceDiffDst] = outbackprop_buf_;
    } else {
      pooling_res_[dnnResourceDiffDst] =
        static_cast<void*>(const_cast<T*>(out_backprop.flat<T>().data()));
    }

    // Handle workspace requirements.
    AllocTmpBuffer(context,
                   &workspace_buf_tensor,
                   lt_workspace_prim_,
                   &workspace_);
    pooling_res_[dnnResourceWorkspace] = workspace_;

    // Handle MKL output tensor setup.
    Tensor* output = nullptr;
    TensorShape tensor_out_shape;
    MklShape mkl_out_shape;
    mkl_out_shape.SetMklTensor(true);
    mkl_out_shape.SetMklLayout(pooling_bwd_, dnnResourceDiffSrc);
    mkl_out_shape.SetTfLayout(mkl_params_.in_dim,
                              mkl_params_.in_sizes,
                              mkl_params_.in_strides);

    tensor_out_shape.AddDim(dnnLayoutGetMemorySize_F32(
                              static_cast<dnnLayout_t>(
                                mkl_out_shape.GetMklLayout())) / sizeof(T));

    AllocateOutputSetMklshape(context,
                              0,
                              &output,
                              tensor_out_shape,
                              mkl_out_shape);

    // Set output tensor.
    pooling_res_[dnnResourceDiffSrc] =
      static_cast<void*>(output->flat<T>().data());

    // Execute primitive.
    CHECK_EQ(dnnExecute_F32(pooling_bwd_, pooling_res_), E_SUCCESS);

    MklCleanup();
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;

  bool outbackprop_in_mkl_format_;

  MklShape mkl_out_backprop_shape;

  // Tensors needed to create temporary buffers
  Tensor outbackprop_buf_tensor;
  Tensor workspace_buf_tensor;

  dnnPrimitive_t pooling_bwd_;
  dnnPrimitive_t convert_outbackprop_;
  dnnLayout_t lt_user_input_bwd_;
  dnnLayout_t lt_outbackprop_user_;
  dnnLayout_t lt_outbackprop_prim_;
  dnnLayout_t lt_workspace_prim_;

  void* workspace_;
  void* outbackprop_buf_;
  void* pooling_res_[dnnResourceNumber];  // Pooling resource array

  MklPoolingOpParams mkl_params_;

  void MklCreateLayoutsAndPrimitives(OpKernelContext* context) {
    const Tensor& tensor_in_shape = MklGetInput(context, 0);
    const Tensor &out_backprop = MklGetInput(context, 1);

    if (!outbackprop_in_mkl_format_) {
      // For avgpooling, tensor_in_shape should have 1 dimension, and 4
      // elements.
      OP_REQUIRES(context,
                  tensor_in_shape.dims() == 1 &&
                  tensor_in_shape.NumElements() == 4,
                  errors::InvalidArgument("original input shape must be "
                                          "1-dimensional and 4 elements"));

      // For avgpooling, out_backprop should have 4 dimensions.
      OP_REQUIRES(context, out_backprop.dims() == 4,
                  errors::InvalidArgument("out_backprop must be "
                                          "4-dimensional"));
    } else {
      // Input in MKL format.
      OP_REQUIRES(context, out_backprop.dims() == 2,
                  errors::InvalidArgument("out_backprop in MKL format must be "
                                          "2-dimensional"));

      // For avgpooling, out_backprop should have 4 dimensions.
      OP_REQUIRES(context, mkl_out_backprop_shape.GetDimension() == 4,
                  errors::InvalidArgument("out_backprop must be "
                                          "4-dimensional"));
    }

    TensorShape output_shape;
    auto shape_vec = tensor_in_shape.vec<int32>();
    for (int64 i = 0; i < tensor_in_shape.NumElements(); ++i) {
      output_shape.AddDim(shape_vec(i));
    }

    MklPoolParameters params;
    params.Init(context, ksize_, stride_, padding_, data_format_, output_shape);

    // Extract the parameters for the op from the pooling specs
    ExtractMklOpParams(context, data_format_, params, &mkl_params_);

    // TODO(inteltf): Get outbackprop layout.
    // Do we need to create layout in every invocation?
    if (!outbackprop_in_mkl_format_) {
      CHECK_EQ(dnnLayoutCreate_F32(&lt_outbackprop_user_,
                                   mkl_params_.in_dim,
                                   mkl_params_.out_sizes,
                                   mkl_params_.out_strides),
               E_SUCCESS);
    } else {
      lt_outbackprop_user_ =
        (dnnLayout_t) mkl_out_backprop_shape.GetCurLayout();
    }

    // Create the backward primitive
    // Create DNN user layout
    CHECK_EQ(dnnLayoutCreate_F32(&lt_user_input_bwd_,
                                 mkl_params_.in_dim,
                                 mkl_params_.in_sizes,
                                 mkl_params_.in_strides),
             E_SUCCESS);

    // Create PoolingBackward primitive
    dnnAlgorithm_t algorithm = dnnAlgorithmPoolingAvg;
    dnnPrimitiveAttributes_t primAttr = nullptr;
    CHECK_EQ(dnnPoolingCreateBackward_F32(&pooling_bwd_,
                                          primAttr,
                                          algorithm,
                                          lt_user_input_bwd_,
                                          mkl_params_.kernel_size,
                                          mkl_params_.kernel_stride,
                                          mkl_params_.in_offset,
                                          dnnBorderZerosAsymm),
             E_SUCCESS);

    // Create expected outbackprop layout from the primitive.
    CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(&lt_outbackprop_prim_,
                                              pooling_bwd_,
                                              dnnResourceDiffDst),
             E_SUCCESS);

    CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(&lt_workspace_prim_,
                                              pooling_bwd_,
                                              dnnResourceWorkspace),
             E_SUCCESS);
  }

  void MklCleanup() {
    CHECK_EQ(dnnDelete_F32(pooling_bwd_), E_SUCCESS);
    pooling_bwd_ = nullptr;

    CHECK_EQ(dnnLayoutDelete_F32(lt_user_input_bwd_), E_SUCCESS);
    lt_user_input_bwd_ = nullptr;

    if (!outbackprop_in_mkl_format_) {
      CHECK_EQ(dnnLayoutDelete_F32(lt_outbackprop_user_), E_SUCCESS);
      lt_outbackprop_user_ = nullptr;
    }

    CHECK_EQ(dnnLayoutDelete_F32(lt_outbackprop_prim_), E_SUCCESS);
    lt_outbackprop_prim_ = nullptr;

    CHECK_EQ(dnnLayoutDelete_F32(lt_workspace_prim_), E_SUCCESS);
    lt_workspace_prim_ = nullptr;
  }
};

REGISTER_KERNEL_BUILDER(
  Name("MklAvgPool").Device(DEVICE_CPU).TypeConstraint<float>("T")
  .Label(mkl_layer_registry::kMklLayerLabel),
  MklAvgPoolingOp<CPUDevice, float>);

REGISTER_KERNEL_BUILDER(
  Name("MklAvgPoolGrad").Device(DEVICE_CPU).TypeConstraint<float>("T")
  .Label(mkl_layer_registry::kMklLayerLabel),
  MklAvgPoolingGradOp<CPUDevice, float>);

}       // namespace tensorflow
#endif  // INTEL_MKL
