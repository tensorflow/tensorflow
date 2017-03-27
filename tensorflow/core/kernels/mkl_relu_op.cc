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

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/util/mkl_util.h"
#include "third_party/mkl/include/mkl_dnn.h"
#include "third_party/mkl/include/mkl_dnn_types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

struct MklReluHelpers {
  static void ValidateSameSizeHelper(OpKernelContext* context, const Tensor& g,
                                     const Tensor& a) {
    OP_REQUIRES(context, a.IsSameSize(g),
                errors::InvalidArgument("g and a must be the same size"));
  }
  static bool ValidateSameSize(OpKernelContext* context, const Tensor& g,
                               const Tensor& a) {
    ValidateSameSizeHelper(context, g, a);
    return context->status().ok();
  }
};

template <typename Device, typename T>
class MklReluOp : public OpKernel {
 public:
  ~MklReluOp() {}

  explicit MklReluOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = MklGetInput(context, 0);
    GetMklShape(context, 0, &mkl_params.input_shape);
    void* user_i = static_cast<void*>(const_cast<T*>(input.flat<T>().data()));
    bool input_in_mkl_format = mkl_params.input_shape.IsMklTensor();
    if (!input_in_mkl_format && !input.dims()) {  // handle the case of a scalar
      const TensorShape& o_shape = input.shape();
      Tensor* out_tensor = nullptr;
      mkl_params.output_shape.SetMklTensor(false);
      AllocateOutputSetMklshape(context, 0, &out_tensor, o_shape,
                                mkl_params.output_shape);
      void* out_o = static_cast<void*>(out_tensor->flat<T>().data());
      (static_cast<T*>(out_o))[0] =
          std::max((static_cast<T*>(user_i))[0], static_cast<T>(0));
      return;
    }

    // Generate size, stride for input if input is in MKL format.
    if (input_in_mkl_format) {
      mkl_params.in_dims = mkl_params.input_shape.GetDimension();
      mkl_params.in_sizes = new size_t[mkl_params.in_dims];
      mkl_params.in_strides = new size_t[mkl_params.in_dims];
      for (int i = 0; i < mkl_params.in_dims; i++) {
        mkl_params.in_sizes[i] = mkl_params.input_shape.GetSizes()[i];
        mkl_params.in_strides[i] = mkl_params.input_shape.GetStrides()[i];
      }
    } else {
      mkl_params.in_dims = input.dims();
      mkl_params.in_sizes = new size_t[mkl_params.in_dims];
      mkl_params.in_strides = new size_t[mkl_params.in_dims];
      for (int i = 0; i < mkl_params.in_dims; i++) {
        mkl_params.in_sizes[i] = input.dim_size((mkl_params.in_dims - 1) - i);
      }
      mkl_params.in_strides[0] = 1;
      for (int i = 1; i < mkl_params.in_dims; i++) {
        mkl_params.in_strides[i] =
            mkl_params.in_strides[i - 1] * mkl_params.in_sizes[i - 1];
      }
    }

    float negative_slope = 0.0;
    MklCreateInputLayouts(context);
    CHECK_EQ(dnnReLUCreateForward_F32(&mkl_prim_relu_fwd_, NULL, mkl_lt_input_,
                                      negative_slope),
             E_SUCCESS);

    Tensor* output = nullptr;

    if (input_in_mkl_format) {
      TensorShape tf_shape;
      mkl_params.output_shape.SetMklTensor(true);
      mkl_params.output_shape.SetMklLayout(mkl_prim_relu_fwd_, dnnResourceDst);
      mkl_params.output_shape.SetTfLayout(
          mkl_params.in_dims, mkl_params.in_sizes, mkl_params.in_strides);
      tf_shape.AddDim(dnnLayoutGetMemorySize_F32(static_cast<dnnLayout_t>(
                          mkl_params.output_shape.GetMklLayout())) /
                      sizeof(T));
      AllocateOutputSetMklshape(context, 0, &output, tf_shape,
                                mkl_params.output_shape);
    } else {
      const TensorShape& o_shape = input.shape();
      mkl_params.output_shape.SetMklTensor(false);
      AllocateOutputSetMklshape(context, 0, &output, o_shape,
                                mkl_params.output_shape);
    }

    void* user_o = static_cast<void*>(const_cast<T*>(output->flat<T>().data()));

    relu_res[dnnResourceDst] = user_o;
    relu_res[dnnResourceSrc] = user_i;
    CHECK_EQ(dnnExecute_F32(mkl_prim_relu_fwd_, relu_res), E_SUCCESS);
    Mklcleanup();
  }

 private:
  typedef struct {
    int in_dims;
    size_t* in_sizes;
    size_t* in_strides;
    MklShape input_shape, output_shape;
  } MklReluOpParams_;

  void Mklcleanup() {
    bool input_in_mkl_format = mkl_params.input_shape.IsMklTensor();
    if (!input_in_mkl_format) dnnLayoutDelete_F32(mkl_lt_input_);
    dnnDelete_F32(mkl_prim_relu_fwd_);
  }
  void MklCreateInputLayouts(OpKernelContext* context) {
    bool input_in_mkl_format = mkl_params.input_shape.IsMklTensor();
    if (!input_in_mkl_format) {
      CHECK_EQ(dnnLayoutCreate_F32(&mkl_lt_input_, mkl_params.in_dims,
                                   mkl_params.in_sizes, mkl_params.in_strides),
               E_SUCCESS);
    } else {
      mkl_lt_input_ =
          static_cast<dnnLayout_t>(mkl_params.input_shape.GetCurLayout());
    }
  }

  dnnPrimitive_t mkl_prim_relu_fwd_ = nullptr;
  MklReluOpParams_ mkl_params;
  void* relu_res[dnnResourceNumber];
  dnnLayout_t mkl_lt_input_ = nullptr;
};

template <typename Device, typename T>
class MklReluGradOp : public OpKernel {
 public:
  ~MklReluGradOp() {}

  explicit MklReluGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override;

 private:
  typedef struct {
    int in_dims;
    size_t* in_sizes;
    size_t* in_strides;
    MklShape input_shape, grad_shape, output_shape;
  } MklReluGradOpParams_;
  MklReluGradOpParams_ mkl_params;

  void MklPrepareReluGradInputs(OpKernelContext* context,
                                Tensor* mkl_tmp_grad_buf_tensor,
                                Tensor* mkl_tmp_input_buf_tensor) {
    dnnPrimitive_t cv_user_to_reluB_input = nullptr,
                   cv_user_to_reluB_grad = nullptr;
    dnnLayout_t mkl_lt_internal_input = nullptr, mkl_lt_internal_grad = nullptr;

    const Tensor& g = MklGetInput(context, 0);
    const Tensor& a = MklGetInput(context, 1);

    void* user_i = static_cast<void*>(const_cast<T*>(a.flat<T>().data()));
    void* user_g = static_cast<void*>(const_cast<T*>(g.flat<T>().data()));

    CHECK_EQ(
        dnnLayoutCreateFromPrimitive_F32(
            &mkl_lt_internal_grad, mkl_prim_relu_back_, dnnResourceDiffDst),
        E_SUCCESS);

    CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(
                 &mkl_lt_internal_input, mkl_prim_relu_back_, dnnResourceSrc),
             E_SUCCESS);

    if (!dnnLayoutCompare_F32(mkl_lt_internal_grad, mkl_lt_grad_)) {
      AllocTmpBuffer(context, mkl_tmp_grad_buf_tensor, mkl_lt_internal_grad,
                     &relu_res[dnnResourceDiffDst]);
      CHECK_EQ(dnnConversionCreate_F32(&cv_user_to_reluB_grad, mkl_lt_grad_,
                                       mkl_lt_internal_grad),
               E_SUCCESS);
    }

    if (!dnnLayoutCompare_F32(mkl_lt_internal_input, mkl_lt_input_)) {
      AllocTmpBuffer(context, mkl_tmp_input_buf_tensor, mkl_lt_internal_input,
                     &relu_res[dnnResourceSrc]);
      CHECK_EQ(dnnConversionCreate_F32(&cv_user_to_reluB_input, mkl_lt_input_,
                                       mkl_lt_internal_input),
               E_SUCCESS);
    }
    if (cv_user_to_reluB_input) {
      CHECK_EQ(dnnConversionExecute_F32(cv_user_to_reluB_input, user_i,
                                        relu_res[dnnResourceSrc]),
               E_SUCCESS);
    } else {
      relu_res[dnnResourceSrc] = user_i;
    }
    if (cv_user_to_reluB_input) dnnDelete_F32(cv_user_to_reluB_input);

    dnnLayoutDelete_F32(mkl_lt_internal_input);
    if (cv_user_to_reluB_grad) {
      CHECK_EQ(dnnConversionExecute_F32(cv_user_to_reluB_grad, user_g,
                                        relu_res[dnnResourceDiffDst]),
               E_SUCCESS);
    } else {
      relu_res[dnnResourceDiffDst] = user_g;
    }

    if (cv_user_to_reluB_grad) dnnDelete_F32(cv_user_to_reluB_grad);
    dnnLayoutDelete_F32(mkl_lt_internal_grad);
  }

  void MklCreateInputLayouts(OpKernelContext* context) {
    bool grad_is_mkl = mkl_params.grad_shape.IsMklTensor();
    bool input_is_mkl = mkl_params.input_shape.IsMklTensor();
    if (!input_is_mkl) {
      CHECK_EQ(dnnLayoutCreate_F32(&mkl_lt_input_, mkl_params.in_dims,
                                   mkl_params.in_sizes, mkl_params.in_strides),
               E_SUCCESS);
    } else {
      mkl_lt_input_ =
          static_cast<dnnLayout_t>(mkl_params.input_shape.GetCurLayout());
    }

    if (!grad_is_mkl) {
      CHECK_EQ(dnnLayoutCreate_F32(&mkl_lt_grad_, mkl_params.in_dims,
                                   mkl_params.in_sizes, mkl_params.in_strides),
               E_SUCCESS);
    } else {
      mkl_lt_grad_ =
          static_cast<dnnLayout_t>(mkl_params.grad_shape.GetCurLayout());
    }
  }

  void MklCleanup() {
    bool grad_is_mkl = mkl_params.grad_shape.IsMklTensor();
    bool input_is_mkl = mkl_params.input_shape.IsMklTensor();
    dnnDelete_F32(mkl_prim_relu_back_);
    if (!input_is_mkl) {
      dnnLayoutDelete_F32(mkl_lt_input_);
    }
    if (!grad_is_mkl) {
      dnnLayoutDelete_F32(mkl_lt_grad_);
    }
  }
  void* relu_res[dnnResourceNumber];
  dnnPrimitive_t mkl_prim_relu_back_ = nullptr;
  dnnLayout_t mkl_lt_input_, mkl_lt_grad_;
};

template <typename Device, typename T>

void MklReluGradOp<Device, T>::Compute(OpKernelContext* context) {
  const Tensor& g = MklGetInput(context, 0);
  const Tensor& a = MklGetInput(context, 1);

  void* user_i = static_cast<void*>(const_cast<T*>(a.flat<T>().data()));
  void* user_g = static_cast<void*>(const_cast<T*>(g.flat<T>().data()));

  GetMklShape(context, 0, &mkl_params.grad_shape);
  GetMklShape(context, 1, &mkl_params.input_shape);

  bool grad_is_mkl = mkl_params.grad_shape.IsMklTensor();
  bool input_is_mkl = mkl_params.input_shape.IsMklTensor();
  if (!input_is_mkl && !grad_is_mkl &&
      !MklReluHelpers::ValidateSameSize(context, g, a))
    return;
  Tensor* output = nullptr;
  if (!input_is_mkl && !grad_is_mkl &&
      !a.dims()) {  // handle the case of a scalar
    // Allocate space for g and
    const TensorShape& g_shape = g.shape();
    mkl_params.output_shape.SetMklTensor(false);
    AllocateOutputSetMklshape(context, 0, &output, g_shape,
                              mkl_params.output_shape);
    void* out_o = static_cast<void*>(output->flat<T>().data());
    (static_cast<T*>(out_o))[0] =
        (static_cast<T*>(user_g))[0] * ((static_cast<T*>(user_i))[0] > 0);
    return;
  }

  // Generate size, stride for input if input/grad is in MKL format.
  if (grad_is_mkl || input_is_mkl) {
    const MklShape* tmp_mkl_shape =
        (grad_is_mkl) ? &mkl_params.grad_shape : &mkl_params.input_shape;

    mkl_params.in_dims = tmp_mkl_shape->GetDimension();
    mkl_params.in_strides = new size_t[mkl_params.in_dims];
    mkl_params.in_sizes = new size_t[mkl_params.in_dims];
    for (int i = 0; i < mkl_params.in_dims; i++) {
      mkl_params.in_sizes[i] = tmp_mkl_shape->GetSizes()[i];
      mkl_params.in_strides[i] = tmp_mkl_shape->GetStrides()[i];
    }
  } else {
    mkl_params.in_dims = g.dims();
    mkl_params.in_strides = new size_t[mkl_params.in_dims];
    mkl_params.in_sizes = new size_t[mkl_params.in_dims];

    for (int i = 0; i < mkl_params.in_dims; i++) {
      mkl_params.in_sizes[i] = g.dim_size((mkl_params.in_dims - 1) - i);
    }
    mkl_params.in_strides[0] = 1;
    for (int i = 1; i < mkl_params.in_dims; i++) {
      mkl_params.in_strides[i] =
          mkl_params.in_strides[i - 1] * mkl_params.in_sizes[i - 1];
    }
  }

  MklCreateInputLayouts(context);
  float negative_slope = 0.0;
  CHECK_EQ(dnnReLUCreateBackward_F32(&mkl_prim_relu_back_, NULL, mkl_lt_grad_,
                                     mkl_lt_input_, negative_slope),
           E_SUCCESS);
  Tensor mkl_tmp_grad_buf_tensor, mkl_tmp_input_buf_tensor;
  MklPrepareReluGradInputs(context, &mkl_tmp_grad_buf_tensor,
                           &mkl_tmp_input_buf_tensor);

  if (input_is_mkl ||
      grad_is_mkl) { /*if  grad or input are MKL leave it in MKL*/
    TensorShape tf_shape;
    mkl_params.output_shape.SetMklTensor(true);
    mkl_params.output_shape.SetMklLayout(mkl_prim_relu_back_,
                                         dnnResourceDiffSrc);
    mkl_params.output_shape.SetTfLayout(mkl_params.in_dims, mkl_params.in_sizes,
                                        mkl_params.in_strides);
    tf_shape.AddDim(dnnLayoutGetMemorySize_F32(static_cast<dnnLayout_t>(
                        mkl_params.output_shape.GetMklLayout())) /
                    sizeof(T));
    AllocateOutputSetMklshape(context, 0, &output, tf_shape,
                              mkl_params.output_shape);

  } else {
    const TensorShape& o_shape = g.shape();
    mkl_params.output_shape.SetMklTensor(false);
    AllocateOutputSetMklshape(context, 0, &output, o_shape,
                              mkl_params.output_shape);
  }

  relu_res[dnnResourceDiffSrc] = static_cast<void*>(output->flat<T>().data());

  CHECK_EQ(dnnExecute_F32(mkl_prim_relu_back_, relu_res), E_SUCCESS);
  MklCleanup();
}

/* Register DNN kernels for supported operations and supported types - right now
 * it is only Relu and f32*/
#define REGISTER_RELU_MKL_SUPPORTED_KERNELS_TYPES(type)                   \
  REGISTER_KERNEL_BUILDER(Name("MklRelu")                                 \
                              .Device(DEVICE_CPU)                         \
                              .TypeConstraint<type>("T")                  \
                              .Label(mkl_layer_registry::kMklLayerLabel), \
                          MklReluOp<CPUDevice, type>);                    \
  REGISTER_KERNEL_BUILDER(Name("MklReluGrad")                             \
                              .Device(DEVICE_CPU)                         \
                              .TypeConstraint<type>("T")                  \
                              .Label(mkl_layer_registry::kMklLayerLabel), \
                          MklReluGradOp<CPUDevice, type>);
TF_CALL_float(REGISTER_RELU_MKL_SUPPORTED_KERNELS_TYPES);

}  // namespace tensorflow

#endif  // INTEL_MKL
