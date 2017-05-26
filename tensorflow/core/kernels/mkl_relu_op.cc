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
    MklReluOpContext mkl_context;

    const Tensor& input = MklGetInput(context, 0);
    GetMklShape(context, 0, &mkl_context.input_shape);
    void* user_i = static_cast<void*>(const_cast<T*>(input.flat<T>().data()));
    bool input_in_mkl_format = mkl_context.input_shape.IsMklTensor();
    if (!input_in_mkl_format && !input.dims()) {  // handle the case of a scalar
      const TensorShape& o_shape = input.shape();
      Tensor* out_tensor = nullptr;
      mkl_context.output_shape.SetMklTensor(false);
      AllocateOutputSetMklShape(context, 0, &out_tensor, o_shape,
                                mkl_context.output_shape);
      void* out_o = static_cast<void*>(out_tensor->flat<T>().data());
      (static_cast<T*>(out_o))[0] =
          std::max((static_cast<T*>(user_i))[0], static_cast<T>(0));
      return;
    }

    // Generate size, stride for input if input is in MKL format.
    if (input_in_mkl_format) {
      mkl_context.in_dims = mkl_context.input_shape.GetDimension();
      mkl_context.in_sizes = new size_t[mkl_context.in_dims];
      mkl_context.in_strides = new size_t[mkl_context.in_dims];
      for (int i = 0; i < mkl_context.in_dims; i++) {
        mkl_context.in_sizes[i] = mkl_context.input_shape.GetSizes()[i];
        mkl_context.in_strides[i] = mkl_context.input_shape.GetStrides()[i];
      }
    } else {
      mkl_context.in_dims = input.dims();
      mkl_context.in_sizes = new size_t[mkl_context.in_dims];
      mkl_context.in_strides = new size_t[mkl_context.in_dims];
      for (int i = 0; i < mkl_context.in_dims; i++) {
        mkl_context.in_sizes[i] = input.dim_size((mkl_context.in_dims - 1) - i);
      }
      mkl_context.in_strides[0] = 1;
      for (int i = 1; i < mkl_context.in_dims; i++) {
        mkl_context.in_strides[i] =
            mkl_context.in_strides[i - 1] * mkl_context.in_sizes[i - 1];
      }
    }

    float negative_slope = 0.0;
    mkl_context.MklCreateInputLayouts(context);
    CHECK_EQ(dnnReLUCreateForward_F32(&mkl_context.prim_relu_fwd, NULL,
                                      mkl_context.lt_input, negative_slope),
             E_SUCCESS);

    Tensor* output = nullptr;

    if (input_in_mkl_format) {
      TensorShape tf_shape;
      mkl_context.output_shape.SetMklTensor(true);
      mkl_context.output_shape.SetMklLayout(mkl_context.prim_relu_fwd,
                                            dnnResourceDst);
      mkl_context.output_shape.SetTfLayout(
          mkl_context.in_dims, mkl_context.in_sizes, mkl_context.in_strides);
      mkl_context.output_shape.SetTfDimOrder(
          mkl_context.in_dims, mkl_context.input_shape.GetTfToMklDimMap());
      tf_shape.AddDim(dnnLayoutGetMemorySize_F32(static_cast<dnnLayout_t>(
                          mkl_context.output_shape.GetMklLayout())) /
                      sizeof(T));
      AllocateOutputSetMklShape(context, 0, &output, tf_shape,
                                mkl_context.output_shape);
    } else {
      const TensorShape& o_shape = input.shape();
      mkl_context.output_shape.SetMklTensor(false);
      AllocateOutputSetMklShape(context, 0, &output, o_shape,
                                mkl_context.output_shape);
    }

    void* user_o = static_cast<void*>(const_cast<T*>(output->flat<T>().data()));

    mkl_context.relu_res[dnnResourceDst] = user_o;
    mkl_context.relu_res[dnnResourceSrc] = user_i;
    CHECK_EQ(dnnExecute_F32(mkl_context.prim_relu_fwd, mkl_context.relu_res),
             E_SUCCESS);
    mkl_context.MklCleanup();
  }

 private:
  typedef struct {
    int in_dims;
    size_t* in_sizes;
    size_t* in_strides;
    MklShape input_shape, output_shape;
    dnnPrimitive_t prim_relu_fwd = nullptr;
    void* relu_res[dnnResourceNumber];
    dnnLayout_t lt_input = nullptr;

    void MklCleanup() {
      bool input_in_mkl_format = input_shape.IsMklTensor();
      if (!input_in_mkl_format) {
        dnnLayoutDelete_F32(lt_input);
        free(in_sizes);
        free(in_strides);
      }
      dnnDelete_F32(prim_relu_fwd);
    }

    void MklCreateInputLayouts(OpKernelContext* context) {
      bool input_in_mkl_format = input_shape.IsMklTensor();
      if (!input_in_mkl_format) {
        CHECK_EQ(dnnLayoutCreate_F32(&lt_input, in_dims, in_sizes, in_strides),
                 E_SUCCESS);
      } else {
        lt_input = static_cast<dnnLayout_t>(input_shape.GetCurLayout());
      }
    }
  } MklReluOpContext;
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
    void* relu_res[dnnResourceNumber];
    dnnPrimitive_t prim_relu_bwd;
    dnnLayout_t lt_input, lt_grad;

    void MklPrepareReluGradInputs(OpKernelContext* context,
                                  Tensor* mkl_tmp_input_buf_tensor) {
      const Tensor& g = MklGetInput(context, 0);
      const Tensor& a = MklGetInput(context, 1);
      void* buf_input = static_cast<void*>(const_cast<T*>(a.flat<T>().data()));
      void* mkl_buffer_convert = nullptr;
      dnnPrimitive_t cv_input_to_grad = nullptr;

      // if input and grad are not in the same layout, do a conversion between
      // them.
      if (!dnnLayoutCompare_F32(lt_input, lt_grad)) {
        AllocTmpBuffer(context, mkl_tmp_input_buf_tensor, lt_grad,
                       &mkl_buffer_convert);
        CHECK_EQ(dnnConversionCreate_F32(&cv_input_to_grad, lt_input,
                   lt_grad), E_SUCCESS);
        CHECK_EQ(dnnConversionExecute_F32(cv_input_to_grad, buf_input,
                                          mkl_buffer_convert),
                 E_SUCCESS);
        relu_res[dnnResourceSrc] = mkl_buffer_convert;
        dnnDelete_F32(cv_input_to_grad);
      } else {
        relu_res[dnnResourceSrc] = buf_input;
      }

      void* buf_grad = static_cast<void*>(const_cast<T*>(g.flat<T>().data()));
      relu_res[dnnResourceDiffDst] = buf_grad;
    }

    void MklCreateInputLayouts(OpKernelContext* context) {
      bool grad_is_mkl = grad_shape.IsMklTensor();
      bool input_is_mkl = input_shape.IsMklTensor();
      if (!input_is_mkl) {
        CHECK_EQ(dnnLayoutCreate_F32(&lt_input, in_dims, in_sizes, in_strides),
                 E_SUCCESS);
      } else {
        lt_input = static_cast<dnnLayout_t>(input_shape.GetCurLayout());
      }

      if (!grad_is_mkl) {
        CHECK_EQ(dnnLayoutCreate_F32(&lt_grad, in_dims, in_sizes, in_strides),
                 E_SUCCESS);
      } else {
        lt_grad = static_cast<dnnLayout_t>(grad_shape.GetCurLayout());
      }
    }

    void MklCleanup() {
      bool grad_is_mkl = grad_shape.IsMklTensor();
      bool input_is_mkl = input_shape.IsMklTensor();
      dnnDelete_F32(prim_relu_bwd);
      if (!input_is_mkl) {
        dnnLayoutDelete_F32(lt_input);
        free(in_sizes);
        free(in_strides);
      }
      if (!grad_is_mkl) {
        dnnLayoutDelete_F32(lt_grad);
      }
    }
  } MklReluGradOpContext;
};

template <typename Device, typename T>

void MklReluGradOp<Device, T>::Compute(OpKernelContext* context) {
  MklReluGradOpContext mkl_context;
  const Tensor& g = MklGetInput(context, 0);
  const Tensor& a = MklGetInput(context, 1);

  void* user_i = static_cast<void*>(const_cast<T*>(a.flat<T>().data()));
  void* user_g = static_cast<void*>(const_cast<T*>(g.flat<T>().data()));

  GetMklShape(context, 0, &mkl_context.grad_shape);
  GetMklShape(context, 1, &mkl_context.input_shape);

  bool grad_is_mkl = mkl_context.grad_shape.IsMklTensor();
  bool input_is_mkl = mkl_context.input_shape.IsMklTensor();
  if (!input_is_mkl && !grad_is_mkl &&
      !MklReluHelpers::ValidateSameSize(context, g, a))
    return;
  Tensor* output = nullptr;
  if (!input_is_mkl && !grad_is_mkl &&
      !a.dims()) {  // handle the case of a scalar
    // Allocate space for g and
    const TensorShape& g_shape = g.shape();
    mkl_context.output_shape.SetMklTensor(false);
    AllocateOutputSetMklShape(context, 0, &output, g_shape,
                              mkl_context.output_shape);
    void* out_o = static_cast<void*>(output->flat<T>().data());
    (static_cast<T*>(out_o))[0] =
        (static_cast<T*>(user_g))[0] * ((static_cast<T*>(user_i))[0] > 0);
    return;
  }

  // Generate size, stride for input if input/grad is in MKL format.
  if (grad_is_mkl || input_is_mkl) {
    const MklShape* tmp_mkl_shape =
        (grad_is_mkl) ? &mkl_context.grad_shape : &mkl_context.input_shape;

    mkl_context.in_dims = tmp_mkl_shape->GetDimension();
    mkl_context.in_strides = new size_t[mkl_context.in_dims];
    mkl_context.in_sizes = new size_t[mkl_context.in_dims];
    for (int i = 0; i < mkl_context.in_dims; i++) {
      mkl_context.in_sizes[i] = tmp_mkl_shape->GetSizes()[i];
      mkl_context.in_strides[i] = tmp_mkl_shape->GetStrides()[i];
    }
  } else {
    mkl_context.in_dims = g.dims();
    mkl_context.in_strides = new size_t[mkl_context.in_dims];
    mkl_context.in_sizes = new size_t[mkl_context.in_dims];

    for (int i = 0; i < mkl_context.in_dims; i++) {
      mkl_context.in_sizes[i] = g.dim_size((mkl_context.in_dims - 1) - i);
    }
    mkl_context.in_strides[0] = 1;
    for (int i = 1; i < mkl_context.in_dims; i++) {
      mkl_context.in_strides[i] =
          mkl_context.in_strides[i - 1] * mkl_context.in_sizes[i - 1];
    }
  }

  mkl_context.MklCreateInputLayouts(context);
  float negative_slope = 0.0;
  CHECK_EQ(dnnReLUCreateBackward_F32(&mkl_context.prim_relu_bwd, NULL,
                                     mkl_context.lt_grad, mkl_context.lt_grad,
                                     negative_slope),
           E_SUCCESS);
  Tensor mkl_tmp_input_buf_tensor;
  mkl_context.MklPrepareReluGradInputs(context, &mkl_tmp_input_buf_tensor);

  if (input_is_mkl ||
      grad_is_mkl) { /*if  grad or input are MKL leave it in MKL*/
    TensorShape tf_shape;
    mkl_context.output_shape.SetMklTensor(true);
    mkl_context.output_shape.SetMklLayout(mkl_context.prim_relu_bwd,
                                          dnnResourceDiffSrc);
    mkl_context.output_shape.SetTfLayout(
        mkl_context.in_dims, mkl_context.in_sizes, mkl_context.in_strides);
    // If input_is_mkl or grad_is_mkl, then we copy strides and sizes from Mkl
    // shape of one that is in MKL layout.
    if (grad_is_mkl == true) {
      mkl_context.output_shape.SetTfDimOrder(
          mkl_context.in_dims, mkl_context.grad_shape.GetTfToMklDimMap());
    } else {
      mkl_context.output_shape.SetTfDimOrder(
          mkl_context.in_dims, mkl_context.input_shape.GetTfToMklDimMap());
    }

    tf_shape.AddDim(dnnLayoutGetMemorySize_F32(static_cast<dnnLayout_t>(
                        mkl_context.output_shape.GetMklLayout())) /
                    sizeof(T));
    AllocateOutputSetMklShape(context, 0, &output, tf_shape,
                              mkl_context.output_shape);

  } else {
    const TensorShape& o_shape = g.shape();
    mkl_context.output_shape.SetMklTensor(false);
    AllocateOutputSetMklShape(context, 0, &output, o_shape,
                              mkl_context.output_shape);
  }

  mkl_context.relu_res[dnnResourceDiffSrc] =
      static_cast<void*>(output->flat<T>().data());

  CHECK_EQ(dnnExecute_F32(mkl_context.prim_relu_bwd, mkl_context.relu_res),
           E_SUCCESS);
  mkl_context.MklCleanup();
}

/* Register DNN kernels for supported operations and supported types - right now
 * it is only Relu and f32*/
#define REGISTER_RELU_MKL_SUPPORTED_KERNELS_TYPES(type)             \
  REGISTER_KERNEL_BUILDER(Name("_MklRelu")                          \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<type>("T")            \
                              .Label(mkl_op_registry::kMklOpLabel), \
                          MklReluOp<CPUDevice, type>);              \
  REGISTER_KERNEL_BUILDER(Name("_MklReluGrad")                      \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<type>("T")            \
                              .Label(mkl_op_registry::kMklOpLabel), \
                          MklReluGradOp<CPUDevice, type>);
TF_CALL_float(REGISTER_RELU_MKL_SUPPORTED_KERNELS_TYPES);

}  // namespace tensorflow

#endif  // INTEL_MKL
