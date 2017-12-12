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
#include "mkl_dnn.h"
#include "mkl_dnn_types.h"

#ifdef INTEL_MKL_DNN
#include "mkldnn.hpp"

using mkldnn::stream;
using mkldnn::prop_kind;
using mkldnn::algorithm;
using mkldnn::relu_forward;
using mkldnn::relu_backward;
using mkldnn::eltwise_relu;
using mkldnn::eltwise_elu;
using mkldnn::eltwise_tanh;
#endif

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

#ifndef INTEL_MKL_DNN

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

      // if input and grad are not in the same layout,
      // do a conversion between them.
      if (!dnnLayoutCompare_F32(lt_input, lt_grad)) {
        AllocTmpBuffer(context, mkl_tmp_input_buf_tensor, lt_grad,
                       &mkl_buffer_convert);
        CHECK_EQ(dnnConversionCreate_F32(&cv_input_to_grad, lt_input,
                   lt_grad), E_SUCCESS);
        CHECK_EQ(dnnConversionExecute_F32(cv_input_to_grad, buf_input,
                                          mkl_buffer_convert), E_SUCCESS);
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

  if (!input_is_mkl && !grad_is_mkl && !a.dims()) {
    // handle the scalar case
    const TensorShape& g_shape = g.shape();
    mkl_context.output_shape.SetMklTensor(false);
    AllocateOutputSetMklShape(context, 0, &output, g_shape,
                              mkl_context.output_shape);

    void* out_o = static_cast<void*>(output->flat<T>().data());
    (static_cast<T*>(out_o))[0] =
        (static_cast<T*>(user_g))[0] * ((static_cast<T*>(user_i))[0] > 0);
    return;
  }

  // generate size, stride for input if input/grad is in mkl format.
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
                                     negative_slope), E_SUCCESS);
  Tensor mkl_tmp_input_buf_tensor;
  mkl_context.MklPrepareReluGradInputs(context, &mkl_tmp_input_buf_tensor);

  if (input_is_mkl ||
      grad_is_mkl) { /*if  grad or input are mkl leave it in mkl*/
    TensorShape tf_shape;
    mkl_context.output_shape.SetMklTensor(true);
    mkl_context.output_shape.SetMklLayout(mkl_context.prim_relu_bwd,
                                          dnnResourceDiffSrc);
    mkl_context.output_shape.SetTfLayout(
        mkl_context.in_dims, mkl_context.in_sizes, mkl_context.in_strides);
    // if input_is_mkl or grad_is_mkl, then we copy strides and sizes from mkl
    // shape of one that is in mkl layout.
    if (grad_is_mkl == true) {
      mkl_context.output_shape.SetTfDimOrder(
          mkl_context.in_dims, mkl_context.grad_shape.GetTfToMklDimMap());
    } else {
      mkl_context.output_shape.SetTfDimOrder(
          mkl_context.in_dims, mkl_context.input_shape.GetTfToMklDimMap());
    }

    tf_shape.AddDim(dnnLayoutGetMemorySize_F32(static_cast<dnnLayout_t>(
                    mkl_context.output_shape.GetMklLayout())) / sizeof(T));
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

  CHECK_EQ(dnnExecute_F32(mkl_context.prim_relu_bwd,
                          mkl_context.relu_res),
                          E_SUCCESS);
  mkl_context.MklCleanup();
}


#else  // INTEL_MKL_DNN

template <typename Device, typename T, algorithm alg_kind>
class MklReluOpBase : public OpKernel {
 public:
  ~MklReluOpBase() {}

  explicit MklReluOpBase(OpKernelConstruction* context) : OpKernel(context) {
  }

  virtual void Compute_Scalar(OpKernelContext* context) = 0;

  void Compute(OpKernelContext* context) override {
    try {
      auto cpu_engine = engine(engine::cpu, 0);
      const size_t src_index = 0;  // index of src input tensor
      const size_t dst_index = 0;  // index of dst output tensor
      const Tensor& src_tensor = MklGetInput(context, src_index);
      MklDnnShape dnn_shape_src;
      GetMklShape(context, src_index, &dnn_shape_src);

      Tensor* dst_tensor = nullptr;
      if (src_tensor.dims() == 0) {
        Compute_Scalar(context);
        return;
      }

      // Create relu primitive.
      MklDnnData<T> src(&cpu_engine);
      MklDnnData<T> dst(&cpu_engine);

      // Set DNN primitive - src
      memory::desc src_md({}, memory::data_undef, memory::format_undef);
      if (dnn_shape_src.IsMklTensor()) {
        src_md = dnn_shape_src.GetMklLayout();
      } else {
        auto src_dims = TFShapeToMklDnnDims(src_tensor.shape());
        auto src_strides = CalculateTFStrides(src_dims);
        // Create blocked memory descriptor
        src_md = MklDnnData<T>::CreateBlockedMemDesc(src_dims, src_strides);
      }
      src.SetUsrMem(src_md, &src_tensor);

      T alpha = 0, beta = 0;
      std::shared_ptr<relu_forward::primitive_desc> relu_fwd_pd;
      auto relu_fwd_desc = relu_forward::desc(prop_kind::forward_training,
          // Operator memory descriptor is same as user memory descriptor.
                                              alg_kind, src.GetUsrMemDesc(),
                                              alpha, beta);
      relu_fwd_pd.reset(new relu_forward::primitive_desc(relu_fwd_desc,
                                                         cpu_engine));

      // allocate dst tensor
      MklDnnShape dnn_shape_dst;
      TensorShape tf_shape_dst;
      if (dnn_shape_src.IsMklTensor()) {
        dnn_shape_dst.SetMklTensor(true);
        auto dst_pd = relu_fwd_pd->dst_primitive_desc();
        dnn_shape_dst.SetMklLayout(&dst_pd);
        dnn_shape_dst.SetElemType(MklDnnType<T>());
        dnn_shape_dst.SetTfLayout(dnn_shape_src.GetDimension(),
                                  dnn_shape_src.GetSizesAsMklDnnDims(),
                                  dnn_shape_src.GetTfDataFormat());
        tf_shape_dst.AddDim(dst_pd.get_size()/sizeof(T));
      } else {
        dnn_shape_dst.SetMklTensor(false);
        tf_shape_dst = src_tensor.shape();
      }
      AllocateOutputSetMklShape(context, dst_index, &dst_tensor, tf_shape_dst,
                                dnn_shape_dst);

      // Destination memory descriptor is same as source memory descriptor.
      auto dst_md = src_md;
      dst.SetUsrMem(dst_md, dst_tensor);

      // execute net
      std::vector<primitive> net;
      auto relu_fwd = relu_forward(*relu_fwd_pd, src.GetOpMem(),
                                   dst.GetOpMem());
      net.push_back(relu_fwd);
      stream(stream::kind::eager).submit(net).wait();
    } catch (mkldnn::error &e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) +
                         ", in file " + string(__FILE__) + ":" +
                         std::to_string(__LINE__);
      OP_REQUIRES_OK(context,
                     errors::Aborted("Operation received an exception:",
                        error_msg));
    }
  }
};


template <typename Device, typename T, algorithm alg_kind>
class MklReluGradOpBase : public OpKernel {
 public:
  ~MklReluGradOpBase() {}

  explicit MklReluGradOpBase(OpKernelConstruction* context) :
    OpKernel(context) {}

  virtual void Compute_Scalar(OpKernelContext* context) = 0;

  void Compute(OpKernelContext* context)  {
    try {
      auto cpu_engine = engine(engine::cpu, 0);
      MklDnnData<T> src(&cpu_engine);
      MklDnnData<T> diff_dst(&cpu_engine);
      MklDnnData<T> diff_src(&cpu_engine);

      const size_t diff_dst_index = 0;  // index of diff_dst input tensor
      const size_t src_index = 1;       // index of src input tensor
      const size_t diff_src_index = 0;  // index of diff_src output tensor

      const Tensor& src_tensor      = MklGetInput(context, src_index);
      const Tensor& diff_dst_tensor = MklGetInput(context, diff_dst_index);
      Tensor* diff_src_tensor       = nullptr;

      MklDnnShape dnn_shape_src, dnn_shape_diff_dst;
      GetMklShape(context, src_index, &dnn_shape_src);
      GetMklShape(context, diff_dst_index, &dnn_shape_diff_dst);

      int src_dims_size = src_tensor.dims();
      if (src_dims_size == 0) {
        Compute_Scalar(context);
        return;
      }

      // Set DNN primitives for src & diff_dst
      memory::desc src_md({}, memory::data_undef, memory::format_undef);
      memory::desc diff_dst_md({}, memory::data_undef, memory::format_undef);
      if (dnn_shape_src.IsMklTensor() || dnn_shape_diff_dst.IsMklTensor()) {
        if (dnn_shape_diff_dst.IsMklTensor()) {
          diff_dst_md = dnn_shape_diff_dst.GetMklLayout();
          src_md = diff_dst_md;
        } else {
          src_md = dnn_shape_src.GetMklLayout();
          diff_dst_md = src_md;
        }
      } else {
        auto src_dims = TFShapeToMklDnnDims(src_tensor.shape());
        auto src_strides = CalculateTFStrides(src_dims);
        src_md = MklDnnData<T>::CreateBlockedMemDesc(src_dims, src_strides);
        diff_dst_md = src_md;
      }
      src.SetUsrMem(src_md, &src_tensor);
      diff_dst.SetUsrMem(diff_dst_md, &diff_dst_tensor);

      T alpha = 0, beta = 0;
      std::shared_ptr<relu_forward::primitive_desc> relu_fwd_pd;
      auto relu_fwd_desc = relu_forward::desc(prop_kind::forward_training,
                                              alg_kind, src_md, alpha, beta);
      relu_fwd_pd.reset(new relu_forward::primitive_desc(relu_fwd_desc,
                                                         cpu_engine));
      auto relu_bwd_desc = relu_backward::desc(alg_kind, diff_dst_md, src_md,
                                                alpha, beta);
      auto relu_bwd_pd  = relu_backward::primitive_desc(relu_bwd_desc,
                                                cpu_engine, *relu_fwd_pd);

      // allocate diff_src tensor
      MklDnnShape dnn_shape_diff_src;
      TensorShape tf_shape_diff_src;
      if (dnn_shape_src.IsMklTensor()) {
        dnn_shape_diff_src.SetMklTensor(true);
        auto diff_src_pd = relu_bwd_pd.diff_src_primitive_desc();
        dnn_shape_diff_src.SetMklLayout(&diff_src_pd);
        dnn_shape_diff_src.SetElemType(MklDnnType<T>());
        dnn_shape_diff_src.SetTfLayout(dnn_shape_src.GetDimension(),
                                       dnn_shape_src.GetSizesAsMklDnnDims(),
                                       dnn_shape_src.GetTfDataFormat());
        tf_shape_diff_src.AddDim(diff_src_pd.get_size()/sizeof(T));
      } else {
        dnn_shape_diff_src.SetMklTensor(false);
        tf_shape_diff_src = src_tensor.shape();
      }
      AllocateOutputSetMklShape(context, diff_src_index, &diff_src_tensor,
                                 tf_shape_diff_src, dnn_shape_diff_src);

      // diff_src memory descriptor is same as diff_dst memory descriptor.
      auto diff_src_md = diff_dst_md;
      diff_src.SetUsrMem(diff_src_md, diff_src_tensor);

      PrepareAndExecuteNet(relu_bwd_pd, &src, &diff_src, &diff_dst);
     } catch (mkldnn::error &e) {
       string error_msg = "Status: " + std::to_string(e.status) +
                          ", message: " + string(e.message) +
                          ", in file " + string(__FILE__) + ":" +
                          std::to_string(__LINE__);
       OP_REQUIRES_OK(context,
                      errors::Aborted("Operation received an exception:",
                                      error_msg));
    }
  }

  void PrepareAndExecuteNet(const relu_backward::primitive_desc& relu_prim_desc,
                  MklDnnData<T>* src, MklDnnData<T>* diff_src, MklDnnData<T>*
                  diff_dst) {
    std::vector<primitive> net;
    net.push_back(relu_backward(relu_prim_desc, src->GetOpMem(),
                                diff_dst->GetOpMem(), diff_src->GetOpMem()));
    stream(stream::kind::eager).submit(net).wait();
  }
};


template <typename Device, typename T>
class MklReluOp : public MklReluOpBase<Device, T, eltwise_relu> {
 public:
  ~MklReluOp() {}

  explicit MklReluOp(OpKernelConstruction* context) :
  MklReluOpBase<Device, T, eltwise_relu>(context) {}

  virtual void Compute_Scalar(OpKernelContext* context) {
    const size_t src_index = 0;  // index of src input tensor
    const size_t dst_index = 0;  // index of dst output tensor
    const Tensor& src_tensor = MklGetInput(context, src_index);
    MklDnnShape dnn_shape_src;
    GetMklShape(context, src_index, &dnn_shape_src);

    Tensor* dst_tensor = nullptr;
    void* user_i = static_cast<void*>(const_cast<T*>(
                         src_tensor.flat<T>().data()));
    MklDnnShape dnn_shape_dst;
    dnn_shape_dst.SetMklTensor(false);
    AllocateOutputSetMklShape(context, dst_index, &dst_tensor,
                              src_tensor.shape(), dnn_shape_dst);
    void* out_o = static_cast<void*>(dst_tensor->flat<T>().data());
    (static_cast<T*>(out_o))[0] =
              std::max((static_cast<T*>(user_i))[0], static_cast<T>(0));
    return;
  }
};

template <typename Device, typename T>
class MklReluGradOp : public MklReluGradOpBase<Device, T, eltwise_relu> {
 public:
  ~MklReluGradOp() {}

  explicit MklReluGradOp(OpKernelConstruction* context) :
  MklReluGradOpBase<Device, T, eltwise_relu>(context) {}

  virtual void Compute_Scalar(OpKernelContext* context) {
    const size_t diff_dst_index = 0;  // index of diff_dst input tensor
    const size_t src_index = 1;       // index of src input tensor
    const size_t diff_src_index = 0;  // index of diff_src output tensor
    const Tensor& src_tensor    = MklGetInput(context, src_index);
    const Tensor& diff_dst_tensor = MklGetInput(context, diff_dst_index);
    Tensor* diff_src_tensor = nullptr;

    MklDnnShape dnn_shape_diff_dst;
    GetMklShape(context, diff_dst_index, &dnn_shape_diff_dst);

    int src_dims_size = src_tensor.dims();
    MklDnnShape dnn_shape_diff_src;
    dnn_shape_diff_src.SetMklTensor(false);
    AllocateOutputSetMklShape(context, diff_src_index, &diff_src_tensor,
                              diff_dst_tensor.shape(), dnn_shape_diff_src);
    void* out_o = static_cast<void*>(diff_src_tensor->flat<T>().data());
    void* user_i =
          static_cast<void*>(const_cast<T*>(src_tensor.flat<T>().data()));
    void* user_g =
          static_cast<void*>(const_cast<T*>(diff_dst_tensor.flat<T>().data()));
    (static_cast<T*>(out_o))[0] = (static_cast<T*>(user_g))[0] *
                                  ((static_cast<T*>(user_i))[0] > 0);
    return;
  }
};

template <typename Device, typename T>
class MklEluOp : public MklReluOpBase<Device, T, eltwise_elu> {
 public:
  ~MklEluOp() {}

  explicit MklEluOp(OpKernelConstruction* context) :
  MklReluOpBase<Device, T, eltwise_elu>(context) {}

  virtual void Compute_Scalar(OpKernelContext* context) {
    const size_t src_index = 0;  // index of src input tensor
    const size_t dst_index = 0;  // index of dst output tensor
    const Tensor& src_tensor = MklGetInput(context, src_index);
    MklDnnShape dnn_shape_src;
    GetMklShape(context, src_index, &dnn_shape_src);

    Tensor* dst_tensor = nullptr;
    void* user_i = static_cast<void*>(const_cast<T*>(
                         src_tensor.flat<T>().data()));
    MklDnnShape dnn_shape_dst;
    dnn_shape_dst.SetMklTensor(false);
    AllocateOutputSetMklShape(context, dst_index, &dst_tensor,
                              src_tensor.shape(), dnn_shape_dst);
    void* out_o = static_cast<void*>(dst_tensor->flat<T>().data());
    // return exp(feature) - 1 if feature > 0; feature otherwise
    T feature = (static_cast<T*>(user_i))[0];
    if (feature < 0)
      (static_cast<T*>(out_o))[0] = std::exp(feature);
    else
      (static_cast<T*>(out_o))[0] = feature;
    return;
  }
};

template <typename Device, typename T>
class MklEluGradOp : public MklReluGradOpBase<Device, T, eltwise_elu> {
 public:
  ~MklEluGradOp() {}

  explicit MklEluGradOp(OpKernelConstruction* context) :
  MklReluGradOpBase<Device, T, eltwise_elu>(context) {}

  virtual void Compute_Scalar(OpKernelContext* context) {
    const size_t diff_dst_index = 0;  // index of diff_dst input tensor
    const size_t src_index = 1;       // index of src input tensor
    const size_t diff_src_index = 0;  // index of diff_src output tensor
    const Tensor& src_tensor    = MklGetInput(context, src_index);
    const Tensor& diff_dst_tensor = MklGetInput(context, diff_dst_index);
    Tensor* diff_src_tensor = nullptr;

    MklDnnShape dnn_shape_diff_dst;
    GetMklShape(context, diff_dst_index, &dnn_shape_diff_dst);

    int src_dims_size = src_tensor.dims();
    MklDnnShape dnn_shape_diff_src;
    dnn_shape_diff_src.SetMklTensor(false);
    AllocateOutputSetMklShape(context, diff_src_index, &diff_src_tensor,
                              diff_dst_tensor.shape(), dnn_shape_diff_src);
    void* out_o = static_cast<void*>(diff_src_tensor->flat<T>().data());
    void* user_i =
          static_cast<void*>(const_cast<T*>(src_tensor.flat<T>().data()));
    void* user_g =
          static_cast<void*>(const_cast<T*>(diff_dst_tensor.flat<T>().data()));
    // gradient of elu(x) = 1 if x > 0; elu(x) + 1 otherwise
    T feature = (static_cast<T*>(user_i))[0];
    if (feature > 0) {
      (static_cast<T*>(out_o))[0] = (static_cast<T*>(user_g))[0];
    } else {
      T elu = std::exp(feature) - 1;
      (static_cast<T*>(out_o))[0] = (static_cast<T*>(user_g))[0] * (elu + 1);
    }
  }
};

template <typename Device, typename T>
class MklTanhOp : public MklReluOpBase<Device, T, eltwise_tanh> {
 public:
  ~MklTanhOp() {}

  explicit MklTanhOp(OpKernelConstruction* context) :
  MklReluOpBase<Device, T, eltwise_tanh>(context) {}

  virtual void Compute_Scalar(OpKernelContext* context) {
    const size_t src_index = 0;  // index of src input tensor
    const size_t dst_index = 0;  // index of dst output tensor
    const Tensor& src_tensor = MklGetInput(context, src_index);
    MklDnnShape dnn_shape_src;
    GetMklShape(context, src_index, &dnn_shape_src);

    Tensor* dst_tensor = nullptr;
    void* user_i = static_cast<void*>(const_cast<T*>(
                         src_tensor.flat<T>().data()));
    MklDnnShape dnn_shape_dst;
    dnn_shape_dst.SetMklTensor(false);
    AllocateOutputSetMklShape(context, dst_index, &dst_tensor,
                              src_tensor.shape(), dnn_shape_dst);
    void* out_o = static_cast<void*>(dst_tensor->flat<T>().data());
    // tanh(x) = (e^x - e^(-x))/ (e^x + e^(-x))
    T feature = (static_cast<T*>(user_i))[0];
    T e1 = std::exp(feature);
    T e2 = std::exp(-feature);
    (static_cast<T*>(out_o))[0] = (e1 - e2)/(e1 + e2);
    return;
  }
};

template <typename Device, typename T>
class MklTanhGradOp : public MklReluGradOpBase<Device, T, eltwise_tanh> {
 public:
  ~MklTanhGradOp() {}

  explicit MklTanhGradOp(OpKernelConstruction* context) :
  MklReluGradOpBase<Device, T, eltwise_tanh>(context) {}

  virtual void Compute_Scalar(OpKernelContext* context) {
    const size_t diff_dst_index = 0;  // index of diff_dst input tensor
    const size_t src_index = 1;       // index of src input tensor
    const size_t diff_src_index = 0;  // index of diff_src output tensor
    const Tensor& src_tensor    = MklGetInput(context, src_index);
    const Tensor& diff_dst_tensor = MklGetInput(context, diff_dst_index);
    Tensor* diff_src_tensor = nullptr;

    MklDnnShape dnn_shape_diff_dst;
    GetMklShape(context, diff_dst_index, &dnn_shape_diff_dst);

    int src_dims_size = src_tensor.dims();
    MklDnnShape dnn_shape_diff_src;
    dnn_shape_diff_src.SetMklTensor(false);
    AllocateOutputSetMklShape(context, diff_src_index, &diff_src_tensor,
                              diff_dst_tensor.shape(), dnn_shape_diff_src);
    void* out_o = static_cast<void*>(diff_src_tensor->flat<T>().data());
    void* user_i =
          static_cast<void*>(const_cast<T*>(src_tensor.flat<T>().data()));
    // gradient of tanh(x) = 1 - tanh(x)^2
    T feature = (static_cast<T*>(user_i))[0];
    T e1 = std::exp(feature);
    T e2 = std::exp(-feature);
    T tanh = (e1 - e2)/(e1 + e2);
    void* user_g =
          static_cast<void*>(const_cast<T*>(diff_dst_tensor.flat<T>().data()));
    (static_cast<T*>(out_o))[0] = (static_cast<T*>(user_g))[0] *
                                  (1 - tanh * tanh);
  }
};

#endif

// register dnn kernels for supported operations and supported types
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

#ifdef INTEL_MKL_DNN

// register dnn kernels for supported operations and supported types
#define REGISTER_ELU_MKL_SUPPORTED_KERNELS_TYPES(type)             \
  REGISTER_KERNEL_BUILDER(Name("_MklElu")                          \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<type>("T")            \
                              .Label(mkl_op_registry::kMklOpLabel), \
                          MklEluOp<CPUDevice, type>);              \
  REGISTER_KERNEL_BUILDER(Name("_MklEluGrad")                      \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<type>("T")            \
                              .Label(mkl_op_registry::kMklOpLabel), \
                          MklEluGradOp<CPUDevice, type>);
TF_CALL_float(REGISTER_ELU_MKL_SUPPORTED_KERNELS_TYPES);

#define REGISTER_TANH_MKL_SUPPORTED_KERNELS_TYPES(type)             \
  REGISTER_KERNEL_BUILDER(Name("_MklTanh")                          \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<type>("T")            \
                              .Label(mkl_op_registry::kMklOpLabel), \
                          MklTanhOp<CPUDevice, type>);              \
  REGISTER_KERNEL_BUILDER(Name("_MklTanhGrad")                      \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<type>("T")            \
                              .Label(mkl_op_registry::kMklOpLabel), \
                          MklTanhGradOp<CPUDevice, type>);
TF_CALL_float(REGISTER_TANH_MKL_SUPPORTED_KERNELS_TYPES);

#endif

}  // namespace tensorflow

#endif  // INTEL_MKL

