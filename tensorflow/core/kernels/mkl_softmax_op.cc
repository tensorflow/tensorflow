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
#ifndef INTEL_MKL_ML_ONLY

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/tensor_format.h"

#include "tensorflow/core/util/mkl_util.h"

#include "mkldnn.hpp"
using mkldnn::prop_kind;
using mkldnn::softmax_forward;
using mkldnn::stream;

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class MklSoftmaxOp : public OpKernel {
 public:
  ~MklSoftmaxOp() {}

  explicit MklSoftmaxOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    try {
      auto cpu_engine = engine(engine::cpu, 0);

      // src_tensor now points to the 0-th input of global data struct "context"
      size_t src_idx = 0;
      const Tensor& src_tensor = MklGetInput(context, src_idx);
      const int input_dims = src_tensor.dims();

      // Add: get MklShape
      MklDnnShape src_mkl_shape;
      GetMklShape(context, src_idx, &src_mkl_shape);

      // src_dims is the dimenstion of src_tensor
      // dim of the dst will also be same as src_dims
      auto src_tf_shape = src_mkl_shape.IsMklTensor()
                              ? src_mkl_shape.GetTfShape()
                              : src_tensor.shape();
      auto src_dims = TFShapeToMklDnnDims(src_tf_shape);
      auto output_dims = src_dims;
      memory::format layout_type;
      // In MKL, data format passed to mkl softmax op depends on dimension of the input tensor.
      // Here "x" data format in MKL is used for 1 dim tensor, "nc" for 2 dim tensor, 
      // "tnc" for 3 dim tensor, "nchw" for 4 dim tensor, and "ncdhw" for 5 dim tensor.
      // Each of the simbols has the following meaning:
      // n = batch, c = channels, t = sequence lenght, h = height,
      // w = width, d = depth 
      switch (input_dims) {
        case 1:
          layout_type = memory::format::x;
          break;
        case 2:
          layout_type = memory::format::nc;
          break;
        case 3:
          layout_type = memory::format::tnc;
          break;
        case 4:
          layout_type = memory::format::nchw;
          break;
        case 5:
          layout_type = memory::format::ncdhw;
          break;
        default:
          OP_REQUIRES_OK(context, errors::Aborted("Input dims must be <= 5 and >=1"));
          return;
      }
      // Create softmax memory for src, dst: both are defined in mkl_util.h,
      // they are wrapper
      MklDnnData<T> src(&cpu_engine);
      MklDnnData<T> dst(&cpu_engine);

      // If input is in MKL layout, then simply grab input layout; otherwise,
      // construct input Tf layout. For TF layout, although input shape
      // (src_dims) required is in MKL-DNN order, the layout is Tensorflow's
      // layout
      auto src_md =
          src_mkl_shape.IsMklTensor()
              ? src_mkl_shape.GetMklLayout()
              : memory::desc(src_dims, MklDnnType<T>(), layout_type);

      // src: setting memory descriptor and op memory descriptor
      // Basically following two functions maps the TF "src_tensor" to mkl
      // tensor object "src"
      // following functions are in mkl_util.h
      // data format is "nc" for src and dst; since the src and dst buffer is
      // always in 2D shape
      src.SetUsrMem(src_md, &src_tensor);
      src.SetOpMemDesc(src_dims, layout_type);

      // creating a memory descriptor
      // passing outermost dim as default axis, where the softmax is applied
      int axis = input_dims - 1;
      auto softmax_fwd_desc = softmax_forward::desc(prop_kind::forward_scoring,
                                                    src.GetOpMemDesc(), axis);
      auto softmax_fwd_pd =
          softmax_forward::primitive_desc(softmax_fwd_desc, cpu_engine);

      // add: output
      Tensor* output_tensor = nullptr;
      MklDnnShape output_mkl_shape;
      TensorShape output_tf_shape;  // shape of output TF tensor.
      // Softmax MklDnn output layout is same as input layout.
      auto dst_pd = src.GetUsrMemPrimDesc();

      // if input is MKL shape, output is also MKL shape.
      // if input is TF shape, output is also TF shape
      if (src_mkl_shape.IsMklTensor()) {
        output_mkl_shape.SetMklTensor(true);
        output_mkl_shape.SetMklLayout(&dst_pd);
        output_mkl_shape.SetElemType(MklDnnType<T>());
        output_mkl_shape.SetTfLayout(output_dims.size(), output_dims,
                                     layout_type);
        output_tf_shape.AddDim((dst_pd.get_size() / sizeof(T)));
      } else {  // then output is also TF shape
        output_mkl_shape.SetMklTensor(false);
        output_tf_shape = MklDnnDimsToTFShape(output_dims);
      }
      // Allocate output shape (MKL or TF based on the above)
      AllocateOutputSetMklShape(context, 0, &output_tensor, output_tf_shape,
                                output_mkl_shape);

      // Output_dims and input_dims are same
      dst.SetUsrMem(src_md, output_tensor);

      // finally creating the "softmax op" using the primitive descriptor, src
      // and dst
      auto softmax_fwd =
          softmax_forward(softmax_fwd_pd, src.GetOpMem(), dst.GetOpMem());

      // execute net (pushing to the stream)
      // following 3 are common for all mkl dnn ops
      std::vector<primitive> net;
      net.push_back(softmax_fwd);
      stream(stream::kind::eager).submit(net).wait();
    } catch (mkldnn::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }
};

/* Register DNN kernels for supported operations and supported types - right now
 * it is only Softmax and f32 */
#define REGISTER_SOFTMAX_MKL_SUPPORTED_KERNELS_TYPES(type)          \
  REGISTER_KERNEL_BUILDER(Name("_MklSoftmax")                       \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<type>("T")            \
                              .Label(mkl_op_registry::kMklOpLabel), \
                          MklSoftmaxOp<CPUDevice, type>);
TF_CALL_float(REGISTER_SOFTMAX_MKL_SUPPORTED_KERNELS_TYPES);

}  // namespace tensorflow

#endif  // INTEL_MKL_ML_ONLY
#endif  // INTEL_MKL
