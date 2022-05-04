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

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "dnnl.hpp"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/util/tensor_format.h"
#ifdef DNNL_AARCH64_USE_ACL
#include "tensorflow/core/platform/mutex.h"
#endif

using dnnl::prop_kind;
using dnnl::softmax_forward;
using dnnl::stream;

namespace tensorflow {

class MklSoftmaxParams {
 public:
  memory::dims src_dims;
  MklTensorFormat src_fmt;
  int axis;
#ifdef DNNL_AARCH64_USE_ACL
  int aarch64_counter;
#endif
  MklSoftmaxParams(memory::dims src_dims, MklTensorFormat src_fmt, int axis)
      : src_dims(src_dims), src_fmt(src_fmt), axis(axis) {}
};

template <typename T>
class MklSoftmaxPrimitive : public MklPrimitive {
 public:
  explicit MklSoftmaxPrimitive(const MklSoftmaxParams& fwdParams)
      : MklPrimitive(engine(engine::kind::cpu, 0)) {
    Setup(fwdParams);
  }

  ~MklSoftmaxPrimitive() {}

  // Softmax forward execute
  //   src_data:  input data buffer of src
  //   dst_data:  output data buffer of dst
  void Execute(const T* src_data, T* dst_data,
               std::shared_ptr<stream> fwd_cpu_stream) {
#ifdef DNNL_AARCH64_USE_ACL
    mutex_lock lock(mu_);
#endif
#ifndef ENABLE_ONEDNN_OPENMP
    context_.src_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(src_data)), *fwd_cpu_stream);
    context_.dst_mem->set_data_handle(static_cast<void*>(dst_data),
                                      *fwd_cpu_stream);
#else
    context_.src_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(src_data)));
    context_.dst_mem->set_data_handle(static_cast<void*>(dst_data));
#endif  // !ENABLE_ONEDNN_OPENMP

    DCHECK_EQ(context_.fwd_primitives.size(), context_.fwd_net_args.size());
    execute_primitives(context_.fwd_primitives, fwd_cpu_stream,
                       context_.fwd_net_args);

    // After execution, set data handle back.
    context_.src_mem->set_data_handle(DummyData);
    context_.dst_mem->set_data_handle(DummyData);
  }

  std::shared_ptr<dnnl::softmax_forward::primitive_desc> GetSoftmaxFwdPd() {
    return context_.fwd_pd;
  }

 private:
  struct SoftmaxFwdContext {
    // MKL-DNN memory.
    std::shared_ptr<memory> src_mem;
    std::shared_ptr<memory> dst_mem;

    // Primitive descriptor.
    std::shared_ptr<dnnl::softmax_forward::desc> fwd_desc;

    // Memory descriptor.
    std::shared_ptr<memory::desc> src_md;

    // Softmax primitive.
    std::shared_ptr<dnnl::softmax_forward::primitive_desc> fwd_pd;
    std::shared_ptr<dnnl::primitive> softmax_fwd;

    std::vector<dnnl::primitive> fwd_primitives;
    std::vector<MemoryArgsMap> fwd_net_args;

    SoftmaxFwdContext()
        : src_mem(nullptr),
          dst_mem(nullptr),
          fwd_desc(nullptr),
          src_md(nullptr),
          fwd_pd(nullptr),
          softmax_fwd(nullptr) {}
  };

  // Softmax forward primitive setup
  void Setup(const MklSoftmaxParams& fwdParams) {
    // Create memory descriptors for softmax data with specified format.
    auto src_format = MklTensorFormatToMklDnnDataFormat(fwdParams.src_fmt);
    context_.src_md.reset(
        new memory::desc({fwdParams.src_dims}, MklDnnType<T>(), src_format));

    // Create softmax descriptor and primitive descriptor.
    context_.fwd_desc.reset(new dnnl::softmax_forward::desc(
        prop_kind::forward_scoring, *context_.src_md, fwdParams.axis));
    context_.fwd_pd.reset(new dnnl::softmax_forward::primitive_desc(
        *context_.fwd_desc, cpu_engine_));

    // Create memory primitive based on dummy data.
    context_.src_mem.reset(
        new memory(*context_.src_md, cpu_engine_, DummyData));
    context_.dst_mem.reset(
        new memory(context_.fwd_pd.get()->dst_desc(), cpu_engine_, DummyData));

    // Create softmax primitive and add it to net
    context_.softmax_fwd.reset(new dnnl::softmax_forward(*context_.fwd_pd));
    context_.fwd_net_args.push_back(
        {{DNNL_ARG_SRC, *context_.src_mem}, {DNNL_ARG_DST, *context_.dst_mem}});

    context_.fwd_primitives.push_back(*context_.softmax_fwd);
  }

  struct SoftmaxFwdContext context_;

#ifdef DNNL_AARCH64_USE_ACL
  mutex mu_;
#endif
};

template <typename T>
class MklSoftmaxPrimitiveFactory : public MklPrimitiveFactory<T> {
 public:
  static MklSoftmaxPrimitive<T>* Get(const MklSoftmaxParams& fwdParams) {
    // Get a softmax fwd primitive from the cached pool.
    MklSoftmaxPrimitive<T>* softmax_forward =
        static_cast<MklSoftmaxPrimitive<T>*>(
            MklSoftmaxPrimitiveFactory<T>::GetInstance().GetSoftmaxFwd(
                fwdParams));
    if (softmax_forward == nullptr) {
      softmax_forward = new MklSoftmaxPrimitive<T>(fwdParams);
      MklSoftmaxPrimitiveFactory<T>::GetInstance().SetSoftmaxFwd(
          fwdParams, softmax_forward);
    }
    return softmax_forward;
  }

  static MklSoftmaxPrimitiveFactory& GetInstance() {
    static MklSoftmaxPrimitiveFactory instance_;
    return instance_;
  }

#ifdef DNNL_AARCH64_USE_ACL
  static int IncrementCounter() {
    static std::atomic_int counter{1};
    return counter.fetch_add(1);
  }
#endif

 private:
  MklSoftmaxPrimitiveFactory() {}
  ~MklSoftmaxPrimitiveFactory() {}

  static string CreateKey(const MklSoftmaxParams& fwdParams) {
    string prefix = "softmax_fwd";
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(prefix);
    key_creator.AddAsKey(fwdParams.src_dims);
    key_creator.AddAsKey<int>(static_cast<int>(fwdParams.src_fmt));
    key_creator.AddAsKey<int>(fwdParams.axis);
#ifdef DNNL_AARCH64_USE_ACL
    key_creator.AddAsKey(fwdParams.aarch64_counter);
#endif
    return key_creator.GetKey();
  }

  MklPrimitive* GetSoftmaxFwd(const MklSoftmaxParams& fwdParams) {
    string key = CreateKey(fwdParams);
    return this->GetOp(key);
  }

  void SetSoftmaxFwd(const MklSoftmaxParams& fwdParams, MklPrimitive* op) {
    string key = CreateKey(fwdParams);
    this->SetOp(key, op);
  }
};

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class MklSoftmaxOp : public OpKernel {
 public:
  ~MklSoftmaxOp() {}

  explicit MklSoftmaxOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    try {
      auto cpu_engine = engine(engine::kind::cpu, 0);
      // src_tensor points to the 0-th input of global data struct "context".
      size_t src_idx = 0;
      const Tensor& src_tensor = MklGetInput(context, src_idx);
      MklDnnShape src_mkl_shape;
      GetMklShape(context, src_idx, &src_mkl_shape);

      // src_dims is the dimension of src_tensor.
      // Dim of the dst will also be same as src_dims.
      auto src_tf_shape = src_mkl_shape.IsMklTensor()
                              ? src_mkl_shape.GetTfShape()
                              : src_tensor.shape();
      const int input_dims = src_tf_shape.dims();
      memory::dims src_dims;
      int axis;
      if (src_mkl_shape.IsMklTensor()) {
        src_dims = src_mkl_shape.GetSizesAsMklDnnDims();
        axis = 1;
      } else {
        src_dims = TFShapeToMklDnnDims(src_tf_shape);
        axis = input_dims - 1;
      }
      MklTensorFormat layout_type;
      // In MKL, data format passed to mkl softmax op depends on dimension of
      // the input tensor. Here "x" data format in MKL is used for 1 dim tensor,
      // "nc" for 2 dim tensor, "tnc" for 3 dim tensor, "nchw" for 4 dim tensor,
      // and "ncdhw" for 5 dim tensor. Each of the symbols has the following
      // meaning: n = batch, c = channels, t = sequence length, h = height, w =
      // width, d = depth. When src tensor is MKL, layout_type here is only used
      // for setting TF layout type of output tensor. When input is TF Tensor,
      // layout here is no special sense. We use axis to define on which
      // dimension to do softmax.
      switch (input_dims) {
        case 1:
          layout_type = MklTensorFormat::FORMAT_X;
          break;
        case 2:
          layout_type = MklTensorFormat::FORMAT_NC;
          break;
        case 3:
          layout_type = MklTensorFormat::FORMAT_TNC;
          break;
        case 4:
          if (src_mkl_shape.IsMklTensor()) {
            layout_type = MklTensorFormat::FORMAT_NHWC;
          } else {
            layout_type = MklTensorFormat::FORMAT_NCHW;
          }
          break;
        case 5:
          if (src_mkl_shape.IsMklTensor()) {
            layout_type = MklTensorFormat::FORMAT_NDHWC;
          } else {
            layout_type = MklTensorFormat::FORMAT_NCDHW;
          }
          break;
        default:
          OP_REQUIRES_OK(context,
                         errors::Aborted("Input dims must be <= 5 and >=1"));
          return;
      }

      // If input is in MKL layout, then simply get the format from input;
      // otherwise, use TF layout defined before.
      auto src_fmt = src_mkl_shape.IsMklTensor()
                         ? MklTensorFormat::FORMAT_BLOCKED
                         : layout_type;

      // Get a softmax fwd primitive from primitive pool.
      MklSoftmaxParams fwdParams(src_dims, src_fmt, axis);
#ifdef DNNL_AARCH64_USE_ACL
      // ACL does not support reuse of primitives with different data.
      // For softmax, the previous approach (PR #47775) of using Tensor
      // addresses does not work, as the addresses are re-used in matmul with
      // different data The counter ensures we still benefit from caching via
      // SetSoftmaxFwd().
      fwdParams.aarch64_counter =
          MklSoftmaxPrimitiveFactory<T>::IncrementCounter();
#endif
      MklSoftmaxPrimitive<T>* softmax_fwd =
          MklSoftmaxPrimitiveFactory<T>::Get(fwdParams);

      // Prepare for creating output tensor.
      Tensor* output_tensor = nullptr;
      MklDnnShape output_mkl_shape;
      TensorShape output_tf_shape;  // shape of output TF tensor.

      auto dst_pd = softmax_fwd->GetSoftmaxFwdPd()->dst_desc();

      // If input is MKL shape, output is also MKL shape.
      // If input is TF shape, output is also TF shape.
      if (src_mkl_shape.IsMklTensor()) {
        output_mkl_shape.SetMklTensor(true);
        output_mkl_shape.SetMklLayout(&dst_pd);
        output_mkl_shape.SetElemType(MklDnnType<T>());
        output_mkl_shape.SetTfLayout(src_dims.size(), src_dims, layout_type);
        output_tf_shape.AddDim((dst_pd.get_size() / sizeof(T)));
      } else {
        output_mkl_shape.SetMklTensor(false);
        output_tf_shape = MklDnnDimsToTFShape(src_dims);
      }
      // Allocate output tensor.
      AllocateOutputSetMklShape(context, 0, &output_tensor, output_tf_shape,
                                output_mkl_shape);

      const T* src_data = src_tensor.flat<T>().data();
      T* dst_data = reinterpret_cast<T*>(output_tensor->flat<T>().data());
      std::shared_ptr<stream> fwd_cpu_stream;
      MklDnnThreadPool eigen_tp(context);
      fwd_cpu_stream.reset(CreateStream(&eigen_tp, softmax_fwd->GetEngine()));
      softmax_fwd->Execute(src_data, dst_data, fwd_cpu_stream);
    } catch (dnnl::error& e) {
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
#define REGISTER_SOFTMAX_MKL_SUPPORTED_KERNELS_TYPES(type)     \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklSoftmax")                                      \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklSoftmaxOp<CPUDevice, type>);

TF_CALL_float(REGISTER_SOFTMAX_MKL_SUPPORTED_KERNELS_TYPES);
TF_CALL_bfloat16(REGISTER_SOFTMAX_MKL_SUPPORTED_KERNELS_TYPES);

}  // namespace tensorflow

#endif  // INTEL_MKL
