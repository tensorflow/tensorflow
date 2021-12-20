/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_MKL_MKL_ELTWISE_ACTIVATION_BASE_OP_H_
#define TENSORFLOW_CORE_KERNELS_MKL_MKL_ELTWISE_ACTIVATION_BASE_OP_H_

// See docs in ../ops/mkl_nn_ops.cc.

#ifdef INTEL_MKL

#include <unordered_map>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "dnnl.hpp"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/mkl_util.h"

using dnnl::algorithm;
using dnnl::eltwise_forward;
using dnnl::memory;
using dnnl::prop_kind;
using dnnl::stream;

using EltwiseFwdPd = dnnl::eltwise_forward::primitive_desc;

namespace tensorflow {

template <typename T>
class MklEltwiseFwdParams {
 public:
  memory::dims src_dims;
  memory::desc src_md;
  algorithm alg_kind;
  float alpha;
  float beta;

  MklEltwiseFwdParams(memory::dims src_dims, memory::desc src_md,
                      algorithm alg_kind, float alpha, float beta)
      : src_dims(src_dims),
        src_md(src_md),
        alg_kind(alg_kind),
        alpha(alpha),
        beta(beta) {}
};

template <typename T>
class MklEltwiseFwdPrimitive : public MklPrimitive {
 public:
  explicit MklEltwiseFwdPrimitive(const MklEltwiseFwdParams<T>& fwdParams)
      : MklPrimitive(engine(engine::kind::cpu, 0)) {
    // create eltwise primitive
    if (context_.eltwise_fwd == nullptr) {
      Setup(fwdParams);
    }
  }

  ~MklEltwiseFwdPrimitive() {}

  // Eltwise forward execute
  //   src_data:  input data buffer of src
  //   dst_data:  output data buffer of dst
  void Execute(const T* src_data, T* dst_data, OpKernelContext* op_context) {
    context_.src_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(src_data)));
    context_.dst_mem->set_data_handle(static_cast<void*>(dst_data));
    DCHECK_EQ(context_.fwd_primitives.size(),
              context_.fwd_primitives_args.size());

    std::vector<primitive> net;
    net.push_back(eltwise_forward(*context_.fwd_pd));
    std::vector<MemoryArgsMap> net_args;
    net_args.push_back(
        {{DNNL_ARG_SRC, *context_.src_mem}, {DNNL_ARG_DST, *context_.dst_mem}});
    // execute eltwise_fwd primitve
    ExecutePrimitive(net, &net_args, GetEngine(), op_context);

    // After execution, set data handle back.
    context_.src_mem->set_data_handle(DummyData);
    context_.dst_mem->set_data_handle(DummyData);
  }

  std::shared_ptr<EltwiseFwdPd> GetEltwiseFwdPd() { return context_.fwd_pd; }

 private:
  // Primitive reuse context for eltwise Fwd ops: Relu, Elu, Tanh
  struct EltwiseFwdContext {
    // oneDNN memory
    std::shared_ptr<memory> src_mem;
    std::shared_ptr<memory> dst_mem;

    // desc & primitive desc
    std::shared_ptr<dnnl::eltwise_forward::desc> fwd_desc;
    std::shared_ptr<EltwiseFwdPd> fwd_pd;

    // memory desc
    std::shared_ptr<memory::desc> src_md;
    std::shared_ptr<memory::desc> dst_md;

    // memory primitive desc
    std::shared_ptr<memory::desc> src_mpd;

    // Eltwise primitive
    std::shared_ptr<dnnl::primitive> eltwise_fwd;

    std::vector<dnnl::primitive> fwd_primitives;

    std::vector<std::unordered_map<int, memory>> fwd_primitives_args;

    EltwiseFwdContext()
        : src_mem(nullptr),
          dst_mem(nullptr),
          fwd_desc(nullptr),
          fwd_pd(nullptr),
          src_md(nullptr),
          dst_md(nullptr),
          src_mpd(nullptr),
          eltwise_fwd(nullptr) {}
  };

  // Eltwise forward primitive setup
  void Setup(const MklEltwiseFwdParams<T>& fwdParams) {
    // create memory descriptors for eltwise data with specified format
    context_.src_md.reset(new memory::desc(fwdParams.src_md.data));
    context_.src_mpd.reset(new memory::desc(*context_.src_md));

    // Create an eltwise forward descriptor and primitive descriptor
    context_.fwd_desc.reset(new eltwise_forward::desc(
        prop_kind::forward, fwdParams.alg_kind, *context_.src_md,
        fwdParams.alpha, fwdParams.beta));
    context_.fwd_pd.reset(new EltwiseFwdPd(*context_.fwd_desc, cpu_engine_));
    auto fwd_pd = context_.fwd_pd.get();

    // Create memory primitive based on dummy data
    context_.src_mem.reset(
        new memory(fwd_pd->src_desc(), cpu_engine_, DummyData));
    context_.dst_mem.reset(
        new memory(fwd_pd->dst_desc(), cpu_engine_, DummyData));
    // Create eltwise primitive and add it to net
    context_.eltwise_fwd.reset(new eltwise_forward(*context_.fwd_pd));
    context_.fwd_primitives_args.push_back(
        {{DNNL_ARG_SRC, *context_.src_mem}, {DNNL_ARG_DST, *context_.dst_mem}});
    context_.fwd_primitives.push_back(*context_.eltwise_fwd);
  }

  struct EltwiseFwdContext context_;
};

template <typename T>
class MklEltwiseFwdPrimitiveFactory : public MklPrimitiveFactory<T> {
 public:
  static MklEltwiseFwdPrimitive<T>* Get(
      const MklEltwiseFwdParams<T>& fwdParams) {
    MklEltwiseFwdPrimitive<T>* eltwise_forward = nullptr;

    // Get a eltwise fwd primitive from the cached pool
    eltwise_forward = static_cast<MklEltwiseFwdPrimitive<T>*>(
        MklEltwiseFwdPrimitiveFactory<T>::GetInstance().GetEltwiseFwd(
            fwdParams));
    if (eltwise_forward == nullptr) {
      eltwise_forward = new MklEltwiseFwdPrimitive<T>(fwdParams);
      MklEltwiseFwdPrimitiveFactory<T>::GetInstance().SetEltwiseFwd(
          fwdParams, eltwise_forward);
    }

    return eltwise_forward;
  }

  static MklEltwiseFwdPrimitiveFactory& GetInstance() {
    static MklEltwiseFwdPrimitiveFactory instance_;
    return instance_;
  }

 private:
  MklEltwiseFwdPrimitiveFactory() {}
  ~MklEltwiseFwdPrimitiveFactory() {}

  static string CreateKey(const MklEltwiseFwdParams<T>& fwdParams) {
    string prefix = "eltwise_fwd";
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(prefix);
    key_creator.AddAsKey(fwdParams.src_dims);
    key_creator.AddAsKey<int>(static_cast<int>(fwdParams.alg_kind));
    key_creator.AddAsKey<float>(static_cast<float>(fwdParams.alpha));
    key_creator.AddAsKey<float>(static_cast<float>(fwdParams.beta));
    return key_creator.GetKey();
  }

  MklPrimitive* GetEltwiseFwd(const MklEltwiseFwdParams<T>& fwdParams) {
    string key = CreateKey(fwdParams);
    return this->GetOp(key);
  }

  void SetEltwiseFwd(const MklEltwiseFwdParams<T>& fwdParams,
                     MklPrimitive* op) {
    string key = CreateKey(fwdParams);
    this->SetOp(key, op);
  }
};

template <typename Device, typename T, algorithm alg_kind>
class MklEltwiseFwdActivationOpBase : public OpKernel {
 public:
  ~MklEltwiseFwdActivationOpBase() {}

  explicit MklEltwiseFwdActivationOpBase(OpKernelConstruction* context,
                                         float alpha, float beta)
      : OpKernel(context), alpha_(alpha), beta_(beta) {}
  virtual void Compute_Scalar(OpKernelContext* context) = 0;

  void Compute(OpKernelContext* context) override {
    try {
      const Tensor& src_tensor = context->input(0);
      TensorShape src_shape = src_tensor.shape();
      if (src_tensor.dims() == 0) {
        Compute_Scalar(context);
        return;
      }
      // Allocate output (dst) tensor
      TensorShape dst_shape = src_shape;
      Tensor* dst_tensor = nullptr;
      // Nothing to compute, return.
      if (src_shape.num_elements() == 0) {
        OP_REQUIRES_OK(context,
                       context->allocate_output(
                           GetTensorDataIndex(0, context->num_outputs()),
                           dst_shape, &dst_tensor));
        return;
      }
      // Set DNN primitive - src
      MklDnnData<T> src(&cpu_engine);
      memory::dims src_dims;
      memory::desc src_md({}, memory::data_type::undef,
                          memory::format_tag::undef);

      src_dims = TFShapeToMklDnnDims(src_tensor.shape());
      auto src_strides = CalculateTFStrides(src_dims);

      // Create blocked memory descriptor
      src_md = MklDnnData<T>::CreateBlockedMemDesc(src_dims, src_strides);

      // Try to get an eltwise forward primitive from caching pool
      MklEltwiseFwdParams<T> fwdParams(src_dims, src_md, alg_kind, alpha_,
                                       beta_);
      MklEltwiseFwdPrimitive<T>* eltwise_fwd =
          MklEltwiseFwdPrimitiveFactory<T>::Get(fwdParams);

      const T* src_data = src_tensor.flat<T>().data();

      OP_REQUIRES_OK(context, context->allocate_output(
                                  GetTensorDataIndex(0, context->num_outputs()),
                                  dst_shape, &dst_tensor));

      T* dst_data = dst_tensor->flat<T>().data();
      // execute eltwise
      eltwise_fwd->Execute(src_data, dst_data, context);
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }

 private:
  engine cpu_engine = engine(engine::kind::cpu, 0);

 protected:
  float alpha_;
  float beta_;
};

// TODO : Implement Eltwise bwd / eltwiseGrad class

}  // namespace tensorflow

#endif  // INTEL_MKL
#endif  // TENSORFLOW_CORE_KERNELS_MKL_MKL_ELTWISE_ACTIVATION_BASE_OP_H_
