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

#include <unordered_map>

#include "mkldnn.hpp"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/mkl_types.h"
#include "tensorflow/core/util/mkl_util.h"

using mkldnn::algorithm;
using mkldnn::eltwise_forward;
using mkldnn::memory;
using mkldnn::prop_kind;
using mkldnn::stream;

using EltwiseFwdPd = mkldnn::eltwise_forward::primitive_desc;
using EltwiseBwdPd = mkldnn::eltwise_backward::primitive_desc;

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
      : MklPrimitive(engine(ENGINE_CPU, 0)) {
#ifndef ENABLE_MKLDNN_V1
    context_.src_fmt =
        static_cast<mkldnn::memory::format>(fwdParams.src_md.data.format);
#endif
    // create eltwise primitive
    if (context_.eltwise_fwd == nullptr) {
      Setup(fwdParams);
    }
  }

  ~MklEltwiseFwdPrimitive() {}

  // Eltwise forward execute
  //   src_data:  input data buffer of src
  //   dst_data:  output data buffer of dst
  void Execute(const T* src_data, T* dst_data,
               std::shared_ptr<stream> fwd_stream) {
#ifdef ENABLE_MKLDNN_THREADPOOL
    context_.src_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(src_data)), *fwd_stream);
    context_.dst_mem->set_data_handle(static_cast<void*>(dst_data),
                                      *fwd_stream);
#else
    context_.src_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(src_data)));
    context_.dst_mem->set_data_handle(static_cast<void*>(dst_data));
#endif  // ENABLE_MKLDNN_THREADPOOL
#ifdef ENABLE_MKLDNN_V1
    DCHECK_EQ(context_.fwd_primitives.size(),
              context_.fwd_primitives_args.size());
    execute_primitives(context_.fwd_primitives, fwd_stream,
                       context_.fwd_primitives_args);
#else
    fwd_stream->submit(context_.fwd_primitives);
#endif

    // After execution, set data handle back.
    context_.src_mem->set_data_handle(DummyData);
    context_.dst_mem->set_data_handle(DummyData);
  }

  std::shared_ptr<EltwiseFwdPd> GetEltwiseFwdPd() { return context_.fwd_pd; }

#ifndef ENABLE_MKLDNN_V1
  // In MKL-DNN v1.x, memory format tags only provide a partial description
  // of the memory layout. Hence, these functions are disabled for v1.x.
  memory::format GetSrcMemoryFormat() { return context_.src_fmt; }
#endif

 private:
  // Primitive reuse context for eltwise Fwd ops: Relu, Elu, Tanh
  struct EltwiseFwdContext {
#ifndef ENABLE_MKLDNN_V1
    // Expected memory format for this primitive instance
    mkldnn::memory::format src_fmt;
#endif

    // MKLDNN memory
    std::shared_ptr<memory> src_mem;
    std::shared_ptr<memory> dst_mem;

    // desc & primitive desc
    std::shared_ptr<mkldnn::eltwise_forward::desc> fwd_desc;
    std::shared_ptr<EltwiseFwdPd> fwd_pd;

    // memory desc
    std::shared_ptr<memory::desc> src_md;
    std::shared_ptr<memory::desc> dst_md;

    // memory primitive desc
    std::shared_ptr<MEMORY_PRIMITIVE_DESC> src_mpd;

    // Eltwise primitive
    std::shared_ptr<mkldnn::primitive> eltwise_fwd;

    std::vector<mkldnn::primitive> fwd_primitives;

#ifdef ENABLE_MKLDNN_V1
    std::vector<std::unordered_map<int, memory>> fwd_primitives_args;
#endif

    EltwiseFwdContext()
        :
#ifndef ENABLE_MKLDNN_V1
          src_fmt(memory::format::any),
#endif
          src_mem(nullptr),
          dst_mem(nullptr),
          fwd_desc(nullptr),
          fwd_pd(nullptr),
          src_md(nullptr),
          dst_md(nullptr),
          src_mpd(nullptr),
          eltwise_fwd(nullptr) {
    }
  };

  // Eltwise forward primitive setup
  void Setup(const MklEltwiseFwdParams<T>& fwdParams) {
    // create memory descriptors for eltwise data with specified format
    context_.src_md.reset(new memory::desc(fwdParams.src_md.data));

    context_.src_mpd.reset(
#ifdef ENABLE_MKLDNN_V1
        new MEMORY_PRIMITIVE_DESC(*context_.src_md));
#else
        new MEMORY_PD_CONSTRUCTOR_2_PARAMS(*context_.src_md, cpu_engine_));
#endif
    // Create an eltwise forward descriptor and primitive descriptor
    context_.fwd_desc.reset(new eltwise_forward::desc(
        prop_kind::forward, fwdParams.alg_kind, *context_.src_md,
        fwdParams.alpha, fwdParams.beta));
    context_.fwd_pd.reset(new EltwiseFwdPd(*context_.fwd_desc, cpu_engine_));
    auto fwd_pd = context_.fwd_pd.get();
#ifdef ENABLE_MKLDNN_V1
    // Create memory primitive based on dummy data
    context_.src_mem.reset(new MEMORY_CONSTRUCTOR(fwd_pd->PRIMITIVE_DESC_SRC,
                                                  cpu_engine_, DummyData));
    context_.dst_mem.reset(new MEMORY_CONSTRUCTOR(fwd_pd->PRIMITIVE_DESC_DST,
                                                  cpu_engine_, DummyData));
    // Create eltwise primitive and add it to net
    context_.eltwise_fwd.reset(new eltwise_forward(*context_.fwd_pd));
    context_.fwd_primitives_args.push_back({{MKLDNN_ARG_SRC, *context_.src_mem},
                                            { MKLDNN_ARG_DST,
                                              *context_.dst_mem }});
#else
    context_.src_mem.reset(new memory(*context_.src_mpd, DummyData));
    context_.dst_mem.reset(
        new memory(context_.fwd_pd.get()->dst_primitive_desc(), DummyData));
    context_.eltwise_fwd.reset(new eltwise_forward(
        *context_.fwd_pd, *context_.src_mem, *context_.dst_mem));
#endif
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
#ifndef ENABLE_MKLDNN_V1
    key_creator.AddAsKey<int>(static_cast<int>(fwdParams.src_md.data.format));
#endif  // !ENABLE_MKLDNN_V1
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

template <typename T>
class MklEltwiseBwdParams {
 public:
  memory::dims src_dims;
  memory::desc common_md;
  algorithm alg_kind;
  float alpha;
  float beta;
  // Whether the input that grad op gets from forward op is SRC
  // of forward op or DST of forward op.
  int forward_input_type;

  MklEltwiseBwdParams(const memory::dims& src_dims,
                      const memory::desc& common_md, algorithm alg_kind,
                      float alpha, float beta, int forward_input_type = -1)
      : src_dims(src_dims),
        common_md(common_md),
        alg_kind(alg_kind),
        alpha(alpha),
        beta(beta),
        forward_input_type(forward_input_type) {}
};

template <typename T>
class MklEltwiseBwdPrimitive : public MklPrimitive {
 public:
  explicit MklEltwiseBwdPrimitive(const MklEltwiseBwdParams<T>& bwdParams)
      : MklPrimitive(engine(ENGINE_CPU, 0)) {
#ifndef ENABLE_MKLDNN_V1
    context_.src_fmt =
        static_cast<mkldnn::memory::format>(bwdParams.common_md.data.format);
    context_.diff_dst_fmt =
        static_cast<mkldnn::memory::format>(bwdParams.common_md.data.format);
#endif
    // create eltwise primitive
    if (context_.eltwise_bwd == nullptr) {
      Setup(bwdParams);
    }
  }

  ~MklEltwiseBwdPrimitive() {}

  // Eltwise backward execute
  //   src_data:       input data buffer of src
  //   diff_dst_data:  input data buffer of diff_dst
  //   diff_src_data:  output data buffer of diff_src
  void Execute(const T* src_data, const T* diff_dst_data, T* diff_src_data,
               std::shared_ptr<stream> bwd_stream) {
#ifdef ENABLE_MKLDNN_THREADPOOL
    context_.src_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(src_data)), *bwd_stream);
    context_.diff_dst_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(diff_dst_data)), *bwd_stream);
    context_.diff_src_mem->set_data_handle(static_cast<void*>(diff_src_data),
                                           *bwd_stream);
#else
    context_.src_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(src_data)));
    context_.diff_dst_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(diff_dst_data)));
    context_.diff_src_mem->set_data_handle(static_cast<void*>(diff_src_data));
#endif  // ENABLE_MKLDNN_THREADPOOL
#ifdef ENABLE_MKLDNN_V1
    DCHECK_EQ(context_.bwd_primitives.size(),
              context_.bwd_primitives_args.size());
    execute_primitives(context_.bwd_primitives, bwd_stream,
                       context_.bwd_primitives_args);
#else
    bwd_stream->submit(context_.bwd_primitives);
#endif  // ENABLE_MKLDNN_V1

    // after execution, set data handle back
    context_.src_mem->set_data_handle(DummyData);
    context_.diff_dst_mem->set_data_handle(DummyData);
    context_.diff_src_mem->set_data_handle(DummyData);
  }

  std::shared_ptr<EltwiseBwdPd> GetEltwiseBwdPd() { return context_.bwd_pd; }

#ifndef ENABLE_MKLDNN_V1
  memory::format GetSrcMemoryFormat() { return context_.src_fmt; }
  memory::format GetDiffDstMemoryFormat() { return context_.diff_dst_fmt; }
#endif  // !ENABLE_MKLDNN_V1

 private:
  // Primitive reuse context for eltwise Bwd ops: Relu, Elu, Tanh
  struct EltwiseBwdContext {
#ifndef ENABLE_MKLDNN_V1
    memory::format src_fmt;
    memory::format diff_dst_fmt;
#endif

    // MKLDNN memory
    std::shared_ptr<memory> src_mem;
    std::shared_ptr<memory> diff_dst_mem;
    std::shared_ptr<memory> diff_src_mem;

    // Backward Eltwise descriptor.
    std::shared_ptr<mkldnn::eltwise_backward::desc> bwd_desc;

    // Memory descriptors.
    std::shared_ptr<memory::desc> src_md;
    std::shared_ptr<memory::desc> diff_dst_md;
    std::shared_ptr<memory::desc> common_md;

    // Memory primitive descriptor.
    // TODO(gzmkl): for MKL-DNN 1.0, src_mpd is same as src_md
    //              So it should be removed once MKL-DNN 0.x is cleaned.
    std::shared_ptr<MEMORY_PRIMITIVE_DESC> src_mpd;
    std::shared_ptr<MEMORY_PRIMITIVE_DESC> diff_dst_mpd;

    // Forward and backward descriptors and primitive descriptors.
    std::shared_ptr<mkldnn::eltwise_forward::desc> fwd_desc;
    std::shared_ptr<EltwiseFwdPd> fwd_pd;
    std::shared_ptr<EltwiseBwdPd> bwd_pd;

    // Eltwise primitive.
    std::shared_ptr<mkldnn::primitive> eltwise_bwd;

    std::vector<mkldnn::primitive> bwd_primitives;

#ifdef ENABLE_MKLDNN_V1
    std::vector<MemoryArgsMap> bwd_primitives_args;
#endif  // ENABLE_MKLDNN_V1

    EltwiseBwdContext()
        :
#ifndef ENABLE_MKLDNN_V1
          src_fmt(memory::format::any),
          diff_dst_fmt(memory::format::any),
#endif  // !ENABLE_MKLDNN_V1
          src_mem(nullptr),
          diff_dst_mem(nullptr),
          diff_src_mem(nullptr),
          src_md(nullptr),
          diff_dst_md(nullptr),
          common_md(nullptr),
          src_mpd(nullptr),
          diff_dst_mpd(nullptr),
          fwd_desc(nullptr),
          fwd_pd(nullptr),
          bwd_pd(nullptr),
          eltwise_bwd(nullptr) {
    }
  };

  // Eltwise backward primitive setup
  void Setup(const MklEltwiseBwdParams<T>& bwdParams) {
    // Create memory descriptors for eltwise data w/ no specified format
    context_.src_md.reset(new memory::desc(bwdParams.common_md.data));
    context_.diff_dst_md.reset(new memory::desc(bwdParams.common_md.data));
    context_.src_mpd.reset(
        new MEMORY_PD_CONSTRUCTOR_2_PARAMS(*context_.src_md, cpu_engine_));
    context_.diff_dst_mpd.reset(
        new MEMORY_PD_CONSTRUCTOR_2_PARAMS(*context_.diff_dst_md, cpu_engine_));

    // Create forward eltwise primitive.
    context_.fwd_desc.reset(new mkldnn::eltwise_forward::desc(
        prop_kind::forward_training, bwdParams.alg_kind, *context_.src_md,
        bwdParams.alpha, bwdParams.beta));
    context_.fwd_pd.reset(new EltwiseFwdPd(*context_.fwd_desc, cpu_engine_));
    context_.bwd_desc.reset(new mkldnn::eltwise_backward::desc(
        bwdParams.alg_kind, *context_.diff_dst_md, *context_.src_md,
        bwdParams.alpha, bwdParams.beta));
    context_.bwd_pd.reset(
        new EltwiseBwdPd(*context_.bwd_desc, cpu_engine_, *context_.fwd_pd));

    auto bwd_pd = context_.bwd_pd.get();

#ifdef ENABLE_MKLDNN_V1
    // Create memory primitive based on dummy data.
    context_.src_mem.reset(new MEMORY_CONSTRUCTOR(bwd_pd->PRIMITIVE_DESC_SRC,
                                                  cpu_engine_, DummyData));
    context_.diff_dst_mem.reset(new MEMORY_CONSTRUCTOR(
        bwd_pd->PRIMITIVE_DESC_DIFF_DST, cpu_engine_, DummyData));
    context_.diff_src_mem.reset(new MEMORY_CONSTRUCTOR(
        bwd_pd->PRIMITIVE_DESC_DIFF_SRC, cpu_engine_, DummyData));
    // Create eltwise primitive and add it to net.
    context_.eltwise_bwd.reset(new mkldnn::eltwise_backward(*context_.bwd_pd));
    context_.bwd_primitives_args.push_back(
        {{bwdParams.forward_input_type, *context_.src_mem},
         {MKLDNN_ARG_DIFF_DST, *context_.diff_dst_mem},
         { MKLDNN_ARG_DIFF_SRC,
           *context_.diff_src_mem }});
#else
    context_.src_mem.reset(new memory(*context_.src_mpd, DummyData));
    context_.diff_dst_mem.reset(new memory(*context_.diff_dst_mpd, DummyData));
    context_.diff_src_mem.reset(new memory(
        context_.bwd_pd.get()->diff_src_primitive_desc(), DummyData));
    context_.eltwise_bwd.reset(new mkldnn::eltwise_backward(
        *context_.bwd_pd, *context_.src_mem, *context_.diff_dst_mem,
        *context_.diff_src_mem));
#endif  // ENABLE_MKLDNN_V1

    context_.bwd_primitives.push_back(*context_.eltwise_bwd);
  }

  struct EltwiseBwdContext context_;
};

template <typename T>
class MklEltwiseBwdPrimitiveFactory : public MklPrimitiveFactory<T> {
 private:
  MklEltwiseBwdPrimitiveFactory() {}
  ~MklEltwiseBwdPrimitiveFactory() {}

 public:
  static MklEltwiseBwdPrimitive<T>* Get(
      const MklEltwiseBwdParams<T>& bwdParams) {
    MklEltwiseBwdPrimitive<T>* eltwise_backward = nullptr;

    // try to find a suitable one in pool
    eltwise_backward = static_cast<MklEltwiseBwdPrimitive<T>*>(
        MklEltwiseBwdPrimitiveFactory<T>::GetInstance().GetEltwiseBwd(
            bwdParams));

    if (eltwise_backward == nullptr) {
      eltwise_backward = new MklEltwiseBwdPrimitive<T>(bwdParams);
      MklEltwiseBwdPrimitiveFactory<T>::GetInstance().SetEltwiseBwd(
          bwdParams, eltwise_backward);
    }
    return eltwise_backward;
  }

  static MklEltwiseBwdPrimitiveFactory& GetInstance() {
    static MklEltwiseBwdPrimitiveFactory instance_;
    return instance_;
  }

 private:
  static string CreateKey(const MklEltwiseBwdParams<T>& bwdParams) {
    string prefix = "eltwise_bwd";
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(prefix);
    key_creator.AddAsKey(bwdParams.src_dims);
    key_creator.AddAsKey(static_cast<int>(bwdParams.alg_kind));
    key_creator.AddAsKey(static_cast<float>(bwdParams.alpha));
    key_creator.AddAsKey(static_cast<float>(bwdParams.beta));
#ifndef ENABLE_MKLDNN_V1
    key_creator.AddAsKey(static_cast<int>(bwdParams.common_md.data.format));
#endif  // !ENABLE_MKLDNN_V1
    return key_creator.GetKey();
  }

  MklPrimitive* GetEltwiseBwd(const MklEltwiseBwdParams<T>& bwdParams) {
    string key = CreateKey(bwdParams);
    return this->GetOp(key);
  }

  void SetEltwiseBwd(const MklEltwiseBwdParams<T>& bwdParams,
                     MklPrimitive* op) {
    string key = CreateKey(bwdParams);
    this->SetOp(key, op);
  }
};

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T, algorithm alg_kind>
class MklReluOpBase : public OpKernel {
 public:
  ~MklReluOpBase() {}

  explicit MklReluOpBase(OpKernelConstruction* context, float alpha, float beta)
      : OpKernel(context), alpha_(alpha), beta_(beta) {}
  virtual void Compute_Scalar(OpKernelContext* context) = 0;

  void Compute(OpKernelContext* context) override {
    try {
      const size_t src_index = 0;  // index of src input tensor
      const size_t dst_index = 0;  // index of dst output tensor
      const Tensor& src_tensor = MklGetInput(context, src_index);
      MklDnnShape dnn_shape_src;
      GetMklShape(context, src_index, &dnn_shape_src);
      if (src_tensor.dims() == 0) {
        Compute_Scalar(context);
        return;
      }
      MklDnnShape dnn_shape_dst;
      TensorShape tf_shape_dst;
      Tensor* dst_tensor = nullptr;
      // Nothing to compute, return.
      if (src_tensor.shape().num_elements() == 0) {
        dnn_shape_dst.SetMklTensor(false);
        tf_shape_dst = MklGetInput(context, src_index).shape();
        AllocateOutputSetMklShape(context, dst_index, &dst_tensor, tf_shape_dst,
                                  dnn_shape_dst);
        return;
      }
      // Set DNN primitive - src
      MklDnnData<T> src(&cpu_engine);
      memory::dims src_dims;
      memory::desc src_md({}, MEMORY_DATA_TYPE_UNDEF, MEMORY_FORMAT_UNDEF);
      if (dnn_shape_src.IsMklTensor()) {
        src_md = dnn_shape_src.GetMklLayout();
        src_dims = dnn_shape_src.GetSizesAsMklDnnDims();
      } else {
        src_dims = TFShapeToMklDnnDims(src_tensor.shape());
        auto src_strides = CalculateTFStrides(src_dims);
        // Create blocked memory descriptor
        src_md = MklDnnData<T>::CreateBlockedMemDesc(src_dims, src_strides);
      }
      // Try to get an eltwise forward primitive from caching pool
      MklEltwiseFwdParams<T> fwdParams(src_dims, src_md, alg_kind, alpha_,
                                       beta_);
      MklEltwiseFwdPrimitive<T>* eltwise_fwd =
          MklEltwiseFwdPrimitiveFactory<T>::Get(fwdParams);
      auto eltwise_fwd_pd = eltwise_fwd->GetEltwiseFwdPd();
      std::shared_ptr<stream> fwd_cpu_stream;
      fwd_cpu_stream.reset(CreateStream(context, eltwise_fwd->GetEngine()));
      // Check if src needs to be reordered
      const T* src_data = src_tensor.flat<T>().data();
      if (IS_SRC_REORDER_NEEDED(src_md, eltwise_fwd_pd, eltwise_fwd)) {
        src.SetUsrMem(src_md, &src_tensor);
        src.CheckReorderToOpMem(
            MEMORY_PD_WITHOUT_DATA(eltwise_fwd_pd->PRIMITIVE_DESC_SRC,
                                   cpu_engine),
            context);
        src_data = const_cast<T*>(
            reinterpret_cast<T*>(src.GetOpMem().get_data_handle()));
      }
      // Allocate dst tensor, always set it as MKL-DNN layout
      if (dnn_shape_src.IsMklTensor()) {
        dnn_shape_dst.SetMklTensor(true);
        auto dst_pd = eltwise_fwd_pd->PRIMITIVE_DESC_DST;
        dnn_shape_dst.SetMklLayout(&dst_pd);
        dnn_shape_dst.SetElemType(MklDnnType<T>());
        dnn_shape_dst.SetTfLayout(dnn_shape_src.GetDimension(),
                                  dnn_shape_src.GetSizesAsMklDnnDims(),
                                  dnn_shape_src.GetTfDataFormat());
        tf_shape_dst.AddDim(dst_pd.get_size() / sizeof(T));
      } else {
        dnn_shape_dst.SetMklTensor(false);
        tf_shape_dst = src_tensor.shape();
      }
      OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                  {static_cast<const int>(src_index)},
                                  static_cast<const int>(dst_index),
                                  tf_shape_dst, &dst_tensor));
      AllocateOutputSetMklShape(context, dst_index, dnn_shape_dst);

      T* dst_data = dst_tensor->flat<T>().data();

      // execute eltwise
      eltwise_fwd->Execute(src_data, dst_data, fwd_cpu_stream);
    } catch (mkldnn::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }

 private:
  engine cpu_engine = engine(ENGINE_CPU, 0);
  std::shared_ptr<EltwiseFwdPd> relu_fwd_pd;

 protected:
  float alpha_;
  float beta_;
};

template <typename Device, typename T, algorithm alg_kind>
class MklReluGradOpBase : public OpKernel {
 public:
  ~MklReluGradOpBase() {}

  explicit MklReluGradOpBase(OpKernelConstruction* context, float alpha,
                             float beta)
      : OpKernel(context), alpha_(alpha), beta_(beta) {}

  virtual void Compute_Scalar(OpKernelContext* context) = 0;

  // All activation functions that are part of NN ops, such as Relu, Elu,
  // LeakyRelu, Relu6, etc have dy at index 0 and y at index 1.
  //
  // if forward op is defined as: y = f(x),
  // {Relu,Elu,Relu6,LeakyRelu}Grad is: z = f_grad(dy,x)
  // TanhGrad is: z = tanh_grad(y,dy)
  //
  // Src below refers to a tensor that gradient op receives from forward
  // operator. From Relu-family ops, it is 'x'; while for TanhGrad, it is 'y'.
  virtual int GetDiffDstIndex() const { return 0; }
  virtual int GetSrcIndex() const { return 1; }
  virtual int GetDiffSrcIndex() const { return 0; }
  // What is the type of input tensor that grad op receives from forward op --
  // is it 'x' (SRC) or 'y' (DST). For Relu-family, it is 'x', so fwd op SRC.

#ifdef ENABLE_MKLDNN_V1
  virtual int GetTypeOfInputTensorFromFwdOp() const { return MKLDNN_ARG_SRC; }
#endif

  void Compute(OpKernelContext* context) {
    try {
      MklDnnData<T> src(&cpu_engine);
      MklDnnData<T> diff_dst(&cpu_engine);

      size_t diff_dst_index = GetDiffDstIndex();
      size_t src_index = GetSrcIndex();
      const size_t diff_src_index = GetDiffSrcIndex();

      const Tensor& src_tensor = MklGetInput(context, src_index);
      const Tensor& diff_dst_tensor = MklGetInput(context, diff_dst_index);
      Tensor* diff_src_tensor = nullptr;

      MklDnnShape dnn_shape_src, dnn_shape_diff_dst;
      GetMklShape(context, src_index, &dnn_shape_src);
      GetMklShape(context, diff_dst_index, &dnn_shape_diff_dst);

      int src_dims_size = src_tensor.dims();
      if (src_dims_size == 0) {
        Compute_Scalar(context);
        return;
      }

      TensorShape tf_shape_diff_src;
      MklDnnShape dnn_shape_diff_src;
      // Nothing to compute, return.
      if (src_tensor.shape().num_elements() == 0) {
        dnn_shape_diff_src.SetMklTensor(false);
        tf_shape_diff_src = MklGetInput(context, diff_src_index).shape();
        AllocateOutputSetMklShape(context, diff_src_index, &diff_src_tensor,
                                  tf_shape_diff_src, dnn_shape_diff_src);
        return;
      }

      // get a eltwise bwd from primitive pool
      memory::dims src_dims = {};
      memory::desc src_md({}, MEMORY_DATA_TYPE_UNDEF, MEMORY_FORMAT_UNDEF);
      memory::desc diff_dst_md({}, MEMORY_DATA_TYPE_UNDEF, MEMORY_FORMAT_UNDEF);
      if (!dnn_shape_src.IsMklTensor() && !dnn_shape_diff_dst.IsMklTensor()) {
        src_dims = TFShapeToMklDnnDims(src_tensor.shape());
        auto src_strides = CalculateTFStrides(src_dims);
        src_md = MklDnnData<T>::CreateBlockedMemDesc(src_dims, src_strides);
        diff_dst_md = src_md;
      } else if (dnn_shape_src.IsMklTensor() &&
                 !dnn_shape_diff_dst.IsMklTensor()) {
        src_md = dnn_shape_src.GetMklLayout();
        src_dims = dnn_shape_src.GetSizesAsMklDnnDims();

        MKL_TENSOR_FORMAT src_mkl_data_format = dnn_shape_src.GetTfDataFormat();
        auto src_tf_data_format =
            MklDnnDataFormatToTFDataFormat(src_mkl_data_format);
        auto diff_dst_dims = TFShapeToMklDnnDimsInNCHW(diff_dst_tensor.shape(),
                                                       src_tf_data_format);
        diff_dst_md = memory::desc(diff_dst_dims, MklDnnType<T>(),
                                   GET_TENSOR_FORMAT(src_mkl_data_format));
      } else if (!dnn_shape_src.IsMklTensor() &&
                 dnn_shape_diff_dst.IsMklTensor()) {
        diff_dst_md = dnn_shape_diff_dst.GetMklLayout();

        MKL_TENSOR_FORMAT diff_dst_mkl_data_format =
            dnn_shape_diff_dst.GetTfDataFormat();
        auto diff_dst_tf_data_format =
            MklDnnDataFormatToTFDataFormat(diff_dst_mkl_data_format);

        src_dims = (src_tensor.dims() == 4)
                       ? TFShapeToMklDnnDimsInNCHW(src_tensor.shape(),
                                                   diff_dst_tf_data_format)
                       : TFShapeToMklDnnDimsInNCDHW(src_tensor.shape(),
                                                    diff_dst_tf_data_format);
        src_md = memory::desc(src_dims, MklDnnType<T>(),
                              GET_TENSOR_FORMAT(diff_dst_mkl_data_format));
      } else {
        src_md = dnn_shape_src.GetMklLayout();
        diff_dst_md = dnn_shape_diff_dst.GetMklLayout();
        src_dims = dnn_shape_src.GetSizesAsMklDnnDims();
      }

      // As per comment above, we tell MKLDNN that both the inputs are in same
      // format. So we set common memory descriptor in MKL format, if any of the
      // inputs are in MKL format. Let's get memory descriptor that we will use
      // for both the inputs.
      memory::desc common_md({}, MEMORY_DATA_TYPE_UNDEF, MEMORY_FORMAT_UNDEF);
      if (dnn_shape_src.IsMklTensor() || dnn_shape_diff_dst.IsMklTensor()) {
        common_md = dnn_shape_src.IsMklTensor() ? src_md : diff_dst_md;
      } else {
        // Since both the inputs are in Tensorflow format, and have
        // same shape, we can get memory descriptor from any input.
        common_md = src_md;
      }

#ifdef ENABLE_MKLDNN_V1
      MklEltwiseBwdParams<T> bwdParams(src_dims, common_md, alg_kind, alpha_,
                                       beta_, GetTypeOfInputTensorFromFwdOp());
#else
      // MKLDNN V0 does not support reusing output of forward op in backward.
      // So this optimization works only in MKLDNN v1.
      MklEltwiseBwdParams<T> bwdParams(src_dims, common_md, alg_kind, alpha_,
                                       beta_);
#endif  // ENABLE_MKLDNN_V1

      MklEltwiseBwdPrimitive<T>* eltwise_bwd =
          MklEltwiseBwdPrimitiveFactory<T>::Get(bwdParams);

      auto eltwise_bwd_pd = eltwise_bwd->GetEltwiseBwdPd();
      std::shared_ptr<stream> bwd_cpu_stream;
      bwd_cpu_stream.reset(CreateStream(context, eltwise_bwd->GetEngine()));
      // check whether need reorder for src / diff_dst
      const T* src_data = src_tensor.flat<T>().data();
      if (IS_SRC_REORDER_NEEDED(src_md, eltwise_bwd_pd, eltwise_bwd)) {
        src.SetUsrMem(src_md, &src_tensor);
        src.CheckReorderToOpMem(
            MEMORY_PD_WITHOUT_DATA(
                eltwise_bwd_pd.get()->PRIMITIVE_DESC_DIFF_SRC, cpu_engine),
            context);
        src_data = const_cast<T*>(
            reinterpret_cast<T*>(src.GetOpMem().get_data_handle()));
      }

      const T* diff_dst_data = diff_dst_tensor.flat<T>().data();
      if (IS_DIFF_DST_REORDER_NEEDED(diff_dst_md, eltwise_bwd_pd,
                                     eltwise_bwd)) {
        diff_dst.SetUsrMem(diff_dst_md, &diff_dst_tensor);
        diff_dst.CheckReorderToOpMem(
            MEMORY_PD_WITHOUT_DATA(
                eltwise_bwd_pd.get()->PRIMITIVE_DESC_DIFF_SRC, cpu_engine),
            context);
        diff_dst_data = const_cast<T*>(
            reinterpret_cast<T*>(diff_dst.GetOpMem().get_data_handle()));
      }

      // allocate diff_src tensor
      if (dnn_shape_src.IsMklTensor() || dnn_shape_diff_dst.IsMklTensor()) {
        auto diff_src_pd = eltwise_bwd_pd->PRIMITIVE_DESC_DIFF_SRC;
        dnn_shape_diff_src.SetMklTensor(true);
        dnn_shape_diff_src.SetMklLayout(&diff_src_pd);
        dnn_shape_diff_src.SetElemType(MklDnnType<T>());
        if (dnn_shape_src.IsMklTensor()) {
          dnn_shape_diff_src.SetTfLayout(dnn_shape_src.GetDimension(),
                                         dnn_shape_src.GetSizesAsMklDnnDims(),
                                         dnn_shape_src.GetTfDataFormat());
        } else {
          dnn_shape_diff_src.SetTfLayout(
              dnn_shape_diff_dst.GetDimension(),
              dnn_shape_diff_dst.GetSizesAsMklDnnDims(),
              dnn_shape_diff_dst.GetTfDataFormat());
        }
        tf_shape_diff_src.AddDim(diff_src_pd.get_size() / sizeof(T));
      } else {
        dnn_shape_diff_src.SetMklTensor(false);
        tf_shape_diff_src = src_tensor.shape();
      }

      OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                  {static_cast<const int>(diff_dst_index)},
                                  static_cast<const int>(diff_src_index),
                                  tf_shape_diff_src, &diff_src_tensor));
      AllocateOutputSetMklShape(context, diff_src_index, dnn_shape_diff_src);

      T* diff_src_data = diff_src_tensor->flat<T>().data();

      // execute eltwise bwd
      eltwise_bwd->Execute(src_data, diff_dst_data, diff_src_data,
                           bwd_cpu_stream);
    } catch (mkldnn::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }

 private:
  engine cpu_engine = engine(ENGINE_CPU, 0);
  std::shared_ptr<EltwiseFwdPd> relu_fwd_pd;

 protected:
  float alpha_;
  float beta_;
};

template <typename Device, typename T>
class MklReluOp : public MklReluOpBase<Device, T, ALGORITHM::eltwise_relu> {
 public:
  ~MklReluOp() {}

  explicit MklReluOp(OpKernelConstruction* context)
      : MklReluOpBase<Device, T, ALGORITHM::eltwise_relu>(context, 0.0f, 0.0f) {
  }

  virtual void Compute_Scalar(OpKernelContext* context) {
    const size_t src_index = 0;  // index of src input tensor
    const size_t dst_index = 0;  // index of dst output tensor
    const Tensor& src_tensor = MklGetInput(context, src_index);
    MklDnnShape dnn_shape_src;
    GetMklShape(context, src_index, &dnn_shape_src);

    Tensor* dst_tensor = nullptr;
    void* user_i =
        static_cast<void*>(const_cast<T*>(src_tensor.flat<T>().data()));
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
class MklReluGradOp
    : public MklReluGradOpBase<Device, T, ALGORITHM::eltwise_relu> {
 public:
  ~MklReluGradOp() {}

  explicit MklReluGradOp(OpKernelConstruction* context)
      : MklReluGradOpBase<Device, T, ALGORITHM::eltwise_relu>(context, 0.0f,
                                                              0.0f) {}

  virtual void Compute_Scalar(OpKernelContext* context) {
    const size_t diff_dst_index = 0;  // index of diff_dst input tensor
    const size_t src_index = 1;       // index of src input tensor
    const size_t diff_src_index = 0;  // index of diff_src output tensor
    const Tensor& src_tensor = MklGetInput(context, src_index);
    const Tensor& diff_dst_tensor = MklGetInput(context, diff_dst_index);
    Tensor* diff_src_tensor = nullptr;

    MklDnnShape dnn_shape_diff_dst;
    GetMklShape(context, diff_dst_index, &dnn_shape_diff_dst);

    MklDnnShape dnn_shape_diff_src;
    dnn_shape_diff_src.SetMklTensor(false);
    AllocateOutputSetMklShape(context, diff_src_index, &diff_src_tensor,
                              diff_dst_tensor.shape(), dnn_shape_diff_src);
    void* out_o = static_cast<void*>(diff_src_tensor->flat<T>().data());
    void* user_i =
        static_cast<void*>(const_cast<T*>(src_tensor.flat<T>().data()));
    void* user_g =
        static_cast<void*>(const_cast<T*>(diff_dst_tensor.flat<T>().data()));
    (static_cast<T*>(out_o))[0] =
        (static_cast<T*>(user_g))[0] *
        (static_cast<T>((static_cast<T*>(user_i))[0] > static_cast<T>(0)));
    return;
  }
};

template <typename Device, typename T>
class MklEluOp : public MklReluOpBase<Device, T, ALGORITHM::eltwise_elu> {
 public:
  ~MklEluOp() {}

  explicit MklEluOp(OpKernelConstruction* context)
      : MklReluOpBase<Device, T, ALGORITHM::eltwise_elu>(context, 0.0f, 0.0f) {}

  virtual void Compute_Scalar(OpKernelContext* context) {
    const size_t src_index = 0;  // index of src input tensor
    const size_t dst_index = 0;  // index of dst output tensor
    const Tensor& src_tensor = MklGetInput(context, src_index);
    MklDnnShape dnn_shape_src;
    GetMklShape(context, src_index, &dnn_shape_src);

    Tensor* dst_tensor = nullptr;
    void* user_i =
        static_cast<void*>(const_cast<T*>(src_tensor.flat<T>().data()));
    MklDnnShape dnn_shape_dst;
    dnn_shape_dst.SetMklTensor(false);
    AllocateOutputSetMklShape(context, dst_index, &dst_tensor,
                              src_tensor.shape(), dnn_shape_dst);
    void* out_o = static_cast<void*>(dst_tensor->flat<T>().data());
    // return exp(feature) - 1 if feature > 0; feature otherwise
    T feature = (static_cast<T*>(user_i))[0];
    if (feature < static_cast<T>(0))
      (static_cast<T*>(out_o))[0] = std::exp(feature);
    else
      (static_cast<T*>(out_o))[0] = feature;
    return;
  }
};

template <typename Device, typename T>
class MklEluGradOp
    : public MklReluGradOpBase<Device, T, ALGORITHM::eltwise_elu> {
 public:
  ~MklEluGradOp() {}

  explicit MklEluGradOp(OpKernelConstruction* context)
      : MklReluGradOpBase<Device, T, ALGORITHM::eltwise_elu>(context, 0.0f,
                                                             0.0f) {}

  virtual void Compute_Scalar(OpKernelContext* context) {
    const size_t diff_dst_index = 0;  // index of diff_dst input tensor
    const size_t src_index = 1;       // index of src input tensor
    const size_t diff_src_index = 0;  // index of diff_src output tensor
    const Tensor& src_tensor = MklGetInput(context, src_index);
    const Tensor& diff_dst_tensor = MklGetInput(context, diff_dst_index);
    Tensor* diff_src_tensor = nullptr;

    MklDnnShape dnn_shape_diff_dst;
    GetMklShape(context, diff_dst_index, &dnn_shape_diff_dst);

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
    if (feature > static_cast<T>(0)) {
      (static_cast<T*>(out_o))[0] = (static_cast<T*>(user_g))[0];
    } else {
      T elu = std::exp(feature) - static_cast<T>(1);
      (static_cast<T*>(out_o))[0] =
          (static_cast<T*>(user_g))[0] * (elu + static_cast<T>(1));
    }
  }
};

#ifdef ENABLE_MKLDNN_V1
// Optimized TanhGrad support exists in DNNL1.x only
// (eltwise_tanh_use_dst_for_bwd). We can still support it with DNNL0.x, but
// it will not be optimized. So we disable it for DNNL0.x.

template <typename Device, typename T>
class MklTanhOp : public MklReluOpBase<Device, T, ALGORITHM::eltwise_tanh> {
 public:
  ~MklTanhOp() {}

  explicit MklTanhOp(OpKernelConstruction* context)
      : MklReluOpBase<Device, T, ALGORITHM::eltwise_tanh>(context, 0.0f, 0.0f) {
  }

  virtual void Compute_Scalar(OpKernelContext* context) {
    const size_t src_index = 0;  // index of src input tensor
    const size_t dst_index = 0;  // index of dst output tensor
    const Tensor& src_tensor = MklGetInput(context, src_index);
    MklDnnShape dnn_shape_src;
    GetMklShape(context, src_index, &dnn_shape_src);

    Tensor* dst_tensor = nullptr;
    void* user_i =
        static_cast<void*>(const_cast<T*>(src_tensor.flat<T>().data()));
    MklDnnShape dnn_shape_dst;
    dnn_shape_dst.SetMklTensor(false);
    AllocateOutputSetMklShape(context, dst_index, &dst_tensor,
                              src_tensor.shape(), dnn_shape_dst);
    void* out_o = static_cast<void*>(dst_tensor->flat<T>().data());
    // tanh(x) = (e^x - e^(-x))/ (e^x + e^(-x))
    T feature = (static_cast<T*>(user_i))[0];
    T e1 = std::exp(feature);
    T e2 = std::exp(-feature);
    (static_cast<T*>(out_o))[0] = (e1 - e2) / (e1 + e2);
    return;
  }
};

template <typename Device, typename T>
class MklTanhGradOp
    : public MklReluGradOpBase<Device, T,
                               ALGORITHM::eltwise_tanh_use_dst_for_bwd> {
 public:
  ~MklTanhGradOp() {}

  explicit MklTanhGradOp(OpKernelConstruction* context)
      : MklReluGradOpBase<Device, T, ALGORITHM::eltwise_tanh_use_dst_for_bwd>(
            context, 0.0f, 0.0f) {}

  virtual int GetDiffDstIndex() const { return 1; }
  virtual int GetSrcIndex() const { return 0; }
  virtual int GetDiffSrcIndex() const { return 0; }

  // TanhGrad gets 'y' from Tanh, where 'y' is output of Tanh(x).
  virtual int GetTypeOfInputTensorFromFwdOp() const { return MKLDNN_ARG_DST; }

  virtual void Compute_Scalar(OpKernelContext* context) {
    // NOTE: Order of y and dy for Tanh is reverse of that for Relu/Elu/other
    // element-wise ops. Tanh is math op in Tensorflow; others are NN ops.
    const size_t diff_dst_index = GetDiffDstIndex();
    const size_t src_index = GetSrcIndex();
    const size_t diff_src_index = GetDiffSrcIndex();
    const Tensor& src_tensor = MklGetInput(context, src_index);
    const Tensor& diff_dst_tensor = MklGetInput(context, diff_dst_index);
    Tensor* diff_src_tensor = nullptr;

    MklDnnShape dnn_shape_diff_dst;
    GetMklShape(context, diff_dst_index, &dnn_shape_diff_dst);

    MklDnnShape dnn_shape_diff_src;
    dnn_shape_diff_src.SetMklTensor(false);
    AllocateOutputSetMklShape(context, diff_src_index, &diff_src_tensor,
                              diff_dst_tensor.shape(), dnn_shape_diff_src);
    void* out_o = static_cast<void*>(diff_src_tensor->flat<T>().data());
    void* user_i =
        static_cast<void*>(const_cast<T*>(src_tensor.flat<T>().data()));
    // gradient of tanh(x) = 1 - tanh(x)^2
    // Input to TanhGrad is output of Tanh. So we do not need to compute
    // Tanh again.
    T tanh = (static_cast<T*>(user_i))[0];
    void* user_g =
        static_cast<void*>(const_cast<T*>(diff_dst_tensor.flat<T>().data()));
    (static_cast<T*>(out_o))[0] =
        (static_cast<T*>(user_g))[0] * (static_cast<T>(1) - tanh * tanh);
  }
};
#endif  // ENABLE_MKLDNN_V1

#define RELU6_UPPER_BOUND 6.0f
template <typename Device, typename T>
class MklRelu6Op
    : public MklReluOpBase<Device, T, ALGORITHM::eltwise_bounded_relu> {
 public:
  ~MklRelu6Op() {}

  explicit MklRelu6Op(OpKernelConstruction* context)
      : MklReluOpBase<Device, T, ALGORITHM::eltwise_bounded_relu>(
            context, RELU6_UPPER_BOUND, 0.0f) {}

  virtual void Compute_Scalar(OpKernelContext* context) {
    const size_t src_index = 0;  // index of src input tensor
    const size_t dst_index = 0;  // index of dst output tensor
    const Tensor& src_tensor = MklGetInput(context, src_index);
    MklDnnShape dnn_shape_src;
    GetMklShape(context, src_index, &dnn_shape_src);

    Tensor* dst_tensor = nullptr;
    T* user_i = const_cast<T*>(src_tensor.flat<T>().data());
    MklDnnShape dnn_shape_dst;
    dnn_shape_dst.SetMklTensor(false);
    AllocateOutputSetMklShape(context, dst_index, &dst_tensor,
                              src_tensor.shape(), dnn_shape_dst);
    T* out_o = dst_tensor->flat<T>().data();
    out_o[0] = std::min(std::max(user_i[0], static_cast<T>(0)),
                        static_cast<T>(RELU6_UPPER_BOUND));
    return;
  }
};

template <typename Device, typename T>
class MklRelu6GradOp
    : public MklReluGradOpBase<Device, T, ALGORITHM::eltwise_bounded_relu> {
 public:
  ~MklRelu6GradOp() {}

  explicit MklRelu6GradOp(OpKernelConstruction* context)
      : MklReluGradOpBase<Device, T, ALGORITHM::eltwise_bounded_relu>(
            context, RELU6_UPPER_BOUND, 0.0f) {}

  virtual void Compute_Scalar(OpKernelContext* context) {
    const size_t diff_dst_index = 0;  // index of diff_dst input tensor
    const size_t src_index = 1;       // index of src input tensor
    const size_t diff_src_index = 0;  // index of diff_src output tensor
    const Tensor& src_tensor = MklGetInput(context, src_index);
    const Tensor& diff_dst_tensor = MklGetInput(context, diff_dst_index);
    Tensor* diff_src_tensor = nullptr;

    MklDnnShape dnn_shape_diff_dst;
    GetMklShape(context, diff_dst_index, &dnn_shape_diff_dst);

    MklDnnShape dnn_shape_diff_src;
    dnn_shape_diff_src.SetMklTensor(false);
    AllocateOutputSetMklShape(context, diff_src_index, &diff_src_tensor,
                              diff_dst_tensor.shape(), dnn_shape_diff_src);
    T* out_o = diff_src_tensor->flat<T>().data();
    T* user_i = const_cast<T*>(src_tensor.flat<T>().data());
    T* user_g = const_cast<T*>(diff_dst_tensor.flat<T>().data());
    out_o[0] = user_g[0] *
               static_cast<T>(user_i[0] > static_cast<T>(0) &&
                              (user_i[0] < static_cast<T>(RELU6_UPPER_BOUND)));
    return;
  }
};

template <typename Device, typename T>
class MklLeakyReluOp
    : public MklReluOpBase<Device, T, ALGORITHM::eltwise_relu> {
 public:
  ~MklLeakyReluOp() {}

  explicit MklLeakyReluOp(OpKernelConstruction* context)
      : MklReluOpBase<Device, T, ALGORITHM::eltwise_relu>(context, 0.0f, 0.0f) {
    float alpha;
    OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha));
    OP_REQUIRES(
        context, alpha <= 1,
        errors::InvalidArgument("MKL LeakyRelu only supports alpha <= 1. "
                                "alpha is: ",
                                alpha));

    this->alpha_ = alpha;
  }

  virtual void Compute_Scalar(OpKernelContext* context) {
    const size_t src_index = 0;  // index of src input tensor
    const size_t dst_index = 0;  // index of dst output tensor
    const Tensor& src_tensor = MklGetInput(context, src_index);
    MklDnnShape dnn_shape_src;
    GetMklShape(context, src_index, &dnn_shape_src);

    Tensor* dst_tensor = nullptr;
    T* user_i = const_cast<T*>(src_tensor.flat<T>().data());
    MklDnnShape dnn_shape_dst;
    dnn_shape_dst.SetMklTensor(false);
    AllocateOutputSetMklShape(context, dst_index, &dst_tensor,
                              src_tensor.shape(), dnn_shape_dst);
    T* out_o = dst_tensor->flat<T>().data();
    out_o[0] = user_i[0] >= T(0) ? user_i[0] : user_i[0] * T(this->alpha_);
    return;
  }
};

template <typename Device, typename T>
class MklLeakyReluGradOp
    : public MklReluGradOpBase<Device, T, ALGORITHM::eltwise_relu> {
 public:
  ~MklLeakyReluGradOp() {}

  explicit MklLeakyReluGradOp(OpKernelConstruction* context)
      : MklReluGradOpBase<Device, T, ALGORITHM::eltwise_relu>(context, 0.0f,
                                                              0.0f) {
    float alpha;
    OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha));
    OP_REQUIRES(
        context, alpha <= 1,
        errors::InvalidArgument("MKL LeakyRelu only supports alpha <= 1. "
                                "alpha is: ",
                                alpha));

    this->alpha_ = alpha;
  }

  virtual void Compute_Scalar(OpKernelContext* context) {
    const size_t diff_dst_index = 0;  // index of diff_dst input tensor
    const size_t src_index = 1;       // index of src input tensor
    const size_t diff_src_index = 0;  // index of diff_src output tensor
    const Tensor& src_tensor = MklGetInput(context, src_index);
    const Tensor& diff_dst_tensor = MklGetInput(context, diff_dst_index);
    Tensor* diff_src_tensor = nullptr;

    MklDnnShape dnn_shape_diff_dst;
    GetMklShape(context, diff_dst_index, &dnn_shape_diff_dst);

    MklDnnShape dnn_shape_diff_src;
    dnn_shape_diff_src.SetMklTensor(false);
    AllocateOutputSetMklShape(context, diff_src_index, &diff_src_tensor,
                              diff_dst_tensor.shape(), dnn_shape_diff_src);
    T* out_o = diff_src_tensor->flat<T>().data();
    T* user_i = const_cast<T*>(src_tensor.flat<T>().data());
    T* user_g = const_cast<T*>(diff_dst_tensor.flat<T>().data());
    out_o[0] = user_i[0] >= static_cast<T>(0)
                   ? user_g[0]
                   : user_g[0] * static_cast<T>(this->alpha_);
    return;
  }
};

// register dnn kernels for supported operations and supported types
#define REGISTER_RELU_MKL_SUPPORTED_KERNELS_TYPES(type)        \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklRelu")                                         \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklReluOp<CPUDevice, type>);                             \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklReluGrad")                                     \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklReluGradOp<CPUDevice, type>);
TF_CALL_float(REGISTER_RELU_MKL_SUPPORTED_KERNELS_TYPES);
TF_CALL_bfloat16(REGISTER_RELU_MKL_SUPPORTED_KERNELS_TYPES);

// register dnn kernels for supported operations and supported types
#define REGISTER_ELU_MKL_SUPPORTED_KERNELS_TYPES(type)         \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklElu")                                          \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklEluOp<CPUDevice, type>);                              \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklEluGrad")                                      \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklEluGradOp<CPUDevice, type>);
TF_CALL_float(REGISTER_ELU_MKL_SUPPORTED_KERNELS_TYPES);
TF_CALL_bfloat16(REGISTER_ELU_MKL_SUPPORTED_KERNELS_TYPES);

#ifdef ENABLE_MKLDNN_V1
#define REGISTER_TANH_MKL_SUPPORTED_KERNELS_TYPES(type)        \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklTanh")                                         \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklTanhOp<CPUDevice, type>);                             \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklTanhGrad")                                     \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklTanhGradOp<CPUDevice, type>);
TF_CALL_float(REGISTER_TANH_MKL_SUPPORTED_KERNELS_TYPES);
TF_CALL_bfloat16(REGISTER_TANH_MKL_SUPPORTED_KERNELS_TYPES);
#endif

#define REGISTER_RELU6_MKL_SUPPORTED_KERNELS_TYPES(type)       \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklRelu6")                                        \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklRelu6Op<CPUDevice, type>);                            \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklRelu6Grad")                                    \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklRelu6GradOp<CPUDevice, type>);
TF_CALL_float(REGISTER_RELU6_MKL_SUPPORTED_KERNELS_TYPES);
TF_CALL_bfloat16(REGISTER_RELU6_MKL_SUPPORTED_KERNELS_TYPES);

#define REGISTER_LeakyRelu_MKL_SUPPORTED_KERNELS_TYPES(type)   \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklLeakyRelu")                                    \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklLeakyReluOp<CPUDevice, type>);                        \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklLeakyReluGrad")                                \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklLeakyReluGradOp<CPUDevice, type>);
TF_CALL_float(REGISTER_LeakyRelu_MKL_SUPPORTED_KERNELS_TYPES);
TF_CALL_bfloat16(REGISTER_LeakyRelu_MKL_SUPPORTED_KERNELS_TYPES);

}  // namespace tensorflow

#endif  // INTEL_MKL
