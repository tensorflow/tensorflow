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

#include "mkldnn.hpp"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/mkl_util.h"

using mkldnn::algorithm;
using mkldnn::eltwise_bounded_relu;
using mkldnn::eltwise_elu;
using mkldnn::eltwise_forward;
using mkldnn::eltwise_relu;
using mkldnn::eltwise_tanh;
using mkldnn::memory;
using mkldnn::prop_kind;
using mkldnn::stream;

namespace tensorflow {

template <typename T>
class MklEltwiseFwdParams {
 public:
  memory::dims src_dims;  // check if this is needed
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
      : cpu_engine_(engine::cpu, 0) {
    // store expected format
    context_.src_fmt =
        static_cast<mkldnn::memory::format>(fwdParams.src_md.data.format);
    context_.fwd_stream.reset(new stream(stream::kind::eager));

    // create eltwise primitive
    if (context_.eltwise_fwd == nullptr) {
      Setup(fwdParams);
    }
  }

  ~MklEltwiseFwdPrimitive() {}

  // Eltwise forward execute
  //   src_data:  input data buffer of src
  //   dst_data:  output data buffer of dst
  void Execute(const T* src_data, T* dst_data) {
    context_.src_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(src_data)));
    context_.dst_mem->set_data_handle(static_cast<void*>(dst_data));
    context_.fwd_stream->submit(context_.fwd_primitives);

    // after execution, set data handle back
    context_.src_mem->set_data_handle(DummyData);
    context_.dst_mem->set_data_handle(DummyData);
  }

  std::shared_ptr<mkldnn::eltwise_forward::primitive_desc> GetEltwiseFwdPd() {
    return context_.fwd_pd;
  }

  memory::format GetSrcMemoryFormat() { return context_.src_fmt; }

 private:
  // Primitive reuse context for eltwise Fwd ops: Relu, Elu, Tanh
  struct EltwiseFwdContext {
    // expected memory format for this primitive instance
    mkldnn::memory::format src_fmt;

    // MKLDNN memory
    std::shared_ptr<memory> src_mem;
    std::shared_ptr<memory> dst_mem;

    // desc & prmitive desc
    std::shared_ptr<mkldnn::eltwise_forward::desc> fwd_desc;
    std::shared_ptr<mkldnn::eltwise_forward::primitive_desc> fwd_pd;

    // memory desc
    std::shared_ptr<memory::desc> src_md;
    std::shared_ptr<memory::desc> dst_md;

    // memory primitive desc
    std::shared_ptr<memory::primitive_desc> src_mpd;

    // Eltwise primitive
    std::shared_ptr<mkldnn::primitive> eltwise_fwd;

    std::shared_ptr<stream> fwd_stream;
    std::vector<mkldnn::primitive> fwd_primitives;

    EltwiseFwdContext()
        : src_fmt(memory::format::any),
          src_mem(nullptr),
          dst_mem(nullptr),
          fwd_desc(nullptr),
          fwd_pd(nullptr),
          src_md(nullptr),
          dst_md(nullptr),
          src_mpd(nullptr),
          eltwise_fwd(nullptr),
          fwd_stream(nullptr) {}
  };

  // Eltwise forward primitive setup
  void Setup(const MklEltwiseFwdParams<T>& fwdParams) {
    // create memory descriptors for eltwise data with specified format
    context_.src_md.reset(new memory::desc(fwdParams.src_md.data));
    context_.src_mpd.reset(
        new memory::primitive_desc(*context_.src_md, cpu_engine_));

    // create a eltwise
    context_.fwd_desc.reset(new mkldnn::eltwise_forward::desc(
        prop_kind::forward, fwdParams.alg_kind, *context_.src_md,
        fwdParams.alpha, fwdParams.beta));
    context_.fwd_pd.reset(new mkldnn::eltwise_forward::primitive_desc(
        *context_.fwd_desc, cpu_engine_));

    // create memory primitive based on dummy data
    context_.src_mem.reset(new memory(*context_.src_mpd, DummyData));
    context_.dst_mem.reset(
        new memory(context_.fwd_pd.get()->dst_primitive_desc(), DummyData));

    // create eltwise primitive and add it to net
    context_.eltwise_fwd.reset(new mkldnn::eltwise_forward(
        *context_.fwd_pd, *context_.src_mem, *context_.dst_mem));

    context_.fwd_primitives.push_back(*context_.eltwise_fwd);
  }

  struct EltwiseFwdContext context_;
  engine cpu_engine_;
};

template <typename T>
class MklEltwiseFwdPrimitiveFactory : public MklPrimitiveFactory<T> {
 public:
  static MklEltwiseFwdPrimitive<T>* Get(
      const MklEltwiseFwdParams<T>& fwdParams) {
    MklEltwiseFwdPrimitive<T>* eltwise_forward = nullptr;

    auto src_fmt =
        static_cast<mkldnn::memory::format>(fwdParams.src_md.data.format);

    // Get a eltwise fwd primitive from the cached pool
    eltwise_forward = static_cast<MklEltwiseFwdPrimitive<T>*>(
        MklEltwiseFwdPrimitiveFactory<T>::GetInstance().GetEltwiseFwd(fwdParams,
                                                                      src_fmt));
    if (eltwise_forward == nullptr) {
      eltwise_forward = new MklEltwiseFwdPrimitive<T>(fwdParams);
      MklEltwiseFwdPrimitiveFactory<T>::GetInstance().SetEltwiseFwd(
          fwdParams, src_fmt, eltwise_forward);
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

  static string CreateKey(const MklEltwiseFwdParams<T>& fwdParams,
                          memory::format src_fmt) {
    string prefix = "eltwise_fwd";
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(prefix);
    key_creator.AddAsKey(fwdParams.src_dims);
    key_creator.AddAsKey<int>(static_cast<int>(fwdParams.alg_kind));
    key_creator.AddAsKey<float>(static_cast<float>(fwdParams.alpha));
    key_creator.AddAsKey<float>(static_cast<float>(fwdParams.beta));
    key_creator.AddAsKey<int>(static_cast<int>(src_fmt));
    return key_creator.GetKey();
  }

  MklPrimitive* GetEltwiseFwd(const MklEltwiseFwdParams<T>& fwdParams,
                              memory::format src_fmt) {
    string key = CreateKey(fwdParams, src_fmt);
    return this->GetOp(key);
  }

  void SetEltwiseFwd(const MklEltwiseFwdParams<T>& fwdParams,
                     memory::format src_fmt, MklPrimitive* op) {
    string key = CreateKey(fwdParams, src_fmt);
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

  MklEltwiseBwdParams(const memory::dims& src_dims,
                      const memory::desc& common_md, algorithm alg_kind,
                      float alpha, float beta)
      : src_dims(src_dims),
        common_md(common_md),
        alg_kind(alg_kind),
        alpha(alpha),
        beta(beta) {}
};

template <typename T>
class MklEltwiseBwdPrimitive : public MklPrimitive {
 public:
  explicit MklEltwiseBwdPrimitive(const MklEltwiseBwdParams<T>& bwdParams)
      : cpu_engine_(engine::cpu, 0) {
    context_.src_fmt =
        static_cast<mkldnn::memory::format>(bwdParams.common_md.data.format);
    context_.diff_dst_fmt =
        static_cast<mkldnn::memory::format>(bwdParams.common_md.data.format);
    context_.bwd_stream.reset(new stream(stream::kind::eager));
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
  void Execute(const T* src_data, const T* diff_dst_data, T* diff_src_data) {
    context_.src_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(src_data)));
    context_.diff_dst_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(diff_dst_data)));
    context_.diff_src_mem->set_data_handle(static_cast<void*>(diff_src_data));
    context_.bwd_stream->submit(context_.bwd_primitives);

    // after execution, set data handle back
    context_.src_mem->set_data_handle(DummyData);
    context_.diff_dst_mem->set_data_handle(DummyData);
    context_.diff_src_mem->set_data_handle(DummyData);
  }

  std::shared_ptr<mkldnn::eltwise_backward::primitive_desc> GetEltwiseBwdPd() {
    return context_.bwd_pd;
  }

  memory::format GetSrcMemoryFormat() { return context_.src_fmt; }

  memory::format GetDiffDstMemoryFormat() { return context_.diff_dst_fmt; }

 private:
  // Primitive reuse context for eltwise Bwd ops: Relu, Elu, Tanh
  struct EltwiseBwdContext {
    // expected memory format for this primitive instance
    memory::format src_fmt;
    memory::format diff_dst_fmt;

    // MKLDNN memory
    std::shared_ptr<memory> src_mem;
    std::shared_ptr<memory> diff_dst_mem;
    std::shared_ptr<memory> diff_src_mem;

    // desc & prmitive desc
    std::shared_ptr<mkldnn::eltwise_backward::desc> bwd_desc;

    // memory desc
    std::shared_ptr<memory::desc> src_md;
    std::shared_ptr<memory::desc> diff_dst_md;
    std::shared_ptr<memory::desc> common_md;

    // memory primitive desc
    std::shared_ptr<memory::primitive_desc> src_mpd;
    std::shared_ptr<memory::primitive_desc> diff_dst_mpd;

    // fwd primitive desc
    std::shared_ptr<mkldnn::eltwise_forward::desc> fwd_desc;
    std::shared_ptr<mkldnn::eltwise_forward::primitive_desc> fwd_pd;
    std::shared_ptr<mkldnn::eltwise_backward::primitive_desc> bwd_pd;

    // Eltwise primitive
    std::shared_ptr<mkldnn::primitive> eltwise_bwd;

    std::shared_ptr<stream> bwd_stream;
    std::vector<mkldnn::primitive> bwd_primitives;

    EltwiseBwdContext()
        : src_fmt(memory::format::any),
          diff_dst_fmt(memory::format::any),
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
          eltwise_bwd(nullptr),
          bwd_stream(nullptr) {}
  };

  // Eltwise backward primitive setup
  void Setup(const MklEltwiseBwdParams<T>& bwdParams) {
    // create memory descriptors for eltwise data w/ no specified format
    context_.src_md.reset(new memory::desc(bwdParams.common_md.data));
    context_.diff_dst_md.reset(new memory::desc(bwdParams.common_md.data));

    context_.src_mpd.reset(
        new memory::primitive_desc(*context_.src_md, cpu_engine_));
    context_.diff_dst_mpd.reset(
        new memory::primitive_desc(*context_.diff_dst_md, cpu_engine_));

    // create forward eltwise primitive
    context_.fwd_desc.reset(new mkldnn::eltwise_forward::desc(
        prop_kind::forward_training, bwdParams.alg_kind, *context_.src_md,
        bwdParams.alpha, bwdParams.beta));
    context_.fwd_pd.reset(new mkldnn::eltwise_forward::primitive_desc(
        *context_.fwd_desc, cpu_engine_));
    context_.bwd_desc.reset(new mkldnn::eltwise_backward::desc(
        bwdParams.alg_kind, *context_.diff_dst_md, *context_.src_md,
        bwdParams.alpha, bwdParams.beta));
    context_.bwd_pd.reset(new mkldnn::eltwise_backward::primitive_desc(
        *context_.bwd_desc, cpu_engine_, *context_.fwd_pd));

    // create memory primitive based on dummy data
    context_.src_mem.reset(new memory(*context_.src_mpd, DummyData));
    context_.diff_dst_mem.reset(new memory(*context_.diff_dst_mpd, DummyData));
    context_.diff_src_mem.reset(new memory(
        context_.bwd_pd.get()->diff_src_primitive_desc(), DummyData));

    // create eltwise primitive and add it to net
    context_.eltwise_bwd.reset(new mkldnn::eltwise_backward(
        *context_.bwd_pd, *context_.src_mem, *context_.diff_dst_mem,
        *context_.diff_src_mem));

    context_.bwd_primitives.push_back(*context_.eltwise_bwd);
  }

  struct EltwiseBwdContext context_;
  engine cpu_engine_;
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

    auto src_fmt =
        static_cast<mkldnn::memory::format>(bwdParams.common_md.data.format);
    auto diff_dst_fmt =
        static_cast<mkldnn::memory::format>(bwdParams.common_md.data.format);

    // try to find a suitable one in pool
    eltwise_backward = static_cast<MklEltwiseBwdPrimitive<T>*>(
        MklEltwiseBwdPrimitiveFactory<T>::GetInstance().GetEltwiseBwd(
            bwdParams, src_fmt, diff_dst_fmt));

    if (eltwise_backward == nullptr) {
      eltwise_backward = new MklEltwiseBwdPrimitive<T>(bwdParams);
      MklEltwiseBwdPrimitiveFactory<T>::GetInstance().SetEltwiseBwd(
          bwdParams, src_fmt, diff_dst_fmt, eltwise_backward);
    }
    return eltwise_backward;
  }

  static MklEltwiseBwdPrimitiveFactory& GetInstance() {
    static MklEltwiseBwdPrimitiveFactory instance_;
    return instance_;
  }

 private:
  static string CreateKey(const MklEltwiseBwdParams<T>& bwdParams,
                          const memory::format& src_fmt,
                          const memory::format& diff_dst_fmt) {
    string prefix = "eltwise_bwd";
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(prefix);
    key_creator.AddAsKey(bwdParams.src_dims);
    key_creator.AddAsKey(static_cast<int>(bwdParams.alg_kind));
    key_creator.AddAsKey(static_cast<float>(bwdParams.alpha));
    key_creator.AddAsKey(static_cast<float>(bwdParams.beta));
    key_creator.AddAsKey(static_cast<int>(src_fmt));
    key_creator.AddAsKey(static_cast<int>(diff_dst_fmt));
    return key_creator.GetKey();
  }

  MklPrimitive* GetEltwiseBwd(const MklEltwiseBwdParams<T>& bwdParams,
                              const memory::format& src_fmt,
                              const memory::format& diff_dst_fmt) {
    string key = CreateKey(bwdParams, src_fmt, diff_dst_fmt);
    return this->GetOp(key);
  }

  void SetEltwiseBwd(const MklEltwiseBwdParams<T>& bwdParams,
                     const memory::format& src_fmt,
                     const memory::format& diff_dst_fmt, MklPrimitive* op) {
    string key = CreateKey(bwdParams, src_fmt, diff_dst_fmt);
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

      // Set DNN primitive - src
      MklDnnData<T> src(&cpu_engine);
      memory::dims src_dims;
      memory::desc src_md({}, memory::data_undef, memory::format_undef);
      if (dnn_shape_src.IsMklTensor()) {
        src_md = dnn_shape_src.GetMklLayout();
        src_dims = dnn_shape_src.GetSizesAsMklDnnDims();
      } else {
        src_dims = TFShapeToMklDnnDims(src_tensor.shape());
        auto src_strides = CalculateTFStrides(src_dims);
        // Create blocked memory descriptor
        src_md = MklDnnData<T>::CreateBlockedMemDesc(src_dims, src_strides);
      }

      // get a eltwise fwd from primitive pool
      MklEltwiseFwdParams<T> fwdParams(src_dims, src_md, alg_kind, alpha_,
                                       beta_);
      MklEltwiseFwdPrimitive<T>* eltwise_fwd =
          MklEltwiseFwdPrimitiveFactory<T>::Get(fwdParams);

      // prepare for execuation
      const T* src_data = src_tensor.flat<T>().data();
      // check wehther src need to reorder
      if (src_md.data.format != eltwise_fwd->GetSrcMemoryFormat()) {
        src.SetUsrMem(src_md, &src_tensor);
        auto src_target_pd = memory::primitive_desc(
            {{src_dims}, MklDnnType<T>(), eltwise_fwd->GetSrcMemoryFormat()},
            cpu_engine);
        src.CheckReorderToOpMem(src_target_pd);
        src_data = const_cast<T*>(
            reinterpret_cast<T*>(src.GetOpMem().get_data_handle()));
      }

      // allocate dst tensor, always set it as MKL-DNN layout
      std::shared_ptr<mkldnn::eltwise_forward::primitive_desc> eltwise_fwd_pd =
          eltwise_fwd->GetEltwiseFwdPd();
      MklDnnShape dnn_shape_dst;
      TensorShape tf_shape_dst;
      if (dnn_shape_src.IsMklTensor()) {
        dnn_shape_dst.SetMklTensor(true);
        auto dst_pd = eltwise_fwd_pd->dst_primitive_desc();
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

      Tensor* dst_tensor = nullptr;
      OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                  {static_cast<const int>(src_index)},
                                  static_cast<const int>(dst_index),
                                  tf_shape_dst, &dst_tensor));
      AllocateOutputSetMklShape(context, dst_index, dnn_shape_dst);

      T* dst_data = dst_tensor->flat<T>().data();

      // execute eltwise
      eltwise_fwd->Execute(src_data, dst_data);
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
  engine cpu_engine = engine(engine::cpu, 0);
  std::shared_ptr<eltwise_forward::primitive_desc> relu_fwd_pd;

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

  void Compute(OpKernelContext* context) {
    try {
      MklDnnData<T> src(&cpu_engine);
      MklDnnData<T> diff_dst(&cpu_engine);

      const size_t diff_dst_index = 0;  // index of diff_dst input tensor
      const size_t src_index = 1;       // index of src input tensor
      const size_t diff_src_index = 0;  // index of diff_src output tensor

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

      // get a eltwise bwd from primitive pool
      memory::dims src_dims = {};
      memory::desc src_md({}, memory::data_undef, memory::format_undef);
      memory::desc diff_dst_md({}, memory::data_undef, memory::format_undef);
      if (!dnn_shape_src.IsMklTensor() && !dnn_shape_diff_dst.IsMklTensor()) {
        src_dims = TFShapeToMklDnnDims(src_tensor.shape());
        auto src_strides = CalculateTFStrides(src_dims);
        src_md = MklDnnData<T>::CreateBlockedMemDesc(src_dims, src_strides);
        diff_dst_md = src_md;
      } else if (dnn_shape_src.IsMklTensor() &&
                 !dnn_shape_diff_dst.IsMklTensor()) {
        src_md = dnn_shape_src.GetMklLayout();
        src_dims = dnn_shape_src.GetSizesAsMklDnnDims();

        memory::format src_mkl_data_format = dnn_shape_src.GetTfDataFormat();
        auto src_tf_data_format =
            MklDnnDataFormatToTFDataFormat(src_mkl_data_format);
        auto diff_dst_dims = TFShapeToMklDnnDimsInNCHW(diff_dst_tensor.shape(),
                                                       src_tf_data_format);
        diff_dst_md =
            memory::desc(diff_dst_dims, MklDnnType<T>(), src_mkl_data_format);
      } else if (!dnn_shape_src.IsMklTensor() &&
                 dnn_shape_diff_dst.IsMklTensor()) {
        diff_dst_md = dnn_shape_diff_dst.GetMklLayout();

        memory::format diff_dst_mkl_data_format =
            dnn_shape_diff_dst.GetTfDataFormat();
        auto diff_dst_tf_data_format =
            MklDnnDataFormatToTFDataFormat(diff_dst_mkl_data_format);

        src_dims = (src_tensor.dims() == 4)
                       ? TFShapeToMklDnnDimsInNCHW(src_tensor.shape(),
                                                   diff_dst_tf_data_format)
                       : TFShapeToMklDnnDimsInNCDHW(src_tensor.shape(),
                                                    diff_dst_tf_data_format);
        src_md =
            memory::desc(src_dims, MklDnnType<T>(), diff_dst_mkl_data_format);
      } else {
        src_md = dnn_shape_src.GetMklLayout();
        diff_dst_md = dnn_shape_diff_dst.GetMklLayout();
        src_dims = dnn_shape_src.GetSizesAsMklDnnDims();
      }

      // As per comment above, we tell MKLDNN that both the inputs are in same
      // format. So we set common memory descriptor in MKL format, if any of the
      // inputs are in MKL format. Let's get memory descriptor that we will use
      // for both the inputs.
      memory::desc common_md({}, memory::data_undef, memory::format_undef);
      if (dnn_shape_src.IsMklTensor() || dnn_shape_diff_dst.IsMklTensor()) {
        common_md = dnn_shape_src.IsMklTensor() ? src_md : diff_dst_md;
      } else {
        // Since both the inputs are in Tensorflow format, and have
        // same shape, we can get memory descriptor from any input.
        common_md = src_md;
      }

      MklEltwiseBwdParams<T> bwdParams(src_dims, common_md, alg_kind, alpha_,
                                       beta_);
      MklEltwiseBwdPrimitive<T>* eltwise_bwd =
          MklEltwiseBwdPrimitiveFactory<T>::Get(bwdParams);
      auto eltwise_bwd_pd = eltwise_bwd->GetEltwiseBwdPd();

      // check whether need reorder for src / diff_dst
      const T* src_data = src_tensor.flat<T>().data();
      if (src_md.data.format != eltwise_bwd->GetSrcMemoryFormat()) {
        src.SetUsrMem(src_md, &src_tensor);
        src.CheckReorderToOpMem(
            eltwise_bwd_pd.get()->diff_src_primitive_desc());
        src_data = const_cast<T*>(
            reinterpret_cast<T*>(src.GetOpMem().get_data_handle()));
      }

      const T* diff_dst_data = diff_dst_tensor.flat<T>().data();
      if (diff_dst_md.data.format != eltwise_bwd->GetDiffDstMemoryFormat()) {
        diff_dst.SetUsrMem(diff_dst_md, &diff_dst_tensor);
        diff_dst.CheckReorderToOpMem(
            eltwise_bwd_pd.get()->diff_src_primitive_desc());
        diff_dst_data = const_cast<T*>(
            reinterpret_cast<T*>(diff_dst.GetOpMem().get_data_handle()));
      }

      // allocate diff_src tensor
      MklDnnShape dnn_shape_diff_src;
      TensorShape tf_shape_diff_src;
      if (dnn_shape_src.IsMklTensor() || dnn_shape_diff_dst.IsMklTensor()) {
        auto diff_src_pd = eltwise_bwd_pd->diff_src_primitive_desc();
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
      eltwise_bwd->Execute(src_data, diff_dst_data, diff_src_data);
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
  engine cpu_engine = engine(engine::cpu, 0);
  std::shared_ptr<eltwise_forward::primitive_desc> relu_fwd_pd;

 protected:
  float alpha_;
  float beta_;
};

template <typename Device, typename T>
class MklReluOp : public MklReluOpBase<Device, T, eltwise_relu> {
 public:
  ~MklReluOp() {}

  explicit MklReluOp(OpKernelConstruction* context)
      : MklReluOpBase<Device, T, eltwise_relu>(context, 0.0f, 0.0f) {}

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
class MklReluGradOp : public MklReluGradOpBase<Device, T, eltwise_relu> {
 public:
  ~MklReluGradOp() {}

  explicit MklReluGradOp(OpKernelConstruction* context)
      : MklReluGradOpBase<Device, T, eltwise_relu>(context, 0.0f, 0.0f) {}

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
class MklEluOp : public MklReluOpBase<Device, T, eltwise_elu> {
 public:
  ~MklEluOp() {}

  explicit MklEluOp(OpKernelConstruction* context)
      : MklReluOpBase<Device, T, eltwise_elu>(context, 0.0f, 0.0f) {}

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
class MklEluGradOp : public MklReluGradOpBase<Device, T, eltwise_elu> {
 public:
  ~MklEluGradOp() {}

  explicit MklEluGradOp(OpKernelConstruction* context)
      : MklReluGradOpBase<Device, T, eltwise_elu>(context, 0.0f, 0.0f) {}

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

template <typename Device, typename T>
class MklTanhOp : public MklReluOpBase<Device, T, eltwise_tanh> {
 public:
  ~MklTanhOp() {}

  explicit MklTanhOp(OpKernelConstruction* context)
      : MklReluOpBase<Device, T, eltwise_tanh>(context, 0.0f, 0.0f) {}

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
class MklTanhGradOp : public MklReluGradOpBase<Device, T, eltwise_tanh> {
 public:
  ~MklTanhGradOp() {}

  explicit MklTanhGradOp(OpKernelConstruction* context)
      : MklReluGradOpBase<Device, T, eltwise_tanh>(context, 0.0f, 0.0f) {}

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
    // gradient of tanh(x) = 1 - tanh(x)^2
    T feature = (static_cast<T*>(user_i))[0];
    T e1 = std::exp(feature);
    T e2 = std::exp(-feature);
    T tanh = (e1 - e2) / (e1 + e2);
    void* user_g =
        static_cast<void*>(const_cast<T*>(diff_dst_tensor.flat<T>().data()));
    (static_cast<T*>(out_o))[0] =
        (static_cast<T*>(user_g))[0] * (static_cast<T>(1) - tanh * tanh);
  }
};

#define RELU6_UPPER_BOUND 6.0f
template <typename Device, typename T>
class MklRelu6Op : public MklReluOpBase<Device, T, eltwise_bounded_relu> {
 public:
  ~MklRelu6Op() {}

  explicit MklRelu6Op(OpKernelConstruction* context)
      : MklReluOpBase<Device, T, eltwise_bounded_relu>(
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
    : public MklReluGradOpBase<Device, T, eltwise_bounded_relu> {
 public:
  ~MklRelu6GradOp() {}

  explicit MklRelu6GradOp(OpKernelConstruction* context)
      : MklReluGradOpBase<Device, T, eltwise_bounded_relu>(
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
class MklLeakyReluOp : public MklReluOpBase<Device, T, eltwise_relu> {
 public:
  ~MklLeakyReluOp() {}

  explicit MklLeakyReluOp(OpKernelConstruction* context)
      : MklReluOpBase<Device, T, eltwise_relu>(context, 0.0f, 0.0f) {
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
class MklLeakyReluGradOp : public MklReluGradOpBase<Device, T, eltwise_relu> {
 public:
  ~MklLeakyReluGradOp() {}

  explicit MklLeakyReluGradOp(OpKernelConstruction* context)
      : MklReluGradOpBase<Device, T, eltwise_relu>(context, 0.0f, 0.0f) {
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
TF_CALL_bfloat16(REGISTER_RELU_MKL_SUPPORTED_KERNELS_TYPES);

// register dnn kernels for supported operations and supported types
#define REGISTER_ELU_MKL_SUPPORTED_KERNELS_TYPES(type)              \
  REGISTER_KERNEL_BUILDER(Name("_MklElu")                           \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<type>("T")            \
                              .Label(mkl_op_registry::kMklOpLabel), \
                          MklEluOp<CPUDevice, type>);               \
  REGISTER_KERNEL_BUILDER(Name("_MklEluGrad")                       \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<type>("T")            \
                              .Label(mkl_op_registry::kMklOpLabel), \
                          MklEluGradOp<CPUDevice, type>);
TF_CALL_float(REGISTER_ELU_MKL_SUPPORTED_KERNELS_TYPES);
TF_CALL_bfloat16(REGISTER_ELU_MKL_SUPPORTED_KERNELS_TYPES);

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
TF_CALL_bfloat16(REGISTER_TANH_MKL_SUPPORTED_KERNELS_TYPES);

#define REGISTER_RELU6_MKL_SUPPORTED_KERNELS_TYPES(type)            \
  REGISTER_KERNEL_BUILDER(Name("_MklRelu6")                         \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<type>("T")            \
                              .Label(mkl_op_registry::kMklOpLabel), \
                          MklRelu6Op<CPUDevice, type>);             \
  REGISTER_KERNEL_BUILDER(Name("_MklRelu6Grad")                     \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<type>("T")            \
                              .Label(mkl_op_registry::kMklOpLabel), \
                          MklRelu6GradOp<CPUDevice, type>);
TF_CALL_float(REGISTER_RELU6_MKL_SUPPORTED_KERNELS_TYPES);
TF_CALL_bfloat16(REGISTER_RELU6_MKL_SUPPORTED_KERNELS_TYPES);

#define REGISTER_LeakyRelu_MKL_SUPPORTED_KERNELS_TYPES(type)        \
  REGISTER_KERNEL_BUILDER(Name("_MklLeakyRelu")                     \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<type>("T")            \
                              .Label(mkl_op_registry::kMklOpLabel), \
                          MklLeakyReluOp<CPUDevice, type>);         \
  REGISTER_KERNEL_BUILDER(Name("_MklLeakyReluGrad")                 \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<type>("T")            \
                              .Label(mkl_op_registry::kMklOpLabel), \
                          MklLeakyReluGradOp<CPUDevice, type>);
TF_CALL_float(REGISTER_LeakyRelu_MKL_SUPPORTED_KERNELS_TYPES);
TF_CALL_bfloat16(REGISTER_LeakyRelu_MKL_SUPPORTED_KERNELS_TYPES);

}  // namespace tensorflow

#endif  // INTEL_MKL
