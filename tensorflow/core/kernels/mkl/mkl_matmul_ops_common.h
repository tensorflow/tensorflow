/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_MKL_MKL_MATMUL_OPS_COMMON_H_
#define TENSORFLOW_CORE_KERNELS_MKL_MKL_MATMUL_OPS_COMMON_H_

#ifdef INTEL_MKL
#include <memory>
#include <string>
#include <vector>

#include "mkldnn.hpp"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/mkl_util.h"

using mkldnn::inner_product_forward;
using mkldnn::primitive_attr;
using mkldnn::prop_kind;
using mkldnn::stream;

namespace tensorflow {

#define L1_SIZE 32 * 1024
typedef Eigen::ThreadPoolDevice CPUDevice;

inline bool ExecuteSingleThreadedGemm(int m, int n, int k) {
  // Ideally we would like to determine blocking and then come up with
  // a heuristic but what we are targeting are very small models whose
  // total size is < few L1's. So we will do this simple calculation
  // to determine if the matrix multiplication should be run on a single thread.
  constexpr int kHeuristicMultiplier = 8;
  return ((sizeof(float) * (m * n + k * (m + n))) <
          L1_SIZE * kHeuristicMultiplier);
}

// This structure aggregates multiple inputs to MklDnnMatMul* methods.
struct MklDnnMatMulFwdParams {
  memory::dims src_dims;
  memory::dims weight_dims;
  memory::dims bias_dims;
  memory::dims dst_dims;
  memory::format_tag src_format;
  memory::format_tag weight_format;
  memory::format_tag dst_format;
  string dtypes = string("");
  struct PostOpParam {
    string name;
    std::vector<float> param;
  };
  std::vector<PostOpParam> post_op_params;

  MklDnnMatMulFwdParams(
      memory::dims src_dims, memory::dims weight_dims, memory::dims bias_dims,
      memory::dims dst_dims,
      memory::format_tag src_format = memory::format_tag::any,
      memory::format_tag weight_format = memory::format_tag::any,
      memory::format_tag dst_format = memory::format_tag::any)
      : src_dims(src_dims),
        weight_dims(weight_dims),
        bias_dims(bias_dims),
        dst_dims(dst_dims),
        src_format(src_format),
        weight_format(weight_format),
        dst_format(dst_format) {}
};

// With quantization, input, weight, bias, and output can have different types.
// So we use different template parameters for each type.
// TODO(intel-tf): The template type "T" is currently used to match the
// templatized class MklPrimitiveFactory (tensorflow/core/util/mkl_util.h).
// In the future, with the removal of "T" from MklPrimitiveFactory, this class
// needs to drop "T".
template <typename T, typename Tinput, typename Tweight, typename Tbias,
          typename Toutput>
class MklDnnMatMulFwdPrimitive : public MklPrimitive {
 public:
  explicit MklDnnMatMulFwdPrimitive(
      const MklDnnMatMulFwdParams& matmulFwdParams)
      : MklPrimitive(engine(engine::kind::cpu, 0)) {
    // Create matmul primitive
    if (context_.matmul_fwd == nullptr) {
      Setup(matmulFwdParams);
    }
  }

  ~MklDnnMatMulFwdPrimitive() {}

  // Inner-product forward execute with bias:
  //  - src_data: input data buffer of src
  //  - weight_data: input data buffer of weight
  //  - bias_data: input data buffer of bias
  //  - dst_data: output data buffer of dst
  void Execute(const Tinput* src_data, const Tweight* weight_data,
               const Tbias* bias_data, Toutput* dst_data,
               std::shared_ptr<stream> fwd_stream) {
#ifndef ENABLE_ONEDNN_OPENMP
    context_.src_mem->set_data_handle(
        static_cast<void*>(const_cast<Tinput*>(src_data)), *fwd_stream);
    context_.weight_mem->set_data_handle(
        static_cast<void*>(const_cast<Tweight*>(weight_data)), *fwd_stream);
    context_.bias_mem->set_data_handle(
        static_cast<void*>(const_cast<Tbias*>(bias_data)));
    context_.dst_mem->set_data_handle(static_cast<void*>(dst_data),
                                      *fwd_stream);
#else
    context_.src_mem->set_data_handle(
        static_cast<void*>(const_cast<Tinput*>(src_data)));
    context_.weight_mem->set_data_handle(
        static_cast<void*>(const_cast<Tweight*>(weight_data)));
    context_.bias_mem->set_data_handle(
        static_cast<void*>(const_cast<Tbias*>(bias_data)));
    context_.dst_mem->set_data_handle(static_cast<void*>(dst_data));
#endif  // !ENABLE_ONEDNN_OPENMP

    execute_primitives(context_.fwd_primitives, fwd_stream, context_.net_args);

    // After execution, set data handle back
    context_.src_mem->set_data_handle(DummyData);
    context_.weight_mem->set_data_handle(DummyData);
    context_.bias_mem->set_data_handle(DummyData);
    context_.dst_mem->set_data_handle(DummyData);
  }

  std::shared_ptr<mkldnn::inner_product_forward::primitive_desc>
  GetPrimitiveDesc() const {
    return context_.fwd_pd;
  }

 private:
  // Primitive reuse context for inner-product Fwd op
  struct MklDnnMatMulFwdContext {
    // MKL-DNN memory.
    std::shared_ptr<mkldnn::memory> src_mem;
    std::shared_ptr<mkldnn::memory> weight_mem;
    std::shared_ptr<mkldnn::memory> bias_mem;
    std::shared_ptr<mkldnn::memory> dst_mem;

    // Descriptor and primitive-descriptor for forward inner-product.
    std::shared_ptr<mkldnn::inner_product_forward::desc> fwd_desc;
    std::shared_ptr<mkldnn::inner_product_forward::primitive_desc> fwd_pd;

    // Memory descriptors.
    std::shared_ptr<mkldnn::memory::desc> src_md;
    std::shared_ptr<mkldnn::memory::desc> weight_md;
    std::shared_ptr<mkldnn::memory::desc> bias_md;
    std::shared_ptr<mkldnn::memory::desc> dst_md;

    // Inner-product primitive.
    std::shared_ptr<mkldnn::primitive> matmul_fwd;
    std::vector<mkldnn::primitive> fwd_primitives;

    std::vector<std::unordered_map<int, memory>> net_args;

    MklDnnMatMulFwdContext()
        : src_mem(nullptr),
          weight_mem(nullptr),
          bias_mem(nullptr),
          dst_mem(nullptr),
          fwd_desc(nullptr),
          fwd_pd(nullptr),
          src_md(nullptr),
          weight_md(nullptr),
          bias_md(nullptr),
          dst_md(nullptr),
          matmul_fwd(nullptr) {}
  };

  void Setup(const MklDnnMatMulFwdParams& matmul_fwd_params) {
    // Create memory descriptors for inner-product data without specified
    // format.
    context_.src_md.reset(new memory::desc({matmul_fwd_params.src_dims},
                                           MklDnnType<Tinput>(),
                                           matmul_fwd_params.src_format));

    context_.weight_md.reset(new memory::desc({matmul_fwd_params.weight_dims},
                                              MklDnnType<Tweight>(),
                                              matmul_fwd_params.weight_format));

    context_.dst_md.reset(new memory::desc({matmul_fwd_params.dst_dims},
                                           MklDnnType<Toutput>(),
                                           matmul_fwd_params.dst_format));

    context_.bias_md.reset(new memory::desc({matmul_fwd_params.bias_dims},
                                            MklDnnType<Tbias>(),
                                            memory::format_tag::any));
    // Create an inner-product.
    context_.fwd_desc.reset(new inner_product_forward::desc(
        prop_kind::forward_inference, *context_.src_md, *context_.weight_md,
        *context_.bias_md, *context_.dst_md));
    context_.fwd_pd.reset(new inner_product_forward::primitive_desc(
        *context_.fwd_desc, cpu_engine_));

    // Check if there is any fusion as post-ops
    auto const& post_op_params = matmul_fwd_params.post_op_params;
    mkldnn::primitive_attr post_ops_attr;
    mkldnn::post_ops post_ops;
    if (!post_op_params.empty()) {
      for (auto const& post_op_param : post_op_params) {
        if (post_op_param.name == "relu" || post_op_param.name == "leakyrelu") {
          DCHECK_EQ(post_op_param.param.size(), 3);
          float op_scale = post_op_param.param[0];
          float op_alpha = post_op_param.param[1];
          float op_beta = post_op_param.param[2];
          post_ops.append_eltwise(op_scale, mkldnn::algorithm::eltwise_relu,
                                  op_alpha, op_beta);
        } else if (post_op_param.name == "relu6") {
          DCHECK_EQ(post_op_param.param.size(), 3);
          float op_scale = post_op_param.param[0];
          float op_alpha = post_op_param.param[1];
          float op_beta = post_op_param.param[2];
          post_ops.append_eltwise(op_scale,
                                  mkldnn::algorithm::eltwise_bounded_relu,
                                  op_alpha, op_beta);
        } else if (post_op_param.name == "elu") {
          DCHECK_EQ(post_op_param.param.size(), 3);
          float op_scale = post_op_param.param[0];
          float op_alpha = post_op_param.param[1];
          float op_beta = post_op_param.param[2];
          post_ops.append_eltwise(op_scale, mkldnn::algorithm::eltwise_elu,
                                  op_alpha, op_beta);
        } else if (post_op_param.name == "tanh") {
          DCHECK_EQ(post_op_param.param.size(), 3);
          float op_scale = post_op_param.param[0];
          float op_alpha = post_op_param.param[1];
          float op_beta = post_op_param.param[2];
          post_ops.append_eltwise(op_scale, mkldnn::algorithm::eltwise_tanh,
                                  op_alpha, op_beta);
        } else if (post_op_param.name == "output_scale") {
          DCHECK_EQ(post_op_param.param.size(), 1);
          std::vector<float> scales;
          scales.push_back(post_op_param.param[0]);
          post_ops_attr.set_output_scales(0, scales);
        } else if (post_op_param.name == "sum") {
          DCHECK_EQ(post_op_param.param.size(), 1);
          float op_scale = post_op_param.param[0];
          post_ops.append_sum(op_scale);

        } else {
          DCHECK((post_op_param.name == "relu") ||
                 (post_op_param.name == "relu6") ||
                 (post_op_param.name == "elu") ||
                 (post_op_param.name == "tanh") ||
                 (post_op_param.name == "sum") ||
                 (post_op_param.name == "leakyrelu") ||
                 (post_op_param.name == "output_scale"));
        }
      }
      post_ops_attr.set_post_ops(post_ops);
      context_.fwd_pd.reset(new inner_product_forward::primitive_desc(
          *context_.fwd_desc, post_ops_attr, cpu_engine_));
    } else {
      context_.fwd_pd.reset(new inner_product_forward::primitive_desc(
          *context_.fwd_desc, cpu_engine_));
    }

    // Create memory primitive based on dummy data
    context_.src_mem.reset(
        new memory(context_.fwd_pd.get()->src_desc(), cpu_engine_, DummyData));
    context_.weight_mem.reset(new memory(context_.fwd_pd.get()->weights_desc(),
                                         cpu_engine_, DummyData));
    context_.dst_mem.reset(
        new memory(context_.fwd_pd.get()->dst_desc(), cpu_engine_, DummyData));
    context_.bias_mem.reset(new memory({{matmul_fwd_params.bias_dims},
                                        MklDnnType<Tbias>(),
                                        memory::format_tag::x},
                                       cpu_engine_, DummyData));

    // Create inner-product primitive.
    context_.matmul_fwd.reset(new inner_product_forward(*context_.fwd_pd));
    context_.net_args.push_back({{MKLDNN_ARG_SRC, *context_.src_mem},
                                 {MKLDNN_ARG_WEIGHTS, *context_.weight_mem},
                                 {MKLDNN_ARG_BIAS, *context_.bias_mem},
                                 {MKLDNN_ARG_DST, *context_.dst_mem}});

    context_.fwd_primitives.push_back(*context_.matmul_fwd);
    return;
  }

  struct MklDnnMatMulFwdContext context_;
};

template <typename T, typename Tinput, typename Tweight, typename Tbias,
          typename Toutput>
class MklDnnMatMulFwdPrimitiveFactory : public MklPrimitiveFactory<T> {
 public:
  static MklDnnMatMulFwdPrimitive<T, Tinput, Tweight, Tbias, Toutput>* Get(
      const MklDnnMatMulFwdParams& mkldnn_matmul_fwd_dims, bool do_not_cache) {
    MklDnnMatMulFwdPrimitive<T, Tinput, Tweight, Tbias, Toutput>* matmul_fwd =
        nullptr;

    if (do_not_cache) {
      // Always create new primitive
      matmul_fwd =
          new MklDnnMatMulFwdPrimitive<T, Tinput, Tweight, Tbias, Toutput>(
              mkldnn_matmul_fwd_dims);
    } else {
      // Try to find a suitable one in pool
      matmul_fwd = dynamic_cast<
          MklDnnMatMulFwdPrimitive<T, Tinput, Tweight, Tbias, Toutput>*>(
          MklDnnMatMulFwdPrimitiveFactory<T, Tinput, Tweight, Tbias,
                                          Toutput>::GetInstance()
              .GetMklDnnMatMulFwd(mkldnn_matmul_fwd_dims));
      if (matmul_fwd == nullptr) {
        matmul_fwd =
            new MklDnnMatMulFwdPrimitive<T, Tinput, Tweight, Tbias, Toutput>(
                mkldnn_matmul_fwd_dims);
        MklDnnMatMulFwdPrimitiveFactory<T, Tinput, Tweight, Tbias,
                                        Toutput>::GetInstance()
            .SetMklDnnMatMulFwd(mkldnn_matmul_fwd_dims, matmul_fwd);
      }
    }
    return matmul_fwd;
  }

 private:
  MklDnnMatMulFwdPrimitiveFactory() {}
  ~MklDnnMatMulFwdPrimitiveFactory() {}

  static MklDnnMatMulFwdPrimitiveFactory& GetInstance() {
    static MklDnnMatMulFwdPrimitiveFactory instance_;
    return instance_;
  }

  static string CreateKey(const MklDnnMatMulFwdParams& mkldnn_matmul_fwd_dims) {
    string prefix = "matmul_fwd_";
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(prefix);
    key_creator.AddAsKey(mkldnn_matmul_fwd_dims.src_dims);
    key_creator.AddAsKey(mkldnn_matmul_fwd_dims.weight_dims);
    key_creator.AddAsKey(mkldnn_matmul_fwd_dims.bias_dims);
    key_creator.AddAsKey(mkldnn_matmul_fwd_dims.dst_dims);
    key_creator.AddAsKey(mkldnn_matmul_fwd_dims.dtypes);
    key_creator.AddAsKey(mkldnn_matmul_fwd_dims.weight_format);

    // Generate keys for post-ops
    for (auto const& post_op_param : mkldnn_matmul_fwd_dims.post_op_params) {
      if (post_op_param.name == "relu" || post_op_param.name == "relu6" ||
          post_op_param.name == "elu" || post_op_param.name == "tanh" ||
          post_op_param.name == "leakyrelu") {
        DCHECK_EQ(post_op_param.param.size(), 3);
        key_creator.AddAsKey(post_op_param.name);
        key_creator.AddAsKey(post_op_param.param[0]);
        key_creator.AddAsKey(post_op_param.param[1]);
        key_creator.AddAsKey(post_op_param.param[2]);
      } else if (post_op_param.name == "sum") {
        DCHECK_EQ(post_op_param.param.size(), 1);
        key_creator.AddAsKey(post_op_param.name);
        key_creator.AddAsKey(post_op_param.param[0]);
      } else if (post_op_param.name == "output_scale") {
        DCHECK_EQ(post_op_param.param.size(), 1);
        key_creator.AddAsKey(post_op_param.name);
        key_creator.AddAsKey(post_op_param.param[0]);
      } else {
        return string("not_a_key");
      }
    }
    return key_creator.GetKey();
  }

  MklPrimitive* GetMklDnnMatMulFwd(
      const MklDnnMatMulFwdParams& mkldnn_matmul_fwd_dims) {
    string key = CreateKey(mkldnn_matmul_fwd_dims);
    return this->GetOp(key);
  }

  void SetMklDnnMatMulFwd(const MklDnnMatMulFwdParams& mkldnn_matmul_fwd_dims,
                          MklPrimitive* op) {
    string key = CreateKey(mkldnn_matmul_fwd_dims);
    this->SetOp(key, op);
  }
};

template <class Tweight, class Toutput>
class MklDnnMatMulOpBase : public OpKernel {
 public:
  explicit MklDnnMatMulOpBase(OpKernelConstruction* context)
      : OpKernel(context) {}
  void Compute(OpKernelContext* context) override = 0;

  // Allocate output tensor.
  virtual void AllocateOutputTensor(
      OpKernelContext* context,
      const inner_product_forward::primitive_desc& mkldnn_matmul_prim_desc,
      const memory::dims& output_dims_mkl_order,
      MklTensorFormat output_tf_format, Tensor** output_tensor,
      bool native_format = false) {
    DCHECK(output_tensor);
    auto dst_pd = mkldnn_matmul_prim_desc.dst_desc();

    MklDnnShape output_mkl_shape;
    output_mkl_shape.SetMklTensor(true);
    output_mkl_shape.SetMklLayout(&dst_pd);
    output_mkl_shape.SetElemType(MklDnnType<Toutput>());
    output_mkl_shape.SetTfLayout(output_dims_mkl_order.size(),
                                 output_dims_mkl_order, output_tf_format);

    TensorShape output_tf_shape;
    output_tf_shape.AddDim((dst_pd.get_size() / sizeof(Toutput)));

    if (native_format) {
      output_tf_shape = output_mkl_shape.GetTfShape();
    }
    // Allocate Output Tensor
    AllocateOutputSetMklShape(context, kOutputIndexDst, output_tensor,
                              output_tf_shape, output_mkl_shape, native_format);
  }

  // TF_LOCKS_EXCLUDED annotation ensures that the lock (mu_) cannot
  // be acquired before entering the function, since it is acquired
  // inside the function.
  inline bool IsWeightCacheEmpty(OpKernelContext* context)
      TF_LOCKS_EXCLUDED(mu_) {
    tf_shared_lock lock(mu_);
    return (weight_oi_.NumElements() == 0);
  }

  // Cache the converted weight in a tensor.
  // Only one thread can execute this method at any given time.
  void CacheWeight(
      OpKernelContext* context,
      const std::shared_ptr<mkldnn::inner_product_forward::primitive_desc>&
          matmul_fwd_pd,
      Tweight* weight_data, const Tensor& weight_tensor,
      MklDnnData<Tweight>& weight, const memory::desc& weight_md)
      TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock lock(mu_);
    const Tensor& weight_t = weight_oi_;

    // If the weights are already cached, there's nothing to do
    if (weight_t.NumElements() > 0) {
      return;
    }

    // reorder and cache the weight
    weight.SetUsrMem(weight_md, &weight_tensor);
    weight.CheckReorderToOpMem(matmul_fwd_pd.get()->weights_desc(), cpu_engine_,
                               context);
    weight_data = static_cast<Tweight*>(weight.GetOpMem().get_data_handle());

    size_t weight_size = matmul_fwd_pd.get()->weights_desc().get_size();
    TensorShape weight_tf_shape;
    weight_tf_shape.AddDim(weight_size / sizeof(Tweight));

    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<Tweight>::value,
                                          weight_tf_shape, &weight_oi_));

    void* weight_oi_t_data = weight.GetTensorBuffer(&weight_oi_);
    memcpy(weight_oi_t_data, weight_data, weight_size);

    // cache the memory descriptor
    auto expected_md = matmul_fwd_pd->weights_desc();
    TensorShape weight_mkl_format;
    weight_mkl_format.AddDim(sizeof(expected_md) / sizeof(Tweight));

    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<Tweight>::value,
                                          weight_mkl_format, &weight_oi_md_));
    *reinterpret_cast<memory::desc*>(weight_oi_md_.flat<Tweight>().data()) =
        expected_md;
  }

  Tweight* GetCachedWeight(OpKernelContext* context,
                           const memory::desc& expected_md)
      TF_LOCKS_EXCLUDED(mu_) {
    tf_shared_lock lock(mu_);
    const Tensor& weight_t = weight_oi_;
    const Tensor& weight_md_t = weight_oi_md_;

    // Check if the memory descriptor of the cached weight is same as
    // expected_md. if so use the cached memory, else return NULL
    if (weight_md_t.flat<Tweight>().size()) {
      const memory::desc& stored_md =
          *(static_cast<memory::desc*>(weight_md_t.data()));
      if (stored_md == expected_md) {
        return static_cast<Tweight*>(
            const_cast<Tweight*>(weight_t.flat<Tweight>().data()));
      }
    }
    return nullptr;
  }

  engine cpu_engine_ = engine(engine::kind::cpu, 0);

 protected:
  // Tensor to save reordered weight
  mutex mu_;
  Tensor weight_oi_ TF_GUARDED_BY(mu_);
  Tensor weight_oi_md_ TF_GUARDED_BY(mu_);

  bool is_weight_const_;

  const int kInputIndexSrc = 0;
  const int kInputIndexWeight = 1;
  const int kInputIndexBias = 2;
  const int kOutputIndexDst = 0;
};

using mkldnn::matmul;

namespace {

struct MklMatMulParams {
  memory::dims a_dims;
  memory::dims b_dims;
  memory::dims c_dims;
  memory::dims a_strides;
  memory::dims b_strides;
  memory::dims c_strides;

  MklMatMulParams(memory::dims a_dims, memory::dims b_dims, memory::dims c_dims,
                  memory::dims a_strides, memory::dims b_strides,
                  memory::dims c_strides)
      : a_dims(a_dims),
        b_dims(b_dims),
        c_dims(c_dims),
        a_strides(a_strides),
        b_strides(b_strides),
        c_strides(c_strides) {}
};

template <typename T>
class MklMatMulPrimitive : public MklPrimitive {
 public:
  explicit MklMatMulPrimitive(const MklMatMulParams& params)
      : MklPrimitive(engine(engine::kind::cpu, 0)) {
    // Create matmul primitive
    Setup(params);
  }

  ~MklMatMulPrimitive() {}

  void Execute(const T* a_data, const T* b_data, T* c_data,
               std::shared_ptr<stream> stream) {
#ifndef ENABLE_ONEDNN_OPENMP
    context_.a_mem->set_data_handle(static_cast<void*>(const_cast<T*>(a_data)),
                                    *stream);
    context_.b_mem->set_data_handle(static_cast<void*>(const_cast<T*>(b_data)),
                                    *stream);
    context_.c_mem->set_data_handle(static_cast<void*>(const_cast<T*>(c_data)),
                                    *stream);
#else
    context_.a_mem->set_data_handle(static_cast<void*>(const_cast<T*>(a_data)));
    context_.b_mem->set_data_handle(static_cast<void*>(const_cast<T*>(b_data)));
    context_.c_mem->set_data_handle(static_cast<void*>(const_cast<T*>(c_data)));
#endif  // !ENABLE_ONEDNN_OPENMP
    execute_primitives(context_.matmul_primitives, stream, context_.net_args);

    // After execution, set data handle back
    context_.a_mem->set_data_handle(DummyData);
    context_.b_mem->set_data_handle(DummyData);
    context_.c_mem->set_data_handle(DummyData);
  }

 private:
  // Primitive reuse context for MatMul op
  struct MklMatMulContext {
    // MKL-DNN memory.
    std::shared_ptr<mkldnn::memory> a_mem;
    std::shared_ptr<mkldnn::memory> b_mem;
    std::shared_ptr<mkldnn::memory> c_mem;

    // Descriptor and primitive-descriptor for MatMul.
    std::shared_ptr<matmul::desc> desc;
    std::shared_ptr<matmul::primitive_desc> prim_desc;

    // Memory descriptors.
    std::shared_ptr<mkldnn::memory::desc> a_md;
    std::shared_ptr<mkldnn::memory::desc> b_md;
    std::shared_ptr<mkldnn::memory::desc> c_md;

    // MatMul primitive.
    std::vector<mkldnn::primitive> matmul_primitives;
    std::vector<std::unordered_map<int, memory>> net_args;

    MklMatMulContext()
        : a_mem(nullptr),
          b_mem(nullptr),
          c_mem(nullptr),
          desc(nullptr),
          prim_desc(nullptr),
          a_md(nullptr),
          b_md(nullptr),
          c_md(nullptr) {}
  };

  void Setup(const MklMatMulParams& params) {
    std::shared_ptr<mkldnn::primitive> matmul_primitive = nullptr;

    // Create MatMul descriptor and primitive descriptor.
    context_.a_md.reset(
        new memory::desc({params.a_dims}, MklDnnType<T>(), params.a_strides));

    context_.b_md.reset(
        new memory::desc({params.b_dims}, MklDnnType<T>(), params.b_strides));

    context_.c_md.reset(
        new memory::desc({params.c_dims}, MklDnnType<T>(), params.c_strides));

    // Create matmul.
    context_.desc.reset(
        new matmul::desc(*context_.a_md, *context_.b_md, *context_.c_md));
    context_.prim_desc.reset(
        new matmul::primitive_desc(*context_.desc, cpu_engine_));

    // Create memory primitive based on dummy data.
    context_.a_mem.reset(
        new mkldnn::memory(*context_.a_md, cpu_engine_, DummyData));
    context_.b_mem.reset(
        new mkldnn::memory(*context_.b_md, cpu_engine_, DummyData));
    context_.c_mem.reset(
        new mkldnn::memory(*context_.b_md, cpu_engine_, DummyData));

    // Create matmul primitive.
    matmul_primitive.reset(new mkldnn::matmul(*context_.prim_desc));
    context_.net_args.push_back({{MKLDNN_ARG_SRC, *context_.a_mem},
                                 {MKLDNN_ARG_WEIGHTS, *context_.b_mem},
                                 {MKLDNN_ARG_DST, *context_.c_mem}});

    context_.matmul_primitives.push_back(*matmul_primitive);
    return;
  }

  struct MklMatMulContext context_;
};

template <typename T>
class MklMatMulPrimitiveFactory : public MklPrimitiveFactory<T> {
 public:
  static MklMatMulPrimitive<T>* Get(const MklMatMulParams& params,
                                    bool do_not_cache) {
    MklMatMulPrimitive<T>* matmul_prim = nullptr;

    if (do_not_cache) {
      // Always create new primitive
      matmul_prim = new MklMatMulPrimitive<T>(params);
    } else {
      // Try to find a suitable one in pool
      matmul_prim = dynamic_cast<MklMatMulPrimitive<T>*>(
          MklMatMulPrimitiveFactory<T>::GetInstance().GetMklMatMul(params));
      if (matmul_prim == nullptr) {
        matmul_prim = new MklMatMulPrimitive<T>(params);
        MklMatMulPrimitiveFactory<T>::GetInstance().SetMklMatMul(params,
                                                                 matmul_prim);
      }
    }

    return matmul_prim;
  }

 private:
  MklMatMulPrimitiveFactory() {}
  ~MklMatMulPrimitiveFactory() {}

  static MklMatMulPrimitiveFactory& GetInstance() {
    static MklMatMulPrimitiveFactory instance_;
    return instance_;
  }

  static string CreateKey(const MklMatMulParams& params) {
    string prefix = "matmul_";
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(prefix);
    key_creator.AddAsKey(params.a_dims);
    key_creator.AddAsKey(params.b_dims);
    key_creator.AddAsKey(params.c_dims);
    key_creator.AddAsKey(params.a_strides);
    key_creator.AddAsKey(params.b_strides);
    key_creator.AddAsKey(params.c_strides);
    key_creator.AddAsKey(typeid(T).name());

    return key_creator.GetKey();
  }

  MklPrimitive* GetMklMatMul(const MklMatMulParams& params) {
    string key = CreateKey(params);
    return this->GetOp(key);
  }

  void SetMklMatMul(const MklMatMulParams& params, MklPrimitive* op) {
    string key = CreateKey(params);
    this->SetOp(key, op);
  }
};

template <typename T>
void dnnl_gemm(char transa, char transb, int64_t m, int64_t n, int64_t k,
               float alpha, const T* a, int64_t lda, const T* b, int64_t ldb,
               float beta, T* c, int64_t ldc, OpKernelContext* ctx = nullptr) {
  using dims = mkldnn::memory::dims;

  // Prepare strides based on the transa and transb flags: transposed
  // matrices have strides swapped
  dims a_dims = dims{m, k};
  dims b_dims = dims{k, n};
  dims c_dims = dims{m, n};
  dims a_strides = tolower(transa) == 'n' ? dims{lda, 1} : dims{1, lda};
  dims b_strides = tolower(transb) == 'n' ? dims{ldb, 1} : dims{1, ldb};
  dims c_strides = dims{ldc, 1};

  // MklMatMul uses const alpha and beta, make guarantee here to ensure
  // they are never changed.
  DCHECK_EQ(alpha, 1.0f);
  DCHECK_EQ(beta, 0.f);

  MklMatMulParams params(a_dims, b_dims, c_dims, a_strides, b_strides,
                         c_strides);
  MklMatMulPrimitive<T>* matmul_prim =
      MklMatMulPrimitiveFactory<T>::Get(params, 0);

  // Execute matmul primitive.
  std::shared_ptr<stream> cpu_stream;
  MklDnnThreadPool eigen_tp(ctx);
  cpu_stream.reset(CreateStream(&eigen_tp, matmul_prim->GetEngine()));
  matmul_prim->Execute(a, b, c, cpu_stream);
}

}  // anonymous namespace

}  // namespace tensorflow

#endif  // INTEL_MKL
#endif  // TENSORFLOW_CORE_KERNELS_MKL_MKL_MATMUL_OPS_COMMON_H_
