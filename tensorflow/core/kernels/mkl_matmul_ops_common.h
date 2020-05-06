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

#ifndef TENSORFLOW_CORE_KERNELS_MKL_MATMUL_OPS_COMMON_H_
#define TENSORFLOW_CORE_KERNELS_MKL_MATMUL_OPS_COMMON_H_

#ifdef INTEL_MKL
#include <memory>
#include <string>
#include <vector>

#include "mkldnn.hpp"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/mkl_types.h"
#include "tensorflow/core/util/mkl_util.h"

using mkldnn::inner_product_forward;
using mkldnn::primitive_attr;
using mkldnn::prop_kind;
using mkldnn::stream;

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

// This structure aggregates multiple inputs to MklDnnMatMul* methods.
struct MklDnnMatMulFwdParams {
  memory::dims src_dims;
  memory::dims weight_dims;
  memory::dims bias_dims;
  memory::dims dst_dims;
  MEMORY_FORMAT src_format;
  MEMORY_FORMAT weight_format;
  string dtypes = string("");
  struct PostOpParam {
    string name;
    std::vector<float> param;
  };
  std::vector<PostOpParam> post_op_params;

  MklDnnMatMulFwdParams(memory::dims src_dims, memory::dims weight_dims,
                        memory::dims bias_dims, memory::dims dst_dims,
                        MEMORY_FORMAT src_format = MEMORY_FORMAT::any,
                        MEMORY_FORMAT weight_format = MEMORY_FORMAT::any)
      : src_dims(src_dims),
        weight_dims(weight_dims),
        bias_dims(bias_dims),
        dst_dims(dst_dims),
        src_format(src_format),
        weight_format(weight_format) {}
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
      : cpu_engine_(ENGINE_CPU, 0) {
    context_.fwd_stream.reset(new CPU_STREAM(cpu_engine_));
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
               const Tbias* bias_data, Toutput* dst_data) {
    context_.src_mem->set_data_handle(
        static_cast<void*>(const_cast<Tinput*>(src_data)));
    context_.weight_mem->set_data_handle(
        static_cast<void*>(const_cast<Tweight*>(weight_data)));
    context_.bias_mem->set_data_handle(
        static_cast<void*>(const_cast<Tbias*>(bias_data)));
    context_.dst_mem->set_data_handle(static_cast<void*>(dst_data));

#ifdef ENABLE_MKLDNN_V1
    execute_primitives(context_.fwd_primitives, context_.fwd_stream,
                       context_.net_args);
#else
    context_.fwd_stream->submit(context_.fwd_primitives);
#endif  // ENABLE_MKLDNN_V1

    // After execution, set data handle back
    context_.src_mem->set_data_handle(DummyData);
    context_.weight_mem->set_data_handle(DummyData);
    context_.bias_mem->set_data_handle(DummyData);
    context_.dst_mem->set_data_handle(DummyData);
  }

#ifndef ENABLE_MKLDNN_V1
  // In MKL-DNN v1.x, memory format tags only provide a partial description
  // of the memory layout. Hence, these functions are disabled for v1.x.
  memory::format GetSrcMemoryFormat() const { return context_.src_fmt; }
  memory::format GetWeightMemoryFormat() const { return context_.weight_fmt; }
#endif  // !ENABLE_MKLDNN_V1

  std::shared_ptr<mkldnn::inner_product_forward::primitive_desc>
  GetPrimitiveDesc() const {
    return context_.fwd_pd;
  }

 private:
  // Primitive reuse context for inner-product Fwd op
  struct MklDnnMatMulFwdContext {
#ifndef ENABLE_MKLDNN_V1
    // Expected memory format for this primitive instance
    MEMORY_FORMAT src_fmt;
    MEMORY_FORMAT weight_fmt;
#endif  // !ENABLE_MKLDNN_V1

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
    std::shared_ptr<mkldnn::stream> fwd_stream;
    std::vector<mkldnn::primitive> fwd_primitives;

#ifdef ENABLE_MKLDNN_V1
    std::vector<std::unordered_map<int, memory>> net_args;
#endif  // ENABLE_MKLDNN_V1

    MklDnnMatMulFwdContext()
        :
#ifndef ENABLE_MKLDNN_V1
          src_fmt(MEMORY_FORMAT::any),
          weight_fmt(MEMORY_FORMAT::any),
#endif  // !ENABLE_MKLDNN_V1
          src_mem(nullptr),
          weight_mem(nullptr),
          bias_mem(nullptr),
          dst_mem(nullptr),
          fwd_desc(nullptr),
          fwd_pd(nullptr),
          src_md(nullptr),
          weight_md(nullptr),
          bias_md(nullptr),
          dst_md(nullptr),
          matmul_fwd(nullptr),
          fwd_stream(nullptr) {
    }
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
                                           MEMORY_FORMAT::any));

    context_.bias_md.reset(new memory::desc({matmul_fwd_params.bias_dims},
                                            MklDnnType<Tbias>(),
                                            MEMORY_FORMAT::any));
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
        if (post_op_param.name == "relu") {
          DCHECK_EQ(post_op_param.param.size(), 3);
          float op_scale = post_op_param.param[0];
          float op_alpha = post_op_param.param[1];
          float op_beta = post_op_param.param[2];
          post_ops.append_eltwise(op_scale, ALGORITHM::eltwise_relu, op_alpha,
                                  op_beta);
        } else if (post_op_param.name == "relu6") {
          DCHECK_EQ(post_op_param.param.size(), 3);
          float op_scale = post_op_param.param[0];
          float op_alpha = post_op_param.param[1];
          float op_beta = post_op_param.param[2];
          post_ops.append_eltwise(op_scale, ALGORITHM::eltwise_bounded_relu,
                                  op_alpha, op_beta);
        } else if (post_op_param.name == "elu") {
          DCHECK_EQ(post_op_param.param.size(), 3);
          float op_scale = post_op_param.param[0];
          float op_alpha = post_op_param.param[1];
          float op_beta = post_op_param.param[2];
          post_ops.append_eltwise(op_scale, ALGORITHM::eltwise_elu, op_alpha,
                                  op_beta);
        } else if (post_op_param.name == "output_scale") {
          DCHECK_EQ(post_op_param.param.size(), 1);
          std::vector<float> scales;
          scales.push_back(post_op_param.param[0]);
          post_ops_attr.set_output_scales(0, scales);
        } else {
          DCHECK((post_op_param.name == "relu") ||
                 (post_op_param.name == "relu6") ||
                 (post_op_param.name == "elu") ||
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

#ifndef ENABLE_MKLDNN_V1
    // Store the expected memory format.
    context_.src_fmt = static_cast<mkldnn::memory::format>(
        context_.fwd_pd.get()->src_primitive_desc().desc().data.format);

    context_.weight_fmt = static_cast<mkldnn::memory::format>(
        context_.fwd_pd.get()->weights_primitive_desc().desc().data.format);
#endif  // !ENABLE_MKLDNN_V1

    // Create memory primitive based on dummy data
    context_.src_mem.reset(new MEMORY_CONSTRUCTOR(
        context_.fwd_pd.get()->PRIMITIVE_DESC_SRC, cpu_engine_, DummyData));
    context_.weight_mem.reset(new MEMORY_CONSTRUCTOR(
        context_.fwd_pd.get()->PRIMITIVE_DESC_WEIGHTS, cpu_engine_, DummyData));
    context_.dst_mem.reset(new MEMORY_CONSTRUCTOR(
        context_.fwd_pd.get()->PRIMITIVE_DESC_DST, cpu_engine_, DummyData));
    context_.bias_mem.reset(new MEMORY_CONSTRUCTOR_USING_MEM_PD(
        matmul_fwd_params.bias_dims, Tbias, MEMORY_FORMAT::x, cpu_engine_,
        DummyData));

#ifdef ENABLE_MKLDNN_V1
    // Create inner-product primitive.
    context_.matmul_fwd.reset(new inner_product_forward(*context_.fwd_pd));
    context_.net_args.push_back({{MKLDNN_ARG_SRC, *context_.src_mem},
                                 {MKLDNN_ARG_WEIGHTS, *context_.weight_mem},
                                 {MKLDNN_ARG_BIAS, *context_.bias_mem},
                                 { MKLDNN_ARG_DST,
                                   *context_.dst_mem }});
#else
    context_.matmul_fwd.reset(new inner_product_forward(
        *context_.fwd_pd, *context_.src_mem, *context_.weight_mem,
        *context_.bias_mem, *context_.dst_mem));
#endif  // ENABLE_MKLDNN_V1

    context_.fwd_primitives.push_back(*context_.matmul_fwd);
    return;
  }

  struct MklDnnMatMulFwdContext context_;
  engine cpu_engine_;
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

    // Generate keys for post-ops
    for (auto const& post_op_param : mkldnn_matmul_fwd_dims.post_op_params) {
      if (post_op_param.name == "relu" || post_op_param.name == "relu6" ||
          post_op_param.name == "elu") {
        DCHECK_EQ(post_op_param.param.size(), 3);
        key_creator.AddAsKey(post_op_param.name);
        key_creator.AddAsKey(post_op_param.param[0]);
        key_creator.AddAsKey(post_op_param.param[1]);
        key_creator.AddAsKey(post_op_param.param[2]);
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
      MKL_TENSOR_FORMAT output_tf_format, Tensor** output_tensor) {
    DCHECK(output_tensor);
    auto dst_pd = mkldnn_matmul_prim_desc.PRIMITIVE_DESC_DST;

    MklDnnShape output_mkl_shape;
    output_mkl_shape.SetMklTensor(true);
    output_mkl_shape.SetMklLayout(&dst_pd);
    output_mkl_shape.SetElemType(MklDnnType<Toutput>());
    output_mkl_shape.SetTfLayout(output_dims_mkl_order.size(),
                                 output_dims_mkl_order, output_tf_format);

    TensorShape output_tf_shape;
    output_tf_shape.AddDim((dst_pd.get_size() / sizeof(Toutput)));

    // Allocate Output Tensor
    AllocateOutputSetMklShape(context, kOutputIndexDst, output_tensor,
                              output_tf_shape, output_mkl_shape);
  }

  // TF_LOCKS_EXCLUDED annotation ensures that the lock (mu_) cannot
  // be acquired before entering the function, since it is acquired
  // inside the function.
  inline bool IsWeightCacheEmpty(OpKernelContext* context)
      TF_LOCKS_EXCLUDED(mu_) {
    tf_shared_lock lock(mu_);
    return (weight_oi_.NumElements() == 0);
  }

  // Cache the converted weight in a persistent tensor.
  // Only one thread can execute this method at any given time.
  void CacheWeight(
      OpKernelContext* context,
      const std::shared_ptr<mkldnn::inner_product_forward::primitive_desc>&
          matmul_fwd_pd,
      Tweight* weight_data, const Tensor& weight_tensor,
      MklDnnData<Tweight>& weight, const memory::desc& weight_md)
      TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock lock(mu_);
    const Tensor& weight_t = *weight_oi_.AccessTensor(context);

    // If the weights are already cached, there's nothing to do
    if (weight_t.NumElements() > 0) {
      return;
    }

    // reorder and cache the weight
    weight.SetUsrMem(weight_md, &weight_tensor);
    weight.CheckReorderToOpMem(MEMORY_PD_WITHOUT_DATA(
        matmul_fwd_pd.get()->PRIMITIVE_DESC_WEIGHTS, cpu_engine_));
    weight_data = static_cast<Tweight*>(weight.GetOpMem().get_data_handle());

    Tensor* weight_tensor_ptr = nullptr;

    size_t weight_size = matmul_fwd_pd.get()->PRIMITIVE_DESC_WEIGHTS.get_size();
    TensorShape weight_tf_shape;
    weight_tf_shape.AddDim(weight_size / sizeof(Tweight));

    OP_REQUIRES_OK(context, context->allocate_persistent(
                                DataTypeToEnum<Tweight>::value, weight_tf_shape,
                                &weight_oi_, &weight_tensor_ptr));

    void* weight_oi_t_data = weight.GetTensorBuffer(weight_tensor_ptr);
    memcpy(weight_oi_t_data, weight_data, weight_size);

// cache the memory descriptor
#ifdef ENABLE_MKLDNN_V1
    auto expected_md = GET_WEIGHTS_DESC_FROM_OP_PD(matmul_fwd_pd);
#else
    auto expected_md = GET_WEIGHTS_DESC_FROM_OP_PD(matmul_fwd_pd).desc();
#endif
    Tensor* weight_md_tensor_ptr = nullptr;
    TensorShape weight_mkl_format;
    weight_mkl_format.AddDim(sizeof(expected_md) / sizeof(Tweight));

    OP_REQUIRES_OK(
        context, context->allocate_persistent(DataTypeToEnum<Tweight>::value,
                                              weight_mkl_format, &weight_oi_md_,
                                              &weight_md_tensor_ptr));
    *reinterpret_cast<memory::desc*>(
        weight_md_tensor_ptr->flat<Tweight>().data()) = expected_md;
  }

  Tweight* GetCachedWeight(OpKernelContext* context,
                           const memory::desc& expected_md)
      TF_LOCKS_EXCLUDED(mu_) {
    tf_shared_lock lock(mu_);
    const Tensor& weight_t = *weight_oi_.AccessTensor(context);
    const Tensor& weight_md_t = *weight_oi_md_.AccessTensor(context);

    // Check if the memory descriptor of the cached weight is same as
    // expected_md. if so use the cached memory, else return NULL
    if (weight_md_t.flat<Tweight>().size()) {
      const memory::desc& stored_md =
          *(static_cast<memory::desc*>(weight_md_t.data()));
#ifdef ENABLE_MKLDNN_V1
      if (stored_md == expected_md) {
#else
      if (stored_md.data.format == expected_md.data.format) {
#endif
        return static_cast<Tweight*>(
            const_cast<Tweight*>(weight_t.flat<Tweight>().data()));
      }
    }
    return nullptr;
  }

  engine cpu_engine_ = engine(ENGINE_CPU, 0);

 protected:
  // Tensor to save reordered weight
  mutex mu_;
  PersistentTensor weight_oi_ TF_GUARDED_BY(mu_);
  PersistentTensor weight_oi_md_ TF_GUARDED_BY(mu_);

  bool is_weight_const_;

  const int kInputIndexSrc = 0;
  const int kInputIndexWeight = 1;
  const int kInputIndexBias = 2;
  const int kOutputIndexDst = 0;
};

// MatMul support for bfloat16 and int8 types is introduced in DNNLv1.2.
#ifdef ENABLE_MKLDNN_V1
namespace {

void dnnl_gemm_exec(const memory::desc& a_md, const memory::desc& b_md,
                    const memory::desc& c_md, const void* a, const void* b,
                    void* c, const primitive_attr& attr) {
  // Create a MatMul primitive
  mkldnn::engine cpu_engine = mkldnn::engine(ENGINE_CPU, 0);
  mkldnn::matmul::desc matmul_desc(a_md, b_md, c_md);
  mkldnn::matmul::primitive_desc matmul_pd(matmul_desc, attr, cpu_engine);
  mkldnn::matmul matmul_prim(matmul_pd);
  // Wrap raw pointers into DNNL memory objects
  mkldnn::memory a_memory(a_md, cpu_engine, const_cast<void*>(a));
  mkldnn::memory b_memory(b_md, cpu_engine, const_cast<void*>(b));
  mkldnn::memory c_memory(c_md, cpu_engine, c);
  // Execute the MatMul primitive.
  // Since here all shapes and parameters are static, please note that we
  // don't need to pass alpha (scales) again, as they are already hard-coded
  // in the primitive descriptor. Also, we are not allowed to change the
  // shapes of matrices A, B, and C -- they should exactly match
  // the memory descriptors passed to MatMul operation descriptor.
  mkldnn::stream s(cpu_engine);
  matmul_prim.execute(s, {{DNNL_ARG_SRC, a_memory},
                          {DNNL_ARG_WEIGHTS, b_memory},
                          { DNNL_ARG_DST,
                            c_memory }});
  s.wait();
}

template <typename T>
void dnnl_gemm_batch(const std::vector<bool>& transa,
                     const std::vector<bool>& transb, const std::vector<int>& m,
                     const std::vector<int>& n, const std::vector<int>& k,
                     const std::vector<float>& alpha, const T* a, const T* b,
                     const std::vector<float>& beta, T* c,
                     const int group_count,
                     const std::vector<int>& group_size) {
  // Current BatchMatMul support in Tensorflow is narrower than the one offered
  // by MKL and MKL-DNN. Current BatchMatMul support in Tensorflow uses only 1
  // group of size equal to batch_size, and all MatMul parameters (m, n, k,
  // alpha, beta) within that group are same.
  DCHECK(group_size.size() == 1);
  DCHECK(transa.size() == group_size[0]);
  DCHECK(transb.size() == group_size[0]);
  DCHECK(alpha.size() == group_size[0]);
  DCHECK(beta.size() == group_size[0]);
  DCHECK(m.size() == group_size[0]);
  DCHECK(n.size() == group_size[0]);
  DCHECK(k.size() == group_size[0]);
  for (int64_t idx = 0; idx < group_size[0]; idx++)
    DCHECK(transa[0] == transa[idx]);
  for (int64_t idx = 0; idx < group_size[0]; idx++)
    DCHECK(transb[0] == transb[idx]);
  for (int64_t idx = 0; idx < group_size[0]; idx++)
    DCHECK(alpha[0] == alpha[idx]);
  for (int64_t idx = 0; idx < group_size[0]; idx++)
    DCHECK(beta[0] == beta[idx]);
  for (int64_t idx = 0; idx < group_size[0]; idx++) DCHECK(m[0] == m[idx]);
  for (int64_t idx = 0; idx < group_size[0]; idx++) DCHECK(n[0] == n[idx]);
  for (int64_t idx = 0; idx < group_size[0]; idx++) DCHECK(k[0] == k[idx]);

  using dims = mkldnn::memory::dims;
  // Prepare strides based on the transa and transb flags: transposed
  // matrices have strides swapped BatchMatMul in MKL-DNN supports 3D metrices
  // so far. That is why strides are 3D also.
  dims a_sizes = dims{group_size[0], m[0], k[0]};
  dims b_sizes = dims{group_size[0], k[0], n[0]};
  dims c_sizes = dims{group_size[0], m[0], n[0]};
  dims a_strides =
      !transa[0] ? dims{m[0] * k[0], k[0], 1} : dims{k[0] * m[0], 1, m[0]};
  dims b_strides =
      !transb[0] ? dims{k[0] * n[0], n[0], 1} : dims{n[0] * k[0], 1, k[0]};
  dims c_strides = dims{m[0] * n[0], n[0], 1};

  // Prepare memory descriptors
  memory::desc a_md(a_sizes, MklDnnType<T>(), a_strides);
  memory::desc b_md(b_sizes, MklDnnType<T>(), b_strides);
  memory::desc c_md(c_sizes, MklDnnType<T>(), c_strides);
  // Create attributes (to handle alpha and beta if necessary)
  mkldnn::primitive_attr attr;
  if (alpha[0] != 1.f) attr.set_output_scales(/* mask */ 0, {alpha[0]});
  if (beta[0] != 0.f) {
    mkldnn::post_ops po;
    po.append_sum(beta[0]);
    attr.set_post_ops(po);
  }
  dnnl_gemm_exec(a_md, b_md, c_md, static_cast<const void*>(a),
                 static_cast<const void*>(b), static_cast<void*>(c), attr);
}

template <typename T>
void dnnl_gemm(char transa, char transb, int64_t m, int64_t n, int64_t k,
               float alpha, const T* a, int64_t lda, const T* b, int64_t ldb,
               float beta, float* c, int64_t ldc) {
  using dims = mkldnn::memory::dims;
  // Prepare strides based on the transa and transb flags: transposed
  // matrices have strides swapped
  dims a_strides = tolower(transa) == 'n' ? dims{lda, 1} : dims{1, lda};
  dims b_strides = tolower(transb) == 'n' ? dims{ldb, 1} : dims{1, ldb};
  // Prepare memory descriptors
  memory::desc a_md({m, k}, MklDnnType<T>(), a_strides);
  memory::desc b_md({k, n}, MklDnnType<T>(), b_strides);
  memory::desc c_md({m, n}, MklDnnType<float>(), {ldc, 1});
  // Create attributes (to handle alpha and beta if necessary)
  mkldnn::primitive_attr attr;
  if (alpha != 1.f) attr.set_output_scales(/* mask */ 0, {alpha});
  if (beta != 0.f) {
    mkldnn::post_ops po;
    po.append_sum(beta);
    attr.set_post_ops(po);
  }
  dnnl_gemm_exec(a_md, b_md, c_md, static_cast<const void*>(a),
                 static_cast<const void*>(b), static_cast<void*>(c), attr);
}

}  // anonymous namespace
#endif  // ENABLE_MKLDNN_V1

}  // namespace tensorflow

#endif  // INTEL_MKL
#endif  // TENSORFLOW_CORE_KERNELS_MKL_MATMUL_OPS_COMMON_H_
