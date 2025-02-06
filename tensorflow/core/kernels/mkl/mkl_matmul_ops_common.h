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

#if defined(INTEL_MKL)
#include <memory>
#include <string>
#include <vector>

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "dnnl.hpp"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/kernels/mkl/mkl_kernel_util.h"
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/util/onednn_env_vars.h"
#if defined(DNNL_AARCH64_USE_ACL) && defined(ENABLE_ONEDNN_OPENMP)
#include "tensorflow/core/platform/mutex.h"
#endif

using dnnl::inner_product_forward;
using dnnl::primitive_attr;
using dnnl::prop_kind;
using dnnl::stream;

namespace tensorflow {

#ifndef ENABLE_ONEDNN_V3
#define APPEND_ELTWISE(scale, alg, alpha, beta) \
  append_eltwise(scale, alg, alpha, beta)
#define APPEND_ELTWISE_RELU6(scale, alpha, beta) \
  append_eltwise(scale, dnnl::algorithm::eltwise_bounded_relu, alpha, beta)
#define OUTPUT_SCALE_DCHECK (post_op_param.name == "output_scale")
#define SET_MKL_LAYOUT(md) SetMklLayout(&md)
#define TSCALED_BIAS Tbias
#else
#define APPEND_ELTWISE(scale, alg, alpha, beta) \
  append_eltwise(alg, alpha, beta);             \
  (void)scale
#define APPEND_ELTWISE_RELU6(scale, alpha, beta)             \
  append_eltwise(dnnl::algorithm::eltwise_clip, 0.0, alpha); \
  (void)scale;                                               \
  (void)beta
#define OUTPUT_SCALE_DCHECK                  \
  (post_op_param.name == "src_scale") ||     \
      (post_op_param.name == "wei_scale") || \
      (post_op_param.name == "dst_scale")
#define SET_MKL_LAYOUT(md) SetMklLayout(md)
#define TSCALED_BIAS float
#endif  // !ENABLE_ONEDNN_V3

#if !defined(ENABLE_ONEDNN_OPENMP) && !defined(ENABLE_ONEDNN_V3)
#define FWD_STREAM , *fwd_stream
#else
#define FWD_STREAM
#endif  // !ENABLE_ONEDNN_OPENMP && !ENABLE_ONEDNN_V3

static Eigen::internal::CacheSizes cache_sizes = Eigen::internal::CacheSizes();

typedef Eigen::ThreadPoolDevice CPUDevice;
inline bool ExecuteSingleThreadedGemm(int64_t m, int64_t n, int64_t k,
                                      int bytes) {
  // Ideally we would like to determine blocking and then come up with
  // a heuristic but what we are targeting are very small models whose
  // total size is < x*L2. So we will do this simple calculation
  // to determine if the matrix multiplication should be run on a single thread.
  // TODO(Intel-tf): this needs to be vastly improved, perhaps at a lower level
  // than the integration.
  ptrdiff_t l2_size = cache_sizes.m_l2;
  constexpr float kHeuristicMultiplier = 1.01;
  const float mul_size = bytes * (m * n + k * (m + n));
  const float l2_heur = l2_size * kHeuristicMultiplier;
  return (mul_size >= 0 && mul_size < l2_heur);
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
  bool const_weight;
  struct PostOpParam {
    string name;
    std::vector<float> param;
    string partial_key;
  };
  std::vector<PostOpParam> post_op_params;
  string input_quant_mode;

  MklDnnMatMulFwdParams(
      memory::dims src_dims, memory::dims weight_dims, memory::dims bias_dims,
      memory::dims dst_dims,
      memory::format_tag src_format = memory::format_tag::any,
      memory::format_tag weight_format = memory::format_tag::any,
      memory::format_tag dst_format = memory::format_tag::any,
      bool const_weight = false)
      : src_dims(src_dims),
        weight_dims(weight_dims),
        bias_dims(bias_dims),
        dst_dims(dst_dims),
        src_format(src_format),
        weight_format(weight_format),
        dst_format(dst_format),
        const_weight(const_weight) {}
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

  dnnl::memory::desc GetScratchPadDesc() {
    return context_.fwd_pd->scratchpad_desc();
  }

  // Inner-product forward execute with bias:
  //  - src_data: input data buffer of src
  //  - weight_data: input data buffer of weight
  //  - bias_data: input data buffer of bias
  //  - dst_data: output data buffer of dst
  //  - sp_data: scratchpad data
  void Execute(const Tinput* src_data, const Tweight* weight_data,
               const void* bias_data, Toutput* dst_data,
               const MklDnnMatMulFwdParams& matmul_fwd_params, void* sp_data,
               std::shared_ptr<stream> fwd_stream) {
#if defined(DNNL_AARCH64_USE_ACL) && defined(ENABLE_ONEDNN_OPENMP)
    mutex_lock lock(primitive_execution_mu_);
#endif
    context_.src_mem->set_data_handle(
        static_cast<void*>(const_cast<Tinput*>(src_data)) FWD_STREAM);
    context_.weight_mem->set_data_handle(
        static_cast<void*>(const_cast<Tweight*>(weight_data)) FWD_STREAM);
    context_.bias_mem->set_data_handle(const_cast<void*>(bias_data) FWD_STREAM);
    context_.dst_mem->set_data_handle(static_cast<void*>(dst_data) FWD_STREAM);
    context_.sp_mem->set_data_handle(sp_data FWD_STREAM);
    auto const& post_op_params = matmul_fwd_params.post_op_params;
    if (!post_op_params.empty()) {
      for (auto const& post_op_param : post_op_params) {
        if (post_op_param.name == "src_scale") {
          context_.src_scale_mem->set_data_handle(static_cast<void*>(
              const_cast<float*>(post_op_param.param.data())) FWD_STREAM);
        } else if (post_op_param.name == "wei_scale") {
          context_.wei_scale_mem->set_data_handle(static_cast<void*>(
              const_cast<float*>(post_op_param.param.data())) FWD_STREAM);
        } else if (post_op_param.name == "dst_scale") {
          context_.dst_scale_mem->set_data_handle(static_cast<void*>(
              const_cast<float*>(post_op_param.param.data())) FWD_STREAM);
        }
      }
    }

    execute_primitives(context_.fwd_primitives, fwd_stream, context_.net_args);

    // After execution, set data handle back
    context_.src_mem->set_data_handle(DummyData);
    context_.weight_mem->set_data_handle(DummyData);
    context_.bias_mem->set_data_handle(DummyData);
    context_.dst_mem->set_data_handle(DummyData);
  }

  std::shared_ptr<dnnl::inner_product_forward::primitive_desc>
  GetPrimitiveDesc() const {
    return context_.fwd_pd;
  }

 private:
  // Primitive reuse context for inner-product Fwd op
  struct MklDnnMatMulFwdContext {
    // oneDNN memory.
    std::shared_ptr<dnnl::memory> src_mem;
    std::shared_ptr<dnnl::memory> weight_mem;
    std::shared_ptr<dnnl::memory> bias_mem;
    std::shared_ptr<dnnl::memory> dst_mem;
    std::shared_ptr<dnnl::memory> sp_mem;
    // Quantization scale related memory
    std::shared_ptr<dnnl::memory> src_scale_mem;
    std::shared_ptr<dnnl::memory> wei_scale_mem;
    std::shared_ptr<dnnl::memory> dst_scale_mem;

    // Descriptor and primitive-descriptor for forward inner-product.
#ifndef ENABLE_ONEDNN_V3
    std::shared_ptr<dnnl::inner_product_forward::desc> fwd_desc;
#endif  // !ENABLE_ONEDNN_V3
    std::shared_ptr<dnnl::inner_product_forward::primitive_desc> fwd_pd;

    // Memory descriptors.
    std::shared_ptr<dnnl::memory::desc> src_md;
    std::shared_ptr<dnnl::memory::desc> weight_md;
    std::shared_ptr<dnnl::memory::desc> bias_md;
    std::shared_ptr<dnnl::memory::desc> dst_md;
    // Quantization scale related memory descriptors
    std::shared_ptr<dnnl::memory::desc> src_scale_md;
    std::shared_ptr<dnnl::memory::desc> wei_scale_md;
    std::shared_ptr<dnnl::memory::desc> dst_scale_md;

    // Inner-product primitive.
    std::shared_ptr<dnnl::primitive> matmul_fwd;
    std::vector<dnnl::primitive> fwd_primitives;

    std::vector<std::unordered_map<int, memory>> net_args;

    MklDnnMatMulFwdContext()
        : src_mem(nullptr),
          weight_mem(nullptr),
          bias_mem(nullptr),
          dst_mem(nullptr),
          sp_mem(nullptr),
          src_scale_mem(nullptr),
          wei_scale_mem(nullptr),
          dst_scale_mem(nullptr),
#ifndef ENABLE_ONEDNN_V3
          fwd_desc(nullptr),
#endif  // ENABLE_ONEDNN_V3
          fwd_pd(nullptr),
          src_md(nullptr),
          weight_md(nullptr),
          bias_md(nullptr),
          dst_md(nullptr),
          src_scale_md(nullptr),
          wei_scale_md(nullptr),
          dst_scale_md(nullptr),
          matmul_fwd(nullptr) {
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
#ifdef DNNL_AARCH64_USE_ACL
                                              memory::format_tag::any));
#else
                                              matmul_fwd_params.weight_format));
#endif

    context_.dst_md.reset(new memory::desc({matmul_fwd_params.dst_dims},
                                           MklDnnType<Toutput>(),
                                           matmul_fwd_params.dst_format));

    memory::data_type bias_dt;
#ifndef ENABLE_ONEDNN_V3
    bias_dt = MklDnnType<Tbias>();
#else
    if (std::is_same<Tweight, qint8>::value) {
      // For QuantizedMatMul, bias needs to be passed to oneDNN as float of
      // bfloat16 (even if Tbias is qint32).
      if (std::is_same<Tbias, bfloat16>::value &&
          matmul_fwd_params.input_quant_mode == "SCALED") {
        bias_dt = MklDnnType<bfloat16>();
      } else {
        bias_dt = MklDnnType<float>();
      }
    } else {
      bias_dt = MklDnnType<Tbias>();
    }
#endif  // !ENABLE_ONEDNN_V3
    context_.bias_md.reset(new memory::desc({matmul_fwd_params.bias_dims},
                                            bias_dt, memory::format_tag::any));

    // Create an inner-product.
#ifndef ENABLE_ONEDNN_V3
    context_.fwd_desc.reset(new inner_product_forward::desc(
        matmul_fwd_params.const_weight ? prop_kind::forward_inference
                                       : prop_kind::forward_training,
        *context_.src_md, *context_.weight_md, *context_.bias_md,
        *context_.dst_md));
    context_.fwd_pd.reset(new inner_product_forward::primitive_desc(
        *context_.fwd_desc, cpu_engine_));
#endif  // !ENABLE_ONEDNN_V3

    // Check if there is any fusion as post-ops
    auto const& post_op_params = matmul_fwd_params.post_op_params;
    dnnl::primitive_attr post_ops_attr;
    post_ops_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    dnnl::post_ops post_ops;
    std::unordered_map<string, bool> is_scale_set;
    if (!post_op_params.empty()) {
      for (auto const& post_op_param : post_op_params) {
        if (post_op_param.name == "Relu" || post_op_param.name == "LeakyRelu") {
          DCHECK_EQ(post_op_param.param.size(), 3);
          float op_scale = post_op_param.param[0];
          float op_alpha = post_op_param.param[1];
          float op_beta = post_op_param.param[2];
          post_ops.APPEND_ELTWISE(op_scale, dnnl::algorithm::eltwise_relu,
                                  op_alpha, op_beta);
        } else if (post_op_param.name == "Relu6") {
          DCHECK_EQ(post_op_param.param.size(), 3);
          float op_scale = post_op_param.param[0];
          float op_alpha = post_op_param.param[1];
          float op_beta = post_op_param.param[2];
          post_ops.APPEND_ELTWISE_RELU6(op_scale, op_alpha, op_beta);
        } else if (post_op_param.name == "Elu") {
          DCHECK_EQ(post_op_param.param.size(), 3);
          float op_scale = post_op_param.param[0];
          float op_alpha = post_op_param.param[1];
          float op_beta = post_op_param.param[2];
          post_ops.APPEND_ELTWISE(op_scale, dnnl::algorithm::eltwise_elu,
                                  op_alpha, op_beta);
        } else if (post_op_param.name == "GeluApproximate") {
          DCHECK_EQ(post_op_param.param.size(), 3);
          float op_scale = post_op_param.param[0];
          float op_alpha = post_op_param.param[1];
          float op_beta = post_op_param.param[2];
          post_ops.APPEND_ELTWISE(op_scale, dnnl::algorithm::eltwise_gelu_tanh,
                                  op_alpha, op_beta);
        } else if (post_op_param.name == "GeluExact") {
          DCHECK_EQ(post_op_param.param.size(), 3);
          float op_scale = post_op_param.param[0];
          float op_alpha = post_op_param.param[1];
          float op_beta = post_op_param.param[2];
          post_ops.APPEND_ELTWISE(op_scale, dnnl::algorithm::eltwise_gelu_erf,
                                  op_alpha, op_beta);
        } else if (post_op_param.name == "Tanh") {
          DCHECK_EQ(post_op_param.param.size(), 3);
          float op_scale = post_op_param.param[0];
          float op_alpha = post_op_param.param[1];
          float op_beta = post_op_param.param[2];
          post_ops.APPEND_ELTWISE(op_scale, dnnl::algorithm::eltwise_tanh,
                                  op_alpha, op_beta);
        } else if (post_op_param.name == "Sigmoid") {
          DCHECK_EQ(post_op_param.param.size(), 3);
          float op_scale = post_op_param.param[0];
          float op_alpha = post_op_param.param[1];
          float op_beta = post_op_param.param[2];
          post_ops.APPEND_ELTWISE(op_scale, dnnl::algorithm::eltwise_logistic,
                                  op_alpha, op_beta);
        } else if (post_op_param.name == "linear") {
          DCHECK_EQ(post_op_param.param.size(), 3);
          float op_scale = post_op_param.param[0];
          float op_alpha = post_op_param.param[1];
          float op_beta = post_op_param.param[2];
          post_ops.APPEND_ELTWISE(op_scale, dnnl::algorithm::eltwise_linear,
                                  op_alpha, op_beta);
#ifndef ENABLE_ONEDNN_V3
        } else if (post_op_param.name == "output_scale") {
          if (post_op_param.param.size() == 1) {
            post_ops_attr.set_output_scales(0, post_op_param.param);
          } else {
            post_ops_attr.set_output_scales(2, post_op_param.param);
          }
#else
        } else if (post_op_param.name == "src_scale") {
          is_scale_set.insert({"src", true});
          post_ops_attr.set_scales_mask(DNNL_ARG_SRC, 0);
          context_.src_scale_md.reset(new memory::desc({1}, MklDnnType<float>(),
                                                       memory::format_tag::x));
          context_.src_scale_mem.reset(
              new memory(*context_.src_scale_md, cpu_engine_, DummyData));
        } else if (post_op_param.name == "wei_scale") {
          is_scale_set.insert({"wei", true});
          const int scale_size = post_op_param.param.size();
          const int mask = scale_size == 1 ? 0 : 1;
          post_ops_attr.set_scales_mask(DNNL_ARG_WEIGHTS, mask);
          context_.wei_scale_md.reset(new memory::desc(
              {scale_size}, MklDnnType<float>(), memory::format_tag::x));
          context_.wei_scale_mem.reset(
              new memory(*context_.wei_scale_md, cpu_engine_, DummyData));
        } else if (post_op_param.name == "dst_scale") {
          is_scale_set.insert({"dst", true});
          const int scale_size = post_op_param.param.size();
          const int mask = scale_size == 1 ? 0 : 1;
          post_ops_attr.set_scales_mask(DNNL_ARG_DST, mask);
          context_.dst_scale_md.reset(new memory::desc({1}, MklDnnType<float>(),
                                                       memory::format_tag::x));
          context_.dst_scale_mem.reset(
              new memory(*context_.dst_scale_md, cpu_engine_, DummyData));
#endif  // !ENABLE_ONEDNN_V3
        } else if (post_op_param.name == "sum") {
          DCHECK_EQ(post_op_param.param.size(), 1);
          float op_scale = post_op_param.param[0];
          post_ops.append_sum(op_scale);

        } else {
          DCHECK((post_op_param.name == "Relu") ||
                 (post_op_param.name == "Relu6") ||
                 (post_op_param.name == "Elu") ||
                 (post_op_param.name == "GeluApproximate") ||
                 (post_op_param.name == "GeluExact") ||
                 (post_op_param.name == "Tanh") ||
                 (post_op_param.name == "Sigmoid") ||
                 (post_op_param.name == "sum") ||
                 (post_op_param.name == "Leakyrelu") || OUTPUT_SCALE_DCHECK);
        }
      }
      post_ops_attr.set_post_ops(post_ops);
    }

#ifndef ENABLE_ONEDNN_V3
    context_.fwd_pd.reset(new inner_product_forward::primitive_desc(
        *context_.fwd_desc, post_ops_attr, cpu_engine_));
#else
    context_.fwd_pd.reset(new inner_product_forward::primitive_desc(
        cpu_engine_,
        matmul_fwd_params.const_weight ? prop_kind::forward_inference
                                       : prop_kind::forward_training,
        *context_.src_md, *context_.weight_md, *context_.bias_md,
        *context_.dst_md, post_ops_attr));
#endif  // !ENABLE_ONEDNN_V3

    // Create memory primitive based on dummy data
    context_.src_mem.reset(
        new memory(context_.fwd_pd.get()->src_desc(), cpu_engine_, DummyData));
    context_.weight_mem.reset(new memory(context_.fwd_pd.get()->weights_desc(),
                                         cpu_engine_, DummyData));
    context_.dst_mem.reset(
        new memory(context_.fwd_pd.get()->dst_desc(), cpu_engine_, DummyData));
    context_.bias_mem.reset(
        new memory(context_.fwd_pd.get()->bias_desc(), cpu_engine_, DummyData));
    auto scratchpad_md = context_.fwd_pd->scratchpad_desc();
    context_.sp_mem.reset(
        new dnnl::memory(scratchpad_md, cpu_engine_, DummyData));

    // Create inner-product primitive.
    context_.matmul_fwd.reset(new inner_product_forward(*context_.fwd_pd));
    std::unordered_map<int, memory> net_args = {
        {DNNL_ARG_SRC, *context_.src_mem},
        {DNNL_ARG_WEIGHTS, *context_.weight_mem},
        {DNNL_ARG_BIAS, *context_.bias_mem},
        {DNNL_ARG_SCRATCHPAD, *context_.sp_mem},
        {DNNL_ARG_DST, *context_.dst_mem}};
#ifdef ENABLE_ONEDNN_V3
    if (is_scale_set["src"]) {
      net_args.insert(
          {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, *context_.src_scale_mem});
    }
    if (is_scale_set["wei"]) {
      net_args.insert(
          {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, *context_.wei_scale_mem});
    }
    if (is_scale_set["dst"]) {
      net_args.insert(
          {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, *context_.dst_scale_mem});
    }
#endif  // ENABLE_ONEDNN_V3
    context_.net_args.push_back(net_args);
    context_.fwd_primitives.push_back(*context_.matmul_fwd);
    return;
  }

  struct MklDnnMatMulFwdContext context_;

#if defined(DNNL_AARCH64_USE_ACL) && defined(ENABLE_ONEDNN_OPENMP)
  // Guards Execution()
  mutex primitive_execution_mu_;
#endif
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
      if (post_op_param.name == "Relu" || post_op_param.name == "Relu6" ||
          post_op_param.name == "Elu" || post_op_param.name == "Tanh" ||
          post_op_param.name == "Sigmoid" ||
          post_op_param.name == "LeakyRelu" ||
          post_op_param.name == "GeluApproximate" ||
          post_op_param.name == "GeluExact" || post_op_param.name == "linear") {
        DCHECK_EQ(post_op_param.param.size(), 3);
        key_creator.AddAsKey(post_op_param.name);
        key_creator.AddAsKey(post_op_param.param[0]);
        key_creator.AddAsKey(post_op_param.param[1]);
        key_creator.AddAsKey(post_op_param.param[2]);
      } else if (post_op_param.name == "sum") {
        DCHECK_EQ(post_op_param.param.size(), 1);
        key_creator.AddAsKey(post_op_param.name);
        key_creator.AddAsKey(post_op_param.param[0]);
#ifndef ENABLE_ONEDNN_V3
      } else if (post_op_param.name == "output_scale") {
#else
      } else if (post_op_param.name == "src_scale" ||
                 post_op_param.name == "wei_scale" ||
                 post_op_param.name == "dst_scale") {
#endif  // !ENABLE_ONEDNN_V3
        key_creator.AddAsKey(post_op_param.name);
        if (post_op_param.partial_key.empty()) {
          DCHECK_GE(post_op_param.param.size(), 1);
          // Old Quantized MatMul kernels do not create part of key beforehand
          // as primitive caching-key-creation optimization.
          key_creator.AddAsKey(post_op_param.param[0]);
        } else {
          // New Quantized MatMul kernels pre-create partial key.
          key_creator.AddAsKey(post_op_param.partial_key);
        }
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

template <class Tweight, class Tbias, class Toutput>
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
    output_mkl_shape.SET_MKL_LAYOUT(dst_pd);
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
      const std::shared_ptr<dnnl::inner_product_forward::primitive_desc>&
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

#ifdef ENABLE_ONEDNN_V3
    // For now, cache weights only for blocked format
    if (weight_md.get_format_kind() != memory::format_kind::blocked) {
      return;
    }
#endif  // ENABLE_ONEDNN_V3

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
#ifndef ENABLE_ONEDNN_V3
    TensorShape weight_mkl_format;
    weight_mkl_format.AddDim(sizeof(expected_md) / sizeof(Tweight));

    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<Tweight>::value,
                                          weight_mkl_format, &weight_oi_md_));
    *reinterpret_cast<memory::desc*>(weight_oi_md_.flat<Tweight>().data()) =
        expected_md;
#else
    weight_oi_md_ = FilterMemoryDesc(
        expected_md.get_ndims(), expected_md.get_inner_nblks(),
        expected_md.get_data_type(), expected_md.get_dims(),
        expected_md.get_inner_blks(), expected_md.get_inner_idxs(),
        expected_md.get_strides());
#endif  // !ENABLE_ONEDNN_V3
  }

  Tweight* GetCachedWeight(OpKernelContext* context,
                           const memory::desc& expected_md)
      TF_LOCKS_EXCLUDED(mu_) {
    tf_shared_lock lock(mu_);
    const Tensor& weight_t = weight_oi_;
#ifndef ENABLE_ONEDNN_V3
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
#else
    // Return the cached weights only if the dimensions of the cached weights
    // and the current weights match. Otherwise, return nullptr.
    //
    // TODO(intel-tf): The following check assumes that all dimensions are
    // known before checking for equality. We may have to modify it in the
    // future once we support runtime dimensions (especially if the dimensions
    // are still unknown at this point).
    if (weight_oi_md_ ==
        FilterMemoryDesc(expected_md.get_ndims(), expected_md.get_inner_nblks(),
                         expected_md.get_data_type(), expected_md.get_dims(),
                         expected_md.get_inner_blks(),
                         expected_md.get_inner_idxs(),
                         expected_md.get_strides())) {
      return static_cast<Tweight*>(
          const_cast<Tweight*>(weight_t.flat<Tweight>().data()));
    }
    return nullptr;
#endif  // !ENABLE_ONEDNN_V3
  }

  bool IsBiasCacheEmpty() TF_LOCKS_EXCLUDED(bias_cache_mutex_) {
    tf_shared_lock lock(bias_cache_mutex_);
    return (cached_bias_data_pt_.NumElements() == 0);
  }

  virtual bool IsCachedBiasValid(float, float)
      TF_SHARED_LOCKS_REQUIRED(bias_cache_mutex_) {
    return false;
  }

  void CacheBias(OpKernelContext* ctx, const Tensor& temp_scaled_bias_tensor,
                 float min_input, float max_input)
      TF_LOCKS_EXCLUDED(bias_cache_mutex_) {
    mutex_lock lock(bias_cache_mutex_);
    if (cached_bias_data_pt_.NumElements() > 0) {
      return;
    }
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(temp_scaled_bias_tensor.dtype(),
                                           temp_scaled_bias_tensor.shape(),
                                           &cached_bias_data_pt_));
    tensor::DeepCopy(temp_scaled_bias_tensor, &cached_bias_data_pt_);
    saved_min_input_ = min_input;
    saved_max_input_ = max_input;
  }

  void GetCachedBias(float min_input, float max_input, void** bias_data)
      TF_LOCKS_EXCLUDED(bias_cache_mutex_) {
    tf_shared_lock lock(bias_cache_mutex_);
    const Tensor& cached_bias_data = cached_bias_data_pt_;
    if (IsCachedBiasValid(min_input, max_input)) {
      *bias_data = static_cast<void*>(const_cast<TSCALED_BIAS*>(
          cached_bias_data.flat<TSCALED_BIAS>().data()));
    } else {
      *bias_data = nullptr;
    }
  }

  engine cpu_engine_ = engine(engine::kind::cpu, 0);

 protected:
  // Tensor to save reordered weight
  mutex mu_;
  Tensor weight_oi_ TF_GUARDED_BY(mu_);
#ifndef ENABLE_ONEDNN_V3
  Tensor weight_oi_md_ TF_GUARDED_BY(mu_);
#else
  FilterMemoryDesc weight_oi_md_ TF_GUARDED_BY(mu_);
#endif  // !ENABLE_ONEDNN_V3

  bool is_weight_const_;

  bool is_bias_const_;
  mutex bias_cache_mutex_;
  // Persistent tensor for cached bias.
  Tensor cached_bias_data_pt_ TF_GUARDED_BY(bias_cache_mutex_);
  float saved_min_input_ = -std::numeric_limits<float>::infinity();
  float saved_max_input_ = std::numeric_limits<float>::infinity();

  const int kInputIndexSrc = 0;
  const int kInputIndexWeight = 1;
  const int kInputIndexBias = 2;
  const int kOutputIndexDst = 0;
};

using dnnl::matmul;

namespace {

struct MklMatMulParams {
  string prefix;
  memory::dims a_dims;
  memory::dims b_dims;
  memory::dims c_dims;
  memory::dims a_strides;
  memory::dims b_strides;
  memory::dims c_strides;
  memory::dim a_nnz;
  struct PostOpParam {
    string name;
    std::vector<float> param;
    memory::dims dims;
    memory::data_type data_type;
    memory::format_tag format_tag;
  };
  std::vector<PostOpParam> post_op_params;

  MklMatMulParams(string prefix, memory::dims a_dims, memory::dims b_dims,
                  memory::dims c_dims, memory::dims a_strides,
                  memory::dims b_strides, memory::dims c_strides,
                  memory::dim a_nnz = 0)
      : prefix(prefix),
        a_dims(a_dims),
        b_dims(b_dims),
        c_dims(c_dims),
        a_strides(a_strides),
        b_strides(b_strides),
        c_strides(c_strides),
        a_nnz(a_nnz) {}
};

template <typename Tlhs, typename Trhs, typename Toutput, bool CSR = false>
class MklMatMulPrimitive : public MklPrimitive {
 public:
  explicit MklMatMulPrimitive(const MklMatMulParams& params)
      : MklPrimitive(engine(engine::kind::cpu, 0)) {
    // Create matmul primitive
    Setup(params);
  }

  ~MklMatMulPrimitive() {}

  dnnl::memory::desc GetScratchPadDesc() {
    return context_.prim_desc->scratchpad_desc();
  }

  void Execute(const std::shared_ptr<stream>& stream, const Tlhs* a_data,
               const Trhs* b_data, const Toutput* c_data, void* sp_data,
               void* mul_data = nullptr, void* add_data = nullptr,
               const int32_t* a_col_indices = nullptr,
               const int32_t* a_row_pointers = nullptr) {
#if defined(DNNL_AARCH64_USE_ACL) && defined(ENABLE_ONEDNN_OPENMP)
    mutex_lock lock(primitive_execution_mu_);
#endif
#if !defined(ENABLE_ONEDNN_OPENMP) && !defined(ENABLE_ONEDNN_V3)
    context_.a_mem->set_data_handle(
        static_cast<void*>(const_cast<Tlhs*>(a_data)), *stream);
    context_.b_mem->set_data_handle(
        static_cast<void*>(const_cast<Trhs*>(b_data)), *stream);
    context_.c_mem->set_data_handle(
        static_cast<void*>(const_cast<Toutput*>(c_data)), *stream);

    if (sp_data != nullptr) context_.sp_mem->set_data_handle(sp_data, *stream);
    if (mul_data != nullptr)
      context_.mul_mem->set_data_handle(mul_data, *stream);
    if (add_data != nullptr)
      context_.add_mem->set_data_handle(add_data, *stream);
#else
    if constexpr (CSR) {
      context_.a_mem->set_data_handle(
          static_cast<void*>(const_cast<Tlhs*>(a_data)), 0);
      context_.a_mem->set_data_handle(
          static_cast<void*>(const_cast<int32_t*>(a_col_indices)), 1);
      context_.a_mem->set_data_handle(
          static_cast<void*>(const_cast<int32_t*>(a_row_pointers)), 2);
    } else {
      context_.a_mem->set_data_handle(
          static_cast<void*>(const_cast<Tlhs*>(a_data)));
    }
    context_.b_mem->set_data_handle(
        static_cast<void*>(const_cast<Trhs*>(b_data)));
    context_.c_mem->set_data_handle(
        static_cast<void*>(const_cast<Toutput*>(c_data)));
    if (sp_data != nullptr) context_.sp_mem->set_data_handle(sp_data);
    if (mul_data != nullptr) context_.mul_mem->set_data_handle(mul_data);
    if (add_data != nullptr) context_.add_mem->set_data_handle(add_data);
#endif  // !ENABLE_ONEDNN_OPENMP && !ENABLE_ONEDNN_V3
    execute_primitives(context_.matmul_primitives, stream, context_.net_args);

    // After execution, set data handle back
    context_.a_mem->set_data_handle(DummyData);
    context_.b_mem->set_data_handle(DummyData);
    context_.c_mem->set_data_handle(DummyData);
    if (sp_data != nullptr) context_.sp_mem->set_data_handle(DummyData);
    if (mul_data != nullptr) context_.mul_mem->set_data_handle(DummyData);
    if (add_data != nullptr) context_.add_mem->set_data_handle(DummyData);
  }

  std::shared_ptr<dnnl::matmul::primitive_desc> GetPrimitiveDesc() const {
    return context_.prim_desc;
  }

 private:
  // Primitive reuse context for MatMul op
  struct MklMatMulContext {
    // oneDNN memory.
    std::shared_ptr<dnnl::memory> a_mem;
    std::shared_ptr<dnnl::memory> b_mem;
    std::shared_ptr<dnnl::memory> c_mem;
    std::shared_ptr<dnnl::memory> mul_mem;
    std::shared_ptr<dnnl::memory> add_mem;
    std::shared_ptr<dnnl::memory> sp_mem;

    // Descriptor and primitive-descriptor for MatMul.
#ifndef ENABLE_ONEDNN_V3
    std::shared_ptr<matmul::desc> desc;
#endif  // !ENABLE_ONEDNN_V3
    std::shared_ptr<matmul::primitive_desc> prim_desc;

    // Memory descriptors.
    std::shared_ptr<dnnl::memory::desc> a_md;
    std::shared_ptr<dnnl::memory::desc> b_md;
    std::shared_ptr<dnnl::memory::desc> c_md;
    std::shared_ptr<dnnl::memory::desc> mul_md;
    std::shared_ptr<dnnl::memory::desc> add_md;

    // MatMul primitive.
    std::vector<dnnl::primitive> matmul_primitives;
    std::vector<std::unordered_map<int, memory>> net_args;

    MklMatMulContext()
        : a_mem(nullptr),
          b_mem(nullptr),
          c_mem(nullptr),
          mul_mem(nullptr),
          add_mem(nullptr),
          sp_mem(nullptr),
#ifndef ENABLE_ONEDNN_V3
          desc(nullptr),
#endif  // !ENABLE_ONEDNN_V3
          prim_desc(nullptr),
          a_md(nullptr),
          b_md(nullptr),
          c_md(nullptr),
          mul_md(nullptr),
          add_md(nullptr) {
    }
  };

  void Setup(const MklMatMulParams& params) {
    std::shared_ptr<dnnl::primitive> matmul_primitive = nullptr;

    // Create MatMul descriptor and primitive descriptor.
    if constexpr (CSR) {
      // If it's a CSR matrix.
#ifdef ENABLE_ONEDNN_V3
      const auto tmp = memory::desc::csr(
          params.a_dims, MklDnnType<Tlhs>(), params.a_nnz,
          dnnl::memory::data_type::s32, dnnl::memory::data_type::s32);
      context_.a_md.reset(new memory::desc(tmp));
#endif  // ENABLE_ONEDNN_V3
    } else {
      context_.a_md.reset(new memory::desc({params.a_dims}, MklDnnType<Tlhs>(),
                                           params.a_strides));
    }

    context_.b_md.reset(new memory::desc({params.b_dims}, MklDnnType<Trhs>(),
#ifdef DNNL_AARCH64_USE_ACL
                                         memory::format_tag::any));
#else
                                         params.b_strides));
#endif
    context_.c_md.reset(new memory::desc({params.c_dims}, MklDnnType<Toutput>(),
                                         params.c_strides));

    // Create matmul.
#ifndef ENABLE_ONEDNN_V3
    context_.desc.reset(
        new matmul::desc(*context_.a_md, *context_.b_md, *context_.c_md));
#endif  // !ENABLE_ONEDNN_V3

    // Check if there is any fusion as post-ops
    auto const& post_op_params = params.post_op_params;
    dnnl::primitive_attr post_ops_attr;
    dnnl::post_ops post_ops;
    if (!post_op_params.empty()) {
      for (auto const& post_op_param : post_op_params) {
        if (post_op_param.name == "output_scale") {
#ifndef ENABLE_ONEDNN_V3
          // TODO(intel-tf): Verify if this code is needed. If not, it needs to
          // be removed.
          DCHECK_EQ(post_op_param.param.size(), 1);
          std::vector<float> scales;
          scales.push_back(post_op_param.param[0]);
          post_ops_attr.set_output_scales(0, scales);
#endif  // !ENABLE_ONEDNN_V3
        } else if (post_op_param.name == "mul") {
          context_.mul_md.reset(new memory::desc({post_op_param.dims},
                                                 post_op_param.data_type,
                                                 post_op_param.format_tag));
          post_ops.append_binary(dnnl::algorithm::binary_mul, *context_.mul_md);
        } else if (post_op_param.name == "add") {
          context_.add_md.reset(new memory::desc({post_op_param.dims},
                                                 post_op_param.data_type,
                                                 post_op_param.format_tag));
          post_ops.append_binary(dnnl::algorithm::binary_add, *context_.add_md);
        } else {
          DCHECK((post_op_param.name == "output_scale"));
        }
      }
      post_ops_attr.set_post_ops(post_ops);
    }
    post_ops_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#ifndef ENABLE_ONEDNN_V3
    context_.prim_desc.reset(
        new matmul::primitive_desc(*context_.desc, post_ops_attr, cpu_engine_));
#else
    context_.prim_desc.reset(
        new matmul::primitive_desc(cpu_engine_, *context_.a_md, *context_.b_md,
                                   *context_.c_md, post_ops_attr));
#endif  // !ENABLE_ONEDNN_V3

    // Create memory primitive based on dummy data.
    if constexpr (CSR) {
      context_.a_mem.reset(new dnnl::memory(*context_.a_md, cpu_engine_,
                                            std::vector<void*>(3, DummyData)));
    } else {
      context_.a_mem.reset(
          new dnnl::memory(*context_.a_md, cpu_engine_, DummyData));
    }
#ifdef DNNL_AARCH64_USE_ACL
    context_.b_mem.reset(new dnnl::memory(
        context_.prim_desc.get()->weights_desc(), cpu_engine_, DummyData));
#else
    context_.b_mem.reset(
        new dnnl::memory(*context_.b_md, cpu_engine_, DummyData));
#endif
    context_.c_mem.reset(
        new dnnl::memory(*context_.c_md, cpu_engine_, DummyData));
    auto scratchpad_md = context_.prim_desc->scratchpad_desc();
    context_.sp_mem.reset(
        new dnnl::memory(scratchpad_md, cpu_engine_, DummyData));

    // Create matmul primitive.
    matmul_primitive.reset(new dnnl::matmul(*context_.prim_desc));
    context_.net_args.push_back({{DNNL_ARG_SRC, *context_.a_mem},
                                 {DNNL_ARG_WEIGHTS, *context_.b_mem},
                                 {DNNL_ARG_SCRATCHPAD, *context_.sp_mem},
                                 {DNNL_ARG_DST, *context_.c_mem}});
    if (!post_op_params.empty()) {
      int count = 0;
      for (auto const& post_op_param : post_op_params) {
        if (post_op_param.name == "mul") {
          context_.mul_mem.reset(
              new dnnl::memory(*context_.mul_md, cpu_engine_, DummyData));
          context_.net_args[0].insert(
              {DNNL_ARG_ATTR_MULTIPLE_POST_OP(count) | DNNL_ARG_SRC_1,
               *context_.mul_mem});
          count++;
        } else if (post_op_param.name == "add") {
          context_.add_mem.reset(
              new dnnl::memory(*context_.add_md, cpu_engine_, DummyData));
          context_.net_args[0].insert(
              {DNNL_ARG_ATTR_MULTIPLE_POST_OP(count) | DNNL_ARG_SRC_1,
               *context_.add_mem});
          count++;
        }
      }
    }

    context_.matmul_primitives.push_back(*matmul_primitive);
    return;
  }

  struct MklMatMulContext context_;
#if defined(DNNL_AARCH64_USE_ACL) && defined(ENABLE_ONEDNN_OPENMP)
  mutex primitive_execution_mu_;
#endif
};

template <typename T, typename Tlhs, typename Trhs, typename Toutput,
          bool CSR = false>
class MklMatMulPrimitiveFactory : public MklPrimitiveFactory<T> {
 public:
  static MklMatMulPrimitive<Tlhs, Trhs, Toutput, CSR>* Get(
      const MklMatMulParams& params, bool do_not_cache) {
    MklMatMulPrimitive<Tlhs, Trhs, Toutput, CSR>* matmul_prim = nullptr;

    if (do_not_cache) {
      // Always create new primitive
      matmul_prim = new MklMatMulPrimitive<Tlhs, Trhs, Toutput, CSR>(params);
    } else {
      // Try to find a suitable one in pool
      matmul_prim = dynamic_cast<MklMatMulPrimitive<Tlhs, Trhs, Toutput, CSR>*>(
          MklMatMulPrimitiveFactory<T, Tlhs, Trhs, Toutput, CSR>::GetInstance()
              .GetMklMatMul(params));
      if (matmul_prim == nullptr) {
        matmul_prim = new MklMatMulPrimitive<Tlhs, Trhs, Toutput, CSR>(params);
        MklMatMulPrimitiveFactory<T, Tlhs, Trhs, Toutput, CSR>::GetInstance()
            .SetMklMatMul(params, matmul_prim);
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
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(params.prefix);
    key_creator.AddAsKey(params.a_dims);
    key_creator.AddAsKey(params.b_dims);
    key_creator.AddAsKey(params.c_dims);
    key_creator.AddAsKey(params.a_strides);
    key_creator.AddAsKey(params.b_strides);
    key_creator.AddAsKey(params.c_strides);
    key_creator.AddAsKey(typeid(T).name());
    key_creator.AddAsKey(typeid(Tlhs).name());
    key_creator.AddAsKey(typeid(Trhs).name());
    key_creator.AddAsKey(typeid(Toutput).name());

    // Generate keys for post-ops
    for (auto const& post_op_param : params.post_op_params) {
      if (post_op_param.name == "output_scale") {
        DCHECK_EQ(post_op_param.param.size(), 1);
        key_creator.AddAsKey(post_op_param.name);
        key_creator.AddAsKey(post_op_param.param[0]);
      } else if (post_op_param.name == "mul" || post_op_param.name == "add") {
        key_creator.AddAsKey(post_op_param.name);
        key_creator.AddAsKey(post_op_param.dims);
      } else {
        return string("not_a_key");
      }
    }
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
  using dims = dnnl::memory::dims;

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

  MklMatMulParams params("dnnl_gemm", a_dims, b_dims, c_dims, a_strides,
                         b_strides, c_strides);
  auto st = ExecuteSingleThreadedGemm(m, n, k, sizeof(T));
  // Create the oneDNN wrapper over Eigen threadpool and set max threads
  // in oneDNN.
  Eigen::ThreadPoolInterface* eigen_interface =
      EigenThreadPoolFromTfContext(ctx);
  tsl::OneDnnThreadPool eigen_tp(eigen_interface, ThreadPoolUseCallerThread(),
                                 st ? 1 : -1);
  MklMatMulPrimitive<T, T, T>* matmul_prim =
      MklMatMulPrimitiveFactory<T, T, T, T>::Get(params, 0);

  UserScratchPad<unsigned char> scratch_pad;
  scratch_pad.AllocateSPTensor(matmul_prim, ctx);
  // Execute matmul primitive.

  std::shared_ptr<stream> cpu_stream;

  cpu_stream.reset(CreateStream(&eigen_tp, matmul_prim->GetEngine()));
  matmul_prim->Execute(cpu_stream, a, b, c, scratch_pad.Get());
}

}  // anonymous namespace

#undef APPEND_ELTWISE
#undef APPEND_ELTWISE_RELU6
#undef OUTPUT_SCALE_DCHECK
#undef SET_MKL_LAYOUT
#undef TSCALED_BIAS

}  // namespace tensorflow

#endif  // INTEL_MKL
#endif  // TENSORFLOW_CORE_KERNELS_MKL_MKL_MATMUL_OPS_COMMON_H_
