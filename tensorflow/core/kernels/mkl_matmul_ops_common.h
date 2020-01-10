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
#include "tensorflow/core/util/mkl_util.h"

using mkldnn::inner_product_forward;
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
  memory::format weight_fmt;
  string dtypes = string("");
  struct PostOpParam {
    string name;
    std::vector<float> param;
  };
  std::vector<PostOpParam> post_op_params;

  MklDnnMatMulFwdParams(memory::dims src_dims, memory::dims weight_dims,
                        memory::dims bias_dims, memory::dims dst_dims,
                        memory::format weight_fmt = memory::format::any)
      : src_dims(src_dims),
        weight_dims(weight_dims),
        bias_dims(bias_dims),
        dst_dims(dst_dims),
        weight_fmt(weight_fmt) {}
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
      : cpu_engine_(engine::cpu, 0) {
    context_.fwd_stream.reset(new stream(stream::kind::eager));
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
    context_.fwd_stream->submit(context_.fwd_primitives);

    // After execution, set data handle back
    context_.src_mem->set_data_handle(DummyData);
    context_.weight_mem->set_data_handle(DummyData);
    context_.bias_mem->set_data_handle(DummyData);
    context_.dst_mem->set_data_handle(DummyData);
  }

  memory::format GetSrcMemoryFormat() const { return context_.src_fmt; }
  memory::format GetweightMemoryFormat() const { return context_.weight_fmt; }
  std::shared_ptr<mkldnn::inner_product_forward::primitive_desc>
  GetPrimitiveDesc() const {
    return context_.fwd_pd;
  }

 private:
  // Primitive reuse context for inner-product Fwd op
  struct MklDnnMatMulFwdContext {
    // Expected memory format for this primitive instance
    memory::format src_fmt;
    memory::format weight_fmt;

    // MKL-DNN memory
    std::shared_ptr<mkldnn::memory> src_mem;
    std::shared_ptr<mkldnn::memory> weight_mem;
    std::shared_ptr<mkldnn::memory> bias_mem;
    std::shared_ptr<mkldnn::memory> dst_mem;

    // Descriptor and primitive-descriptor for forward inner-product
    std::shared_ptr<mkldnn::inner_product_forward::desc> fwd_desc;
    std::shared_ptr<mkldnn::inner_product_forward::primitive_desc> fwd_pd;

    // Memory descriptors
    std::shared_ptr<mkldnn::memory::desc> src_md;
    std::shared_ptr<mkldnn::memory::desc> weight_md;
    std::shared_ptr<mkldnn::memory::desc> bias_md;
    std::shared_ptr<mkldnn::memory::desc> dst_md;

    // Inner-product primitive
    std::shared_ptr<mkldnn::primitive> matmul_fwd;
    std::shared_ptr<mkldnn::stream> fwd_stream;
    std::vector<mkldnn::primitive> fwd_primitives;

    MklDnnMatMulFwdContext()
        : src_fmt(memory::format::any),
          weight_fmt(memory::format::any),
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
          fwd_stream(nullptr) {}
  };

  void Setup(const MklDnnMatMulFwdParams& matmul_fwd_params) {
    // Create memory descriptors for inner-product data with no specified format
    context_.src_md.reset(new memory::desc({matmul_fwd_params.src_dims},
                                           MklDnnType<Tinput>(),
                                           memory::format::any));

    context_.weight_md.reset(new memory::desc({matmul_fwd_params.weight_dims},
                                              MklDnnType<Tweight>(),
                                              matmul_fwd_params.weight_fmt));

    context_.dst_md.reset(new memory::desc({matmul_fwd_params.dst_dims},
                                           MklDnnType<Toutput>(),
                                           memory::format::any));

    context_.bias_md.reset(new memory::desc({matmul_fwd_params.bias_dims},
                                            MklDnnType<Tbias>(),
                                            memory::format::any));
    // Create an inner-product
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
          post_ops.append_eltwise(op_scale, mkldnn::eltwise_relu, op_alpha,
                                  op_beta);
        } else if (post_op_param.name == "relu6") {
          DCHECK_EQ(post_op_param.param.size(), 3);
          float op_scale = post_op_param.param[0];
          float op_alpha = post_op_param.param[1];
          float op_beta = post_op_param.param[2];
          post_ops.append_eltwise(op_scale, mkldnn::eltwise_bounded_relu,
                                  op_alpha, op_beta);
        } else if (post_op_param.name == "elu") {
          DCHECK_EQ(post_op_param.param.size(), 3);
          float op_scale = post_op_param.param[0];
          float op_alpha = post_op_param.param[1];
          float op_beta = post_op_param.param[2];
          post_ops.append_eltwise(op_scale, mkldnn::eltwise_elu, op_alpha,
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

    // Store the expected memory format
    context_.src_fmt = static_cast<mkldnn::memory::format>(
        context_.fwd_pd.get()->src_primitive_desc().desc().data.format);

    context_.weight_fmt = static_cast<mkldnn::memory::format>(
        context_.fwd_pd.get()->weights_primitive_desc().desc().data.format);

    // Create memory primitive based on dummy data
    context_.src_mem.reset(
        new memory(context_.fwd_pd.get()->src_primitive_desc(), DummyData));
    context_.weight_mem.reset(
        new memory(context_.fwd_pd.get()->weights_primitive_desc(), DummyData));
    context_.dst_mem.reset(
        new memory(context_.fwd_pd.get()->dst_primitive_desc(), DummyData));
    context_.bias_mem.reset(new memory({{{matmul_fwd_params.bias_dims},
                                         MklDnnType<Tbias>(),
                                         memory::format::x},
                                        cpu_engine_},
                                       DummyData));

    // Create inner-product primitive
    context_.matmul_fwd.reset(new inner_product_forward(
        *context_.fwd_pd, *context_.src_mem, *context_.weight_mem,
        *context_.bias_mem, *context_.dst_mem));

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

template <class Toutput>
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
      memory::format output_tf_format, Tensor** output_tensor) {
    DCHECK(output_tensor);
    auto dst_pd = mkldnn_matmul_prim_desc.dst_primitive_desc();

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

  engine cpu_engine_ = engine(engine::cpu, 0);

 protected:
  const int kInputIndexSrc = 0;
  const int kInputIndexWeight = 1;
  const int kInputIndexBias = 2;
  const int kOutputIndexDst = 0;
};

}  // namespace tensorflow

#endif  // INTEL_MKL
#endif  // TENSORFLOW_CORE_KERNELS_MKL_MATMUL_OPS_COMMON_H_
