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

#include "tensorflow/core/kernels/mkl/mkl_conv_ops.h"

#include <algorithm>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "mkldnn.hpp"
#include "absl/strings/str_join.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/mkl/mkl_quantized_conv_ops.h"
#include "tensorflow/core/kernels/no_op.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

using mkldnn::convolution_forward;
using mkldnn::prop_kind;
using mkldnn::stream;
using ConvFwdPd = mkldnn::convolution_forward::primitive_desc;
using ReorderPd = mkldnn::reorder::primitive_desc;

namespace tensorflow {
// This structure aggregates multiple inputs to Conv2DFwd* methods.
struct MklConvFwdParams {
  memory::dims src_dims;
  memory::dims filter_dims;
  memory::dims bias_dims;
  memory::dims dst_dims;
  memory::dims strides;
  memory::dims dilations;
  memory::dims padding_left;
  memory::dims padding_right;
  memory::dims fuse_bn_dims;
  MklTensorFormat tf_fmt;
  bool native_format;
  string dtypes = string("");
#ifdef DNNL_AARCH64_USE_ACL
  void* filter_address = nullptr;
#endif
  struct PostOpParam {
    string name;
    mkldnn::algorithm alg;
    std::vector<float> param;
    std::string partial_key;
  };
  std::vector<PostOpParam> post_op_params;

  MklConvFwdParams(memory::dims src_dims, memory::dims filter_dims,
                   memory::dims bias_dims, memory::dims dst_dims,
                   memory::dims strides, memory::dims dilations,
                   memory::dims padding_left, memory::dims padding_right,
                   memory::dims fuse_bn_dims, MklTensorFormat tf_fmt,
                   bool native_format)
      : src_dims(src_dims),
        filter_dims(filter_dims),
        bias_dims(bias_dims),
        dst_dims(dst_dims),
        strides(strides),
        dilations(dilations),
        padding_left(padding_left),
        padding_right(padding_right),
        fuse_bn_dims(fuse_bn_dims),
        tf_fmt(tf_fmt),
        native_format(native_format) {}
};

// With quantization, input, filter, and output can have different types
// so we use different template parameter for each type
template <typename Tinput, typename Tfilter, typename Tbias, typename Toutput>
class MklConvFwdPrimitive : public MklPrimitive {
 public:
  explicit MklConvFwdPrimitive(const MklConvFwdParams& convFwdDims)
      : MklPrimitive(engine(engine::kind::cpu, 0)) {
    // Create convolution primitive
    if (context_.conv_fwd == nullptr) {
      Setup(convFwdDims);
    }
  }
  ~MklConvFwdPrimitive() {}

  // Convolution forward execute with bias
  //   src_data:    input data buffer of src
  //   filter_data: input data buffer of filter (weights)
  //   bias_data:   input data buffer of bias
  //   dst_data:    output data buffer of dst
  void Execute(const Tinput* src_data, const Tfilter* filter_data,
               const Tbias* bias_data, const Toutput* dst_data,
               std::shared_ptr<stream> fwd_stream) {
    Execute(src_data, filter_data, bias_data, dst_data, nullptr, nullptr,
            nullptr, nullptr, fwd_stream);
  }

  void Execute(const Tinput* src_data, const Tfilter* filter_data,
               const Tbias* bias_data, const Toutput* dst_data,
               const Tinput* bn_scale_data, const Tinput* bn_mean_data,
               const Tinput* bn_offset_data, const Tinput* bn_rsqrt_data,
               std::shared_ptr<stream> fwd_stream) {
#ifndef ENABLE_ONEDNN_OPENMP
    // TODO: Create a common function and avoid the duplicate code
    context_.src_mem->set_data_handle(
        static_cast<void*>(const_cast<Tinput*>(src_data)), *fwd_stream);
    context_.filter_mem->set_data_handle(
        static_cast<void*>(const_cast<Tfilter*>(filter_data)), *fwd_stream);
    if (bias_data != nullptr) {
      context_.bias_mem->set_data_handle(
          static_cast<void*>(const_cast<Tbias*>(bias_data)), *fwd_stream);
    }
    if (bn_scale_data != nullptr) {
      context_.bn_scale_mem->set_data_handle(
          static_cast<void*>(const_cast<Tinput*>(bn_scale_data)), *fwd_stream);
      context_.bn_mean_mem->set_data_handle(
          static_cast<void*>(const_cast<Tinput*>(bn_mean_data)), *fwd_stream);
      context_.bn_rsqrt_mem->set_data_handle(
          static_cast<void*>(const_cast<Tinput*>(bn_rsqrt_data)), *fwd_stream);
      context_.bn_offset_mem->set_data_handle(
          static_cast<void*>(const_cast<Tinput*>(bn_offset_data)), *fwd_stream);
    }
    context_.dst_mem->set_data_handle(
        static_cast<void*>(const_cast<Toutput*>(dst_data)), *fwd_stream);
#else
    context_.src_mem->set_data_handle(
        static_cast<void*>(const_cast<Tinput*>(src_data)));
    context_.filter_mem->set_data_handle(
        static_cast<void*>(const_cast<Tfilter*>(filter_data)));
    if (bias_data != nullptr) {
      context_.bias_mem->set_data_handle(
          static_cast<void*>(const_cast<Tbias*>(bias_data)));
    }
    if (bn_scale_data != nullptr) {
      context_.bn_scale_mem->set_data_handle(
          static_cast<void*>(const_cast<Tinput*>(bn_scale_data)));
      context_.bn_mean_mem->set_data_handle(
          static_cast<void*>(const_cast<Tinput*>(bn_mean_data)));
      context_.bn_rsqrt_mem->set_data_handle(
          static_cast<void*>(const_cast<Tinput*>(bn_rsqrt_data)));
      context_.bn_offset_mem->set_data_handle(
          static_cast<void*>(const_cast<Tinput*>(bn_offset_data)));
    }
    context_.dst_mem->set_data_handle(
        static_cast<void*>(const_cast<Toutput*>(dst_data)));
#endif  // !ENABLE_ONEDNN_OPENMP

    DCHECK_EQ(context_.fwd_primitives.size(),
              context_.fwd_primitives_args.size());
    for (size_t i = 0; i < context_.fwd_primitives.size(); ++i) {
      context_.fwd_primitives.at(i).execute(*fwd_stream,
                                            context_.fwd_primitives_args.at(i));
    }

    // After execution, set data handle back
    context_.src_mem->set_data_handle(DummyData);
    context_.filter_mem->set_data_handle(DummyData);
    if (bias_data != nullptr) {
      context_.bias_mem->set_data_handle(DummyData);
    }
    if (bn_scale_data != nullptr) {
      context_.bn_scale_mem->set_data_handle(DummyData);
      context_.bn_mean_mem->set_data_handle(DummyData);
      context_.bn_rsqrt_mem->set_data_handle(DummyData);
      context_.bn_offset_mem->set_data_handle(DummyData);
    }
    context_.dst_mem->set_data_handle(DummyData);
  }

  // Convolution forward execute without bias
  //   src_data:    input data buffer of src
  //   filter_data: input data buffer of filter (weights)
  //   dst_data:    output data buffer of dst
  void Execute(const Tinput* src_data, const Tfilter* filter_data,
               const Toutput* dst_data, std::shared_ptr<stream> fwd_stream) {
    Execute(src_data, filter_data, nullptr, dst_data, nullptr, nullptr, nullptr,
            nullptr, fwd_stream);
  }

  std::shared_ptr<ConvFwdPd> GetPrimitiveDesc() const {
    return context_.fwd_pd;
  }

 private:
  // Primitive reuse context for Conv2D Fwd op
  struct ConvFwdContext {
    // MKL-DNN memory
    std::shared_ptr<mkldnn::memory> src_mem;
    std::shared_ptr<mkldnn::memory> filter_mem;
    std::shared_ptr<mkldnn::memory> bias_mem;
    std::shared_ptr<mkldnn::memory> dst_mem;

    // FusedBatchNorm related memory
    std::shared_ptr<mkldnn::memory> bn_scale_mem;
    std::shared_ptr<mkldnn::memory> bn_mean_mem;
    std::shared_ptr<mkldnn::memory> bn_rsqrt_mem;
    std::shared_ptr<mkldnn::memory> bn_offset_mem;

    // Desc & primitive desc
    std::shared_ptr<mkldnn::convolution_forward::desc> fwd_desc;

    // Memory desc
    std::shared_ptr<mkldnn::memory::desc> src_md;
    std::shared_ptr<mkldnn::memory::desc> filter_md;
    std::shared_ptr<mkldnn::memory::desc> bias_md;
    std::shared_ptr<mkldnn::memory::desc> dst_md;

    // TODO(intel-tf, yimeisun123): Only need one? FusedBatchNorm related.
    std::shared_ptr<mkldnn::memory::desc> bn_scale_md;
    std::shared_ptr<mkldnn::memory::desc> bn_mean_md;
    std::shared_ptr<mkldnn::memory::desc> bn_rsqrt_md;
    std::shared_ptr<mkldnn::memory::desc> bn_offset_md;

    // Convolution primitive
    std::shared_ptr<ConvFwdPd> fwd_pd;
    std::shared_ptr<mkldnn::primitive> conv_fwd;

    std::vector<mkldnn::primitive> fwd_primitives;
    std::vector<std::unordered_map<int, memory>> fwd_primitives_args;

    ConvFwdContext()
        : src_mem(nullptr),
          filter_mem(nullptr),
          bias_mem(nullptr),
          dst_mem(nullptr),
          bn_scale_mem(nullptr),
          bn_mean_mem(nullptr),
          bn_rsqrt_mem(nullptr),
          bn_offset_mem(nullptr),
          fwd_desc(nullptr),
          src_md(nullptr),
          filter_md(nullptr),
          bias_md(nullptr),
          dst_md(nullptr),
          bn_scale_md(nullptr),
          bn_mean_md(nullptr),
          bn_rsqrt_md(nullptr),
          bn_offset_md(nullptr),
          fwd_pd(nullptr),
          conv_fwd(nullptr) {}
  };

  void Setup(const MklConvFwdParams& convFwdDims) {
    memory::format_tag user_data_fmt;
    if (convFwdDims.native_format) {
      user_data_fmt = MklTensorFormatToMklDnnDataFormat(convFwdDims.tf_fmt);
    } else {
      // Create memory descriptors for convolution data w/ no specified format
      user_data_fmt = memory::format_tag::any;
    }
    context_.src_md.reset(new memory::desc(
        {convFwdDims.src_dims}, MklDnnType<Tinput>(), user_data_fmt));

    context_.filter_md.reset(new memory::desc({convFwdDims.filter_dims},
                                              MklDnnType<Tfilter>(),
                                              memory::format_tag::any));

    context_.dst_md.reset(new memory::desc(
        {convFwdDims.dst_dims}, MklDnnType<Toutput>(), user_data_fmt));

    if (!convFwdDims.bias_dims.empty()) {
      context_.bias_md.reset(new memory::desc({convFwdDims.bias_dims},
                                              MklDnnType<Tbias>(),
                                              memory::format_tag::any));
      // Create a convolution descriptor
      context_.fwd_desc.reset(new convolution_forward::desc(
          prop_kind::forward, mkldnn::algorithm::convolution_direct,
          *context_.src_md, *context_.filter_md, *context_.bias_md,
          *context_.dst_md, convFwdDims.strides, convFwdDims.dilations,
          convFwdDims.padding_left, convFwdDims.padding_right));
    } else {
      context_.fwd_desc.reset(new convolution_forward::desc(
          prop_kind::forward, mkldnn::algorithm::convolution_direct,
          *context_.src_md, *context_.filter_md, *context_.dst_md,
          convFwdDims.strides, convFwdDims.dilations, convFwdDims.padding_left,
          convFwdDims.padding_right));
    }

    if (!convFwdDims.fuse_bn_dims.empty()) {
      const memory::format_tag fused_bn_arg_fmt =
          convFwdDims.native_format
              ? user_data_fmt
              : MklTensorFormatToMklDnnDataFormat(convFwdDims.tf_fmt);

      context_.bn_scale_md.reset(new memory::desc(
          {convFwdDims.fuse_bn_dims}, MklDnnType<Tinput>(), fused_bn_arg_fmt));
      context_.bn_mean_md.reset(new memory::desc(
          {convFwdDims.fuse_bn_dims}, MklDnnType<Tinput>(), fused_bn_arg_fmt));
      context_.bn_rsqrt_md.reset(new memory::desc(
          {convFwdDims.fuse_bn_dims}, MklDnnType<Tinput>(), fused_bn_arg_fmt));
      context_.bn_offset_md.reset(new memory::desc(
          {convFwdDims.fuse_bn_dims}, MklDnnType<Tinput>(), fused_bn_arg_fmt));
    }

    // Check if there is any fusions as post-ops
    auto const& post_op_params = convFwdDims.post_op_params;
    mkldnn::primitive_attr post_ops_attr;
    mkldnn::post_ops post_ops;
    if (!post_op_params.empty()) {
      for (auto const& post_op_param : post_op_params) {
        if (post_op_param.name == "activation") {
          DCHECK_EQ(post_op_param.param.size(), 3);
          float op_scale = post_op_param.param[0];
          float op_alpha = post_op_param.param[1];
          float op_beta = post_op_param.param[2];
          post_ops.append_eltwise(op_scale, post_op_param.alg, op_alpha,
                                  op_beta);
        } else if (post_op_param.name == "sum") {
          DCHECK_EQ(post_op_param.param.size(), 1);
          float op_scale = post_op_param.param[0];
          post_ops.append_sum(op_scale);
        } else if (post_op_param.name == "output_scale") {
          if (post_op_param.param.size() == 1) {
            post_ops_attr.set_output_scales(0, post_op_param.param);
          } else {
            post_ops_attr.set_output_scales(2, post_op_param.param);
          }
        } else if (post_op_param.name == "fuse_bn") {
          post_ops.append_binary(mkldnn::algorithm::binary_sub,
                                 *context_.bn_mean_md);
          post_ops.append_binary(mkldnn::algorithm::binary_mul,
                                 *context_.bn_rsqrt_md);
          post_ops.append_binary(mkldnn::algorithm::binary_mul,
                                 *context_.bn_scale_md);
          post_ops.append_binary(mkldnn::algorithm::binary_add,
                                 *context_.bn_offset_md);
        } else {
          DCHECK((post_op_param.name == "activation") ||
                 (post_op_param.name == "sum") ||
                 (post_op_param.name == "output_scale") ||
                 (post_op_param.name == "fuse_bn"));
        }
      }
      post_ops_attr.set_post_ops(post_ops);
      context_.fwd_pd.reset(
          new ConvFwdPd(*context_.fwd_desc, post_ops_attr, cpu_engine_));
    } else {
      context_.fwd_pd.reset(new ConvFwdPd(*context_.fwd_desc, cpu_engine_));
    }

    // Create memory primitive based on dummy data
    context_.src_mem.reset(
        new memory(context_.fwd_pd.get()->src_desc(), cpu_engine_, DummyData));
    context_.filter_mem.reset(new memory(context_.fwd_pd.get()->weights_desc(),
                                         cpu_engine_, DummyData));
    context_.dst_mem.reset(
        new memory(context_.fwd_pd.get()->dst_desc(), cpu_engine_, DummyData));

    context_.conv_fwd.reset(new convolution_forward(*context_.fwd_pd));

    // Create convolution primitive and add it to net
    if (!convFwdDims.bias_dims.empty()) {
      context_.bias_mem.reset(new memory(
          {{convFwdDims.bias_dims}, MklDnnType<Tbias>(), memory::format_tag::x},
          cpu_engine_, DummyData));
      context_.fwd_primitives_args.push_back(
          {{MKLDNN_ARG_SRC, *context_.src_mem},
           {MKLDNN_ARG_WEIGHTS, *context_.filter_mem},
           {MKLDNN_ARG_BIAS, *context_.bias_mem},
           {MKLDNN_ARG_DST, *context_.dst_mem}});
    } else if (!convFwdDims.fuse_bn_dims.empty()) {
      context_.bn_scale_mem.reset(
          new memory(*context_.bn_scale_md, cpu_engine_, DummyData));
      context_.bn_mean_mem.reset(
          new memory(*context_.bn_mean_md, cpu_engine_, DummyData));
      context_.bn_offset_mem.reset(
          new memory(*context_.bn_offset_md, cpu_engine_, DummyData));
      context_.bn_rsqrt_mem.reset(
          new memory(*context_.bn_rsqrt_md, cpu_engine_, DummyData));

      context_.fwd_primitives_args.push_back(
          {{MKLDNN_ARG_SRC, *context_.src_mem},
           {MKLDNN_ARG_WEIGHTS, *context_.filter_mem},
           {MKLDNN_ARG_DST, *context_.dst_mem},
           {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | MKLDNN_ARG_SRC_1,
            *context_.bn_mean_mem},
           {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | MKLDNN_ARG_SRC_1,
            *context_.bn_rsqrt_mem},
           {DNNL_ARG_ATTR_MULTIPLE_POST_OP(2) | MKLDNN_ARG_SRC_1,
            *context_.bn_scale_mem},
           {DNNL_ARG_ATTR_MULTIPLE_POST_OP(3) | MKLDNN_ARG_SRC_1,
            *context_.bn_offset_mem}});
    } else {
      context_.fwd_primitives_args.push_back(
          {{MKLDNN_ARG_SRC, *context_.src_mem},
           {MKLDNN_ARG_WEIGHTS, *context_.filter_mem},
           {MKLDNN_ARG_DST, *context_.dst_mem}});
    }
    context_.fwd_primitives.push_back(*context_.conv_fwd);
  }

  struct ConvFwdContext context_;
};

// TODO(nhasabni): We should not require passing a type to MklPrimitiveFactory.
// But removing the need for type in MklPrimitiveFactory is going to require
// change to every MKL op. So not doing it now. Instead passing float.
template <typename Tinput, typename Tfilter, typename Tbias, typename Toutput>
class MklConvFwdPrimitiveFactory : public MklPrimitiveFactory<float> {
 public:
  static MklConvFwdPrimitive<Tinput, Tfilter, Tbias, Toutput>* Get(
      const MklConvFwdParams& convFwdDims, bool do_not_cache) {
    MklConvFwdPrimitive<Tinput, Tfilter, Tbias, Toutput>* conv_fwd = nullptr;

    if (do_not_cache) {
      // Always create a new primitive
      conv_fwd =
          new MklConvFwdPrimitive<Tinput, Tfilter, Tbias, Toutput>(convFwdDims);
    } else {
      // Try to find a suitable one in pool
      conv_fwd =
          dynamic_cast<MklConvFwdPrimitive<Tinput, Tfilter, Tbias, Toutput>*>(
              MklConvFwdPrimitiveFactory<Tinput, Tfilter, Tbias,
                                         Toutput>::GetInstance()
                  .GetConvFwd(convFwdDims));
      if (conv_fwd == nullptr) {
        conv_fwd = new MklConvFwdPrimitive<Tinput, Tfilter, Tbias, Toutput>(
            convFwdDims);
        MklConvFwdPrimitiveFactory<Tinput, Tfilter, Tbias,
                                   Toutput>::GetInstance()
            .SetConvFwd(convFwdDims, conv_fwd);
      }
    }

    return conv_fwd;
  }

 private:
  MklConvFwdPrimitiveFactory() {}
  ~MklConvFwdPrimitiveFactory() {}

  static const int kDilationH = 0, kDilationW = 1;

  static MklConvFwdPrimitiveFactory& GetInstance() {
    static MklConvFwdPrimitiveFactory instance_;
    return instance_;
  }

  static string CreateKey(const MklConvFwdParams& convFwdDims) {
    string prefix = "conv_fwd_";
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(prefix);
    key_creator.AddAsKey(convFwdDims.src_dims);
    key_creator.AddAsKey(convFwdDims.filter_dims);
#ifdef DNNL_AARCH64_USE_ACL
    key_creator.AddAsKey(convFwdDims.filter_address);
#endif
    key_creator.AddAsKey(convFwdDims.bias_dims);
    key_creator.AddAsKey(convFwdDims.dst_dims);
    key_creator.AddAsKey(convFwdDims.strides);
    key_creator.AddAsKey(convFwdDims.dilations);
    key_creator.AddAsKey(convFwdDims.padding_left);
    key_creator.AddAsKey(convFwdDims.padding_right);
    key_creator.AddAsKey(convFwdDims.dtypes);
    if (convFwdDims.native_format) {
      key_creator.AddAsKey(convFwdDims.tf_fmt);
    }

    // Generate keys for post-ops
    for (auto const& post_op_param : convFwdDims.post_op_params) {
      key_creator.AddAsKey(post_op_param.name);
      if (post_op_param.name == "activation") {
        DCHECK_EQ(post_op_param.param.size(), 3);
        for (auto& param : post_op_param.param) {
          key_creator.AddAsKey(param);
        }
      } else if (post_op_param.name == "sum") {
        DCHECK_EQ(post_op_param.param.size(), 1);
        for (auto& param : post_op_param.param) {
          key_creator.AddAsKey(param);
        }
      } else if (post_op_param.name == "output_scale") {
        key_creator.AddAsKey(post_op_param.partial_key);
      } else if (post_op_param.name == "fuse_bn") {
        key_creator.AddAsKey(post_op_param.name);
        key_creator.AddAsKey(convFwdDims.fuse_bn_dims);
      } else {
        return string("not_a_key");
      }
    }

    return key_creator.GetKey();
  }

  MklPrimitive* GetConvFwd(const MklConvFwdParams& convFwdDims) {
    string key = CreateKey(convFwdDims);
    return this->GetOp(key);
  }

  void SetConvFwd(const MklConvFwdParams& convFwdDims, MklPrimitive* op) {
    string key = CreateKey(convFwdDims);
    this->SetOp(key, op);
  }
};

// Base class for convolution forward operations
template <typename Device, typename Tinput, typename Tfilter, typename Tbias,
          typename Toutput, typename Ttemp_output, typename Tpadding,
          bool bias_enabled, bool pad_enabled, bool is_depthwise,
          bool native_format>
class MklConvOp : public OpKernel {
 public:
  ~MklConvOp() {}

  explicit MklConvOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations_));

    // Conv and QuantizedConv ops have different padding attributes
    // (`padding_list` versus `explicit_paddings`). But one and only one
    // attribute is expected.
    OP_REQUIRES(
        context,
        !(context->HasAttr("padding_list") &&
          context->HasAttr("explicit_paddings")),
        errors::InvalidArgument("Can only have 1 `padding` list at most"));
    if (context->HasAttr("padding_list")) {
      OP_REQUIRES_OK(context, context->GetAttr("padding_list", &padding_list_));
    }
    if (context->HasAttr("explicit_paddings")) {
      OP_REQUIRES_OK(context,
                     context->GetAttr("explicit_paddings", &padding_list_));
    }

    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(context, (strides_.size() == 4 || strides_.size() == 5),
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 or 5 dimensions"));

    const int64 stride_n = GetTensorDim(strides_, data_format_, 'N');
    const int64 stride_c = GetTensorDim(strides_, data_format_, 'C');
    OP_REQUIRES(
        context, stride_n == 1 && stride_c == 1,
        errors::Unimplemented("Current implementation does not yet support "
                              "strides in the batch and depth dimensions."));

    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    is_filter_const_ = false;
    if (context->HasAttr("is_filter_const")) {
      OP_REQUIRES_OK(context,
                     context->GetAttr("is_filter_const", &is_filter_const_));
    }

    if (strides_.size() == 4) {
      OP_REQUIRES(context, dilations_.size() == 4,
                  errors::InvalidArgument("Sliding window dilations field must "
                                          "specify 4 dimensions"));
      const int64 dilation_n = GetTensorDim(dilations_, data_format_, 'N');
      const int64 dilation_c = GetTensorDim(dilations_, data_format_, 'C');
      const int64 dilation_h = GetTensorDim(dilations_, data_format_, 'H');
      const int64 dilation_w = GetTensorDim(dilations_, data_format_, 'W');
      OP_REQUIRES(context, dilation_n == 1 && dilation_c == 1,
                  errors::InvalidArgument(
                      "Current implementation does not yet support "
                      "dilations in the batch and depth dimensions."));
      OP_REQUIRES(
          context, dilation_h > 0 && dilation_w > 0,
          errors::InvalidArgument("Dilated rates should be larger than 0."));
    } else if (strides_.size() == 5) {
      OP_REQUIRES(context, dilations_.size() == 5,
                  errors::InvalidArgument("Dilation rates field must "
                                          "specify 5 dimensions"));
      OP_REQUIRES(context,
                  (GetTensorDim(dilations_, data_format_, 'N') == 1 &&
                   GetTensorDim(dilations_, data_format_, 'C') == 1),
                  errors::InvalidArgument(
                      "Current implementation does not yet support "
                      "dilations rates in the batch and depth dimensions."));
      OP_REQUIRES(
          context,
          (GetTensorDim(dilations_, data_format_, '0') > 0 &&
           GetTensorDim(dilations_, data_format_, '1') > 0 &&
           GetTensorDim(dilations_, data_format_, '2') > 0),
          errors::InvalidArgument("Dilated rates should be larger than 0."));
    }
  }

  void Compute(OpKernelContext* context) override {
    try {
      // Input tensors
      const Tensor& src_tensor = MklGetInput(context, kInputIndex_Src);
      const Tensor& filter_tensor = MklGetInput(context, kInputIndex_Filter);
      MklDnnShape src_mkl_shape, filter_mkl_shape;
      GetMklShape(context, kInputIndex_Src, &src_mkl_shape, native_format);
      GetMklShape(context, kInputIndex_Filter, &filter_mkl_shape,
                  native_format);

      OP_REQUIRES(context, !filter_mkl_shape.IsMklTensor(),
                  errors::InvalidArgument("Filter should not be in "
                                          "Mkl Layout"));

      MklDnnData<Tinput> src(&cpu_engine_);
      MklDnnData<Tfilter> filter(&cpu_engine_);

      memory::dims src_dims, filter_dims, padding_left, padding_right,
          dilations, strides;
      memory::dims dst_dims_tf_order, dst_dims_mkl_order;

      // For any Conv with `EXPLICIT` padding, get padding from `padding_list`
      // attribute. Otherwise, get it from one of the inputs.
      bool pad_attr_enabled = false;
      for (auto const& padding_val : padding_list_) {
        if (padding_val) {
          pad_attr_enabled = true;

          break;
        }
      }

      if (fuse_pad_ || pad_attr_enabled) {
        PadWithConvFusion(context, padding_left, padding_right,
                          pad_attr_enabled);
      }

      // Get shapes of input tensors in MKL-DNN order
      MklDnnConvUtil conv_utl(context, strides_, padding_, data_format_,
                              dilations_);
      auto src_tf_shape = GetTfShape(context, kInputIndex_Src, native_format);
      auto filter_tf_shape =
          GetTfShape(context, kInputIndex_Filter, native_format);
      conv_utl.GetConvFwdSizesInMklOrder(
          src_tf_shape, filter_tf_shape, &src_dims, &filter_dims, &strides,
          &dilations, &dst_dims_tf_order, &dst_dims_mkl_order, &padding_left,
          &padding_right, (fuse_pad_ || pad_attr_enabled), is_depthwise);

      if (!context->status().ok()) return;

      // Check for corner case - if there is nothing to compute, return.
      TensorShape dst_tf_shape = MklDnnDimsToTFShape(dst_dims_tf_order);

      // Corner cases: output with 0 elements and 0 batch size.
      Tensor* dst_tensor = nullptr;
      bool emit_filter_output = (typeid(Tinput) == typeid(Tfilter) &&
                                 typeid(Tinput) == typeid(Toutput) &&
                                 (typeid(Tinput) == typeid(float) ||
                                  typeid(Tinput) == typeid(bfloat16))) &&
                                !native_format;
      if (dst_tf_shape.num_elements() == 0 || dst_dims_tf_order[0] == 0) {
        MklDnnShape dst_mkl_shape;
        dst_mkl_shape.SetMklTensor(false);
        AllocateOutputSetMklShape(context, kOutputIndex_Dst, &dst_tensor,
                                  src_tf_shape, dst_mkl_shape, native_format);

        // MklConv2D/3D also outputs converted filter as 2nd output.
        filter_mkl_shape.SetMklTensor(false);
        Tensor* output_filter_tensor = nullptr;
        if (emit_filter_output) {
          filter_mkl_shape.SetMklTensor(false);
          AllocateOutputSetMklShape(context, kOutputIndex_Filter,
                                    &output_filter_tensor, filter_tf_shape,
                                    filter_mkl_shape);
        }
        return;
      }

      bool is_conv2d = (strides_.size() == 4);

      if (!is_conv2d) {
        OP_REQUIRES(
            context, !pad_enabled,
            errors::InvalidArgument("Pad + Conv fusion only works for 2D"));
        OP_REQUIRES(
            context, !fuse_pad_,
            errors::InvalidArgument("Pad+Conv fusion only works for 2D"));
      }

      // TODO(gzmkl) 3-D support for Depthwise is not there
      if (is_depthwise) {
        OP_REQUIRES(context, is_conv2d,
                    errors::InvalidArgument(
                        "Only 2D convolution is supported for depthwise."));
      }

      // Create memory for user data.
      // Describe how the inputs and outputs of Convolution look like. Also
      // specify buffers containing actual input and output data.
      auto tf_fmt = is_conv2d ? TFDataFormatToMklDnnDataFormat(data_format_)
                              : TFDataFormatToMklDnn3DDataFormat(data_format_);

      auto mkl_fmt_tag = MklTensorFormatToMklDnnDataFormat(tf_fmt);
      // NOTE: `mkl_fmt_tag` will be `format_tag::undef` for ReLU
      OP_REQUIRES(context, mkl_fmt_tag != memory::format_tag::undef,
                  errors::InvalidArgument("Invalid data format"));

      // If input is in MKL layout, then simply grab the layout; otherwise,
      // construct TF layout for input.
      // For constructing TF layout for input, although input shape (src_dims)
      // is required to be in MKL-DNN order, the input layout is actually in
      // TF layout depending on the data format:
      //     Conv2D: NHWC or NCHW
      //     Conv3D: NDHWC or NCDHW
      auto src_md =
          src_mkl_shape.IsMklTensor()
              ? src_mkl_shape.GetMklLayout()
              : memory::desc(src_dims, MklDnnType<Tinput>(), mkl_fmt_tag);
      src.SetUsrMem(src_md, &src_tensor);

      // Although filter shape (filter_dims) required is in MKL-DNN order,
      // the layout is Tensorflow's layout (HWIO) and (HWIGO) for
      // depthwise/group convolutions.
      auto filter_format = is_conv2d ? (is_depthwise ? memory::format_tag::hwigo
                                                     : memory::format_tag::hwio)
                                     : memory::format_tag::dhwio;

      DCHECK(!filter_mkl_shape.IsMklTensor());
      auto filter_md =
          filter_mkl_shape.IsMklTensor()
              ? filter_mkl_shape.GetMklLayout()
              : memory::desc(filter_dims, MklDnnType<Tfilter>(), filter_format);
      filter.SetUsrMem(filter_md, &filter_tensor);

      // MKL-DNN dilations start from 0.
      for (int i = 0; i < dilations.size(); ++i) --dilations[i];

      // In some cases, primitive descriptor could potentially contain
      // large buffers. As a result, we don't cache these primitives if the
      // environment variable `TF_MKL_OPTIMIZE_PRIMITIVE_MEMUSE` is set to True.
      // MKL-DNN allocates buffers in the following cases:
      //   1. Legacy CPU without AVX512/AVX2, or
      //   2. 1x1 convolution with strides != 1
      bool do_not_cache =
          MklPrimitiveFactory<Tinput>::IsPrimitiveMemOptEnabled() &&
          (src_dims[MklDnnDims::Dim_N] > kSmallBatchSize) &&
          (MklPrimitiveFactory<Tinput>::IsLegacyPlatform() ||
           IsConv1x1StrideNot1(filter_dims, strides));

      // Get a conv2d fwd from primitive pool
      MklConvFwdPrimitive<Tinput, Tfilter, Tbias, Ttemp_output>* conv_fwd =
          nullptr;
      memory::dims bias_dims = {};
      if (fuse_biasadd_) {
        conv_utl.GetBiasSizeInMklOrder(kInputIndex_Bias, &bias_dims);
      }
      memory::dims fuse_bn_dims = {};
      TensorShape fuse_bn_shape;
      if (fuse_bn_) {
        // Inputs to FusedBatchNorm have same 1D shape
        fuse_bn_shape = MklGetInput(context, kInputIndex_BN_Mean).shape();
        OP_REQUIRES(context, fuse_bn_shape.dims() == 1,
                    errors::InvalidArgument("FusedBatchNorm must be 1D, not: ",
                                            fuse_bn_shape.DebugString()));

        // Note - MKL-DNN expects {1, C, 1, 1} for binary post-op even for NHWC
        fuse_bn_dims = {1, fuse_bn_shape.dim_size(0), 1, 1};
      }

      MklConvFwdParams convFwdDims(
          src_dims, filter_dims, fuse_biasadd_ ? bias_dims : NONE_DIMS,
          dst_dims_mkl_order, strides, dilations, padding_left, padding_right,
          fuse_bn_dims, tf_fmt, native_format);

      // TODO(mdfaijul): Extend the basic parameters for data types and fusions
      this->ExtendConvFwdParams(context, convFwdDims);
#ifdef DNNL_AARCH64_USE_ACL
      // Specifics of ACL: a primitive per constant weights ptr
      convFwdDims.filter_address = const_cast<void*>(
          static_cast<const void*>(filter_tensor.flat<Tfilter>().data()));
#endif

      conv_fwd =
          MklConvFwdPrimitiveFactory<Tinput, Tfilter, Tbias, Ttemp_output>::Get(
              convFwdDims, do_not_cache);
      // Allocate output tensors `dst_tensor` and `filter_out_tensor`
      MklDnnShape output_mkl_shape;
      std::shared_ptr<ConvFwdPd> conv_fwd_pd = conv_fwd->GetPrimitiveDesc();
      AllocateOutputTensor(context, *conv_fwd_pd, dst_dims_mkl_order, tf_fmt,
                           &output_mkl_shape, &dst_tensor);

      Tensor* filter_out_tensor = nullptr;
      if (emit_filter_output) {
        AllocateFilterOutputTensor(context, *conv_fwd_pd,
                                   TFShapeToMklDnnDims(filter_tf_shape),
                                   &filter_out_tensor);
      }

      Ttemp_output* dst_data =
          reinterpret_cast<Ttemp_output*>(dst_tensor->flat<Toutput>().data());

      // Check whether src and filter need to be reordered.
      Tinput* src_data = nullptr;
      if (src_md != conv_fwd_pd->src_desc()) {
        src.SetUsrMem(src_md, &src_tensor);
        src.CheckReorderToOpMem(conv_fwd_pd->src_desc(), cpu_engine_, context);
        src_data = static_cast<Tinput*>(src.GetOpMem().get_data_handle());
      } else {
        src_data = static_cast<Tinput*>(
            const_cast<Tinput*>(src_tensor.flat<Tinput>().data()));
      }

      Tfilter* filter_data = nullptr;
      if (filter_md != conv_fwd_pd->weights_desc()) {
        bool is_filter_cached = false;
        // If filter is a constant, we can avoid the conversion of filter from
        // Tensorflow format to MKL format by caching the filter when it is
        // converted for the first time. This cached filter can then be reused
        // in subsequent iterations.
        if (is_filter_const_) {
          if (IsFilterCacheEmpty(context)) {
            // Cache filter if it is not already cached.
            CacheFilter(context, conv_fwd_pd, filter_data, filter_tensor,
                        filter, filter_md, filter_mkl_shape);
          }
          filter_data = GetCachedFilter(context, conv_fwd_pd->weights_desc());
          is_filter_cached = (filter_data != nullptr);
        }
        if (!is_filter_cached) {
          filter.SetUsrMem(filter_md, &filter_tensor);
          if (filter_out_tensor == nullptr) {
            filter.CheckReorderToOpMem(conv_fwd_pd->weights_desc(), cpu_engine_,
                                       context);
          } else {
            filter.CheckReorderToOpMem(
                conv_fwd_pd->weights_desc(),
                filter.GetTensorBuffer(filter_out_tensor), cpu_engine_,
                context);
          }
          filter_data =
              static_cast<Tfilter*>(filter.GetOpMem().get_data_handle());
        }
      } else {
        filter_data = static_cast<Tfilter*>(
            const_cast<Tfilter*>(filter_tensor.flat<Tfilter>().data()));
      }

      // Execute convolution
      std::shared_ptr<stream> fwd_cpu_stream;
      MklDnnThreadPool eigen_tp(context);
      fwd_cpu_stream.reset(CreateStream(&eigen_tp, conv_fwd->GetEngine()));
      if (fuse_biasadd_) {
        const Tensor& bias_tensor = MklGetInput(context, kInputIndex_Bias);
        Tbias* bias_data =
            this->GetBiasHandle(context, conv_fwd_pd, bias_tensor);
        conv_fwd->Execute(src_data, filter_data, bias_data, dst_data,
                          fwd_cpu_stream);
      } else if (fuse_bn_) {
        const Tensor& bn_scale_tensor =
            MklGetInput(context, kInputIndex_BN_Scale);
        Tinput* bn_scale_data = static_cast<Tinput*>(
            const_cast<Tinput*>(bn_scale_tensor.flat<Tinput>().data()));
        const Tensor& bn_mean_tensor =
            MklGetInput(context, kInputIndex_BN_Mean);
        Tinput* bn_mean_data = static_cast<Tinput*>(
            const_cast<Tinput*>(bn_mean_tensor.flat<Tinput>().data()));
        const Tensor& bn_offset_tensor =
            MklGetInput(context, kInputIndex_BN_Offset);
        Tinput* bn_offset_data = static_cast<Tinput*>(
            const_cast<Tinput*>(bn_offset_tensor.flat<Tinput>().data()));

        Tensor bn_rsqrt_tensor;
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<Tinput>::v(),
                                              fuse_bn_shape, &bn_rsqrt_tensor));
        Tinput* bn_rsqrt_data = static_cast<Tinput*>(
            const_cast<Tinput*>(bn_rsqrt_tensor.flat<Tinput>().data()));
        this->ComputeBNScale(context, epsilon_, kInputIndex_BN_Variance,
                             bn_rsqrt_data);
        conv_fwd->Execute(src_data, filter_data, nullptr, dst_data,
                          bn_scale_data, bn_mean_data, bn_offset_data,
                          bn_rsqrt_data, fwd_cpu_stream);
      } else {
        conv_fwd->Execute(src_data, filter_data, dst_data, fwd_cpu_stream);
      }

      // Delete primitive since it is not cached.
      if (do_not_cache) delete conv_fwd;

    } catch (mkldnn::error& e) {
      string error_msg = tensorflow::strings::StrCat(
          "Status: ", e.status, ", message: ", string(e.message), ", in file ",
          __FILE__, ":", __LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }

  void PadWithConvFusion(OpKernelContext* context, memory::dims& padding_left,
                         memory::dims& padding_right, bool pad_attr_enabled) {
    Tpadding* paddings = nullptr;
    if (pad_attr_enabled) {
      paddings = padding_list_.data();
    } else {
      const Tensor& paddings_tf = MklGetInput(context, input_index_pad_);
      OP_REQUIRES(context, paddings_tf.dims() == 2,
                  errors::InvalidArgument("paddings must be 2-dimensional: ",
                                          paddings_tf.shape().DebugString()));
      // Flatten tensor to get individual paddings.
      paddings = static_cast<Tpadding*>(
          const_cast<Tpadding*>(paddings_tf.flat<Tpadding>().data()));
    }
    // If the data format is NHWC, indices 0, 1, 6 and 7 of paddings(_tf)
    // will be zero.
    // Example:
    // paddings_tf = [ [0, 0] [1, 2] [3, 4] [0, 0] ],
    // flat method = row-major, then:
    // paddings = {0, 0, 1, 2, 3, 4, 0, 0}.
    // Hence, the values are: top = 1, bottom = 2, left = 3, right = 4.
    //
    // Similarly, if the data format is NCHW, indices 0, 1, 2 and 3 of
    // paddings(_tf) will be zero.
    // i.e. for the above example, paddings = {0, 0, 0, 0, 1, 2, 3, 4}.
    int64 pad_top = 0, pad_left = 0;
    int64 pad_bottom = 0, pad_right = 0;
    string data_format = ToString(data_format_);
    if (data_format == "NHWC") {
      pad_top = paddings[2];
      pad_bottom = paddings[3];
      pad_left = paddings[4];
      pad_right = paddings[5];
    } else if (data_format == "NCHW") {
      pad_top = paddings[4];
      pad_bottom = paddings[5];
      pad_left = paddings[6];
      pad_right = paddings[7];
    }
    // Create padding arrays for MKL-DNN convolutions.
    // MKL-DNN uses asymmetric padding.
    padding_left = {static_cast<int>(pad_top), static_cast<int>(pad_left)};
    padding_right = {static_cast<int>(pad_bottom), static_cast<int>(pad_right)};
  }

 protected:
  void set_fuse_biasadd(bool fuse_biasadd) { fuse_biasadd_ = fuse_biasadd; }
  void set_fuse_activation(bool fuse_activation,
                           mkldnn::algorithm activation_alg,
                           float alpha_or_upbound = 0.0) {
    fuse_activation_ = fuse_activation;
    activation_alg_ = activation_alg;
    // This variable is used for alpha in leakyrelu or upper bound in relu6
    // depending on the context
    alpha_or_upbound_ = alpha_or_upbound;
  }
  void set_fuse_pad(bool fuse_pad) {
    fuse_pad_ = fuse_pad;
    if (fuse_bn_) {
      // If FusedBatchNorm is fused in PadWithFusedConv2D, pad is the 7th input
      input_index_pad_ = 6;
    } else if (fuse_add_ && fuse_biasadd_) {
      // If Bias and Add are fused in PadWithFusedConv2D, pad is the 5th input
      input_index_pad_ = 4;
    } else {
      // Case of Bias is fused in PadwithFusedConv OP, pad is the fourth input
      input_index_pad_ = 3;
    }
  }
  void set_fuse_add(bool fuse_add) { fuse_add_ = fuse_add; }
  void set_fuse_bn(bool fuse_bn, float epsilon) {
    fuse_bn_ = fuse_bn;
    epsilon_ = epsilon;
  }

  virtual void ComputeBNScale(OpKernelContext* context, float epsilon,
                              int bn_variance_index, Tinput* scale_buf_ptr) {
    OP_REQUIRES(
        context, false,
        errors::Unimplemented("Compute BN scale not expected in base class"));
    return;
  }

  // This method is for the base class MklConvOp, which handles the
  // floating point implementation of Conv. The quantized conv implementations
  // will use overridden versions of this method.
  virtual void ExtendConvFwdParams(OpKernelContext* context,
                                   MklConvFwdParams& params) {
    // Create a string from data types of input, filter, bias, and output.
    params.dtypes.append(typeid(Tinput).name());
    params.dtypes.append(typeid(Tfilter).name());
    params.dtypes.append(typeid(Tbias).name());
    params.dtypes.append(typeid(Toutput).name());

    // Add fusions as post ops
    // NOTE: Fusion of BiasAdd is handled directly inside MklConvOp by
    // checking `fuse_biasadd_` flag.
    if (fuse_add_) {
      params.post_op_params.push_back(
          {"sum", mkldnn::algorithm::undef, {1.0}, ""});
    }
    // NOTE - fuse_bn post_op entry must be before fuse_activation
    if (fuse_bn_) {
      params.post_op_params.push_back(
          {"fuse_bn", mkldnn::algorithm::undef, {1.0}, ""});
    }
    if (fuse_activation_) {
      params.post_op_params.push_back(
          {"activation", activation_alg_, {1.0, alpha_or_upbound_, 0.0}, ""});
    }
  }

  virtual Tbias* GetBiasHandle(OpKernelContext* context,
                               std::shared_ptr<ConvFwdPd>& conv2d_fwd_pd,
                               const Tensor& bias_tensor) {
    if (fuse_biasadd_) {
      return static_cast<Tbias*>(
          const_cast<Tbias*>(bias_tensor.flat<Tbias>().data()));
    }
    return nullptr;
  }

  virtual void AllocateOutputTensor(OpKernelContext* context,
                                    const ConvFwdPd& conv_prim_desc,
                                    const memory::dims& output_dims_mkl_order,
                                    MklTensorFormat output_tf_format,
                                    MklDnnShape* output_mkl_shape,
                                    Tensor** output_tensor) {
    DCHECK(output_tensor);
    auto dst_md = conv_prim_desc.dst_desc();

    if (!std::is_same<Ttemp_output, Toutput>::value) {
      dst_md.data.data_type =
          static_cast<mkldnn_data_type_t>(MklDnnType<Toutput>());
    }

    // Allocate shape of MKL tensor
    output_mkl_shape->SetMklTensor(true);
    output_mkl_shape->SetMklLayout(&dst_md);
    output_mkl_shape->SetElemType(MklDnnType<Toutput>());
    output_mkl_shape->SetTfLayout(output_dims_mkl_order.size(),
                                  output_dims_mkl_order, output_tf_format);

    // Allocate shape of TF tensor
    TensorShape output_tf_shape;
    output_tf_shape.AddDim((dst_md.get_size() / sizeof(Toutput)));
    if (native_format) {
      output_tf_shape = output_mkl_shape->GetTfShape();
    }

    if (fuse_add_) {
      const Tensor& add_tensor = MklGetInput(context, kInputIndex_Add);
      MklDnnShape add_mkl_shape;
      GetMklShape(context, kInputIndex_Add, &add_mkl_shape, native_format);
      // Forward the summand tensor to the output only if it has no other
      // references, otherwise make a copy of it.
      if (native_format && context->forward_input_to_output_with_shape(
                               kInputIndex_Add, kOutputIndex_Dst,
                               output_tf_shape, output_tensor)) {
        return;
      }
      // Check if reorder is needed
      if (!native_format && add_mkl_shape == *output_mkl_shape &&
          ForwardMklTensorInToOutWithMklShape(context, kInputIndex_Add,
                                              kOutputIndex_Dst, output_tensor,
                                              add_mkl_shape, false)) {
        return;
      } else {
        AllocateOutputSetMklShape(context, kOutputIndex_Dst, output_tensor,
                                  output_tf_shape, *output_mkl_shape,
                                  native_format);
        auto output_format_tag = MklTensorFormatToMklDnnDataFormat(
            output_mkl_shape->GetTfDataFormat());
        OP_REQUIRES(context, output_format_tag != memory::format_tag::undef,
                    errors::InvalidArgument(
                        "MklConvOp: AddN fusion: Invalid data format"));
        auto add_md =
            add_mkl_shape.IsMklTensor()
                ? add_mkl_shape.GetMklLayout()
                : memory::desc(output_dims_mkl_order, MklDnnType<Toutput>(),
                               output_format_tag);
        void* add_buf = static_cast<void*>(
            const_cast<Toutput*>(add_tensor.flat<Toutput>().data()));
        void* dst_buf =
            static_cast<void*>((*output_tensor)->flat<Ttemp_output>().data());
        if (native_format) {
          // We are simply deep copying the add_tensor to output_tensor without
          // changing memory layout, hence using same memory descriptor.
          add_md = dst_md =
              memory::desc({add_tensor.NumElements()}, MklDnnType<Toutput>(),
                           mkldnn::memory::format_tag::x);
        }
        fuse_add_src_.reset(new memory(add_md, this->cpu_engine_, add_buf));
        fuse_add_dst_.reset(new memory(dst_md, this->cpu_engine_, dst_buf));
        auto reorder_desc =
            ReorderPd(this->cpu_engine_, add_md, this->cpu_engine_, dst_md);

        CreateAndExecuteReorder(reorder_desc, *fuse_add_src_, *fuse_add_dst_,
                                this->cpu_engine_, context);
      }
    } else {
      AllocateOutputSetMklShape(context, kOutputIndex_Dst, output_tensor,
                                output_tf_shape, *output_mkl_shape,
                                native_format);
    }
  }

  engine cpu_engine_ = engine(engine::kind::cpu, 0);

 private:
  std::shared_ptr<mkldnn::memory> fuse_add_src_;
  std::shared_ptr<mkldnn::memory> fuse_add_dst_;
  std::vector<int32> strides_;
  std::vector<int32> dilations_;
  std::vector<Tpadding> padding_list_;
  bool is_filter_const_;
  mutex mu_;
  Padding padding_;
  TensorFormat data_format_;
  Tensor cached_filter_data_ TF_GUARDED_BY(mu_);
  Tensor cached_filter_md_ TF_GUARDED_BY(mu_);

  // Initialize to values the template is instantiated with
  bool fuse_biasadd_ = bias_enabled;
  bool fuse_activation_ = false;
  bool fuse_pad_ = pad_enabled;
  bool fuse_add_ = false;
  bool fuse_bn_ = false;
  float epsilon_ = 0.0001;

  // This variable is used for alpha in leakyrelu or upper bound in relu6
  // depending on the context
  float alpha_or_upbound_ = 0.0;
  mkldnn::algorithm activation_alg_ = mkldnn::algorithm::undef;

  int input_index_pad_ = 2;

  const int kInputIndex_Src = 0, kInputIndex_Filter = 1, kInputIndex_Bias = 2;
  const int kInputIndex_Add = 3;
  const int kOutputIndex_Dst = 0, kOutputIndex_Filter = 1;
  const int kDilationH = 0, kDilationW = 1;

  // Input indices for FusedBatchNorm
  const int kInputIndex_BN_Scale = 2, kInputIndex_BN_Offset = 3;
  const int kInputIndex_BN_Mean = 4, kInputIndex_BN_Variance = 5;

  MklTensorFormat GetFilterTfDataFormat(const MklDnnShape* filter_mkl_shape,
                                        const ConvFwdPd& conv_prim_desc) const {
    DCHECK(filter_mkl_shape);
    return filter_mkl_shape->GetTfDataFormat();
  }

  // Allocate tensors for cached filter data and cached filter memory
  // descriptor (data format)
  void AllocateTensor(OpKernelContext* context, const ConvFwdPd& conv_prim_desc,
                      Tensor** filter_tensor,
                      const MklDnnShape* filter_mkl_shape) {
    DCHECK(filter_tensor);
    TensorShape filter_tf_shape;
    filter_tf_shape.AddDim(
        (conv_prim_desc.weights_desc().get_size() / sizeof(Tfilter)));
    OP_REQUIRES_OK(
        context, context->allocate_temp(DataTypeToEnum<Tfilter>::value,
                                        filter_tf_shape, &cached_filter_data_));

    *filter_tensor = &cached_filter_data_;

    // There is no tensor format in DNNL 1.x. So we cache the complete filter
    // descriptor as flat byte array.
    TensorShape cached_filter_md_shape;
    memory::desc weights_desc = conv_prim_desc.weights_desc();
    // We don't use .get_size() method of memory::desc since it returns size
    // required to store primitive's input memory. It is much more than size of
    // memory::desc itself.
    cached_filter_md_shape.AddDim(sizeof(weights_desc) / sizeof(uint8));
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DT_UINT8, cached_filter_md_shape,
                                          &cached_filter_md_));
    *reinterpret_cast<memory::desc*>(cached_filter_md_.flat<uint8>().data()) =
        weights_desc;
  }

  void AllocateTensor(OpKernelContext* context, const ConvFwdPd& conv_prim_desc,
                      Tensor** filter_tensor) {
    AllocateTensor(context, conv_prim_desc, filter_tensor, nullptr);
  }

  void AllocateFilterOutputTensor(OpKernelContext* context,
                                  const ConvFwdPd& conv_prim_desc,
                                  const memory::dims& filter_dims_tf_order,
                                  Tensor** filter_tensor) {
    DCHECK(filter_tensor);
    auto filter_md = conv_prim_desc.weights_desc();

    // Allocate shape of MKL tensor
    MklDnnShape filter_mkl_shape;
    filter_mkl_shape.SetMklTensor(true);
    filter_mkl_shape.SetMklLayout(&filter_md);
    filter_mkl_shape.SetElemType(MklDnnType<Tfilter>());

    // The format of the filter is actually OIhw8i8o, but TF doesn't support
    // this format. Just use format::blocked for now because the layout
    // is stored in the MKL data.
    filter_mkl_shape.SetTfLayout(filter_dims_tf_order.size(),
                                 filter_dims_tf_order,
                                 MklTensorFormat::FORMAT_BLOCKED);

    // Allocate the data space for the filter to propagate as TF tensor.
    TensorShape filter_tf_shape;
    filter_tf_shape.AddDim((filter_md.get_size() / sizeof(Tfilter)));

    AllocateOutputSetMklShape(context, kOutputIndex_Filter, filter_tensor,
                              filter_tf_shape, filter_mkl_shape);
  }

  // TODO(intel-mkl): This function does not seem to be called. Remove it.
  // Prepare and execute net - checks for input and output reorders.
  void PrepareAndExecuteNet(const ConvFwdPd& conv_prim_desc,
                            MklDnnData<Tinput>* src,
                            MklDnnData<Tfilter>* filter,
                            MklDnnData<Tbias>* bias,
                            MklDnnData<Toutput>* output,
                            Tensor* filter_out_tensor) {
    DCHECK(filter_out_tensor);

    // Create reorders between user layout and MKL layout if it is needed and
    // add it to the net before convolution. No need to check for output
    // reorder as we propagate output layout to the next layer.
    src->CheckReorderToOpMem(conv_prim_desc.src_desc(), cpu_engine_);

    // Rather than re-ordering to a temp buffer, reorder directly to the
    // filter output tensor
    filter->CheckReorderToOpMem(conv_prim_desc.weights_desc(),
                                filter->GetTensorBuffer(filter_out_tensor));

    // Create convolution primitive and add it to net.
    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;
    if (bias) {
      DCHECK(fuse_biasadd_);
      net.push_back(convolution_forward(conv_prim_desc));
      net_args.push_back({{MKLDNN_ARG_SRC, src->GetOpMem()},
                          {MKLDNN_ARG_WEIGHTS, filter->GetOpMem()},
                          {MKLDNN_ARG_BIAS, bias->GetOpMem()},
                          {MKLDNN_ARG_DST, output->GetOpMem()}});
    } else {
      DCHECK(!fuse_biasadd_);
      net.push_back(convolution_forward(conv_prim_desc));
      net_args.push_back({{MKLDNN_ARG_SRC, src->GetOpMem()},
                          {MKLDNN_ARG_WEIGHTS, filter->GetOpMem()},
                          {MKLDNN_ARG_DST, output->GetOpMem()}});
    }
    ExecutePrimitive(net, &net_args, cpu_engine_);
  }

  // TF_LOCKS_EXCLUDED annotation ensures that the lock (mu_) cannot
  // be acquired before entering the function, since it is acquired
  // inside the function.
  inline bool IsFilterCacheEmpty(OpKernelContext* context)
      TF_LOCKS_EXCLUDED(mu_) {
    tf_shared_lock lock(mu_);
    const Tensor& cached_filter_data_tensor = cached_filter_data_;
    return (cached_filter_data_tensor.NumElements() == 0);
  }

  // Cache the converted filter in a tensor.
  // Only one thread can execute this method at any given time.
  void CacheFilter(OpKernelContext* context,
                   const std::shared_ptr<ConvFwdPd>& conv_fwd_pd,
                   Tfilter* filter_data, const Tensor& filter_tensor,
                   MklDnnData<Tfilter>& filter, const memory::desc& filter_md,
                   const MklDnnShape& filter_mkl_shape) TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock lock(mu_);
    const Tensor& cached_filter_data_tensor = cached_filter_data_;

    // If filter is already cached, there's nothing to do.
    if (cached_filter_data_tensor.NumElements() > 0) {
      return;
    }

    // Otherwise, cache filter
    filter.SetUsrMem(filter_md, &filter_tensor);
    filter.CheckReorderToOpMem(conv_fwd_pd.get()->weights_desc(),
                               this->cpu_engine_, context);
    filter_data = static_cast<Tfilter*>(filter.GetOpMem().get_data_handle());

    Tensor* filter_tensor_ptr = nullptr;
    AllocateTensor(context, *conv_fwd_pd, &filter_tensor_ptr,
                   &filter_mkl_shape);
    void* cached_filter_data = filter.GetTensorBuffer(filter_tensor_ptr);
    size_t cached_filter_data_size = filter.GetOpMem().get_desc().get_size();
    memcpy(cached_filter_data, filter_data, cached_filter_data_size);
  }

  bool AreMemoryDescriptorsEqual(const memory::desc& filter_md,
                                 const Tensor& cached_filter_md) {
    auto filter_md_data = filter_md.data;
    const char* filter_data = reinterpret_cast<const char*>(&filter_md_data);

    auto cached_filter_md_data = cached_filter_md.scalar<int64>()();
    const char* cached_filter_data =
        reinterpret_cast<const char*>(&cached_filter_md_data);

    for (size_t i = 0; i < sizeof(filter_md_data); ++i) {
      if (*filter_data++ != *cached_filter_data++) {
        return false;
      }
    }
    return true;
  }

  Tfilter* GetCachedFilter(OpKernelContext* context,
                           const memory::desc& filter_md)
      TF_LOCKS_EXCLUDED(mu_) {
    tf_shared_lock lock(mu_);
    const Tensor& cached_filter_data = cached_filter_data_;
    const Tensor& cached_filter_md = cached_filter_md_;

    // Check if the memory descriptor of the cached weights is the same as
    // filter_md. If so, we can use the cached weights; otherwise
    // return nullptr.
    if (filter_md == *static_cast<memory::desc*>(cached_filter_md.data())) {
      return static_cast<Tfilter*>(
          const_cast<Tfilter*>(cached_filter_data.flat<Tfilter>().data()));
    }
    return nullptr;
  }
};

// Base class for fused convolution forward operations
template <typename Device, typename Tinput, typename Tfilter, typename Tbias,
          typename Toutput, typename Ttemp_output, typename Tpadding,
          bool pad_enabled, bool native_format>
class MklFusedConvOp
    : public MklConvOp<Device, Tinput, Tfilter, Tbias, Toutput, Ttemp_output,
                       Tpadding, false, false, false, native_format> {
 public:
  explicit MklFusedConvOp(OpKernelConstruction* context)
      : MklConvOp<Device, Tinput, Tfilter, Tbias, Toutput, Ttemp_output,
                  Tpadding, false, false, false, native_format>(context) {
    // Since we came here through the registration of _MklFusedConv2D, get
    // all information from 'fused_ops' and 'num_args'
    std::vector<string> fused_ops;
    OP_REQUIRES_OK(context, context->GetAttr("fused_ops", &fused_ops));

    int num_args;
    OP_REQUIRES_OK(context, context->GetAttr("num_args", &num_args));
    OP_REQUIRES(context, !fused_ops.empty(),
                errors::InvalidArgument(
                    "Fused Conv2D must have at least one fused op."));

    // TODO: Compact the code for activation checking
    if (fused_ops == std::vector<string>{"BiasAdd"}) {
      this->set_fuse_biasadd(true);
      OP_REQUIRES(context, num_args == 1,
                  errors::InvalidArgument(
                      "Fused Conv2D must have one extra argument: bias."));
    } else if (fused_ops == std::vector<string>{"Relu"}) {
      this->set_fuse_activation(true, mkldnn::algorithm::eltwise_relu);
    } else if (fused_ops == std::vector<string>{"Relu6"}) {
      this->set_fuse_activation(true, mkldnn::algorithm::eltwise_bounded_relu,
                                6.0);
    } else if (fused_ops == std::vector<string>{"Elu"}) {
      this->set_fuse_activation(true, mkldnn::algorithm::eltwise_elu, 1.0);
    } else if (fused_ops == std::vector<string>{"LeakyRelu"}) {
      float leakyrelu_alpha;
      OP_REQUIRES_OK(context,
                     context->GetAttr("leakyrelu_alpha", &leakyrelu_alpha));
      this->set_fuse_activation(true, mkldnn::algorithm::eltwise_relu,
                                leakyrelu_alpha);
    } else if (fused_ops == std::vector<string>{"FusedBatchNorm"}) {
      float epsilon;
      OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon));
      OP_REQUIRES(
          context, num_args == 4,
          errors::InvalidArgument(
              "Fused Conv2D with batchnorm must have 4 extra argument"));
      this->set_fuse_bn(true, epsilon);
    } else if (fused_ops == std::vector<string>{"BiasAdd", "Relu"}) {
      this->set_fuse_biasadd(true);
      this->set_fuse_activation(true, mkldnn::algorithm::eltwise_relu);
      OP_REQUIRES(context, num_args == 1,
                  errors::InvalidArgument(
                      "Fused Conv2D must have one extra argument: bias."));
    } else if (fused_ops == std::vector<string>{"BiasAdd", "Relu6"}) {
      this->set_fuse_biasadd(true);
      this->set_fuse_activation(true, mkldnn::algorithm::eltwise_bounded_relu,
                                6.0);
      OP_REQUIRES(context, num_args == 1,
                  errors::InvalidArgument(
                      "Fused Conv2D must have one extra argument: bias."));
    } else if (fused_ops == std::vector<string>{"BiasAdd", "Elu"}) {
      this->set_fuse_biasadd(true);
      this->set_fuse_activation(true, mkldnn::algorithm::eltwise_elu, 1.0);
      OP_REQUIRES(context, num_args == 1,
                  errors::InvalidArgument(
                      "Fused Conv2D must have one extra argument: bias."));
    } else if (fused_ops == std::vector<string>{"BiasAdd", "LeakyRelu"}) {
      this->set_fuse_biasadd(true);
      float leakyrelu_alpha;
      OP_REQUIRES_OK(context,
                     context->GetAttr("leakyrelu_alpha", &leakyrelu_alpha));
      this->set_fuse_activation(true, mkldnn::algorithm::eltwise_relu,
                                leakyrelu_alpha);
      OP_REQUIRES(context, num_args == 1,
                  errors::InvalidArgument(
                      "Fused Conv2D must have one extra argument: bias."));
    } else if (fused_ops == std::vector<string>{"BiasAdd", "Add"}) {
      this->set_fuse_biasadd(true);
      this->set_fuse_add(true);
      OP_REQUIRES(
          context, num_args == 2,
          errors::InvalidArgument(
              "Fused Conv2D must have two extra arguments: bias and add."));
    } else if (fused_ops == std::vector<string>{"FusedBatchNorm", "Relu"}) {
      float epsilon;
      OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon));
      OP_REQUIRES(
          context, num_args == 4,
          errors::InvalidArgument(
              "Fused Conv2D with batchnorm must have 4 extra argument"));
      this->set_fuse_bn(true, epsilon);
      this->set_fuse_activation(true, mkldnn::algorithm::eltwise_relu);
    } else if (fused_ops == std::vector<string>{"FusedBatchNorm", "Relu6"}) {
      float epsilon;
      OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon));
      OP_REQUIRES(
          context, num_args == 4,
          errors::InvalidArgument(
              "Fused Conv2D with batchnorm must have 4 extra argument"));
      this->set_fuse_bn(true, epsilon);
      this->set_fuse_activation(true, mkldnn::algorithm::eltwise_bounded_relu,
                                6.0);
    } else if (fused_ops == std::vector<string>{"FusedBatchNorm", "Elu"}) {
      float epsilon;
      OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon));
      OP_REQUIRES(
          context, num_args == 4,
          errors::InvalidArgument(
              "Fused Conv2D with batchnorm must have 4 extra argument"));
      this->set_fuse_bn(true, epsilon);
      this->set_fuse_activation(true, mkldnn::algorithm::eltwise_elu, 1.0);
    } else if (fused_ops ==
               std::vector<string>{"FusedBatchNorm", "LeakyRelu"}) {
      float epsilon, leakyrelu_alpha;
      OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon));
      OP_REQUIRES_OK(context,
                     context->GetAttr("leakyrelu_alpha", &leakyrelu_alpha));
      OP_REQUIRES(
          context, num_args == 4,
          errors::InvalidArgument(
              "Fused Conv2D with batchnorm must have 4 extra argument"));
      this->set_fuse_bn(true, epsilon);
      this->set_fuse_activation(true, mkldnn::algorithm::eltwise_relu,
                                leakyrelu_alpha);
    } else if (fused_ops == std::vector<string>{"BiasAdd", "Add", "Relu"}) {
      this->set_fuse_biasadd(true);
      this->set_fuse_add(true);
      this->set_fuse_activation(true, mkldnn::algorithm::eltwise_relu);
      OP_REQUIRES(
          context, num_args == 2,
          errors::InvalidArgument(
              "Fused Conv2D must have two extra arguments: bias and add."));
    } else if (fused_ops == std::vector<string>{"BiasAdd", "Add", "Relu6"}) {
      this->set_fuse_biasadd(true);
      this->set_fuse_add(true);
      this->set_fuse_activation(true, mkldnn::algorithm::eltwise_bounded_relu,
                                6.0);
      OP_REQUIRES(
          context, num_args == 2,
          errors::InvalidArgument(
              "Fused Conv2D must have two extra arguments: bias and add."));
    } else if (fused_ops == std::vector<string>{"BiasAdd", "Add", "Elu"}) {
      this->set_fuse_biasadd(true);
      this->set_fuse_add(true);
      this->set_fuse_activation(true, mkldnn::algorithm::eltwise_elu, 1.0);
      OP_REQUIRES(
          context, num_args == 2,
          errors::InvalidArgument(
              "Fused Conv2D must have two extra arguments: bias and add."));
    } else if (fused_ops ==
               std::vector<string>{"BiasAdd", "Add", "LeakyRelu"}) {
      this->set_fuse_biasadd(true);
      this->set_fuse_add(true);
      float leakyrelu_alpha;
      OP_REQUIRES_OK(context,
                     context->GetAttr("leakyrelu_alpha", &leakyrelu_alpha));
      this->set_fuse_activation(true, mkldnn::algorithm::eltwise_relu,
                                leakyrelu_alpha);
      OP_REQUIRES(
          context, num_args == 2,
          errors::InvalidArgument(
              "Fused Conv2D must have two extra arguments: bias and add."));
    } else {
      OP_REQUIRES(context, false,
                  errors::Unimplemented("Fusion is not implemented: [",
                                        absl::StrJoin(fused_ops, ","), "]"));
    }

    if (pad_enabled) {
      this->set_fuse_pad(true);
    }
  }

  void ComputeBNScale(OpKernelContext* context, float epsilon,
                      int bn_variance_index, Tinput* scale_buf_ptr) override {
    const Tensor& bn_var_tensor = MklGetInput(context, bn_variance_index);

    Eigen::Tensor<Tinput, 1, Eigen::RowMajor> bn_rsqrt =
        (bn_var_tensor.flat<Tinput>() + static_cast<Tinput>(epsilon)).rsqrt();
    Tinput* bn_rsqrt_data = bn_rsqrt.data();
    size_t num_elem = bn_var_tensor.shape().dim_size(0);
    for (size_t i = 0; i < num_elem; i++) {
      scale_buf_ptr[i] = bn_rsqrt_data[i];
    }
    return;
  }

  virtual ~MklFusedConvOp() {}
};

template <typename Device, typename Tinput, typename Tfilter, typename Tbias,
          typename Toutput, typename Ttemp_output, typename Tpadding,
          bool pad_enabled, bool bias_enabled, bool is_depthwise,
          bool native_format>
class MklFusedDepthwiseConvOp
    : public MklConvOp<Device, Tinput, Tfilter, Tbias, Toutput, Ttemp_output,
                       Tpadding, bias_enabled, false, is_depthwise,
                       native_format> {
 public:
  explicit MklFusedDepthwiseConvOp(OpKernelConstruction* context)
      : MklConvOp<Device, Tinput, Tfilter, Tbias, Toutput, Ttemp_output,
                  Tpadding, bias_enabled, false, is_depthwise, native_format>(
            context) {
    // Since we came here through the registration of
    // _MklFusedDepthwiseConv2dNative, get all
    // information from 'fused_ops' and 'num_args'
    std::vector<string> fused_ops;
    OP_REQUIRES_OK(context, context->GetAttr("fused_ops", &fused_ops));

    int num_args;
    OP_REQUIRES_OK(context, context->GetAttr("num_args", &num_args));
    OP_REQUIRES(context, !fused_ops.empty(),
                errors::InvalidArgument(
                    "Fused DepthwiseConv2D must have at least one fused op."));

    if (fused_ops == std::vector<string>{"BiasAdd"}) {
      this->set_fuse_biasadd(true);
    } else if (fused_ops == std::vector<string>{"BiasAdd", "Relu"}) {
      this->set_fuse_biasadd(true);
      this->set_fuse_activation(true, mkldnn::algorithm::eltwise_relu);
    } else if (fused_ops == std::vector<string>{"BiasAdd", "Relu6"}) {
      this->set_fuse_biasadd(true);
      this->set_fuse_activation(true, mkldnn::algorithm::eltwise_bounded_relu,
                                6.0);
    } else if (fused_ops == std::vector<string>{"BiasAdd", "Elu"}) {
      this->set_fuse_biasadd(true);
      this->set_fuse_activation(true, mkldnn::algorithm::eltwise_elu, 1.0);
    } else {
      OP_REQUIRES(context, false,
                  errors::Unimplemented("Fusion is not implemented: [",
                                        absl::StrJoin(fused_ops, ","), "]"));
    }

    OP_REQUIRES(
        context, num_args == 1,
        errors::InvalidArgument(
            "Fused DepthwiseConv2D must have one extra argument: bias."));

    if (pad_enabled) {
      this->set_fuse_pad(true);
    }
  }

  virtual ~MklFusedDepthwiseConvOp() {}
};

// We create new class for each version of Quantized Convolution and inherit
// from the FP32 version of the base class
template <typename Device, typename Tinput, typename Tbias, typename Toutput,
          typename Ttemp_output, bool bias_enabled, bool is_depthwise,
          bool native_format = false>
class MklQuantizedConv2DOp
    : public MklConvOp<Device, Tinput, qint8, Tbias, Toutput, Ttemp_output,
                       int32, bias_enabled, false, is_depthwise,
                       native_format> {
 public:
  virtual ~MklQuantizedConv2DOp() {
    if (this->input_bias_ != nullptr) {
      delete this->input_bias_;
      input_bias_ = nullptr;
    }

    if (this->scaled_bias_ != nullptr) {
      delete this->scaled_bias_;
      scaled_bias_ = nullptr;
    }
  }

  explicit MklQuantizedConv2DOp(OpKernelConstruction* context)
      : MklConvOp<Device, Tinput, qint8, Tbias, Toutput, Ttemp_output, int32,
                  bias_enabled, false, is_depthwise, native_format>(context) {
    bool is_filter_const;
    OP_REQUIRES_OK(context,
                   context->GetAttr("is_filter_const", &is_filter_const));

    if (bias_enabled) {
      OP_REQUIRES_OK(context,
                     context->GetAttr("is_bias_const", &is_bias_const_));
    }

    OP_REQUIRES(context, is_filter_const,
                errors::InvalidArgument("Filter must be a constant"));
  }

  void Compute(OpKernelContext* context) override {
    // Compute int32 output tensor
    MklConvOp<Device, Tinput, qint8, Tbias, Toutput, Ttemp_output, int32,
              bias_enabled, false, is_depthwise,
              native_format>::Compute(context);

    // Compute additional outputs: min/max scalars.
    int bias_index_offset;
    bias_index_offset = bias_enabled ? 1 : 0;

    const float min_input =
        context->input(2 + bias_index_offset).flat<float>()(0);
    const float max_input =
        context->input(3 + bias_index_offset).flat<float>()(0);

    MklDnnShape output_min_mkl_shape, output_max_mkl_shape;
    output_min_mkl_shape.SetMklTensor(false);
    output_max_mkl_shape.SetMklTensor(false);

    Tensor* output_min = nullptr;
    Tensor* output_max = nullptr;
    if (std::is_same<Toutput, quint8>::value ||
        std::is_same<Toutput, qint8>::value) {
      AllocateOutputSetMklShape(context, 1, &output_min, {},
                                output_min_mkl_shape, native_format);
      AllocateOutputSetMklShape(context, 2, &output_max, {},
                                output_max_mkl_shape, native_format);
      // This is the case the convolution and requantization are fused.
      output_min->flat<float>()(0) =
          context->input(6 + bias_index_offset).flat<float>()(0);
      output_max->flat<float>()(0) =
          context->input(7 + bias_index_offset).flat<float>()(0);
    } else {
      const Tensor& min_filter = context->input(4 + bias_index_offset);
      const Tensor& max_filter = context->input(5 + bias_index_offset);
      if (min_filter.dims() == 0) {
        float min_output_value;
        float max_output_value;
        MklQuantizationRangeForMultiplication<Tinput, qint8, qint32>(
            min_input, max_input, min_filter.flat<float>()(0),
            max_filter.flat<float>()(0), &min_output_value, &max_output_value);
        AllocateOutputSetMklShape(context, 1, &output_min, {},
                                  output_min_mkl_shape, native_format);
        AllocateOutputSetMklShape(context, 2, &output_max, {},
                                  output_max_mkl_shape, native_format);
        output_min->flat<float>()(0) = min_output_value;
        output_max->flat<float>()(0) = max_output_value;
      } else {
        size_t depth = min_filter.NumElements();
        AllocateOutputSetMklShape(context, 1, &output_min,
                                  {static_cast<ptrdiff_t>(depth)},
                                  output_min_mkl_shape, native_format);
        AllocateOutputSetMklShape(context, 2, &output_max,
                                  {static_cast<ptrdiff_t>(depth)},
                                  output_max_mkl_shape, native_format);
        MklQuantizationRangeForMultiplication<Tinput, qint8, qint32>(
            min_input, max_input, min_filter, max_filter, &output_min,
            &output_max);
      }
    }
  }

 protected:
  void ExtendConvFwdParams(OpKernelContext* context,
                           MklConvFwdParams& params) override {
    MklConvOp<Device, Tinput, qint8, Tbias, Toutput, Ttemp_output, int32,
              bias_enabled, false, is_depthwise,
              native_format>::ExtendConvFwdParams(context, params);

    // When the output type is quint8, the output data id requantized
    // into quint8. A post_op "output_scale" is added to do the conversion.
    if (std::is_same<Toutput, quint8>::value ||
        std::is_same<Toutput, qint8>::value) {
      int bias_index_offset;
      bias_index_offset = bias_enabled ? 1 : 0;

      const float min_input =
          context->input(2 + bias_index_offset).flat<float>()(0);
      const float max_input =
          context->input(3 + bias_index_offset).flat<float>()(0);
      const Tensor& min_filter_vector = context->input(4 + bias_index_offset);
      const Tensor& max_filter_vector = context->input(5 + bias_index_offset);

      // min_freezed_output and max_freezed_output are the actual range
      // for the output.
      const float min_freezed_output =
          context->input(6 + bias_index_offset).flat<float>()(0);
      const float max_freezed_output =
          context->input(7 + bias_index_offset).flat<float>()(0);

      float int_output_limit =
          std::is_same<Toutput, quint8>::value ? 255.0f : 127.0f;
      size_t depth = min_filter_vector.NumElements();
      const float* min_filter = min_filter_vector.flat<float>().data();
      const float* max_filter = max_filter_vector.flat<float>().data();
      std::vector<float> scales(depth);
      float float_input_range =
          std::max(std::abs(min_input), std::abs(max_input));
      float float_output_range =
          std::max(std::abs(min_freezed_output), std::abs(max_freezed_output));
      const float int_const_scale_limit =
          (std::is_same<Tinput, quint8>::value) ? 255.0 * 127.0 : 127.0 * 127.0;
      for (size_t i = 0; i < depth; ++i) {
        // For simplicity and symmetry, we set filter range to be outer
        // bounds of min_filter and max_filter.
        float float_filter_range =
            std::max(std::abs(min_filter[i]), std::abs(max_filter[i]));
        // To understand the scaling, please see mkl_requantize_ops_test.
        scales[i] = int_output_limit * float_input_range * float_filter_range /
                    (int_const_scale_limit * float_output_range);
      }
      // we are creating a partial key here to use with primitive key caching to
      // improve key creation performance. Instead of using actual values we are
      // using the pointers for min/max_filter_vector, and this works since the
      // filter vector here is a constant.
      FactoryKeyCreator param_key;
      param_key.AddAsKey<float>(min_input);
      param_key.AddAsKey<float>(max_input);
      param_key.AddAsKey<float>(min_freezed_output);
      param_key.AddAsKey<float>(max_freezed_output);
      param_key.AddAsKey<const float*>(min_filter);
      param_key.AddAsKey<const float*>(max_filter);
      params.post_op_params.push_back({"output_scale", mkldnn::algorithm::undef,
                                       scales, param_key.GetKey()});
    }
  }

  Tbias* GetBiasHandle(OpKernelContext* context,
                       std::shared_ptr<ConvFwdPd>& conv_fwd_pd,
                       const Tensor& bias_tensor) override {
    if (!bias_enabled) {
      return nullptr;
    }
    if (std::is_same<Tbias, qint32>::value) {
      return static_cast<Tbias*>(
          const_cast<Tbias*>(bias_tensor.flat<Tbias>().data()));
    }
    int bias_index_offset;
    bias_index_offset = bias_enabled ? 1 : 0;

    const float min_input =
        context->input(2 + bias_index_offset).flat<float>()(0);
    const float max_input =
        context->input(3 + bias_index_offset).flat<float>()(0);
    const Tensor& min_filter_vector = context->input(4 + bias_index_offset);
    const Tensor& max_filter_vector = context->input(5 + bias_index_offset);
    const float* min_filter = min_filter_vector.flat<float>().data();
    const float* max_filter = max_filter_vector.flat<float>().data();

    const float int_const_scale_limit =
        (std::is_same<Tinput, quint8>::value) ? 255.0 * 127.0 : 127.0 * 127.0;
    // Re-scale bias if either of following 2 conditions are met:
    // 1. Bias is not const;
    // 2. Bias is const, but bias cache is empty (first iteration).

    size_t depth = min_filter_vector.NumElements();
    bool scales_are_valid = (depth == scales_.size());
    scales_.resize(depth);
    for (size_t i = 0; i < depth; ++i) {
      float tmp_scale =
          int_const_scale_limit /
          (std::max(std::abs(max_input), std::abs(min_input)) *
           std::max(std::abs(max_filter[i]), std::abs(min_filter[i])));
      if (scales_are_valid && std::abs(tmp_scale - scales_[i]) > 1e-6) {
        scales_are_valid = false;
      }
      scales_[i] = tmp_scale;
    }
    if (!is_bias_const_ || IsBiasCacheEmpty(context) || !scales_are_valid) {
      mkldnn::primitive_attr bias_attr;
      if (depth == 1) {
        bias_attr.set_output_scales(0, scales_);
      } else {
        bias_attr.set_output_scales(1, scales_);
      }

      auto bias_md = memory::desc({static_cast<int>(bias_tensor.NumElements())},
                                  MklDnnType<Tbias>(), memory::format_tag::x);
      void* bias_buf = static_cast<void*>(
          const_cast<Tbias*>(bias_tensor.flat<Tbias>().data()));
      if (!input_bias_) {
        input_bias_ = new memory(bias_md, this->cpu_engine_, bias_buf);
      } else {
        input_bias_->set_data_handle(bias_buf);
      }

      if (!scaled_bias_buf_)
        AllocTmpBuffer<Tbias>(context, &scaled_bias_tensor_,
                              conv_fwd_pd->bias_desc(), &scaled_bias_buf_);
      if (!scaled_bias_) {
        scaled_bias_ = new memory(bias_md, this->cpu_engine_, scaled_bias_buf_);
      } else {
        scaled_bias_->set_data_handle(scaled_bias_buf_);
      }
      auto reorder_desc =
          ReorderPd(this->cpu_engine_, input_bias_->get_desc(),
                    this->cpu_engine_, scaled_bias_->get_desc(), bias_attr);
      CreateAndExecuteReorder(reorder_desc, *input_bias_, *scaled_bias_,
                              this->cpu_engine_, context);

      Tbias* bias_data =
          reinterpret_cast<Tbias*>(scaled_bias_->get_data_handle());
      if (is_bias_const_)
        CacheBias(context, conv_fwd_pd, bias_data, scaled_bias_);

      return bias_data;
    }
    return GetCachedBias(context);
  }

  bool is_bias_const_;
  Tensor cached_bias_data_ TF_GUARDED_BY(bias_cache_mu_);

  memory* input_bias_ = nullptr;
  memory* scaled_bias_ = nullptr;

  Tensor scaled_bias_tensor_;
  void* scaled_bias_buf_ = nullptr;

 private:
  std::vector<float> scales_;
  mutex bias_cache_mu_;
  // Allocate tensors for cached bias data and
  // cached bias memory descriptor (data format)
  void AllocateTensor(OpKernelContext* context, const ConvFwdPd& conv_prim_desc,
                      Tensor** bias_tensor) {
    DCHECK(bias_tensor);
    TensorShape bias_tf_shape;
    bias_tf_shape.AddDim(
        (conv_prim_desc.bias_desc().get_size() / sizeof(Tbias)));
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<Tbias>::value,
                                          bias_tf_shape, &cached_bias_data_));
    *bias_tensor = &cached_bias_data_;
  }

  // TF_LOCKS_EXCLUDED annotation ensures that the lock (mu_) cannot
  // be acquired before entering the function, since it is acquired
  // inside the function.
  inline bool IsBiasCacheEmpty(OpKernelContext* context)
      TF_LOCKS_EXCLUDED(bias_cache_mu_) {
    tf_shared_lock lock(bias_cache_mu_);
    return (cached_bias_data_.NumElements() == 0);
  }

  // Cache the converted bias in a tensor.
  // Only one thread can execute this method at any given time.
  void CacheBias(OpKernelContext* context,
                 const std::shared_ptr<ConvFwdPd>& conv_fwd_pd,
                 Tbias* bias_data, const memory* scaled_bias)
      TF_LOCKS_EXCLUDED(bias_cache_mu_) {
    mutex_lock lock(bias_cache_mu_);

    // If bias is already cached, there's nothing to do.
    if (cached_bias_data_.NumElements() > 0) {
      return;
    }

    // Otherwise, cache bias
    Tensor* bias_tensor_ptr = nullptr;
    AllocateTensor(context, *conv_fwd_pd, &bias_tensor_ptr);
    void* cached_bias_data = const_cast<void*>(
        static_cast<const void*>(bias_tensor_ptr->flat<Tbias>().data()));
    size_t cached_bias_data_size = scaled_bias->get_desc().get_size();
    memcpy(cached_bias_data, bias_data, cached_bias_data_size);
  }

  Tbias* GetCachedBias(OpKernelContext* context)
      TF_LOCKS_EXCLUDED(bias_cache_mu_) {
    tf_shared_lock lock(bias_cache_mu_);
    const Tensor& cached_bias_data = cached_bias_data_;

    return static_cast<Tbias*>(
        const_cast<Tbias*>(cached_bias_data.flat<Tbias>().data()));
  }
};

template <typename Device, typename Tinput, typename Tbias, typename Toutput,
          typename Ttemp_output, bool bias_enabled, bool is_depthwise,
          bool native_format = false>
class MklQuantizedConv2DReluOp
    : public MklQuantizedConv2DOp<Device, Tinput, Tbias, Toutput, Ttemp_output,
                                  bias_enabled, is_depthwise, native_format> {
 public:
  virtual ~MklQuantizedConv2DReluOp() {}

  explicit MklQuantizedConv2DReluOp(OpKernelConstruction* context)
      : MklQuantizedConv2DOp<Device, Tinput, Tbias, Toutput, Ttemp_output,
                             bias_enabled, is_depthwise, native_format>(
            context) {}

 protected:
  void ExtendConvFwdParams(OpKernelContext* context,
                           MklConvFwdParams& params) override {
    MklQuantizedConv2DOp<Device, Tinput, Tbias, Toutput, Ttemp_output,
                         bias_enabled, is_depthwise,
                         native_format>::ExtendConvFwdParams(context, params);

    params.post_op_params.push_back(
        {"activation", mkldnn::algorithm::eltwise_relu, {1.0, 0.0, 0.0}, ""});
  }
};

template <typename Device, typename Tinput, typename Tbias, typename Toutput,
          typename Ttemp_output, bool bias_enabled, bool is_depthwise,
          bool native_format = false>
class MklQuantizedConv2DSumReluOp
    : public MklQuantizedConv2DOp<Device, Tinput, Tbias, Toutput, Ttemp_output,
                                  bias_enabled, is_depthwise, native_format> {
 public:
  virtual ~MklQuantizedConv2DSumReluOp() {}

  explicit MklQuantizedConv2DSumReluOp(OpKernelConstruction* context)
      : MklQuantizedConv2DOp<Device, Tinput, Tbias, Toutput, Ttemp_output,
                             bias_enabled, is_depthwise, native_format>(
            context) {}

 protected:
  void ExtendConvFwdParams(OpKernelContext* context,
                           MklConvFwdParams& params) override {
    MklQuantizedConv2DOp<Device, Tinput, Tbias, Toutput, Ttemp_output,
                         bias_enabled, is_depthwise,
                         native_format>::ExtendConvFwdParams(context, params);
    // Calculate the scale (beta in mkldnn api term) for sum
    if (std::is_same<Toutput, quint8>::value) {
      int summand_idx = native_format ? context->num_inputs() - 1 - 2
                                      : context->num_inputs() / 2 - 1 - 2;
      DataType summand_type = this->input_type(summand_idx);
      bool summand_condition =
          (summand_type == DT_QINT8) || (summand_type == DT_QUINT8);
      CHECK((summand_condition));
      int bias_index_offset = bias_enabled ? 1 : 0;
      const float min_freezed_output =
          context->input(6 + bias_index_offset).flat<float>()(0);
      const float max_freezed_output =
          context->input(7 + bias_index_offset).flat<float>()(0);
      const float min_freezed_summand =
          context->input(9 + bias_index_offset).flat<float>()(0);
      const float max_freezed_summand =
          context->input(10 + bias_index_offset).flat<float>()(0);

      float scale_output =
          std::max(std::abs(min_freezed_output), std::abs(max_freezed_output));
      float scale_summand = std::max(std::abs(min_freezed_summand),
                                     std::abs(max_freezed_summand));
      // if summand_type is also DT_QUINT8 as the scale_output,
      // the scaling factor of 255.0f cancels each other and thus is avoided.
      // If it is not then  it is DT_INT8 and is scaled appropriately.
      if (summand_type == DT_QUINT8) {
        params.post_op_params.push_back({"sum",
                                         mkldnn::algorithm::undef,
                                         {scale_summand / scale_output},
                                         ""});
      } else {
        params.post_op_params.push_back(
            {"sum",
             mkldnn::algorithm::undef,
             {255.0f * scale_summand / (scale_output * 127.0f)},
             ""});
      }
    } else {
      params.post_op_params.push_back(
          {"sum", mkldnn::algorithm::undef, {1.0}, ""});
    }
    params.post_op_params.push_back(
        {"activation", mkldnn::algorithm::eltwise_relu, {1.0, 0.0, 0.0}, ""});
  }

  void AllocateOutputTensor(OpKernelContext* context,
                            const ConvFwdPd& conv_prim_desc,
                            const memory::dims& output_dims_mkl_order,
                            MklTensorFormat output_tf_format,
                            MklDnnShape* output_mkl_shape,
                            Tensor** output_tensor) override {
    int summand_idx = native_format ? context->num_inputs() - 1
                                    : context->num_inputs() / 2 - 1;
    if (std::is_same<Toutput, quint8>::value) {
      summand_idx -= 2;
      DataType summand_type = this->input_type(summand_idx);
      bool summand_condition =
          (summand_type == DT_QINT8) || (summand_type == DT_QUINT8);
      CHECK((summand_condition));
      Tensor& summand = const_cast<Tensor&>(MklGetInput(context, summand_idx));
      MklDnnShape summand_mkl_shape;
      GetMklShape(context, summand_idx, &summand_mkl_shape, native_format);
      auto dst_md = summand_mkl_shape.GetMklLayout();

      // TODO(intel-tf): Handle both non-MKL and MKL tensors
      if (summand_type == DT_QINT8) {
        OP_REQUIRES_OK(
            context, summand.BitcastFrom(summand, DT_QUINT8, summand.shape()));
        dst_md.data.data_type =
            static_cast<mkldnn_data_type_t>(MklDnnType<Toutput>());
        summand_mkl_shape.SetMklLayout(&dst_md);
        summand_mkl_shape.SetElemType(MklDnnType<Toutput>());
      }
      // TODO(intel-tf): Support cases when summand cannot be forwarded.
      OP_REQUIRES(context,
                  native_format
                      ? context->forward_input_to_output_with_shape(
                            summand_idx, 0, summand.shape(), output_tensor)
                      : ForwardMklTensorInToOutWithMklShape(
                            context, summand_idx, 0, output_tensor,
                            summand_mkl_shape, false),
                  errors::InvalidArgument(
                      "Summand cannot be forwarded in the current fusion."));
      return;
    }
    MklConvOp<Device, Tinput, qint8, Tbias, Toutput, Ttemp_output, int32,
              bias_enabled, false, false,
              native_format>::AllocateOutputTensor(context, conv_prim_desc,
                                                   output_dims_mkl_order,
                                                   output_tf_format,
                                                   output_mkl_shape,
                                                   output_tensor);
    const Tensor& summand = MklGetInput(context, summand_idx);
    if (summand.dtype() != DT_FLOAT)
      TF_CHECK_OK(Status(error::Code::FAILED_PRECONDITION,
                         "Current fusion requires summand to be float"));
    MklDnnShape summand_mkl_shape;
    GetMklShape(context, summand_idx, &summand_mkl_shape, native_format);
    // We need to compute scale for the summand
    int bias_index_offset = bias_enabled ? 1 : 0;
    const float min_input =
        context->input(2 + bias_index_offset).flat<float>()(0);
    const float max_input =
        context->input(3 + bias_index_offset).flat<float>()(0);
    const Tensor& min_filter_vector = context->input(4 + bias_index_offset);
    const Tensor& max_filter_vector = context->input(5 + bias_index_offset);
    const float* min_filter = min_filter_vector.flat<float>().data();
    const float* max_filter = max_filter_vector.flat<float>().data();

    const float int_const_scale_limit =
        (std::is_same<Tinput, quint8>::value) ? 255.0 * 127.0 : 127.0 * 127.0;
    size_t depth = min_filter_vector.NumElements();
    std::vector<float> scales(depth);
    for (size_t i = 0; i < depth; ++i) {
      // TODO(nammbash): scale factors for UINT8(inputs) & INT8(weights) are
      // done regularly. A Cleaner design to address all mapping in one
      // function needs to be implemented in future which also supports other
      // quantized type mapping in future.
      scales[i] = int_const_scale_limit /
                  (std::max(std::abs(max_input), std::abs(min_input)) *
                   std::max(std::abs(max_filter[i]), std::abs(min_filter[i])));
    }
    mkldnn::primitive_attr reorder_attr;
    if (depth == 1) {
      reorder_attr.set_output_scales(0, scales);
    } else {
      reorder_attr.set_output_scales(2, scales);
    }
    auto summand_md =
        summand_mkl_shape.IsMklTensor()
            ? summand_mkl_shape.GetMklLayout()
            : memory::desc(output_dims_mkl_order, MklDnnType<Tbias>(),
                           memory::format_tag::nhwc);
    void* summand_buf =
        static_cast<void*>(const_cast<Tbias*>(summand.flat<Tbias>().data()));
    void* dst_buf =
        static_cast<void*>((*output_tensor)->flat<Ttemp_output>().data());
    summand_.reset(new memory(summand_md, this->cpu_engine_, summand_buf));
    dst_.reset(
        new memory(conv_prim_desc.dst_desc(), this->cpu_engine_, dst_buf));
    auto reorder_desc =
        ReorderPd(this->cpu_engine_, summand_md, this->cpu_engine_,
                  conv_prim_desc.dst_desc(), reorder_attr);
    CreateAndExecuteReorder(reorder_desc, *summand_, *dst_, this->cpu_engine_,
                            context);
  }

  std::shared_ptr<mkldnn::memory> summand_;
  std::shared_ptr<mkldnn::memory> dst_;
};

#define REGISTER_MKL_KERNEL(op, kernel, input_type, bias_type, output_type, \
                            accu_type, has_bias, is_depthwise, is_native)   \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name(op)                                                              \
          .Device(DEVICE_CPU)                                               \
          .TypeConstraint<input_type>("Tinput")                             \
          .TypeConstraint<qint8>("Tfilter") BIAS_TYPE_CONSTRAINT(bias_type) \
          .TypeConstraint<output_type>("out_type") LABEL,                   \
      kernel TEMPLATE_ARGS(CPUDevice, input_type, bias_type, output_type,   \
                           accu_type, has_bias, is_depthwise, is_native));

#define REGISTER_MKL_KERNEL_ALL_INPUT_TYPES(op, kernel, bias_type,            \
                                            output_type, accu_type, has_bias, \
                                            is_depthwise, is_native)          \
  REGISTER_MKL_KERNEL(op, kernel, qint8, bias_type, output_type, accu_type,   \
                      has_bias, is_depthwise, is_native);                     \
  REGISTER_MKL_KERNEL(op, kernel, quint8, bias_type, output_type, accu_type,  \
                      has_bias, is_depthwise, is_native);

#define REGISTER_MKL_KERNEL_ALL_BIAS_TYPES(op, kernel, input_type,            \
                                           output_type, accu_type, has_bias,  \
                                           is_depthwise, is_native)           \
  REGISTER_MKL_KERNEL(op, kernel, input_type, qint32, output_type, accu_type, \
                      has_bias, is_depthwise, is_native);                     \
  REGISTER_MKL_KERNEL(op, kernel, input_type, float, output_type, accu_type,  \
                      has_bias, is_depthwise, is_native);

#define REGISTER_MKL_KERNEL_ALL_INPUT_AND_BIAS_TYPES(                      \
    op, kernel, output_type, accu_type, has_bias, is_depthwise, is_native) \
  REGISTER_MKL_KERNEL_ALL_INPUT_TYPES(op, kernel, qint32, output_type,     \
                                      accu_type, has_bias, is_depthwise,   \
                                      is_native);                          \
  REGISTER_MKL_KERNEL_ALL_INPUT_TYPES(op, kernel, float, output_type,      \
                                      accu_type, has_bias, is_depthwise,   \
                                      is_native);

#define LABEL
#define TEMPLATE_ARGS(CPUDevice, input_type, bias_type, output_type, \
                      accu_type, has_bias, is_depthwise, is_native)
#define BIAS_TYPE_CONSTRAINT(bias_type)

REGISTER_MKL_KERNEL("QuantizedConv2D", NoOp, quint8, float, qint32, qint32,
                    false, false, false);
REGISTER_MKL_KERNEL_ALL_INPUT_TYPES("QuantizedConv2DWithBias", NoOp, float,
                                    qint32, qint32, false, false, false);
REGISTER_MKL_KERNEL_ALL_INPUT_TYPES("QuantizedConv2DWithBiasAndRelu", NoOp,
                                    float, qint32, qint32, false, false, false);
REGISTER_MKL_KERNEL("QuantizedConv2DWithBiasSumAndRelu", NoOp, quint8, float,
                    qint32, qint32, false, false, false);
REGISTER_MKL_KERNEL("QuantizedConv2DAndRequantize", NoOp, quint8, float, qint8,
                    qint8, false, false, false);
REGISTER_MKL_KERNEL("QuantizedConv2DPerChannel", NoOp, quint8, float, qint32,
                    qint32, false, false, false);
REGISTER_MKL_KERNEL("QuantizedConv2DAndRelu", NoOp, quint8, float, qint32,
                    qint32, false, false, false);
REGISTER_MKL_KERNEL("QuantizedConv2DAndReluAndRequantize", NoOp, quint8, float,
                    quint8, quint8, false, false, false);
REGISTER_MKL_KERNEL("QuantizedDepthwiseConv2D", NoOp, quint8, float, qint32,
                    qint32, false, false, false);
REGISTER_MKL_KERNEL("QuantizedDepthwiseConv2DWithBias", NoOp, quint8, float,
                    qint32, qint32, false, false, false);
REGISTER_MKL_KERNEL("QuantizedDepthwiseConv2DWithBiasAndRelu", NoOp, quint8,
                    float, qint32, qint32, false, false, false);
#undef BIAS_TYPE_CONSTRAINT

#define BIAS_TYPE_CONSTRAINT(bias_type) .TypeConstraint<bias_type>("Tbias")
REGISTER_MKL_KERNEL_ALL_INPUT_AND_BIAS_TYPES(
    "QuantizedConv2DWithBiasAndRequantize", NoOp, qint8, qint8, false, false,
    false);
REGISTER_MKL_KERNEL_ALL_INPUT_AND_BIAS_TYPES(
    "QuantizedConv2DWithBiasAndReluAndRequantize", NoOp, quint8, quint8, false,
    false, false);
REGISTER_MKL_KERNEL_ALL_BIAS_TYPES(
    "QuantizedConv2DWithBiasSumAndReluAndRequantize", NoOp, quint8, quint8,
    quint8, false, false, false);
REGISTER_MKL_KERNEL_ALL_BIAS_TYPES(
    "QuantizedConv2DWithBiasSignedSumAndReluAndRequantize", NoOp, quint8,
    quint8, qint8, false, false, false);
REGISTER_MKL_KERNEL_ALL_BIAS_TYPES(
    "QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize", NoOp, quint8,
    quint8, quint8, false, false, false);
#undef BIAS_TYPE_CONSTRAINT
#undef TEMPLATE_ARGS
#undef LABEL

#define LABEL .Label(mkl_op_registry::kMklQuantizedOpLabel)
#define TEMPLATE_ARGS(CPUDevice, input_type, bias_type, output_type, \
                      accu_type, has_bias, is_depthwise, is_native)  \
<CPUDevice, input_type, bias_type, output_type, accu_type, has_bias, is_depthwise, is_native>
#define BIAS_TYPE_CONSTRAINT(bias_type)
REGISTER_MKL_KERNEL_ALL_INPUT_TYPES("_MklQuantizedConv2D", MklQuantizedConv2DOp,
                                    float, qint32, qint32, false, false, true);
REGISTER_MKL_KERNEL("_MklQuantizedConv2DPerChannel", MklQuantizedConv2DOp,
                    quint8, float, qint32, qint32, false, false, true);
REGISTER_MKL_KERNEL_ALL_INPUT_TYPES("_MklQuantizedConv2DWithBias",
                                    MklQuantizedConv2DOp, float, qint32, qint32,
                                    true, false, true);
REGISTER_MKL_KERNEL_ALL_INPUT_TYPES("_MklQuantizedConv2DWithBiasAndRelu",
                                    MklQuantizedConv2DReluOp, float, qint32,
                                    qint32, true, false, true);
REGISTER_MKL_KERNEL("_MklQuantizedConv2DWithBiasSumAndRelu",
                    MklQuantizedConv2DSumReluOp, quint8, float, qint32, qint32,
                    true, false, true);
REGISTER_MKL_KERNEL("_MklQuantizedConv2DAndRequantize", MklQuantizedConv2DOp,
                    quint8, float, qint8, qint8, false, false, true);
REGISTER_MKL_KERNEL("_MklQuantizedConv2DAndRelu", MklQuantizedConv2DReluOp,
                    quint8, float, qint32, qint32, false, false, true);
REGISTER_MKL_KERNEL("_MklQuantizedConv2DAndReluAndRequantize",
                    MklQuantizedConv2DReluOp, quint8, float, quint8, quint8,
                    false, false, true);
REGISTER_MKL_KERNEL("_MklQuantizedDepthwiseConv2D", MklQuantizedConv2DOp,
                    quint8, float, qint32, qint32, false, true, true);
REGISTER_MKL_KERNEL("_MklQuantizedDepthwiseConv2DWithBias",
                    MklQuantizedConv2DOp, quint8, float, qint32, qint32, true,
                    true, true);
REGISTER_MKL_KERNEL("_MklQuantizedDepthwiseConv2DWithBiasAndRelu",
                    MklQuantizedConv2DReluOp, quint8, float, qint32, qint32,
                    true, true, true);
#undef BIAS_TYPE_CONSTRAINT

#define BIAS_TYPE_CONSTRAINT(bias_type) .TypeConstraint<bias_type>("Tbias")
REGISTER_MKL_KERNEL_ALL_INPUT_AND_BIAS_TYPES(
    "_MklQuantizedConv2DWithBiasAndRequantize", MklQuantizedConv2DOp, qint8,
    qint8, true, false, true);
REGISTER_MKL_KERNEL_ALL_INPUT_AND_BIAS_TYPES(
    "_MklQuantizedConv2DWithBiasAndReluAndRequantize", MklQuantizedConv2DReluOp,
    quint8, quint8, true, false, true);
REGISTER_MKL_KERNEL_ALL_BIAS_TYPES(
    "_MklQuantizedConv2DWithBiasSumAndReluAndRequantize",
    MklQuantizedConv2DSumReluOp, quint8, quint8, quint8, true, false, true);
REGISTER_MKL_KERNEL_ALL_BIAS_TYPES(
    "_MklQuantizedConv2DWithBiasSignedSumAndReluAndRequantize",
    MklQuantizedConv2DSumReluOp, quint8, quint8, qint8, true, false, true);
REGISTER_MKL_KERNEL_ALL_BIAS_TYPES(
    "_MklQuantizedDepthwiseConv2DWithBiasAndReluAndRequantize",
    MklQuantizedConv2DReluOp, quint8, quint8, quint8, true, true, true);
#undef BIAS_TYPE_CONSTRAINT
#undef TEMPLATE_ARGS
#undef LABEL

// Register NoOp kernel for ops that will be rewritten to the _Mkl* version
REGISTER_KERNEL_BUILDER(Name("DepthwiseConv2dNative")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<bfloat16>("T"),
                        NoOp);

#define REGISTER_NO_OP_CPU_2D_DEPTHWISE(T)                    \
  REGISTER_KERNEL_BUILDER(Name("_FusedDepthwiseConv2dNative") \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<T>("T"),        \
                          NoOp);

TF_CALL_float(REGISTER_NO_OP_CPU_2D_DEPTHWISE);
TF_CALL_bfloat16(REGISTER_NO_OP_CPU_2D_DEPTHWISE);

// Register 2D operations
#define REGISTER_MKL_CPU_2D(T)                                                 \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_MklConv2D")                                                       \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<T>("T")                                              \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),                 \
      MklConvOp<CPUDevice, T, T, T, T, T, int32, false, false, false, false>); \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_MklConv2DWithBias")                                               \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<T>("T")                                              \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),                 \
      MklConvOp<CPUDevice, T, T, T, T, T, int32, true, false, false, false>);  \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("__MklDummyConv2DWithBias")                                         \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<T>("T")                                              \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),                 \
      MklDummyOp<CPUDevice, T>);                                               \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_MklPadWithConv2D")                                                \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<T>("T")                                              \
          .TypeConstraint<int32>("Tpaddings")                                  \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),                 \
      MklConvOp<CPUDevice, T, T, T, T, T, int32, false, true, false, false>);  \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_MklPadWithConv2D")                                                \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<T>("T")                                              \
          .TypeConstraint<int64>("Tpaddings")                                  \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),                 \
      MklConvOp<CPUDevice, T, T, T, T, T, int64, false, true, false, false>);  \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("__MklDummyPadWithConv2D")                                          \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<T>("T")                                              \
          .TypeConstraint<int32>("Tpaddings")                                  \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),                 \
      MklDummyOp<CPUDevice, T>);                                               \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_MklNativeConv2D")                                                 \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<T>("T")                                              \
          .Label(mkl_op_registry::kMklNameChangeOpLabel),                      \
      MklConvOp<CPUDevice, T, T, T, T, T, int32, false, false, false, true>);  \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_MklNativeConv2DWithBias")                                         \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<T>("T")                                              \
          .Label(mkl_op_registry::kMklNameChangeOpLabel),                      \
      MklConvOp<CPUDevice, T, T, T, T, T, int32, true, false, false, true>);   \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_MklNativePadWithConv2D")                                          \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<T>("T")                                              \
          .TypeConstraint<int32>("Tpaddings")                                  \
          .Label(mkl_op_registry::kMklNameChangeOpLabel),                      \
      MklConvOp<CPUDevice, T, T, T, T, T, int32, false, true, false, true>);   \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_MklNativePadWithConv2D")                                          \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<T>("T")                                              \
          .TypeConstraint<int64>("Tpaddings")                                  \
          .Label(mkl_op_registry::kMklNameChangeOpLabel),                      \
      MklConvOp<CPUDevice, T, T, T, T, T, int64, false, true, false, true>);

TF_CALL_float(REGISTER_MKL_CPU_2D);
TF_CALL_bfloat16(REGISTER_MKL_CPU_2D);

#define REGISTER_MKL_CPU_2D_DEPTHWISE(T)                                      \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_MklDepthwiseConv2dNative")                                       \
          .Device(DEVICE_CPU)                                                 \
          .TypeConstraint<T>("T")                                             \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),                \
      MklConvOp<CPUDevice, T, T, T, T, T, int32, false, false, true, false>); \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_MklFusedDepthwiseConv2dNative")                                  \
          .Device(DEVICE_CPU)                                                 \
          .TypeConstraint<T>("T")                                             \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),                \
      MklFusedDepthwiseConvOp<CPUDevice, T, T, T, T, T, int32, false, true,   \
                              true, false>);                                  \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_MklNativeFusedDepthwiseConv2dNative")                            \
          .Device(DEVICE_CPU)                                                 \
          .TypeConstraint<T>("T")                                             \
          .Label(mkl_op_registry::kMklNameChangeOpLabel),                     \
      MklFusedDepthwiseConvOp<CPUDevice, T, T, T, T, T, int32, false, true,   \
                              true, true>);                                   \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_MklNativeDepthwiseConv2dNative")                                 \
          .Device(DEVICE_CPU)                                                 \
          .TypeConstraint<T>("T")                                             \
          .Label(mkl_op_registry::kMklNameChangeOpLabel),                     \
      MklConvOp<CPUDevice, T, T, T, T, T, int32, false, false, true, true>);

TF_CALL_float(REGISTER_MKL_CPU_2D_DEPTHWISE);
TF_CALL_bfloat16(REGISTER_MKL_CPU_2D_DEPTHWISE);

// Note we are registering _MklFusedConv2D.
// We check the fused_ops attributes to decide if bias is enabled or not.
#define REGISTER_MKL_CPU_2D_FUSED(T)                                  \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("_MklFusedConv2D")                                         \
          .Device(DEVICE_CPU)                                         \
          .TypeConstraint<T>("T")                                     \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),        \
      MklFusedConvOp<CPUDevice, T, T, T, T, T, int32, false, false>); \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("_MklPadWithFusedConv2D")                                  \
          .Device(DEVICE_CPU)                                         \
          .TypeConstraint<int32>("Tpaddings")                         \
          .TypeConstraint<T>("T")                                     \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),        \
      MklFusedConvOp<CPUDevice, T, T, T, T, T, int32, true, false>);  \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("_MklPadWithFusedConv2D")                                  \
          .Device(DEVICE_CPU)                                         \
          .TypeConstraint<T>("T")                                     \
          .TypeConstraint<int64>("Tpaddings")                         \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),        \
      MklFusedConvOp<CPUDevice, T, T, T, T, T, int64, true, false>);  \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("__MklDummyPadWithFusedConv2D")                            \
          .Device(DEVICE_CPU)                                         \
          .TypeConstraint<T>("T")                                     \
          .TypeConstraint<int32>("Tpaddings")                         \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),        \
      MklDummyOp<CPUDevice, T>);                                      \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("_MklNativeFusedConv2D")                                   \
          .Device(DEVICE_CPU)                                         \
          .TypeConstraint<T>("T")                                     \
          .Label(mkl_op_registry::kMklNameChangeOpLabel),             \
      MklFusedConvOp<CPUDevice, T, T, T, T, T, int32, false, true>);  \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("_MklNativePadWithFusedConv2D")                            \
          .Device(DEVICE_CPU)                                         \
          .TypeConstraint<int32>("Tpaddings")                         \
          .TypeConstraint<T>("T")                                     \
          .Label(mkl_op_registry::kMklNameChangeOpLabel),             \
      MklFusedConvOp<CPUDevice, T, T, T, T, T, int32, true, true>);   \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("_MklNativePadWithFusedConv2D")                            \
          .Device(DEVICE_CPU)                                         \
          .TypeConstraint<T>("T")                                     \
          .TypeConstraint<int64>("Tpaddings")                         \
          .Label(mkl_op_registry::kMklNameChangeOpLabel),             \
      MklFusedConvOp<CPUDevice, T, T, T, T, T, int64, true, true>);

TF_CALL_float(REGISTER_MKL_CPU_2D_FUSED);
TF_CALL_bfloat16(REGISTER_MKL_CPU_2D_FUSED);

// Register 3D operations
#define REGISTER_MKL_CPU_3D(T)                                                 \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_MklConv3D")                                                       \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<T>("T")                                              \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),                 \
      MklConvOp<CPUDevice, T, T, T, T, T, int32, false, false, false, false>); \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_MklNativeConv3D")                                                 \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<T>("T")                                              \
          .Label(mkl_op_registry::kMklNameChangeOpLabel),                      \
      MklConvOp<CPUDevice, T, T, T, T, T, int32, false, false, false, true>);
TF_CALL_float(REGISTER_MKL_CPU_3D);
TF_CALL_bfloat16(REGISTER_MKL_CPU_3D);

}  // namespace tensorflow
#endif  // INTEL_MKL
