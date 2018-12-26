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

#include <string.h>
#include <algorithm>
#include <map>
#include <vector>

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/mkl_conv_ops.h"
#include "tensorflow/core/kernels/mkl_quantized_conv_ops.h"
#include "tensorflow/core/kernels/no_op.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

#include "tensorflow/core/util/mkl_util.h"

#ifndef INTEL_MKL_ML_ONLY
#include "mkldnn.hpp"

using mkldnn::prop_kind;
using mkldnn::stream;
using mkldnn::convolution_forward;
using mkldnn::convolution_direct;

#else
#include "mkl_dnn.h"
#include "mkl_dnn_types.h"
#endif

namespace tensorflow {

#ifndef INTEL_MKL_ML_ONLY

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
  string dtypes = string("");
  struct PostOpParam {
    string name;
    std::vector<float> param;
  };
  std::vector<PostOpParam> post_op_params;

  MklConvFwdParams(memory::dims src_dims, memory::dims filter_dims,
                   memory::dims bias_dims, memory::dims dst_dims,
                   memory::dims strides, memory::dims dilations,
                   memory::dims padding_left, memory::dims padding_right)
      : src_dims(src_dims),
        filter_dims(filter_dims),
        bias_dims(bias_dims),
        dst_dims(dst_dims),
        strides(strides),
        dilations(dilations),
        padding_left(padding_left),
        padding_right(padding_right) {}
};
// With quantization, input, filter, and output can have different types
// so we use differnt template parameter for each type
template <typename T, typename Tinput, typename Tfilter, typename Tbias,
          typename Toutput>
class MklConvFwdPrimitive : public MklPrimitive {
 public:
  explicit MklConvFwdPrimitive(const MklConvFwdParams& convFwdDims)
      : cpu_engine_(engine::cpu, 0) {
    context_.fwd_stream.reset(new stream(stream::kind::eager));
    // create conv primitive
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
               const Tbias* bias_data, const Toutput* dst_data) {
    context_.src_mem->set_data_handle(
        static_cast<void*>(const_cast<Tinput*>(src_data)));
    context_.filter_mem->set_data_handle(
        static_cast<void*>(const_cast<Tfilter*>(filter_data)));
    context_.bias_mem->set_data_handle(
        static_cast<void*>(const_cast<Tbias*>(bias_data)));
    context_.dst_mem->set_data_handle(
        static_cast<void*>(const_cast<Toutput*>(dst_data)));
    context_.fwd_stream->submit(context_.fwd_primitives);

    // after exec, set data handle back
    context_.src_mem->set_data_handle(DummyData);
    context_.filter_mem->set_data_handle(DummyData);
    context_.bias_mem->set_data_handle(DummyData);
    context_.dst_mem->set_data_handle(DummyData);

    return;
  }

  // Convolution forward execute without bias
  //   src_data:    input data buffer of src
  //   filter_data: input data buffer of filter (weights)
  //   dst_data:    output data buffer of dst
  void Execute(const Tinput* src_data, const Tfilter* filter_data,
               const Toutput* dst_data) {
    context_.src_mem->set_data_handle(
        static_cast<void*>(const_cast<Tinput*>(src_data)));
    context_.filter_mem->set_data_handle(
        static_cast<void*>(const_cast<Tfilter*>(filter_data)));
    context_.dst_mem->set_data_handle(
        static_cast<void*>(const_cast<Toutput*>(dst_data)));
    context_.fwd_stream->submit(context_.fwd_primitives);

    // after execution, set data handle back
    context_.src_mem->set_data_handle(DummyData);
    context_.filter_mem->set_data_handle(DummyData);
    context_.dst_mem->set_data_handle(DummyData);
  }

  memory::format GetSrcMemoryFormat() const { return context_.src_fmt; }

  memory::format GetFilterMemoryFormat() const { return context_.filter_fmt; }

  std::shared_ptr<mkldnn::convolution_forward::primitive_desc>
  GetPrimitiveDesc() const {
    return context_.fwd_pd;
  }

 private:
  // Primitive reuse context for Conv2D Fwd op
  struct ConvFwdContext {
    // expected memory format for this primitive instance
    memory::format src_fmt;
    memory::format filter_fmt;

    // MKLDNN memory
    std::shared_ptr<mkldnn::memory> src_mem;
    std::shared_ptr<mkldnn::memory> filter_mem;
    std::shared_ptr<mkldnn::memory> bias_mem;
    std::shared_ptr<mkldnn::memory> dst_mem;

    // desc & prmitive desc
    std::shared_ptr<mkldnn::convolution_forward::desc> fwd_desc;

    // memory desc
    std::shared_ptr<mkldnn::memory::desc> src_md;
    std::shared_ptr<mkldnn::memory::desc> filter_md;
    std::shared_ptr<mkldnn::memory::desc> bias_md;
    std::shared_ptr<mkldnn::memory::desc> dst_md;

    // convolution primitive
    std::shared_ptr<mkldnn::convolution_forward::primitive_desc> fwd_pd;
    std::shared_ptr<mkldnn::primitive> conv_fwd;

    std::shared_ptr<mkldnn::stream> fwd_stream;
    std::vector<mkldnn::primitive> fwd_primitives;

    ConvFwdContext()
        : src_fmt(memory::format::any),
          filter_fmt(memory::format::any),
          src_mem(nullptr),
          filter_mem(nullptr),
          bias_mem(nullptr),
          dst_mem(nullptr),
          fwd_desc(nullptr),
          src_md(nullptr),
          filter_md(nullptr),
          bias_md(nullptr),
          fwd_pd(nullptr),
          conv_fwd(nullptr),
          fwd_stream(nullptr) {}
  };

  void Setup(const MklConvFwdParams& convFwdDims) {
    // create memory descriptors for convolution data w/ no specified format
    context_.src_md.reset(new memory::desc(
        {convFwdDims.src_dims}, MklDnnType<Tinput>(), memory::format::any));

    context_.filter_md.reset(new memory::desc(
        {convFwdDims.filter_dims}, MklDnnType<Tfilter>(), memory::format::any));

    context_.dst_md.reset(new memory::desc(
        {convFwdDims.dst_dims}, MklDnnType<Toutput>(), memory::format::any));

    if (!convFwdDims.bias_dims.empty())
      context_.bias_md.reset(new memory::desc(
          {convFwdDims.bias_dims}, MklDnnType<Tbias>(), memory::format::any));

    // create a convolution
    if (!convFwdDims.bias_dims.empty()) {
      context_.fwd_desc.reset(new convolution_forward::desc(
          prop_kind::forward, convolution_direct, *context_.src_md,
          *context_.filter_md, *context_.bias_md, *context_.dst_md,
          convFwdDims.strides, convFwdDims.dilations, convFwdDims.padding_left,
          convFwdDims.padding_right, padding_kind::zero));
    } else {
      context_.fwd_desc.reset(new convolution_forward::desc(
          prop_kind::forward, convolution_direct, *context_.src_md,
          *context_.filter_md, *context_.dst_md, convFwdDims.strides,
          convFwdDims.dilations, convFwdDims.padding_left,
          convFwdDims.padding_right, padding_kind::zero));
    }

    context_.fwd_pd.reset(new convolution_forward::primitive_desc(
        *context_.fwd_desc, cpu_engine_));

    // Check if there is any fusions as post-ops
    auto const& post_op_params = convFwdDims.post_op_params;
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
        } else {
          DCHECK((post_op_param.name == "relu") ||
                 (post_op_param.name == "sum") ||
                 (post_op_param.name == "output_scale"));
        }
      }
      post_ops_attr.set_post_ops(post_ops);
      context_.fwd_pd.reset(new convolution_forward::primitive_desc(
          *context_.fwd_desc, post_ops_attr, cpu_engine_));
    } else {
      context_.fwd_pd.reset(new convolution_forward::primitive_desc(
          *context_.fwd_desc, cpu_engine_));
    }

    // store the expected memory format
    context_.src_fmt = static_cast<mkldnn::memory::format>(
        context_.fwd_pd.get()->src_primitive_desc().desc().data.format);

    context_.filter_fmt = static_cast<mkldnn::memory::format>(
        context_.fwd_pd.get()->weights_primitive_desc().desc().data.format);

    // create memory primitive based on dummy data
    context_.src_mem.reset(
        new memory(context_.fwd_pd.get()->src_primitive_desc(), DummyData));
    context_.filter_mem.reset(
        new memory(context_.fwd_pd.get()->weights_primitive_desc(), DummyData));
    context_.dst_mem.reset(
        new memory(context_.fwd_pd.get()->dst_primitive_desc(), DummyData));

    // create convolution primitive and add it to net
    if (!convFwdDims.bias_dims.empty()) {
      context_.bias_mem.reset(new memory(
          {{{convFwdDims.bias_dims}, MklDnnType<T>(), memory::format::x},
           cpu_engine_},
          DummyData));
      context_.conv_fwd.reset(new convolution_forward(
          *context_.fwd_pd, *context_.src_mem, *context_.filter_mem,
          *context_.bias_mem, *context_.dst_mem));
    } else {
      context_.conv_fwd.reset(
          new convolution_forward(*context_.fwd_pd, *context_.src_mem,
                                  *context_.filter_mem, *context_.dst_mem));
    }

    context_.fwd_primitives.push_back(*context_.conv_fwd);
    return;
  }

  struct ConvFwdContext context_;
  engine cpu_engine_;
};

template <typename T, typename Tinput, typename Tfilter, typename Tbias,
          typename Toutput>
class MklConvFwdPrimitiveFactory : public MklPrimitiveFactory<T> {
 public:
  static MklConvFwdPrimitive<T, Tinput, Tfilter, Tbias, Toutput>* Get(
      const MklConvFwdParams& convFwdDims, bool do_not_cache) {
    MklConvFwdPrimitive<T, Tinput, Tfilter, Tbias, Toutput>* conv_fwd = nullptr;

    if (do_not_cache) { /* Always create new primitive */
      conv_fwd = new MklConvFwdPrimitive<T, Tinput, Tfilter, Tbias, Toutput>(
          convFwdDims);
    } else {
      // try to find a suitable one in pool
      conv_fwd = dynamic_cast<
          MklConvFwdPrimitive<T, Tinput, Tfilter, Tbias, Toutput>*>(
          MklConvFwdPrimitiveFactory<T, Tinput, Tfilter, Tbias,
                                     Toutput>::GetInstance()
              .GetConvFwd(convFwdDims));
      if (conv_fwd == nullptr) {
        conv_fwd = new MklConvFwdPrimitive<T, Tinput, Tfilter, Tbias, Toutput>(
            convFwdDims);
        MklConvFwdPrimitiveFactory<T, Tinput, Tfilter, Tbias,
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
    key_creator.AddAsKey(convFwdDims.bias_dims);
    key_creator.AddAsKey(convFwdDims.dst_dims);
    key_creator.AddAsKey(convFwdDims.strides);
    key_creator.AddAsKey(convFwdDims.dilations);
    key_creator.AddAsKey(convFwdDims.padding_left);
    key_creator.AddAsKey(convFwdDims.padding_right);
    key_creator.AddAsKey(convFwdDims.dtypes);

    // Generate keys for post-ops
    for (auto const& post_op_param : convFwdDims.post_op_params) {
      if (post_op_param.name == "relu") {
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
        key_creator.AddAsKey(post_op_param.name);
        size_t nelems = post_op_param.param.size();
        for (size_t i = 0; i < nelems; i++)
          key_creator.AddAsKey(post_op_param.param[i]);
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

#endif

typedef Eigen::ThreadPoolDevice CPUDevice;

// For now, MKL-ML is default. So making MKL-DNN not a default choice.
#ifdef INTEL_MKL_ML_ONLY
template <typename Device, typename T, bool bias_enabled>
class MklConvOp : public OpKernel {
 public:
  ~MklConvOp() {}

  explicit MklConvOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));

    const int64 stride_n = GetTensorDim(strides_, data_format_, 'N');
    const int64 stride_c = GetTensorDim(strides_, data_format_, 'C');
    OP_REQUIRES(
        context, stride_n == 1 && stride_c == 1,
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    MklConv2DOpContext mkl_context;
    const Tensor& input = MklGetInput(context, 0);
    GetMklShape(context, 0, &(mkl_context.input_shape));
    bool input_in_mkl_format = mkl_context.input_shape.IsMklTensor();

    const Tensor& filter = MklGetInput(context, 1);
    MklShape mkl_filter_shape;
    GetMklShape(context, 1, &mkl_filter_shape);
    CHECK(!mkl_filter_shape.IsMklTensor())
        << "Conv filter should not be in MKL Layout";

    if (bias_enabled) {
      const Tensor& bias = MklGetInput(context, 2);
      OP_REQUIRES(context, bias.dims() == 1,
                  errors::InvalidArgument("bias must be 1-dimensional: ",
                                          bias.shape().DebugString()));
    }

    if (!input_in_mkl_format) {
      OP_REQUIRES(context, input.dims() == 4,
                  errors::InvalidArgument("input must be 4-dimensional",
                                          input.shape().DebugString()));
    }

    OP_REQUIRES(context, filter.dims() == 4,
                errors::InvalidArgument("filter must be 4-dimensional: ",
                                        filter.shape().DebugString()));

    for (int i = 0; i < 3; i++) {
      OP_REQUIRES(context, FastBoundsCheck(filter.dim_size(i),
                                           std::numeric_limits<int>::max()),
                  errors::InvalidArgument("filter too large"));
    }

    const int64 input_depth =
        input_in_mkl_format ? GetMklTensorDim(mkl_context.input_shape, 'C')
                            : GetTensorDim(input, data_format_, 'C');
    OP_REQUIRES(
        context, input_depth == filter.dim_size(2),
        errors::InvalidArgument("input and filter must have the same depth: ",
                                input_depth, " vs ", filter.dim_size(2)));
    // The last dimension for filter is out_depth.
    const int out_depth = static_cast<int>(filter.dim_size(3));

    // The second dimension for input is rows/height.
    // The first dimension for filter is rows/height.
    const int64 input_rows_raw =
        input_in_mkl_format ? GetMklTensorDim(mkl_context.input_shape, 'H')
                            : GetTensorDim(input, data_format_, 'H');
    OP_REQUIRES(context, FastBoundsCheck(input_rows_raw,
                                         std::numeric_limits<int>::max()),
                errors::InvalidArgument("Input rows too large"));
    const int input_rows = static_cast<int>(input_rows_raw);
    const int filter_rows = static_cast<int>(filter.dim_size(0));

    // The third dimension for input is columns/width.
    // The second dimension for filter is columns/width.
    const int64 input_cols_raw =
        input_in_mkl_format ? GetMklTensorDim(mkl_context.input_shape, 'W')
                            : GetTensorDim(input, data_format_, 'W');
    OP_REQUIRES(context, FastBoundsCheck(input_cols_raw,
                                         std::numeric_limits<int>::max()),
                errors::InvalidArgument("Input cols too large"));
    const int input_cols = static_cast<int>(input_cols_raw);
    const int filter_cols = static_cast<int>(filter.dim_size(1));

    // The first dimension for input is batch.
    const int64 input_batch_raw =
        input_in_mkl_format ? GetMklTensorDim(mkl_context.input_shape, 'N')
                            : GetTensorDim(input, data_format_, 'N');
    OP_REQUIRES(context, FastBoundsCheck(input_batch_raw,
                                         std::numeric_limits<int>::max()),
                errors::InvalidArgument("batch is too large"));
    const int batch = static_cast<int>(input_batch_raw);

    // For now we take the stride from the second and third dimensions only (we
    // do not support striding on the batch or depth dimension).
    const int stride_rows = GetTensorDim(strides_, data_format_, 'H');
    const int stride_cols = GetTensorDim(strides_, data_format_, 'W');

    int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_rows, filter_rows, stride_rows,
                                         padding_, &out_rows, &pad_rows));
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_cols, filter_cols, stride_cols,
                                         padding_, &out_cols, &pad_cols));
    TensorShape out_shape =
        ShapeFromFormat(data_format_, batch, out_rows, out_cols, out_depth);

    // Output tensor is of the following dimensions:
    // [ in_batch, out_rows, out_cols, out_depth ]
    Tensor* output = nullptr;

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      // Nothing to do, allocate output tensor and return
      MklShape mkl_output_mkl_shape;
      mkl_output_mkl_shape.SetMklTensor(false);
      AllocateOutputSetMklShape(context, 0, &output, input.shape(),
                                mkl_output_mkl_shape);
      return;
    }

    if (batch == 0) {
      // Nothing to do, allocate output tensor and return
      MklShape mkl_output_mkl_shape;
      mkl_output_mkl_shape.SetMklTensor(false);
      AllocateOutputSetMklShape(context, 0, &output, input.shape(),
                                mkl_output_mkl_shape);
      return;
    }

    // Create MKL convolution primitives
    mkl_context.in_dims = input_in_mkl_format
                              ? mkl_context.input_shape.GetDimension()
                              : input.dims();
    mkl_context.filter_dims = filter.dims();

    mkl_context.in_sizes[MklDims::W] = static_cast<size_t>(input_cols);
    mkl_context.in_sizes[MklDims::H] = static_cast<size_t>(input_rows);
    mkl_context.in_sizes[MklDims::C] = static_cast<size_t>(input_depth);
    mkl_context.in_sizes[MklDims::N] = static_cast<size_t>(batch);

    mkl_context.out_sizes[MklDims::W] = static_cast<size_t>(out_cols);
    mkl_context.out_sizes[MklDims::H] = static_cast<size_t>(out_rows);
    mkl_context.out_sizes[MklDims::C] = static_cast<size_t>(out_depth);
    mkl_context.out_sizes[MklDims::N] = static_cast<size_t>(batch);

    mkl_context.input_offset[0] = static_cast<int>(-pad_cols);
    mkl_context.input_offset[1] = static_cast<int>(-pad_rows);

    mkl_context.conv_stride[0] = static_cast<size_t>(stride_cols);
    mkl_context.conv_stride[1] = static_cast<size_t>(stride_rows);

    GetStridesFromSizes(data_format_, mkl_context.out_strides,
                        mkl_context.out_sizes);
    GetStridesFromSizes(data_format_, mkl_context.in_strides,
                        mkl_context.in_sizes);

    // TF filter dimension order (out_depth, in_depth, cols, rows) ->
    // MKL filter dimension order (out_depth, in_depth, rows, cols)
    mkl_context.filter_sizes[0] = filter.dim_size(1);  // cols
    mkl_context.filter_sizes[1] = filter.dim_size(0);  // rows
    mkl_context.filter_sizes[2] = filter.dim_size(2);  // in_depth
    mkl_context.filter_sizes[3] = filter.dim_size(3);  // out_depth

    // TF filter layout - (rows, cols, in_depth, out_depth)
    mkl_context.filter_strides[0] =
        filter.dim_size(2) * filter.dim_size(3);  // cols
    mkl_context.filter_strides[1] =
        filter.dim_size(1) * filter.dim_size(2) * filter.dim_size(3);  // rows
    mkl_context.filter_strides[2] = filter.dim_size(3);  // in_depth
    mkl_context.filter_strides[3] = 1;                   // out_depth

    if (bias_enabled) {
      const Tensor& bias = MklGetInput(context, 2);
      mkl_context.bias_sizes[0] = {static_cast<size_t>(bias.dim_size(0))};
      mkl_context.bias_strides[0] = {1};
    }

    // Create Convolution Primitive
    if (bias_enabled) {
      CHECK_EQ(
          dnnConvolutionCreateForwardBias_F32(
              &mkl_context.prim_fwd, nullptr, dnnAlgorithmConvolutionDirect,
              mkl_context.in_dims, mkl_context.in_sizes, mkl_context.out_sizes,
              mkl_context.filter_sizes, mkl_context.conv_stride,
              mkl_context.input_offset, dnnBorderZeros),
          E_SUCCESS);
    } else {
      CHECK_EQ(
          dnnConvolutionCreateForward_F32(
              &mkl_context.prim_fwd, nullptr, dnnAlgorithmConvolutionDirect,
              mkl_context.in_dims, mkl_context.in_sizes, mkl_context.out_sizes,
              mkl_context.filter_sizes, mkl_context.conv_stride,
              mkl_context.input_offset, dnnBorderZeros),
          E_SUCCESS);
    }

    TensorShape mkl_output_tf_shape;
    MklShape mkl_output_mkl_shape;
    mkl_output_mkl_shape.SetMklTensor(true);
    mkl_output_mkl_shape.SetMklLayout(mkl_context.prim_fwd, dnnResourceDst);
    mkl_output_mkl_shape.SetTfLayout(mkl_context.in_dims, mkl_context.out_sizes,
                                     mkl_context.out_strides);
    // MKL might change the dimension ordering
    // Create mapping to recover the original TF dimension order
    mkl_output_mkl_shape.SetTfDimOrder(mkl_context.in_dims, data_format_);

    mkl_output_tf_shape.AddDim(
        dnnLayoutGetMemorySize_F32(
            static_cast<dnnLayout_t>(mkl_output_mkl_shape.GetMklLayout())) /
        sizeof(T));
    AllocateOutputSetMklShape(context, 0, &output, mkl_output_tf_shape,
                              mkl_output_mkl_shape);
    // Filter output to be used in the backprop_input
    TensorShape mkl_filter_output_tf_shape;
    MklShape mkl_filter_output_mkl_shape;
    mkl_filter_output_mkl_shape.SetMklTensor(true);
    mkl_filter_output_mkl_shape.SetMklLayout(mkl_context.prim_fwd,
                                             dnnResourceFilter);

    size_t filter_sizes[4] = {static_cast<size_t>(filter.dim_size(0)),
                              static_cast<size_t>(filter.dim_size(1)),
                              static_cast<size_t>(filter.dim_size(2)),
                              static_cast<size_t>(filter.dim_size(3))};
    mkl_filter_output_mkl_shape.SetTfLayout(filter.dims(), filter_sizes,
                                            mkl_context.filter_strides);

    mkl_filter_output_mkl_shape.SetTfDimOrder(mkl_context.filter_dims,
                                              data_format_);
    mkl_filter_output_tf_shape.AddDim(
        dnnLayoutGetMemorySize_F32(static_cast<dnnLayout_t>(
            mkl_filter_output_mkl_shape.GetMklLayout())) /
        sizeof(T));
    AllocateOutputSetMklShape(context, 1, &mkl_context.output_filter,
                              mkl_filter_output_tf_shape,
                              mkl_filter_output_mkl_shape);

    mkl_context.conv_res[dnnResourceDst] =
        static_cast<void*>(output->flat<T>().data());

    mkl_context.MklCreateInputLayouts(context);

    // Temp tensor used to allocate tmp buffers
    Tensor mkl_tmp_input_buf_tensor, mkl_tmp_filter_buf_tensor,
        mkl_tmp_bias_buf_tensor;
    mkl_context.MklPrepareConvolutionInputs(context, &mkl_tmp_input_buf_tensor,
                                            &mkl_tmp_filter_buf_tensor,
                                            &mkl_tmp_bias_buf_tensor);

    // Execute convolution
    CHECK_EQ(dnnExecute_F32(mkl_context.prim_fwd, mkl_context.conv_res),
             E_SUCCESS);

    mkl_context.MklCleanup();
  }

 private:
  typedef struct {
    int in_dims;
    size_t in_sizes[4];
    size_t in_strides[4];
    size_t out_sizes[4];
    size_t out_strides[4];
    int filter_dims;
    size_t filter_sizes[4];
    size_t filter_strides[4];
    size_t bias_sizes[1];
    size_t bias_strides[1];
    int input_offset[2];
    size_t conv_stride[2];
    MklShape input_shape;
    dnnPrimitive_t prim_fwd;
    void* conv_res[dnnResourceNumber];
    dnnLayout_t lt_filter, lt_bias, lt_input;
    Tensor* output_filter = nullptr;

    // Create MKL dnnLayout_t objects for tensors coming into the layer
    void MklCreateInputLayouts(OpKernelContext* context) {
      bool input_in_mkl_format = input_shape.IsMklTensor();
      if (input_in_mkl_format) {
        lt_input = static_cast<dnnLayout_t>(input_shape.GetCurLayout());
      } else {
        CHECK_EQ(dnnLayoutCreate_F32(&lt_input, in_dims, in_sizes, in_strides),
                 E_SUCCESS);
      }

      CHECK_EQ(dnnLayoutCreate_F32(&lt_filter, filter_dims, filter_sizes,
                                   filter_strides),
               E_SUCCESS);

      if (bias_enabled) {
        CHECK_EQ(dnnLayoutCreate_F32(&lt_bias, 1, bias_sizes, bias_strides),
                 E_SUCCESS);
      }
    }

    // Compare incoming tensor layouts with MKL preferred layouts and convert
    // data to the preferred layout if necessary
    void MklPrepareConvolutionInputs(OpKernelContext* context,
                                     Tensor* mkl_tmp_input_buf_tensor,
                                     Tensor* mkl_tmp_filter_buf_tensor,
                                     Tensor* mkl_tmp_bias_buf_tensor) {
      bool mkl_convert_input, mkl_convert_filter, mkl_convert_bias;
      dnnPrimitive_t mkl_prim_convert_filter, mkl_prim_convert_bias,
          mkl_prim_convert_input;
      dnnLayout_t mkl_lt_internal_filter, mkl_lt_internal_bias,
          mkl_lt_internal_input;
      void *mkl_buf_convert_input, *mkl_buf_convert_filter,
          *mkl_buf_convert_bias;
      mkl_prim_convert_filter = nullptr;
      mkl_prim_convert_bias = nullptr;
      mkl_prim_convert_input = nullptr;
      mkl_lt_internal_filter = nullptr;
      mkl_lt_internal_bias = nullptr;
      mkl_lt_internal_input = nullptr;
      mkl_buf_convert_input = nullptr;
      mkl_buf_convert_filter = nullptr;
      mkl_buf_convert_bias = nullptr;

      // Compare with internal layouts and convert if needed
      const Tensor& input = MklGetInput(context, 0);
      void* mkl_buf_input =
          const_cast<void*>(static_cast<const void*>(input.flat<T>().data()));
      CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(&mkl_lt_internal_input,
                                                prim_fwd, dnnResourceSrc),
               E_SUCCESS);
      mkl_convert_input =
          !dnnLayoutCompare_F32(mkl_lt_internal_input, lt_input);
      if (mkl_convert_input) {
        CHECK_EQ(dnnConversionCreate_F32(&mkl_prim_convert_input, lt_input,
                                         mkl_lt_internal_input),
                 E_SUCCESS);
        AllocTmpBuffer(context, mkl_tmp_input_buf_tensor, mkl_lt_internal_input,
                       &mkl_buf_convert_input);
        CHECK_EQ(dnnConversionExecute_F32(mkl_prim_convert_input, mkl_buf_input,
                                          mkl_buf_convert_input),
                 E_SUCCESS);
        dnnDelete_F32(mkl_prim_convert_input);
      }
      dnnLayoutDelete_F32(mkl_lt_internal_input);

      conv_res[dnnResourceSrc] =
          (mkl_convert_input) ? mkl_buf_convert_input : mkl_buf_input;

      const Tensor& filter = MklGetInput(context, 1);
      void* mkl_buf_filter =
          const_cast<void*>(static_cast<const void*>(filter.flat<T>().data()));
      CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(&mkl_lt_internal_filter,
                                                prim_fwd, dnnResourceFilter),
               E_SUCCESS);
      mkl_convert_filter =
          !dnnLayoutCompare_F32(mkl_lt_internal_filter, lt_filter);
      if (mkl_convert_filter) {
        CHECK_EQ(dnnConversionCreate_F32(&mkl_prim_convert_filter, lt_filter,
                                         mkl_lt_internal_filter),
                 E_SUCCESS);

        mkl_buf_convert_filter = const_cast<void*>(
            static_cast<const void*>(output_filter->flat<T>().data()));

        CHECK_EQ(
            dnnConversionExecute_F32(mkl_prim_convert_filter, mkl_buf_filter,
                                     mkl_buf_convert_filter),
            E_SUCCESS);
        dnnDelete_F32(mkl_prim_convert_filter);
      }
      dnnLayoutDelete_F32(mkl_lt_internal_filter);

      conv_res[dnnResourceFilter] =
          (mkl_convert_filter) ? mkl_buf_convert_filter : mkl_buf_filter;

      if (bias_enabled) {
        const Tensor& bias = MklGetInput(context, 2);
        void* mkl_buf_bias =
            const_cast<void*>(static_cast<const void*>(bias.flat<T>().data()));
        CHECK_EQ(dnnLayoutCreateFromPrimitive_F32(&mkl_lt_internal_bias,
                                                  prim_fwd, dnnResourceBias),
                 E_SUCCESS);
        mkl_convert_bias = !dnnLayoutCompare_F32(mkl_lt_internal_bias, lt_bias);
        if (mkl_convert_bias) {
          CHECK_EQ(dnnConversionCreate_F32(&mkl_prim_convert_bias, lt_bias,
                                           mkl_lt_internal_bias),
                   E_SUCCESS);
          AllocTmpBuffer(context, mkl_tmp_bias_buf_tensor, mkl_lt_internal_bias,
                         &mkl_buf_convert_bias);
          CHECK_EQ(dnnConversionExecute_F32(mkl_prim_convert_bias, mkl_buf_bias,
                                            mkl_buf_convert_bias),
                   E_SUCCESS);
          dnnDelete_F32(mkl_prim_convert_bias);
        }
        dnnLayoutDelete_F32(mkl_lt_internal_bias);

        conv_res[dnnResourceBias] =
            (mkl_convert_bias) ? mkl_buf_convert_bias : mkl_buf_bias;
      }
    }

    void MklCleanup() {
      bool input_in_mkl_format = input_shape.IsMklTensor();
      dnnDelete_F32(prim_fwd);
      if (!input_in_mkl_format) dnnLayoutDelete_F32(lt_input);
      dnnLayoutDelete_F32(lt_filter);
      if (bias_enabled) dnnLayoutDelete_F32(lt_bias);
    }
  } MklConv2DOpContext;

  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;
};

// FP32 kernel registration for INTEL_MKL_ML
REGISTER_KERNEL_BUILDER(Name("_MklConv2D")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T")
                            .Label(mkl_op_registry::kMklOpLabel),
                        MklConv2DOp<CPUDevice, float, false>);
REGISTER_KERNEL_BUILDER(Name("_MklConv2DWithBias")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T")
                            .Label(mkl_op_registry::kMklOpLabel),
                        MklConv2DOp<CPUDevice, float, true>);

#else

// Base class for convolution forward operations
template <typename Device, typename Tinput, typename Tfilter, typename Tbias,
          typename Toutput, typename Ttemp_output, typename Tpadding,
          bool bias_enabled, bool pad_enabled, bool is_depthwise>
class MklConvOp : public OpKernel {
 public:
  ~MklConvOp() {}

  explicit MklConvOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations_));
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
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));

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
      OP_REQUIRES(context, (GetTensorDim(dilations_, data_format_, 'N') == 1 &&
                            GetTensorDim(dilations_, data_format_, 'C') == 1),
                  errors::InvalidArgument(
                      "Current implementation does not yet support "
                      "dilations rates in the batch and depth dimensions."));
      OP_REQUIRES(
          context, (GetTensorDim(dilations_, data_format_, '0') > 0 &&
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
      GetMklShape(context, kInputIndex_Src, &src_mkl_shape);
      GetMklShape(context, kInputIndex_Filter, &filter_mkl_shape);
      OP_REQUIRES(context, filter_mkl_shape.IsMklTensor() == false,
                  errors::InvalidArgument("Filter should not be in "
                                          "Mkl Layout"));

      MklDnnData<Tinput> src(&cpu_engine_);
      MklDnnData<Tfilter> filter(&cpu_engine_);

      memory::dims src_dims, filter_dims, padding_left, padding_right,
          dilations, strides;
      memory::dims dst_dims_tf_order, dst_dims_mkl_order;

      // If pad with conv2d fusion is enabled
      if (pad_enabled) {
        PadWithConvFusion(context, padding_left, padding_right);
      }

      // Get shapes of input tensors in MKL-DNN order
      MklDnnConvUtil conv_utl(context, strides_, padding_, data_format_,
                              dilations_);
      auto src_tf_shape = GetTfShape(context, kInputIndex_Src);
      auto filter_tf_shape = GetTfShape(context, kInputIndex_Filter);
      conv_utl.GetConvFwdSizesInMklOrder(
          src_tf_shape, filter_tf_shape, &src_dims, &filter_dims, &strides,
          &dilations, &dst_dims_tf_order, &dst_dims_mkl_order, &padding_left,
          &padding_right, pad_enabled, is_depthwise);
      if (!context->status().ok()) return;

      // Check for corner case - if there is nothing to compute, return.
      TensorShape dst_tf_shape = MklDnnDimsToTFShape(dst_dims_tf_order);

      // Corner cases: output with 0 elements and 0 batch size.
      Tensor* dst_tensor = nullptr;
      if (dst_tf_shape.num_elements() == 0 || dst_dims_tf_order[0] == 0) {
        MklDnnShape dst_mkl_shape;
        dst_mkl_shape.SetMklTensor(false);
        AllocateOutputSetMklShape(context, kOutputIndex_Dst, &dst_tensor,
                                  src_tf_shape, dst_mkl_shape);

        // MklConv2D/3D also outputs converted filter
        // as 2nd output of Conv2D/3D.
        filter_mkl_shape.SetMklTensor(false);
        Tensor* output_filter_tensor = nullptr;
        // MklConv2D also outputs converted filter as 2nd output.
        if (typeid(Tinput) == typeid(float) &&
            typeid(Tfilter) == typeid(float) &&
            typeid(Toutput) == typeid(float)) {
          filter_mkl_shape.SetMklTensor(false);
          AllocateOutputSetMklShape(context, kOutputIndex_Filter,
                                    &output_filter_tensor, filter_tf_shape,
                                    filter_mkl_shape);
        }
        return;
      }

      bool is_conv2d = (strides_.size() == 4);

      // TODO 3-D support for Depthwise is not there
      if (is_depthwise) {
        OP_REQUIRES(context, is_conv2d,
                    errors::InvalidArgument(
                        "Only 2D convolution is supported for depthwise."));
      }

      // TODO(Intel-tf) Add check to make sure pad_enabled is true only for 2D
      if (!is_conv2d) {
        OP_REQUIRES(
            context, !pad_enabled,
            errors::InvalidArgument("Pad+Conv fusion only works for 2D"));
      }
      // Create memory for user data.
      // Describe how the inputs and outputs of Convolution look like. Also
      // specify buffers containing actual input and output data.
      auto tf_fmt = is_conv2d ? TFDataFormatToMklDnnDataFormat(data_format_)
                              : TFDataFormatToMklDnn3DDataFormat(data_format_);

      // If input is in MKL layout, then simply grab input layout; otherwise,
      // construct input Tf layout. For TF layout, although input shape
      // (src_dims) required is in MKL-DNN order, the layout is Tensorflow's
      // layout depending on data format:
      //     Conv2D: NHWC or NCHW
      //     Conv3D: NDHWC or NCDHW
      auto src_md = src_mkl_shape.IsMklTensor()
                        ? src_mkl_shape.GetMklLayout()
                        : memory::desc(src_dims, MklDnnType<Tinput>(), tf_fmt);
      src.SetUsrMem(src_md, &src_tensor);

      // Although filter shape (filter_dims) required is in MKL-DNN order,
      // the layout is Tensorflow's layout (HWIO) and (HWIGO)for depthwise/group
      // convolutions

      auto filter_format = is_conv2d ? (is_depthwise ? memory::format::hwigo
                                                     : memory::format::hwio)
                                     : memory::format::dhwio;

      DCHECK(!filter_mkl_shape.IsMklTensor());
      auto filter_md =
          filter_mkl_shape.IsMklTensor()
              ? filter_mkl_shape.GetMklLayout()
              : memory::desc(filter_dims, MklDnnType<Tfilter>(), filter_format);
      filter.SetUsrMem(filter_md, &filter_tensor);
      // MKLDNN dilation starts from 0.
      for (int i = 0; i < dilations.size(); i++) dilations[i] -= 1;

      // In some cases, primitve descriptor includes potentialy large buffers,
      // we don't cache those primitves if the env variable
      // TF_MKL_OPTIMIZE_PRIMITIVE_MEMUSE is true. MKL DNN allocates buffers
      // in the following cases
      //   1. Legacy CPU without AVX512/AVX2, or
      //   2. 1x1 convolution with stride != 1
      bool do_not_cache =
          MklPrimitiveFactory<Tinput>::IsPrimitiveMemOptEnabled() &&
          (src_dims[MklDnnDims::Dim_N] > kSmallBatchSize) &&
          (MklPrimitiveFactory<Tinput>::IsLegacyPlatform() ||
           IsConv1x1StrideNot1(filter_dims, strides));

      // get a conv2d fwd from primitive pool
      MklConvFwdPrimitive<float, Tinput, Tfilter, Tbias, Ttemp_output>*
          conv_fwd = nullptr;
      if (fuse_biasadd_) {
        memory::dims bias_dims = {};
        conv_utl.GetBiasSizeInMklOrder(kInputIndex_Bias, &bias_dims);
        MklConvFwdParams convFwdDims(src_dims, filter_dims, bias_dims,
                                     dst_dims_mkl_order, strides, dilations,
                                     padding_left, padding_right);

        // TODO(mdfaijul):  Extend the basic parameters for data types and
        // fusions
        this->ExtendConvFwdParams(context, convFwdDims);

        conv_fwd = MklConvFwdPrimitiveFactory<float, Tinput, Tfilter, Tbias,
                                              Ttemp_output>::Get(convFwdDims,
                                                                 do_not_cache);
      } else {
        MklConvFwdParams convFwdDims(src_dims, filter_dims, NONE_DIMS,
                                     dst_dims_mkl_order, strides, dilations,
                                     padding_left, padding_right);

        // Extend the basic parameters for data types and fusions
        this->ExtendConvFwdParams(context, convFwdDims);

        conv_fwd = MklConvFwdPrimitiveFactory<float, Tinput, Tfilter, Tbias,
                                              Ttemp_output>::Get(convFwdDims,
                                                                 do_not_cache);
      }

      // allocate output tensors output_tensor and filter_out_tensor
      std::shared_ptr<mkldnn::convolution_forward::primitive_desc> conv_fwd_pd =
          conv_fwd->GetPrimitiveDesc();
      AllocateOutputTensor(context, *conv_fwd_pd, dst_dims_mkl_order, tf_fmt,
                           &dst_tensor);
      Tensor* filter_out_tensor = nullptr;
      if (typeid(Tinput) == typeid(float) && typeid(Tfilter) == typeid(float) &&
          typeid(Toutput) == typeid(float)) {
        AllocateFilterOutputTensor(context, *conv_fwd_pd,
                                   TFShapeToMklDnnDims(filter_tf_shape),
                                   &filter_out_tensor);
      }

      Ttemp_output* dst_data =
          reinterpret_cast<Ttemp_output*>(dst_tensor->flat<Toutput>().data());

      // check whether src/filter need reorder
      Tinput* src_data = nullptr;
      if (src_md.data.format != conv_fwd->GetSrcMemoryFormat()) {
        src.SetUsrMem(src_md, &src_tensor);
        src.CheckReorderToOpMem(conv_fwd_pd.get()->src_primitive_desc());
        src_data = static_cast<Tinput*>(src.GetOpMem().get_data_handle());
      } else {
        src_data = static_cast<Tinput*>(
            const_cast<Tinput*>(src_tensor.flat<Tinput>().data()));
      }
      Tfilter* filter_data = nullptr;
      if (filter_md.data.format != conv_fwd->GetFilterMemoryFormat()) {
        filter.SetUsrMem(filter_md, &filter_tensor);
        if (filter_out_tensor == nullptr) {
          filter.CheckReorderToOpMem(
              conv_fwd_pd.get()->weights_primitive_desc());
        } else {
          filter.CheckReorderToOpMem(
              conv_fwd_pd.get()->weights_primitive_desc(),
              filter.GetTensorBuffer(filter_out_tensor));
        }
        filter_data =
            static_cast<Tfilter*>(filter.GetOpMem().get_data_handle());
      } else {
        filter_data = static_cast<Tfilter*>(
            const_cast<Tfilter*>(filter_tensor.flat<Tfilter>().data()));
      }

      // execute convolution
      if (fuse_biasadd_) {
        const Tensor& bias_tensor = MklGetInput(context, kInputIndex_Bias);
        Tbias* bias_data =
            this->GetBiasHandle(context, conv_fwd_pd, bias_tensor);
        conv_fwd->Execute(src_data, filter_data, bias_data, dst_data);
      } else {
        conv_fwd->Execute(src_data, filter_data, dst_data);
      }

      // delete primitive since it is not cached.
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
                         memory::dims& padding_right) {
    const Tensor& paddings_tf = MklGetInput(context, 2);
    OP_REQUIRES(context, paddings_tf.dims() == 2,
                errors::InvalidArgument("paddings must be 2-dimensional: ",
                                        paddings_tf.shape().DebugString()));
    Tpadding* paddings = nullptr;
    // To get individual pad, need to flatten the tensor
    paddings = static_cast<Tpadding*>(
        const_cast<Tpadding*>(paddings_tf.flat<Tpadding>().data()));
    // For NHWC format:
    // paddings[0], paddings[1], paddings[6], paddings[7] should be zero
    // if the paddings_tf is [ [0, 0] [1,2] [3,4] [0,0] ]
    // paddings = {0, 0, 1, 2, 3, 4, 0, 0} ; flat method is row major
    // then, values are: top = 1, bottom =2, left=3, right=4
    // For NCHW format:
    // paddings[0], paddings[1], paddings[2], paddings[3] should be zero
    // similar explanation as NHWC format will apply.
    int64 pad_top, pad_left;
    int64 pad_bottom, pad_right;
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
    // Create padding arrays for MKL DNN convolutions.
    // MKL-DNN uses asymetric padding.
    padding_left = {static_cast<int>(pad_top), static_cast<int>(pad_left)};
    padding_right = {static_cast<int>(pad_bottom), static_cast<int>(pad_right)};
  }

 protected:
  void set_fuse_biasadd(bool fuse_biasadd) { fuse_biasadd_ = fuse_biasadd; }
  void set_fuse_relu(bool fuse_relu) { fuse_relu_ = fuse_relu; }

  // This method is for the base class MklConvOp, which handles the
  // floating point implementation of Conv. The quantized conv implementations
  // will use overidden versions of this method.
  virtual void ExtendConvFwdParams(OpKernelContext* context,
                                   MklConvFwdParams& params) {
    // Create a string from data types of input, filter, bias, and output.
    params.dtypes.append(typeid(Tinput).name());
    params.dtypes.append(typeid(Tfilter).name());
    params.dtypes.append(typeid(Tbias).name());
    params.dtypes.append(typeid(Toutput).name());

    // Add fusions as post ops
    // Note: Fusion of BiasAdd is handled directly inside MklConvOp by
    // checking fuse_biasadd_ flag.
    if (fuse_relu_) params.post_op_params.push_back({"relu", {1.0, 0.0, 0.0}});
  }

  virtual Tbias* GetBiasHandle(
      OpKernelContext* context,
      std::shared_ptr<mkldnn::convolution_forward::primitive_desc>&
          conv2d_fwd_pd,
      const Tensor& bias_tensor) {
    if (fuse_biasadd_) {
      return static_cast<Tbias*>(
          const_cast<Tbias*>(bias_tensor.flat<Tbias>().data()));
    } else {
      return nullptr;
    }
  }

  // Allocate output tensor.
  virtual void AllocateOutputTensor(
      OpKernelContext* context,
      const convolution_forward::primitive_desc& conv_prim_desc,
      const memory::dims& output_dims_mkl_order,
      memory::format output_tf_format, Tensor** output_tensor) {
    CHECK_NOTNULL(output_tensor);
    auto dst_pd = conv_prim_desc.dst_primitive_desc();

    auto dst_md = dst_pd.desc();
    if (!std::is_same<Ttemp_output, Toutput>::value) {
      dst_md.data.data_type =
          static_cast<mkldnn_data_type_t>(MklDnnType<Toutput>());
      dst_pd = memory::primitive_desc(dst_md, cpu_engine_);
    }
    // Allocate shape of Mkl tensor.
    MklDnnShape output_mkl_shape;
    output_mkl_shape.SetMklTensor(true);
    output_mkl_shape.SetMklLayout(&dst_pd);
    output_mkl_shape.SetElemType(MklDnnType<Toutput>());
    output_mkl_shape.SetTfLayout(output_dims_mkl_order.size(),
                                 output_dims_mkl_order, output_tf_format);

    // Allocate shape of TF tensor.
    TensorShape output_tf_shape;
    output_tf_shape.AddDim((dst_pd.get_size() / sizeof(Toutput)));

    AllocateOutputSetMklShape(context, kOutputIndex_Dst, output_tensor,
                              output_tf_shape, output_mkl_shape);
  }

  engine cpu_engine_ = engine(engine::cpu, 0);

 private:
  std::vector<int32> strides_;
  std::vector<int32> dilations_;
  Padding padding_;
  TensorFormat data_format_;

  // Initialize to values the template is instantiated with
  bool fuse_biasadd_ = bias_enabled;
  bool fuse_relu_ = false;

  const int kInputIndex_Src = 0, kInputIndex_Filter = 1, kInputIndex_Bias = 2;
  const int kInputIndex_Pad = 2;
  const int kOutputIndex_Dst = 0, kOutputIndex_Filter = 1;
  const int kDilationH = 0, kDilationW = 1;

  // Allocate filter output tensor.
  void AllocateFilterOutputTensor(
      OpKernelContext* context,
      const convolution_forward::primitive_desc& conv_prim_desc,
      const memory::dims& filter_dims_tf_order, Tensor** filter_tensor) {
    CHECK_NOTNULL(filter_tensor);
    auto filter_pd = conv_prim_desc.weights_primitive_desc();

    // Allocate shape of Mkl tensor.
    MklDnnShape filter_mkl_shape;
    filter_mkl_shape.SetMklTensor(true);
    filter_mkl_shape.SetMklLayout(&filter_pd);
    filter_mkl_shape.SetElemType(MklDnnType<Tfilter>());

    // The format of the filter is actually OIhw8i8o, but TF doesn't support
    // this format. Just use format::blocked for now because the layout
    // is stored in the MKL data.
    filter_mkl_shape.SetTfLayout(filter_dims_tf_order.size(),
                                 filter_dims_tf_order, memory::format::blocked);

    // Allocate the data space for the filter to propagate as TF tensor.
    TensorShape filter_tf_shape;
    filter_tf_shape.AddDim((filter_pd.get_size() / sizeof(Tfilter)));

    AllocateOutputSetMklShape(context, kOutputIndex_Filter, filter_tensor,
                              filter_tf_shape, filter_mkl_shape);
  }
  // Prepare and execute net - checks for input and output reorders.
  void PrepareAndExecuteNet(
      const convolution_forward::primitive_desc& conv_prim_desc,
      MklDnnData<Tinput>* src, MklDnnData<Tfilter>* filter,
      MklDnnData<Tbias>* bias, MklDnnData<Toutput>* output,
      Tensor* filter_out_tensor) {
    CHECK_NOTNULL(filter_out_tensor);

    // Create reorders between user layout and MKL layout if it is needed and
    // add it to the net before convolution. No need to check for output
    // reorder as we propagate output layout to the next layer.
    src->CheckReorderToOpMem(conv_prim_desc.src_primitive_desc());

    // rather than re-order to a temp buffer, reorder directly to the
    // filter output tensor
    filter->CheckReorderToOpMem(conv_prim_desc.weights_primitive_desc(),
                                filter->GetTensorBuffer(filter_out_tensor));

    // Create convolution primitive and add it to net.
    std::vector<primitive> net;
    if (bias) {
      DCHECK(fuse_biasadd_);
      net.push_back(convolution_forward(conv_prim_desc, src->GetOpMem(),
                                        filter->GetOpMem(), bias->GetOpMem(),
                                        output->GetOpMem()));
    } else {
      DCHECK(!fuse_biasadd_);
      net.push_back(convolution_forward(conv_prim_desc, src->GetOpMem(),
                                        filter->GetOpMem(),
                                        output->GetOpMem()));
    }

    stream(stream::kind::eager).submit(net).wait();
  }
};

// Base class for fused convolution forward operations
template <typename Device, typename Tinput, typename Tfilter, typename Tbias,
          typename Toutput, typename Ttemp_output>
class MklFusedConvOp
    : public MklConvOp<Device, Tinput, Tfilter, Tbias, Toutput, Ttemp_output,
                       int32, false, false, false> {
 public:
  explicit MklFusedConvOp(OpKernelConstruction* context)
      : MklConvOp<Device, Tinput, Tfilter, Tbias, Toutput, Ttemp_output, int32,
                  false, false, false>(context) {
    // Since we came here through the registration of _MklFusedConv2D, get
    // all information from 'fused_ops' and 'num_args'
    std::vector<string> fused_ops;
    OP_REQUIRES_OK(context, context->GetAttr("fused_ops", &fused_ops));

    int num_args;
    OP_REQUIRES_OK(context, context->GetAttr("num_args", &num_args));
    OP_REQUIRES(context, !fused_ops.empty(),
                errors::InvalidArgument(
                    "Fused Conv2D must have at least one fused op."));

    if (fused_ops == std::vector<string>{"BiasAdd"}) {
      this->set_fuse_biasadd(true);
      OP_REQUIRES(context, num_args == 1,
                  errors::InvalidArgument(
                      "Fused Conv2D must have one extra argument: bias."));
    } else if (fused_ops == std::vector<string>{"Relu"}) {
      this->set_fuse_relu(true);
    } else if (fused_ops == std::vector<string>{"BiasAdd", "Relu"}) {
      this->set_fuse_biasadd(true);
      this->set_fuse_relu(true);
      OP_REQUIRES(context, num_args == 1,
                  errors::InvalidArgument(
                      "Fused Conv2D must have one extra argument: bias."));
    } else {
      OP_REQUIRES(context, false,
                  errors::Unimplemented("Fusion is not implemented: [",
                                        str_util::Join(fused_ops, ","), "]"));
    }
  }

  virtual ~MklFusedConvOp() {}
};

// We create new class for each verison of Quantized Convolution and inherit
// from the FP32 version of the base class
template <typename Device, typename Tbias, typename Toutput,
          typename Ttemp_output, bool bias_enabled>
class MklQuantizedConv2DOp
    : public MklConvOp<Device, quint8, qint8, Tbias, Toutput, Ttemp_output,
                       int32, bias_enabled, false, false> {
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
      : MklConvOp<Device, quint8, qint8, Tbias, Toutput, Ttemp_output, int32,
                  bias_enabled, false, false>(context) {}

  void Compute(OpKernelContext* context) override {
    // Compute int32 output tensor
    MklConvOp<Device, quint8, qint8, Tbias, Toutput, Ttemp_output, int32,
              bias_enabled, false, false>::Compute(context);

    // Compute additional outputs: min/max scalars.
    int bias_index_offset;
    bias_index_offset = bias_enabled ? 1 : 0;

    const float min_input =
        context->input(2 + bias_index_offset).flat<float>()(0);
    const float max_input =
        context->input(3 + bias_index_offset).flat<float>()(0);

    Tensor* output_min = nullptr;
    Tensor* output_max = nullptr;
    MklDnnShape output_min_mkl_shape, output_max_mkl_shape;
    output_min_mkl_shape.SetMklTensor(false);
    output_max_mkl_shape.SetMklTensor(false);

    if (std::is_same<Toutput, quint8>::value ||
        std::is_same<Toutput, qint8>::value) {
      AllocateOutputSetMklShape(context, 1, &output_min, {},
                                output_min_mkl_shape);
      AllocateOutputSetMklShape(context, 2, &output_max, {},
                                output_max_mkl_shape);
      // This is the case the convolution and requantization are fused.
      // min_freezed_output and max_freezed_output are the actual range
      // for the output
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
        MklQuantizationRangeForMultiplication<quint8, qint8, qint32>(
            min_input, max_input, min_filter.flat<float>()(0),
            max_filter.flat<float>()(0), &min_output_value, &max_output_value);
        AllocateOutputSetMklShape(context, 1, &output_min, {},
                                  output_min_mkl_shape);
        AllocateOutputSetMklShape(context, 2, &output_max, {},
                                  output_max_mkl_shape);
        output_min->flat<float>()(0) = min_output_value;
        output_max->flat<float>()(0) = max_output_value;
      } else {
        size_t depth = min_filter.NumElements();
        AllocateOutputSetMklShape(context, 1, &output_min, {depth},
                                  output_min_mkl_shape);
        AllocateOutputSetMklShape(context, 2, &output_max, {depth},
                                  output_max_mkl_shape);
        MklQuantizationRangeForMultiplication<quint8, qint8, qint32>(
            min_input, max_input, min_filter, max_filter, &output_min,
            &output_max);
      }
    }
  }

 protected:
  void ExtendConvFwdParams(OpKernelContext* context,
                           MklConvFwdParams& params) override {
    MklConvOp<Device, quint8, qint8, Tbias, Toutput, Ttemp_output, int32,
              bias_enabled, false, false>::ExtendConvFwdParams(context, params);

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
      const float min_freezed_output =
          context->input(6 + bias_index_offset).flat<float>()(0);
      const float max_freezed_output =
          context->input(7 + bias_index_offset).flat<float>()(0);

      float factor = std::is_same<Toutput, quint8>::value ? 255.0f : 127.0f;
      size_t depth = min_filter_vector.NumElements();
      const float* min_filter = min_filter_vector.flat<float>().data();
      const float* max_filter = max_filter_vector.flat<float>().data();
      std::vector<float> scales(depth);
      float input_range = std::max(std::abs(min_input), std::abs(max_input));
      float output_range =
          std::max(std::abs(min_freezed_output), std::abs(max_freezed_output));
      for (size_t i = 0; i < depth; i++) {
        float filter_range =
            std::max(std::abs(min_filter[i]), std::abs(max_filter[i]));
        scales[i] = factor * input_range * filter_range /
                    (255.0f * 127.0f * output_range);
      }
      params.post_op_params.push_back({"output_scale", scales});
    }
  }

  Tbias* GetBiasHandle(
      OpKernelContext* context,
      std::shared_ptr<mkldnn::convolution_forward::primitive_desc>& conv_fwd_pd,
      const Tensor& bias_tensor) override {
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

    std::vector<mkldnn::primitive> net;
    if (bias_enabled) {
      if (std::is_same<Tbias, qint32>::value) {
        return static_cast<Tbias*>(
            const_cast<Tbias*>(bias_tensor.flat<Tbias>().data()));
      }
      // If bias is enabled and requantization is not fused, scale the
      // bias to be consistent with quantized-input and quantized-filter.
      size_t depth = min_filter_vector.NumElements();
      std::vector<float> scales(depth);
      for (size_t i = 0; i < depth; i++) {
        scales[i] =
            255.0 * 127.0 /
            (std::max(std::abs(max_input), std::abs(min_input)) *
             std::max(std::abs(max_filter[i]), std::abs(min_filter[i])));
      }
      mkldnn::primitive_attr bias_attr;
      if (depth == 1) {
        bias_attr.set_output_scales(0, scales);
      } else {
        bias_attr.set_output_scales(1, scales);
      }
      auto bias_pd = memory::primitive_desc(
          {{bias_tensor.NumElements()}, MklDnnType<Tbias>(), memory::format::x},
          this->cpu_engine_);

      void* bias_buf = static_cast<void*>(
          const_cast<Tbias*>(bias_tensor.flat<Tbias>().data()));
      input_bias_ = new memory(bias_pd, bias_buf);
      scaled_bias_ = new memory(conv_fwd_pd->bias_primitive_desc());
      auto reorder_desc = mkldnn::reorder::primitive_desc(
          input_bias_->get_primitive_desc(), scaled_bias_->get_primitive_desc(),
          bias_attr);
      net.push_back(mkldnn::reorder(reorder_desc, *input_bias_, *scaled_bias_));
      stream(stream::kind::eager).submit(net).wait();
      return reinterpret_cast<Tbias*>(scaled_bias_->get_data_handle());
    } else {
      return nullptr;
    }
  }

  memory* input_bias_ = nullptr;
  memory* scaled_bias_ = nullptr;
};

template <typename Device, typename Tbias, typename Toutput,
          typename Ttemp_output, bool bias_enabled>
class MklQuantizedConv2DReluOp
    : public MklQuantizedConv2DOp<Device, Tbias, Toutput, Ttemp_output,
                                  bias_enabled> {
 public:
  virtual ~MklQuantizedConv2DReluOp() {}

  explicit MklQuantizedConv2DReluOp(OpKernelConstruction* context)
      : MklQuantizedConv2DOp<Device, Tbias, Toutput, Ttemp_output,
                             bias_enabled>(context) {}

 protected:
  void ExtendConvFwdParams(OpKernelContext* context,
                           MklConvFwdParams& params) override {
    MklQuantizedConv2DOp<Device, Tbias, Toutput, Ttemp_output,
                         bias_enabled>::ExtendConvFwdParams(context, params);
    params.post_op_params.push_back({"relu", {1.0, 0.0, 0.0}});
  }
};

template <typename Device, typename Tbias, typename Toutput,
          typename Ttemp_output, bool bias_enabled>
class MklQuantizedConv2DSumReluOp
    : public MklQuantizedConv2DOp<Device, Tbias, Toutput, Ttemp_output,
                                  bias_enabled> {
 public:
  virtual ~MklQuantizedConv2DSumReluOp() {
    if (this->summand_ != nullptr) {
      delete this->summand_;
      summand_ = nullptr;
    }

    if (this->dst_ != nullptr) {
      delete this->dst_;
      dst_ = nullptr;
    }
  }

  explicit MklQuantizedConv2DSumReluOp(OpKernelConstruction* context)
      : MklQuantizedConv2DOp<Device, Tbias, Toutput, Ttemp_output,
                             bias_enabled>(context) {}

 protected:
  void ExtendConvFwdParams(OpKernelContext* context,
                           MklConvFwdParams& params) override {
    MklQuantizedConv2DOp<Device, Tbias, Toutput, Ttemp_output,
                         bias_enabled>::ExtendConvFwdParams(context, params);
    // Calculate the scale (beta in mkldnn api term) for sum
    if (std::is_same<Toutput, quint8>::value) {
      int summand_idx = context->num_inputs() / 2 - 1 - 2;
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
      if (summand_type == DT_QUINT8)
        params.post_op_params.push_back(
            {"sum", {scale_summand / scale_output}});
      else
        params.post_op_params.push_back(
            {"sum", {255.0f * scale_summand / (scale_output * 127.0f)}});
    } else {
      params.post_op_params.push_back({"sum", {1.0}});
    }
    params.post_op_params.push_back({"relu", {1.0, 0.0, 0.0}});
  }

  // Allocate output tensor.
  void AllocateOutputTensor(
      OpKernelContext* context,
      const convolution_forward::primitive_desc& conv_prim_desc,
      const memory::dims& output_dims_mkl_order,
      memory::format output_tf_format, Tensor** output_tensor) override {
    int summand_idx = context->num_inputs() / 2 - 1;
    float reorder_sum_scale = 1.0;
    if (std::is_same<Toutput, quint8>::value) {
      summand_idx -= 2;
      DataType summand_type = this->input_type(summand_idx);
      bool summand_condition =
          (summand_type == DT_QINT8) || (summand_type == DT_QUINT8);
      CHECK((summand_condition));
      Tensor& summand = const_cast<Tensor&>(MklGetInput(context, summand_idx));
      MklDnnShape summand_mkl_shape;
      GetMklShape(context, summand_idx, &summand_mkl_shape);
      auto dst_md = summand_mkl_shape.GetMklLayout();
      if (summand_mkl_shape.IsMklTensor()) {
        if (summand_type == DT_QINT8) {
          summand.UnsafeCopyFromInternal(summand, DT_QUINT8, summand.shape());
          dst_md.data.data_type =
              static_cast<mkldnn_data_type_t>(MklDnnType<Toutput>());
          summand_mkl_shape.SetMklLayout(&dst_md);
          summand_mkl_shape.SetElemType(MklDnnType<Toutput>());
        }
        ForwardMklTensorInToOutWithMklShape(context, summand_idx, 0,
                                            summand_mkl_shape);
        *output_tensor = const_cast<Tensor*>(&summand);
        return;
      } else {
        TF_CHECK_OK(Status(error::Code::FAILED_PRECONDITION,
                           "Current fusion is not successful."));
      }
    }
    // TODO(mdfaijul): Add cleaner code for non-mkl tensor
    MklConvOp<Device, quint8, qint8, Tbias, Toutput, Ttemp_output, int32,
              bias_enabled, false,
              false>::AllocateOutputTensor(context, conv_prim_desc,
                                           output_dims_mkl_order,
                                           output_tf_format, output_tensor);
    const Tensor& summand = MklGetInput(context, summand_idx);
    if (summand.dtype() != DT_FLOAT)
      TF_CHECK_OK(Status(error::Code::FAILED_PRECONDITION,
                         "Current fusion requires summand to be float"));
    MklDnnShape summand_mkl_shape;
    GetMklShape(context, summand_idx, &summand_mkl_shape);
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

    size_t depth = min_filter_vector.NumElements();
    std::vector<float> scales(depth);
    for (size_t i = 0; i < depth; i++) {
      scales[i] = 255.0 * 127.0 /
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
                           memory::format::nhwc);
    auto summand_pd = memory::primitive_desc(summand_md, this->cpu_engine_);
    void* summand_buf =
        static_cast<void*>(const_cast<Tbias*>(summand.flat<Tbias>().data()));
    void* dst_buf =
        static_cast<void*>((*output_tensor)->flat<Ttemp_output>().data());
    summand_ = new memory(summand_pd, summand_buf);
    dst_ = new memory(conv_prim_desc.dst_primitive_desc(), dst_buf);
    auto reorder_desc = mkldnn::reorder::primitive_desc(
        summand_pd, conv_prim_desc.dst_primitive_desc(), reorder_attr);

    std::vector<mkldnn::primitive> net;
    net.push_back(mkldnn::reorder(reorder_desc, *summand_, *dst_));
    stream(stream::kind::eager).submit(net).wait();
  }

  memory* summand_ = nullptr;
  memory* dst_ = nullptr;
};

// INT8 kernel registration
// Register NoOp kernel for QunatizedConv2D for qint8 filter
REGISTER_KERNEL_BUILDER(Name("QuantizedConv2D")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<qint32>("out_type"),
                        NoOp);

REGISTER_KERNEL_BUILDER(Name("QuantizedConv2DAndRequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<qint8>("out_type"),
                        NoOp);

// Register NoOp kernel for QuantizedPerChannelConv2D
REGISTER_KERNEL_BUILDER(Name("QuantizedConv2DPerChannel")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<qint32>("out_type"),
                        NoOp);
// Register a templatized implementation of MklQuntizedConv2D.
REGISTER_KERNEL_BUILDER(
    Name("_MklQuantizedConv2DPerChannel")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<qint32>("out_type")
        .Label(mkl_op_registry::kMklQuantizedOpLabel),
    MklQuantizedConv2DOp<CPUDevice, float, qint32, qint32, false>);

// Register a templatized implementation of MklQuntizedConv2D.
REGISTER_KERNEL_BUILDER(
    Name("_MklQuantizedConv2D")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<qint32>("out_type")
        .Label(mkl_op_registry::kMklQuantizedOpLabel),
    MklQuantizedConv2DOp<CPUDevice, float, qint32, qint32, false>);

REGISTER_KERNEL_BUILDER(
    Name("_MklQuantizedConv2DAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<qint8>("out_type")
        .Label(mkl_op_registry::kMklQuantizedOpLabel),
    MklQuantizedConv2DOp<CPUDevice, qint32, qint8, qint8, false>);

// Register NoOp kernel for QuantizedConv2DWithBias to get a python interface.
// This kernel will be replaced by an MKL kernel during graph
// optimization pass.
REGISTER_KERNEL_BUILDER(Name("QuantizedConv2DWithBias")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<qint32>("out_type"),
                        NoOp);

REGISTER_KERNEL_BUILDER(Name("QuantizedConv2DWithBiasAndRequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<qint8>("out_type"),
                        NoOp);

// Register a templatized implementation MklQuantizedConv2DWithBias.
REGISTER_KERNEL_BUILDER(
    Name("_MklQuantizedConv2DWithBias")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<qint32>("out_type")
        .Label(mkl_op_registry::kMklQuantizedOpLabel),
    MklQuantizedConv2DOp<CPUDevice, float, qint32, qint32, true>);

REGISTER_KERNEL_BUILDER(
    Name("_MklQuantizedConv2DWithBiasAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<qint32>("Tbias")
        .TypeConstraint<qint8>("out_type")
        .Label(mkl_op_registry::kMklQuantizedOpLabel),
    MklQuantizedConv2DOp<CPUDevice, qint32, qint8, qint8, true>);
REGISTER_KERNEL_BUILDER(
    Name("_MklQuantizedConv2DWithBiasAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<float>("Tbias")
        .TypeConstraint<qint8>("out_type")
        .Label(mkl_op_registry::kMklQuantizedOpLabel),
    MklQuantizedConv2DOp<CPUDevice, float, qint8, qint8, true>);

// Register NoOp kernel for QuantizedConv2DAndRelu to get a python interface.
// This kernel will be replaced by an MKL kernel during graph-optimization pass.
REGISTER_KERNEL_BUILDER(Name("QuantizedConv2DAndRelu")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<qint32>("out_type"),
                        NoOp);

REGISTER_KERNEL_BUILDER(Name("QuantizedConv2DAndReluAndRequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<quint8>("out_type"),
                        NoOp);

// Register a templatized implementation of MklQuantizedConv2DAndRelu.
REGISTER_KERNEL_BUILDER(
    Name("_MklQuantizedConv2DAndRelu")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<qint32>("out_type")
        .Label(mkl_op_registry::kMklQuantizedOpLabel),
    MklQuantizedConv2DReluOp<CPUDevice, float, qint32, qint32, false>);

REGISTER_KERNEL_BUILDER(
    Name("_MklQuantizedConv2DAndReluAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<quint8>("out_type")
        .Label(mkl_op_registry::kMklQuantizedOpLabel),
    MklQuantizedConv2DReluOp<CPUDevice, qint32, quint8, quint8, false>);

// Register NoOp kernel for QuantizedConv2DWithBiasAndRelu to get a python
// interface.
// This kernel will be replaced by an MKL kernel during graph-optimization pass.
REGISTER_KERNEL_BUILDER(Name("QuantizedConv2DWithBiasAndRelu")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<qint32>("out_type"),
                        NoOp);

// Register NoOp kernel for QuantizedConv2DWithBiasAndReluAndRequantize
// to get a python interface.
// This kernel will be replaced by an MKL kernel during graph-optimization pass.
REGISTER_KERNEL_BUILDER(Name("QuantizedConv2DWithBiasAndReluAndRequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<quint8>("out_type"),
                        NoOp);

// Register a templatized implementation of MklQuantizedConv2DWithBiasAndRelu.
REGISTER_KERNEL_BUILDER(
    Name("_MklQuantizedConv2DWithBiasAndRelu")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<qint32>("out_type")
        .Label(mkl_op_registry::kMklQuantizedOpLabel),
    MklQuantizedConv2DReluOp<CPUDevice, float, qint32, qint32, true>);

// Register a templatized implementation of
// MklQuantizedConv2DWithBiasAndReluAndRequantize.
REGISTER_KERNEL_BUILDER(
    Name("_MklQuantizedConv2DWithBiasAndReluAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<float>("Tbias")
        .TypeConstraint<quint8>("out_type")
        .Label(mkl_op_registry::kMklQuantizedOpLabel),
    MklQuantizedConv2DReluOp<CPUDevice, float, quint8, quint8, true>);
REGISTER_KERNEL_BUILDER(
    Name("_MklQuantizedConv2DWithBiasAndReluAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<qint32>("Tbias")
        .TypeConstraint<quint8>("out_type")
        .Label(mkl_op_registry::kMklQuantizedOpLabel),
    MklQuantizedConv2DReluOp<CPUDevice, qint32, quint8, quint8, true>);

// Register NoOp kernel for QuantizedConv2DWithBiasSumAndRelu to get a python
// interface.
// This kernel will be replaced by an MKL kernel during graph-optimization pass.
REGISTER_KERNEL_BUILDER(Name("QuantizedConv2DWithBiasSumAndRelu")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<qint32>("out_type"),
                        NoOp);

REGISTER_KERNEL_BUILDER(Name("QuantizedConv2DWithBiasSumAndReluAndRequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<quint8>("out_type"),
                        NoOp);
REGISTER_KERNEL_BUILDER(
    Name("QuantizedConv2DWithBiasSignedSumAndReluAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<quint8>("out_type"),
    NoOp);
// Register a templatized implementation of MklQuantizedConv2DWithBiasAndRelu.
REGISTER_KERNEL_BUILDER(
    Name("_MklQuantizedConv2DWithBiasSumAndRelu")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<qint32>("out_type")
        .Label(mkl_op_registry::kMklQuantizedOpLabel),
    MklQuantizedConv2DSumReluOp<CPUDevice, float, qint32, qint32, true>);

REGISTER_KERNEL_BUILDER(
    Name("_MklQuantizedConv2DWithBiasSumAndReluAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<qint32>("Tbias")
        .TypeConstraint<quint8>("out_type")
        .Label(mkl_op_registry::kMklQuantizedOpLabel),
    MklQuantizedConv2DSumReluOp<CPUDevice, qint32, quint8, quint8, true>);
REGISTER_KERNEL_BUILDER(
    Name("_MklQuantizedConv2DWithBiasSignedSumAndReluAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<qint32>("Tbias")
        .TypeConstraint<quint8>("out_type")
        .Label(mkl_op_registry::kMklQuantizedOpLabel),
    MklQuantizedConv2DSumReluOp<CPUDevice, qint32, quint8, qint8, true>);

REGISTER_KERNEL_BUILDER(
    Name("_MklQuantizedConv2DWithBiasSumAndReluAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<float>("Tbias")
        .TypeConstraint<quint8>("out_type")
        .Label(mkl_op_registry::kMklQuantizedOpLabel),
    MklQuantizedConv2DSumReluOp<CPUDevice, float, quint8, quint8, true>);
REGISTER_KERNEL_BUILDER(
    Name("_MklQuantizedConv2DWithBiasSignedSumAndReluAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<float>("Tbias")
        .TypeConstraint<quint8>("out_type")
        .Label(mkl_op_registry::kMklQuantizedOpLabel),
    MklQuantizedConv2DSumReluOp<CPUDevice, float, quint8, qint8, true>);
#endif  // INTEL_MKL_ML

// Register 2D operations
#define REGISTER_MKL_CPU_2D(T)                                             \
  REGISTER_KERNEL_BUILDER(Name("_MklConv2D")                               \
                              .Device(DEVICE_CPU)                          \
                              .TypeConstraint<T>("T")                      \
                              .Label(mkl_op_registry::kMklOpLabel),        \
                          MklConvOp<CPUDevice, float, float, float, float, \
                                    float, int32, false, false, false>);   \
  REGISTER_KERNEL_BUILDER(Name("_MklConv2DWithBias")                       \
                              .Device(DEVICE_CPU)                          \
                              .TypeConstraint<T>("T")                      \
                              .Label(mkl_op_registry::kMklOpLabel),        \
                          MklConvOp<CPUDevice, float, float, float, float, \
                                    float, int32, true, false, false>);    \
  REGISTER_KERNEL_BUILDER(Name("__MklDummyConv2DWithBias")                 \
                              .Device(DEVICE_CPU)                          \
                              .TypeConstraint<T>("T")                      \
                              .Label(mkl_op_registry::kMklOpLabel),        \
                          MklDummyOp<CPUDevice, T>);                       \
  REGISTER_KERNEL_BUILDER(Name("_MklPadWithConv2D")                        \
                              .Device(DEVICE_CPU)                          \
                              .TypeConstraint<T>("T")                      \
                              .TypeConstraint<int32>("Tpaddings")          \
                              .Label(mkl_op_registry::kMklOpLabel),        \
                          MklConvOp<CPUDevice, float, float, float, float, \
                                    float, int32, false, true, false>);    \
  REGISTER_KERNEL_BUILDER(Name("_MklPadWithConv2D")                        \
                              .Device(DEVICE_CPU)                          \
                              .TypeConstraint<T>("T")                      \
                              .TypeConstraint<int64>("Tpaddings")          \
                              .Label(mkl_op_registry::kMklOpLabel),        \
                          MklConvOp<CPUDevice, float, float, float, float, \
                                    float, int64, false, true, false>);    \
  REGISTER_KERNEL_BUILDER(Name("__MklDummyPadWithConv2D")                  \
                              .Device(DEVICE_CPU)                          \
                              .TypeConstraint<T>("T")                      \
                              .TypeConstraint<int32>("Tpaddings")          \
                              .Label(mkl_op_registry::kMklOpLabel),        \
                          MklDummyOp<CPUDevice, T>);

TF_CALL_float(REGISTER_MKL_CPU_2D);

#define REGISTER_MKL_CPU_2D_DEPTHWISE(T)                                   \
  REGISTER_KERNEL_BUILDER(Name("_MklDepthwiseConv2dNative")                \
                              .Device(DEVICE_CPU)                          \
                              .TypeConstraint<float>("T")                  \
                              .Label(mkl_op_registry::kMklOpLabel),        \
                          MklConvOp<CPUDevice, float, float, float, float, \
                                    float, int32, false, false, true>);

TF_CALL_float(REGISTER_MKL_CPU_2D_DEPTHWISE);

#define REGISTER_MKL_CPU_2D_FUSED(T)                                \
  REGISTER_KERNEL_BUILDER(Name("_MklFusedConv2D")                   \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<T>("T")               \
                              .Label(mkl_op_registry::kMklOpLabel), \
                          MklFusedConvOp<CPUDevice, T, T, T, T, T>);
// We check the fused_ops attributes to decide if bias is enabled or not.

TF_CALL_float(REGISTER_MKL_CPU_2D_FUSED);

// Register 3D operations
#define REGISTER_MKL_CPU_3D(T)                  \
  REGISTER_KERNEL_BUILDER(                      \
      Name("_MklConv3D")                        \
          .Device(DEVICE_CPU)                   \
          .TypeConstraint<T>("T")               \
          .Label(mkl_op_registry::kMklOpLabel), \
      MklConvOp<CPUDevice, T, T, T, T, T, int32, false, false, false>);
TF_CALL_float(REGISTER_MKL_CPU_3D);

}  // namespace tensorflow
#endif  // INTEL_MKL
