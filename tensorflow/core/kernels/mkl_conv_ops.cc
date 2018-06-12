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
#include <map>
#include <memory>
#include <string>
#include <typeinfo>
#include <vector>
#include <algorithm>

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
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

#include "tensorflow/core/util/mkl_util.h"

#ifndef INTEL_MKL_ML
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

#ifndef INTEL_MKL_ML

struct ConvFwdParams {
  memory::dims src_dims;
  memory::dims filter_dims;
  memory::dims bias_dims;
  memory::dims dst_dims;
  memory::dims strides;
  memory::dims dilations;
  memory::dims padding_left;
  memory::dims padding_right;

  std::string dtypes = std::string("");
  std::vector<std::string> post_ops = std::vector<std::string>();

  ConvFwdParams(memory::dims src_dims, memory::dims filter_dims,
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

template <typename T, typename Tinput, typename Tfilter, typename Tbias,
          typename Toutput>
class Conv2DFwd : public DnnOp {
 public:
  explicit Conv2DFwd(const ConvFwdParams& conv_fwd_params) {
    fwd_stream_.reset(new stream(stream::kind::eager));
    // create conv primitive
    if (conv_fwd_ == nullptr) {
      Setup(conv_fwd_params);
    }
  }

  ~Conv2DFwd() {}

  // Convolution forward execute with bias
  //   src_data:    input data buffer of src
  //   filter_data: input data buffer of filter (weights)
  //   bias_data:   input data buffer of bias
  //   dst_data:    output data buffer of dst
  void Execute(Tinput* src_data, Tfilter* filter_data, Tbias* bias_data,
               Toutput* dst_data) {
    src_mem_->set_data_handle(static_cast<void*>(src_data));
    filter_mem_->set_data_handle(static_cast<void*>(filter_data));
    bias_mem_->set_data_handle(static_cast<void*>(bias_data));
    dst_mem_->set_data_handle(static_cast<void*>(dst_data));
    fwd_stream_->submit(fwd_primitives_);

    // after exec, set data handle back
    src_mem_->set_data_handle(DummyData);
    filter_mem_->set_data_handle(DummyData);
    bias_mem_->set_data_handle(DummyData);
    dst_mem_->set_data_handle(DummyData);

    return;
  }

  // Convolution forward execute without bias
  //   src_data:    input data buffer of src
  //   filter_data: input data buffer of filter (weights)
  //   dst_data:    output data buffer of dst
  void Execute(Tinput* src_data, Tfilter* filter_data, Toutput* dst_data) {
    src_mem_->set_data_handle(static_cast<void*>(src_data));
    filter_mem_->set_data_handle(static_cast<void*>(filter_data));
    dst_mem_->set_data_handle(static_cast<void*>(dst_data));
    fwd_stream_->submit(fwd_primitives_);

    // after exec, set data handle back
    src_mem_->set_data_handle(DummyData);
    filter_mem_->set_data_handle(DummyData);
    dst_mem_->set_data_handle(DummyData);

    return;
  }

  // expected memory format for this primitive instance
  memory::format src_fmt_;
  memory::format filter_fmt_;

  // convolution primitive
  std::shared_ptr<mkldnn::convolution_forward::primitive_desc> fwd_pd_;
  std::shared_ptr<mkldnn::primitive> conv_fwd_;

 private:
  void Setup(const ConvFwdParams& conv_fwd_params) {
    // create memory descriptors for convolution data w/ no specified format
    src_md_.reset(new memory::desc({conv_fwd_params.src_dims},
                                   MklDnnType<Tinput>(), memory::format::any));

    filter_md_.reset(new memory::desc({conv_fwd_params.filter_dims},
                                      MklDnnType<Tfilter>(),
                                      memory::format::any));

    dst_md_.reset(new memory::desc({conv_fwd_params.dst_dims},
                                   MklDnnType<Toutput>(), memory::format::any));

    if (!conv_fwd_params.bias_dims.empty())
      bias_md_.reset(new memory::desc({conv_fwd_params.bias_dims},
                                      MklDnnType<Tbias>(),
                                      memory::format::any));

    // create a convolution
    if (!conv_fwd_params.bias_dims.empty()) {
      fwd_desc_.reset(new convolution_forward::desc(
          prop_kind::forward, convolution_direct, *src_md_, *filter_md_,
          *bias_md_, *dst_md_, conv_fwd_params.strides,
          conv_fwd_params.dilations, conv_fwd_params.padding_left,
          conv_fwd_params.padding_right, padding_kind::zero));
    } else {
      fwd_desc_.reset(new convolution_forward::desc(
          prop_kind::forward, convolution_direct, *src_md_, *filter_md_,
          *dst_md_, conv_fwd_params.strides, conv_fwd_params.dilations,
          conv_fwd_params.padding_left, conv_fwd_params.padding_right,
          padding_kind::zero));
    }

    // Check if there is any fusions as post-ops
    if (!conv_fwd_params.post_ops.empty()) {
      // Add ReLU as post op
      if (conv_fwd_params.post_ops.at(0) == "relu") {
        mkldnn::post_ops ops;
        mkldnn::primitive_attr post_ops_attr;
        const float ops_scale = 1.f;
        const float ops_alpha = 0.f;  // relu negative slope
        const float ops_beta = 0.f;
        ops.append_eltwise(ops_scale, mkldnn::eltwise_relu, ops_alpha,
                           ops_beta);
        post_ops_attr.set_post_ops(ops);

        fwd_pd_.reset(new convolution_forward::primitive_desc(
            *fwd_desc_, post_ops_attr, cpu_engine_));
      } else {
        fwd_pd_.reset(
            new convolution_forward::primitive_desc(*fwd_desc_, cpu_engine_));
      }
    } else {
      fwd_pd_.reset(
          new convolution_forward::primitive_desc(*fwd_desc_, cpu_engine_));
    }

    // store the expected memory format
    src_fmt_ = static_cast<mkldnn::memory::format>(
        fwd_pd_.get()->src_primitive_desc().desc().data.format);

    filter_fmt_ = static_cast<mkldnn::memory::format>(
        fwd_pd_.get()->weights_primitive_desc().desc().data.format);

    // create memory primitive based on dummy data
    src_mem_.reset(new memory(fwd_pd_.get()->src_primitive_desc(), DummyData));
    filter_mem_.reset(
        new memory(fwd_pd_.get()->weights_primitive_desc(), DummyData));
    dst_mem_.reset(new memory(fwd_pd_.get()->dst_primitive_desc(), DummyData));

    // create convolution primitive and add it to net
    if (!conv_fwd_params.bias_dims.empty()) {
      bias_mem_.reset(new memory(
          {{{conv_fwd_params.bias_dims}, MklDnnType<T>(), memory::format::x},
           cpu_engine_},
          DummyData));
      conv_fwd_.reset(new convolution_forward(*fwd_pd_, *src_mem_, *filter_mem_,
                                              *bias_mem_, *dst_mem_));
    } else {
      conv_fwd_.reset(new convolution_forward(*fwd_pd_, *src_mem_, *filter_mem_,
                                              *dst_mem_));
    }

    fwd_primitives_.push_back(*conv_fwd_);
    return;
  }

  // MKLDNN memory
  std::shared_ptr<mkldnn::memory> src_mem_;
  std::shared_ptr<mkldnn::memory> filter_mem_;
  std::shared_ptr<mkldnn::memory> bias_mem_;
  std::shared_ptr<mkldnn::memory> dst_mem_;

  std::shared_ptr<mkldnn::stream> fwd_stream_;
  std::vector<mkldnn::primitive> fwd_primitives_;

  // desc & prmitive desc
  std::shared_ptr<mkldnn::convolution_forward::desc> fwd_desc_;

  // memory desc
  std::shared_ptr<mkldnn::memory::desc> src_md_;
  std::shared_ptr<mkldnn::memory::desc> filter_md_;
  std::shared_ptr<mkldnn::memory::desc> bias_md_;
  std::shared_ptr<mkldnn::memory::desc> dst_md_;

  engine cpu_engine_ = engine(engine::cpu, 0);
};

template <typename T, typename Tinput, typename Tfilter, typename Tbias,
          typename Toutput>
class Conv2DFwdFactory : public DnnOpFactory<T> {
 public:
  static Conv2DFwd<T, Tinput, Tfilter, Tbias, Toutput>* Get(
      const ConvFwdParams& conv_fwd_params) {
    Conv2DFwd<T, Tinput, Tfilter, Tbias, Toutput>* conv2d_fwd = nullptr;

    // try to find a suitable one in pool
    conv2d_fwd = dynamic_cast<Conv2DFwd<T, Tinput, Tfilter, Tbias, Toutput>*>(
        Conv2DFwdFactory<T, Tinput, Tfilter, Tbias, Toutput>::GetInstance()
            .GetConv2DFwd(conv_fwd_params));

    if (conv2d_fwd == nullptr) {
      conv2d_fwd =
          new Conv2DFwd<T, Tinput, Tfilter, Tbias, Toutput>(conv_fwd_params);
      Conv2DFwdFactory<T, Tinput, Tfilter, Tbias, Toutput>::GetInstance()
          .SetConv2DFwd(conv_fwd_params, conv2d_fwd);
    }
    return conv2d_fwd;
  }

 private:
  Conv2DFwdFactory() {}
  ~Conv2DFwdFactory() {}

  static const int kDilationH = 0, kDilationW = 1;

  static Conv2DFwdFactory& GetInstance() {
    static Conv2DFwdFactory instance_;
    return instance_;
  }

  static std::string CreateKey(const ConvFwdParams& conv_fwd_params) {
    std::string prefix = "conv2d_fwd_";
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(prefix);
    key_creator.AddAsKey(conv_fwd_params.src_dims);
    key_creator.AddAsKey(conv_fwd_params.filter_dims);
    key_creator.AddAsKey(conv_fwd_params.bias_dims);
    key_creator.AddAsKey(conv_fwd_params.dst_dims);
    key_creator.AddAsKey(conv_fwd_params.strides);
    key_creator.AddAsKey(conv_fwd_params.dilations);
    key_creator.AddAsKey(conv_fwd_params.padding_left);
    key_creator.AddAsKey(conv_fwd_params.padding_right);
    key_creator.AddAsKey(conv_fwd_params.dtypes);
    for (auto const& post_op : conv_fwd_params.post_ops)
      key_creator.AddAsKey(post_op);

    return key_creator.GetKey();
  }

  DnnOp* GetConv2DFwd(const ConvFwdParams& conv_fwd_params) {
    std::string key = CreateKey(conv_fwd_params);
    return this->GetOp(key);
  }

  void SetConv2DFwd(const ConvFwdParams& conv_fwd_params, DnnOp* op) {
    std::string key = CreateKey(conv_fwd_params);
    this->SetOp(key, op);
  }
};

#endif

typedef Eigen::ThreadPoolDevice CPUDevice;

// For now, MKL-ML is default. So making MKL-DNN not a default choice.
#ifdef INTEL_MKL_ML
template <typename Device, typename T, bool biasEnabled>
class MklConv2DOp : public OpKernel {
 public:
  ~MklConv2DOp() {}

  explicit MklConv2DOp(OpKernelConstruction* context) : OpKernel(context) {
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

    if (biasEnabled) {
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

    if (biasEnabled) {
      const Tensor& bias = MklGetInput(context, 2);
      mkl_context.bias_sizes[0] = {static_cast<size_t>(bias.dim_size(0))};
      mkl_context.bias_strides[0] = {1};
    }

    // Create Convolution Primitive
    if (biasEnabled) {
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

      if (biasEnabled) {
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

      if (biasEnabled) {
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
      if (biasEnabled) dnnLayoutDelete_F32(lt_bias);
    }
  } MklConv2DOpContext;

  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;
};

#else

template <typename Device, typename Tinput, typename Tfilter, typename Tbias,
          typename Toutput, bool biasEnabled>
class MklConv2DOp : public OpKernel {
 public:
  ~MklConv2DOp() {}

  explicit MklConv2DOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations_));
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
      MklDnnData<Toutput> dst(&cpu_engine_);  // output

      memory::dims src_dims, filter_dims, padding_left, padding_right,
          dilations, strides;
      memory::dims dst_dims_tf_order, dst_dims_mkl_order;

      // Get shapes of input tensors in MKL-DNN order
      MklDnnConvUtil conv_utl(context, strides_, padding_, data_format_,
                              dilations_);
      auto src_tf_shape = GetTfShape(context, kInputIndex_Src);
      auto filter_tf_shape = GetTfShape(context, kInputIndex_Filter);
      conv_utl.GetConvFwdSizesInMklOrder(
          src_tf_shape, filter_tf_shape, &src_dims, &filter_dims, &strides,
          &dilations, &dst_dims_tf_order, &dst_dims_mkl_order, &padding_left,
          &padding_right);
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

        Tensor* output_filter_tensor = nullptr;
        // MklConv2D also outputs converted filter as 2nd output of Conv2D.
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

      // Create memory for user data.
      // Describe how the inputs and outputs of Convolution look like. Also
      // specify buffers containing actual input and output data.
      auto tf_fmt = TFDataFormatToMklDnnDataFormat(data_format_);

      // If input is in MKL layout, then simply grab input layout; otherwise,
      // construct input Tf layout. For TF layout, although input shape
      // (src_dims) required is in MKL-DNN order, the layout is Tensorflow's
      // layout (NHWC or NCHW depending on data format).
      auto src_md = src_mkl_shape.IsMklTensor()
                        ? src_mkl_shape.GetMklLayout()
                        : memory::desc(src_dims, MklDnnType<Tinput>(), tf_fmt);
      src.SetUsrMem(src_md, &src_tensor);

      // Although filter shape (filter_dims) required is in MKL-DNN order,
      // the layout is Tensorflow's layout (HWIO).
      auto filter_md = filter_mkl_shape.IsMklTensor()  // Should NEVER be true
                           ? filter_mkl_shape.GetMklLayout()
                           : memory::desc(filter_dims, MklDnnType<Tfilter>(),
                                          memory::format::hwio);
      filter.SetUsrMem(filter_md, &filter_tensor);

      // MKLDNN dilation starts from 0.
      dilations[kDilationH] -= 1;
      dilations[kDilationW] -= 1;

      // get a conv2d fwd from primitive pool
      Conv2DFwd<float, Tinput, Tfilter, Tbias, Toutput>* conv2d_fwd = nullptr;
      if (biasEnabled) {
        memory::dims bias_dims = {};
        conv_utl.GetBiasSizeInMklOrder(kInputIndex_Bias, &bias_dims);
        ConvFwdParams conv_fwd_params(src_dims, filter_dims, bias_dims,
                                      dst_dims_mkl_order, strides, dilations,
                                      padding_left, padding_right);

        // TODO(mdfaijul):  Extend the basic parameters for data types and
        // fusions
        this->ExtendConvFwdParams(conv_fwd_params);

        conv2d_fwd =
            Conv2DFwdFactory<float, Tinput, Tfilter, Tbias, Toutput>::Get(
                conv_fwd_params);
      } else {
        ConvFwdParams conv_fwd_params(src_dims, filter_dims, NONE_DIMS,
                                      dst_dims_mkl_order, strides, dilations,
                                      padding_left, padding_right);

        // Extend the basic parameters for data types and fusions
        this->ExtendConvFwdParams(conv_fwd_params);

        conv2d_fwd =
            Conv2DFwdFactory<float, Tinput, Tfilter, Tbias, Toutput>::Get(
                conv_fwd_params);
      }

      // allocate output tensors output_tensor and filter_out_tensor
      std::shared_ptr<mkldnn::convolution_forward::primitive_desc> conv_fwd_pd =
          conv2d_fwd->fwd_pd_;
      AllocateOutputTensor(context, *conv_fwd_pd, dst_dims_mkl_order, tf_fmt,
                           &dst_tensor);

      Tensor* filter_out_tensor = nullptr;
      if (typeid(Tinput) == typeid(float) && typeid(Tfilter) == typeid(float) &&
          typeid(Toutput) == typeid(float)) {
        AllocateFilterOutputTensor(context, *conv_fwd_pd,
                                   TFShapeToMklDnnDims(filter_tf_shape),
                                   &filter_out_tensor);
      }

      Toutput* dst_data =
          static_cast<Toutput*>(dst_tensor->flat<Toutput>().data());

      // check whether src/filter need reorder
      std::vector<primitive> net;
      if (src_md.data.format != conv2d_fwd->src_fmt_)
        src.CheckReorderToOpMem(conv_fwd_pd.get()->src_primitive_desc(), &net);

      if (filter_md.data.format != conv2d_fwd->filter_fmt_) {
        if (filter_out_tensor == nullptr) {
          filter.CheckReorderToOpMem(
              conv_fwd_pd.get()->weights_primitive_desc(), &net);
        } else {
          filter.CheckReorderToOpMem(
              conv_fwd_pd.get()->weights_primitive_desc(),
              filter.GetTensorBuffer(filter_out_tensor), &net);
        }
      }

      stream(stream::kind::eager).submit(net).wait();

      Tinput* src_data = static_cast<Tinput*>(src.GetOpMem().get_data_handle());
      Tfilter* filter_data =
          static_cast<Tfilter*>(filter.GetOpMem().get_data_handle());

      // execute convolution
      if (biasEnabled) {
        const Tensor& bias_tensor = MklGetInput(context, kInputIndex_Bias);
        Tbias* bias_data =
            this->GetBiasHandle(context, conv_fwd_pd, bias_tensor);
        conv2d_fwd->Execute(src_data, filter_data, bias_data, dst_data);
      } else {
        conv2d_fwd->Execute(src_data, filter_data, dst_data);
      }
    } catch (mkldnn::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) + ", message: " +
                         std::string(e.message) + ", in file " +
                         std::string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }

  virtual void ExtendConvFwdParams(ConvFwdParams& params) {
    // Create a string from data types of input, filter, bias, and output.
    params.dtypes.append(typeid(Tinput).name());
    params.dtypes.append(typeid(Tfilter).name());
    params.dtypes.append(typeid(Tbias).name());
    params.dtypes.append(typeid(Toutput).name());
  }

  virtual Tbias* GetBiasHandle(
      OpKernelContext* context,
      std::shared_ptr<mkldnn::convolution_forward::primitive_desc>&
          conv2d_fwd_pd,
      const Tensor& bias_tensor) {
    if (biasEnabled) {
      Tbias* bias_data = static_cast<Tbias*>(
          const_cast<Tbias*>(bias_tensor.flat<Tbias>().data()));
      return bias_data;
    } else {
      return nullptr;
    }
  }

 private:
  std::vector<int32> strides_;
  std::vector<int32> dilations_;
  Padding padding_;
  TensorFormat data_format_;
  const int kInputIndex_Src = 0, kInputIndex_Filter = 1, kInputIndex_Bias = 2;
  const int kOutputIndex_Dst = 0, kOutputIndex_Filter = 1;
  const int kDilationH = 0, kDilationW = 1;
  engine cpu_engine_ = engine(engine::cpu, 0);

  // Allocate output tensor.
  void AllocateOutputTensor(
      OpKernelContext* context,
      const convolution_forward::primitive_desc& conv_prim_desc,
      const memory::dims& output_dims_mkl_order,
      memory::format output_tf_format, Tensor** output_tensor) {
    CHECK_NOTNULL(output_tensor);
    auto dst_pd = conv_prim_desc.dst_primitive_desc();

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

  // Allocate output tensor.
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
    std::vector<primitive> net;
    src->CheckReorderToOpMem(conv_prim_desc.src_primitive_desc(), &net);

    // rather than re-order to a temp buffer, reorder directly to the
    // filter output tensor
    filter->CheckReorderToOpMem(conv_prim_desc.weights_primitive_desc(),
                                filter->GetTensorBuffer(filter_out_tensor),
                                &net);

    // Create convolution primitive and add it to net.
    if (bias) {
      CHECK_EQ(biasEnabled, true);
      net.push_back(convolution_forward(conv_prim_desc, src->GetOpMem(),
                                        filter->GetOpMem(), bias->GetOpMem(),
                                        output->GetOpMem()));
    } else {
      CHECK_EQ(biasEnabled, false);
      net.push_back(convolution_forward(conv_prim_desc, src->GetOpMem(),
                                        filter->GetOpMem(),
                                        output->GetOpMem()));
    }

    stream(stream::kind::eager).submit(net).wait();
  }
};

template <typename Device, bool biasEnabled>
class MklQuantizedConv2DOp
    : public MklConv2DOp<Device, quint8, qint8, float, qint32, biasEnabled> {
 public:
  virtual ~MklQuantizedConv2DOp() {
    if (this->input_bias_ != nullptr) {
      delete this->input_bias_;
    }

    if (this->scaled_bias_ != nullptr) {
      delete this->scaled_bias_;
    }
  }

  explicit MklQuantizedConv2DOp(OpKernelConstruction* context)
      : MklConv2DOp<Device, quint8, qint8, float, qint32, biasEnabled>(
            context) {}

  void Compute(OpKernelContext* context) override {
    // Compute int32 output tensor
    MklConv2DOp<Device, quint8, qint8, float, qint32, biasEnabled>::Compute(
        context);

    // Compute additional outputs fp32 min/max sclars.
    int bias_index_offset;
    bias_index_offset = biasEnabled ? 1 : 0;

    const float min_input =
        context->input(2 + bias_index_offset).flat<float>()(0);
    const float max_input =
        context->input(3 + bias_index_offset).flat<float>()(0);
    const float min_filter =
        context->input(4 + bias_index_offset).flat<float>()(0);
    const float max_filter =
        context->input(5 + bias_index_offset).flat<float>()(0);

    float min_output_value;
    float max_output_value;
    MklQuantizationRangeForMultiplication<quint8, qint8, qint32>(
        min_input, max_input, min_filter, max_filter, &min_output_value,
        &max_output_value);
    Tensor* output_min = nullptr;
    Tensor* output_max = nullptr;
    MklDnnShape output_min_mkl_shape, output_max_mkl_shape;
    output_min_mkl_shape.SetMklTensor(false);
    output_max_mkl_shape.SetMklTensor(false);
    AllocateOutputSetMklShape(context, 1, &output_min, {},
                              output_min_mkl_shape);
    AllocateOutputSetMklShape(context, 2, &output_max, {},
                              output_max_mkl_shape);
    output_min->flat<float>()(0) = min_output_value;
    output_max->flat<float>()(0) = max_output_value;
  }

  void ExtendConvFwdParams(ConvFwdParams& params) override {
    MklConv2DOp<Device, quint8, qint8, float, qint32,
                biasEnabled>::ExtendConvFwdParams(params);
  }

  float* GetBiasHandle(
      OpKernelContext* context,
      std::shared_ptr<mkldnn::convolution_forward::primitive_desc>& conv_fwd_pd,
      const Tensor& bias_tensor) override {
    int bias_index_offset;
    bias_index_offset = biasEnabled ? 1 : 0;

    const float min_input =
        context->input(2 + bias_index_offset).flat<float>()(0);
    const float max_input =
        context->input(3 + bias_index_offset).flat<float>()(0);
    const float min_filter =
        context->input(4 + bias_index_offset).flat<float>()(0);
    const float max_filter =
        context->input(5 + bias_index_offset).flat<float>()(0);

    std::vector<mkldnn::primitive> net;
    // If bias is enabled, scale the bias to be consistent with quantized-input
    // and quantized-filter.
    if (biasEnabled) {
      float bias_scale = 255.0 * 127.0 /
                         (std::max(std::abs(max_input), std::abs(min_input)) *
                          std::max(std::abs(max_filter), std::abs(min_filter)));
      std::vector<float> scales;
      scales.push_back(bias_scale);
      mkldnn::primitive_attr bias_attr;
      bias_attr.set_output_scales(0, scales);

      void* bias_buf = static_cast<void*>(
          const_cast<float*>(bias_tensor.flat<float>().data()));
      input_bias_ = new memory(conv_fwd_pd->bias_primitive_desc(), bias_buf);
      scaled_bias_ = new memory(conv_fwd_pd->bias_primitive_desc());
      auto reorder_desc = mkldnn::reorder::primitive_desc(
          input_bias_->get_primitive_desc(), scaled_bias_->get_primitive_desc(),
          bias_attr);
      net.push_back(mkldnn::reorder(reorder_desc, *input_bias_, *scaled_bias_));
      stream(stream::kind::eager).submit(net).wait();
      return reinterpret_cast<float*>(scaled_bias_->get_data_handle());
    } else {
      return nullptr;
    }
  }

 private:
  memory* input_bias_ = nullptr;
  memory* scaled_bias_ = nullptr;
};

template <typename Device, bool biasEnabled>
class MklQuantizedConv2DReluOp
    : public MklQuantizedConv2DOp<Device, biasEnabled> {
 public:
  virtual ~MklQuantizedConv2DReluOp() {}

  explicit MklQuantizedConv2DReluOp(OpKernelConstruction* context)
      : MklQuantizedConv2DOp<Device, biasEnabled>(context) {}

  void ExtendConvFwdParams(ConvFwdParams& params) override {
    MklQuantizedConv2DOp<Device, biasEnabled>::ExtendConvFwdParams(params);
    params.post_ops.push_back("relu");
  }
};

// Register NoOp kernel for QunatizedConv2D since none is registered for
// filter data type of qint8.
REGISTER_KERNEL_BUILDER(Name("QuantizedConv2D")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<qint32>("out_type"),
                        NoOp);

// Register a templatized implementation of MklQuntizedConv2D.
REGISTER_KERNEL_BUILDER(Name("_MklQuantizedConv2D")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<qint32>("out_type")
                            .Label(mkl_op_registry::kMklQuantizedOpLabel),
                        MklQuantizedConv2DOp<CPUDevice, false>);

// Register NoOp kernel for QuantizedConv2DWithBias to get a python interface.
// This kernel will be replaced by an MKL kernel during graph
// optimization pass.
REGISTER_KERNEL_BUILDER(Name("QuantizedConv2DWithBias")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<qint32>("out_type"),
                        NoOp);

// Register a templatized implementation MklQuantizedConv2DWithBias.
REGISTER_KERNEL_BUILDER(Name("_MklQuantizedConv2DWithBias")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<qint32>("out_type")
                            .Label(mkl_op_registry::kMklQuantizedOpLabel),
                        MklQuantizedConv2DOp<CPUDevice, true>);

// Register NoOp kernel for QuantizedConv2DAndRelu to get a python interface.
// This kernel will be replaced by an MKL kernel during graph-optimization pass.
REGISTER_KERNEL_BUILDER(Name("QuantizedConv2DAndRelu")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<qint32>("out_type"),
                        NoOp);

// Register a templatized implementation of MklQuantizedConv2DAndRelu.
REGISTER_KERNEL_BUILDER(Name("_MklQuantizedConv2DAndRelu")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<qint32>("out_type")
                            .Label(mkl_op_registry::kMklQuantizedOpLabel),
                        MklQuantizedConv2DReluOp<CPUDevice, false>);

// Register NoOp kernel for QuantizedConv2DWithBiasAndRelu to get a python
// interface.
// This kernel will be replaced by an MKL kernel during graph-optimization pass.
REGISTER_KERNEL_BUILDER(Name("QuantizedConv2DWithBiasAndRelu")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<qint32>("out_type"),
                        NoOp);

// Register a templatized implementation of MklQuantizedConv2DWithBiasAndRelu.
REGISTER_KERNEL_BUILDER(Name("_MklQuantizedConv2DWithBiasAndRelu")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<qint32>("out_type")
                            .Label(mkl_op_registry::kMklQuantizedOpLabel),
                        MklQuantizedConv2DReluOp<CPUDevice, true>);

#endif  // INTEL_MKL_ML

#define REGISTER_MKL_CPU(T)                                           \
  REGISTER_KERNEL_BUILDER(Name("_MklConv2D")                          \
                              .Device(DEVICE_CPU)                     \
                              .TypeConstraint<T>("T")                 \
                              .Label(mkl_op_registry::kMklOpLabel),   \
                          MklConv2DOp<CPUDevice, T, T, T, T, false>); \
  REGISTER_KERNEL_BUILDER(Name("_MklConv2DWithBias")                  \
                              .Device(DEVICE_CPU)                     \
                              .TypeConstraint<T>("T")                 \
                              .Label(mkl_op_registry::kMklOpLabel),   \
                          MklConv2DOp<CPUDevice, T, T, T, T, true>);  \
  REGISTER_KERNEL_BUILDER(Name("__MklDummyConv2DWithBias")            \
                              .Device(DEVICE_CPU)                     \
                              .TypeConstraint<T>("T")                 \
                              .Label(mkl_op_registry::kMklOpLabel),   \
                          MklDummyOp<CPUDevice, T>);

TF_CALL_float(REGISTER_MKL_CPU);

}  // namespace tensorflow
#endif  // INTEL_MKL
