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

#include <algorithm>
#include <vector>

#include "mkldnn.hpp"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/conv_grad_ops.h"
#include "tensorflow/core/kernels/mkl_conv_ops.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/mkl_types.h"
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"
#include "tensorflow/core/util/work_sharder.h"

using mkldnn::convolution_backward_weights;
using mkldnn::memory;
using mkldnn::prop_kind;
using mkldnn::stream;

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

using ConvBwdFilterDesc = mkldnn::convolution_backward_weights::desc;
using ConvBwdFilterPd = mkldnn::convolution_backward_weights::primitive_desc;

struct MklConvBwdFilterParams {
  memory::dims src_dims;
  memory::dims diff_filter_dims;
  memory::dims diff_bias_dims;
  memory::dims diff_dst_dims;
  memory::dims strides;
  memory::dims dilations;
  memory::dims padding_left;
  memory::dims padding_right;
#ifndef ENABLE_MKLDNN_V1
  padding_kind padding;
#endif  // !ENABLE_MKLDNN_V1

  MklConvBwdFilterParams(memory::dims src_dims, memory::dims diff_filter_dims,
                         memory::dims diff_bias_dims,
                         memory::dims diff_dst_dims, memory::dims strides,
                         memory::dims dilations, memory::dims padding_left,
#ifndef ENABLE_MKLDNN_V1
                         memory::dims padding_right, padding_kind padding)
#else
                         memory::dims padding_right)
#endif  // !ENABLE_MKLDNN_V1
      : src_dims(src_dims),
        diff_filter_dims(diff_filter_dims),
        diff_bias_dims(diff_bias_dims),
        diff_dst_dims(diff_dst_dims),
        strides(strides),
        dilations(dilations),
        padding_left(padding_left),
#ifndef ENABLE_MKLDNN_V1
        padding_right(padding_right),
        padding(padding) {
  }
#else
        padding_right(padding_right) {
  }
#endif  // !ENABLE_MKLDNN_V1
};

template <typename T>
class MklConvBwdFilterPrimitive : public MklPrimitive {
 public:
  explicit MklConvBwdFilterPrimitive(
      const MklConvBwdFilterParams& convBwdFilterDims)
      : cpu_engine_(ENGINE_CPU, 0) {
    context_.bwd_filter_stream.reset(new CPU_STREAM(cpu_engine_));

    // Create convolution backward filter primitive.
    if (context_.conv_bwd_filter == nullptr) {
      Setup(convBwdFilterDims);
    }
  }

  ~MklConvBwdFilterPrimitive() {}

  // Convolution backward weights execution with bias
  //   src_data:         input data buffer for src
  //   diff_filter_data: output data buffer for diff_filter
  //   diff_bias_data:   output data buffer for diff_bias
  //   diff_dst_data:    input data buffer for diff_dst
  void Execute(const T* src_data, const T* diff_filter_data,
               const T* diff_bias_data, const T* diff_dst_data) {
    context_.src_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(src_data)));
    context_.diff_filter_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(diff_filter_data)));
    if (diff_bias_data != nullptr) {
      context_.diff_bias_mem->set_data_handle(
          static_cast<void*>(const_cast<T*>(diff_bias_data)));
    }
    context_.diff_dst_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(diff_dst_data)));

#ifdef ENABLE_MKLDNN_V1
    execute_primitives(context_.bwd_filter_primitives,
                       context_.bwd_filter_stream,
                       context_.bwd_filter_primitives_args);
#else
    context_.bwd_filter_stream->submit(context_.bwd_filter_primitives);
#endif

    context_.src_mem->set_data_handle(DummyData);
    context_.diff_filter_mem->set_data_handle(DummyData);
    if (diff_bias_data != nullptr) {
      context_.diff_bias_mem->set_data_handle(DummyData);
    }
    context_.diff_dst_mem->set_data_handle(DummyData);
  }

  // Convolution backward weights without bias.
  //   src_data:         input data buffer of src
  //   diff_filter_data: output data buffer of diff_filter
  //   diff_dst_data:    input data buffer of diff_dst
  void Execute(const T* src_data, const T* diff_filter_data,
               const T* diff_dst_data) {
    Execute(src_data, diff_filter_data, nullptr, diff_dst_data);
  }

#ifndef ENABLE_MKLDNN_V1
  memory::format GetSrcMemoryFormat() const { return context_.src_fmt; }
  memory::format GetDiffDstMemoryFormat() const {
    return context_.diff_dst_fmt;
  }
  memory::format GetDiffFilterMemoryFormat() const {
    return context_.diff_filter_fmt;
  }
#endif

  std::shared_ptr<ConvBwdFilterPd> GetPrimitiveDesc() const {
    return context_.bwd_filter_pd;
  }

 private:
  // Primitive reuse context for Conv2D backward filter op.
  struct ConvBwdFilterContext {
#ifndef ENABLE_MKLDNN_V1
    // Expected memory format for this primitive instance
    memory::format src_fmt;
    memory::format diff_dst_fmt;
    memory::format diff_filter_fmt;
#endif  // !ENABLE_MKLDNN_V1

    // MKL-DNN memory for inputs and outputs.
    std::shared_ptr<mkldnn::memory> src_mem;
    std::shared_ptr<mkldnn::memory> diff_filter_mem;
    std::shared_ptr<mkldnn::memory> diff_bias_mem;
    std::shared_ptr<mkldnn::memory> diff_dst_mem;

    // Primitive descriptor and descriptor for convolution backward filter.
    std::shared_ptr<ConvBwdFilterPd> bwd_filter_pd;
    std::shared_ptr<ConvBwdFilterDesc> bwd_filter_desc;

    // Primitive descriptor and descriptor for convolution forward.
    std::shared_ptr<ConvFwdPd> fwd_pd;
    std::shared_ptr<ConvFwdDesc> fwd_desc;

    // Convolution backward filter primitive.
    std::shared_ptr<mkldnn::primitive> conv_bwd_filter;

    // Memory descriptors: forward & backward share the same memory descriptors
    std::shared_ptr<mkldnn::memory::desc> src_md;
    std::shared_ptr<mkldnn::memory::desc> diff_filter_md;
    std::shared_ptr<mkldnn::memory::desc> diff_bias_md;
    std::shared_ptr<mkldnn::memory::desc> diff_dst_md;

    // MKL-DNN pipeline for executing primitives.
    std::shared_ptr<mkldnn::stream> bwd_filter_stream;
    std::vector<mkldnn::primitive> bwd_filter_primitives;

#ifdef ENABLE_MKLDNN_V1
    std::vector<MemoryArgsMap> bwd_filter_primitives_args;
#endif

    ConvBwdFilterContext()
        :
#ifndef ENABLE_MKLDNN_V1
          src_fmt(memory::format::any),
          diff_dst_fmt(memory::format::any),
          diff_filter_fmt(memory::format::any),
#endif
          src_mem(nullptr),
          diff_filter_mem(nullptr),
          diff_bias_mem(nullptr),
          diff_dst_mem(nullptr),
          bwd_filter_desc(nullptr),
          fwd_desc(nullptr),
          fwd_pd(nullptr),
          src_md(nullptr),
          diff_filter_md(nullptr),
          diff_bias_md(nullptr),
          diff_dst_md(nullptr),
          bwd_filter_stream(nullptr) {
    }
  };

  void Setup(const MklConvBwdFilterParams& convBwdFilterDims) {
    // Create memory descriptors for convolution backward filter without any
    // specific format so that MKL-DNN can pick an appropriate one depending
    // on the input parameters.
    context_.src_md.reset(new memory::desc(
        {convBwdFilterDims.src_dims}, MklDnnType<T>(), MEMORY_FORMAT::any));

    context_.diff_dst_md.reset(
        new memory::desc({convBwdFilterDims.diff_dst_dims}, MklDnnType<T>(),
                         MEMORY_FORMAT::any));

    context_.diff_filter_md.reset(
        new memory::desc({convBwdFilterDims.diff_filter_dims}, MklDnnType<T>(),
                         MEMORY_FORMAT::any));

    if (!convBwdFilterDims.diff_bias_dims.empty())
      context_.diff_bias_md.reset(
          new memory::desc({convBwdFilterDims.diff_bias_dims}, MklDnnType<T>(),
                           MEMORY_FORMAT::x));

    // Create descriptor and primitive descriptor for convolution forward.
    context_.fwd_desc.reset(new ConvFwdDesc(
        prop_kind::forward, ALGORITHM::convolution_direct, *context_.src_md,
        *context_.diff_filter_md, *context_.diff_dst_md,
        convBwdFilterDims.strides, convBwdFilterDims.dilations,
#ifndef ENABLE_MKLDNN_V1
        convBwdFilterDims.padding_left, convBwdFilterDims.padding_right,
        convBwdFilterDims.padding));
#else
        convBwdFilterDims.padding_left, convBwdFilterDims.padding_right));
#endif  // !ENABLE_MKLDNN_V1
    context_.fwd_pd.reset(new ConvFwdPd(*context_.fwd_desc, cpu_engine_));

    // Create descriptor and primitive descriptor for convolution bwd filter.
    if (!convBwdFilterDims.diff_bias_dims.empty()) {
      context_.bwd_filter_desc.reset(new ConvBwdFilterDesc(
          ALGORITHM::convolution_direct, *context_.src_md,
          *context_.diff_filter_md, *context_.diff_bias_md,
          *context_.diff_dst_md, convBwdFilterDims.strides,
          convBwdFilterDims.dilations, convBwdFilterDims.padding_left,
#ifndef ENABLE_MKLDNN_V1
          convBwdFilterDims.padding_right, convBwdFilterDims.padding));
#else
          convBwdFilterDims.padding_right));
#endif  // !ENABLE_MKLDNN_V1
    } else {
      context_.bwd_filter_desc.reset(new ConvBwdFilterDesc(
          ALGORITHM::convolution_direct, *context_.src_md,
          *context_.diff_filter_md, *context_.diff_dst_md,
          convBwdFilterDims.strides, convBwdFilterDims.dilations,
#ifndef ENABLE_MKLDNN_V1
          convBwdFilterDims.padding_left, convBwdFilterDims.padding_right,
          convBwdFilterDims.padding));
#else
          convBwdFilterDims.padding_left, convBwdFilterDims.padding_right));
#endif  // !ENABLE_MKLDNN_V1
    }
    context_.bwd_filter_pd.reset(new ConvBwdFilterPd(
        *context_.bwd_filter_desc, cpu_engine_, *context_.fwd_pd));

    auto bwd_filter_pd = context_.bwd_filter_pd.get();

#ifndef ENABLE_MKLDNN_V1
    // Store the expected memory format.
    context_.src_fmt = static_cast<mkldnn::memory::format>(
        bwd_filter_pd->src_primitive_desc().desc().data.format);
    context_.diff_filter_fmt = static_cast<mkldnn::memory::format>(
        bwd_filter_pd->diff_weights_primitive_desc().desc().data.format);
    context_.diff_dst_fmt = static_cast<mkldnn::memory::format>(
        bwd_filter_pd->diff_dst_primitive_desc().desc().data.format);
#endif  // !ENABLE_MKLDNN_V1

    // Create memory using dummy data.
    context_.src_mem.reset(new MEMORY_CONSTRUCTOR(
        bwd_filter_pd->PRIMITIVE_DESC_SRC, cpu_engine_, DummyData));
    context_.diff_filter_mem.reset(new MEMORY_CONSTRUCTOR(
        bwd_filter_pd->PRIMITIVE_DESC_DIFF_WEIGHTS, cpu_engine_, DummyData));
    context_.diff_dst_mem.reset(new MEMORY_CONSTRUCTOR(
        bwd_filter_pd->PRIMITIVE_DESC_DIFF_DST, cpu_engine_, DummyData));

    // Create convolution backward filter primitive and add it to the net.
    if (!convBwdFilterDims.diff_bias_dims.empty()) {
      context_.diff_bias_mem.reset(new MEMORY_CONSTRUCTOR_USING_MEM_PD(
          convBwdFilterDims.diff_bias_dims, T, MEMORY_FORMAT::x, cpu_engine_,
          DummyData));
#ifdef ENABLE_MKLDNN_V1
      context_.conv_bwd_filter.reset(
          new convolution_backward_weights(*context_.bwd_filter_pd));
      context_.bwd_filter_primitives_args.push_back(
          {{MKLDNN_ARG_SRC, *context_.src_mem},
           {MKLDNN_ARG_DIFF_WEIGHTS, *context_.diff_filter_mem},
           {MKLDNN_ARG_DIFF_BIAS, *context_.diff_bias_mem},
           { MKLDNN_ARG_DIFF_DST,
             *context_.diff_dst_mem }});
    } else {
      context_.conv_bwd_filter.reset(
          new convolution_backward_weights(*context_.bwd_filter_pd));
      context_.bwd_filter_primitives_args.push_back(
          {{MKLDNN_ARG_SRC, *context_.src_mem},
           {MKLDNN_ARG_DIFF_WEIGHTS, *context_.diff_filter_mem},
           { MKLDNN_ARG_DIFF_DST,
             *context_.diff_dst_mem }});
    }
#else
      context_.conv_bwd_filter.reset(new convolution_backward_weights(
          *context_.bwd_filter_pd, *context_.src_mem, *context_.diff_dst_mem,
          *context_.diff_filter_mem, *context_.diff_bias_mem));
    } else {
      context_.conv_bwd_filter.reset(new convolution_backward_weights(
          *context_.bwd_filter_pd, *context_.src_mem, *context_.diff_dst_mem,
          *context_.diff_filter_mem));
    }
#endif  // ENABLE_MKLDNN_V1
    context_.bwd_filter_primitives.push_back(*context_.conv_bwd_filter);
  }

  struct ConvBwdFilterContext context_;
  engine cpu_engine_;
};

template <typename T>
class MklConvBwdFilterPrimitiveFactory : public MklPrimitiveFactory<T> {
 public:
  static MklConvBwdFilterPrimitive<T>* Get(
      const MklConvBwdFilterParams& convBwdFilterDims, bool do_not_cache) {
    MklConvBwdFilterPrimitive<T>* conv_bwd_filter = nullptr;

    if (do_not_cache) { /* Create new primitive always */
      conv_bwd_filter = new MklConvBwdFilterPrimitive<T>(convBwdFilterDims);
    } else {
      // Look into the pool for reusable primitive.
      conv_bwd_filter = dynamic_cast<MklConvBwdFilterPrimitive<T>*>(
          MklConvBwdFilterPrimitiveFactory<T>::GetInstance().GetConvBwdFilter(
              convBwdFilterDims));

      if (conv_bwd_filter == nullptr) {
        conv_bwd_filter = new MklConvBwdFilterPrimitive<T>(convBwdFilterDims);
        MklConvBwdFilterPrimitiveFactory<T>::GetInstance().SetConvBwdFilter(
            convBwdFilterDims, conv_bwd_filter);
      }
    }

    return conv_bwd_filter;
  }

 private:
  MklConvBwdFilterPrimitiveFactory() {}
  ~MklConvBwdFilterPrimitiveFactory() {}

  static MklConvBwdFilterPrimitiveFactory& GetInstance() {
    static MklConvBwdFilterPrimitiveFactory instance_;
    return instance_;
  }

  static string CreateKey(const MklConvBwdFilterParams& convBwdFilterDims) {
    string prefix = "conv_bwd_filter";
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(prefix);
    key_creator.AddAsKey(convBwdFilterDims.src_dims);
    key_creator.AddAsKey(convBwdFilterDims.diff_filter_dims);
    key_creator.AddAsKey(convBwdFilterDims.diff_bias_dims);
    key_creator.AddAsKey(convBwdFilterDims.diff_dst_dims);
    key_creator.AddAsKey(convBwdFilterDims.strides);
    key_creator.AddAsKey(convBwdFilterDims.dilations);
    key_creator.AddAsKey(convBwdFilterDims.padding_left);
    key_creator.AddAsKey(convBwdFilterDims.padding_right);
    return key_creator.GetKey();
  }

  MklPrimitive* GetConvBwdFilter(
      const MklConvBwdFilterParams& convBwdFilterDims) {
    string key = CreateKey(convBwdFilterDims);
    return this->GetOp(key);
  }

  void SetConvBwdFilter(const MklConvBwdFilterParams& convBwdFilterDims,
                        MklPrimitive* op) {
    string key = CreateKey(convBwdFilterDims);
    this->SetOp(key, op);
  }
};

template <typename Device, class T, bool bias_enabled, bool is_depthwise,
          bool eager_mode>
class MklConvCustomBackpropFilterOp
    : public MklConvBackpropCommonOp<Device, T, is_depthwise> {
 public:
  explicit MklConvCustomBackpropFilterOp(OpKernelConstruction* context)
      : MklConvBackpropCommonOp<Device, T, is_depthwise>(context) {}

  ~MklConvCustomBackpropFilterOp() {}

  void Compute(OpKernelContext* context) {
    try {
      // Input tensors.
      const Tensor& src_tensor = MklGetInput(context, kInputIdx);
      const Tensor& filter_tensor = MklGetInput(context, kFilterIdx);
      const Tensor& diff_dst_tensor = MklGetInput(context, kDiffDstIdx);

      MklDnnShape src_mkl_shape, filter_mkl_shape, diff_dst_mkl_shape;
      GetMklShape(context, kInputIdx, &src_mkl_shape, eager_mode);
      GetMklShape(context, kFilterIdx, &filter_mkl_shape, eager_mode);
      GetMklShape(context, kDiffDstIdx, &diff_dst_mkl_shape, eager_mode);
      // Allow operator-specific sanity checking of shapes.
      ValidateMklShapes(src_mkl_shape, filter_mkl_shape, diff_dst_mkl_shape);

      // Allow operator-specific generation of shapes.
      // E.g., Conv2DBackpropFilter gets filter as filter_sizes. It is a
      // tensor containing shape of filter. So filter.shape() is not
      // a correct way to get filter shape. These operator-specific calls
      // allow this class to handle this case.
      TensorShape src_tf_shape = MakeInputTfShape(context, src_tensor);
      TensorShape filter_tf_shape = MakeFilterTfShape(context, filter_tensor);
      TensorShape diff_dst_tf_shape =
          GetTfShape(context, kDiffDstIdx, eager_mode);

      // Corner cases: output with 0 elements and 0 batch size.
      Tensor* diff_filter_tensor = nullptr;
      if (src_tf_shape.num_elements() == 0 ||
          filter_tf_shape.num_elements() == 0 ||
          diff_dst_tf_shape.num_elements() == 0) {
        MklDnnShape diff_filter_mkl_shape;
        diff_filter_mkl_shape.SetMklTensor(false);
        TensorShape diff_filter_tf_shape =
            GetOutputTfShape(src_tf_shape, filter_tf_shape, diff_dst_tf_shape);
        const int kOutputIdx = 0;
        AllocateOutputSetMklShape(context, kOutputIdx, &diff_filter_tensor,
                                  diff_filter_tf_shape, diff_filter_mkl_shape,
                                  eager_mode);
        DCHECK(diff_filter_tensor != nullptr);

        // If output tensor has more than 0 elements, we need to 0 them out.
        auto diff_filter_data = diff_filter_tensor->flat<T>().data();
        for (size_t i = 0; i < diff_filter_tf_shape.num_elements(); ++i) {
          diff_filter_data[i] = static_cast<T>(0);
        }
        return;
      }

      // By default, all dims are in MKL order except those that are suffixed
      // with `tf_order`
      memory::dims diff_dst_dims, fwd_src_dims, fwd_filter_dims;
      memory::dims padding_left, padding_right, dilations, strides;
      memory::dims fwd_dst_dims, fwd_dst_dims_tf_order;

      // Get forward convolution parameters.
      MklDnnConvUtil conv_util(context, this->strides_, this->padding_,
                               this->data_format_, this->dilations_);
      conv_util.GetConvFwdSizesInMklOrder(
          src_tf_shape, filter_tf_shape, &fwd_src_dims, &fwd_filter_dims,
          &strides, &dilations, &fwd_dst_dims_tf_order, &fwd_dst_dims,
          &padding_left, &padding_right, false, is_depthwise);
      if (!context->status().ok()) return;

      bool is_conv2d = (this->strides_.size() == 4);

      auto tf_fmt = is_conv2d
                        ? TFDataFormatToMklDnnDataFormat(this->data_format_)
                        : TFDataFormatToMklDnn3DDataFormat(this->data_format_);
#ifdef ENABLE_MKLDNN_V1
      auto mkl_fmt_tag = MklTensorFormatToMklDnnDataFormat(tf_fmt);
      OP_REQUIRES(context, mkl_fmt_tag != memory::format_tag::undef,
                  errors::InvalidArgument("Invalid data format"));
#endif

      auto fwd_src_md =
          src_mkl_shape.IsMklTensor()
              ? src_mkl_shape.GetMklLayout()
              : memory::desc(fwd_src_dims, MklDnnType<T>(), MKL_FMT_TAG);

      conv_util.GetInputSizeInMklOrder(diff_dst_tf_shape, &diff_dst_dims);
      if (!context->status().ok()) return;

      auto diff_dst_md =
          diff_dst_mkl_shape.IsMklTensor()
              ? diff_dst_mkl_shape.GetMklLayout()
              : memory::desc(diff_dst_dims, MklDnnType<T>(), MKL_FMT_TAG);

      memory::dims diff_bias_dims = {};
      int64 depth = 0;
      if (bias_enabled) {
        TensorShape obp_tf_shape = GetTfShape(context, 2);
        depth = (this->data_format_ == FORMAT_NCHW)
                    ? obp_tf_shape.dim_size(1)
                    : obp_tf_shape.dim_size(is_conv2d ? 3 : 4);
        diff_bias_dims = {static_cast<int>(depth)};
      }

      // The default dilation factor for each dimension is 1 in TF and
      // 0 in MKL-DNN.
      for (int i = 0; i < dilations.size(); ++i) --dilations[i];
      MklConvBwdFilterParams convBwdFilterDims(
          fwd_src_dims, fwd_filter_dims, diff_bias_dims, diff_dst_dims, strides,
#ifndef ENABLE_MKLDNN_V1
          dilations, padding_left, padding_right,
          TFPaddingToMklDnnPadding(this->padding_));
#else
          dilations, padding_left, padding_right);
#endif  // !ENABLE_MKLDNN_V1

      // MKL-DNN allocates large buffers when a conv gradient filter primitive
      // is created. So we don't cache conv backward primitives when the env
      // variable TF_MKL_OPTIMIZE_PRIMITIVE_MEMUSE is set to true.
      bool do_not_cache = MklPrimitiveFactory<T>::IsPrimitiveMemOptEnabled();

      MklConvBwdFilterPrimitive<T>* conv_bwd_filter =
          MklConvBwdFilterPrimitiveFactory<T>::Get(convBwdFilterDims,
                                                   do_not_cache);

      // Allocate output tensors: diff_filter and diff_bias (w bias).
      auto diff_filter_dims = GetOutputDims(fwd_src_dims, fwd_filter_dims);

      MklDnnShape diff_filter_mkl_shape;
      diff_filter_mkl_shape.SetMklTensor(false);

      if (is_conv2d) {
        if (!is_depthwise) {
          // Conv2D: output_dims_mkl_order is in OIHW format.
          TensorShape diff_filter_tf_shape(
              {diff_filter_dims[MklDnnDims::Dim_H],
               diff_filter_dims[MklDnnDims::Dim_W],
               diff_filter_dims[MklDnnDims::Dim_I],
               diff_filter_dims[MklDnnDims::Dim_O]});
          AllocateOutputSetMklShape(context, 0, &diff_filter_tensor,
                                    diff_filter_tf_shape, diff_filter_mkl_shape,
                                    eager_mode);
        } else {
          // Depthwise Conv2d: diff_filter_dims is GOIHW format.
          //                  | TensorFlow       | MKLDNN
          // ----------------------------------------------------------------
          // filter_out_depth | depth_multiplier | depth_multiplier *
          //                  |                  | group_count
          // ----------------------------------------------------------------
          // filter_in_depth  | in_depth         | in_depth / group_count
          // For depthwise convolution, we have group_count == in_depth.
          // So here G = original I, and I = 1.
          // And the GOIHW is mkldnn format, here we try to extract the TF
          // format, TF format is HWIO, as G = original I, so here is HWGO.
          TensorShape diff_filter_tf_shape(
              {diff_filter_dims[MklDnnFilterGroupDims::MKL_GROUP_FILTER_DIM_H],
               diff_filter_dims[MklDnnFilterGroupDims::MKL_GROUP_FILTER_DIM_W],
               diff_filter_dims[MklDnnFilterGroupDims::MKL_GROUP_FILTER_DIM_G],
               diff_filter_dims
                   [MklDnnFilterGroupDims::MKL_GROUP_FILTER_DIM_O]});
          AllocateOutputSetMklShape(context, 0, &diff_filter_tensor,
                                    diff_filter_tf_shape,
                                    diff_filter_mkl_shape);
        }
      } else {
        // Conv3D: output_dims_mkl_order is in OIDHW format.
        TensorShape diff_filter_tf_shape(
            {diff_filter_dims[MklDnnDims3D::Dim3d_D],
             diff_filter_dims[MklDnnDims3D::Dim3d_H],
             diff_filter_dims[MklDnnDims3D::Dim3d_W],
             diff_filter_dims[MklDnnDims3D::Dim3d_I],
             diff_filter_dims[MklDnnDims3D::Dim3d_O]});
        AllocateOutputSetMklShape(context, 0, &diff_filter_tensor,
                                  diff_filter_tf_shape, diff_filter_mkl_shape);
      }

      Tensor* diff_bias_tensor = nullptr;
      if (bias_enabled) {
        TensorShape diff_bias_shape({depth});
        AllocateBiasGradTensor(context, diff_bias_shape, &diff_bias_tensor);
      }

      // Check if src and diff_dst need to be reordered.
      T* src_data = nullptr;
      MklDnnData<T> src(&cpu_engine_);
      auto bwd_filter_pd = conv_bwd_filter->GetPrimitiveDesc();
      if (IS_SRC_REORDER_NEEDED(fwd_src_md, bwd_filter_pd, conv_bwd_filter)) {
        src.SetUsrMem(fwd_src_md, &src_tensor);
        src.CheckReorderToOpMem(MEMORY_PD_WITHOUT_DATA(
            bwd_filter_pd->PRIMITIVE_DESC_SRC, cpu_engine_));
        src_data = static_cast<T*>(src.GetOpMem().get_data_handle());
      } else {
        src_data = static_cast<T*>(const_cast<T*>(src_tensor.flat<T>().data()));
      }

      T* diff_dst_data = nullptr;
      MklDnnData<T> diff_dst(&cpu_engine_);
      if (IS_DIFF_DST_REORDER_NEEDED(diff_dst_md, bwd_filter_pd,
                                     conv_bwd_filter)) {
        diff_dst.SetUsrMem(diff_dst_md, &diff_dst_tensor);
        diff_dst.CheckReorderToOpMem(MEMORY_PD_WITHOUT_DATA(
            bwd_filter_pd->PRIMITIVE_DESC_DIFF_DST, cpu_engine_));
        diff_dst_data = static_cast<T*>(diff_dst.GetOpMem().get_data_handle());
      } else {
        diff_dst_data =
            static_cast<T*>(const_cast<T*>(diff_dst_tensor.flat<T>().data()));
      }

      DCHECK(!diff_filter_mkl_shape.IsMklTensor());
      auto diff_filter_format = GetOutputFormat(MKL_FMT_TAG);
      auto diff_filter_md =
          memory::desc(diff_filter_dims, MklDnnType<T>(), diff_filter_format);

      // Convert diff_filter (output) back to TF layout if needed
      // (i.e. reorder op memory back to user memory)
      MklDnnData<T> diff_filter(&cpu_engine_);
      bool diff_filter_reorder_required = false;
      T* diff_filter_data = nullptr;
      if (IS_DIFF_FILTER_REORDER_NEEDED(diff_filter_md, diff_filter_format,
                                        bwd_filter_pd, conv_bwd_filter)) {
        // Allocate diff_filter tensor as Tensorflow layout.
        diff_filter.SetUsrMem(diff_filter_dims, diff_filter_format,
                              diff_filter_tensor);
        diff_filter_reorder_required = true;
        diff_filter.PrepareReorderToUserMemIfReq(
            bwd_filter_pd->PRIMITIVE_DESC_DIFF_WEIGHTS);
        diff_filter_data =
            static_cast<T*>(diff_filter.GetOpMem().get_data_handle());
      } else {
        diff_filter_data = static_cast<T*>(
            const_cast<T*>(diff_filter_tensor->flat<T>().data()));
      }

      // Execute convolution backward filter.
      if (bias_enabled) {
        T* diff_bias_data =
            static_cast<T*>(const_cast<T*>(diff_bias_tensor->flat<T>().data()));
        conv_bwd_filter->Execute(src_data, diff_filter_data, diff_bias_data,
                                 diff_dst_data);
      } else {
        conv_bwd_filter->Execute(src_data, diff_filter_data, diff_dst_data);
      }

      // Reorder diff_filter back to Tensorflow layout if necessary.
      if (diff_filter_reorder_required) {
        diff_filter.InsertReorderToUserMem();
      }

      // Delete primitive since it is not cached.
      if (do_not_cache) delete conv_bwd_filter;
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
  const int kInputIdx = 0, kFilterIdx = 1, kDiffDstIdx = 2;
  const int kDilationH = 0, kDilationW = 1;

  engine cpu_engine_ = engine(ENGINE_CPU, 0);

  // Assert that input shapes are valid.
  void ValidateMklShapes(const MklDnnShape& input_mkl_shape,
                         const MklDnnShape& filter_mkl_shape,
                         const MklDnnShape& obp_mkl_shape) {
    CHECK(!filter_mkl_shape.IsMklTensor())
        << "ConvBackpropFilter: filter should not be in MKL Layout";
  }

  // Get TensorFlow shape of input tensor.
  TensorShape MakeInputTfShape(OpKernelContext* context,
                               const Tensor& input_tensor) {
    size_t input_idx = 0;
    return GetTfShape(context, input_idx, eager_mode);
  }

  // Get TensorFlow shape of filter tensor.
  TensorShape MakeFilterTfShape(OpKernelContext* context,
                                const Tensor& filter_tensor) {
    TensorShape filter_tf_shape;
    CHECK_EQ(TensorShapeUtils::IsVector(filter_tensor.shape()), true);
    CHECK_EQ(TensorShapeUtils::MakeShape(filter_tensor.vec<int32>(),
                                         &filter_tf_shape)
                 .ok(),
             true);
    return filter_tf_shape;
  }

  // Get Tensorflow shape of output tensor (diff_filter),
  // which is same as shape of filter.
  TensorShape GetOutputTfShape(const TensorShape& input_shape,
                               const TensorShape& filter_shape,
                               const TensorShape& outbprop_shape) {
    return filter_shape;
  }

  // Get the shape of output (diff_filter) in MKL-DNN order.
  // Computes shape of output from input shape (fwd_input_dims)
  // and filter shape (fwd_filter_dims).
  const memory::dims& GetOutputDims(const memory::dims& fwd_input_dims,
                                    const memory::dims& fwd_filter_dims) {
    return fwd_filter_dims;
  }

  // Output layout is Tensorflow's filter layout
  //   Conv2D: HWIO;  Conv3D: DHWIO; Depthwise Conv: HWIGO
  MEMORY_FORMAT GetOutputFormat(const MEMORY_FORMAT data_format) {
    return is_depthwise ? MEMORY_FORMAT::hwigo
                        : ((this->strides_.size() == 4) ? MEMORY_FORMAT::hwio
                                                        : MEMORY_FORMAT::dhwio);
  }

  void AllocateOutputTensor(OpKernelContext* context,
                            const memory::dims& output_dims_mkl_order,
                            Tensor** output_tensor) {
    DCHECK(output_tensor != nullptr);

    // For BackpropFilter, we convert the output tensor back in Tensorflow
    // layout. Because typically, BackpropFilter is the last operator in the
    // graph that emit filter gradient that is provided to ApplyGradient
    // method to update the filter. But it may be possible to eliminate this
    // by forwarding filter in MKL layout if we support ApplyGradient method
    // for MKL layout propagation.
    MklDnnShape output_mkl_shape;
    output_mkl_shape.SetMklTensor(false);
    // output_dims_mkl_order is in OIHW format.
    // Allocate shape of TF tensor in HWIO format.
    TensorShape output_tf_shape({output_dims_mkl_order[MklDnnDims::Dim_H],
                                 output_dims_mkl_order[MklDnnDims::Dim_W],
                                 output_dims_mkl_order[MklDnnDims::Dim_I],
                                 output_dims_mkl_order[MklDnnDims::Dim_O]});
    AllocateOutputSetMklShape(context, 0, output_tensor, output_tf_shape,
                              output_mkl_shape);
  }

  void AllocateBiasGradTensor(OpKernelContext* context,
                              const TensorShape& bias_grad_shape,
                              Tensor** bias_grad_tensor) {
    DCHECK(bias_grad_tensor);

    MklDnnShape bias_grad_mkl_shape;
    bias_grad_mkl_shape.SetMklTensor(false);
    AllocateOutputSetMklShape(context, 1, bias_grad_tensor, bias_grad_shape,
                              bias_grad_mkl_shape);
  }
};

#define REGISTER_MKL_FILTER_KERNELS(T)                                   \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("_MklConv2DBackpropFilter")                                   \
          .Device(DEVICE_CPU)                                            \
          .TypeConstraint<T>("T")                                        \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),           \
      MklConvCustomBackpropFilterOp<CPUDevice, T, false, false, false>); \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("_MklEagerConv2DBackpropFilter")                              \
          .Device(DEVICE_CPU)                                            \
          .TypeConstraint<T>("T")                                        \
          .Label(mkl_op_registry::kMklNameChangeOpLabel),                \
      MklConvCustomBackpropFilterOp<CPUDevice, T, false, false, true>);  \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("_MklConv2DBackpropFilterWithBias")                           \
          .Device(DEVICE_CPU)                                            \
          .TypeConstraint<T>("T")                                        \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),           \
      MklConvCustomBackpropFilterOp<CPUDevice, T, true, false, false>);  \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("_MklDepthwiseConv2dNativeBackpropFilter")                    \
          .Device(DEVICE_CPU)                                            \
          .TypeConstraint<T>("T")                                        \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),           \
      MklConvCustomBackpropFilterOp<CPUDevice, T, false, true, false>);  \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("__MklDummyConv2DBackpropFilterWithBias")                     \
          .Device(DEVICE_CPU)                                            \
          .TypeConstraint<T>("T")                                        \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),           \
      MklDummyOp<CPUDevice, T>);                                         \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("_MklConv3DBackpropFilterV2")                                 \
          .Device(DEVICE_CPU)                                            \
          .TypeConstraint<T>("T")                                        \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),           \
      MklConvCustomBackpropFilterOp<CPUDevice, T, false, false, false>);

TF_CALL_float(REGISTER_MKL_FILTER_KERNELS);
TF_CALL_bfloat16(REGISTER_MKL_FILTER_KERNELS);

#undef REGISTER_MKL_FILTER_KERNELS

}  // namespace tensorflow

#endif  // INTEL_MKL
