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

struct MklConvBwdFilterParams {
  memory::dims src_dims;
  memory::dims diff_filter_dims;
  memory::dims diff_bias_dims;
  memory::dims diff_dst_dims;
  memory::dims strides;
  memory::dims dilations;
  memory::dims padding_left;
  memory::dims padding_right;
  padding_kind padding;

  MklConvBwdFilterParams(memory::dims src_dims, memory::dims diff_filter_dims,
                         memory::dims diff_bias_dims,
                         memory::dims diff_dst_dims, memory::dims strides,
                         memory::dims dilations, memory::dims padding_left,
                         memory::dims padding_right, padding_kind padding)
      : src_dims(src_dims),
        diff_filter_dims(diff_filter_dims),
        diff_bias_dims(diff_bias_dims),
        diff_dst_dims(diff_dst_dims),
        strides(strides),
        dilations(dilations),
        padding_left(padding_left),
        padding_right(padding_right),
        padding(padding) {}
};

template <typename T>
class MklConvBwdFilterPrimitive : public MklPrimitive {
 public:
  explicit MklConvBwdFilterPrimitive(
      const MklConvBwdFilterParams& convBwdFilterDims)
      : cpu_engine_(engine::cpu, 0) {
    context_.bwd_filter_stream.reset(new stream(stream::kind::eager));
    // create conv primitive
    if (context_.conv_bwd_filter == nullptr) {
      Setup(convBwdFilterDims);
    }
  }

  ~MklConvBwdFilterPrimitive() {}

  // Convolution backward weights with bias
  //   src_data:         input data buffer of src
  //   diff_filter_data: output data buffer of diff_filter
  //   diff_bias_data:   output data buffer of diff_bias
  //   diff_dst_data:    input data buffer of diff_dst
  void Execute(const T* src_data, const T* diff_filter_data,
               const T* diff_bias_data, const T* diff_dst_data) {
    context_.src_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(src_data)));
    context_.diff_filter_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(diff_filter_data)));
    context_.diff_bias_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(diff_bias_data)));
    context_.diff_dst_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(diff_dst_data)));

    context_.bwd_filter_stream->submit(context_.bwd_filter_primitives);

    context_.src_mem->set_data_handle(DummyData);
    context_.diff_filter_mem->set_data_handle(DummyData);
    context_.diff_bias_mem->set_data_handle(DummyData);
    context_.diff_dst_mem->set_data_handle(DummyData);
    return;
  }

  // Convolution backward weights without bias
  //   src_data:         input data buffer of src
  //   diff_filter_data: output data buffer of diff_filter
  //   diff_dst_data:    input data buffer of diff_dst
  void Execute(const T* src_data, const T* diff_filter_data,
               const T* diff_dst_data) {
    context_.src_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(src_data)));
    context_.diff_filter_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(diff_filter_data)));
    context_.diff_dst_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(diff_dst_data)));

    context_.bwd_filter_stream->submit(context_.bwd_filter_primitives);

    context_.src_mem->set_data_handle(DummyData);
    context_.diff_filter_mem->set_data_handle(DummyData);
    context_.diff_dst_mem->set_data_handle(DummyData);
    return;
  }

  memory::format GetSrcMemoryFormat() const { return context_.src_fmt; }

  memory::format GetDiffDstMemoryFormat() const {
    return context_.diff_dst_fmt;
  }

  memory::format GetDiffFilterMemoryFormat() const {
    return context_.diff_filter_fmt;
  }

  // convolution primitive
  std::shared_ptr<mkldnn::convolution_backward_weights::primitive_desc>
  GetPrimitiveDesc() const {
    return context_.bwd_filter_pd;
  }

 private:
  // Primitive reuse context for Conv2D bwd filter op
  struct ConvBwdFilterContext {
    // expected memory format for this primitive instance
    memory::format src_fmt;
    memory::format diff_dst_fmt;
    memory::format diff_filter_fmt;

    // convolution bwd input primitive
    std::shared_ptr<mkldnn::convolution_backward_weights::primitive_desc>
        bwd_filter_pd;
    std::shared_ptr<mkldnn::primitive> conv_bwd_filter;

    // MKLDNN memory
    std::shared_ptr<mkldnn::memory> src_mem;
    std::shared_ptr<mkldnn::memory> diff_filter_mem;
    std::shared_ptr<mkldnn::memory> diff_bias_mem;
    std::shared_ptr<mkldnn::memory> diff_dst_mem;

    // desc & prmitive desc
    std::shared_ptr<mkldnn::convolution_backward_weights::desc> bwd_filter_desc;
    std::shared_ptr<mkldnn::convolution_forward::desc> fwd_desc;
    std::shared_ptr<mkldnn::convolution_forward::primitive_desc> fwd_pd;

    // memory desc: forward & backward can share same memory desc
    std::shared_ptr<mkldnn::memory::desc> src_md;
    std::shared_ptr<mkldnn::memory::desc> diff_filter_md;
    std::shared_ptr<mkldnn::memory::desc> diff_bias_md;
    std::shared_ptr<mkldnn::memory::desc> diff_dst_md;

    // MKL pipeline
    std::shared_ptr<mkldnn::stream> bwd_filter_stream;
    std::vector<mkldnn::primitive> bwd_filter_primitives;

    ConvBwdFilterContext()
        : src_fmt(memory::format::any),
          diff_dst_fmt(memory::format::any),
          diff_filter_fmt(memory::format::any),
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
          bwd_filter_stream(nullptr) {}
  };

  // Setup Conv2d backward filter (weights) primitives.
  void Setup(const MklConvBwdFilterParams& convBwdFilterDims) {
    // create memory descriptors for convolution data w/ no specified format
    context_.src_md.reset(new memory::desc(
        {convBwdFilterDims.src_dims}, MklDnnType<T>(), memory::format::any));

    context_.diff_dst_md.reset(
        new memory::desc({convBwdFilterDims.diff_dst_dims}, MklDnnType<T>(),
                         memory::format::any));

    context_.diff_filter_md.reset(
        new memory::desc({convBwdFilterDims.diff_filter_dims}, MklDnnType<T>(),
                         memory::format::any));

    if (!convBwdFilterDims.diff_bias_dims.empty())
      context_.diff_bias_md.reset(
          new memory::desc({convBwdFilterDims.diff_bias_dims}, MklDnnType<T>(),
                           memory::format::x));

    // create a convolution
    if (!convBwdFilterDims.diff_bias_dims.empty()) {
      context_.bwd_filter_desc.reset(new convolution_backward_weights::desc(
          convolution_direct, *context_.src_md, *context_.diff_filter_md,
          *context_.diff_bias_md, *context_.diff_dst_md,
          convBwdFilterDims.strides, convBwdFilterDims.dilations,
          convBwdFilterDims.padding_left, convBwdFilterDims.padding_right,
          convBwdFilterDims.padding));
    } else {
      context_.bwd_filter_desc.reset(new convolution_backward_weights::desc(
          convolution_direct, *context_.src_md, *context_.diff_filter_md,
          *context_.diff_dst_md, convBwdFilterDims.strides,
          convBwdFilterDims.dilations, convBwdFilterDims.padding_left,
          convBwdFilterDims.padding_right, convBwdFilterDims.padding));
    }

    // create fwd primitive_desc
    context_.fwd_desc.reset(new convolution_forward::desc(
        prop_kind::forward, convolution_direct, *context_.src_md,
        *context_.diff_filter_md, *context_.diff_dst_md,
        convBwdFilterDims.strides, convBwdFilterDims.dilations,
        convBwdFilterDims.padding_left, convBwdFilterDims.padding_right,
        convBwdFilterDims.padding));
    context_.fwd_pd.reset(new convolution_forward::primitive_desc(
        *context_.fwd_desc, cpu_engine_));

    // create backward conv primitive_desc
    context_.bwd_filter_pd.reset(
        new convolution_backward_weights::primitive_desc(
            *context_.bwd_filter_desc, cpu_engine_, *context_.fwd_pd));

    // store the expected memory format
    auto bwd_filter_pd = context_.bwd_filter_pd.get();
    context_.src_fmt = static_cast<mkldnn::memory::format>(
        bwd_filter_pd->src_primitive_desc().desc().data.format);
    context_.diff_filter_fmt = static_cast<mkldnn::memory::format>(
        bwd_filter_pd->diff_weights_primitive_desc().desc().data.format);
    context_.diff_dst_fmt = static_cast<mkldnn::memory::format>(
        bwd_filter_pd->diff_dst_primitive_desc().desc().data.format);

    // create memory primitive based on dummy data
    context_.src_mem.reset(
        new memory(bwd_filter_pd->src_primitive_desc(), DummyData));
    context_.diff_filter_mem.reset(
        new memory(bwd_filter_pd->diff_weights_primitive_desc(), DummyData));
    context_.diff_dst_mem.reset(
        new memory(bwd_filter_pd->diff_dst_primitive_desc(), DummyData));

    // create convolution primitive and add it to net
    if (!convBwdFilterDims.diff_bias_dims.empty()) {
      context_.diff_bias_mem.reset(
          new memory({{{convBwdFilterDims.diff_bias_dims},
                       MklDnnType<T>(),
                       memory::format::x},
                      cpu_engine_},
                     DummyData));
      context_.conv_bwd_filter.reset(new convolution_backward_weights(
          *context_.bwd_filter_pd, *context_.src_mem, *context_.diff_dst_mem,
          *context_.diff_filter_mem, *context_.diff_bias_mem));
    } else {
      context_.conv_bwd_filter.reset(new convolution_backward_weights(
          *context_.bwd_filter_pd, *context_.src_mem, *context_.diff_dst_mem,
          *context_.diff_filter_mem));
    }

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
      // look into the pool for reusable primitive
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
      MklDnnData<T> src(&cpu_engine_);
      MklDnnData<T> diff_dst(&cpu_engine_);
      MklDnnData<T> diff_filter(&cpu_engine_);  // output

      // This flag indicates Conv2D or Conv3D
      bool is_conv2d = (this->strides_.size() == 4);

      // Input tensors
      const int kInputIdx = 0, kFilterIdx = 1, kOutbpropIdx = 2;
      const Tensor& src_tensor = MklGetInput(context, kInputIdx);
      const Tensor& filter_tensor = MklGetInput(context, kFilterIdx);
      const Tensor& diff_dst_tensor = MklGetInput(context, kOutbpropIdx);

      MklDnnShape src_mkl_shape, filter_mkl_shape, diff_dst_mkl_shape;
      GetMklShape(context, kInputIdx, &src_mkl_shape, eager_mode);
      GetMklShape(context, kFilterIdx, &filter_mkl_shape, eager_mode);
      GetMklShape(context, kOutbpropIdx, &diff_dst_mkl_shape, eager_mode);
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
          GetTfShape(context, kOutbpropIdx, eager_mode);

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
        CHECK_NOTNULL(diff_filter_tensor);

        // if output tensor has more than 0 elements, we need to 0 them out.
        auto diff_filter_data = diff_filter_tensor->flat<T>().data();
        for (size_t i = 0; i < diff_filter_tf_shape.num_elements(); ++i) {
          diff_filter_data[i] = static_cast<T>(0);
        }
        return;
      }

      // By default, all dims are in MKL order. Only dims in TF order
      // are those with prefix tf_order.
      memory::dims diff_dst_dims, fwd_src_dims, fwd_filter_dims;
      memory::dims padding_left, padding_right, dilations, strides,
          fwd_dst_dims;
      memory::dims fwd_dst_dims_tf_order;

      // Get forward convolution parameters.
      MklDnnConvUtil conv_utl(context, this->strides_, this->padding_,
                              this->data_format_, this->dilations_);
      conv_utl.GetConvFwdSizesInMklOrder(
          src_tf_shape, filter_tf_shape, &fwd_src_dims, &fwd_filter_dims,
          &strides, &dilations, &fwd_dst_dims_tf_order, &fwd_dst_dims,
          &padding_left, &padding_right, false, is_depthwise);
      if (!context->status().ok()) return;

      auto tf_fmt = is_conv2d
                        ? TFDataFormatToMklDnnDataFormat(this->data_format_)
                        : TFDataFormatToMklDnn3DDataFormat(this->data_format_);

      auto fwd_src_md =
          src_mkl_shape.IsMklTensor()
              ? src_mkl_shape.GetMklLayout()
              : memory::desc(fwd_src_dims, MklDnnType<T>(), tf_fmt);

      conv_utl.GetInputSizeInMklOrder(diff_dst_tf_shape, &diff_dst_dims);
      if (!context->status().ok()) return;

      auto diff_dst_md =
          diff_dst_mkl_shape.IsMklTensor()
              ? diff_dst_mkl_shape.GetMklLayout()
              : memory::desc(diff_dst_dims, MklDnnType<T>(), tf_fmt);

      memory::dims diff_bias_dims = {};
      int64 depth = 0;
      if (bias_enabled) {
        TensorShape obp_tf_shape = GetTfShape(context, 2);
        depth = (this->data_format_ == FORMAT_NCHW)
                    ? obp_tf_shape.dim_size(1)
                    : obp_tf_shape.dim_size(is_conv2d ? 3 : 4);
        diff_bias_dims = {static_cast<int>(depth)};
      }
      for (int i = 0; i < dilations.size(); i++) dilations[i] -= 1;

      MklConvBwdFilterPrimitive<T>* conv_bwd_filter = nullptr;
      MklConvBwdFilterParams convBwdFilterDims(
          fwd_src_dims, fwd_filter_dims, diff_bias_dims, diff_dst_dims, strides,
          dilations, padding_left, padding_right,
          TFPaddingToMklDnnPadding(this->padding_));

      // MKL DNN allocates large buffers when a conv gradient filter primtive is
      // created. So we don't cache conv backward primitives when the env
      // variable TF_MKL_OPTIMIZE_PRIMITIVE_MEMUSE is set to true.
      bool do_not_cache = MklPrimitiveFactory<T>::IsPrimitiveMemOptEnabled();
      conv_bwd_filter = MklConvBwdFilterPrimitiveFactory<T>::Get(
          convBwdFilterDims, do_not_cache);
      auto bwd_filter_pd = conv_bwd_filter->GetPrimitiveDesc();

      // allocate output tensors: diff_fitler and diff_bias (w bias)
      auto bwd_output_dims = GetOutputDims(fwd_src_dims, fwd_filter_dims);

      // diff_filter
      MklDnnShape diff_filter_mkl_shape;
      diff_filter_mkl_shape.SetMklTensor(false);

      if (is_conv2d) {
        if (!is_depthwise) {
          // Conv2D: output_dims_mkl_order is in OIHW format.
          TensorShape diff_filter_tf_shape(
              {bwd_output_dims[MklDnnDims::Dim_H],
               bwd_output_dims[MklDnnDims::Dim_W],
               bwd_output_dims[MklDnnDims::Dim_I],
               bwd_output_dims[MklDnnDims::Dim_O]});
          AllocateOutputSetMklShape(context, 0, &diff_filter_tensor,
                                    diff_filter_tf_shape, diff_filter_mkl_shape,
                                    eager_mode);
        } else {
          // Depthwise Conv2d: bwd_output_dims is GOIHW format
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
              {bwd_output_dims[MklDnnFilterGroupDims::MKL_GROUP_FILTER_DIM_H],
               bwd_output_dims[MklDnnFilterGroupDims::MKL_GROUP_FILTER_DIM_W],
               bwd_output_dims[MklDnnFilterGroupDims::MKL_GROUP_FILTER_DIM_G],
               bwd_output_dims[MklDnnFilterGroupDims::MKL_GROUP_FILTER_DIM_O]});
          AllocateOutputSetMklShape(context, 0, &diff_filter_tensor,
                                    diff_filter_tf_shape,
                                    diff_filter_mkl_shape);
        }
      } else {
        // Conv3D: output_dims_mkl_order is in OIDHW format.
        TensorShape diff_filter_tf_shape(
            {bwd_output_dims[MklDnnDims3D::Dim3d_D],
             bwd_output_dims[MklDnnDims3D::Dim3d_H],
             bwd_output_dims[MklDnnDims3D::Dim3d_W],
             bwd_output_dims[MklDnnDims3D::Dim3d_I],
             bwd_output_dims[MklDnnDims3D::Dim3d_O]});
        AllocateOutputSetMklShape(context, 0, &diff_filter_tensor,
                                  diff_filter_tf_shape, diff_filter_mkl_shape);
      }

      Tensor* diff_bias_tensor = nullptr;
      if (bias_enabled) {
        TensorShape diff_bias_shape({depth});
        AllocateBiasGradTensor(context, diff_bias_shape, &diff_bias_tensor);
      }

      // check if src and diff_dst need reorder
      T* src_data = nullptr;
      if (fwd_src_md.data.format != conv_bwd_filter->GetSrcMemoryFormat()) {
        src.SetUsrMem(fwd_src_md, &src_tensor);
        src.CheckReorderToOpMem(bwd_filter_pd->src_primitive_desc());
        src_data = static_cast<T*>(src.GetOpMem().get_data_handle());
      } else {
        src_data = static_cast<T*>(const_cast<T*>(src_tensor.flat<T>().data()));
      }

      T* diff_dst_data = nullptr;
      if (diff_dst_md.data.format !=
          conv_bwd_filter->GetDiffDstMemoryFormat()) {
        diff_dst.SetUsrMem(diff_dst_md, &diff_dst_tensor);
        diff_dst.CheckReorderToOpMem(bwd_filter_pd->diff_dst_primitive_desc());
        diff_dst_data = static_cast<T*>(diff_dst.GetOpMem().get_data_handle());
      } else {
        diff_dst_data =
            static_cast<T*>(const_cast<T*>(diff_dst_tensor.flat<T>().data()));
      }

      // For backward filter, convert diff_filter back to Tensorflow layout
      // Here we prepare to reorder op memory back to user memory
      bool diff_filter_reorder_required = false;
      T* diff_filter_data = nullptr;
      if (GetOutputFormat(tf_fmt) !=
          conv_bwd_filter->GetDiffFilterMemoryFormat()) {
        // Allocate diff filter tensor as Tensorflow layout
        diff_filter.SetUsrMem(bwd_output_dims, GetOutputFormat(tf_fmt),
                              diff_filter_tensor);
        diff_filter_reorder_required = true;
        diff_filter.PrepareReorderToUserMemIfReq(
            bwd_filter_pd->diff_weights_primitive_desc());
        diff_filter_data =
            static_cast<T*>(diff_filter.GetOpMem().get_data_handle());
      } else {
        diff_filter_data = static_cast<T*>(
            const_cast<T*>(diff_filter_tensor->flat<T>().data()));
      }

      // Execute convolution filter bwd
      if (bias_enabled) {
        T* diff_bias_data =
            static_cast<T*>(const_cast<T*>(diff_bias_tensor->flat<T>().data()));
        conv_bwd_filter->Execute(src_data, diff_filter_data, diff_bias_data,
                                 diff_dst_data);
      } else {
        conv_bwd_filter->Execute(src_data, diff_filter_data, diff_dst_data);
      }

      // Reorder diff_filter back to Tensorflow layout if necessary
      if (diff_filter_reorder_required) {
        diff_filter.InsertReorderToUserMem();
      }

      // delete primitive since it is not cached.
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
  const int kInputIndex_Filter = 1;
  const int kInputIndex_InputSizes = 0;
  const int kDilationH = 0, kDilationW = 1;
  engine cpu_engine_ = engine(engine::cpu, 0);

  // Validate input shapes.
  // Function asserts that input shapes are valid.
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
  memory::format GetOutputFormat(const memory::format data_format) {
    return is_depthwise
               ? memory::format::hwigo
               : ((this->strides_.size() == 4) ? memory::format::hwio
                                               : memory::format::dhwio);
  }

  // Allocate output tensor.
  void AllocateOutputTensor(
      OpKernelContext* context,
      const convolution_backward_weights::primitive_desc& conv_pd,
      const memory::dims& output_dims_mkl_order,
      memory::format output_tf_format, Tensor** output_tensor) {
    CHECK_NOTNULL(output_tensor);

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

  // Allocate tensor for bias grad
  void AllocateBiasGradTensor(OpKernelContext* context,
                              const TensorShape& bias_grad_shape,
                              Tensor** bias_grad_tensor) {
    CHECK_NOTNULL(bias_grad_tensor);

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
