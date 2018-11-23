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

// See docs in ../ops/nn_ops.cc. This opkernel uses MKL library, create MKL
// layout and primitives, use MKL dnn primitives to compute convolution backward
// input

#ifdef INTEL_MKL

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS
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

using mkldnn::convolution_backward_data;
using mkldnn::prop_kind;
using mkldnn::stream;

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;

/// utility classes enabling primitive reuse for backward conv ops.
struct MklConvBwdInputParams {
  memory::dims diff_src_dims;
  memory::dims filter_dims;
  memory::dims diff_dst_dims;
  memory::dims strides;
  memory::dims dilations;
  memory::dims padding_left;
  memory::dims padding_right;
  padding_kind padding;

  MklConvBwdInputParams(memory::dims diff_src_dims, memory::dims filter_dims,
                        memory::dims diff_dst_dims, memory::dims strides,
                        memory::dims dilations, memory::dims padding_left,
                        memory::dims padding_right, padding_kind padding)
      : diff_src_dims(diff_src_dims),
        filter_dims(filter_dims),
        diff_dst_dims(diff_dst_dims),
        strides(strides),
        dilations(dilations),
        padding_left(padding_left),
        padding_right(padding_right),
        padding(padding) {}
};

template <typename T>
class MklConvBwdInputPrimitive : public MklPrimitive {
 public:
  explicit MklConvBwdInputPrimitive(
      const MklConvBwdInputParams& convBwdInputDims)
      : cpu_engine_(engine::cpu, 0) {
    context_.bwd_input_stream.reset(new stream(stream::kind::eager));

    // create conv primitive
    if (context_.conv_bwd_input == nullptr) {
      Setup(convBwdInputDims);
    }
  }
  ~MklConvBwdInputPrimitive() {}

  // Convolution backward filter (weights)
  //   diff_src_data: output data buffer of diff_src
  //   filter_data:   input data buffer of filter (weights)
  //   diff_dst_data: input data buffer of dst
  // Bias does not matter here
  void Execute(const T* diff_src_data, const T* filter_data,
               const T* diff_dst_data) {
    context_.diff_src_mem->set_data_handle(
        static_cast<T*>(const_cast<T*>(diff_src_data)));
    context_.filter_mem->set_data_handle(
        static_cast<T*>(const_cast<T*>(filter_data)));
    context_.diff_dst_mem->set_data_handle(
        static_cast<T*>(const_cast<T*>(diff_dst_data)));

    context_.bwd_input_stream->submit(context_.bwd_input_primitives);

    // set back data handle
    context_.diff_src_mem->set_data_handle(DummyData);
    context_.filter_mem->set_data_handle(DummyData);
    context_.diff_dst_mem->set_data_handle(DummyData);
    return;
  }

  memory::format GetFilterMemoryFormat() const { return context_.filter_fmt; }

  memory::format GetDiffDstMemoryFormat() const {
    return context_.diff_dst_fmt;
  }

  std::shared_ptr<mkldnn::convolution_backward_data::primitive_desc>
  GetPrimitiveDesc() const {
    return context_.bwd_input_pd;
  }

 private:
  // Primitive reuse context for Conv Bwd Input op
  struct ConvBwdInputContext {
    // expected memory format for this primitive instance
    memory::format filter_fmt;
    memory::format diff_dst_fmt;

    // MKLDNN memory
    std::shared_ptr<mkldnn::memory> diff_src_mem;
    std::shared_ptr<mkldnn::memory> filter_mem;
    std::shared_ptr<mkldnn::memory> diff_dst_mem;

    // convolution primitive
    std::shared_ptr<mkldnn::convolution_backward_data::primitive_desc>
        bwd_input_pd;
    std::shared_ptr<mkldnn::primitive> conv_bwd_input;

    // desc & prmitive desc
    std::shared_ptr<mkldnn::convolution_backward_data::desc> bwd_input_desc;
    std::shared_ptr<mkldnn::convolution_forward::desc> fwd_desc;
    std::shared_ptr<mkldnn::convolution_forward::primitive_desc> fwd_pd;

    // memory desc: forward & backward can share same memory::desc
    std::shared_ptr<memory::desc> diff_src_md;
    std::shared_ptr<memory::desc> filter_md;
    std::shared_ptr<memory::desc> diff_dst_md;

    // MKL pipeline
    std::shared_ptr<mkldnn::stream> bwd_input_stream;
    std::vector<mkldnn::primitive> bwd_input_primitives;

    ConvBwdInputContext()
        : filter_fmt(memory::format::any),
          diff_dst_fmt(memory::format::any),
          diff_src_mem(nullptr),
          filter_mem(nullptr),
          diff_dst_mem(nullptr),
          bwd_input_pd(nullptr),
          conv_bwd_input(nullptr),
          bwd_input_desc(nullptr),
          fwd_desc(nullptr),
          fwd_pd(nullptr),
          diff_src_md(nullptr),
          filter_md(nullptr),
          diff_dst_md(nullptr),
          bwd_input_stream(nullptr) {}
  };

  void Setup(const MklConvBwdInputParams& convBwdInputDims) {
    // create memory descriptors for convolution data w/ no specified format
    context_.diff_src_md.reset(
        new memory::desc({convBwdInputDims.diff_src_dims}, MklDnnType<T>(),
                         memory::format::any));
    context_.filter_md.reset(new memory::desc(
        {convBwdInputDims.filter_dims}, MklDnnType<T>(), memory::format::any));
    context_.diff_dst_md.reset(
        new memory::desc({convBwdInputDims.diff_dst_dims}, MklDnnType<T>(),
                         memory::format::any));

    // create convolution primitives
    context_.bwd_input_desc.reset(new convolution_backward_data::desc(
        convolution_direct, *context_.diff_src_md, *context_.filter_md,
        *context_.diff_dst_md, convBwdInputDims.strides,
        convBwdInputDims.dilations, convBwdInputDims.padding_left,
        convBwdInputDims.padding_right, convBwdInputDims.padding));

    context_.fwd_desc.reset(new convolution_forward::desc(
        prop_kind::forward, convolution_direct, *context_.diff_src_md,
        *context_.filter_md, *context_.diff_dst_md, convBwdInputDims.strides,
        convBwdInputDims.dilations, convBwdInputDims.padding_left,
        convBwdInputDims.padding_right, convBwdInputDims.padding));

    context_.fwd_pd.reset(new convolution_forward::primitive_desc(
        *context_.fwd_desc, cpu_engine_));

    // create backward conv prim desc
    context_.bwd_input_pd.reset(new convolution_backward_data::primitive_desc(
        *context_.bwd_input_desc, cpu_engine_, *context_.fwd_pd));

    // create memory primitive based on dummy data
    context_.diff_src_mem.reset(new memory(
        context_.bwd_input_pd.get()->diff_src_primitive_desc(), DummyData));
    context_.filter_mem.reset(new memory(
        context_.bwd_input_pd.get()->weights_primitive_desc(), DummyData));
    context_.diff_dst_mem.reset(new memory(
        context_.bwd_input_pd.get()->diff_dst_primitive_desc(), DummyData));

    // store the expected memory format
    context_.filter_fmt =
        static_cast<memory::format>(context_.bwd_input_pd.get()
                                        ->weights_primitive_desc()
                                        .desc()
                                        .data.format);
    context_.diff_dst_fmt =
        static_cast<memory::format>(context_.bwd_input_pd.get()
                                        ->diff_dst_primitive_desc()
                                        .desc()
                                        .data.format);

    // create convolution primitive and add it to net
    context_.conv_bwd_input.reset(new convolution_backward_data(
        *context_.bwd_input_pd, *context_.diff_dst_mem, *context_.filter_mem,
        *context_.diff_src_mem));

    context_.bwd_input_primitives.push_back(*context_.conv_bwd_input);
  }

  struct ConvBwdInputContext context_;
  engine cpu_engine_;
};

template <typename T>
class MklConvBwdInputPrimitiveFactory : public MklPrimitiveFactory<T> {
 private:
  MklConvBwdInputPrimitiveFactory() {}
  ~MklConvBwdInputPrimitiveFactory() {}

 public:
  static MklConvBwdInputPrimitive<T>* Get(
      const MklConvBwdInputParams& convBwdInputDims, bool do_not_cache) {
    MklConvBwdInputPrimitive<T>* conv_bwd_input = nullptr;

    if (do_not_cache) { /* Always allocate primitive */
      conv_bwd_input = new MklConvBwdInputPrimitive<T>(convBwdInputDims);
    } else {
      // look into the pool for reusable primitive
      conv_bwd_input = dynamic_cast<MklConvBwdInputPrimitive<T>*>(
          MklConvBwdInputPrimitiveFactory<T>::GetInstance().GetConvBwdInput(
              convBwdInputDims));
      if (conv_bwd_input == nullptr) {
        conv_bwd_input = new MklConvBwdInputPrimitive<T>(convBwdInputDims);
        MklConvBwdInputPrimitiveFactory<T>::GetInstance().SetConvBwdInput(
            convBwdInputDims, conv_bwd_input);
      }
    }

    return conv_bwd_input;
  }

 private:
  static MklConvBwdInputPrimitiveFactory& GetInstance() {
    static MklConvBwdInputPrimitiveFactory instance_;
    return instance_;
  }

  static string CreateKey(const MklConvBwdInputParams& convBwdInputDims) {
    string prefix = "conv_bwd_input";
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(prefix);
    key_creator.AddAsKey(convBwdInputDims.diff_src_dims);
    key_creator.AddAsKey(convBwdInputDims.filter_dims);
    key_creator.AddAsKey(convBwdInputDims.diff_dst_dims);
    key_creator.AddAsKey(convBwdInputDims.strides);
    key_creator.AddAsKey(convBwdInputDims.dilations);
    key_creator.AddAsKey(convBwdInputDims.padding_left);
    key_creator.AddAsKey(convBwdInputDims.padding_right);
    return key_creator.GetKey();
  }

  MklPrimitive* GetConvBwdInput(const MklConvBwdInputParams& convBwdInputDims) {
    string key = CreateKey(convBwdInputDims);
    return this->GetOp(key);
  }

  void SetConvBwdInput(const MklConvBwdInputParams& convBwdInputDims,
                       MklPrimitive* op) {
    string key = CreateKey(convBwdInputDims);
    this->SetOp(key, op);
  }
};

template <typename Device, class T>
class MklConvCustomBackpropInputOp : public MklConvBackpropCommonOp<Device, T> {
 public:
  explicit MklConvCustomBackpropInputOp(OpKernelConstruction* context)
      : MklConvBackpropCommonOp<Device, T>(context) {}

  ~MklConvCustomBackpropInputOp() {}

  void Compute(OpKernelContext* context) {
    try {
      MklDnnData<T> filter(&cpu_engine);
      MklDnnData<T> diff_dst(&cpu_engine);

      // This flag indicate Conv2D or Conv3D
      bool is_Conv2D = (this->strides_.size() == 4);

      // Input tensors
      const int kInputIdx = 0, kFilterIdx = 1, kOutbpropIdx = 2;
      const Tensor& src_tensor = MklGetInput(context, kInputIdx);
      const Tensor& filter_tensor = MklGetInput(context, kFilterIdx);
      const Tensor& diff_dst_tensor = MklGetInput(context, kOutbpropIdx);

      MklDnnShape src_mkl_shape, filter_mkl_shape, diff_dst_mkl_shape;
      GetMklShape(context, kInputIdx, &src_mkl_shape);
      GetMklShape(context, kFilterIdx, &filter_mkl_shape);
      GetMklShape(context, kOutbpropIdx, &diff_dst_mkl_shape);
      // Allow operator-specific sanity checking of shapes.
      ValidateMklShapes(src_mkl_shape, filter_mkl_shape, diff_dst_mkl_shape);

      // Allow operator-specific generation of shapes.
      // E.g., ConvBackpropFilter gets filter as filter_sizes. It is a
      // tensor containing shape of filter. So filter.shape() is not
      // a correct way to get filter shape. These operator-specific calls
      // allow this class to handle this case.
      TensorShape src_tf_shape = MakeInputTfShape(context, src_tensor);
      TensorShape filter_tf_shape = MakeFilterTfShape(context, filter_tensor);
      TensorShape diff_dst_tf_shape = GetTfShape(context, kOutbpropIdx);

      // Corner cases: output with 0 elements and 0 batch size.
      Tensor* diff_src_tensor = nullptr;
      if (src_tf_shape.num_elements() == 0 ||
          filter_tf_shape.num_elements() == 0 ||
          diff_dst_tf_shape.num_elements() == 0) {
        MklDnnShape diff_src_mkl_shape;
        diff_src_mkl_shape.SetMklTensor(false);
        TensorShape diff_src_tf_shape =
            GetOutputTfShape(src_tf_shape, filter_tf_shape, diff_dst_tf_shape);
        const int kOutputIdx = 0;
        AllocateOutputSetMklShape(context, kOutputIdx, &diff_src_tensor,
                                  diff_src_tf_shape, diff_src_mkl_shape);
        CHECK_NOTNULL(diff_src_tensor);

        // if output tensor has more than 0 elements, we need to 0 them out.
        auto diff_src_data = diff_src_tensor->flat<T>().data();
        for (size_t i = 0; i < diff_src_tf_shape.num_elements(); ++i) {
          diff_src_data[i] = 0;
        }
        return;
      }

      // By default, all dims are in MKL order. Only dims in TF order
      // are those with postfix tf_order.
      memory::dims diff_dst_dims, fwd_src_dims, fwd_filter_dims;
      memory::dims padding_left, padding_right, dilations, strides;
      memory::dims fwd_output_dims, fwd_output_dims_tf_order;

      // Get forward convolution parameters.
      MklDnnConvUtil conv_utl(context, this->strides_, this->padding_,
                              this->data_format_, this->dilations_);
      conv_utl.GetConvFwdSizesInMklOrder(
          src_tf_shape, filter_tf_shape, &fwd_src_dims, &fwd_filter_dims,
          &strides, &dilations, &fwd_output_dims_tf_order, &fwd_output_dims,
          &padding_left, &padding_right, false);
      if (!context->status().ok()) return;

      // Create Convolution forward descriptor since Convolution backward
      // API needs it. For that, we first need to create input, filter
      // and output memory descriptors.
      auto tf_fmt = is_Conv2D
                        ? TFDataFormatToMklDnnDataFormat(this->data_format_)
                        : TFDataFormatToMklDnn3DDataFormat(this->data_format_);

      // If filter is in MKL layout, then simply grab filter layout;
      // otherwise, construct filter in TF layout.
      // For TF layout, filter is in HWIO format.
      auto fwd_filter_md =
          filter_mkl_shape.IsMklTensor()
              ? filter_mkl_shape.GetMklLayout()
              : memory::desc(
                    fwd_filter_dims, MklDnnType<T>(),
                    is_Conv2D ? memory::format::hwio : memory::format::dhwio);

      conv_utl.GetInputSizeInMklOrder(diff_dst_tf_shape, &diff_dst_dims);
      if (!context->status().ok()) return;
      auto diff_dst_md =
          diff_dst_mkl_shape.IsMklTensor()
              ? diff_dst_mkl_shape.GetMklLayout()
              : memory::desc(diff_dst_dims, MklDnnType<T>(), tf_fmt);
      for (int i = 0; i < dilations.size(); i++) dilations[i] -= 1;

      MklConvBwdInputPrimitive<T>* conv_bwd_input = nullptr;
      MklConvBwdInputParams convBwdInputDims(
          fwd_src_dims, fwd_filter_dims, diff_dst_dims, strides, dilations,
          padding_left, padding_right,
          TFPaddingToMklDnnPadding(this->padding_));

      // We don't cache those primitves if the env variable
      // TF_MKL_OPTIMIZE_PRIMITIVE_MEMUSE is true and if primitve descriptor
      // includes potentialy large buffers. MKL DNN allocates buffers
      // in the following cases
      //   1. Legacy CPU without AVX512/AVX2, or
      //   2. 1x1 convolution with stride != 1
      bool do_not_cache = MklPrimitiveFactory<T>::IsPrimitiveMemOptEnabled() &&
                          (MklPrimitiveFactory<T>::IsLegacyPlatform() ||
                           IsConv1x1StrideNot1(fwd_filter_dims, strides));
      conv_bwd_input = MklConvBwdInputPrimitiveFactory<T>::Get(convBwdInputDims,
                                                               do_not_cache);
      auto bwd_input_pd = conv_bwd_input->GetPrimitiveDesc();

      // allocate output tensor
      auto diff_src_pd = bwd_input_pd->diff_src_primitive_desc();
      auto bwd_diff_src_dims = GetOutputDims(fwd_src_dims, fwd_filter_dims);
      auto bwd_diff_src_format = GetOutputFormat(tf_fmt);
      MklDnnShape diff_src_mkl_shape;
      diff_src_mkl_shape.SetMklTensor(true);
      diff_src_mkl_shape.SetMklLayout(&diff_src_pd);
      diff_src_mkl_shape.SetElemType(MklDnnType<T>());
      diff_src_mkl_shape.SetTfLayout(bwd_diff_src_dims.size(),
                                     bwd_diff_src_dims, bwd_diff_src_format);
      TensorShape diff_src_tf_shape;
      diff_src_tf_shape.AddDim(diff_src_pd.get_size() / sizeof(T));
      AllocateOutputSetMklShape(context, 0, &diff_src_tensor, diff_src_tf_shape,
                                diff_src_mkl_shape);

      T* diff_src_data =
          static_cast<T*>(const_cast<T*>(diff_src_tensor->flat<T>().data()));

      // check if filter and diff_dst need reorder
      T* filter_data = nullptr;
      if (fwd_filter_md.data.format !=
          conv_bwd_input->GetFilterMemoryFormat()) {
        filter.SetUsrMem(fwd_filter_md, &filter_tensor);
        filter.CheckReorderToOpMem(bwd_input_pd->weights_primitive_desc());
        filter_data = static_cast<T*>(filter.GetOpMem().get_data_handle());
      } else {
        filter_data =
            static_cast<T*>(const_cast<T*>(filter_tensor.flat<T>().data()));
      }

      T* diff_dst_data = nullptr;
      if (diff_dst_md.data.format != conv_bwd_input->GetDiffDstMemoryFormat()) {
        diff_dst.SetUsrMem(diff_dst_md, &diff_dst_tensor);
        diff_dst.CheckReorderToOpMem(bwd_input_pd->diff_dst_primitive_desc());
        diff_dst_data = static_cast<T*>(diff_dst.GetOpMem().get_data_handle());
      } else {
        diff_dst_data =
            static_cast<T*>(const_cast<T*>(diff_dst_tensor.flat<T>().data()));
      }

      // execute convolution input bwd
      conv_bwd_input->Execute(diff_src_data, filter_data, diff_dst_data);

      // delete primitive since it is not cached.
      if (do_not_cache) {
        delete conv_bwd_input;
      }
    } catch (mkldnn::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) + ", message: " +
                         string(e.message) + ", in file " + string(__FILE__) +
                         ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }

 private:
  const int kInputIndex_Filter = 1, kInputIndex_InputSizes = 0;
  const int kDilationH = 0, kDilationW = 1;
  engine cpu_engine = engine(engine::cpu, 0);

  // Validate input shapes.
  // Function asserts that input shapes are valid.
  void ValidateMklShapes(const MklDnnShape& input_mkl_shape,
                         const MklDnnShape& filter_mkl_shape,
                         const MklDnnShape& obp_mkl_shape) {
    // Tensor that feeds to 'Input' slot of BackpropInput is always just a shape
    // of the Tensor and never an actual tensor. So it will never be in MKL
    // layout.
    CHECK(!input_mkl_shape.IsMklTensor())
        << "ConvBackpropInput: input should not be in MKL Layout";
  }

  // Get TensorFlow shape of input tensor.
  TensorShape MakeInputTfShape(OpKernelContext* context,
                               const Tensor& input_tensor) {
    TensorShape input_tf_shape;
    CHECK_EQ(TensorShapeUtils::IsVector(input_tensor.shape()), true);
    // Conv[2D|3D]BackpropInputV2 supports both DT_INT32 and DT_INT64
    // output_shape MakeShape is able to handle both DT_INT32 and DT_INT64 for
    // input_tensor.
    CHECK_EQ(this->MakeShape(input_tensor, &input_tf_shape).ok(), true);
    return input_tf_shape;
  }

  // Get TensorFlow shape of filter tensor.
  TensorShape MakeFilterTfShape(OpKernelContext* context,
                                const Tensor& filter_tensor) {
    return GetTfShape(context, kInputIndex_Filter);
  }

  // Get the Tensorflow shape of Output (diff_src),
  // which is same as shape of Conv 'input'.
  TensorShape GetOutputTfShape(const TensorShape& input_shape,
                               const TensorShape& filter_shape,
                               const TensorShape& outbprop_shape) {
    return input_shape;
  }

  // Get the Tensorflow shape of Output (diff_src),
  // which is same as shape of Conv 'input'.
  const memory::dims& GetOutputDims(const memory::dims& fwd_input_dims,
                                    const memory::dims& fwd_filter_dims) {
    return fwd_input_dims;
  }

  // Output layout is Tensorflow's layout in data format order.
  memory::format GetOutputFormat(const memory::format data_format) {
    return data_format;
  }

  // Allocate output tensor.
  void AllocateOutputTensor(
      OpKernelContext* context,
      const convolution_backward_data::primitive_desc& conv_pd,
      const memory::dims& output_dims_mkl_order,
      memory::format output_tf_format, Tensor** output_tensor) {
    CHECK_NOTNULL(output_tensor);

    // Output primitive descriptor for backward data is diff_src.
    auto dst_pd = conv_pd.diff_src_primitive_desc();

    // Allocate shape of Mkl tensor.
    MklDnnShape output_mkl_shape;
    output_mkl_shape.SetMklTensor(true);
    output_mkl_shape.SetMklLayout(&dst_pd);
    output_mkl_shape.SetElemType(MklDnnType<T>());
    output_mkl_shape.SetTfLayout(output_dims_mkl_order.size(),
                                 output_dims_mkl_order, output_tf_format);

    // Allocate shape of TF tensor.
    TensorShape output_tf_shape;
    output_tf_shape.AddDim(dst_pd.get_size() / sizeof(T));

    AllocateOutputSetMklShape(context, 0, output_tensor, output_tf_shape,
                              output_mkl_shape);
  }
};

#define REGISTER_MKL_CPU_KERNELS(T)                                    \
  REGISTER_KERNEL_BUILDER(Name("_MklConv2DBackpropInput")              \
                              .Device(DEVICE_CPU)                      \
                              .TypeConstraint<T>("T")                  \
                              .Label(mkl_op_registry::kMklOpLabel),    \
                          MklConvCustomBackpropInputOp<CPUDevice, T>); \
  REGISTER_KERNEL_BUILDER(Name("_MklConv3DBackpropInputV2")            \
                              .Device(DEVICE_CPU)                      \
                              .TypeConstraint<T>("T")                  \
                              .Label(mkl_op_registry::kMklOpLabel),    \
                          MklConvCustomBackpropInputOp<CPUDevice, T>);

TF_CALL_float(REGISTER_MKL_CPU_KERNELS);
#undef REGISTER_MKL_CPU_KERNELS

}  // namespace tensorflow
#endif  // INTEL_MKL
