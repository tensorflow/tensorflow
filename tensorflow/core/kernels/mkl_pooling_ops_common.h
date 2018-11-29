/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_MKL_POOLING_OPS_COMMON_H_
#define TENSORFLOW_CORE_KERNELS_MKL_POOLING_OPS_COMMON_H_

#ifdef INTEL_MKL
#include <memory>
#include <string>
#include <vector>
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/util/padding.h"

#ifndef INTEL_MKL_ML_ONLY
#include "mkldnn.hpp"
using mkldnn::memory;
using mkldnn::pooling_backward;
using mkldnn::pooling_forward;
using mkldnn::stream;
#endif

namespace tensorflow {

#ifndef INTEL_MKL_ML_ONLY

using mkldnn::memory;
using mkldnn::pooling_avg;
using mkldnn::pooling_avg_exclude_padding;
using mkldnn::pooling_avg_include_padding;
using mkldnn::pooling_max;
using mkldnn::prop_kind;

struct MklPoolingParams {
  memory::dims src_dims;
  memory::dims dst_dims;
  memory::dims filter_dims;
  memory::dims strides;
  memory::dims padding_left;
  memory::dims padding_right;
  mkldnn::algorithm alg_kind;
  mkldnn::prop_kind prop_kind;

  MklPoolingParams(memory::dims src_dims, memory::dims dst_dims,
                   memory::dims filter_dims, memory::dims strides,
                   memory::dims padding_left, memory::dims padding_right,
                   mkldnn::algorithm alg_kind, mkldnn::prop_kind prop_kind)
      : src_dims(src_dims),
        dst_dims(dst_dims),
        filter_dims(filter_dims),
        strides(strides),
        padding_left(padding_left),
        padding_right(padding_right),
        alg_kind(alg_kind),
        prop_kind(prop_kind) {}
};

template <typename T>
class MklPoolingFwdPrimitive : public MklPrimitive {
 public:
  explicit MklPoolingFwdPrimitive(const MklPoolingParams& fwdParams)
      : cpu_engine_(engine::cpu, 0) {
    context_.fwd_stream.reset(new stream(stream::kind::eager));
    if (context_.fwd == nullptr) Setup(fwdParams);
  }

  ~MklPoolingFwdPrimitive() {}

  // Pooling forward execute
  //   src_data:  input data buffer of src
  //   ws_data:   output data buffer of workspace
  //   dst_data:  output data buffer of dst
  void Execute(const T* src_data, T* dst_data, void* ws_data = nullptr);

  std::shared_ptr<mkldnn::pooling_forward::primitive_desc> GetPoolingFwdPd()
      const {
    return context_.fwd_pd;
  }

  memory::format GetSrcMemoryFormat() const { return context_.src_fmt; }

  memory::format GetDstMemoryFormat() const { return context_.dst_fmt; }

 private:
  void Setup(const MklPoolingParams& fwdParams);

  struct PoolingFwdContext {
    // algorithm
    mkldnn::algorithm alg_kind;

    // Kind of propagation, forward or backward
    mkldnn::prop_kind prop_kind;

    // expected memory format
    memory::format src_fmt;
    memory::format dst_fmt;
    memory::format ws_fmt;

    // workspace shape
    memory::dims ws_dims;
    memory::data_type ws_dt;
    size_t ws_size;

    // MKL-DNN memory, just dummy data
    std::shared_ptr<mkldnn::memory> ws_mem;
    std::shared_ptr<mkldnn::memory> src_mem;
    std::shared_ptr<mkldnn::memory> dst_mem;

    // desc & primitive desc
    std::shared_ptr<mkldnn::pooling_forward::desc> fwd_desc;
    std::shared_ptr<mkldnn::pooling_forward::primitive_desc> fwd_pd;

    // memory desc
    std::shared_ptr<mkldnn::memory::desc> src_md;
    std::shared_ptr<mkldnn::memory::desc> dst_md;

    // Pooling primitive
    std::shared_ptr<mkldnn::pooling_forward> fwd;
    std::shared_ptr<mkldnn::stream> fwd_stream;
    std::vector<mkldnn::primitive> fwd_primitives;

    PoolingFwdContext()
        : src_fmt(memory::format::any),
          dst_fmt(memory::format::any),
          ws_fmt(memory::format::any),
          ws_mem(nullptr),
          src_mem(nullptr),
          dst_mem(nullptr),
          fwd_desc(nullptr),
          fwd_pd(nullptr),
          src_md(nullptr),
          dst_md(nullptr),
          fwd(nullptr),
          fwd_stream(nullptr) {}
  };

  struct PoolingFwdContext context_;
  engine cpu_engine_;
};

template <typename T>
class MklPoolingFwdPrimitiveFactory : public MklPrimitiveFactory<T> {
 public:
  static MklPoolingFwdPrimitive<T>* Get(const MklPoolingParams& fwdParams) {
    MklPoolingFwdPrimitive<T>* pooling_forward = nullptr;

    // Get pooling primitive from the pool
    pooling_forward = static_cast<MklPoolingFwdPrimitive<T>*>(
        MklPoolingFwdPrimitiveFactory<T>::GetInstance().GetPoolingFwd(
            fwdParams));

    if (pooling_forward == nullptr) {
      pooling_forward = new MklPoolingFwdPrimitive<T>(fwdParams);
      MklPoolingFwdPrimitiveFactory<T>::GetInstance().SetPoolingFwd(
          fwdParams, pooling_forward);
    }
    return pooling_forward;
  }

  static MklPoolingFwdPrimitiveFactory& GetInstance() {
    static MklPoolingFwdPrimitiveFactory instance_;
    return instance_;
  }

 private:
  MklPoolingFwdPrimitiveFactory() {}
  ~MklPoolingFwdPrimitiveFactory() {}

  // The key to be created will be used to get/set pooling
  // primitive op from reuse perspective.
  // A pooling key is a string which concates key parameters
  // as well as algorithm kind (max versus avg).
  static string CreateKey(const MklPoolingParams& fwdParams) {
    string prefix = "pooling_fwd";
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(prefix);
    key_creator.AddAsKey(fwdParams.src_dims);
    key_creator.AddAsKey(fwdParams.dst_dims);
    key_creator.AddAsKey(fwdParams.filter_dims);
    key_creator.AddAsKey(fwdParams.strides);
    key_creator.AddAsKey(fwdParams.padding_left);
    key_creator.AddAsKey(fwdParams.padding_right);
    key_creator.AddAsKey<int>(static_cast<int>(fwdParams.alg_kind));
    key_creator.AddAsKey<int>(static_cast<int>(fwdParams.prop_kind));
    return key_creator.GetKey();
  }

  MklPrimitive* GetPoolingFwd(const MklPoolingParams& fwdParams) {
    string key = CreateKey(fwdParams);
    return this->GetOp(key);
  }

  void SetPoolingFwd(const MklPoolingParams& fwdParams, MklPrimitive* op) {
    string key = CreateKey(fwdParams);
    this->SetOp(key, op);
  }
};

template <typename T>
class MklPoolingBwdPrimitive : public MklPrimitive {
 public:
  explicit MklPoolingBwdPrimitive(const MklPoolingParams& bwdParams)
      : cpu_engine(engine::cpu, 0) {
    context_.bwd_stream.reset(new stream(stream::kind::eager));
    if (context_.bwd == nullptr) Setup(bwdParams);
  }

  ~MklPoolingBwdPrimitive() {}

  // Pooling backward execute
  //   diff_dst_data:  input data buffer of diff_dst
  //   diff_src_data:  output data buffer of diff_src
  //   ws_data:        input data buffer of workspace
  void Execute(const T* diff_dst_data, T* diff_src_data,
               const void* ws_data = nullptr);

 public:
  std::shared_ptr<mkldnn::pooling_forward::primitive_desc> GetPoolingFwdPd()
      const {
    return context_.fwd_pd;
  }
  std::shared_ptr<mkldnn::pooling_backward::primitive_desc> GetPoolingBwdPd()
      const {
    return context_.bwd_pd;
  }

  memory::format GetDiffDstFormat() const { return context_.diff_dst_fmt; }

  mkldnn::memory::data_type GetWorkspaceDataType() const {
    return context_.ws_dt;
  }
  memory::format GetWorkspaceFormat() const { return context_.ws_fmt; }

 private:
  void Setup(const MklPoolingParams& bwdParams);

  // Primitive reuse context for pooling bwd ops
  struct PoolingBwdContext {
    // algorithm
    mkldnn::algorithm alg_kind;

    // expected memory format
    mkldnn::memory::format diff_src_fmt;
    mkldnn::memory::format diff_dst_fmt;
    mkldnn::memory::format ws_fmt;

    // workspace attribute
    mkldnn::memory::dims ws_dims;
    mkldnn::memory::data_type ws_dt;

    // MKL-DNN memory
    std::shared_ptr<mkldnn::memory> ws_mem;
    std::shared_ptr<mkldnn::memory> diff_src_mem;
    std::shared_ptr<mkldnn::memory> diff_dst_mem;

    // memory desc
    std::shared_ptr<mkldnn::memory::desc> diff_src_md;
    std::shared_ptr<mkldnn::memory::desc> diff_dst_md;

    // desc & primitive desc
    std::shared_ptr<mkldnn::pooling_forward::desc> fwd_desc;
    std::shared_ptr<mkldnn::pooling_backward::desc> bwd_desc;
    std::shared_ptr<mkldnn::pooling_forward::primitive_desc> fwd_pd;
    std::shared_ptr<mkldnn::pooling_backward::primitive_desc> bwd_pd;

    // pooling primitive
    std::shared_ptr<mkldnn::pooling_backward> bwd;
    std::shared_ptr<mkldnn::stream> bwd_stream;

    std::vector<mkldnn::primitive> bwd_primitives;

    PoolingBwdContext()
        : diff_src_fmt(memory::format::any),
          diff_dst_fmt(memory::format::any),
          ws_fmt(memory::format::any),
          ws_mem(nullptr),
          diff_src_mem(nullptr),
          diff_dst_mem(nullptr),
          diff_src_md(nullptr),
          diff_dst_md(nullptr),
          fwd_desc(nullptr),
          bwd_desc(nullptr),
          fwd_pd(nullptr),
          bwd_pd(nullptr),
          bwd(nullptr),
          bwd_stream(nullptr) {}
  };

  struct PoolingBwdContext context_;
  engine cpu_engine;
};

template <typename T>
class MklPoolingBwdPrimitiveFactory : public MklPrimitiveFactory<T> {
 public:
  static MklPoolingBwdPrimitive<T>* Get(const MklPoolingParams& bwdParams) {
    MklPoolingBwdPrimitive<T>* pooling_backward = nullptr;

    // Find a pooling backward primitive from the pool
    // If it does not exist, create a new one
    pooling_backward = static_cast<MklPoolingBwdPrimitive<T>*>(
        MklPoolingBwdPrimitiveFactory<T>::GetInstance().GetPoolingBwd(
            bwdParams));
    if (pooling_backward == nullptr) {
      pooling_backward = new MklPoolingBwdPrimitive<T>(bwdParams);
      MklPoolingBwdPrimitiveFactory<T>::GetInstance().SetPoolingBwd(
          bwdParams, pooling_backward);
    }
    return pooling_backward;
  }

  static MklPoolingBwdPrimitiveFactory& GetInstance() {
    static MklPoolingBwdPrimitiveFactory instance_;
    return instance_;
  }

 private:
  MklPoolingBwdPrimitiveFactory() {}
  ~MklPoolingBwdPrimitiveFactory() {}

  // The key to be created will be used to get/set pooling
  // primitive op from reuse perspective.
  // A pooling key is a string which concates key parameters
  // as well as algorithm kind (max versus avg).
  static string CreateKey(const MklPoolingParams& bwdParams) {
    string prefix = "pooling_bwd";
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(prefix);
    key_creator.AddAsKey(bwdParams.src_dims);
    key_creator.AddAsKey(bwdParams.dst_dims);
    key_creator.AddAsKey(bwdParams.filter_dims);
    key_creator.AddAsKey(bwdParams.strides);
    key_creator.AddAsKey(bwdParams.padding_left);
    key_creator.AddAsKey(bwdParams.padding_right);
    key_creator.AddAsKey<int>(static_cast<int>(bwdParams.alg_kind));
    return key_creator.GetKey();
  }

  MklPrimitive* GetPoolingBwd(const MklPoolingParams& bwdParams) {
    string key = CreateKey(bwdParams);
    return this->GetOp(key);
  }

  void SetPoolingBwd(const MklPoolingParams& bwdParams, MklPrimitive* op) {
    string key = CreateKey(bwdParams);
    this->SetOp(key, op);
  }
};
#endif

typedef Eigen::ThreadPoolDevice CPUDevice;

struct MklPoolParameters {
  int depth;

  int tensor_in_planes;  // Pool3D
  int tensor_in_cols;
  int tensor_in_rows;
  int tensor_in_batch;

  int window_planes;  // Pool3D
  int window_rows;
  int window_cols;
  int depth_window;

  int planes_stride;  // Pool3D
  int row_stride;
  int col_stride;
  int depth_stride;

  int64 out_planes;  // Pool3D
  int64 out_height;
  int64 out_width;
  int out_depth;

  int64 pad_P1;  // Pool3D
  int64 pad_P2;  // Pool3D
  int64 pad_left;
  int64 pad_right;
  int64 pad_top;
  int64 pad_bottom;
  int pad_depth;

  TensorFormat data_format;
  MklPoolParameters()
      : depth(0),
        tensor_in_planes(0),
        tensor_in_cols(0),
        tensor_in_rows(0),
        tensor_in_batch(0),
        window_planes(0),
        window_rows(0),
        window_cols(0),
        depth_window(0),
        planes_stride(0),
        row_stride(0),
        col_stride(0),
        depth_stride(0),
        out_planes(0),
        out_height(0),
        out_width(0),
        out_depth(0),
        pad_P1(0),
        pad_P2(0),
        pad_left(0),
        pad_right(0),
        pad_top(0),
        pad_bottom(0),
        pad_depth(0),
        data_format(TensorFormat::FORMAT_NCHW) {}

  // Updates context->status if there is an invalid input.
  void Init(OpKernelContext* context, const std::vector<int32>& ksize,
            const std::vector<int32>& stride, Padding padding,
            TensorFormat data_format, const TensorShape& tensor_in_shape);
#ifdef INTEL_MKL_ML_ONLY
  void Init(OpKernelContext* context, const std::vector<int32>& ksize,
            const std::vector<int32>& stride, Padding padding,
            TensorFormat data_format, const MklShape* mkl_in_shape);
#else
  void Init(OpKernelContext* context, const std::vector<int32>& ksize,
            const std::vector<int32>& stride, Padding padding,
            TensorFormat data_format, const MklDnnShape* mkl_in_shape);
#endif

 private:
  // Common initialization for TensorFlow and MKL formats
  void Init(OpKernelContext* context, const std::vector<int32>& ksize,
            const std::vector<int32>& stride, Padding padding,
            TensorFormat data_format);
};

#ifndef INTEL_MKL_ML_ONLY

template <class T>
class MklPoolingOpBase : public OpKernel {
 public:
  explicit MklPoolingOpBase(OpKernelConstruction* context)
      : OpKernel(context), workspace_enabled_(false) {
    string data_format;
    if (std::is_same<T, qint8>::value || std::is_same<T, quint8>::value) {
      // current quantized convolution doesn't have data_format attribute.
      data_format = "NHWC";
    } else {
      OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    }
    OP_REQUIRES(context, FormatFromString(data_format, &this->data_format_tf_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &this->ksize_));
    OP_REQUIRES(context, this->ksize_.size() == 4 || this->ksize_.size() == 5,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 or 5 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &this->stride_));
    OP_REQUIRES(context, this->stride_.size() == 4 || this->stride_.size() == 5,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 or 5 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &this->padding_));
    OP_REQUIRES(context, this->ksize_[0] == 1 && this->stride_[0] == 1,
                errors::Unimplemented("Pooling is not yet supported on the "
                                      "batch dimension."));
    bool is_pool2d = (this->ksize_.size() == 4);
    this->data_format_mkldnn_ =
        is_pool2d ? TFDataFormatToMklDnnDataFormat(this->data_format_tf_)
                  : TFDataFormatToMklDnn3DDataFormat(this->data_format_tf_);

    // We may not get this attribute for this node if it does not go through
    // graph rewrite pass. So we do not check for error while retrieving this
    // attribute value.
    context->GetAttr("workspace_enabled", &this->workspace_enabled_);
  }
  void Compute(OpKernelContext* context) override = 0;

 protected:
  // Calculate output shape of pooling op in MKL-DNN and TensorFlow order.
  // MKL-DNN uses NCHW(Pool2D) or NCDHW(Pool3D) for output order.
  // But TensorFlow output will be in NHWC/NCHW(Pool2D) or
  // NDHWC/NCDHW(Pool3D) format depending on data format. Function expects
  // output height and width to have already been int32 bounds-checked.
  void GetOutputDims(const MklPoolParameters& mkl_pool_params,
                     memory::dims* output_dims_mkl_order) {
    if (this->ksize_.size() == 4) {
      // Pooling2D: MKL-DNN always needs output in NCHW format.
      *output_dims_mkl_order = {mkl_pool_params.tensor_in_batch,
                                mkl_pool_params.out_depth,
                                static_cast<int>(mkl_pool_params.out_height),
                                static_cast<int>(mkl_pool_params.out_width)};
    } else {
      // Pooling3D: MKL-DNN always needs output in NCDHW format.
      *output_dims_mkl_order = {mkl_pool_params.tensor_in_batch,
                                mkl_pool_params.out_depth,
                                static_cast<int>(mkl_pool_params.out_planes),
                                static_cast<int>(mkl_pool_params.out_height),
                                static_cast<int>(mkl_pool_params.out_width)};
    }
  }

  void InitMklPoolParameters(OpKernelContext* context,
                             MklPoolParameters* pool_params,
                             const MklDnnShape& original_input_mkl_shape,
                             const TensorShape& input_tensor_shape) {
    if (!original_input_mkl_shape.IsMklTensor()) {
      pool_params->Init(context, this->ksize_, this->stride_, this->padding_,
                        this->data_format_tf_, input_tensor_shape);
    } else {
      pool_params->Init(context, this->ksize_, this->stride_, this->padding_,
                        this->data_format_tf_, &original_input_mkl_shape);
    }
  }

  void PoolParamsToDims(const MklPoolParameters* pool_params,
                        memory::dims* filter_dims, memory::dims* strides,
                        memory::dims* padding_left, memory::dims* padding_right,
                        bool is_pool2d) {
    if (is_pool2d) {
      // Pool2D
      *filter_dims =
          memory::dims({pool_params->window_rows, pool_params->window_cols});
      *strides =
          memory::dims({pool_params->row_stride, pool_params->col_stride});
      *padding_left = memory::dims({static_cast<int>(pool_params->pad_top),
                                    static_cast<int>(pool_params->pad_left)});
      *padding_right = memory::dims({static_cast<int>(pool_params->pad_bottom),
                                     static_cast<int>(pool_params->pad_right)});
    } else {
      // Pool3D
      *filter_dims =
          memory::dims({pool_params->window_planes, pool_params->window_rows,
                        pool_params->window_cols});
      *strides =
          memory::dims({pool_params->planes_stride, pool_params->row_stride,
                        pool_params->col_stride});

      *padding_left = memory::dims({static_cast<int>(pool_params->pad_P1),
                                    static_cast<int>(pool_params->pad_top),
                                    static_cast<int>(pool_params->pad_left)});
      *padding_right = memory::dims({static_cast<int>(pool_params->pad_P2),
                                     static_cast<int>(pool_params->pad_bottom),
                                     static_cast<int>(pool_params->pad_right)});
    }
  }

  void AllocateEmptyOutputTensor(OpKernelContext* context,
                                 const int kOutputIndex,
                                 MklPoolParameters* pool_params,
                                 const memory::dims output_dims_mkl_order,
                                 Tensor** output_tensor) {
    MklDnnShape output_mkl_shape;
    output_mkl_shape.SetMklTensor(false);
    TensorShape output_tf_shape;
    if (pool_params->data_format == TensorFormat::FORMAT_NCHW) {
      output_tf_shape = MklDnnDimsToTFShape(output_dims_mkl_order);
    } else {
      memory::dims output_dims_NHWC_order;
      output_dims_NHWC_order = {pool_params->tensor_in_batch,
                                static_cast<int>(pool_params->out_height),
                                static_cast<int>(pool_params->out_width),
                                pool_params->out_depth};
      output_tf_shape = MklDnnDimsToTFShape(output_dims_NHWC_order);
    }
    AllocateOutputSetMklShape(context, kOutputIndex, output_tensor,
                              output_tf_shape, output_mkl_shape);
    CHECK_NOTNULL(output_tensor);
  }

  // Checks to make sure that the memory we need to allocate
  // is a multiple of sizeof(T)
  // returns the number of elements
  size_t GetNumTElements(const memory::primitive_desc& pd) {
    size_t num_bytes = pd.get_size();
    size_t ret_val = num_bytes / sizeof(T);
    if (num_bytes % sizeof(T) != 0) {
      ret_val++;
    }
    return ret_val;
  }

  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_tf_;
  memory::format data_format_mkldnn_;
  bool workspace_enabled_;
};

template <class T>
class MklPoolingForwardOpBase : public MklPoolingOpBase<T> {
 public:
  explicit MklPoolingForwardOpBase<T>(OpKernelConstruction* context)
      : MklPoolingOpBase<T>(context) {}
  void Compute(OpKernelContext* context) override = 0;

 protected:
  void ConfigureInput(OpKernelContext* context,
                      const MklDnnShape& input_mkl_shape,
                      const Tensor& input_tensor,
                      MklPoolParameters* pool_params,
                      MklDnnData<T>* dnn_data_input) {
    CHECK_NOTNULL(pool_params);
    CHECK_NOTNULL(dnn_data_input);
    TensorShape input_tensor_shape = input_tensor.shape();
    if (input_tensor.NumElements() != 0) {
      memory::desc input_md =
          input_mkl_shape.IsMklTensor()
              ? input_mkl_shape.GetMklLayout()
              : memory::desc(
                    (this->ksize_.size() == 4)
                        ? TFShapeToMklDnnDimsInNCHW(input_tensor_shape,
                                                    this->data_format_tf_)
                        : TFShapeToMklDnnDimsInNCDHW(input_tensor_shape,
                                                     this->data_format_tf_),
                    MklDnnType<T>(), this->data_format_mkldnn_);
      dnn_data_input->SetUsrMem(input_md, &input_tensor);

      if (this->ksize_.size() == 5) {
        // Pool3D
        std::vector<int> mkldnn_sizes(5, -1);
        mkldnn_sizes[MklDnnDims3D::Dim3d_N] = input_md.data.dims[0];
        mkldnn_sizes[MklDnnDims3D::Dim3d_C] = input_md.data.dims[1];
        mkldnn_sizes[MklDnnDims3D::Dim3d_D] = input_md.data.dims[2];
        mkldnn_sizes[MklDnnDims3D::Dim3d_H] = input_md.data.dims[3];
        mkldnn_sizes[MklDnnDims3D::Dim3d_W] = input_md.data.dims[4];
        dnn_data_input->SetOpMemDesc(mkldnn_sizes, this->data_format_mkldnn_);
      }
    }
    this->InitMklPoolParameters(context, pool_params, input_mkl_shape,
                                input_tensor_shape);
  }

  void AllocateOutputTensor(
      OpKernelContext* context,
      const pooling_forward::primitive_desc& pool_fwd_prim_desc,
      const memory::dims output_dims_mkl_order,
      const memory::format& output_tf_format, Tensor** output_tensor) {
    CHECK_NOTNULL(output_tensor);
    memory::primitive_desc dst_pd = pool_fwd_prim_desc.dst_primitive_desc();

    MklDnnShape output_mkl_shape;
    output_mkl_shape.SetMklTensor(true);
    output_mkl_shape.SetMklLayout(&dst_pd);
    output_mkl_shape.SetElemType(MklDnnType<T>());
    output_mkl_shape.SetTfLayout(output_dims_mkl_order.size(),
                                 output_dims_mkl_order, output_tf_format);
    TensorShape output_tf_shape;

    // only allocate enough space for the elements we need.
    output_tf_shape.AddDim(this->GetNumTElements(dst_pd));
    AllocateOutputSetMklShape(context, kOutputTensorIndexOutput, output_tensor,
                              output_tf_shape, output_mkl_shape);
    CHECK_NOTNULL(*output_tensor);
  }

  void SanityCheckInput(OpKernelContext* context, const Tensor& input_tensor,
                        const MklDnnShape& input_mkl_shape) {
    if (!input_mkl_shape.IsMklTensor()) {
      OP_REQUIRES(context, input_tensor.dims() == 4 || input_tensor.dims() == 5,
                  errors::InvalidArgument("Input must be 4 or 5-dimensional"));
    } else {
      OP_REQUIRES(
          context, input_mkl_shape.GetDimension() == 4 ||
                       input_mkl_shape.GetDimension() == 5,
          errors::InvalidArgument("Input shape must be 4 or 5-dimensional"));
    }
  }
  // .Input("value: T")
  // .Output("output: T")
  const int kInputTensorIndexInput = 0;
  const int kOutputTensorIndexOutput = 0;
};  // MklPoolingForwardBaseOp

template <class T>
class MklPoolingBackwardOpBase : public MklPoolingOpBase<T> {
 public:
  explicit MklPoolingBackwardOpBase<T>(OpKernelConstruction* context)
      : MklPoolingOpBase<T>(context) {}
  void Compute(OpKernelContext* context) override = 0;

 protected:
  const int kOutputTensorIndexOutput = 0;

  void AllocateOutputTensor(
      OpKernelContext* context,
      const pooling_backward::primitive_desc& pool_bkwd_prim_desc,
      const memory::dims output_dims_mkl_order,
      const memory::format& output_tf_format, Tensor** output_tensor) {
    CHECK_NOTNULL(output_tensor);
    memory::primitive_desc dst_pd =
        pool_bkwd_prim_desc.diff_src_primitive_desc();
    MklDnnShape output_mkl_shape;
    output_mkl_shape.SetMklTensor(true);
    output_mkl_shape.SetMklLayout(&dst_pd);
    output_mkl_shape.SetElemType(MklDnnType<T>());
    output_mkl_shape.SetTfLayout(output_dims_mkl_order.size(),
                                 output_dims_mkl_order, output_tf_format);

    TensorShape output_tf_shape;
    output_tf_shape.AddDim(this->GetNumTElements(dst_pd));
    AllocateOutputSetMklShape(context, kOutputTensorIndexOutput, output_tensor,
                              output_tf_shape, output_mkl_shape);
    CHECK_NOTNULL(*output_tensor);
  }

  memory::desc ConfigureInputGradient(
      const MklDnnShape& input_gradient_mkl_shape,
      const Tensor& input_gradient_tensor,
      MklDnnData<T>* input_gradient_dnn_data,
      const memory::desc& original_output_md) {
    // Configure the gradient as is
    memory::desc original_input_grad_md =
        input_gradient_mkl_shape.IsMklTensor()
            ? input_gradient_mkl_shape.GetMklLayout()
            : memory::desc(
                  (this->ksize_.size() == 4)
                      ? TFShapeToMklDnnDimsInNCHW(input_gradient_tensor.shape(),
                                                  this->data_format_tf_)
                      : TFShapeToMklDnnDimsInNCDHW(
                            input_gradient_tensor.shape(),
                            this->data_format_tf_),
                  MklDnnType<T>(), this->data_format_mkldnn_);

    input_gradient_dnn_data->SetUsrMem(original_input_grad_md,
                                       &input_gradient_tensor);

    // Check to see if input grad diff dst is in the right format
    // Create a new memory descriptor with the same shape as the
    // original, but the format of the other tensors.
    memory::format original_output_format =
        static_cast<memory::format>(original_output_md.data.format);
    bool grad_reorder_needed =
        input_gradient_dnn_data->IsReorderNeeded(original_output_format);
    memory::dims diff_dst_dims =
        input_gradient_mkl_shape.IsMklTensor()
            ? input_gradient_mkl_shape.GetSizesAsMklDnnDims()
            : TFShapeToMklDnnDimsInNCHW(input_gradient_tensor.shape(),
                                        this->data_format_tf_);
    memory::desc target_diff_dst_md =
        memory::desc(diff_dst_dims, MklDnnType<T>(), original_output_format);

    return grad_reorder_needed ? target_diff_dst_md : original_input_grad_md;
  }
};
#endif  // INTEL_MKL_ML_ONLY

//-------------------------------------------------------------------
// Utility functions

typedef struct {
  size_t in_dim;
  size_t in_sizes[4];
  size_t in_strides[4];
  size_t out_sizes[4];
  size_t out_strides[4];
  int in_offset[4];
  size_t kernel_stride[2];
  size_t kernel_size[2];
} MklPoolingOpParams;

// Transfers the right parameters for pooling to the op parameters
// Updates context->status if there is an invalid input.
void ExtractMklOpParams(OpKernelContext* context, TensorFormat data_format,
                        const MklPoolParameters& params,
                        MklPoolingOpParams* mkl_params);
}  // namespace tensorflow

#endif  // INTEL_MKL
#endif  // TENSORFLOW_CORE_KERNELS_MKL_POOLING_OPS_COMMON_H_
