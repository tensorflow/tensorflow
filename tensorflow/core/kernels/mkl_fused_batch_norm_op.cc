/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#ifdef INTEL_MKL
#include "mkldnn.hpp"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/util/tensor_format.h"

using mkldnn::batch_normalization_backward;
using mkldnn::batch_normalization_forward;
using mkldnn::prop_kind;
using mkldnn::stream;
using mkldnn::use_global_stats;
using mkldnn::use_scale_shift;

namespace tensorflow {
using CPUDevice = Eigen::ThreadPoolDevice;

struct MklBatchNormFwdParams {
  memory::dims src_dims;
  int depth;
  float eps;
  bool training;

  MklBatchNormFwdParams(const memory::dims& src_dims, int depth, float eps,
                        bool training)
      : src_dims(src_dims), depth(depth), eps(eps), training(training) {}
};

template <typename T, typename U>
class MklFusedBatchNormFwdPrimitive : public MklPrimitive {
 public:
  explicit MklFusedBatchNormFwdPrimitive(const MklBatchNormFwdParams& fwdParams)
      : cpu_engine_(engine::cpu, 0) {
    context_.fwd_stream.reset(new mkldnn::stream(mkldnn::stream::kind::eager));
    if (context_.bn_fwd == nullptr) Setup(fwdParams);
  }

  ~MklFusedBatchNormFwdPrimitive() {}

  // BatchNormalization forward execute
  //   src_data:     input data buffer of src
  //   weights_data: input data buffer of weights
  //   dst_data:     output data buffer of dst
  //   mean_data:     output data buffer of means
  //   variance_data: output data buffer of variances
  void Execute(const T* src_data, const U* weights_data, T* dst_data,
               U* mean_data, U* variance_data) {
    context_.src_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(src_data)));
    context_.dst_mem->set_data_handle(static_cast<void*>(dst_data));

    if (context_.flags & use_scale_shift)
      context_.weights_mem->set_data_handle(
          static_cast<void*>(const_cast<U*>(weights_data)));

    if ((context_.pkind == prop_kind::forward_training) ||
        (context_.flags & use_global_stats)) {
      context_.mean_mem->set_data_handle(static_cast<void*>(mean_data));
      context_.variance_mem->set_data_handle(static_cast<void*>(variance_data));
    }

    // execution
    context_.fwd_stream->submit(context_.fwd_primitives);

    context_.src_mem->set_data_handle(DummyData);
    context_.dst_mem->set_data_handle(DummyData);

    if (context_.flags & use_scale_shift)
      context_.weights_mem->set_data_handle(DummyData);

    if ((context_.pkind == prop_kind::forward_training) ||
        (context_.flags & use_global_stats)) {
      context_.mean_mem->set_data_handle(DummyData);
      context_.variance_mem->set_data_handle(DummyData);
    }
  }

  memory::primitive_desc GetDstPd() const {
    return (*context_.dst_mem).get_primitive_desc();
  }

  mkldnn_memory_format_t GetSrcFmt() const {
    return (*context_.src_mem).get_primitive_desc().desc().data.format;
  }

  mkldnn_memory_format_t GetDstFmt() const {
    return (*context_.dst_mem).get_primitive_desc().desc().data.format;
  }

 private:
  // Primitive reuse context for BatchNorm fwd op
  struct BatchNormFwdContext {
    // flags indict if it is training or inference mode
    int64 flags;

    // algorithm
    mkldnn::prop_kind pkind;

    // Mkldnn Memory
    std::shared_ptr<mkldnn::memory> src_mem;
    std::shared_ptr<mkldnn::memory> weights_mem;
    std::shared_ptr<mkldnn::memory> dst_mem;
    std::shared_ptr<mkldnn::memory> mean_mem;
    std::shared_ptr<mkldnn::memory> variance_mem;

    // BatchNorm forward primitive
    std::shared_ptr<mkldnn::primitive> bn_fwd;
    std::shared_ptr<mkldnn::stream> fwd_stream;
    std::vector<mkldnn::primitive> fwd_primitives;

    BatchNormFwdContext()
        : flags(0),
          pkind(mkldnn::forward_training),
          src_mem(nullptr),
          weights_mem(nullptr),
          dst_mem(nullptr),
          mean_mem(nullptr),
          variance_mem(nullptr),
          bn_fwd(nullptr),
          fwd_stream(nullptr) {}
  };

  void Setup(const MklBatchNormFwdParams& fwdParams) {
    context_.flags = fwdParams.training ? use_scale_shift
                                        : (use_scale_shift | use_global_stats);
    context_.pkind = fwdParams.training ? prop_kind::forward_training
                                        : prop_kind::forward_scoring;

    // memory desc
    auto src_md = memory::desc({fwdParams.src_dims}, MklDnnType<T>(),
                               get_desired_format(fwdParams.src_dims[1]));

    // fwd desc & primitive desc
    auto fwd_desc = batch_normalization_forward::desc(
        context_.pkind, src_md, fwdParams.eps, context_.flags);
    auto fwd_pd =
        batch_normalization_forward::primitive_desc(fwd_desc, cpu_engine_);

    // memory primitive
    context_.src_mem.reset(new memory({src_md, cpu_engine_}, DummyData));
    context_.dst_mem.reset(new memory(fwd_pd.dst_primitive_desc(), DummyData));

    if (context_.flags & use_scale_shift) {
      auto weights_desc = memory::desc({2, fwdParams.depth}, MklDnnType<U>(),
                                       memory::format::nc);
      context_.weights_mem.reset(
          new memory({weights_desc, cpu_engine_}, DummyData));
    }

    if (fwdParams.training || (context_.flags & use_global_stats)) {
      auto mean_desc = memory::desc({1, fwdParams.depth}, MklDnnType<U>(),
                                    memory::format::nc);
      context_.mean_mem.reset(new memory({mean_desc, cpu_engine_}, DummyData));

      auto variance_desc =
          memory::desc({1, fwdParams.depth}, MklDnnType<U>(), memory::nc);
      context_.variance_mem.reset(
          new memory({variance_desc, cpu_engine_}, DummyData));
    }

    // BatchNorm forward primitive
    if (!fwdParams.training && !(context_.flags & use_global_stats)) {
      if ((context_.flags & use_scale_shift) && mkldnn_use_scaleshift) {
        context_.bn_fwd.reset(new batch_normalization_forward(
            fwd_pd, *context_.src_mem, *context_.weights_mem,
            *context_.dst_mem));
      } else {
        context_.bn_fwd.reset(new batch_normalization_forward(
            fwd_pd, *context_.src_mem, *context_.dst_mem));
      }
    } else if (context_.flags & use_global_stats) {
      if ((context_.flags & use_scale_shift) && mkldnn_use_scaleshift) {
        context_.bn_fwd.reset(new batch_normalization_forward(
            fwd_pd, *context_.src_mem, (const primitive::at)*context_.mean_mem,
            (const primitive::at)*context_.variance_mem, *context_.weights_mem,
            *context_.dst_mem));
      } else {
        context_.bn_fwd.reset(new batch_normalization_forward(
            fwd_pd, *context_.src_mem, (const primitive::at)*context_.mean_mem,
            (const primitive::at)*context_.variance_mem, *context_.dst_mem));
      }
    } else {
      if ((context_.flags & use_scale_shift) && mkldnn_use_scaleshift) {
        context_.bn_fwd.reset(new batch_normalization_forward(
            fwd_pd, *context_.src_mem, *context_.weights_mem, *context_.dst_mem,
            *context_.mean_mem, *context_.variance_mem));
      } else {
        context_.bn_fwd.reset(new batch_normalization_forward(
            fwd_pd, *context_.src_mem, *context_.dst_mem, *context_.mean_mem,
            *context_.variance_mem));
      }
    }

    context_.fwd_primitives.push_back(*context_.bn_fwd);
  }

  mkldnn::memory::desc get_desc_data(const mkldnn::memory& m) const {
    return m.get_primitive_desc().desc().data;
  }

  struct BatchNormFwdContext context_;
  engine cpu_engine_;
};

template <typename T, typename U>
class MklFusedBatchNormFwdPrimitiveFactory : public MklPrimitiveFactory<T> {
 public:
  static MklFusedBatchNormFwdPrimitive<T, U>* Get(
      const MklBatchNormFwdParams& fwdParams) {
    auto bn_fwd = static_cast<MklFusedBatchNormFwdPrimitive<T, U>*>(
        MklFusedBatchNormFwdPrimitiveFactory<T, U>::GetInstance()
            .GetBatchNormFwd(fwdParams));

    if (bn_fwd == nullptr) {
      bn_fwd = new MklFusedBatchNormFwdPrimitive<T, U>(fwdParams);
      MklFusedBatchNormFwdPrimitiveFactory<T, U>::GetInstance().SetBatchNormFwd(
          fwdParams, bn_fwd);
    }
    return bn_fwd;
  }

  static MklFusedBatchNormFwdPrimitiveFactory& GetInstance() {
    static MklFusedBatchNormFwdPrimitiveFactory instance_;
    return instance_;
  }

 private:
  MklFusedBatchNormFwdPrimitiveFactory() {}
  ~MklFusedBatchNormFwdPrimitiveFactory() {}

  static string CreateKey(const MklBatchNormFwdParams& fwdParams) {
    string prefix = "bn_fwd";
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(prefix);
    key_creator.AddAsKey(fwdParams.src_dims);
    key_creator.AddAsKey<int>(fwdParams.depth);
    key_creator.AddAsKey<float>(fwdParams.eps);
    key_creator.AddAsKey<bool>(fwdParams.training);
    key_creator.AddAsKey(typeid(T).name());
    key_creator.AddAsKey(typeid(U).name());
    return key_creator.GetKey();
  }

  MklPrimitive* GetBatchNormFwd(const MklBatchNormFwdParams& fwdParams) {
    string key = CreateKey(fwdParams);
    return this->GetOp(key);
  }

  void SetBatchNormFwd(const MklBatchNormFwdParams& fwdParams,
                       MklPrimitive* op) {
    string key = CreateKey(fwdParams);
    this->SetOp(key, op);
  }
};

struct MklBatchNormBwdParams {
  memory::dims src_dims;
  memory::dims diff_dst_dims;
  int depth;
  float eps;
  bool training;

  MklBatchNormBwdParams(memory::dims src_dims, memory::dims diff_dst_dims,
                        int depth, float eps, bool training)
      : src_dims(src_dims),
        diff_dst_dims(diff_dst_dims),
        depth(depth),
        eps(eps),
        training(training) {}
};

template <typename T, typename U>
class MklFusedBatchNormBwdPrimitive : public MklPrimitive {
 public:
  explicit MklFusedBatchNormBwdPrimitive(const MklBatchNormBwdParams& bwdParams)
      : cpu_engine_(engine::cpu, 0) {
    context_.bwd_stream.reset(new mkldnn::stream(mkldnn::stream::kind::eager));
    if (context_.bn_bwd == nullptr) Setup(bwdParams);
  }

  ~MklFusedBatchNormBwdPrimitive() {}

  // BatchNormalization backward execute
  //   src_data:       input data buffer of src
  //   mean_data:      input data buffer of mean
  //   variance_data:  input data buffer of variance
  //   diff_dst_data:  input data buffer of diff_dst
  //   weights_data:   input data buffer of weights
  //   diff_src_data:      output data buffer of diff_src
  //   diff_weights_data:  output data buffer of diff_weights
  void Execute(const T* src_data, const U* mean_data, const U* variance_data,
               const T* diff_dst_data, const U* weights_data, T* diff_src_data,
               U* diff_weights_data) {
    context_.src_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(src_data)));
    context_.mean_mem->set_data_handle(
        static_cast<void*>(const_cast<U*>(mean_data)));
    context_.variance_mem->set_data_handle(
        static_cast<void*>(const_cast<U*>(variance_data)));
    context_.diff_dst_mem->set_data_handle(
        static_cast<void*>(const_cast<T*>(diff_dst_data)));

    // TODO: type for weights?
    if (context_.flags & use_scale_shift) {
      context_.weights_mem->set_data_handle(
          static_cast<void*>(const_cast<U*>(weights_data)));
      context_.diff_weights_mem->set_data_handle(
          static_cast<void*>(diff_weights_data));
    }

    context_.diff_src_mem->set_data_handle(static_cast<void*>(diff_src_data));

    // execution
    context_.bwd_stream->submit(context_.bwd_primitives);

    context_.src_mem->set_data_handle(DummyData);
    context_.mean_mem->set_data_handle(DummyData);
    context_.variance_mem->set_data_handle(DummyData);
    context_.diff_dst_mem->set_data_handle(DummyData);
    if (context_.flags & use_scale_shift) {
      context_.weights_mem->set_data_handle(DummyData);
      context_.diff_weights_mem->set_data_handle(DummyData);
    }
    context_.diff_src_mem->set_data_handle(DummyData);
  }

  mkldnn_memory_format_t GetSrcFmt() {
    return (*context_.src_mem).get_primitive_desc().desc().data.format;
  }

  mkldnn_memory_format_t GetDiffDstFmt() {
    return (*context_.diff_dst_mem).get_primitive_desc().desc().data.format;
  }

  memory::primitive_desc GetDiffSrcPd() {
    return (*context_.diff_src_mem).get_primitive_desc();
  }

 private:
  struct BatchNormBwdContext {
    // Flags to indicate whether it is training or inference
    int64 flags;

    // MKLDNN memory
    std::shared_ptr<mkldnn::memory> src_mem;
    std::shared_ptr<mkldnn::memory> mean_mem;
    std::shared_ptr<mkldnn::memory> variance_mem;
    std::shared_ptr<mkldnn::memory> diff_dst_mem;
    std::shared_ptr<mkldnn::memory> weights_mem;
    std::shared_ptr<mkldnn::memory> diff_weights_mem;
    std::shared_ptr<mkldnn::memory> diff_src_mem;

    // Batch Norm primitive
    std::shared_ptr<mkldnn::primitive> bn_bwd;
    std::vector<mkldnn::primitive> bwd_primitives;
    std::shared_ptr<mkldnn::stream> bwd_stream;

    BatchNormBwdContext()
        : src_mem(nullptr),
          mean_mem(nullptr),
          variance_mem(nullptr),
          diff_dst_mem(nullptr),
          weights_mem(nullptr),
          diff_weights_mem(nullptr),
          diff_src_mem(nullptr),
          bwd_stream(nullptr) {}
  };

  void Setup(const MklBatchNormBwdParams& bwdParams) {
    context_.flags = bwdParams.training ? use_scale_shift
                                        : (use_scale_shift | use_global_stats);

    // memory desc
    auto src_md = memory::desc({bwdParams.src_dims}, MklDnnType<T>(),
                               get_desired_format(bwdParams.src_dims[1]));
    auto diff_dst_md =
        memory::desc({bwdParams.diff_dst_dims}, MklDnnType<T>(),
                     get_desired_format(bwdParams.diff_dst_dims[1]));
    auto variance_desc =
        memory::desc({1, bwdParams.depth}, MklDnnType<U>(), memory::nc);
    auto mean_desc =
        memory::desc({1, bwdParams.depth}, MklDnnType<U>(), memory::format::nc);
    auto weights_desc =
        memory::desc({2, bwdParams.depth}, MklDnnType<U>(), memory::format::nc);
    auto diff_weights_desc = weights_desc;

    // fwd desc & primitive desc
    auto fwd_desc = batch_normalization_forward::desc(
        prop_kind::forward_training, src_md, bwdParams.eps,
        bwdParams.training ? use_scale_shift
                           : (use_scale_shift | use_global_stats));
    auto fwd_pd =
        batch_normalization_forward::primitive_desc(fwd_desc, cpu_engine_);

    // BatchNorm backward primtive
    //
    // For inference, specify use_global_stats
    //   1. on fwd propagation, use mean and variance provided as inputs.
    //   2. on bwd propagation, mean and variance are considered as constants.
    //      Thus, reduce the amount of MKL computation.
    auto bwd_desc = batch_normalization_backward::desc(
        prop_kind::backward, diff_dst_md, src_md, bwdParams.eps,
        bwdParams.training ? use_scale_shift
                           : (use_scale_shift | use_global_stats));
    auto bn_bwd_pd = batch_normalization_backward::primitive_desc(
        bwd_desc, cpu_engine_, fwd_pd);

    // memory primitive
    context_.src_mem.reset(new memory({src_md, cpu_engine_}, DummyData));
    context_.diff_dst_mem.reset(
        new memory({diff_dst_md, cpu_engine_}, DummyData));
    context_.variance_mem.reset(
        new memory({variance_desc, cpu_engine_}, DummyData));
    context_.mean_mem.reset(new memory({mean_desc, cpu_engine_}, DummyData));
    context_.weights_mem.reset(
        new memory({weights_desc, cpu_engine_}, DummyData));
    context_.diff_weights_mem.reset(
        new memory({diff_weights_desc, cpu_engine_}, DummyData));
    context_.diff_src_mem.reset(new memory({src_md, cpu_engine_}, DummyData));

    context_.bn_bwd.reset(new batch_normalization_backward(
        bn_bwd_pd, *context_.src_mem, *context_.mean_mem,
        *context_.variance_mem, *context_.diff_dst_mem, *context_.weights_mem,
        *context_.diff_src_mem, *context_.diff_weights_mem));
    context_.bwd_primitives.push_back(*context_.bn_bwd);
  }

  struct BatchNormBwdContext context_;
  engine cpu_engine_;
};

template <typename T, typename U>
class MklFusedBatchNormBwdPrimitiveFactory : public MklPrimitiveFactory<T> {
 public:
  static MklFusedBatchNormBwdPrimitive<T, U>* Get(
      const MklBatchNormBwdParams& bwdParams) {
    auto bn_bwd = static_cast<MklFusedBatchNormBwdPrimitive<T, U>*>(
        MklFusedBatchNormBwdPrimitiveFactory<T, U>::GetInstance()
            .GetBatchNormBwd(bwdParams));
    if (bn_bwd == nullptr) {
      bn_bwd = new MklFusedBatchNormBwdPrimitive<T, U>(bwdParams);
      MklFusedBatchNormBwdPrimitiveFactory<T, U>::GetInstance().SetBatchNormBwd(
          bwdParams, bn_bwd);
    }
    return bn_bwd;
  }

  static MklFusedBatchNormBwdPrimitiveFactory& GetInstance() {
    static MklFusedBatchNormBwdPrimitiveFactory instance_;
    return instance_;
  }

 private:
  MklFusedBatchNormBwdPrimitiveFactory() {}
  ~MklFusedBatchNormBwdPrimitiveFactory() {}

  static string CreateKey(const MklBatchNormBwdParams& bwdParams) {
    string prefix = "bn_bwd";
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(prefix);
    key_creator.AddAsKey(bwdParams.src_dims);
    key_creator.AddAsKey(bwdParams.diff_dst_dims);
    key_creator.AddAsKey<int>(bwdParams.depth);
    key_creator.AddAsKey<float>(bwdParams.eps);
    key_creator.AddAsKey<bool>(bwdParams.training);
    key_creator.AddAsKey(typeid(T).name());
    key_creator.AddAsKey(typeid(U).name());
    return key_creator.GetKey();
  }

  MklPrimitive* GetBatchNormBwd(const MklBatchNormBwdParams& bwdParams) {
    string key = CreateKey(bwdParams);
    return this->GetOp(key);
  }

  void SetBatchNormBwd(const MklBatchNormBwdParams& bwdParams,
                       MklPrimitive* op) {
    string key = CreateKey(bwdParams);
    this->SetOp(key, op);
  }
};

template <typename Device, typename T, typename U>
class MklFusedBatchNormOp : public OpKernel {
 public:
  explicit MklFusedBatchNormOp(OpKernelConstruction* context)
      : OpKernel(context) {
    float epsilon;
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon));
    epsilon_ = epsilon;
    string tensor_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &tensor_format));
    OP_REQUIRES(context, FormatFromString(tensor_format, &tensor_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("is_training", &is_training_));
  }

  void Compute(OpKernelContext* context) override {
    try {
      const size_t kSrcIndex = 0;       // index of src input tensor
      const size_t kScaleIndex = 1;     // index of scale tensor
      const size_t kShiftIndex = 2;     // index of shift tensor
      const size_t kMeanIndex = 3;      // index of est_mean tensor
      const size_t kVarianceIndex = 4;  // index of est_variance tensor

      const Tensor& src_tensor = MklGetInput(context, kSrcIndex);
      const Tensor& scale_tensor = MklGetInput(context, kScaleIndex);
      const Tensor& shift_tensor = MklGetInput(context, kShiftIndex);
      const Tensor& est_mean_tensor = MklGetInput(context, kMeanIndex);
      const Tensor& est_variance_tensor = MklGetInput(context, kVarianceIndex);

      TensorShape tf_shape_src;
      MklDnnShape dnn_shape_src;
      GetMklShape(context, kSrcIndex, &dnn_shape_src);

      if (dnn_shape_src.IsMklTensor()) {
        tf_shape_src = dnn_shape_src.GetTfShape();
        OP_REQUIRES(context, dnn_shape_src.GetDimension() == 4,
                    errors::InvalidArgument("input must be 4-dimensional",
                                            src_tensor.shape().DebugString()));
      } else {
        tf_shape_src = src_tensor.shape();
        OP_REQUIRES(context, src_tensor.dims() == 4,
                    errors::InvalidArgument("input must be 4-dimensional",
                                            src_tensor.shape().DebugString()));
      }
      OP_REQUIRES(context, scale_tensor.dims() == 1,
                  errors::InvalidArgument("scale must be 1-dimensional",
                                          scale_tensor.shape().DebugString()));
      OP_REQUIRES(context, shift_tensor.dims() == 1,
                  errors::InvalidArgument("offset must be 1-dimensional",
                                          shift_tensor.shape().DebugString()));
      OP_REQUIRES(
          context, est_mean_tensor.dims() == 1,
          errors::InvalidArgument("estimated_mean must be 1-dimensional",
                                  est_mean_tensor.shape().DebugString()));
      OP_REQUIRES(
          context, est_variance_tensor.dims() == 1,
          errors::InvalidArgument("estimated_variance must be 1-dimensional",
                                  est_variance_tensor.shape().DebugString()));

      if (is_training_) {
        OP_REQUIRES(
            context, est_mean_tensor.dim_size(0) == 0,
            errors::InvalidArgument("estimated_mean must be empty for training",
                                    est_mean_tensor.shape().DebugString()));
        OP_REQUIRES(context, est_variance_tensor.dim_size(0) == 0,
                    errors::InvalidArgument(
                        "estimated_variance must be empty for training",
                        est_variance_tensor.shape().DebugString()));
      }

      // special case: input with 0 element and 0 batch size
      Tensor* dst_tensor = nullptr;
      if (tf_shape_src.num_elements() == 0) {
        HandleEmptyInput(context, tf_shape_src, scale_tensor.shape(),
                         &dst_tensor);
        return;
      }

      if (dnn_shape_src.IsMklTensor())
        depth_ = dnn_shape_src.DimSize(MklDnnDims::Dim_C);
      else
        ExtractParams(context);

      // Indices of output tensors
      const size_t kDstIndex = 0;

      // allocate 4 output TF tensors
      Tensor* batch_mean_tensor = nullptr;
      Tensor* batch_variance_tensor = nullptr;
      Tensor* saved_mean_tensor = nullptr;
      Tensor* saved_variance_tensor = nullptr;
      AllocateTFOutputs(context, scale_tensor.shape(), &batch_mean_tensor,
                        &batch_variance_tensor, &saved_mean_tensor,
                        &saved_variance_tensor);

      if (is_training_)
        SetMeanVariance(*batch_mean_tensor, *batch_variance_tensor);
      else
        SetMeanVariance(est_mean_tensor, est_variance_tensor);

      MklDnnData<T> src(&cpu_engine);
      MklDnnData<U> weights(&cpu_engine);

      memory::format format_m;
      if (dnn_shape_src.IsMklTensor()) {
        if (dnn_shape_src.IsTensorInNCHWFormat()) {
          format_m = memory::format::nchw;
        } else {
          format_m = memory::format::nhwc;
        }
      } else {
        format_m = TFDataFormatToMklDnnDataFormat(tensor_format_);
      }

      // set src primitive
      memory::dims src_dims =
          dnn_shape_src.IsMklTensor()
              ? dnn_shape_src.GetSizesAsMklDnnDims()
              : TFShapeToMklDnnDimsInNCHW(src_tensor.shape(), tensor_format_);

      auto src_md = dnn_shape_src.IsMklTensor()
                        ? dnn_shape_src.GetMklLayout()
                        : memory::desc(src_dims, MklDnnType<T>(), format_m);

      // MKL-DNN packs scale & shift as "weights":
      // <scale>...<scale><shift>...<shift>
      weights.AllocateBuffer(2 * depth_ * sizeof(U));
      U* weights_data = reinterpret_cast<U*>(weights.GetAllocatedBuffer());
      const U* scale_tf = scale_tensor.flat<U>().data();
      const U* shift_tf = shift_tensor.flat<U>().data();

      std::memcpy(weights_data, scale_tf, depth_ * sizeof(U));
      std::memcpy(weights_data + depth_, shift_tf, depth_ * sizeof(U));
      char* saved_mean_data_tf =
          reinterpret_cast<char*>(saved_mean_tensor->flat<U>().data());
      std::memcpy(saved_mean_data_tf, reinterpret_cast<char*>(mean_values_),
                  depth_ * sizeof(U));

      char* saved_variance_data_tf =
          reinterpret_cast<char*>(saved_variance_tensor->flat<U>().data());
      std::memcpy(saved_variance_data_tf,
                  reinterpret_cast<char*>(variance_values_),
                  depth_ * sizeof(U));

      // get batchnorm op from the pool
      MklBatchNormFwdParams fwdParams(src_dims, depth_, epsilon_, is_training_);
      MklFusedBatchNormFwdPrimitive<T, U>* bn_fwd =
          MklFusedBatchNormFwdPrimitiveFactory<T, U>::Get(fwdParams);

      // check if reorder is needed for src, weights, mean, variance
      const T* src_data = src_tensor.flat<T>().data();
      if (src_md.data.format != bn_fwd->GetSrcFmt()) {
        src.SetUsrMem(src_md, &src_tensor);
        auto src_target = memory::primitive_desc(
            {{src_dims},
             MklDnnType<T>(),
             static_cast<memory::format>(bn_fwd->GetSrcFmt())},
            cpu_engine);
        src.CheckReorderToOpMem(src_target);
        src_data = const_cast<T*>(
            reinterpret_cast<T*>(src.GetOpMem().get_data_handle()));
      }

      // allocate output (dst) tensor; always set it as MKL-DNN layout
      MklDnnShape dnn_shape_dst;
      TensorShape tf_shape_dst;
      dnn_shape_dst.SetMklTensor(true);
      auto dst_pd = bn_fwd->GetDstPd();
      dnn_shape_dst.SetMklLayout(&dst_pd);
      dnn_shape_dst.SetElemType(MklDnnType<T>());
      auto ndims = dnn_shape_src.IsMklTensor() ? dnn_shape_src.GetDimension()
                                               : src_tensor.shape().dims();
      dnn_shape_dst.SetTfLayout(ndims, src_dims, format_m);
      tf_shape_dst.AddDim(dst_pd.get_size() / sizeof(T));
      AllocateOutputSetMklShape(context, kDstIndex, &dst_tensor, tf_shape_dst,
                                dnn_shape_dst);

      U* weights_op_data = weights_data;
      U* mean_op_data = saved_mean_tensor->flat<U>().data();
      U* variance_op_data = saved_variance_tensor->flat<U>().data();
      T* dst_data = dst_tensor->flat<T>().data();

      // execution
      bn_fwd->Execute(src_data, weights_op_data, dst_data, mean_op_data,
                      variance_op_data);

      // copy batch_mean data
      U* batch_mean_data_tf = batch_mean_tensor->flat<U>().data();
      std::memcpy(reinterpret_cast<char*>(batch_mean_data_tf),
                  reinterpret_cast<char*>(saved_mean_data_tf),
                  depth_ * sizeof(U));
      // TODO(yli135): OpMem is same as usr mem since
      // since its format is hard-coded as nc when primitive is created.

      // copy batch_variance data with Bessel's correction
      float adjust_factor = 1.0;
      if (is_training_) {
        size_t orig_size = src_dims[0] * src_dims[2] * src_dims[3];
        size_t adjust_size = orig_size - 1;
        adjust_factor = (static_cast<float>(orig_size)) / adjust_size;
      }

      auto variance_data = reinterpret_cast<U*>(saved_variance_data_tf);
      auto batch_variance_data = batch_variance_tensor->flat<U>().data();
      if (is_training_) {
        for (int k = 0; k < depth_; k++) {
          batch_variance_data[k] =
              variance_data[k] * static_cast<U>(adjust_factor);
        }
      } else {
        std::memcpy(batch_variance_data, variance_data, depth_ * sizeof(U));
      }
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
  float epsilon_;
  TensorFormat tensor_format_;
  bool is_training_;
  U* mean_values_;
  U* variance_values_;
  size_t depth_;  // batch normalization is done for per channel.
  engine cpu_engine = engine(engine::cpu, 0);

  void ExtractParams(OpKernelContext* context) {
    const Tensor& input = MklGetInput(context, 0);
    depth_ = static_cast<int>(GetTensorDim(input, tensor_format_, 'C'));
  }

  void SetMeanVariance(const Tensor& mean, const Tensor& variance) {
    mean_values_ = reinterpret_cast<U*>(const_cast<U*>(mean.flat<U>().data()));
    variance_values_ =
        reinterpret_cast<U*>(const_cast<U*>(variance.flat<U>().data()));
  }

  void HandleEmptyInput(OpKernelContext* context, TensorShape tf_shape_src,
                        TensorShape tf_shape_scale, Tensor** dst_tensor) {
    CHECK_NOTNULL(dst_tensor);

    const size_t kDstIndex = 0;
    MklDnnShape dnn_shape_dst;
    dnn_shape_dst.SetMklTensor(false);
    AllocateOutputSetMklShape(context, kDstIndex, dst_tensor, tf_shape_src,
                              dnn_shape_dst);
    CHECK_NOTNULL(*dst_tensor);
    memset(const_cast<char*>((*dst_tensor)->tensor_data().data()), 0,
           (*dst_tensor)->tensor_data().size());

    Tensor* batch_mean_tensor = nullptr;
    Tensor* batch_variance_tensor = nullptr;
    Tensor* saved_mean_tensor = nullptr;
    Tensor* saved_variance_tensor = nullptr;
    AllocateTFOutputs(context, tf_shape_scale, &batch_mean_tensor,
                      &batch_variance_tensor, &saved_mean_tensor,
                      &saved_variance_tensor);
  }

  void AllocateTFOutputs(OpKernelContext* context, TensorShape tf_shape_scale,
                         Tensor** batch_mean_tensor,
                         Tensor** batch_variance_tensor,
                         Tensor** saved_mean_tensor,
                         Tensor** saved_variance_tensor) {
    CHECK_NOTNULL(batch_mean_tensor);
    CHECK_NOTNULL(batch_variance_tensor);
    CHECK_NOTNULL(saved_mean_tensor);
    CHECK_NOTNULL(saved_variance_tensor);

    const size_t kBatchMeanIndex = 1;
    const size_t kBatchVarianceIndex = 2;
    const size_t kSavedMeanIndex = 3;
    const size_t kSavedVarianceIndex = 4;

    // allocate batch mean output tensor
    MklDnnShape mkl_shape_batch_mean;
    mkl_shape_batch_mean.SetMklTensor(false);
    AllocateOutputSetMklShape(context, kBatchMeanIndex, batch_mean_tensor,
                              tf_shape_scale, mkl_shape_batch_mean);
    CHECK_NOTNULL(*batch_mean_tensor);
    // set NAN mean value in case of empty input tensor
    int num_elements = tf_shape_scale.num_elements();
    auto batch_mean_data = (*batch_mean_tensor)->flat<U>().data();
    std::fill_n(batch_mean_data, num_elements, static_cast<U>(NAN));

    // allocate batch variance output tensor
    MklDnnShape mkl_shape_batch_variance;
    mkl_shape_batch_variance.SetMklTensor(false);
    AllocateOutputSetMklShape(context, kBatchVarianceIndex,
                              batch_variance_tensor, tf_shape_scale,
                              mkl_shape_batch_variance);
    CHECK_NOTNULL(*batch_variance_tensor);
    // set NAN variance value in case of empty input tensor
    auto batch_variance_data = (*batch_variance_tensor)->flat<U>().data();
    std::fill_n(batch_variance_data, num_elements, static_cast<U>(NAN));

    // Mean and variance (without Bessel's correction) saved for backward
    // computation to serve as pre-computed mean and variance.
    MklDnnShape mkl_shape_saved_mean;
    mkl_shape_saved_mean.SetMklTensor(false);
    AllocateOutputSetMklShape(context, kSavedMeanIndex, saved_mean_tensor,
                              tf_shape_scale, mkl_shape_saved_mean);
    CHECK_NOTNULL(*saved_mean_tensor);
    // set NAN mean value in case of empty input tensor
    auto saved_mean_data = (*saved_mean_tensor)->flat<U>().data();
    std::fill_n(saved_mean_data, num_elements, static_cast<U>(NAN));

    MklDnnShape mkl_shape_saved_variance;
    mkl_shape_saved_variance.SetMklTensor(false);
    AllocateOutputSetMklShape(context, kSavedVarianceIndex,
                              saved_variance_tensor, tf_shape_scale,
                              mkl_shape_saved_variance);
    CHECK_NOTNULL(*saved_variance_tensor);
    // set NAN variance value in case of empty input tensor
    auto saved_variance_data = (*saved_variance_tensor)->flat<U>().data();
    std::fill_n(saved_variance_data, num_elements, static_cast<U>(NAN));
  }
};

template <typename Device, typename T, typename U>
class MklFusedBatchNormGradOp : public OpKernel {
 public:
  explicit MklFusedBatchNormGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    float epsilon;
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon));
    epsilon_ = epsilon;
    string tensor_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &tensor_format));
    OP_REQUIRES(context, FormatFromString(tensor_format, &tensor_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("is_training", &is_training_));
  }

  void Compute(OpKernelContext* context) override {
    try {
      const size_t kDiffDstIndex = 0;   // index of diff_dst tensor
      const size_t kSrcIndex = 1;       // index of src input tensor
      const size_t kScaleIndex = 2;     // index of scale tensor
      const size_t kMeanIndex = 3;      // index of saved_mean tensor
      const size_t kVarianceIndex = 4;  // index of saved_variance tensor

      const Tensor& diff_dst_tensor = MklGetInput(context, kDiffDstIndex);
      const Tensor& src_tensor = MklGetInput(context, kSrcIndex);
      const Tensor& scale_tensor = MklGetInput(context, kScaleIndex);
      const Tensor& saved_mean_tensor = MklGetInput(context, kMeanIndex);
      const Tensor& saved_variance_tensor =
          MklGetInput(context, kVarianceIndex);

      MklDnnShape dnn_shape_src, dnn_shape_diff_dst;
      GetMklShape(context, kSrcIndex, &dnn_shape_src);
      GetMklShape(context, kDiffDstIndex, &dnn_shape_diff_dst);

      TensorShape tf_shape_src, tf_shape_diff_dst;
      if (dnn_shape_diff_dst.IsMklTensor()) {
        tf_shape_diff_dst = dnn_shape_diff_dst.GetTfShape();
        OP_REQUIRES(
            context, dnn_shape_diff_dst.GetDimension() == 4,
            errors::InvalidArgument("input must be 4-dimensional",
                                    diff_dst_tensor.shape().DebugString()));
      } else {
        tf_shape_diff_dst = diff_dst_tensor.shape();
        OP_REQUIRES(
            context, diff_dst_tensor.dims() == 4,
            errors::InvalidArgument("input must be 4-dimensional",
                                    diff_dst_tensor.shape().DebugString()));
      }

      if (dnn_shape_src.IsMklTensor()) {
        tf_shape_src = dnn_shape_src.GetTfShape();
        OP_REQUIRES(context, dnn_shape_src.GetDimension() == 4,
                    errors::InvalidArgument("input must be 4-dimensional",
                                            src_tensor.shape().DebugString()));
      } else {
        tf_shape_src = src_tensor.shape();
        OP_REQUIRES(context, src_tensor.dims() == 4,
                    errors::InvalidArgument("input must be 4-dimensional",
                                            src_tensor.shape().DebugString()));
      }

      OP_REQUIRES(context, scale_tensor.dims() == 1,
                  errors::InvalidArgument("scale must be 1-dimensional",
                                          scale_tensor.shape().DebugString()));
      OP_REQUIRES(
          context, saved_mean_tensor.dims() == 1,
          errors::InvalidArgument("saved mean must be 1-dimensional",
                                  saved_mean_tensor.shape().DebugString()));

      OP_REQUIRES(
          context, saved_variance_tensor.dims() == 1,
          errors::InvalidArgument("saved variance must be 1-dimensional",
                                  saved_variance_tensor.shape().DebugString()));

      Tensor* diff_src_tensor = nullptr;
      // special case: input with 0 element and 0 batch size
      if (tf_shape_src.num_elements() == 0 ||
          tf_shape_diff_dst.num_elements() == 0) {
        HandleEmptyInput(context, tf_shape_src, scale_tensor.shape(),
                         &diff_src_tensor);
        return;
      }

      if (dnn_shape_src.IsMklTensor()) {
        depth_ = dnn_shape_src.DimSize(MklDnnDims::Dim_C);
      } else if (dnn_shape_diff_dst.IsMklTensor()) {
        depth_ = dnn_shape_diff_dst.DimSize(MklDnnDims::Dim_C);
      } else {
        ExtractParams(context);
      }

      memory::format format_m;
      if (dnn_shape_src.IsMklTensor()) {
        if (dnn_shape_src.IsTensorInNCHWFormat())
          format_m = memory::format::nchw;
        else
          format_m = memory::format::nhwc;
      } else {
        format_m = TFDataFormatToMklDnnDataFormat(tensor_format_);
      }

      MklDnnData<T> src(&cpu_engine);
      MklDnnData<T> diff_dst(&cpu_engine);
      MklDnnData<U> weights(&cpu_engine);
      MklDnnData<U> diff_weights(&cpu_engine);

      memory::dims src_dims =
          dnn_shape_src.IsMklTensor()
              ? dnn_shape_src.GetSizesAsMklDnnDims()
              : TFShapeToMklDnnDimsInNCHW(src_tensor.shape(), tensor_format_);
      memory::dims diff_dst_dims =
          dnn_shape_diff_dst.IsMklTensor()
              ? dnn_shape_diff_dst.GetSizesAsMklDnnDims()
              : TFShapeToMklDnnDimsInNCHW(diff_dst_tensor.shape(),
                                          tensor_format_);

      // set src and diff_dst primitive descriptors
      memory::desc src_md =
          dnn_shape_src.IsMklTensor()
              ? dnn_shape_src.GetMklLayout()
              : memory::desc(src_dims, MklDnnType<T>(), format_m);
      memory::desc diff_dst_md =
          dnn_shape_diff_dst.IsMklTensor()
              ? dnn_shape_diff_dst.GetMklLayout()
              : memory::desc(diff_dst_dims, MklDnnType<T>(), format_m);

      // weights -- MKL DNN packs scales/ shifts as weights in order
      // of scale, ..., scale, shift, ...., shift
      weights.AllocateBuffer(2 * depth_ * sizeof(U));
      U* weights_data_tf = reinterpret_cast<U*>(weights.GetAllocatedBuffer());
      const U* scale_tf = scale_tensor.flat<U>().data();
      for (int k = 0; k < depth_; k++) {
        weights_data_tf[k] = scale_tf[k];
        weights_data_tf[k + depth_] = static_cast<U>(0);
      }

      diff_weights.AllocateBuffer(2 * depth_ * sizeof(U));

      MklBatchNormBwdParams bwdParams(src_dims, diff_dst_dims, depth_, epsilon_,
                                      is_training_);
      MklFusedBatchNormBwdPrimitive<T, U>* bn_bwd =
          MklFusedBatchNormBwdPrimitiveFactory<T, U>::Get(bwdParams);

      // check if src/diff_dst need to be reordered
      const T* src_data = src_tensor.flat<T>().data();
      if (src_md.data.format != bn_bwd->GetSrcFmt()) {
        src.SetUsrMem(src_md, &src_tensor);
        auto src_target = memory::primitive_desc(
            {{src_dims},
             MklDnnType<T>(),
             static_cast<memory::format>(bn_bwd->GetSrcFmt())},
            cpu_engine);
        src.CheckReorderToOpMem(src_target);
        src_data = const_cast<T*>(
            reinterpret_cast<T*>(src.GetOpMem().get_data_handle()));
      }

      const T* diff_dst_data = diff_dst_tensor.flat<T>().data();
      if (diff_dst_md.data.format != bn_bwd->GetDiffDstFmt()) {
        diff_dst.SetUsrMem(diff_dst_md, &diff_dst_tensor);
        auto diff_dst_target = memory::primitive_desc(
            {{diff_dst_dims},
             MklDnnType<T>(),
             static_cast<memory::format>(bn_bwd->GetDiffDstFmt())},
            cpu_engine);
        diff_dst.CheckReorderToOpMem(diff_dst_target);
        diff_dst_data = const_cast<T*>(
            reinterpret_cast<T*>(diff_dst.GetOpMem().get_data_handle()));
      }

      // Indices of output tensors
      const size_t kDiffSrcIndex = 0;  // index of diff_src tensor

      // allocate output tensor: diff_src, always set as MKL-DNN layout
      MklDnnShape dnn_shape_diff_src;
      TensorShape tf_shape_diff_src;
      dnn_shape_diff_src.SetMklTensor(true);
      auto diff_src_pd = bn_bwd->GetDiffSrcPd();
      dnn_shape_diff_src.SetMklLayout(&diff_src_pd);
      dnn_shape_diff_src.SetElemType(MklDnnType<T>());
      dnn_shape_diff_src.SetTfLayout(src_dims.size(), src_dims, format_m);
      dnn_shape_diff_src.SetTfDimOrder(src_dims.size(), tensor_format_);
      tf_shape_diff_src.AddDim(diff_src_pd.get_size() / sizeof(T));
      AllocateOutputSetMklShape(context, kDiffSrcIndex, &diff_src_tensor,
                                tf_shape_diff_src, dnn_shape_diff_src);

      U* mean_data =
          static_cast<U*>(const_cast<U*>(saved_mean_tensor.flat<U>().data()));
      U* variance_data = static_cast<U*>(
          const_cast<U*>(saved_variance_tensor.flat<U>().data()));
      U* weights_data = weights_data_tf;
      T* diff_src_data = static_cast<T*>(diff_src_tensor->flat<T>().data());
      U* diff_weights_data = static_cast<U*>(diff_weights.GetAllocatedBuffer());
      // Execute
      bn_bwd->Execute(src_data, mean_data, variance_data, diff_dst_data,
                      weights_data, diff_src_data, diff_weights_data);

      // allocate output TF tensors: diff_scale and diff_shift
      Tensor* diff_scale_tensor = nullptr;
      Tensor* diff_shift_tensor = nullptr;
      AllocateTFOutputs(context, scale_tensor.shape(), &diff_scale_tensor,
                        &diff_shift_tensor);

      // copy data: diff_scale and diff_shift
      auto diff_scale_data = diff_scale_tensor->flat<U>().data();
      auto diff_shift_data = diff_shift_tensor->flat<U>().data();
      std::memcpy(reinterpret_cast<char*>(diff_scale_data),
                  reinterpret_cast<char*>(diff_weights_data),
                  depth_ * sizeof(U));
      std::memcpy(reinterpret_cast<char*>(diff_shift_data),
                  reinterpret_cast<char*>(diff_weights_data + depth_),
                  depth_ * sizeof(U));
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
  float epsilon_;
  TensorFormat tensor_format_;
  int depth_;  // batch normalization is done for per channel.
  bool is_training_;
  engine cpu_engine = engine(engine::cpu, 0);

  void ExtractParams(OpKernelContext* context) {
    const Tensor& input = MklGetInput(context, 0);
    depth_ = static_cast<int>(GetTensorDim(input, tensor_format_, 'C'));
  }

  void HandleEmptyInput(OpKernelContext* context, TensorShape tf_shape_src,
                        TensorShape tf_shape_scale_shift,
                        Tensor** diff_src_tensor) {
    const size_t kDiffSrcIndex = 0;

    MklDnnShape dnn_shape_diff_src;
    dnn_shape_diff_src.SetMklTensor(false);
    AllocateOutputSetMklShape(context, kDiffSrcIndex, diff_src_tensor,
                              tf_shape_src, dnn_shape_diff_src);
    auto diff_src_data = (*diff_src_tensor)->flat<T>().data();
    std::fill_n(diff_src_data, (*diff_src_tensor)->shape().num_elements(),
                static_cast<T>(0));

    Tensor* diff_scale_tensor = nullptr;
    Tensor* diff_shift_tensor = nullptr;
    AllocateTFOutputs(context, tf_shape_scale_shift, &diff_scale_tensor,
                      &diff_shift_tensor);
  }

  void AllocateTFOutputs(OpKernelContext* context,
                         TensorShape tf_shape_scale_shift,
                         Tensor** diff_scale_tensor,
                         Tensor** diff_shift_tensor) {
    CHECK_NOTNULL(diff_scale_tensor);
    CHECK_NOTNULL(diff_shift_tensor);

    const size_t kDiffScaleIndex = 1;
    const size_t kDiffShiftIndex = 2;
    const size_t kP1Index = 3;
    const size_t kP2Index = 4;

    // separate out scale and shift grad and copy to individual tensors
    MklDnnShape mkl_shape_diff_scale;
    mkl_shape_diff_scale.SetMklTensor(false);
    AllocateOutputSetMklShape(context, kDiffScaleIndex, diff_scale_tensor,
                              tf_shape_scale_shift, mkl_shape_diff_scale);
    CHECK_NOTNULL(*diff_scale_tensor);
    auto diff_scale_data = (*diff_scale_tensor)->flat<U>().data();
    std::fill_n(diff_scale_data, (*diff_scale_tensor)->shape().num_elements(),
                static_cast<U>(0));

    MklDnnShape mkl_shape_diff_shift;
    mkl_shape_diff_shift.SetMklTensor(false);
    AllocateOutputSetMklShape(context, kDiffShiftIndex, diff_shift_tensor,
                              tf_shape_scale_shift, mkl_shape_diff_shift);
    CHECK_NOTNULL(*diff_shift_tensor);
    auto diff_shift_data = (*diff_shift_tensor)->flat<U>().data();
    std::fill_n(diff_shift_data, (*diff_shift_tensor)->shape().num_elements(),
                static_cast<U>(0));

    // Placeholders for estimated_mean and estimated_variance, which are
    // used for inference and thus not needed here for gradient computation.
    Tensor *p1_tensor = nullptr, *p2_tensor = nullptr;
    MklDnnShape mkl_shape_p;
    mkl_shape_p.SetMklTensor(false);
    AllocateOutputSetMklShape(context, kP1Index, &p1_tensor, TensorShape({}),
                              mkl_shape_p);
    AllocateOutputSetMklShape(context, kP2Index, &p2_tensor, TensorShape({}),
                              mkl_shape_p);
  }

  memory::dims GetMeanVarianceDims() { return memory::dims({1, depth_}); }
};

#define REGISTER_MKL_FUSED_BATCHNORM_CPU(T)                         \
  REGISTER_KERNEL_BUILDER(Name("_MklFusedBatchNorm")                \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<T>("T")               \
                              .Label(mkl_op_registry::kMklOpLabel), \
                          MklFusedBatchNormOp<CPUDevice, T, T>);

TF_CALL_float(REGISTER_MKL_FUSED_BATCHNORM_CPU);
#undef REGISTER_MKL_FUSED_BATCHNORM_CPU

#define REGISTER_MKL_FUSED_BATCHNORM_V2_CPU(T, U)                   \
  REGISTER_KERNEL_BUILDER(Name("_MklFusedBatchNormV2")              \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<T>("T")               \
                              .TypeConstraint<U>("U")               \
                              .Label(mkl_op_registry::kMklOpLabel), \
                          MklFusedBatchNormOp<CPUDevice, T, U>);

REGISTER_MKL_FUSED_BATCHNORM_V2_CPU(float, float);
#undef REGISTER_MKL_FUSED_BATCHNORM_V2_CPU

#define REGISTER_MKL_FUSED_BATCHNORM_GRAD_CPU(T)                    \
  REGISTER_KERNEL_BUILDER(Name("_MklFusedBatchNormGrad")            \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<T>("T")               \
                              .Label(mkl_op_registry::kMklOpLabel), \
                          MklFusedBatchNormGradOp<CPUDevice, T, T>);

TF_CALL_float(REGISTER_MKL_FUSED_BATCHNORM_GRAD_CPU);
#undef REGISTER_MKL_FUSED_BATCHNORM_GRAD_CPU

#define REGISTER_MKL_FUSED_BATCHNORM_GRAD_V2_CPU(T, U)              \
  REGISTER_KERNEL_BUILDER(Name("_MklFusedBatchNormGradV2")          \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<T>("T")               \
                              .TypeConstraint<U>("U")               \
                              .Label(mkl_op_registry::kMklOpLabel), \
                          MklFusedBatchNormGradOp<CPUDevice, T, U>);

REGISTER_MKL_FUSED_BATCHNORM_GRAD_V2_CPU(float, float);
#undef REGISTER_MKL_FUSED_BATCHNORM_GRAD_V2_CPU

}  // namespace tensorflow

#endif  // INTEL_MKL
