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
#include <string>
#include <vector>
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/util/padding.h"

#ifndef INTEL_MKL_ML
#include "mkldnn.hpp"
using mkldnn::memory;
using mkldnn::pooling_backward;
using mkldnn::pooling_forward;
using mkldnn::stream;
#endif

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

struct MklPoolParameters {
  int depth;

  int tensor_in_cols;
  int tensor_in_rows;
  int tensor_in_batch;

  int window_rows;
  int window_cols;
  int depth_window;

  int row_stride;
  int col_stride;
  int depth_stride;

  int64 out_height;
  int64 out_width;
  int out_depth;

  int64 pad_left;
  int64 pad_right;
  int64 pad_top;
  int64 pad_bottom;
  int pad_depth;

  TensorFormat data_format;
  MklPoolParameters()
      : depth(0),
        tensor_in_cols(0),
        tensor_in_rows(0),
        tensor_in_batch(0),
        window_rows(0),
        window_cols(0),
        depth_window(0),
        row_stride(0),
        col_stride(0),
        depth_stride(0),
        out_height(0),
        out_width(0),
        out_depth(0),
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
#ifdef INTEL_MKL_ML
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

#ifndef INTEL_MKL_ML

template <class T>
class MklPoolingOpBase : public OpKernel {
 public:
  explicit MklPoolingOpBase(OpKernelConstruction* context)
      : OpKernel(context), workspace_enabled_(false) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &this->data_format_tf_),
                errors::InvalidArgument("Invalid data format"));
    this->data_format_mkldnn_ =
        TFDataFormatToMklDnnDataFormat(this->data_format_tf_);
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &this->ksize_));
    OP_REQUIRES(context, this->ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &this->stride_));
    OP_REQUIRES(context, this->stride_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &this->padding_));
    OP_REQUIRES(context, this->ksize_[0] == 1 && this->stride_[0] == 1,
                errors::Unimplemented("Pooling is not yet supported on the "
                                      "batch dimension."));

    // We may not get this attribute for this node if it does not go through
    // graph rewrite pass. So we do not check for error while retrieving this
    // attribute value.
    context->GetAttr("workspace_enabled", &this->workspace_enabled_);
  }
  void Compute(OpKernelContext* context) override = 0;

 protected:
  // Calculate output shape of pooling op in MKL-DNN and TensorFlow order.
  // MKL-DNN uses NCHW for output order. But TensorFlow output will be in
  // NHWC or NCHW format depending on data format. Function expects
  // output height and output width to have already been int32
  // bounds-checked
  void GetOutputDims(const MklPoolParameters& mkl_pool_params,
                     memory::dims* output_dims_mkl_order) {
    // MKL-DNN always needs output in NCHW format.
    *output_dims_mkl_order = {mkl_pool_params.tensor_in_batch,
                              mkl_pool_params.out_depth,
                              static_cast<int>(mkl_pool_params.out_height),
                              static_cast<int>(mkl_pool_params.out_width)};
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
    memory::desc input_md =
        input_mkl_shape.IsMklTensor()
            ? input_mkl_shape.GetMklLayout()
            : memory::desc(TFShapeToMklDnnDimsInNCHW(input_tensor_shape,
                                                     this->data_format_tf_),
                           MklDnnType<T>(), this->data_format_mkldnn_);
    dnn_data_input->SetUsrMem(input_md, &input_tensor);
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

  void PrepareAndExecuteNet(
      const pooling_forward::primitive_desc& pool_fwd_desc,
      const MklDnnData<T>* src, MklDnnData<T>* dst,
      MklDnnData<uint8>* wksp = nullptr) {
    std::vector<primitive> net;

    // Create pooling primitive and add it to net
    if (wksp != nullptr) {
      net.push_back(pooling_forward(pool_fwd_desc, src->GetOpMem(),
                                    dst->GetOpMem(), wksp->GetOpMem()));
    } else {
      net.push_back(
          pooling_forward(pool_fwd_desc, src->GetOpMem(), dst->GetOpMem()));
    }
    stream(stream::kind::eager).submit(net).wait();
  }

  void SanityCheckInput(OpKernelContext* context, const Tensor& input_tensor,
                        const MklDnnShape& input_mkl_shape) {
    if (!input_mkl_shape.IsMklTensor()) {
      OP_REQUIRES(context, input_tensor.dims() == 4,
                  errors::InvalidArgument("Input must be 4-dimensional"));
    } else {
      OP_REQUIRES(context, input_mkl_shape.GetDimension() == 4,
                  errors::InvalidArgument("Input shape must be "
                                          "4-dimensional"));
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

  void PrepareAndExecuteNet(
      const pooling_backward::primitive_desc& pool_bkwd_desc,
      MklDnnData<T>* input_gradient_diff_dst, MklDnnData<T>* output_diff_src,
      const memory::primitive_desc& target_diff_dst_pd,
      const MklDnnData<uint8>* workspace = nullptr) {
    std::vector<primitive> net;

    // If the input gradient isn't in the same format as the output
    // reorder it to the same format as the output
    input_gradient_diff_dst->CheckReorderToOpMem(target_diff_dst_pd, &net);

    // Create pooling primitive and add it to net
    if (nullptr == workspace) {
      net.push_back(pooling_backward(pool_bkwd_desc,
                                     input_gradient_diff_dst->GetOpMem(),
                                     output_diff_src->GetOpMem()));
    } else {
      net.push_back(
          pooling_backward(pool_bkwd_desc, input_gradient_diff_dst->GetOpMem(),
                           workspace->GetOpMem(), output_diff_src->GetOpMem()));
    }
    stream(stream::kind::eager).submit(net).wait();
  }

  // Max Pooling and Avg Pooling have slightly different implementations
  // Takes the Tensor containing original input data and the original
  // mkl Dnn Shape and populates other data
  memory::desc ConfigureOriginalInput(
      OpKernelContext* context, const Tensor& tensor_original_input_shape,
      const MklDnnShape& original_input_mkl_shape,
      memory::dims* original_input_dims_nchw, MklPoolParameters* pool_params,
      const TensorShape& input_tensor_shape) {
    CHECK_NOTNULL(original_input_dims_nchw);
    CHECK_NOTNULL(pool_params);
    this->InitMklPoolParameters(context, pool_params, original_input_mkl_shape,
                                input_tensor_shape);

    *original_input_dims_nchw =
        original_input_mkl_shape.IsMklTensor()
            ? original_input_mkl_shape.GetSizesAsMklDnnDims()
            : TFShapeToMklDnnDimsInNCHW(input_tensor_shape,
                                        this->data_format_tf_);

    return original_input_mkl_shape.IsMklTensor()
               ? original_input_mkl_shape.GetMklLayout()
               : memory::desc(*original_input_dims_nchw, MklDnnType<T>(),
                              this->data_format_mkldnn_);
  }

  memory::desc ConfigureOriginalOutput(
      const MklPoolParameters& pool_params,
      const MklDnnShape& original_output_mkl_shape,
      memory::dims output_dims_mkl_order) {
    this->GetOutputDims(pool_params, &output_dims_mkl_order);

    return original_output_mkl_shape.IsMklTensor()
               ? original_output_mkl_shape.GetMklLayout()
               : memory::desc(output_dims_mkl_order, MklDnnType<T>(),
                              this->data_format_mkldnn_);
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
                  TFShapeToMklDnnDimsInNCHW(input_gradient_tensor.shape(),
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
#endif  // INTEL_MKL_ML

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
