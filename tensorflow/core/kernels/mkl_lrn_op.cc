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

// LRN = Local Response Normalization
// See docs in ../ops/nn_ops.cc. This opkernel uses MKL library, create MKL
// layout and primitives, use MKL dnn primitives to compute local
// response normalization

#ifdef INTEL_MKL

#define EIGEN_USE_THREADS
#include <vector>
#include "mkldnn.hpp"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/util/tensor_format.h"

#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/util/work_sharder.h"
#endif

using mkldnn::lrn_across_channels;
using mkldnn::lrn_backward;
using mkldnn::lrn_forward;
using mkldnn::prop_kind;
using mkldnn::stream;

namespace tensorflow {

namespace {
// Create a depth-by-depth band matrix with 1s along a swath of size (2 *
// depth_radius + 1) around the diagonal.
template <typename T>
void GetBandMatrix(int depth, int depth_radius,
                   Eigen::Tensor<T, 2, Eigen::RowMajor>* result) {
  result->setZero();
  for (int row = 0; row < depth; ++row) {
    const int begin = std::max<int>(0, row - depth_radius);
    const int end = std::min<int>(depth, row + depth_radius + 1);
    Eigen::DSizes<Eigen::DenseIndex, 2> start(row, begin);
    Eigen::DSizes<Eigen::DenseIndex, 2> sizes(1, end - begin);
    result->slice(start, sizes).setConstant(T(1));
  }
}

}  // namespace

template <typename T>
class MklLRNOp : public OpKernel {
 public:
  ~MklLRNOp() {}

  explicit MklLRNOp(OpKernelConstruction* context) : OpKernel(context) {
    int64 depth_radius64;
    OP_REQUIRES_OK(context, context->GetAttr("depth_radius", &depth_radius64));
    OP_REQUIRES(
        context,
        FastBoundsCheck(depth_radius64, std::numeric_limits<int>::max()),
        errors::InvalidArgument("depth_radius = ", depth_radius64,
                                " larger than int max"));
    depth_radius_ = static_cast<size_t>(depth_radius64);

    OP_REQUIRES_OK(context, context->GetAttr("bias", &bias_));
    OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha_));
    OP_REQUIRES_OK(context, context->GetAttr("beta", &beta_));
    workspace_enabled_ = false;
    OP_REQUIRES_OK(context,
                   context->GetAttr("workspace_enabled", &workspace_enabled_));
  }

  void Compute(OpKernelContext* context) override {
    try {
      SanityCheckInputs(context);
      if (!context->status().ok()) return;

      auto cpu_engine = engine(engine::cpu, 0);
      const Tensor& src_tensor = MklGetInput(context, kIdxInput);
      MklDnnShape src_dnn_shape;
      GetMklShape(context, kIdxInput, &src_dnn_shape);

      // MKL-DNN has a notion of kernel_size and not depth_radius.
      int kernel_size = 2 * depth_radius_ + 1;
      float new_alpha = alpha_ * kernel_size;

      // if the input tensor is not an MKL Tensor, or if the last
      // dimension is not channel, then just use Eigen.
      // MKL only support normalization over the channel dimension.
      if (!src_dnn_shape.IsMklTensor()) {
        MklDefaultToEigen(context, src_tensor);
        return;
      } else if (!src_dnn_shape.IsMklChannelDim(src_dnn_shape.GetDimension() -
                                                1)) {
        Tensor converted_tensor =
            ConvertMklToTF<T>(context, src_tensor, src_dnn_shape);
        MklDefaultToEigen(context, converted_tensor);
        return;
      }
      // At this point, we can assume that the src is an MklTensor
      // and we can enable the workspace
      workspace_enabled_ = true;

      MklDnnData<T> src_dnn_data(&cpu_engine);
      MklDnnData<T> dst_dnn_data(&cpu_engine);
      MklDnnData<uint8> workspace_dnn_data(&cpu_engine);

      TensorShape tf_output_shape = src_tensor.shape();

      memory::desc src_md = src_dnn_shape.GetCurLayout();
      memory::dims input_dims = src_dnn_shape.GetSizesAsMklDnnDims();

      // Create memory for user input.
      // Since Tensorflow always performs normalization over last dimension,
      // and MKL-DNN performs normalization over Channel, we tell MKL-DNN
      // that input is in NHWC layout with Channel being the last dimension.
      src_dnn_data.SetUsrMem(src_md, &src_tensor);
      src_dnn_data.SetOpMemDesc(input_dims, memory::format::nhwc);

      // output_dnn_data and workspace both have the same shape as input
      dst_dnn_data.SetUsrMem(src_md);
      dst_dnn_data.SetOpMemDesc(input_dims, memory::format::nhwc);

      // Create LRN primitive descriptor.
      // Tensorflow's normalization semantics is across channels.
      // MKL-DNN also supports normalization within channel.
      auto lrn_desc = lrn_forward::desc(prop_kind::forward, lrn_across_channels,
                                        src_dnn_data.GetUsrMemDesc(),
                                        kernel_size, new_alpha, beta_, bias_);
      auto lrn_prim_desc = lrn_forward::primitive_desc(lrn_desc, cpu_engine);

      // Allocate output_dnn_data tensor.
      Tensor* output_tensor = nullptr;
      memory::format input_format = src_dnn_shape.GetTfDataFormat();
      AllocateOutputTensor(context, lrn_prim_desc, input_dims, input_format,
                           &output_tensor);
      OP_REQUIRES_OK(context, context->status());
      CHECK_NOTNULL(output_tensor);
      dst_dnn_data.SetUsrMemDataHandle(output_tensor);

      // Handle workspace required for MKL-DNN.
      AllocateWorkspaceTensor(context, lrn_prim_desc, &workspace_dnn_data);
      OP_REQUIRES_OK(context, context->status());

      PrepareAndExecuteNet(lrn_prim_desc, &src_dnn_data, &dst_dnn_data,
                           &workspace_dnn_data);
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
  void PrepareAndExecuteNet(const lrn_forward::primitive_desc& lrn_fwd_desc,
                            MklDnnData<T>* src_dnn_data,
                            MklDnnData<T>* dst_dnn_data,
                            MklDnnData<uint8>* wksp_dnn_data = nullptr) {
    // Check for input reorder
    src_dnn_data->CheckReorderToOpMem(lrn_fwd_desc.src_primitive_desc());

    // Create pooling primitive and add it to net
    std::vector<primitive> net;
    if (wksp_dnn_data != nullptr) {
      net.push_back(lrn_forward(lrn_fwd_desc, src_dnn_data->GetOpMem(),
                                wksp_dnn_data->GetOpMem(),
                                dst_dnn_data->GetOpMem()));
    } else {
      net.push_back(lrn_forward(lrn_fwd_desc, src_dnn_data->GetOpMem(),
                                dst_dnn_data->GetOpMem()));
    }
    stream(stream::kind::eager).submit(net).wait();
  }

  void AllocateOutputTensor(
      OpKernelContext* context,
      const lrn_forward::primitive_desc& lrn_fwd_prim_desc,
      const memory::dims output_dims_mkl_order,
      const memory::format& output_tf_format, Tensor** output_tensor) {
    CHECK_NOTNULL(output_tensor);
    memory::primitive_desc dst_pd = lrn_fwd_prim_desc.dst_primitive_desc();

    MklDnnShape output_mkl_shape;
    // We only handle the case when the inputs and output are in Mkl format
    // Any other case is handled by Eigen
    output_mkl_shape.SetMklTensor(true);
    output_mkl_shape.SetMklLayout(&dst_pd);
    output_mkl_shape.SetElemType(MklDnnType<T>());
    output_mkl_shape.SetTfLayout(output_dims_mkl_order.size(),
                                 output_dims_mkl_order, output_tf_format);
    TensorShape output_tf_shape;
    // only allocate enough space for the elements we need.
    size_t num_bytes = dst_pd.get_size();
    CHECK_EQ(num_bytes % sizeof(T), 0);
    output_tf_shape.AddDim(num_bytes / sizeof(T));
    AllocateOutputSetMklShape(context, kIdxOutput, output_tensor,
                              output_tf_shape, output_mkl_shape);
  }

  // Fallback implementation - Taken from lrn_op.cc
  // TODO(inteltf) Check if we can use EigenLRNOp directly instead of making a
  // copy.
  void MklDefaultToEigen(OpKernelContext* context, const Tensor& input) {
    const int batch = static_cast<int>(input.dim_size(0));
    const int rows = static_cast<int>(input.dim_size(1));
    const int cols = static_cast<int>(input.dim_size(2));
    const int depth = static_cast<int>(input.dim_size(3));
    const int nodes = cols * rows;

    auto in_shaped = input.shaped<T, 2>({nodes * batch, depth});
    // Multiplying the input with the band matrix has the effect of reducing
    // the
    // correct patch along the depth.
    Eigen::Tensor<T, 2, Eigen::RowMajor> multiplier(depth, depth);
    GetBandMatrix<T>(depth, depth_radius_, &multiplier);

    Tensor* output_dnn_data = nullptr;
    MklDnnShape mkl_output_mkl_shape;
    mkl_output_mkl_shape.SetMklTensor(false);
    mkl_output_mkl_shape.SetDimensions(4);
    AllocateOutputSetMklShape(context, kIdxOutput, &output_dnn_data,
                              input.shape(), mkl_output_mkl_shape);
    CHECK_NOTNULL(output_dnn_data);

    Tensor* workspace_tensor = nullptr;
    MklDnnShape workspace_mkl_shape;
    workspace_mkl_shape.SetMklTensor(false);
    TensorShape workspace_tf_shape;
    workspace_tf_shape.AddDim(0);
    AllocateOutputSetMklShape(context, kIdxWorkspace, &workspace_tensor,
                              workspace_tf_shape, workspace_mkl_shape);
    CHECK_NOTNULL(workspace_tensor);

    auto out_shaped = output_dnn_data->shaped<T, 2>({nodes * batch, depth});
    Eigen::array<DimPair, 1> dims = {{DimPair(1, 0)}};
    auto tmp = in_shaped.square().contract(multiplier, dims) * alpha_ + bias_;
    if (beta_ == T(1)) {
      out_shaped.device(context->eigen_cpu_device()) =
          in_shaped * tmp.inverse();
    } else if (beta_ == T(0.5)) {
      out_shaped.device(context->eigen_cpu_device()) = in_shaped * tmp.rsqrt();
    } else {
      out_shaped.device(context->eigen_cpu_device()) =
          in_shaped * (tmp.log() * -beta_).exp();
    }
  }

  void AllocateWorkspaceTensor(
      OpKernelContext* context,
      const lrn_forward::primitive_desc& lrn_fwd_prim_desc,
      MklDnnData<uint8>* dnn_data_wksp) {
    CHECK_NOTNULL(dnn_data_wksp);
    Tensor* workspace_tensor = nullptr;
    memory::primitive_desc workspace_pd =
        lrn_fwd_prim_desc.workspace_primitive_desc();
    size_t workspace_bytes = workspace_pd.get_size();
    MklDnnShape workspace_mkl_shape;
    // the workspace tensor is a uint8 tensor that has
    // exactly the number of bytes necessary
    workspace_mkl_shape.SetMklTensor(false);
    TensorShape workspace_tf_shape;
    workspace_tf_shape.AddDim(workspace_bytes);
    AllocateOutputSetMklShape(context, kIdxWorkspace, &workspace_tensor,
                              workspace_tf_shape, workspace_mkl_shape);
    CHECK_NOTNULL(workspace_tensor);
    dnn_data_wksp->SetUsrMem(workspace_pd, workspace_tensor);
  }

  void SanityCheckInputs(OpKernelContext* context) {
    const Tensor& src_tensor = MklGetInput(context, kIdxInput);
    MklDnnShape src_dnn_shape;
    GetMklShape(context, kIdxInput, &src_dnn_shape);
    if (src_dnn_shape.IsMklTensor()) {
      OP_REQUIRES(context, src_dnn_shape.GetDimension() == 4,
                  errors::InvalidArgument("input must be 4-dimensional"));
      OP_REQUIRES(context,
                  FastBoundsCheck(src_tensor.NumElements(),
                                  std::numeric_limits<int>::max()),
                  errors::InvalidArgument("argument to LRN too large"));
    } else {
      OP_REQUIRES(context, src_tensor.dims() == 4,
                  errors::InvalidArgument("input must be 4-dimensional"));
      OP_REQUIRES(context,
                  FastBoundsCheck(src_tensor.NumElements(),
                                  std::numeric_limits<int>::max()),
                  errors::InvalidArgument("argument to LRN too large"));
    }
  }
  const int kIdxInput = 0, kIdxOutput = 0, kIdxWorkspace = 1;

  typedef typename Eigen::Tensor<T, 1, Eigen::RowMajor>::DimensionPair DimPair;
  bool workspace_enabled_;
  int depth_radius_;
  float bias_;
  float alpha_;
  float beta_;
};

template <typename T>
class MklLRNGradOp : public OpKernel {
 public:
  explicit MklLRNGradOp(OpKernelConstruction* context) : OpKernel(context) {
    int64 depth_radius64;
    OP_REQUIRES_OK(context, context->GetAttr("depth_radius", &depth_radius64));
    OP_REQUIRES(
        context,
        FastBoundsCheck(depth_radius64, std::numeric_limits<int>::max()),
        errors::InvalidArgument("depth_radius = ", depth_radius64,
                                " larger than int max"));
    depth_radius_ = static_cast<int>(depth_radius64);
    OP_REQUIRES_OK(context, context->GetAttr("bias", &bias_));
    OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha_));
    OP_REQUIRES_OK(context, context->GetAttr("beta", &beta_));
    workspace_enabled_ = false;
    OP_REQUIRES_OK(context,
                   context->GetAttr("workspace_enabled", &workspace_enabled_));
  }

  void Compute(OpKernelContext* context) override {
    try {
      SanityCheckInputs(context);
      if (!context->status().ok()) return;

      auto cpu_engine = engine(engine::cpu, 0);
      MklDnnData<T> input_grad_dnn_data(&cpu_engine);
      MklDnnData<T> orig_input_dnn_data(&cpu_engine);
      MklDnnData<T> orig_output_dnn_data(&cpu_engine);
      MklDnnData<T> output_dnn_data(&cpu_engine);

      MklDnnShape input_grad_dnn_shape, orig_input_dnn_shape,
          orig_output_dnn_shape;
      GetMklShape(context, kIdxGradient, &input_grad_dnn_shape);
      GetMklShape(context, kIdxOrigInput, &orig_input_dnn_shape);
      GetMklShape(context, kIdxOrigOutput, &orig_output_dnn_shape);

      // We only use MKLDNN if all of the necessary inputs are present
      // in mkldnn format, and Channel is the last dimension
      bool can_use_mkldnn = workspace_enabled_ &&
                            input_grad_dnn_shape.IsMklTensor() &&
                            orig_input_dnn_shape.IsMklTensor() &&
                            orig_output_dnn_shape.IsMklTensor() &&
                            input_grad_dnn_shape.IsMklChannelDim(
                                input_grad_dnn_shape.GetDimension() - 1) &&
                            orig_input_dnn_shape.IsMklChannelDim(
                                orig_input_dnn_shape.GetDimension() - 1) &&
                            orig_output_dnn_shape.IsMklChannelDim(
                                orig_output_dnn_shape.GetDimension() - 1);

      if (!can_use_mkldnn) {
        // Fallback to eigen
        MklDefaultToEigen(context);
        return;
      }
      // At this point, we have the all clear to use MklDnn constructs
      // Naming: diff_dst is input_gradient_tensor; src is orig_input_tensor.
      const Tensor& input_grad_tensor = MklGetInput(context, kIdxGradient);
      const Tensor& orig_input_tensor = MklGetInput(context, kIdxOrigInput);

      // Get input sizes in MKL-DNN required NCHW format.
      // LRN does not have data_format attribute. But by default it has
      // NHWC format.
      memory::desc original_output_md = orig_output_dnn_shape.GetCurLayout();
      memory::desc target_diff_dst_md = ConfigureInputGradient(
          input_grad_tensor, input_grad_dnn_shape, &input_grad_dnn_data);

      memory::desc orig_input_md = orig_input_dnn_shape.GetCurLayout();
      memory::dims orig_input_dims =
          orig_input_dnn_shape.GetSizesAsMklDnnDims();
      orig_input_dnn_data.SetUsrMem(orig_input_md, &orig_input_tensor);
      orig_input_dnn_data.SetOpMemDesc(orig_input_dims, memory::format::nhwc);

      // output_dnn_data has the same shape as original input
      output_dnn_data.SetUsrMem(orig_input_md);
      output_dnn_data.SetOpMemDesc(orig_input_dims, memory::format::nhwc);

      // MKL-DNN has a notion of kernel_size and not depth_radius.
      int kernel_size = 2 * depth_radius_ + 1;
      float new_alpha = alpha_ * kernel_size;

      // Create LRN backward primitive descriptor. It requires LRN forward
      // primitive descriptor also.
      auto lrn_fwd_desc = lrn_forward::desc(
          prop_kind::forward, lrn_across_channels, orig_input_md, kernel_size,
          new_alpha, beta_, bias_);
      auto lrn_fwd_prim_desc =
          lrn_forward::primitive_desc(lrn_fwd_desc, cpu_engine);
      auto lrn_bwd_desc = lrn_backward::desc(
          lrn_across_channels, original_output_md, target_diff_dst_md,
          kernel_size, new_alpha, beta_, bias_);
      auto lrn_bwd_prim_desc = lrn_backward::primitive_desc(
          lrn_bwd_desc, cpu_engine, lrn_fwd_prim_desc);

      Tensor* output_tensor = nullptr;
      memory::format orig_input_format = orig_input_dnn_shape.GetTfDataFormat();
      AllocateOutputTensor(context, lrn_bwd_prim_desc, orig_input_dims,
                           orig_input_format, &output_tensor);
      OP_REQUIRES_OK(context, context->status());
      CHECK_NOTNULL(output_tensor);
      output_dnn_data.SetUsrMemDataHandle(output_tensor);

      // Create LRN primitive and add it to the net
      // At this point, workspace is enabled, so we don't need
      // to check. Pass input workspace to LRN backward primitive.
      const Tensor& workspace_tensor = MklGetInput(context, kIdxWorkspace);
      MklDnnData<uint8> workspace_dnn_data(&cpu_engine);
      ConfigureWorkspace(workspace_tensor,
                         lrn_fwd_prim_desc.workspace_primitive_desc(),
                         &workspace_dnn_data);

      PrepareAndExecuteNet(
          lrn_bwd_prim_desc, lrn_fwd_prim_desc, &orig_input_dnn_data,
          &input_grad_dnn_data, &output_dnn_data,
          memory::primitive_desc(target_diff_dst_md, cpu_engine),
          &workspace_dnn_data);
    } catch (mkldnn::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }

  void AllocateOutputTensor(
      OpKernelContext* context,
      const lrn_backward::primitive_desc& lrn_bkwd_prim_desc,
      const memory::dims output_dims_mkl_order,
      const memory::format& output_tf_format, Tensor** output_tensor) {
    CHECK_NOTNULL(output_tensor);
    memory::primitive_desc dst_pd =
        lrn_bkwd_prim_desc.diff_src_primitive_desc();
    MklDnnShape output_mkl_shape;

    // We assume that all outputs at this point are MKL Tensors
    output_mkl_shape.SetMklTensor(true);
    output_mkl_shape.SetMklLayout(&dst_pd);
    output_mkl_shape.SetElemType(MklDnnType<T>());
    output_mkl_shape.SetTfLayout(output_dims_mkl_order.size(),
                                 output_dims_mkl_order, output_tf_format);

    TensorShape output_tf_shape;
    size_t num_bytes = dst_pd.get_size();
    CHECK_EQ(num_bytes % sizeof(T), 0);
    output_tf_shape.AddDim(num_bytes / sizeof(T));
    AllocateOutputSetMklShape(context, kIdxOutput, output_tensor,
                              output_tf_shape, output_mkl_shape);
  }

  memory::desc ConfigureInputGradient(const Tensor& input_grad_tensor,
                                      const MklDnnShape& input_grad_dnn_shape,
                                      MklDnnData<T>* input_grad_dnn_data) {
    CHECK_NOTNULL(input_grad_dnn_data);
    // This shouldn't be necessary at this point, but just in case
    CHECK_EQ(input_grad_dnn_shape.IsMklTensor(), true);

    memory::desc input_grad_md = input_grad_dnn_shape.GetCurLayout();
    memory::dims orig_input_dims = input_grad_dnn_shape.GetSizesAsMklDnnDims();
    input_grad_dnn_data->SetUsrMem(input_grad_md, &input_grad_tensor);
    input_grad_dnn_data->SetOpMemDesc(orig_input_dims, memory::format::nhwc);
    return input_grad_md;
  }

  void PrepareAndExecuteNet(
      const lrn_backward::primitive_desc& lrn_bkwd_desc,
      const lrn_forward::primitive_desc& lrn_fwd_desc,
      MklDnnData<T>* src_dnn_data, MklDnnData<T>* input_gradient_diff_dst,
      MklDnnData<T>* output_diff_src,
      const memory::primitive_desc& target_diff_dst_pd,
      const MklDnnData<uint8>* workspace_dnn_data = nullptr) {
    // Check for input reordering on the diff dst input
    input_gradient_diff_dst->CheckReorderToOpMem(
        lrn_bkwd_desc.diff_dst_primitive_desc());

    // Check for input reordering on the original input
    src_dnn_data->CheckReorderToOpMem(lrn_fwd_desc.src_primitive_desc());
    // Create pooling primitive and add it to net
    std::vector<primitive> net;
    if (nullptr == workspace_dnn_data) {
      net.push_back(lrn_backward(lrn_bkwd_desc, src_dnn_data->GetOpMem(),
                                 input_gradient_diff_dst->GetOpMem(),
                                 output_diff_src->GetOpMem()));
    } else {
      net.push_back(lrn_backward(lrn_bkwd_desc, src_dnn_data->GetOpMem(),
                                 input_gradient_diff_dst->GetOpMem(),
                                 workspace_dnn_data->GetOpMem(),
                                 output_diff_src->GetOpMem()));
    }
    stream(stream::kind::eager).submit(net).wait();
  }

  void ConfigureWorkspace(const Tensor& workspace_tensor,
                          memory::primitive_desc workspace_pd,
                          MklDnnData<uint8>* workspace_dnn_data) {
    CHECK_NOTNULL(workspace_dnn_data);

    workspace_dnn_data->SetUsrMem(workspace_pd, &workspace_tensor);
  }

  // Fallback implementation - Taken from lrn_op.cc
  // TODO(intelft) Check if we can use EigenLRNOp directly instead of making a
  // copy.
  void MklDefaultToEigen(OpKernelContext* context) {
    Tensor input_gradient_tensor;
    Tensor orig_input_tensor;
    Tensor orig_output_tensor;

    MklDnnShape input_grad_dnn_shape, orig_input_dnn_shape,
        orig_output_dnn_shape;
    GetMklShape(context, kIdxGradient, &input_grad_dnn_shape);
    GetMklShape(context, kIdxOrigInput, &orig_input_dnn_shape);
    GetMklShape(context, kIdxOrigOutput, &orig_output_dnn_shape);

    if (input_grad_dnn_shape.IsMklTensor()) {
      input_gradient_tensor = ConvertMklToTF<T>(
          context, MklGetInput(context, kIdxGradient), input_grad_dnn_shape);
    } else {
      input_gradient_tensor = MklGetInput(context, kIdxGradient);
    }

    if (orig_input_dnn_shape.IsMklTensor()) {
      orig_input_tensor = ConvertMklToTF<T>(
          context, MklGetInput(context, kIdxOrigInput), orig_input_dnn_shape);
    } else {
      orig_input_tensor = MklGetInput(context, kIdxOrigInput);
    }

    if (orig_output_dnn_shape.IsMklTensor()) {
      orig_output_tensor = ConvertMklToTF<T>(
          context, MklGetInput(context, kIdxOrigOutput), orig_output_dnn_shape);
    } else {
      orig_output_tensor = MklGetInput(context, kIdxOrigOutput);
    }

    const int64 batch = static_cast<int64>(input_gradient_tensor.dim_size(0));
    const int64 rows = static_cast<int64>(input_gradient_tensor.dim_size(1));
    const int64 cols = static_cast<int64>(input_gradient_tensor.dim_size(2));
    const int64 depth = static_cast<int64>(input_gradient_tensor.dim_size(3));
    const auto nodes = cols * rows;

    auto grads_shaped =
        input_gradient_tensor.shaped<T, 2>({nodes * batch, depth});

    auto in_shaped = orig_input_tensor.shaped<T, 2>({nodes * batch, depth});
    auto activations = orig_output_tensor.shaped<T, 2>({nodes * batch, depth});

    Tensor* output_dnn_data;
    MklDnnShape mkl_output_mkl_shape;
    mkl_output_mkl_shape.SetMklTensor(false);
    mkl_output_mkl_shape.SetDimensions(4);
    AllocateOutputSetMklShape(context, kIdxOutput, &output_dnn_data,
                              input_gradient_tensor.shape(),
                              mkl_output_mkl_shape);

    auto out_shaped = output_dnn_data->shaped<T, 2>({nodes * batch, depth});
    out_shaped.setZero();
    auto shard = [this, activations, in_shaped, grads_shaped, out_shaped,
                  depth](int64 begin, int64 end) {
      for (int64 i = begin; i < end; ++i) {
        for (int64 j = 0; j < depth; ++j) {
          int64 depth_begin = std::max<int64>(0, j - depth_radius_);
          int64 depth_end = std::min<int64>(depth, j + depth_radius_ + 1);

          T norm(0);
          for (int64 k = depth_begin; k < depth_end; ++k) {
            norm += in_shaped(i, k) * in_shaped(i, k);
          }
          norm = alpha_ * norm + bias_;
          DCHECK_GT(norm, T(1e-6));
          for (int64 k = depth_begin; k < depth_end; ++k) {
            T dyi = T(-2) * alpha_ * beta_ * in_shaped(i, k) *
                    activations(i, j) / norm;
            if (k == j) {
              dyi += Eigen::numext::pow(norm, -beta_);
            }
            dyi *= grads_shaped(i, j);
            const_cast<typename TTypes<T, 2>::Tensor&>(out_shaped)(i, k) += dyi;
          }
        }
      }
    };
    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers, nodes * batch,
          depth * depth, shard);
  }

  void SanityCheckInputs(OpKernelContext* context) {
    const Tensor& input_gradient_tensor = MklGetInput(context, kIdxGradient);
    const Tensor& orig_input_tensor = MklGetInput(context, kIdxOrigInput);
    const Tensor& orig_output_tensor = MklGetInput(context, kIdxOrigOutput);
    const Tensor& workspace_tensor = MklGetInput(context, kIdxWorkspace);
    MklDnnShape in_grads_dnn_shape, in_image_dnn_shape, out_image_dnn_shape,
        workspace_dnn_shape;
    GetMklShape(context, kIdxGradient, &in_grads_dnn_shape);
    GetMklShape(context, kIdxOrigInput, &in_image_dnn_shape);
    GetMklShape(context, kIdxOrigOutput, &out_image_dnn_shape);
    GetMklShape(context, kIdxWorkspace, &workspace_dnn_shape);
    if (in_grads_dnn_shape.IsMklTensor()) {
      OP_REQUIRES(context, in_grads_dnn_shape.GetDimension() == 4,
                  errors::InvalidArgument("Input gradient must be "
                                          "4-dimensional"));
    } else {
      OP_REQUIRES(
          context, input_gradient_tensor.dims() == 4,
          errors::InvalidArgument("input gradient must be 4-dimensional"));
    }

    if (in_image_dnn_shape.IsMklTensor()) {
      OP_REQUIRES(context, in_image_dnn_shape.GetDimension() == 4,
                  errors::InvalidArgument("input images must be "
                                          "4-dimensional"));
    } else {
      OP_REQUIRES(context, orig_input_tensor.dims() == 4,
                  errors::InvalidArgument("input images must be "
                                          "4-dimensional"));
    }

    if (out_image_dnn_shape.IsMklTensor()) {
      OP_REQUIRES(context, out_image_dnn_shape.GetDimension() == 4,
                  errors::InvalidArgument("Output image must be "
                                          "4-dimensional"));
    } else {
      OP_REQUIRES(
          context, orig_output_tensor.dims() == 4,
          errors::InvalidArgument("Output image must be 4-dimensional"));
    }

    if (workspace_enabled_) {
      if (workspace_dnn_shape.IsMklTensor()) {
        OP_REQUIRES(
            context, workspace_dnn_shape.IsMklTensor() == false,
            errors::InvalidArgument("Workspace should not be MKL Tensor."));
      } else {
        OP_REQUIRES(context, workspace_tensor.dims() == 1,
                    errors::InvalidArgument("Workspace must be 1-dimensional"));
      }
    }
  }

  // Input("input_grads: T")
  // Input("input_image: T")
  // Input("output_image: T")
  // Input("workspace: uint8")
  const int kIdxGradient = 0, kIdxOrigInput = 1, kIdxOrigOutput = 2,
            kIdxWorkspace = 3, kIdxOutput = 0;

  typedef typename Eigen::Tensor<T, 1, Eigen::RowMajor>::DimensionPair DimPair;
  bool workspace_enabled_;
  int depth_radius_;
  float bias_;
  float alpha_;
  float beta_;
};

#define REGISTER_MKL_LRN_CPU(T)                                     \
  REGISTER_KERNEL_BUILDER(Name("_MklLRN")                           \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<T>("T")               \
                              .Label(mkl_op_registry::kMklOpLabel), \
                          MklLRNOp<T>);                             \
  REGISTER_KERNEL_BUILDER(Name("_MklLRNGrad")                       \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<T>("T")               \
                              .Label(mkl_op_registry::kMklOpLabel), \
                          MklLRNGradOp<T>);

TF_CALL_float(REGISTER_MKL_LRN_CPU);

}  // namespace tensorflow

#endif  // INTEL_MKL
