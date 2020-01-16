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

// See docs in ../ops/math_ops.cc.

#ifdef INTEL_MKL
#define EIGEN_USE_THREADS

#include <numeric>
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"

#include "mkldnn.hpp"
#include "tensorflow/core/util/mkl_util.h"
using mkldnn::stream;
using mkldnn::sum;

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class MklAddNOp : public OpKernel {
 public:
  ~MklAddNOp() {}
  explicit MklAddNOp(OpKernelConstruction* context) : OpKernel(context) {}

  TensorShape GetTensorShape(OpKernelContext* ctx, size_t src_index) {
    const Tensor& src_tensor = MklGetInput(ctx, src_index);
    MklDnnShape src_mkl_shape;
    GetMklShape(ctx, src_index, &src_mkl_shape);
    return src_mkl_shape.IsMklTensor() ? src_mkl_shape.GetTfShape()
                                       : src_tensor.shape();
  }

  bool CheckInputShape(OpKernelContext* ctx) {
    const int num_inputs = ctx->num_inputs() / 2;
    const TensorShape src0_shape = GetTensorShape(ctx, 0);

    for (size_t i = 1; i < num_inputs; ++i) {
      if (!src0_shape.IsSameSize(GetTensorShape(ctx, i))) {
        ctx->SetStatus(errors::InvalidArgument(
            "Inputs to operation ", this->name(), " of type ",
            this->type_string(),
            " must have the same size and shape.  Input 0: ",
            src0_shape.DebugString(), " != input : ", i,
            GetTensorShape(ctx, i).DebugString()));

        return false;
      }
    }

    return true;
  }

  // Return first tensor index which is in MKL layout, or -1 with no MKL input.
  int FindMKLInputIndex(OpKernelContext* ctx) {
    int mkl_index = -1;
    const int num_inputs = ctx->num_inputs() / 2;

    MklDnnShape src_mkl_shape;
    for (size_t i = 0; i < num_inputs; ++i) {
      GetMklShape(ctx, i, &src_mkl_shape);
      if (src_mkl_shape.IsMklTensor()) {
        mkl_index = i;
        break;
      }
    }

    return mkl_index;
  }

  void ComputeScalar(OpKernelContext* ctx) {
    const int num_inputs = ctx->num_inputs() / 2;
    const size_t kOutputIdx = 0;
    TensorShape output_tf_shape;
    MklDnnShape output_mkl_shape;
    Tensor* dst_tensor = nullptr;

    T sum = static_cast<T>(0);
    for (int src_idx = 0; src_idx < num_inputs; ++src_idx) {
      const Tensor& src_tensor = MklGetInput(ctx, src_idx);
      T* src_i = const_cast<T*>(src_tensor.flat<T>().data());
      sum += src_i[0];
    }

    output_mkl_shape.SetMklTensor(false);
    output_tf_shape = MklGetInput(ctx, kOutputIdx).shape();
    AllocateOutputSetMklShape(ctx, kOutputIdx, &dst_tensor, output_tf_shape,
                              output_mkl_shape);

    T* out_o = dst_tensor->flat<T>().data();
    out_o[0] = sum;
  }

  void Compute(OpKernelContext* ctx) override {
    // Each input tensor in MKL layout has additional meta-tensor carrying
    // layout information. So the number of actual tensors is half the total
    // number of inputs.
    const int num_inputs = ctx->num_inputs() / 2;

    MklDnnShape mkl_shape;
    const size_t kSrc0Idx = 0;
    const size_t kOutputIdx = 0;

    if (num_inputs == 1) {
      GetMklShape(ctx, kSrc0Idx, &mkl_shape);
      bool input_in_mkl_format = mkl_shape.IsMklTensor();

      if (input_in_mkl_format) {
        ForwardMklTensorInToOut(ctx, kSrc0Idx, kOutputIdx);
      } else {
        ForwardTfTensorInToOut(ctx, kSrc0Idx, kOutputIdx);
      }
      return;
    }

    // Check if the input shape is same
    if (!CheckInputShape(ctx)) return;

    try {
      TensorShape output_tf_shape;
      MklDnnShape output_mkl_shape;
      const Tensor& src_tensor = MklGetInput(ctx, kSrc0Idx);

      Tensor* dst_tensor = nullptr;

      // Nothing to compute, return.
      if (src_tensor.shape().num_elements() == 0) {
        output_mkl_shape.SetMklTensor(false);
        output_tf_shape = src_tensor.shape();
        AllocateOutputSetMklShape(ctx, kOutputIdx, &dst_tensor, output_tf_shape,
                                  output_mkl_shape);
        return;
      }

      if (src_tensor.dims() == 0) {
        ComputeScalar(ctx);
        return;
      }

      auto cpu_engine = engine(engine::cpu, 0);
      std::vector<float> coeff(num_inputs, 1.0);
      std::vector<memory::primitive_desc> srcs_pd;
      std::vector<MklDnnData<T>> srcs(num_inputs, MklDnnData<T>(&cpu_engine));
      std::vector<primitive::at> inputs;

      MklDnnData<T> dst(&cpu_engine);
      bool has_mkl_input = false;
      int mkl_input_index = FindMKLInputIndex(ctx);
      memory::format mkl_data_format;
      TensorFormat tf_data_format;
      if (mkl_input_index >= 0) {
        has_mkl_input = true;
        GetMklShape(ctx, mkl_input_index, &mkl_shape);
        // MKL input has the data format information.
        mkl_data_format = mkl_shape.GetTfDataFormat();
        tf_data_format = MklDnnDataFormatToTFDataFormat(mkl_data_format);
      }

      // Create memory descriptor for MKL-DNN.
      // If all input in Tensorflow format, create block memory descriptor,
      // else convet TF format to MKL memory descriptor
      for (int src_idx = 0; src_idx < num_inputs; ++src_idx) {
        MklDnnShape src_mkl_shape;
        GetMklShape(ctx, src_idx, &src_mkl_shape);
        memory::desc md({}, memory::data_undef, memory::format_undef);
        const Tensor& src_tensor = MklGetInput(ctx, src_idx);

        if (src_mkl_shape.IsMklTensor()) {
          md = src_mkl_shape.GetMklLayout();
        } else {
          if (has_mkl_input) {
            memory::dims src_dims;
            if (src_tensor.dims() == 4) {
              src_dims =
                  TFShapeToMklDnnDimsInNCHW(src_tensor.shape(), tf_data_format);
            } else {
              DCHECK(src_tensor.dims() == 5);
              src_dims = TFShapeToMklDnnDimsInNCDHW(src_tensor.shape(),
                                                    tf_data_format);
            }
            md = memory::desc(src_dims, MklDnnType<T>(), mkl_data_format);
          } else {
            // Create block memory descriptor for TensorFlow format input.
            auto dims = TFShapeToMklDnnDims(src_tensor.shape());
            auto strides = CalculateTFStrides(dims);
            md = MklDnnData<T>::CreateBlockedMemDesc(dims, strides);
          }
        }
        srcs_pd.push_back(memory::primitive_desc(md, cpu_engine));
        srcs[src_idx].SetUsrMem(md, &src_tensor);
        inputs.push_back(srcs[src_idx].GetOpMem());
      }

      auto sum_pd = sum::primitive_desc(coeff, srcs_pd);
      output_mkl_shape.SetMklTensor(has_mkl_input);
      auto output_pd = sum_pd.dst_primitive_desc();
      dst.SetUsrMem(output_pd);

      if (has_mkl_input) {
        output_mkl_shape.SetMklLayout(&output_pd);
        output_mkl_shape.SetElemType(MklDnnType<T>());
        output_mkl_shape.SetTfLayout(mkl_shape.GetDimension(),
                                     mkl_shape.GetSizesAsMklDnnDims(),
                                     mkl_shape.GetTfDataFormat());
        output_tf_shape.AddDim((output_pd.get_size() / sizeof(T)));
      } else {
        // All inputs have TF shapes, get the shape from first one.
        output_tf_shape = MklGetInput(ctx, kSrc0Idx).shape();
      }
      AllocateOutputSetMklShape(ctx, kOutputIdx, &dst_tensor, output_tf_shape,
                                output_mkl_shape);
      dst.SetUsrMemDataHandle(dst_tensor);

      // Create Sum op, and submit net for execution.
      std::vector<primitive> net;
      net.push_back(sum(sum_pd, inputs, dst.GetOpMem()));
      stream(stream::kind::eager).submit(net).wait();
    } catch (mkldnn::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          ctx, errors::Aborted("Operation received an exception:", error_msg));
    }
  }
};

#define REGISTER_MKL_CPU(T)                                    \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklAddN")                                         \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<T>("T")                              \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklAddNOp<CPUDevice, T>);

TF_CALL_float(REGISTER_MKL_CPU);
TF_CALL_bfloat16(REGISTER_MKL_CPU);
#undef REGISTER_MKL_CPU
}  // namespace tensorflow
#endif  // INTEL_MKL
