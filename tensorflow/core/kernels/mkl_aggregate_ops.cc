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

  void Compute(OpKernelContext* ctx) override {
    const int num = ctx->num_inputs();
    // Only additions of 2 input tensors is supported now
    OP_REQUIRES(ctx, num / 2 == 2,
                errors::InvalidArgument("Only additions of two tensors "
                                        "supported by MKL. Num inputs: ",
                                        num));

    try {
      auto cpu_engine = engine(engine::cpu, 0);
      size_t src1_idx = 0, src2_idx = 1, output_idx = 0;
      const Tensor& src1_tensor = MklGetInput(ctx, src1_idx);
      const Tensor& src2_tensor = MklGetInput(ctx, src2_idx);

      MklDnnShape src1_mkl_shape, src2_mkl_shape;
      GetMklShape(ctx, src1_idx, &src1_mkl_shape);
      GetMklShape(ctx, src2_idx, &src2_mkl_shape);
      bool input1_in_mkl_format = src1_mkl_shape.IsMklTensor();
      bool input2_in_mkl_format = src2_mkl_shape.IsMklTensor();
      int src1_dims_size = input1_in_mkl_format ? src1_mkl_shape.GetDimension()
                                                : src1_tensor.dims();
      int src2_dims_size = input2_in_mkl_format ? src2_mkl_shape.GetDimension()
                                                : src2_tensor.dims();
      // if the shapes of two tensors are not same raise op error
      TensorShape src1_shape, src2_shape;
      src1_shape = input1_in_mkl_format ? src1_mkl_shape.GetTfShape()
                                        : src1_tensor.shape();
      src2_shape = input2_in_mkl_format ? src2_mkl_shape.GetTfShape()
                                        : src2_tensor.shape();

      if (!src1_shape.IsSameSize(src2_shape)) {
        ctx->SetStatus(errors::InvalidArgument(
            "Inputs to operation ", this->name(), " of type ",
            this->type_string(),
            " must have the same size and shape.  Input 0: ",
            src1_shape.DebugString(),
            " != input 1: ", src2_shape.DebugString()));
      }

      if (!input1_in_mkl_format && src1_dims_size == 0) {
        Tensor* dst_tensor = nullptr;
        MklDnnShape mkl_shape_dst;
        mkl_shape_dst.SetMklTensor(false);
        AllocateOutputSetMklShape(ctx, output_idx, &dst_tensor,
                                  src1_tensor.shape(), mkl_shape_dst);
        float user_i1 = (src1_tensor.scalar<T>()());
        float user_i2 = (src2_tensor.scalar<T>()());
        dst_tensor->scalar<T>()() = std::plus<float>{}(user_i1, user_i2);
        return;
      }

      // If there is nothing to compute, return.
      if (!input1_in_mkl_format && !input2_in_mkl_format) {
        if (src1_tensor.shape().num_elements() == 0) {
          Tensor* dst_tensor = nullptr;
          MklDnnShape mkl_shape_dst;
          mkl_shape_dst.SetMklTensor(false);
          AllocateOutputSetMklShape(ctx, output_idx, &dst_tensor,
                                    src1_tensor.shape(), mkl_shape_dst);
          return;
        }
      }

      const std::vector<float> coeff(2, 1.0f);
      MklDnnData<T> src1(&cpu_engine);
      MklDnnData<T> src2(&cpu_engine);
      MklDnnData<T> dst(&cpu_engine);

      int tmp_size = input1_in_mkl_format ? src2_dims_size : src1_dims_size;
      memory::dims dims(tmp_size);
      memory::dims strides(tmp_size);
      memory::desc md1({}, memory::data_undef, memory::format_undef);
      memory::desc md2({}, memory::data_undef, memory::format_undef);

      // For creating Sum primitive, we need to ensure that all inputs are in
      // same format. What that means is if we have a mixed input case - where
      // one input is in Tensorflow format and one input is in MKL format -,
      // then we need to ensure that all inputs are in same format for
      // primitive construction. For performance reason, we say that all inputs
      // are in MKL format in such case, and insert reorder for input that is
      // in Tensorflow format into MKL format. On the other hand, if both the
      // inputs are in MKL format or both are in Tensorflow format, then we
      // dont need reorder.
      if (!input1_in_mkl_format && !input2_in_mkl_format) {
        // If both the inputs are in Tensorflow format, we create blocked memory
        // descriptor.
        dims = TFShapeToMklDnnDims(src1_tensor.shape());
        strides = CalculateTFStrides(dims);
        md1 = MklDnnData<T>::CreateBlockedMemDesc(dims, strides);
        md2 = md1;
      } else if (input1_in_mkl_format && !input2_in_mkl_format) {
        // If one input is in MKL format and other is in Tensorflow, then
        // create respective descriptors describing the actual case. For input
        // in Mkl format, we just get Mkl layout from MklDnnShape. For input in
        // Tensorflow format, we create memory descriptor using data format.
        md1 = src1_mkl_shape.GetMklLayout();

        memory::format src1_mkl_data_format = src1_mkl_shape.GetTfDataFormat();
        auto src1_tf_data_format =
            MklDnnDataFormatToTFDataFormat(src1_mkl_data_format);
        memory::dims src2_dims;
        if (src2_tensor.dims() == 4) {
          src2_dims = TFShapeToMklDnnDimsInNCHW(src2_tensor.shape(),
                                                src1_tf_data_format);
        } else {
          src2_dims = TFShapeToMklDnnDimsInNCDHW(src2_tensor.shape(),
                                                 src1_tf_data_format);
        }
        md2 = memory::desc(src2_dims, MklDnnType<T>(), src1_mkl_data_format);
      } else if (input2_in_mkl_format && !input1_in_mkl_format) {
        // Same comment as above.
        memory::format src2_mkl_data_format = src2_mkl_shape.GetTfDataFormat();
        auto src2_tf_data_format =
            MklDnnDataFormatToTFDataFormat(src2_mkl_data_format);
        memory::dims src1_dims;
        if (src1_tensor.dims() == 4) {
          src1_dims = TFShapeToMklDnnDimsInNCHW(src1_tensor.shape(),
                                                src2_tf_data_format);
        } else {
          src1_dims = TFShapeToMklDnnDimsInNCDHW(src1_tensor.shape(),
                                                 src2_tf_data_format);
        }
        md1 = memory::desc(src1_dims, MklDnnType<T>(), src2_mkl_data_format);

        md2 = src2_mkl_shape.GetMklLayout();
      } else {
        // If both the inputs are in MKL format, we use Mkl layout of the input
        // tensors.
        md1 = src1_mkl_shape.GetMklLayout();
        md2 = src2_mkl_shape.GetMklLayout();
      }
      src1.SetUsrMem(md1, &src1_tensor);
      src2.SetUsrMem(md2, &src2_tensor);

      // As per comment above, we tell MKLDNN that both the inputs are in same
      // format. So we set common memory descriptor in MKL format, if any of the
      // inputs are in MKL format. Let's get memory descriptor that we will use
      // for both the inputs.
      // We set output memory descriptor in MKL format, if any of the
      // inputs are in MKL format.
      memory::desc common_md({}, memory::data_undef, memory::format_undef);
      if (input1_in_mkl_format || input2_in_mkl_format) {
        common_md = input1_in_mkl_format ? md1 : md2;
        dst.SetUsrMem(common_md);
      } else {
        // Since both the inputs are in Tensorflow format, and have
        // same shape, we can get memory descriptor from any input.
        common_md = md1;
        dst.SetUsrMem(common_md);
      }

      std::vector<memory::primitive_desc> srcs_pd;
      // Memory descriptor for 1st input
      srcs_pd.push_back(memory::primitive_desc(common_md, cpu_engine));
      // Memory descriptor for 2nd input
      srcs_pd.push_back(memory::primitive_desc(common_md, cpu_engine));
      auto sum_pd = sum::primitive_desc(dst.GetUsrMemDesc(), coeff, srcs_pd);

      // Now we setup resources for primitive execution.
      // First, we need to check if any of the inputs need to be reordered as
      // per the logic described above. Since output will be in MKL format if
      // atleast one input is in MKL format, we choose output descriptor for
      // reorder.
      std::vector<primitive::at> inputs;
      // Check if actual input format of the tensor is different than common_pd
      // we told MKLDNN. In that case, we will need reorder.
      src1.CheckReorderToOpMem(srcs_pd[0]);
      src2.CheckReorderToOpMem(srcs_pd[1]);
      inputs.push_back(src1.GetOpMem());
      inputs.push_back(src2.GetOpMem());

      // Allocate output tensor now.
      Tensor* dst_tensor = nullptr;
      MklDnnShape output_mkl_shape;
      TensorShape output_tf_shape;

      if (input2_in_mkl_format || input1_in_mkl_format) {
        output_mkl_shape.SetMklTensor(true);
        auto output_pd = dst.GetUsrMemPrimDesc();
        output_mkl_shape.SetMklLayout(&output_pd);
        output_mkl_shape.SetElemType(MklDnnType<T>());
        if (input1_in_mkl_format) {
          output_mkl_shape.SetTfLayout(src1_dims_size,
                                       src1_mkl_shape.GetSizesAsMklDnnDims(),
                                       src1_mkl_shape.GetTfDataFormat());
        } else {
          output_mkl_shape.SetTfLayout(src2_dims_size,
                                       src2_mkl_shape.GetSizesAsMklDnnDims(),
                                       src2_mkl_shape.GetTfDataFormat());
        }
        output_tf_shape.AddDim((output_pd.get_size() / sizeof(T)));
      } else {
        output_mkl_shape.SetMklTensor(false);
        output_tf_shape = src1_tensor.shape();
      }
      AllocateOutputSetMklShape(ctx, output_idx, &dst_tensor, output_tf_shape,
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

#define REGISTER_MKL_CPU(T)                                         \
  REGISTER_KERNEL_BUILDER(Name("_MklAddN")                          \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<T>("T")               \
                              .Label(mkl_op_registry::kMklOpLabel), \
                          MklAddNOp<CPUDevice, T>);

TF_CALL_float(REGISTER_MKL_CPU);
#undef REGISTER_MKL_CPU
}  // namespace tensorflow
#endif  // INTEL_MKL
