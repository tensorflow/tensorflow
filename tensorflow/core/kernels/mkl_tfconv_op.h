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

#ifndef TENSORFLOW_CORE_KERNELS_MKL_TFCONV_OP_H_
#define TENSORFLOW_CORE_KERNELS_MKL_TFCONV_OP_H_

#ifdef INTEL_MKL

#include <algorithm>
#include <vector>
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/tensor_format.h"

#include "mkl_dnn.h"
#include "mkl_dnn_types.h"
#include "tensorflow/core/util/mkl_util.h"

#ifndef INTEL_MKL_ML
using mkldnn::stream;
#endif

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;

///////////////////////////////////////////////////////////
//               Op kernel
///////////////////////////////////////////////////////////

template <typename Device, typename T>
class MklToTfOp : public OpKernel {
 public:
  explicit MklToTfOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format_str));
    OP_REQUIRES_OK(context, context->GetAttr("T", &op_data_type));
    has_avx512f_ = port::TestCPUFeature(port::CPUFeature::AVX512F);
  }

  void Compute(OpKernelContext* context) override {
    ConvertMklToTf(this, context, data_format_str, op_data_type, has_avx512f_,
                   0);
    VLOG(1) << "MKLToTFConversion complete successfully.";
  }

#ifndef INTEL_MKL_ML
  static void ConvertMklToTf(OpKernel* op_kernel, OpKernelContext* context,
                             string data_format_str, DataType op_data_type,
                             bool has_avx512f, uint input_number) {
    try {
      // Check that input tensor is in MKL format.
      const Tensor& input_tensor = MklGetInput(context, input_number);
      MklDnnShape input_shape;
      GetMklShape(context, input_number, &input_shape);

      // if input is already in Tf format, then copy input tensor to output.
      if (!input_shape.IsMklTensor()) {
        context->set_output(input_number, input_tensor);
        VLOG(1) << "MKLToTFConversion: No conversion needed, "
                << "copying input to output";
        return;
      }

      // Check that input data type is same as operator data type and that it
      // is same as output data type.
      DataType input_data_type = op_kernel->input_type(input_number);
      DataType output_data_type = op_kernel->output_type(input_number);
      CHECK_EQ(op_data_type, input_data_type);
      CHECK_EQ(op_data_type, output_data_type);

      auto cpu_engine = engine(engine::cpu, 0);
      MklDnnData<T> input(&cpu_engine);

      // Get Mkl layout of input tensor.
      auto input_mkl_md = input_shape.GetMklLayout();
      // Get TensorFlow layout of input tensor. Expected output of conversion
      // has same layout as Tensorflow layout of input tensor.
      auto output_tf_md = input_shape.GetTfLayout();
      auto output_tf_pd = memory::primitive_desc(output_tf_md, cpu_engine);
      // Set input Mkl layout as the user layout.
      input.SetUsrMem(input_mkl_md, &input_tensor);

      // Allocate output tensor.
      TensorShape output_shape = input_shape.GetTfShape();
      Tensor* output_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(
                                  input_number, output_shape, &output_tensor));
      CHECK_NOTNULL(output_tensor);

      // Do we need to reorder Mkl layout into TensorFlow layout?
      if (input.IsReorderNeeded(output_tf_pd)) {
        // Insert reorder between Mkl layout and TensorFlow layout.
        std::vector<primitive> net;
        CHECK_EQ(input.CheckReorderToOpMem(output_tf_pd, output_tensor, &net),
                 true);
        stream(stream::kind::eager).submit(net).wait();
      } else {
        // If not, just forward input tensor to output tensor.
        CHECK(output_tensor->CopyFrom(input_tensor, output_shape));
      }
    } catch (mkldnn::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + std::string(e.message) + ", in file " +
                         std::string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }
#else
  static void ConvertMklToTf(OpKernel* op_kernel, OpKernelContext* context,
                             string data_format_str, DataType op_data_type,
                             bool has_avx512f, uint32 input_number) {
    // Check that input tensor is in MKL format.
    const Tensor& input_tensor = MklGetInput(context, input_number);
    MklShape input_shape;
    GetMklShape(context, input_number, &input_shape);

    // if input is already in Tf format, then just copy input tensor to output.
    if (!input_shape.IsMklTensor()) {
      context->set_output(input_number, input_tensor);
      VLOG(1) << "MKLToTFConversion: No conversion needed, "
              << "copying input to output";
      return;
    }

    // Check that input data type is same as operator data type and that it is
    // same as output data type.
    DataType input_data_type = op_kernel->input_type(input_number);
    DataType output_data_type = op_kernel->output_type(input_number);
    CHECK_EQ(op_data_type, input_data_type);
    CHECK_EQ(op_data_type, output_data_type);

    TensorShape output_shape;
    size_t ndims = input_shape.GetDimension();
    size_t* in_sizes = new size_t[ndims];
    for (size_t i = 0; i < ndims; i++) {
      // Outermost to innermost dimension
      output_shape.AddDim(input_shape.GetSizes()[input_shape.tf_dim_idx(i)]);
      in_sizes[i] = input_shape.GetSizes()[i];
    }

    // Allocate output tensor.
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(input_number, output_shape,
                                                     &output_tensor));

    dnnLayout_t output_layout =
        static_cast<dnnLayout_t>(input_shape.GetTfLayout());
    // Execute DNNConversion.
    void* input_buffer =
        static_cast<void*>(const_cast<T*>(input_tensor.flat<T>().data()));
    delete[] in_sizes;
    void* output_buffer =
        static_cast<void*>(const_cast<T*>(output_tensor->flat<T>().data()));
    input_shape.GetConvertedFlatData(output_layout, input_buffer,
                                     output_buffer);
    VLOG(1) << "MKLToTFConversion complete successfully.";
  }
#endif

 private:
  /// Data format of the operation
  string data_format_str;

  /// Data type of the operation
  DataType op_data_type;

  /// CPUIDInfo
  bool has_avx512f_ = false;
};

///////////////////////////////////////////////////////////
//               Register kernel
///////////////////////////////////////////////////////////

#define REGISTER_CPU(T)                                             \
  REGISTER_KERNEL_BUILDER(Name("_MklToTf")                          \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<T>("T")               \
                              .Label(mkl_op_registry::kMklOpLabel), \
                          MklToTfOp<CPUDevice, T>);

TF_CALL_NUMBER_TYPES(REGISTER_CPU);
#undef REGISTER_CPU
}  // namespace tensorflow
#endif  // INTEL_MKL
#endif  // TENSORFLOW_CORE_KERNELS_MKL_TFCONV_OP_H_
