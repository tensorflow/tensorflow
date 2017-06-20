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

#ifdef INTEL_MKL

#include <algorithm>
#include <vector>
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/tensor_format.h"

#include "third_party/mkl/include/mkl_dnn.h"
#include "third_party/mkl/include/mkl_dnn_types.h"
#include "tensorflow/core/util/mkl_util.h"

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
    // Check that input tensor is in MKL format.
    const Tensor& input_tensor = MklGetInput(context, 0);
    MklShape input_shape;
    GetMklShape(context, 0, &input_shape);

    // if input is already in Tf format, then just copy input tensor to output.
    if (!input_shape.IsMklTensor()) {
      context->set_output(0, input_tensor);
      VLOG(1) << "MKLToTFConversion: No conversion needed, "
              << "copying input to output";
      return;
    }

    // Check that input data type is same as operator data type and that it is
    // same as output data type.
    DataType input_data_type = input_type(0);
    DataType output_data_type = output_type(0);
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
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_tensor));

    // If data format is NHWC, transform MKL tensor to NCHW format and then
    // do NCHW -> NHWC.
    dnnLayout_t lt_trans_input = nullptr;
    Tensor mkl_tmp_trans_input_buf_tensor;
    void* buf_trans_input = nullptr;
    bool input_fmt_nhwc = input_shape.IsTensorInNHWCFormat();
    if (input_fmt_nhwc && ndims == 4 && has_avx512f_) {
      size_t strides_nchw[4];
      GetStridesFromSizes(FORMAT_NCHW, strides_nchw, in_sizes);
      CHECK_EQ(dnnLayoutCreate_F32(&lt_trans_input, ndims, in_sizes,
                                           strides_nchw), E_SUCCESS);
      AllocTmpBuffer(context, &mkl_tmp_trans_input_buf_tensor, lt_trans_input,
                     &buf_trans_input);
    }
    else {
      lt_trans_input = static_cast<dnnLayout_t>(input_shape.GetTfLayout());
      buf_trans_input =
          static_cast<void*>(const_cast<T*>(output_tensor->flat<T>().data()));
    }

    // Execute DNNConversion.
    void* input_buffer =
        static_cast<void*>(const_cast<T*>(input_tensor.flat<T>().data()));
    input_shape.GetConvertedFlatData(lt_trans_input, input_buffer,
                                     buf_trans_input);
    // NCHW -> NHWC, if data format is NHWC
    if (input_fmt_nhwc && ndims == 4 && has_avx512f_) {
      dnnLayoutDelete_F32(lt_trans_input);
      TensorShape nhwc_shape = ShapeFromFormat(FORMAT_NHWC,
          in_sizes[MklDims::N], in_sizes[MklDims::H],
          in_sizes[MklDims::W], in_sizes[MklDims::C]);
      MklNCHWToNHWC(mkl_tmp_trans_input_buf_tensor, &output_tensor);
    }

    delete in_sizes;

    VLOG(1) << "MKLToTFConversion complete successfully.";
  }

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

TF_CALL_float(REGISTER_CPU);
#undef REGISTER_CPU
}  // namespace tensorflow
#endif /* INTEL_MKL */
