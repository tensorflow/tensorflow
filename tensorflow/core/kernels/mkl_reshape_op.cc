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

#ifdef INTEL_MKL

#include <memory>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

#include "third_party/mkl/include/mkl_dnn.h"
#include "third_party/mkl/include/mkl_dnn_types.h"
#include "tensorflow/core/util/mkl_util.h"

namespace tensorflow {
using CPUDevice = Eigen::ThreadPoolDevice;
template <typename Device, typename T>
class MklReshapeOp : public OpKernel {
 public:
  explicit MklReshapeOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = MklGetInput(context, 0);
    const Tensor& sizes = MklGetInput(context, 1);

    // Preliminary validation of sizes.
    OP_REQUIRES(context, IsLegacyVector(sizes.shape()),
                errors::InvalidArgument("sizes input must be 1-D, not shape ",
                                        sizes.shape().DebugString()));
    const int64 num_dims = sizes.NumElements();

    // Compute the output shape.  Determine product of specified
    // dimensions, and find the index of the unspecified one.
    TensorShape shape;
    int64 product = 1;
    int unknown_index = -1;
    auto vec_size = sizes.flat<int32>();
    for (int d = 0; d < num_dims; ++d) {
      const int32 size = vec_size(d);
      if (size == -1) {
        OP_REQUIRES(
            context, unknown_index == -1,
            errors::InvalidArgument("only one input size may be -1, not both ",
                                    unknown_index, " and ", d));
        unknown_index = d;
        shape.AddDim(1);
      } else {
        OP_REQUIRES(context, size >= 0,
                    errors::InvalidArgument(
                        "size ", d, " must be non-negative, not ", size));
        shape.AddDim(size);
        product *= size;
      }
    }
    if (unknown_index != -1) {
      OP_REQUIRES(
          context, product > 0,
          errors::InvalidArgument("Reshape cannot infer the missing input size "
                                  "for an empty tensor unless all specified "
                                  "input sizes are non-zero"));
      const int64 missing = input.NumElements() / product;
      OP_REQUIRES(
          context, product * missing == input.NumElements(),
          errors::InvalidArgument(
              "Input to reshape is a tensor with ", input.NumElements(),
              " values, but the requested shape requires a multiple of ",
              product));
      shape.set_dim(unknown_index, missing);
    }
    OP_REQUIRES(context, shape.num_elements() == input.NumElements(),
                errors::InvalidArgument("Input to reshape is a tensor with ",
                                        input.NumElements(),
                                        " values, but the requested shape has ",
                                        shape.num_elements()));

    MklShape mkl_shape_input;
    GetMklShape(context, 0, &mkl_shape_input);
    bool input_in_mkl_format = mkl_shape_input.IsMklTensor();
    if (input_in_mkl_format) {
      TensorShape& shape_to = shape;
      TensorShape shape_from;
      for (size_t i = 0; i < mkl_shape_input.GetDimension(); i++) {
        // Outermost to innermost dimension
        shape_from.AddDim(
            mkl_shape_input.GetSizes()[mkl_shape_input.tf_dim_idx(i)]);
      }

      if (shape_from == shape_to) {
        CopyMklTensorInToOut(context, 0, 0);
        return;
      } else {
        // Allocate output tensor.
        Tensor* output_tensor = NULL;
        MklShape mkl_shape_output;
        mkl_shape_output.SetMklTensor(false);
        AllocateOutputSetMklShape(context, 0, &output_tensor, shape_to,
                                  mkl_shape_output);

        // Get output layout pointer.
        dnnLayout_t output_layout =
            static_cast<dnnLayout_t>(mkl_shape_input.GetTfLayout());

        // Execute DNNConversion.
        // Note: we  assume an MKL tensor always have float as its data type.
        void* input_buffer =
            static_cast<void*>(const_cast<float*>(input.flat<float>().data()));
        void* output_buffer = static_cast<void*>(
            const_cast<float*>(output_tensor->flat<float>().data()));
        mkl_shape_input.GetConvertedFlatData(output_layout, input_buffer,
                                             output_buffer);

        VLOG(1) << "MKLToTFConversion complete successfully.";
        return;
      }
    } else {
      CopyTfTensorInToOutWithShape(context, 0, 0, shape);
    }
  }
};

#define REGISTER_MKL_CPU(T)                                         \
  REGISTER_KERNEL_BUILDER(Name("_MklReshape")                       \
                              .Device(DEVICE_CPU)                   \
                              .HostMemory("shape")                  \
                              .TypeConstraint<T>("T")               \
                              .TypeConstraint<int32>("Tshape")      \
                              .Label(mkl_op_registry::kMklOpLabel), \
                          MklReshapeOp<CPUDevice, T>);
TF_CALL_float(REGISTER_MKL_CPU);
#undef REGISTER_MKL_CPU
}  // namespace tensorflow

#endif  // INTEL_MKL
