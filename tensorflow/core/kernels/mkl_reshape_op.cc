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

#include "mkldnn.hpp"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/mkl_types.h"
#include "tensorflow/core/util/mkl_util.h"

using mkldnn::stream;

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;

template <typename Device, typename T>
class MklReshapeOp : public OpKernel {
 public:
  explicit MklReshapeOp(OpKernelConstruction* context) : OpKernel(context) {}

 private:
  // When the input tensor is in MKL layout and we are reshaping the tensor to a
  // different shape than its actual shape, then we use MKLDNN reorder primitive
  // to put tensor back in Tensorflow layout. But we can skip this reordering
  // some times. This function checks for all such cases.
  bool SkipReorder(const MklDnnShape& mkl_shape_input,
                   const TensorShape& reshape_to) {
    CHECK_EQ(mkl_shape_input.IsMklTensor(), true);

    // If Tensorflow's data format and the underlying format maintained by
    // MKLDNN are equivalent (both are NHWC or both are NCHW), then we can
    // safely return true.
    // @todo: Future do not force skip reorder for all blocked format. Use
    // blocking_desc_is_equal() for checking all the stride arrays in
    // mkl-dnn/blob/master/src/common/type_helpers.hpp
    auto input_mkl_md = mkl_shape_input.GetMklLayout();
    return SKIP_INPUT_REORDER(mkl_shape_input, input_mkl_md);
  }

 public:
  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = MklGetInput(context, 0);
    const Tensor& sizes = MklGetInput(context, 1);

    MklDnnShape mkl_shape_input;
    GetMklShape(context, kInputSlotIdx, &mkl_shape_input);
    bool input_in_mkl_format = mkl_shape_input.IsMklTensor();
    TensorShape input_shape = input_in_mkl_format ? mkl_shape_input.GetTfShape()
                                                  : input_tensor.shape();
    const int64 nelems = input_in_mkl_format ? input_shape.num_elements()
                                             : input_tensor.NumElements();

    // Preliminary validation of sizes.
    OP_REQUIRES(context, TensorShapeUtils::IsVector(sizes.shape()),
                errors::InvalidArgument("sizes input must be 1-D, not shape ",
                                        sizes.shape().DebugString()));

    // Compute the output shape.  Determine product of specified
    // dimensions, and find the index of the unspecified one.
    TensorShape shape;
    int64 product = 1;
    int unknown_index = -1;
    bool sizes_has_zero_dim = false;
    switch (sizes.dtype()) {
      case DT_INT32:
        OP_REQUIRES_OK(context,
                       ValidateSizes<int32>(sizes, &product, &unknown_index,
                                            &shape, &sizes_has_zero_dim));
        break;
      case DT_INT64:
        OP_REQUIRES_OK(context,
                       ValidateSizes<int64>(sizes, &product, &unknown_index,
                                            &shape, &sizes_has_zero_dim));
        break;
      default:
        context->CtxFailure(errors::InvalidArgument(
            "desired shape must be a DT_INT32 or DT_INT64 vector, not a ",
            DataTypeString(sizes.dtype())));
        return;
    }
    if (unknown_index != -1) {
      int64 input_num_elements = 1;
      bool input_has_zero_dim = false;
      for (int dim = 0; dim < input_shape.dims(); ++dim) {
        // For zero dimension, we don't count it into `input_num_elements`
        // unless `sizes` has no zero dimension, so we are still able to
        // infer shapes for other dimensions.
        if (input_shape.dim_size(dim) > 0 || !sizes_has_zero_dim) {
          input_num_elements *= input_shape.dim_size(dim);
        } else {
          input_has_zero_dim = true;
        }
      }

      const int64 missing = input_num_elements / product;
      if (!input_has_zero_dim) {
        OP_REQUIRES(
            context, product * missing == input_num_elements,
            errors::InvalidArgument(
                "Input to reshape is a tensor with ", input_num_elements,
                " values, but the requested shape requires a multiple of ",
                product));
      }
      shape.set_dim(unknown_index, missing);
    }
    OP_REQUIRES(
        context, shape.num_elements() == nelems,
        errors::InvalidArgument("Input to reshape is a tensor with ", nelems,
                                " values, but the requested shape has ",
                                shape.num_elements()));

    if (input_in_mkl_format && !SkipReorder(mkl_shape_input, shape)) {
      TensorShape& shape_to = shape;
      TensorShape shape_from = mkl_shape_input.GetTfShape();
      if (shape_from == shape_to) {
        CopyMklTensorInToOut(context, kInputSlotIdx, kOutputSlotIdx);
        return;
      } else {
        try {
          auto cpu_engine = engine(ENGINE_CPU, 0);
          MklDnnData<T> dnn_data_input(&cpu_engine);
          // Reshape is just a logical view change operation for a tensor.
          // It does not change underlying layout. But MKLDNN may maintain
          // tensor data in different layout than that specified by Tensorflow.
          // If MKLDNN maintains input tensor in different layout than that
          // specified by Tensorflow, we will need to reorder tensor and then
          // put it in the shape expected by Tensorflow.

          // If dimensions that are being expanded or collapsed are not
          // maintained contiguously by MKLDNN, then we use reorder.

          // Get Mkl layout of input tensor.
          auto input_mkl_md = mkl_shape_input.GetMklLayout();
          // Set input Mkl layout as the user layout.
          dnn_data_input.SetUsrMem(input_mkl_md, &input_tensor);
          // Get expected Tensorflow layout of input tensor.
          auto output_tf_md = mkl_shape_input.GetTfLayout();
#ifndef ENABLE_MKLDNN_V1
          auto output_tf_pd = memory::primitive_desc(output_tf_md, cpu_engine);
#endif  // !ENABLE_MKLDNN_V1

          Tensor* output_tensor = nullptr;
          MklDnnShape mkl_shape_output;
          mkl_shape_output.SetMklTensor(false);
          // We allocate output tensor in the shape expected by Reshape.
          AllocateOutputSetMklShape(context, kOutputSlotIdx, &output_tensor,
                                    shape_to, mkl_shape_output);

          // Insert reorder between Mkl layout and TensorFlow layout if
          // needed. If reorder is not needed but reshape is needed (since
          // shape_from != shape_to), then we just copy input tensor to
          // output tensor with target shape (we cannot forward Mkl layout
          // in such case because shape has changed.)
          if (dnn_data_input.CheckReorderToOpMem(OUTPUT_TF_MD, output_tensor)) {
          } else {
            OP_REQUIRES(context,
                        output_tensor->CopyFrom(input_tensor, shape_to),
                        errors::InvalidArgument("invalid input tensor shape"));
          }
          return;
        } catch (mkldnn::error& e) {
          string error_msg = "Status: " + std::to_string(e.status) +
                             ", message: " + string(e.message) + ", in file " +
                             string(__FILE__) + ":" + std::to_string(__LINE__);
          OP_REQUIRES_OK(
              context,
              errors::Aborted("Operation received an exception:", error_msg));
        }
      }
    } else {
      // If input tensor is not in Mkl format, then just copy Tensorflow tensor
      // to output with specified shape.
      CopyTfTensorInToOutWithShape(context, kInputSlotIdx, kOutputSlotIdx,
                                   shape);
    }
  }

 private:
  const int kInputSlotIdx = 0;
  const int kOutputSlotIdx = 0;

  template <typename Tshape>
  Status ValidateSizes(const Tensor& sizes, int64* product, int* unknown_index,
                       TensorShape* shape, bool* has_zero_dim) {
    *product = 1;
    *unknown_index = -1;
    *has_zero_dim = false;
    const int64 num_dims = sizes.NumElements();
    auto Svec = sizes.flat<Tshape>();
    for (int d = 0; d < num_dims; ++d) {
      const Tshape size = Svec(d);
      if (size == -1) {
        if (*unknown_index != -1) {
          return errors::InvalidArgument(
              "Only one input size may be -1, not both ", *unknown_index,
              " and ", d);
        }
        *unknown_index = d;
        shape->AddDim(1);
      } else if (size < 0) {
        return errors::InvalidArgument("Size ", d,
                                       " must be non-negative, not ", size);
      } else if (size == 0) {
        // We don't include zero-sized dimension in product, so that we can
        // still calculate number of elements for non-zero-sized dimensions and
        // therefore infer their shapes.
        shape->AddDim(size);
        *has_zero_dim = true;
      } else {
        shape->AddDim(size);
        (*product) *= size;
      }
    }
    return Status::OK();
  }
};

#define REGISTER_MKL_CPU(T)                                    \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklReshape")                                      \
          .Device(DEVICE_CPU)                                  \
          .HostMemory("shape")                                 \
          .TypeConstraint<T>("T")                              \
          .TypeConstraint("Tshape", {DT_INT32, DT_INT64})      \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklReshapeOp<CPUDevice, T>);

TF_CALL_float(REGISTER_MKL_CPU);
TF_CALL_bfloat16(REGISTER_MKL_CPU);

#undef REGISTER_MKL_CPU

}  // namespace tensorflow

#endif  // INTEL_MKL
