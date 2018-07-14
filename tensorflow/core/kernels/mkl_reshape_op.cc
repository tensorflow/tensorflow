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


#ifndef INTEL_MKL_ML
#include "mkldnn.hpp"
using mkldnn::stream;
#else
#include "mkl_dnn.h"
#include "mkl_dnn_types.h"
#endif

#include "tensorflow/core/util/mkl_util.h"

namespace tensorflow {
using CPUDevice = Eigen::ThreadPoolDevice;
template <typename Device, typename T>
class MklReshapeOp : public OpKernel {
 public:
  explicit MklReshapeOp(OpKernelConstruction* context) : OpKernel(context) {}

#ifdef INTEL_MKL_ML
  void Compute(OpKernelContext* context) override {
    const Tensor& input = MklGetInput(context, 0);
    const Tensor& sizes = MklGetInput(context, 1);

    // Preliminary validation of sizes.
    OP_REQUIRES(context, IsLegacyVector(sizes.shape()),
                errors::InvalidArgument("sizes input must be 1-D, not shape ",
                                        sizes.shape().DebugString()));

    // Compute the output shape.  Determine product of specified
    // dimensions, and find the index of the unspecified one.
    TensorShape shape;
    int64 product = 1;
    int unknown_index = -1;
    switch (sizes.dtype()) {
      case DT_INT32:
        OP_REQUIRES_OK(context, ValidateSizes<int32>(sizes, &product,
                                                     &unknown_index, &shape));
        break;
      case DT_INT64:
        OP_REQUIRES_OK(context, ValidateSizes<int64>(sizes, &product,
                                                     &unknown_index, &shape));
        break;
      default:
        context->CtxFailure(errors::InvalidArgument(
            "desired shape must be a DT_INT32 or DT_INT64 vector, not a ",
            DataTypeString(sizes.dtype())));
        return;
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

#else

 private:
  // When the input tensor is in MKL layout and we are reshaping the tensor to a
  // different shape than its actual shape, then we use MKLDNN reorder primitive
  // to put tensor back in Tensorflow layout. But we can skip this reordering
  // some times. This function checks for all such cases.
  bool SkipReorder(const MklDnnShape& mkl_shape_input,
                   const TensorShape& reshape_to) {
    CHECK_EQ(mkl_shape_input.IsMklTensor(), true);
    bool ret = false;

    // If Tensorflow's data format and the underlying format maintained by
    // MKLDNN are equivalent (both are NHWC or both are NCHW), then we can
    // safely return true.
    auto input_mkl_md = mkl_shape_input.GetMklLayout();
    if (mkl_shape_input.GetTfDataFormat() == input_mkl_md.data.format) {
      ret = true;
    }

    return ret;
  }

 public:
  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = MklGetInput(context, 0);
    const Tensor& sizes = MklGetInput(context, 1);

    MklDnnShape mkl_shape_input;
    GetMklShape(context, kInputSlotIdx, &mkl_shape_input);
    bool input_in_mkl_format = mkl_shape_input.IsMklTensor();
    const int64 nelems = input_in_mkl_format
                             ? mkl_shape_input.GetTfShape().num_elements()
                             : input_tensor.NumElements();

    // Preliminary validation of sizes.
    OP_REQUIRES(context, IsLegacyVector(sizes.shape()),
                errors::InvalidArgument("sizes input must be 1-D, not shape ",
                                        sizes.shape().DebugString()));

    // Compute the output shape.  Determine product of specified
    // dimensions, and find the index of the unspecified one.
    TensorShape shape;
    int64 product = 1;
    int unknown_index = -1;
    switch (sizes.dtype()) {
      case DT_INT32:
        OP_REQUIRES_OK(context, ValidateSizes<int32>(sizes, &product,
                                                     &unknown_index, &shape));
        break;
      case DT_INT64:
        OP_REQUIRES_OK(context, ValidateSizes<int64>(sizes, &product,
                                                     &unknown_index, &shape));
        break;
      default:
        context->CtxFailure(errors::InvalidArgument(
            "desired shape must be a DT_INT32 or DT_INT64 vector, not a ",
            DataTypeString(sizes.dtype())));
        return;
    }
    if (unknown_index != -1) {
      OP_REQUIRES(
          context, product > 0,
          errors::InvalidArgument("Reshape cannot infer the missing input size "
                                  "for an empty tensor unless all specified "
                                  "input sizes are non-zero"));
      const int64 missing = nelems / product;
      OP_REQUIRES(
          context, product * missing == nelems,
          errors::InvalidArgument(
              "Input to reshape is a tensor with ", nelems,
              " values, but the requested shape requires a multiple of ",
              product));
      shape.set_dim(unknown_index, missing);
    }
    OP_REQUIRES(
        context, shape.num_elements() == nelems,
        errors::InvalidArgument("Input to reshape is a tensor with ", nelems,
                                " values, but the requested shape has ",
                                shape.num_elements()));

    if (input_in_mkl_format) {
      TensorShape& shape_to = shape;
      TensorShape shape_from = mkl_shape_input.GetTfShape();
      if (shape_from == shape_to) {
        CopyMklTensorInToOut(context, kInputSlotIdx, kOutputSlotIdx);
        return;
      } else {
        try {
          auto cpu_engine = engine(engine::cpu, 0);
          MklDnnData<T> dnn_data_input(&cpu_engine);
          // Reshape is just a logical view change operation for a tensor.
          // It does not change underlying layout. But MKLDNN may maintain
          // tensor data in different layout than that specified by Tensorflow.
          // If MKLDNN maintains input tensor in different layout than that
          // specified by Tensorflow, we will need to reorder tensor and then
          // put it in the shape expected by Tensorflow. But if MKLDNN has
          // maintained input tensor in the same layout as it is expected by
          // Tensorflow, we don't need to reorder tensor contents, we just
          // need to update MklDnnShape object associated with the input
          // tensor to reflect the shape change expected by reshape.
          if (!SkipReorder(mkl_shape_input, shape_to)) {
            // If dimensions that are being expanded or collapsed are not
            // maintained contiguously by MKLDNN, then we use reorder.

            // Get Mkl layout of input tensor.
            auto input_mkl_md = mkl_shape_input.GetMklLayout();
            // Set input Mkl layout as the user layout.
            dnn_data_input.SetUsrMem(input_mkl_md, &input_tensor);
            // Get expected Tensorflow layout of input tensor.
            auto output_tf_md = mkl_shape_input.GetTfLayout();
            auto output_tf_pd =
                memory::primitive_desc(output_tf_md, cpu_engine);

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
            if (dnn_data_input.CheckReorderToOpMem(output_tf_pd, output_tensor)) {
            } else {
              OP_REQUIRES(
                  context, output_tensor->CopyFrom(input_tensor, shape_to),
                  errors::InvalidArgument("invalid input tensor shape"));
            }
            return;
          } else {
            // If dimensions that are being expanded or collapsed are
            // maintained contiguously by MKLDNN, then we skip reorder, just
            // update MklDnnShape object for the tensorflow tensor, and forward
            // Tensorflow tensor as it is to the output.
            auto output_dims = TFShapeToMklDnnDims(shape_to);
            auto output_strides = CalculateTFStrides(output_dims);
            auto output_tf_md = MklDnnData<T>::CreateBlockedMemDesc(
                output_dims, output_strides);
            auto output_tf_pd =
                memory::primitive_desc(output_tf_md, cpu_engine);

            // Set MklDnnShape
            MklDnnShape mkl_shape_output;
            mkl_shape_output.SetMklTensor(true);
            mkl_shape_output.SetMklLayout(&output_tf_pd);
            mkl_shape_output.SetElemType(MklDnnType<T>());
            mkl_shape_output.SetTfLayout(output_dims.size(), output_dims,
                                         memory::format::blocked);

            // We now simply forward input Mkl tensor to output and change its
            // output MklDnnShape object.
            ForwardMklTensorInToOutWithMklShape(
                context, kInputSlotIdx, kOutputSlotIdx, mkl_shape_output);
            return;
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
    } else {
      // If input tensor is not in Mkl format, then just copy Tensorflow tensor
      // to output with specified shape.
      CopyTfTensorInToOutWithShape(context, kInputSlotIdx, kOutputSlotIdx,
                                   shape);
    }
  }

#endif  // INTEL_MKL_ML

 private:
  const int kInputSlotIdx = 0;
  const int kOutputSlotIdx = 0;

  template <typename Tshape>
  Status ValidateSizes(const Tensor& sizes, int64* product, int* unknown_index,
                       TensorShape* shape) {
    *product = 1;
    *unknown_index = -1;
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
      } else {
        shape->AddDim(size);
        (*product) *= size;
      }
    }
    return Status::OK();
  }
};

#define REGISTER_MKL_CPU(T)                                         \
  REGISTER_KERNEL_BUILDER(Name("_MklReshape")                       \
                              .Device(DEVICE_CPU)                   \
                              .HostMemory("shape")                  \
                              .TypeConstraint<T>("T")               \
                              .TypeConstraint<int32>("Tshape")      \
                              .Label(mkl_op_registry::kMklOpLabel), \
                          MklReshapeOp<CPUDevice, T>);              \
  REGISTER_KERNEL_BUILDER(Name("_MklReshape")                       \
                              .Device(DEVICE_CPU)                   \
                              .HostMemory("shape")                  \
                              .TypeConstraint<T>("T")               \
                              .TypeConstraint<int64>("Tshape")      \
                              .Label(mkl_op_registry::kMklOpLabel), \
                          MklReshapeOp<CPUDevice, T>);
TF_CALL_float(REGISTER_MKL_CPU);
#undef REGISTER_MKL_CPU
}  // namespace tensorflow

#endif  // INTEL_MKL
