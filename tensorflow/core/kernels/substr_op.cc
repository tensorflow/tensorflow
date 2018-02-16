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

#include <string>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {

// Position/length can be 32 or 64-bit integers
template <typename T>
class SubstrOp : public OpKernel {
 public:
  using OpKernel::OpKernel;

  void Compute(OpKernelContext* context) override {
    // Get inputs
    const Tensor& input_tensor = context->input(0);
    const Tensor& pos_tensor = context->input(1);
    const Tensor& len_tensor = context->input(2);
    const TensorShape& input_shape = input_tensor.shape();
    const TensorShape& pos_shape = pos_tensor.shape();

    bool is_scalar = TensorShapeUtils::IsScalar(pos_shape);

    if (is_scalar || input_shape == pos_shape) {
      // pos/len are either scalar or match the shape of input_tensor
      // Do not need to do broadcasting

      // Reshape input
      auto input = input_tensor.flat<string>();
      // Allocate output
      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output("output", input_tensor.shape(),
                                              &output_tensor));
      auto output = output_tensor->flat<string>();
      if (is_scalar) {
        // Perform Op with scalar pos/len
        const T pos =
            tensorflow::internal::SubtleMustCopy(pos_tensor.scalar<T>()());
        const T len =
            tensorflow::internal::SubtleMustCopy(len_tensor.scalar<T>()());
        for (size_t i = 0; i < input_tensor.NumElements(); ++i) {
          string in = input(i);
          OP_REQUIRES(
              context, FastBoundsCheck(pos, in.size() + 1),
              errors::InvalidArgument("pos ", pos, " out of range for string",
                                      "b'", in, "' at index ", i));
          output(i) = in.substr(pos, len);
        }
      } else {
        // Perform Op element-wise with tensor pos/len
        auto pos_flat = pos_tensor.flat<T>();
        auto len_flat = len_tensor.flat<T>();
        for (size_t i = 0; i < input_tensor.NumElements(); ++i) {
          string in = input(i);
          const T pos = tensorflow::internal::SubtleMustCopy(pos_flat(i));
          const T len = tensorflow::internal::SubtleMustCopy(len_flat(i));
          OP_REQUIRES(
              context, FastBoundsCheck(pos, in.size() + 1),
              errors::InvalidArgument("pos ", pos, " out of range for string",
                                      "b'", in, "' at index ", i));
          output(i) = in.substr(pos, len);
        }
      }
    } else {
      // Perform op with broadcasting
      // TODO: Use ternary broadcasting for once available in Eigen. Current
      //       implementation iterates through broadcasted ops element-wise;
      //       this should be parallelized.

      // Create BCast helper with shape of input and pos/len
      BCast bcast(BCast::FromShape(input_shape), BCast::FromShape(pos_shape));
      OP_REQUIRES(context, bcast.IsValid(),
                  errors::InvalidArgument(
                      "Incompatible shapes: ", input_shape.DebugString(),
                      " vs. ", pos_shape.DebugString()));
      TensorShape output_shape = BCast::ToShape(bcast.result_shape());
      int ndims = output_shape.dims();
      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output("output", output_shape,
                                                       &output_tensor));
      switch (ndims) {
        case 1: {
          // Reshape tensors according to BCast results
          auto input = input_tensor.shaped<string, 1>(bcast.x_reshape());
          auto output = output_tensor->shaped<string, 1>(bcast.result_shape());
          auto pos_shaped = pos_tensor.shaped<T, 1>(bcast.y_reshape());
          auto len_shaped = len_tensor.shaped<T, 1>(bcast.y_reshape());

          // Allocate temporary buffer for broadcasted input tensor
          Tensor input_buffer;
          OP_REQUIRES_OK(context, context->allocate_temp(
                                      DT_STRING, output_shape, &input_buffer));
          TTypes<string, 1>::Tensor input_bcast =
              input_buffer.shaped<string, 1>(bcast.result_shape());
          input_bcast =
              input.broadcast(BCast::ToIndexArray<1>(bcast.x_bcast()));

          // Allocate temporary buffer for broadcasted position tensor
          Tensor pos_buffer;
          OP_REQUIRES_OK(context,
                         context->allocate_temp(DataTypeToEnum<T>::v(),
                                                output_shape, &pos_buffer));
          typename TTypes<T, 1>::Tensor pos_bcast(
              pos_buffer.shaped<T, 1>(bcast.result_shape()));
          pos_bcast =
              pos_shaped.broadcast(BCast::ToIndexArray<1>(bcast.y_bcast()));

          // Allocate temporary buffer for broadcasted length tensor
          Tensor len_buffer;
          OP_REQUIRES_OK(context,
                         context->allocate_temp(DataTypeToEnum<T>::v(),
                                                output_shape, &len_buffer));
          typename TTypes<T, 1>::Tensor len_bcast(
              len_buffer.shaped<T, 1>(bcast.result_shape()));
          len_bcast =
              len_shaped.broadcast(BCast::ToIndexArray<1>(bcast.y_bcast()));

          // Iterate through broadcasted tensors and perform substr
          for (int i = 0; i < output_shape.dim_size(0); ++i) {
            string in = input_bcast(i);
            const T pos = tensorflow::internal::SubtleMustCopy(pos_bcast(i));
            const T len = tensorflow::internal::SubtleMustCopy(len_bcast(i));
            OP_REQUIRES(
                context, FastBoundsCheck(pos, input_bcast(i).size() + 1),
                errors::InvalidArgument("pos ", pos, " out of range for string",
                                        "b'", in, "' at index ", i));
            output(i) = in.substr(pos, len);
          }
          break;
        }
        case 2: {
          // Reshape tensors according to BCast results
          auto input = input_tensor.shaped<string, 2>(bcast.x_reshape());
          auto output = output_tensor->shaped<string, 2>(bcast.result_shape());
          auto pos_shaped = pos_tensor.shaped<T, 2>(bcast.y_reshape());
          auto len_shaped = len_tensor.shaped<T, 2>(bcast.y_reshape());

          // Allocate temporary buffer for broadcasted input tensor
          Tensor input_buffer;
          OP_REQUIRES_OK(context, context->allocate_temp(
                                      DT_STRING, output_shape, &input_buffer));
          TTypes<string, 2>::Tensor input_bcast =
              input_buffer.shaped<string, 2>(bcast.result_shape());
          input_bcast =
              input.broadcast(BCast::ToIndexArray<2>(bcast.x_bcast()));

          // Allocate temporary buffer for broadcasted position tensor
          Tensor pos_buffer;
          OP_REQUIRES_OK(context,
                         context->allocate_temp(DataTypeToEnum<T>::v(),
                                                output_shape, &pos_buffer));
          typename TTypes<T, 2>::Tensor pos_bcast(
              pos_buffer.shaped<T, 2>(bcast.result_shape()));
          pos_bcast =
              pos_shaped.broadcast(BCast::ToIndexArray<2>(bcast.y_bcast()));

          // Allocate temporary buffer for broadcasted length tensor
          Tensor len_buffer;
          OP_REQUIRES_OK(context,
                         context->allocate_temp(DataTypeToEnum<T>::v(),
                                                output_shape, &len_buffer));
          typename TTypes<T, 2>::Tensor len_bcast(
              len_buffer.shaped<T, 2>(bcast.result_shape()));
          len_bcast =
              len_shaped.broadcast(BCast::ToIndexArray<2>(bcast.y_bcast()));

          // Iterate through broadcasted tensors and perform substr
          for (int i = 0; i < output_shape.dim_size(0); ++i) {
            for (int j = 0; j < output_shape.dim_size(1); ++j) {
              string in = input_bcast(i, j);
              const T pos =
                  tensorflow::internal::SubtleMustCopy(pos_bcast(i, j));
              const T len =
                  tensorflow::internal::SubtleMustCopy(len_bcast(i, j));
              OP_REQUIRES(context, FastBoundsCheck(pos, in.size() + 1),
                          errors::InvalidArgument(
                              "pos ", pos, " out of range for ", "string b'",
                              in, "' at index (", i, ", ", j, ")"));
              output(i, j) = in.substr(pos, len);
            }
          }
          break;
        }
        default: {
          context->SetStatus(errors::Unimplemented(
              "Substr broadcast not implemented for ", ndims, " dimensions"));
        }
      }
    }
  }
};

#define REGISTER_SUBSTR(type)                                      \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("Substr").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SubstrOp<type>);
REGISTER_SUBSTR(int32);
REGISTER_SUBSTR(int64);
}  // namespace tensorflow
