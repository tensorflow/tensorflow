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
      const TensorShape input_shape = input_tensor.shape();
      const TensorShape pos_shape = pos_tensor.shape();
      const TensorShape len_shape = len_tensor.shape();
      
      // Validate size of tensors
      OP_REQUIRES(context, pos_tensor.shape() == len_tensor.shape(),
                  errors::InvalidArgument("pos and len shapes must match: ",
                                           pos_shape.DebugString(), " vs. ",
                                           len_shape.DebugString()));
      
      if (TensorShapeUtils::IsScalar(pos_shape)) {
        // Perform Op with scalar pos/len
        auto input = input_tensor.flat<string>();
        T pos = pos_tensor.scalar<T>()();
        T len = len_tensor.scalar<T>()();

        // Allocate output
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output("output", input_tensor.shape(),
                                                &output_tensor));
        auto output = output_tensor->flat<string>();

        // Set output to be substrings of input strings
        for (size_t i = 0; i < input_tensor.NumElements(); ++i) {
          OP_REQUIRES(context, pos >= 0 && pos < input(i).size(),
                      errors::InvalidArgument("pos ", pos, 
                                              " out of range for string b'", 
                                              input(i), "' at index ", i));
          output(i) = input(i).substr(pos, len);
        }
      } else if (input_shape == pos_shape) {
        // Perform Op element-wise
        auto input = input_tensor.flat<string>();
        auto pos = pos_tensor.flat<T>();
        auto len = len_tensor.flat<T>();

        // Allocate output
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output("output", input_tensor.shape(),
                                                &output_tensor));
        auto output = output_tensor->flat<string>();

        // Set output to be substrings of input strings
        for (size_t i = 0; i < input_tensor.NumElements(); ++i) {
          OP_REQUIRES(context, pos(i) >= 0 && pos(i) < input(i).size(),
                      errors::InvalidArgument("pos ", pos(i), 
                                              " out of range for string b'", 
                                              input(i), "' at index ", i));
          output(i) = input(i).substr(pos(i), len(i));
        }
      } else {  
        // Attempt broadcasting this Op
        // context->SetStatus(errors::Unimplemented(
        //          "Substr broadcast is not supported yet."));

        BCast bcast(BCast::FromShape(input_shape), BCast::FromShape(pos_shape));
        OP_REQUIRES(context, bcast.IsValid(), 
                    errors::InvalidArgument("Incompatible shapes: ", 
                                            input_shape.DebugString(), " vs. ",
                                            pos_shape.DebugString()));
        TensorShape output_shape = BCast::ToShape(bcast.result_shape());
        int ndims = output_shape.dims();
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output("output", output_shape,
                                                &output_tensor));
        switch (ndims) {
          case 2: {
            auto output = output_tensor->shaped<string, 2>(bcast.x_reshape());
            auto input_reshaped = input_tensor.shaped<string, 2>(bcast.x_reshape());
            auto pos_reshaped = pos_tensor.shaped<T, 2>(bcast.y_reshape());
            auto len_reshaped = len_tensor.shaped<T, 2>(bcast.y_reshape());
              
            typename TTypes<string, 2>::Tensor input = input_reshaped.broadcast(BCast::ToIndexArray<2>(bcast.x_bcast()))(0);
            typename TTypes<T, 2>::Tensor pos = pos_reshaped.broadcast(BCast::ToIndexArray<2>(bcast.y_bcast()))(0);
            typename TTypes<T, 2>::Tensor len = len_reshaped.broadcast(BCast::ToIndexArray<2>(bcast.y_bcast()))(0);
            
            for (size_t i = 0; i < ndims; ++i) {
              size_t dim_size = output_shape.dim_size(i);
              for (size_t j = 0; j < dim_size; ++j) {
                output(i, j) = input(i, j).substr(pos(i, j), len(i, j));
              }
            }
          }
          default: {
            context->SetStatus(errors::InvalidArgument(
                    "Broadcast rank not supported: ", ndims));
          }
        }
      }
    }
};

#define REGISTER_SUBSTR(type)                         \
  REGISTER_KERNEL_BUILDER(Name("Substr")              \
                          .Device(DEVICE_CPU)         \
                          .TypeConstraint<type>("T"), \
                          SubstrOp<type>);
REGISTER_SUBSTR(int32);
REGISTER_SUBSTR(int64);

}  // namespace tensorflow
