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
      
      if (!TensorShapeUtils::IsScalar(pos_shape) && input_shape != pos_shape) {
        // This Op currently only supports either scalar pos/len or pos/len with 
        // shapes that match the input tensor.
        context->SetStatus(errors::Unimplemented(
                 "Substr broadcast is not yet supported."));
      }
      
      // Reshape input 
      auto input = input_tensor.flat<string>();
      // Allocate output
      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output("output", input_tensor.shape(),
                                              &output_tensor));
      auto output = output_tensor->flat<string>();
      
      if (TensorShapeUtils::IsScalar(pos_shape)) {
        // Perform Op with scalar pos/len
        T pos = pos_tensor.scalar<T>()();
        T len = len_tensor.scalar<T>()();
        for (size_t i = 0; i < input_tensor.NumElements(); ++i) {
          // Make sure pos won't cause a runtime error
          OP_REQUIRES(context, pos >= 0 && pos < input(i).size(),
                      errors::InvalidArgument("pos ", pos, 
                                              " out of range for string b'", 
                                              input(i), "' at index ", i));
          output(i) = input(i).substr(pos, len);
        }
      } else if (input_shape == pos_shape) {
        // Perform Op element-wise
        auto pos = pos_tensor.flat<T>();
        auto len = len_tensor.flat<T>();
        for (size_t i = 0; i < input_tensor.NumElements(); ++i) {
          // Make sure pos won't cause a runtime error
          OP_REQUIRES(context, pos(i) >= 0 && pos(i) < input(i).size(),
                      errors::InvalidArgument("pos ", pos(i), 
                                              " out of range for string b'", 
                                              input(i), "' at index ", i));
          output(i) = input(i).substr(pos(i), len(i));
        }
      } else {
        // TODO: Create broadcast version of this operation
        //
        // Can't use BinaryOp pattern found in cwise_ops_common.h, as Substr
        // has three inputs. It may be worth waiting until ternary broadcasting
        // is implemented before attempting this.
        context->SetStatus(errors::Unimplemented(
                 "Substr broadcast is not yet supported."));  
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
