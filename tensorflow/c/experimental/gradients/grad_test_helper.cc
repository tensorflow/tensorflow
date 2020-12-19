/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/c/experimental/gradients/grad_test_helper.h"

#include "tensorflow/c/eager/gradient_checker.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace gradients {
namespace internal {

void CompareNumericalAndAutodiffGradients(
    Model model, Model grad_model, AbstractContext* ctx,
    absl::Span<AbstractTensorHandle* const> inputs, bool use_function,
    double abs_error) {
  auto num_inputs = inputs.size();
  std::vector<AbstractTensorHandle*> outputs(num_inputs);
  auto s = RunModel(grad_model, ctx, inputs, absl::MakeSpan(outputs),
                    /*use_function=*/use_function);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  for (int i = 0; i < num_inputs; ++i) {
    if (!outputs[i]) continue;

    AbstractTensorHandlePtr numerical_grad;
    {
      AbstractTensorHandle* numerical_grad_raw;
      s = CalcNumericalGrad(ctx, model, inputs,
                            /*input_index=*/i, use_function,
                            &numerical_grad_raw);
      ASSERT_EQ(errors::OK, s.code()) << s.error_message();
      numerical_grad.reset(numerical_grad_raw);
    }

    TF_Tensor* numerical_tensor;
    s = GetValue(numerical_grad.get(), &numerical_tensor);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    auto num_elem_numerical = TF_TensorElementCount(numerical_tensor);

    TF_Tensor* analytical_tensor;
    s = GetValue(outputs[i], &analytical_tensor);
    ASSERT_EQ(errors::OK, s.code()) << s.error_message();
    auto num_elem_analytical = TF_TensorElementCount(analytical_tensor);

    ASSERT_EQ(num_elem_numerical, num_elem_analytical);

    float* dnumerical = new float[num_elem_numerical]{0};
    memcpy(&dnumerical[0], TF_TensorData(numerical_tensor),
           TF_TensorByteSize(numerical_tensor));
    float* danalytical = new float[num_elem_analytical]{0};
    memcpy(&danalytical[0], TF_TensorData(analytical_tensor),
           TF_TensorByteSize(analytical_tensor));

    for (int j = 0; j < num_elem_numerical; j++) {
      ASSERT_NEAR(dnumerical[j], danalytical[j], abs_error);
    }
    TF_DeleteTensor(analytical_tensor);
    TF_DeleteTensor(numerical_tensor);
    delete[] danalytical;
    delete[] dnumerical;
    outputs[i]->Unref();
  }
}

}  // namespace internal
}  // namespace gradients
}  // namespace tensorflow
