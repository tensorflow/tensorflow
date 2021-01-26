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
#ifndef TENSORFLOW_C_EXPERIMENTAL_GRADIENTS_GRAD_TEST_HELPER_H_
#define TENSORFLOW_C_EXPERIMENTAL_GRADIENTS_GRAD_TEST_HELPER_H_

#include "absl/random/random.h"
#include "tensorflow/c/eager/gradients.h"
#include "tensorflow/c/eager/unified_api_testutil.h"

namespace tensorflow {
namespace gradients {
namespace internal {

void CompareNumericalAndAutodiffGradients(
    Model model, Model grad_model, AbstractContext* ctx,
    absl::Span<AbstractTensorHandle* const> inputs, bool use_function,
    double abs_error = 1e-2);

void CheckTensorValue(AbstractTensorHandle* t, absl::Span<const float> manuals,
                      absl::Span<const int64_t> dims, double abs_error = 1e-2);

Model BuildGradModel(Model forward, GradientRegistry registry);

absl::BitGen& GetBitGen();

template <class T, TF_DataType datatype>
Status TestTensorHandleWithDimsRandom(AbstractContext* ctx, const T lower,
                                      const T upper,
                                      absl::Span<const int64_t> dims,
                                      AbstractTensorHandle** tensor) {
  int num_dims = dims.size();
  int64_t num_elems = 1;
  for (size_t i{}; i < num_dims; ++i) {
    num_elems *= dims[i];
  }
  std::unique_ptr<T[]> tensor_data(new T[num_elems]);
  for (size_t i{}; i < num_elems; ++i) {
    tensor_data[i] =
        absl::Uniform<T>(absl::IntervalOpen, GetBitGen(), lower, upper);
  }
  return TestTensorHandleWithDims<T, datatype>(ctx, tensor_data.get(),
                                               dims.data(), num_dims, tensor);
}

template <class T, TF_DataType datatype>
Status TestScalarTensorHandleRandom(AbstractContext* ctx, const T lower,
                                    const T upper,
                                    AbstractTensorHandle** tensor) {
  return TestScalarTensorHandle<T, datatype>(
      ctx, absl::Uniform<T>(absl::IntervalOpen, GetBitGen(), lower, upper),
      tensor);
}

}  // namespace internal
}  // namespace gradients
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_GRADIENTS_GRAD_TEST_HELPER_H_
