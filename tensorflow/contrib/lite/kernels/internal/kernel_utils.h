/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_KERNEL_UTILS_H_
#define TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_KERNEL_UTILS_H_

#include "tensorflow/contrib/lite/builtin_op_data.h"

namespace tflite {
namespace kernel_utils {

// Performs an RNN batch inference step for inputs specified by input_ptr_batch.
// The RNN cell is specified by the pointers to its input and recurrent weights,
// and biases, along with the input size, number of units, activation.
//
// The pointers to the hidden state and the output are updated as a result.
//
// The pointers with the suffix "_batch" point to data aligned in batch_major
// order, and each step processes batch_size many inputs from input_ptr_batch,
// and updates batch_size many outputs and hidden states.
void RnnBatchStep(const float* input_ptr_batch, const float* input_weights_ptr,
                  const float* recurrent_weights_ptr, const float* bias_ptr,
                  int input_size, int num_units, int batch_size,
                  TfLiteFusedActivation activation,
                  float* hidden_state_ptr_batch, float* output_ptr_batch);

}  // namespace kernel_utils
}  // namespace tflite
#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_KERNEL_UTILS_H_
