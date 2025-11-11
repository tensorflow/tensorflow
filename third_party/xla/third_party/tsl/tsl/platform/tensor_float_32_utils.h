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

#ifndef TENSORFLOW_TSL_PLATFORM_TENSOR_FLOAT_32_UTILS_H_
#define TENSORFLOW_TSL_PLATFORM_TENSOR_FLOAT_32_UTILS_H_

namespace tsl {

// NOTE: The usage of this function is only supported through the Tensorflow
// framework.
// If TensorFloat-32 is enabled, float32 inputs of supported ops, such as
// `tf.linalg.matmul`, will be rounded from 23 bits of precision to 10 bits of
// precision in most cases. This allows the ops to execute much faster by
// utilizing the GPU's tensor cores. TensorFloat-32 has the same dynamic range
// as float32, meaning it is no more likely to underflow or overflow than
// float32. Ops still use float32 accumulation when TensorFloat-32 is enabled.
// Enabling or disabling TensorFloat-32 only affects Ampere GPUs and above.
// See enable_tensor_float_32_execution in python/framework/config.py for more
// details.
void enable_tensor_float_32_execution(bool enabled);

bool tensor_float_32_execution_enabled();

}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_TENSOR_FLOAT_32_UTILS_H_
