/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/kernels/rng_converter_utils.h"

#include <gtest/gtest.h>
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/rng_alg.h"

namespace tensorflow {
namespace {

TEST(RngConverterUtilsTest, DefaultRngForCPUEqualsGPU) {
  EXPECT_EQ(DefaultRngAlgForDeviceType(DEVICE_CPU_XLA_JIT),
            DefaultRngAlgForDeviceType(DEVICE_GPU_XLA_JIT));
}

TEST(RngConverterUtilsTest, UnknownDeviceIsDefault) {
  EXPECT_EQ(DefaultRngAlgForDeviceType(/*device_type_string=*/"UNKNOWN DEVICE"),
            xla::RandomAlgorithm::RNG_DEFAULT);
}

TEST(RngConverterUtilsTest, TensorflowAutoSelects) {
  EXPECT_EQ(ToTensorflowAlgorithm(xla::RandomAlgorithm::RNG_DEFAULT),
            tensorflow::RNG_ALG_AUTO_SELECT);
}

TEST(RngConverterUtilsTest, ToTensorflow) {
  EXPECT_EQ(ToTensorflowAlgorithm(xla::RandomAlgorithm::RNG_PHILOX),
            tensorflow::RNG_ALG_PHILOX);

  EXPECT_EQ(ToTensorflowAlgorithm(xla::RandomAlgorithm::RNG_THREE_FRY),
            tensorflow::RNG_ALG_THREEFRY);
}

}  // namespace
}  // namespace tensorflow
