/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/cl/kernels/cl_test.h"
#include "tensorflow/lite/delegates/gpu/common/gpu_model_test_util.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

TEST_F(OpenCLOperationTest, LinkingConvolutionAndCosOp) {
  auto status = TestLinkingConvolutionAndCosOp(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

TEST_F(OpenCLOperationTest, LinkingConvolution2InputMul2InputMul) {
  auto status = TestLinkingConvolution2InputMul2InputMul(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

TEST_F(OpenCLOperationTest, LinkingConvolution2InputBroadcastMul2InputMul) {
  auto status = TestLinkingConvolution2InputBroadcastMul2InputMul(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

TEST_F(OpenCLOperationTest, LinkingConvolution2InputMul2InputBroadcastMul) {
  auto status = TestLinkingConvolution2InputMul2InputBroadcastMul(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

TEST_F(OpenCLOperationTest, LinkingConvolution2InputMul2InputMulCos) {
  auto status = TestLinkingConvolution2InputMul2InputMulCos(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

TEST_F(OpenCLOperationTest, LinkingConvolutionFirstTanh2InputDiff) {
  auto status = TestLinkingConvolutionFirstTanh2InputDiff(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

TEST_F(OpenCLOperationTest, LinkingConvolutionSecondTanh2InputDiff) {
  auto status = TestLinkingConvolutionSecondTanh2InputDiff(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

TEST_F(OpenCLOperationTest, LinkingConvolutionFirstTanhSecondCos2InputDiff) {
  auto status = TestLinkingConvolutionFirstTanhSecondCos2InputDiff(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

TEST_F(OpenCLOperationTest, LinkingComplex0) {
  auto status = TestLinkingComplex0(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

TEST_F(OpenCLOperationTest, LinkingConvElem2InputAddElemsOp) {
  auto status = TestLinkingConvElem2InputAddElemsOp(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

}  // namespace
}  // namespace cl
}  // namespace gpu
}  // namespace tflite
