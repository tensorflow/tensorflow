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

#include "xla/tsl/profiler/convert/xla_op_utils.h"

#include <gtest/gtest.h>
#include "xla/tsl/platform/test.h"

namespace tsl {
namespace profiler {
namespace {

TEST(XlaOpUtilsTest, HloModuleNameWithProgramId) {
  EXPECT_EQ("module(123)", HloModuleNameWithProgramId("module", 123));
}

TEST(XlaOpUtilsTest, IsHloRematerialization) {
  EXPECT_FALSE(IsHloRematerialization("%fusion.4848 = %reshape.19311.remat"));
  EXPECT_TRUE(IsHloRematerialization("%convolution.5.remat"));
  EXPECT_TRUE(IsHloRematerialization("%convolution.4.remat = %abc"));
}

TEST(XlaOpUtilsTest, IsFrameworkRematerialization) {
  EXPECT_TRUE(IsFrameworkRematerialization(
      "test_function_name/rematted_computation/dot_general"));
  EXPECT_FALSE(
      IsFrameworkRematerialization("test_function_name/fusion/dot_general"));
  EXPECT_FALSE(IsFrameworkRematerialization(
      "test_function_name_rematted_computation/reshape/dot_general"));
}

TEST(XlaOpUtilsTest, IsRematerialization) {
  EXPECT_TRUE(IsRematerialization(
      "%convolution.5.remat",
      "test_function_name/rematted_computation/dot_general"));
  EXPECT_TRUE(IsRematerialization(
      "%convolution.5", "test_function_name/rematted_computation/dot_general"));
  EXPECT_TRUE(IsRematerialization("%convolution.5.remat",
                                  "test_function_name/reshape/dot_general"));
  EXPECT_FALSE(IsRematerialization("%convolution.5",
                                   "test_function_name/reshape/dot_general"));
}

TEST(XlaOpUtilsTest, IsHostOrSparseCoreV0Infeed) {
  EXPECT_TRUE(IsHostOrSparseCoreV0Infeed(kHloInfeed));
  EXPECT_TRUE(IsHostOrSparseCoreV0Infeed(kHloSparseCoreV0Infeed));
  EXPECT_FALSE(IsHostOrSparseCoreV0Infeed(kHloSparseCoreV0InfeedWait));
  EXPECT_FALSE(IsHostOrSparseCoreV0Infeed(kHloSparseCoreV0InfeedTransform));
}

TEST(XlaOpUtilsTest, TfOpFullname) {
  EXPECT_EQ("", TfOpFullname("", ""));
  EXPECT_EQ("XLA_Args:XLA_Args", TfOpFullname("", "XLA_Args"));
  EXPECT_EQ("XLA_Retvals:op_type", TfOpFullname("op_type", "XLA_Retvals"));
  EXPECT_EQ("op_name:op_type", TfOpFullname("op_type", "op_name"));
}

TEST(XlaOpUtilsTest, IsXlaArgsOrRetvals) {
  EXPECT_TRUE(IsXlaArgsOrRetvals("XLA_Args"));
  EXPECT_TRUE(IsXlaArgsOrRetvals("XLA_Retvals"));
  EXPECT_FALSE(IsXlaArgsOrRetvals("op_type"));
  EXPECT_FALSE(IsXlaArgsOrRetvals("op_name"));
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
