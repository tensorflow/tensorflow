/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/tools/benchmark/partition_report.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace tflite {
namespace benchmark {
namespace {

using ::testing::HasSubstr;

// A custom op that XNNPACK cannot delegate; copies its input to its output.
TfLiteRegistration FallbackOpRegistration() {
  TfLiteRegistration registration = {};
  registration.builtin_code = kTfLiteBuiltinCustom;
  registration.custom_name = "FALLBACK_OP";
  registration.prepare = [](TfLiteContext* context, TfLiteNode* node) {
    const TfLiteTensor* input = &context->tensors[node->inputs->data[0]];
    TfLiteTensor* output = &context->tensors[node->outputs->data[0]];
    return context->ResizeTensor(context, output,
                                 TfLiteIntArrayCopy(input->dims));
  };
  registration.invoke = [](TfLiteContext* context, TfLiteNode* node) {
    const TfLiteTensor* input = &context->tensors[node->inputs->data[0]];
    TfLiteTensor* output = &context->tensors[node->outputs->data[0]];
    std::memcpy(output->data.raw, input->data.raw, input->bytes);
    return kTfLiteOk;
  };
  return registration;
}

TfLiteRegistration AddRegistration() {
  TfLiteRegistration registration = *ops::builtin::Register_ADD();
  registration.builtin_code = kTfLiteBuiltinAdd;
  return registration;
}

void* NewAddParams() {
  auto* params =
      static_cast<TfLiteAddParams*>(calloc(1, sizeof(TfLiteAddParams)));
  params->activation = kTfLiteActNone;
  return params;
}

// Builds t0 -> ADD -> t1 -> FALLBACK_OP -> t2 -> ADD -> t3 with float [1, 4]
// tensors, so that XNNPACK produces two partitions around one fallback node.
class PartitionReportTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(interpreter_.AddTensors(4), kTfLiteOk);
    for (int i = 0; i < 4; ++i) {
      ASSERT_EQ(interpreter_.SetTensorParametersReadWrite(
                    i, kTfLiteFloat32, "t", {1, 4}, TfLiteQuantization()),
                kTfLiteOk);
    }
    ASSERT_EQ(interpreter_.SetInputs({0}), kTfLiteOk);
    ASSERT_EQ(interpreter_.SetOutputs({3}), kTfLiteOk);
    add_registration_ = AddRegistration();
    fallback_registration_ = FallbackOpRegistration();
    ASSERT_EQ(interpreter_.AddNodeWithParameters(
                  {0, 0}, {1}, nullptr, 0, NewAddParams(), &add_registration_),
              kTfLiteOk);
    ASSERT_EQ(interpreter_.AddNodeWithParameters({1}, {2}, nullptr, 0, nullptr,
                                                 &fallback_registration_),
              kTfLiteOk);
    ASSERT_EQ(interpreter_.AddNodeWithParameters(
                  {2, 2}, {3}, nullptr, 0, NewAddParams(), &add_registration_),
              kTfLiteOk);
  }

  void ApplyXnnpack() {
    TfLiteXNNPackDelegateOptions options =
        TfLiteXNNPackDelegateOptionsDefault();
    delegate_.reset(TfLiteXNNPackDelegateCreate(&options));
    ASSERT_EQ(interpreter_.ModifyGraphWithDelegate(delegate_.get()), kTfLiteOk);
  }

  Interpreter interpreter_;
  TfLiteRegistration add_registration_;
  TfLiteRegistration fallback_registration_;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      delegate_{nullptr, TfLiteXNNPackDelegateDelete};
};

TEST_F(PartitionReportTest, NoDelegateIsOneFallbackIsland) {
  ASSERT_EQ(interpreter_.AllocateTensors(), kTfLiteOk);
  const std::string report = GeneratePartitionReport(interpreter_);
  EXPECT_THAT(report, HasSubstr("0 delegate partition(s)"));
  EXPECT_THAT(report, HasSubstr("3 fallback node(s) in 1 island(s)"));
  EXPECT_THAT(report,
              HasSubstr("island 0: plan[0..2] 3 node(s): ADD x2, CUSTOM "
                        "FALLBACK_OP x1"));
  EXPECT_THAT(report, HasSubstr("PARTITION_REPORT partitions=0 "
                                "delegated_nodes=0 fallback_nodes=3 "
                                "boundary_bytes=0 inter_partition_bytes=0 "
                                "dynamic_boundary_tensors=0"));
}

TEST_F(PartitionReportTest, XnnpackPartitionsAroundFallbackIsland) {
  ApplyXnnpack();
  ASSERT_EQ(interpreter_.AllocateTensors(), kTfLiteOk);
  const std::string report = GeneratePartitionReport(interpreter_);

  EXPECT_THAT(report, HasSubstr("DELEGATE(TfLiteXNNPackDelegate) replaces 1 "
                                "node(s)"));
  EXPECT_THAT(report, HasSubstr("CUSTOM FALLBACK_OP"));
  EXPECT_THAT(report, HasSubstr("2 delegate partition(s) covering 2 node(s); "
                                "1 fallback node(s) in 1 island(s)"));
  EXPECT_THAT(
      report,
      HasSubstr("island 0: plan[1..1] 1 node(s): CUSTOM FALLBACK_OP x1"));
  // Each partition exchanges one 16-byte tensor with the CPU on each side:
  // graph input / fallback input, fallback output / graph output.
  EXPECT_THAT(report, HasSubstr("partition 0: plan[0] TfLiteXNNPackDelegate "
                                "(1 node(s)) in=16 B out=16 B"));
  EXPECT_THAT(report, HasSubstr("partition 1: plan[2] TfLiteXNNPackDelegate "
                                "(1 node(s)) in=16 B out=16 B"));
  EXPECT_THAT(report, HasSubstr("total: in=32 B out=32 B all=64 B "
                                "(delegate->delegate: 0 B)"));
  EXPECT_THAT(report, HasSubstr("PARTITION_REPORT partitions=2 "
                                "delegated_nodes=2 fallback_nodes=1 "
                                "boundary_bytes=64 inter_partition_bytes=0 "
                                "dynamic_boundary_tensors=0"));
}

}  // namespace
}  // namespace benchmark
}  // namespace tflite
