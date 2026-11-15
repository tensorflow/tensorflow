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
// Tests for InterpreterBuilder::SetSubgraphSpecializations.

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/core/signature_runner.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace impl {
namespace {

using ::testing::ElementsAre;
using Specialization = InterpreterBuilder::SubgraphSpecialization;

// multi_signatures.bin: subgraph 0 = signature "add" (y = x + 2), subgraph 1
// = signature "sub" (y = x - 3); input "x" is 1-D with a dynamic dim.
constexpr char kMultiSignatures[] =
    "tensorflow/lite/testdata/multi_signatures.bin";
// no_signatures.bin: one subgraph, inputs "x1", "x2", output "Identity".
constexpr char kNoSignatures[] =
    "tensorflow/lite/testdata/no_signatures.bin";
// Same graph with all tensor names stripped.
constexpr char kNoSignaturesNoTensorNames[] =
    "tensorflow/lite/testdata/no_signatures_no_tensor_names.bin";

std::vector<int> Dims(const TfLiteTensor* t) {
  return std::vector<int>(t->dims->data, t->dims->data + t->dims->size);
}

std::vector<std::string> Keys(const Interpreter& interpreter) {
  std::vector<std::string> keys;
  for (const std::string* k : interpreter.signature_keys()) keys.push_back(*k);
  return keys;
}

class SpecializationTest : public ::testing::Test {
 protected:
  TfLiteStatus Build(const char* path, std::vector<Specialization> specs) {
    model_ = FlatBufferModel::BuildFromFile(path, &reporter_);
    if (!model_) return kTfLiteError;
    InterpreterBuilder builder(*model_, resolver_);
    builder.SetSubgraphSpecializations(std::move(specs));
    return builder(&interpreter_);
  }

  TestErrorReporter reporter_;
  ops::builtin::BuiltinOpResolver resolver_;
  std::unique_ptr<FlatBufferModel> model_;
  std::unique_ptr<Interpreter> interpreter_;
};

TEST_F(SpecializationTest, EmptyListIsNoop) {
  ASSERT_EQ(Build(kMultiSignatures, {}), kTfLiteOk);
  EXPECT_EQ(interpreter_->subgraphs_size(), 2);
  EXPECT_THAT(Keys(*interpreter_), ElementsAre("add", "sub"));
}

TEST_F(SpecializationTest, AddCreatesStaticSignatureAndKeepsSource) {
  ASSERT_EQ(Build(kMultiSignatures, {{.source_subgraph_index = 0,
                                      .signature_key = "add_4",
                                      .input_dims = {{0, {4}}}}}),
            kTfLiteOk);
  EXPECT_EQ(interpreter_->subgraphs_size(), 3);
  EXPECT_THAT(Keys(*interpreter_), ElementsAre("add", "sub", "add_4"));

  SignatureRunner* clone = interpreter_->GetSignatureRunner("add_4");
  ASSERT_NE(clone, nullptr);
  EXPECT_THAT(clone->input_names(), ElementsAre(::testing::StrEq("x")));
  EXPECT_THAT(clone->output_names(), ElementsAre(::testing::StrEq("output_0")));
  // Resized before AllocateTensors, like a user ResizeInputTensor call.
  EXPECT_THAT(Dims(clone->input_tensor("x")), ElementsAre(4));
  ASSERT_EQ(clone->AllocateTensors(), kTfLiteOk);
  EXPECT_THAT(Dims(clone->output_tensor("output_0")), ElementsAre(4));
  for (int i = 0; i < 4; ++i) clone->input_tensor("x")->data.f[i] = i;
  ASSERT_EQ(clone->Invoke(), kTfLiteOk);
  const float* out = clone->output_tensor("output_0")->data.f;
  EXPECT_THAT(std::vector<float>(out, out + 4), ElementsAre(2, 3, 4, 5));

  // Source is untouched: still resizable to another shape.
  SignatureRunner* source = interpreter_->GetSignatureRunner("add");
  ASSERT_NE(source, nullptr);
  ASSERT_EQ(source->ResizeInputTensor("x", {2}), kTfLiteOk);
  ASSERT_EQ(source->AllocateTensors(), kTfLiteOk);
  EXPECT_THAT(Dims(source->output_tensor("output_0")), ElementsAre(2));
  // The clone keeps its own shape.
  EXPECT_THAT(Dims(clone->input_tensor("x")), ElementsAre(4));
}

TEST_F(SpecializationTest, CloneSharesWeightsWithSource) {
  ASSERT_EQ(Build(kMultiSignatures, {{.source_subgraph_index = 0,
                                      .signature_key = "add_4",
                                      .input_dims = {{0, {4}}}}}),
            kTfLiteOk);
  Subgraph* source = interpreter_->subgraph(0);
  Subgraph* clone = interpreter_->subgraph(2);
  ASSERT_EQ(source->tensors_size(), clone->tensors_size());
  int shared = 0;
  for (size_t i = 0; i < source->tensors_size(); ++i) {
    const TfLiteTensor* s = source->tensor(i);
    const TfLiteTensor* c = clone->tensor(i);
    if (s->allocation_type == kTfLiteMmapRo) {
      EXPECT_EQ(c->allocation_type, kTfLiteMmapRo);
      EXPECT_EQ(c->data.data, s->data.data);
      ++shared;
    }
  }
  EXPECT_GT(shared, 0);
}

TEST_F(SpecializationTest, MultipleClonesFromOneSource) {
  ASSERT_EQ(Build(kMultiSignatures, {{.source_subgraph_index = 0,
                                      .signature_key = "add_2",
                                      .input_dims = {{0, {2}}}},
                                     {.source_subgraph_index = 0,
                                      .signature_key = "add_8",
                                      .input_dims = {{0, {8}}}},
                                     {.source_subgraph_index = 1,
                                      .signature_key = "sub_3",
                                      .source_signature_key = "sub",
                                      .input_dims = {{0, {3}}}}}),
            kTfLiteOk);
  EXPECT_EQ(interpreter_->subgraphs_size(), 5);
  EXPECT_THAT(Keys(*interpreter_),
              ElementsAre("add", "sub", "add_2", "add_8", "sub_3"));
  EXPECT_THAT(
      Dims(interpreter_->GetSignatureRunner("add_2")->input_tensor("x")),
      ElementsAre(2));
  EXPECT_THAT(
      Dims(interpreter_->GetSignatureRunner("add_8")->input_tensor("x")),
      ElementsAre(8));
  SignatureRunner* sub3 = interpreter_->GetSignatureRunner("sub_3");
  ASSERT_NE(sub3, nullptr);
  EXPECT_THAT(Dims(sub3->input_tensor("x")), ElementsAre(3));
  ASSERT_EQ(sub3->AllocateTensors(), kTfLiteOk);
  for (int i = 0; i < 3; ++i) sub3->input_tensor("x")->data.f[i] = 10 + i;
  ASSERT_EQ(sub3->Invoke(), kTfLiteOk);
  const float* out = sub3->output_tensor("output_0")->data.f;
  EXPECT_THAT(std::vector<float>(out, out + 3), ElementsAre(7, 8, 9));
}

TEST_F(SpecializationTest, ReplaceResizesSourceInPlace) {
  ASSERT_EQ(Build(kMultiSignatures,
                  {{.source_subgraph_index = 0, .input_dims = {{0, {3}}}}}),
            kTfLiteOk);
  EXPECT_EQ(interpreter_->subgraphs_size(), 2);
  EXPECT_THAT(Keys(*interpreter_), ElementsAre("add", "sub"));
  SignatureRunner* add = interpreter_->GetSignatureRunner("add");
  EXPECT_THAT(Dims(add->input_tensor("x")), ElementsAre(3));
  ASSERT_EQ(add->AllocateTensors(), kTfLiteOk);
  EXPECT_THAT(Dims(add->output_tensor("output_0")), ElementsAre(3));
}

TEST_F(SpecializationTest, ModelWithoutSignaturesUsesTensorNames) {
  ASSERT_EQ(Build(kNoSignatures, {{.source_subgraph_index = 0,
                                   .signature_key = "main_2",
                                   .input_dims = {{0, {2}}, {1, {2}}}}}),
            kTfLiteOk);
  EXPECT_EQ(interpreter_->subgraphs_size(), 2);
  EXPECT_THAT(Keys(*interpreter_), ElementsAre("main_2"));
  SignatureRunner* runner = interpreter_->GetSignatureRunner("main_2");
  ASSERT_NE(runner, nullptr);
  EXPECT_THAT(runner->input_names(),
              ElementsAre(::testing::StrEq("x1"), ::testing::StrEq("x2")));
  EXPECT_THAT(runner->output_names(),
              ElementsAre(::testing::StrEq("Identity")));
  EXPECT_THAT(Dims(runner->input_tensor("x1")), ElementsAre(2));
  EXPECT_THAT(Dims(runner->input_tensor("x2")), ElementsAre(2));
  ASSERT_EQ(runner->AllocateTensors(), kTfLiteOk);
}

TEST_F(SpecializationTest, ModelWithoutTensorNamesUsesPlaceholderNames) {
  ASSERT_EQ(
      Build(kNoSignaturesNoTensorNames, {{.source_subgraph_index = 0,
                                          .signature_key = "main_2",
                                          .input_dims = {{0, {2}}, {1, {2}}}}}),
      kTfLiteOk);
  SignatureRunner* runner = interpreter_->GetSignatureRunner("main_2");
  ASSERT_NE(runner, nullptr);
  // Unnamed tensors get the same placeholder names as the default runner
  // (Interpreter::CreatePlaceholderSignatureDef), not a shared "" key.
  EXPECT_THAT(runner->input_names(), ElementsAre(::testing::StrEq("input0"),
                                                 ::testing::StrEq("input1")));
  EXPECT_THAT(runner->output_names(), ElementsAre(::testing::StrEq("output0")));
  EXPECT_THAT(Dims(runner->input_tensor("input0")), ElementsAre(2));
  EXPECT_THAT(Dims(runner->input_tensor("input1")), ElementsAre(2));
  ASSERT_EQ(runner->AllocateTensors(), kTfLiteOk);
}

TEST_F(SpecializationTest, RejectsDuplicateKey) {
  EXPECT_EQ(Build(kMultiSignatures, {{.source_subgraph_index = 0,
                                      .signature_key = "x4",
                                      .input_dims = {{0, {4}}}},
                                     {.source_subgraph_index = 1,
                                      .signature_key = "x4",
                                      .input_dims = {{0, {4}}}}}),
            kTfLiteError);
  EXPECT_EQ(interpreter_, nullptr);
}

TEST_F(SpecializationTest, RejectsKeyCollidingWithModelSignature) {
  EXPECT_EQ(Build(kMultiSignatures, {{.source_subgraph_index = 0,
                                      .signature_key = "sub",
                                      .input_dims = {{0, {4}}}}}),
            kTfLiteError);
}

TEST_F(SpecializationTest, RejectsBadSourceIndex) {
  EXPECT_EQ(Build(kMultiSignatures, {{.source_subgraph_index = 5,
                                      .signature_key = "x",
                                      .input_dims = {{0, {4}}}}}),
            kTfLiteError);
}

TEST_F(SpecializationTest, RejectsBadInputPosition) {
  EXPECT_EQ(Build(kMultiSignatures, {{.source_subgraph_index = 0,
                                      .signature_key = "x",
                                      .input_dims = {{3, {4}}}}}),
            kTfLiteError);
}

TEST_F(SpecializationTest, RejectsUnknownSourceSignature) {
  EXPECT_EQ(Build(kMultiSignatures, {{.source_subgraph_index = 0,
                                      .signature_key = "x",
                                      .source_signature_key = "nope",
                                      .input_dims = {{0, {4}}}}}),
            kTfLiteError);
}

TEST_F(SpecializationTest, RejectsSourceSignatureOfOtherSubgraph) {
  EXPECT_EQ(Build(kMultiSignatures, {{.source_subgraph_index = 0,
                                      .signature_key = "x",
                                      .source_signature_key = "sub",
                                      .input_dims = {{0, {4}}}}}),
            kTfLiteError);
}

}  // namespace
}  // namespace impl
}  // namespace tflite
