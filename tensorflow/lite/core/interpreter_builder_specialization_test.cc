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
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/core/signature_runner.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/testing/util.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace impl {
namespace {

using ::testing::ElementsAre;
using ::testing::Not;
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

std::vector<uint8_t> FloatBytes(const std::vector<float>& values) {
  const auto* begin = reinterpret_cast<const uint8_t*>(values.data());
  return std::vector<uint8_t>(begin, begin + values.size() * sizeof(float));
}

// Tests for `tensor_dims` and `constant_overrides`. They use the "add"
// signature of multi_signatures.bin (subgraph 0, y = x + 2).
class SpecializationOverridesTest : public SpecializationTest {
 protected:
  // Builds the model without specializations to look up the tensor indices
  // the specializations refer to.
  void SetUp() override {
    ASSERT_EQ(Build(kMultiSignatures, {}), kTfLiteOk);
    const Subgraph* add = interpreter_->subgraph(0);
    ASSERT_EQ(add->inputs().size(), 1);
    ASSERT_EQ(add->outputs().size(), 1);
    input_index_ = add->inputs()[0];
    output_index_ = add->outputs()[0];
    for (size_t i = 0; i < add->tensors_size(); ++i) {
      const TfLiteTensor* t = add->tensor(i);
      if (t->allocation_type == kTfLiteMmapRo && t->type == kTfLiteFloat32 &&
          t->bytes > 0) {
        ASSERT_EQ(constant_index_, -1) << "expected a single float constant";
        constant_index_ = static_cast<int>(i);
        constant_elements_ = t->bytes / sizeof(float);
      }
    }
    ASSERT_NE(constant_index_, -1);
  }

  // Runs signature `key` on {1, 2} and returns the two outputs.
  std::vector<float> RunOnOneTwo(const char* key) {
    SignatureRunner* runner = interpreter_->GetSignatureRunner(key);
    if (runner == nullptr) return {};
    if (runner->ResizeInputTensor("x", {2}) != kTfLiteOk) return {};
    if (runner->AllocateTensors() != kTfLiteOk) return {};
    runner->input_tensor("x")->data.f[0] = 1;
    runner->input_tensor("x")->data.f[1] = 2;
    if (runner->Invoke() != kTfLiteOk) return {};
    const float* out = runner->output_tensor("output_0")->data.f;
    return std::vector<float>(out, out + 2);
  }

  int input_index_ = -1;
  int output_index_ = -1;
  int constant_index_ = -1;
  size_t constant_elements_ = 0;
};

TEST_F(SpecializationOverridesTest, ConstantOverrideAffectsCloneOnly) {
  ASSERT_EQ(
      Build(kMultiSignatures,
            {{.source_subgraph_index = 0,
              .signature_key = "add_10",
              .input_dims = {{0, {2}}},
              .constant_overrides = {{constant_index_,
                                      FloatBytes(std::vector<float>(
                                          constant_elements_, 10.0f))}}}}),
      kTfLiteOk);
  EXPECT_THAT(RunOnOneTwo("add_10"), ElementsAre(11, 12));
  EXPECT_THAT(RunOnOneTwo("add"), ElementsAre(3, 4));

  const Subgraph* source = interpreter_->subgraph(0);
  const Subgraph* clone = interpreter_->subgraph(2);
  const TfLiteTensor* source_constant = source->tensor(constant_index_);
  const TfLiteTensor* clone_constant = clone->tensor(constant_index_);
  // The clone owns a private copy that delegates still see as a constant.
  EXPECT_EQ(clone_constant->allocation_type, kTfLiteMmapRo);
  EXPECT_NE(clone_constant->data.data, source_constant->data.data);
  EXPECT_THAT(Dims(clone_constant), Dims(source_constant));
  // It no longer aliases the model buffer, so it must not be shared by id.
  EXPECT_EQ(source->GetTensorBufferIdentifiers().count(constant_index_), 1);
  EXPECT_EQ(clone->GetTensorBufferIdentifiers().count(constant_index_), 0);
}

TEST_F(SpecializationOverridesTest, ConstantOverrideWithNewDims) {
  ASSERT_EQ(Build(kMultiSignatures,
                  {{.source_subgraph_index = 0,
                    .signature_key = "add_vec",
                    .input_dims = {{0, {2}}},
                    .tensor_dims = {{constant_index_, {2}}},
                    .constant_overrides = {{constant_index_,
                                            FloatBytes({10.0f, 20.0f})}}}}),
            kTfLiteOk);
  EXPECT_THAT(Dims(interpreter_->subgraph(2)->tensor(constant_index_)),
              ElementsAre(2));
  EXPECT_THAT(RunOnOneTwo("add_vec"), ElementsAre(11, 22));
  EXPECT_THAT(RunOnOneTwo("add"), ElementsAre(3, 4));
}

TEST_F(SpecializationOverridesTest, TensorDimsAreSetWithoutPrepare) {
  // Without `tensor_dims` the output keeps the shape stored in the model until
  // the kernels' Prepare runs.
  ASSERT_EQ(Build(kMultiSignatures, {{.source_subgraph_index = 0,
                                      .signature_key = "add_4",
                                      .input_dims = {{0, {4}}}}}),
            kTfLiteOk);
  EXPECT_THAT(Dims(interpreter_->subgraph(2)->tensor(output_index_)),
              Not(ElementsAre(4)));

  ASSERT_EQ(
      Build(kMultiSignatures,
            {{.source_subgraph_index = 0,
              .signature_key = "add_4",
              .tensor_dims = {{input_index_, {4}}, {output_index_, {4}}}}}),
      kTfLiteOk);
  const Subgraph* clone = interpreter_->subgraph(2);
  EXPECT_THAT(Dims(clone->tensor(input_index_)), ElementsAre(4));
  EXPECT_THAT(Dims(clone->tensor(output_index_)), ElementsAre(4));
  // The source keeps the model's shapes.
  EXPECT_THAT(Dims(interpreter_->subgraph(0)->tensor(output_index_)),
              Not(ElementsAre(4)));

  SignatureRunner* runner = interpreter_->GetSignatureRunner("add_4");
  ASSERT_NE(runner, nullptr);
  ASSERT_EQ(runner->AllocateTensors(), kTfLiteOk);
  EXPECT_THAT(Dims(runner->output_tensor("output_0")), ElementsAre(4));
  for (int i = 0; i < 4; ++i) runner->input_tensor("x")->data.f[i] = i;
  ASSERT_EQ(runner->Invoke(), kTfLiteOk);
  const float* out = runner->output_tensor("output_0")->data.f;
  EXPECT_THAT(std::vector<float>(out, out + 4), ElementsAre(2, 3, 4, 5));
}

TEST_F(SpecializationOverridesTest, ReplaceOverridesSourceInPlace) {
  ASSERT_EQ(
      Build(kMultiSignatures,
            {{.source_subgraph_index = 0,
              .constant_overrides = {{constant_index_,
                                      FloatBytes(std::vector<float>(
                                          constant_elements_, 10.0f))}}}}),
      kTfLiteOk);
  EXPECT_EQ(interpreter_->subgraphs_size(), 2);
  EXPECT_THAT(RunOnOneTwo("add"), ElementsAre(11, 12));
}

TEST_F(SpecializationOverridesTest, RejectsTensorIndexOutOfRange) {
  EXPECT_EQ(Build(kMultiSignatures, {{.source_subgraph_index = 0,
                                      .signature_key = "x",
                                      .tensor_dims = {{999, {4}}}}}),
            kTfLiteError);
  EXPECT_EQ(Build(kMultiSignatures,
                  {{.source_subgraph_index = 0,
                    .signature_key = "x",
                    .constant_overrides = {{999, FloatBytes({1.0f})}}}}),
            kTfLiteError);
}

TEST_F(SpecializationOverridesTest, RejectsConstantOverrideOfNonConstant) {
  EXPECT_EQ(
      Build(kMultiSignatures,
            {{.source_subgraph_index = 0,
              .signature_key = "x",
              .constant_overrides = {{input_index_, FloatBytes({1.0f})}}}}),
      kTfLiteError);
}

TEST_F(SpecializationOverridesTest, RejectsConstantOverrideTooSmallForDims) {
  EXPECT_EQ(Build(kMultiSignatures,
                  {{.source_subgraph_index = 0,
                    .signature_key = "x",
                    .tensor_dims = {{constant_index_, {3}}},
                    .constant_overrides = {{constant_index_,
                                            FloatBytes({1.0f, 2.0f})}}}}),
            kTfLiteError);
}

TEST_F(SpecializationOverridesTest, RejectsReshapingConstantWithoutContents) {
  EXPECT_EQ(
      Build(kMultiSignatures, {{.source_subgraph_index = 0,
                                .signature_key = "x",
                                .tensor_dims = {{constant_index_, {7}}}}}),
      kTfLiteError);
}

TEST_F(SpecializationOverridesTest, ConstantOverrideUpdatesByteSize) {
  // The model buffer of a constant may be larger than its shape requires; the
  // override must not keep that size for its own, smaller buffer.
  ASSERT_EQ(
      Build(kMultiSignatures,
            {{.source_subgraph_index = 0,
              .signature_key = "add_10",
              .constant_overrides = {{constant_index_,
                                      FloatBytes(std::vector<float>(
                                          constant_elements_, 10.0f))}}}}),
      kTfLiteOk);
  EXPECT_EQ(interpreter_->subgraph(2)->tensor(constant_index_)->bytes,
            constant_elements_ * sizeof(float));
}

// Builds y = x + c where the constant `c` (tensor 1, shape [2]) is stored in
// an external buffer, so the model itself holds no data for it.
std::vector<uint8_t> BuildExternalConstantModel() {
  auto float_tensor = [](const char* name) {
    auto tensor = std::make_unique<TensorT>();
    tensor->name = name;
    tensor->type = TensorType_FLOAT32;
    tensor->shape = {2};
    return tensor;
  };
  ModelT model;
  model.version = TFLITE_SCHEMA_VERSION;
  model.buffers.push_back(std::make_unique<BufferT>());
  auto add_code = std::make_unique<OperatorCodeT>();
  add_code->builtin_code = BuiltinOperator_ADD;
  add_code->deprecated_builtin_code = BuiltinOperator_ADD;
  model.operator_codes.push_back(std::move(add_code));

  auto subgraph = std::make_unique<SubGraphT>();
  subgraph->tensors.push_back(float_tensor("x"));
  subgraph->tensors.push_back(float_tensor("c"));
  subgraph->tensors[1]->external_buffer = 1;
  subgraph->tensors.push_back(float_tensor("y"));
  subgraph->inputs = {0};
  subgraph->outputs = {2};
  auto add = std::make_unique<OperatorT>();
  add->opcode_index = 0;
  add->inputs = {0, 1};
  add->outputs = {2};
  add->builtin_options.Set(AddOptionsT());
  subgraph->operators.push_back(std::move(add));
  model.subgraphs.push_back(std::move(subgraph));

  flatbuffers::FlatBufferBuilder builder;
  builder.Finish(Model::Pack(builder, &model), ModelIdentifier());
  return std::vector<uint8_t>(builder.GetBufferPointer(),
                              builder.GetBufferPointer() + builder.GetSize());
}

class SpecializationExternalConstantTest : public ::testing::Test {
 protected:
  static constexpr int kConstantIndex = 1;

  TfLiteStatus Build(std::vector<Specialization> specs) {
    buffer_ = BuildExternalConstantModel();
    InterpreterBuilder builder(GetModel(buffer_.data()), resolver_);
    builder.SetSubgraphSpecializations(std::move(specs));
    return builder(&interpreter_);
  }

  ops::builtin::BuiltinOpResolver resolver_;
  std::vector<uint8_t> buffer_;
  std::unique_ptr<Interpreter> interpreter_;
};

TEST_F(SpecializationExternalConstantTest, ConstantOverrideReplacesExternal) {
  ASSERT_EQ(Build({{.source_subgraph_index = 0,
                    .signature_key = "clone",
                    .constant_overrides = {{kConstantIndex,
                                            FloatBytes({10.0f, 20.0f})}}}}),
            kTfLiteOk);
  const Subgraph* source = interpreter_->subgraph(0);
  const Subgraph* clone = interpreter_->subgraph(1);
  // The source still expects its constant from the external buffer; the clone
  // has its own contents and must not have them loaded over.
  EXPECT_EQ(source->GetExternalTensorBufferIdentifiers().count(kConstantIndex),
            1);
  EXPECT_EQ(clone->GetExternalTensorBufferIdentifiers().count(kConstantIndex),
            0);
  const TfLiteTensor* constant = clone->tensor(kConstantIndex);
  EXPECT_EQ(constant->allocation_type, kTfLiteMmapRo);
  EXPECT_EQ(constant->bytes, 2 * sizeof(float));

  SignatureRunner* runner = interpreter_->GetSignatureRunner("clone");
  ASSERT_NE(runner, nullptr);
  ASSERT_EQ(runner->AllocateTensors(), kTfLiteOk);
  runner->input_tensor("x")->data.f[0] = 1;
  runner->input_tensor("x")->data.f[1] = 2;
  ASSERT_EQ(runner->Invoke(), kTfLiteOk);
  const float* out = runner->output_tensor("y")->data.f;
  EXPECT_THAT(std::vector<float>(out, out + 2), ElementsAre(11, 22));
}

TEST_F(SpecializationExternalConstantTest,
       RejectsReshapingExternalConstantWithoutContents) {
  // The external buffer has a fixed size, so the constant cannot grow.
  EXPECT_EQ(Build({{.source_subgraph_index = 0,
                    .signature_key = "clone",
                    .tensor_dims = {{kConstantIndex, {8}}}}}),
            kTfLiteError);
  // Listing its unchanged shape is fine.
  EXPECT_EQ(Build({{.source_subgraph_index = 0,
                    .signature_key = "clone",
                    .tensor_dims = {{kConstantIndex, {2}}}}}),
            kTfLiteOk);
}

}  // namespace
}  // namespace impl
}  // namespace tflite
