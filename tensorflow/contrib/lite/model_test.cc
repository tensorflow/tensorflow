/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <string>

#include "tensorflow/contrib/lite/model.h"

#include <gtest/gtest.h>
#include "tensorflow/contrib/lite/error_reporter.h"
#include "tensorflow/contrib/lite/testing/util.h"

// Comparison for TfLiteRegistration. Since TfLiteRegistration is a C object,
// we must declare this in global namespace, so argument-dependent operator
// lookup works.
inline bool operator==(const TfLiteRegistration& a,
                       const TfLiteRegistration& b) {
  return a.invoke == b.invoke && a.init == b.init && a.prepare == b.prepare &&
         a.free == b.free;
}

namespace tflite {

// Provide a dummy operation that does nothing.
namespace {
void* dummy_init(TfLiteContext*, const char*, size_t) { return nullptr; }
void dummy_free(TfLiteContext*, void*) {}
TfLiteStatus dummy_resize(TfLiteContext*, TfLiteNode*) { return kTfLiteOk; }
TfLiteStatus dummy_invoke(TfLiteContext*, TfLiteNode*) { return kTfLiteOk; }
TfLiteRegistration dummy_reg = {dummy_init, dummy_free, dummy_resize,
                                dummy_invoke};
}  // namespace

// Provide a trivial resolver that returns a constant value no matter what
// op is asked for.
class TrivialResolver : public OpResolver {
 public:
  explicit TrivialResolver(TfLiteRegistration* constant_return = nullptr)
      : constant_return_(constant_return) {}
  // Find the op registration of a custom operator by op name.
  TfLiteRegistration* FindOp(tflite::BuiltinOperator op) const override {
    return constant_return_;
  }
  // Find the op registration of a custom operator by op name.
  TfLiteRegistration* FindOp(const char* op) const override {
    return constant_return_;
  }

 private:
  TfLiteRegistration* constant_return_;
};

TEST(BasicFlatBufferModel, TestNonExistantFiles) {
  ASSERT_TRUE(!FlatBufferModel::BuildFromFile("/tmp/tflite_model_1234"));
}

// Make sure a model with nothing in it loads properly.
TEST(BasicFlatBufferModel, TestEmptyModelsAndNullDestination) {
  auto model = FlatBufferModel::BuildFromFile(
      "tensorflow/contrib/lite/testdata/empty_model.bin");
  ASSERT_TRUE(model);
  // Now try to build it into a model.
  std::unique_ptr<Interpreter> interpreter;
  ASSERT_EQ(InterpreterBuilder(*model, TrivialResolver())(&interpreter),
            kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);
  ASSERT_NE(InterpreterBuilder(*model, TrivialResolver())(nullptr), kTfLiteOk);
}

// Make sure currently unsupported # of subgraphs are checked
// TODO(aselle): Replace this test when multiple subgraphs are supported.
TEST(BasicFlatBufferModel, TestZeroAndMultipleSubgraphs) {
  auto m1 = FlatBufferModel::BuildFromFile(
      "tensorflow/contrib/lite/testdata/0_subgraphs.bin");
  ASSERT_TRUE(m1);
  std::unique_ptr<Interpreter> interpreter1;
  ASSERT_NE(InterpreterBuilder(*m1, TrivialResolver())(&interpreter1),
            kTfLiteOk);

  auto m2 = FlatBufferModel::BuildFromFile(
      "tensorflow/contrib/lite/testdata/2_subgraphs.bin");
  ASSERT_TRUE(m2);
  std::unique_ptr<Interpreter> interpreter2;
  ASSERT_NE(InterpreterBuilder(*m2, TrivialResolver())(&interpreter2),
            kTfLiteOk);
}

// Test what happens if we cannot bind any of the ops.
TEST(BasicFlatBufferModel, TestModelWithoutNullRegistrations) {
  auto model = FlatBufferModel::BuildFromFile(
      "tensorflow/contrib/lite/testdata/test_model.bin");
  ASSERT_TRUE(model);
  // Check that we get an error code and interpreter pointer is reset.
  std::unique_ptr<Interpreter> interpreter(new Interpreter);
  ASSERT_NE(InterpreterBuilder(*model, TrivialResolver(nullptr))(&interpreter),
            kTfLiteOk);
  ASSERT_EQ(interpreter, nullptr);
}

// Make sure model is read to interpreter propelrly
TEST(BasicFlatBufferModel, TestModelInInterpreter) {
  auto model = FlatBufferModel::BuildFromFile(
      "tensorflow/contrib/lite/testdata/test_model.bin");
  ASSERT_TRUE(model);
  // Check that we get an error code and interpreter pointer is reset.
  std::unique_ptr<Interpreter> interpreter(new Interpreter);
  ASSERT_EQ(
      InterpreterBuilder(*model, TrivialResolver(&dummy_reg))(&interpreter),
      kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->tensors_size(), 4);
  ASSERT_EQ(interpreter->nodes_size(), 2);
  std::vector<int> inputs = {0, 1};
  std::vector<int> outputs = {2, 3};
  ASSERT_EQ(interpreter->inputs(), inputs);
  ASSERT_EQ(interpreter->outputs(), outputs);

  EXPECT_EQ(std::string(interpreter->GetInputName(0)), "input0");
  EXPECT_EQ(std::string(interpreter->GetInputName(1)), "input1");
  EXPECT_EQ(std::string(interpreter->GetOutputName(0)), "out1");
  EXPECT_EQ(std::string(interpreter->GetOutputName(1)), "out2");

  // Make sure all input tensors are correct
  TfLiteTensor* i0 = interpreter->tensor(0);
  ASSERT_EQ(i0->type, kTfLiteFloat32);
  ASSERT_NE(i0->data.raw, nullptr);  // mmapped
  ASSERT_EQ(i0->allocation_type, kTfLiteMmapRo);
  TfLiteTensor* i1 = interpreter->tensor(1);
  ASSERT_EQ(i1->type, kTfLiteFloat32);
  ASSERT_EQ(i1->data.raw, nullptr);
  ASSERT_EQ(i1->allocation_type, kTfLiteArenaRw);
  TfLiteTensor* o0 = interpreter->tensor(2);
  ASSERT_EQ(o0->type, kTfLiteFloat32);
  ASSERT_EQ(o0->data.raw, nullptr);
  ASSERT_EQ(o0->allocation_type, kTfLiteArenaRw);
  TfLiteTensor* o1 = interpreter->tensor(3);
  ASSERT_EQ(o1->type, kTfLiteFloat32);
  ASSERT_EQ(o1->data.raw, nullptr);
  ASSERT_EQ(o1->allocation_type, kTfLiteArenaRw);

  // Check op 0 which has inputs {0, 1} outputs {2}.
  {
    const std::pair<TfLiteNode, TfLiteRegistration>* node_and_reg0 =
        interpreter->node_and_registration(0);
    ASSERT_NE(node_and_reg0, nullptr);
    const TfLiteNode& node0 = node_and_reg0->first;
    const TfLiteRegistration& reg0 = node_and_reg0->second;
    TfLiteIntArray* desired_inputs = TfLiteIntArrayCreate(2);
    desired_inputs->data[0] = 0;
    desired_inputs->data[1] = 1;
    TfLiteIntArray* desired_outputs = TfLiteIntArrayCreate(1);
    desired_outputs->data[0] = 2;
    ASSERT_TRUE(TfLiteIntArrayEqual(node0.inputs, desired_inputs));
    ASSERT_TRUE(TfLiteIntArrayEqual(node0.outputs, desired_outputs));
    TfLiteIntArrayFree(desired_inputs);
    TfLiteIntArrayFree(desired_outputs);
    ASSERT_EQ(reg0, dummy_reg);
  }

  // Check op 1 which has inputs {2} outputs {3}.
  {
    const std::pair<TfLiteNode, TfLiteRegistration>* node_and_reg1 =
        interpreter->node_and_registration(1);
    ASSERT_NE(node_and_reg1, nullptr);
    const TfLiteNode& node1 = node_and_reg1->first;
    const TfLiteRegistration& reg1 = node_and_reg1->second;
    TfLiteIntArray* desired_inputs = TfLiteIntArrayCreate(1);
    TfLiteIntArray* desired_outputs = TfLiteIntArrayCreate(1);
    desired_inputs->data[0] = 2;
    desired_outputs->data[0] = 3;
    ASSERT_TRUE(TfLiteIntArrayEqual(node1.inputs, desired_inputs));
    ASSERT_TRUE(TfLiteIntArrayEqual(node1.outputs, desired_outputs));
    TfLiteIntArrayFree(desired_inputs);
    TfLiteIntArrayFree(desired_outputs);
    ASSERT_EQ(reg1, dummy_reg);
  }
}

// This tests on a flatbuffer that defines a shape of 2 to be a memory mapped
// buffer. But the buffer is provided to be only 1 element.
TEST(BasicFlatBufferModel, TestBrokenMmap) {
  ASSERT_FALSE(FlatBufferModel::BuildFromFile(
      "tensorflow/contrib/lite/testdata/test_model_broken.bin"));
}

TEST(BasicFlatBufferModel, TestNullModel) {
  // Check that we get an error code and interpreter pointer is reset.
  std::unique_ptr<Interpreter> interpreter(new Interpreter);
  ASSERT_NE(
      InterpreterBuilder(nullptr, TrivialResolver(&dummy_reg))(&interpreter),
      kTfLiteOk);
  ASSERT_EQ(interpreter.get(), nullptr);
}

struct TestErrorReporter : public ErrorReporter {
  int Report(const char* format, va_list args) override {
    calls++;
    return 0;
  }
  int calls = 0;
};

// This makes sure the ErrorReporter is marshalled from FlatBufferModel to
// the Interpreter.
TEST(BasicFlatBufferModel, TestCustomErrorReporter) {
  TestErrorReporter reporter;
  auto model = FlatBufferModel::BuildFromFile(
      "tensorflow/contrib/lite/testdata/empty_model.bin",
      &reporter);
  ASSERT_TRUE(model);

  std::unique_ptr<Interpreter> interpreter;
  TrivialResolver resolver;
  InterpreterBuilder(*model, resolver)(&interpreter);
  ASSERT_NE(interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(reporter.calls, 1);
}

// This makes sure the ErrorReporter is marshalled from FlatBufferModel to
// the Interpreter.
TEST(BasicFlatBufferModel, TestNullErrorReporter) {
  auto model = FlatBufferModel::BuildFromFile(
      "tensorflow/contrib/lite/testdata/empty_model.bin", nullptr);
  ASSERT_TRUE(model);

  std::unique_ptr<Interpreter> interpreter;
  TrivialResolver resolver;
  InterpreterBuilder(*model, resolver)(&interpreter);
  ASSERT_NE(interpreter->Invoke(), kTfLiteOk);
}

// Test what happens if we cannot bind any of the ops.
TEST(BasicFlatBufferModel, TestBuildModelFromCorruptedData) {
  std::string corrupted_data = "123";
  auto model = FlatBufferModel::BuildFromBuffer(corrupted_data.c_str(),
                                                corrupted_data.length());
  ASSERT_FALSE(model);
}

// TODO(aselle): Add tests for serialization of builtin op data types.
// These tests will occur with the evaluation tests of individual operators,
// not here.

}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
