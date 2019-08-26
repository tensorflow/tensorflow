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

// An ultra-lightweight testing framework designed for use with microcontroller
// applications. Its only dependency is on TensorFlow Lite's ErrorReporter
// interface, where log messages are output. This is designed to be usable even
// when no standard C or C++ libraries are available, and without any dynamic
// memory allocation or reliance on global constructors.
//
// To build a test, you use syntax similar to gunit, but with some extra
// decoration to create a hidden 'main' function containing each of the tests to
// be run. Your code should look something like:
// ----------------------------------------------------------------------------
// #include "path/to/this/header"
//
// TF_LITE_MICRO_TESTS_BEGIN
//
// TF_LITE_MICRO_TEST(SomeTest) {
//   TF_LITE_LOG_EXPECT_EQ(true, true);
// }
//
// TF_LITE_MICRO_TESTS_END
// ----------------------------------------------------------------------------
// If you compile this for your platform, you'll get a normal binary that you
// should be able to run. Executing it will output logging information like this
// to stderr (or whatever equivalent is available and written to by
// ErrorReporter):
// ----------------------------------------------------------------------------
// Testing SomeTest
// 1/1 tests passed
// ~~~ALL TESTS PASSED~~~
// ----------------------------------------------------------------------------
// This is designed to be human-readable, so you can just run tests manually,
// but the string "~~~ALL TESTS PASSED~~~" should only appear if all of the
// tests do pass. This makes it possible to integrate with automated test
// systems by scanning the output logs and looking for that magic value.
//
// This framework is intended to be a rudimentary alternative to no testing at
// all on systems that struggle to run more conventional approaches, so use with
// caution!

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICRO_TESTING_MICRO_TEST_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICRO_TESTING_MICRO_TEST_H_

#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/core/api/op_resolver.h"

namespace micro_test {
extern int tests_passed;
extern int tests_failed;
extern bool is_test_complete;
extern bool did_test_fail;
extern tflite::ErrorReporter* reporter;
}  // namespace micro_test

#define TF_LITE_MICRO_TESTS_BEGIN              \
  namespace micro_test {                       \
  int tests_passed;                            \
  int tests_failed;                            \
  bool is_test_complete;                       \
  bool did_test_fail;                          \
  tflite::ErrorReporter* reporter;             \
  }                                            \
                                               \
  int main(int argc, char** argv) {            \
    micro_test::tests_passed = 0;              \
    micro_test::tests_failed = 0;              \
    tflite::MicroErrorReporter error_reporter; \
    micro_test::reporter = &error_reporter;

#define TF_LITE_MICRO_TESTS_END                                \
  micro_test::reporter->Report(                                \
      "%d/%d tests passed", micro_test::tests_passed,          \
      (micro_test::tests_failed + micro_test::tests_passed));  \
  if (micro_test::tests_failed == 0) {                         \
    micro_test::reporter->Report("~~~ALL TESTS PASSED~~~\n");  \
  } else {                                                     \
    micro_test::reporter->Report("~~~SOME TESTS FAILED~~~\n"); \
  }                                                            \
  }

// TODO(petewarden): I'm going to hell for what I'm doing to this poor for loop.
#define TF_LITE_MICRO_TEST(name)                                           \
  micro_test::reporter->Report("Testing %s", #name);                       \
  for (micro_test::is_test_complete = false,                               \
      micro_test::did_test_fail = false;                                   \
       !micro_test::is_test_complete; micro_test::is_test_complete = true, \
      micro_test::tests_passed += (micro_test::did_test_fail) ? 0 : 1,     \
      micro_test::tests_failed += (micro_test::did_test_fail) ? 1 : 0)

#define TF_LITE_MICRO_EXPECT(x)                                                \
  do {                                                                         \
    if (!(x)) {                                                                \
      micro_test::reporter->Report(#x " failed at %s:%d", __FILE__, __LINE__); \
      micro_test::did_test_fail = true;                                        \
    }                                                                          \
  } while (false)

#define TF_LITE_MICRO_EXPECT_EQ(x, y)                                          \
  do {                                                                         \
    if ((x) != (y)) {                                                          \
      micro_test::reporter->Report(#x " == " #y " failed at %s:%d (%d vs %d)", \
                                   __FILE__, __LINE__, (x), (y));              \
      micro_test::did_test_fail = true;                                        \
    }                                                                          \
  } while (false)

#define TF_LITE_MICRO_EXPECT_NE(x, y)                                         \
  do {                                                                        \
    if ((x) == (y)) {                                                         \
      micro_test::reporter->Report(#x " != " #y " failed at %s:%d", __FILE__, \
                                   __LINE__);                                 \
      micro_test::did_test_fail = true;                                       \
    }                                                                         \
  } while (false)

#define TF_LITE_MICRO_EXPECT_NEAR(x, y, epsilon)                      \
  do {                                                                \
    auto delta = ((x) > (y)) ? ((x) - (y)) : ((y) - (x));             \
    if (delta > epsilon) {                                            \
      micro_test::reporter->Report(#x " near " #y " failed at %s:%d", \
                                   __FILE__, __LINE__);               \
      micro_test::did_test_fail = true;                               \
    }                                                                 \
  } while (false)

#define TF_LITE_MICRO_EXPECT_GT(x, y)                                        \
  do {                                                                       \
    if ((x) <= (y)) {                                                        \
      micro_test::reporter->Report(#x " > " #y " failed at %s:%d", __FILE__, \
                                   __LINE__);                                \
      micro_test::did_test_fail = true;                                      \
    }                                                                        \
  } while (false)

#define TF_LITE_MICRO_EXPECT_LT(x, y)                                        \
  do {                                                                       \
    if ((x) >= (y)) {                                                        \
      micro_test::reporter->Report(#x " < " #y " failed at %s:%d", __FILE__, \
                                   __LINE__);                                \
      micro_test::did_test_fail = true;                                      \
    }                                                                        \
  } while (false)

#define TF_LITE_MICRO_EXPECT_GE(x, y)                                         \
  do {                                                                        \
    if ((x) < (y)) {                                                          \
      micro_test::reporter->Report(#x " >= " #y " failed at %s:%d", __FILE__, \
                                   __LINE__);                                 \
      micro_test::did_test_fail = true;                                       \
    }                                                                         \
  } while (false)

#define TF_LITE_MICRO_EXPECT_LE(x, y)                                         \
  do {                                                                        \
    if ((x) > (y)) {                                                          \
      micro_test::reporter->Report(#x " <= " #y " failed at %s:%d", __FILE__, \
                                   __LINE__);                                 \
      micro_test::did_test_fail = true;                                       \
    }                                                                         \
  } while (false)


namespace tflite {
namespace {

void* MockModelOpInit(TfLiteContext* context, const char* buffer, size_t length) {
  // We don't support delegate in TFL micro. This is a weak check to test if
  // context struct being zero-initialized.
  TF_LITE_MICRO_EXPECT_EQ(nullptr,
                          context->ReplaceNodeSubsetsWithDelegateKernels);
  // Do nothing.
  return nullptr;
}

void MockModelOpFree(TfLiteContext* context, void* buffer) {
  // Do nothing.
}

TfLiteStatus MockModelOpPrepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

TfLiteStatus MockModelOpInvokeAdd(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = &context->tensors[node->inputs->data[0]];
  const int32_t* input_data = input->data.i32;
  const TfLiteTensor* weight = &context->tensors[node->inputs->data[1]];
  const uint8_t* weight_data = weight->data.uint8;
  TfLiteTensor* output = &context->tensors[node->outputs->data[0]];
  int32_t* output_data = output->data.i32;
  output_data[0] = input_data[0] + weight_data[0];
  return kTfLiteOk;
}

TfLiteStatus MockModelOpInvokeId(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = &context->tensors[node->inputs->data[0]];
  const int32_t* input_data = input->data.i32;
  TfLiteTensor* output = &context->tensors[node->outputs->data[0]];
  int32_t* output_data = output->data.i32;
  output_data[0] = input_data[0];
  return kTfLiteOk;
}

class MockOpResolver : public OpResolver {
public:
  const TfLiteRegistration* FindOp(BuiltinOperator op,
                                   int version) const override {
    return nullptr;
  }
  const TfLiteRegistration* FindOp(const char* op, int version) const override {
    if (strcmp(op, "mock_custom_add") == 0) {
      static TfLiteRegistration r = {MockModelOpInit, MockModelOpFree, MockModelOpPrepare,
                                     MockModelOpInvokeAdd};
      return &r;
    } else if (strcmp(op, "mock_custom_id") == 0) {
      static TfLiteRegistration r = {MockModelOpInit, MockModelOpFree, MockModelOpPrepare,
                                     MockModelOpInvokeId};
      return &r;
    } else {
      return nullptr;
    }
  }
};

class StackAllocator : public flatbuffers::Allocator {
public:
  StackAllocator() : data_(data_backing_), data_size_(0) {}

  uint8_t* allocate(size_t size) override {
    if ((data_size_ + size) > kStackAllocatorSize) {
      // TODO(petewarden): Add error reporting beyond returning null!
      return nullptr;
    }
    uint8_t* result = data_;
    data_ += size;
    data_size_ += size;
    return result;
  }

  void deallocate(uint8_t* p, size_t) override {}

//  static StackAllocator& instance() {
//    // Avoid using true dynamic memory allocation to be portable to bare metal.
//    static char inst_memory[sizeof(StackAllocator)];
//    static StackAllocator* inst = new (inst_memory) StackAllocator;
//    return *inst;
//  }

  static constexpr int kStackAllocatorSize = 4096;

private:
  uint8_t data_backing_[kStackAllocatorSize];
  uint8_t* data_;
  int data_size_;
};

const Model* BuildMockModel() {
  using flatbuffers::Offset;
  static StackAllocator stack_allocator;
  flatbuffers::FlatBufferBuilder builder(StackAllocator::kStackAllocatorSize,
                                         &stack_allocator);
  constexpr size_t buffer_data_size = 1;
  const uint8_t buffer_data[buffer_data_size] = {21};
  constexpr size_t buffers_size = 2;
  const Offset<Buffer> buffers[buffers_size] = {
          CreateBuffer(builder),
          CreateBuffer(builder,
                       builder.CreateVector(buffer_data, buffer_data_size))};
  constexpr size_t tensor_shape_size = 1;
  const int32_t tensor_shape[tensor_shape_size] = {1};
  constexpr size_t tensors_size = 3;
  const Offset<Tensor> tensors[tensors_size] = {
          CreateTensor(builder,
                       builder.CreateVector(tensor_shape, tensor_shape_size),
                       TensorType_INT32, 0,
                       builder.CreateString("test_input_tensor"), 0, false),
          CreateTensor(builder,
                       builder.CreateVector(tensor_shape, tensor_shape_size),
                       TensorType_UINT8, 1,
                       builder.CreateString("test_weight_tensor"), 0, false),
          CreateTensor(builder,
                       builder.CreateVector(tensor_shape, tensor_shape_size),
                       TensorType_INT32, 0,
                       builder.CreateString("test_output_tensor"), 0, false),
  };
  constexpr size_t inputs_size = 1;
  const int32_t inputs[inputs_size] = {0};
  constexpr size_t outputs_size = 1;
  const int32_t outputs[outputs_size] = {2};
  constexpr size_t operator_inputs_size = 2;
  const int32_t operator_inputs[operator_inputs_size] = {0, 1};
  constexpr size_t operator_outputs_size = 1;
  const int32_t operator_outputs[operator_outputs_size] = {2};
  constexpr size_t operators_size = 1;
  const Offset<Operator> operators[operators_size] = {CreateOperator(
          builder, 0, builder.CreateVector(operator_inputs, operator_inputs_size),
          builder.CreateVector(operator_outputs, operator_outputs_size),
          BuiltinOptions_NONE)};
  constexpr size_t subgraphs_size = 1;
  const Offset<SubGraph> subgraphs[subgraphs_size] = {
          CreateSubGraph(builder, builder.CreateVector(tensors, tensors_size),
                         builder.CreateVector(inputs, inputs_size),
                         builder.CreateVector(outputs, outputs_size),
                         builder.CreateVector(operators, operators_size),
                         builder.CreateString("test_subgraph"))};
  constexpr size_t operator_codes_size = 1;
  const Offset<OperatorCode> operator_codes[operator_codes_size] = {
          CreateOperatorCodeDirect(builder, BuiltinOperator_CUSTOM, "mock_custom_add",
                                   0)};
  const Offset<Model> model_offset = CreateModel(
          builder, 0, builder.CreateVector(operator_codes, operator_codes_size),
          builder.CreateVector(subgraphs, subgraphs_size),
          builder.CreateString("test_model"),
          builder.CreateVector(buffers, buffers_size));
  FinishModelBuffer(builder, model_offset);
  void* model_pointer = builder.GetBufferPointer();
  const Model* model = flatbuffers::GetRoot<Model>(model_pointer);
  return model;
}

const Model* BuildLargerMockModel() {
  // This model has consists of two operators, arranged as follows:
  // input (id: 0)  --> [ Operator 0 ] --(id: 2)--> [ Operator 1 ] --> output (id: 3)
  // weight (id: 1) ---/
  using flatbuffers::Offset;
  static StackAllocator stack_allocator;
  flatbuffers::FlatBufferBuilder builder(StackAllocator::kStackAllocatorSize,
                                         &stack_allocator);
  constexpr size_t buffer_data_size = 1;
  const uint8_t buffer_data[buffer_data_size] = {21};
  constexpr size_t buffers_size = 2;
  const Offset<Buffer> buffers[buffers_size] = {
          CreateBuffer(builder),
          CreateBuffer(builder,
                       builder.CreateVector(buffer_data, buffer_data_size))};
  constexpr size_t tensor_shape_size = 1;
  const int32_t tensor_shape[tensor_shape_size] = {1};
  constexpr size_t tensors_size = 4;
  const Offset<Tensor> tensors[tensors_size] = {
          CreateTensor(builder,
                       builder.CreateVector(tensor_shape, tensor_shape_size),
                       TensorType_INT32, 0,
                       builder.CreateString("test_input_tensor"), 0, false),
          CreateTensor(builder,
                       builder.CreateVector(tensor_shape, tensor_shape_size),
                       TensorType_UINT8, 1,
                       builder.CreateString("test_weight_tensor"), 0, false),
          CreateTensor(builder,
                       builder.CreateVector(tensor_shape, tensor_shape_size),
                       TensorType_INT32, 0,
                       builder.CreateString("test_op0_output_tensor"), 0, false),
          CreateTensor(builder,
                       builder.CreateVector(tensor_shape, tensor_shape_size),
                       TensorType_INT32, 0,
                       builder.CreateString("test_op1_output_tensor"), 0, false),
  };

  constexpr size_t inputs_size = 1;
  const int32_t inputs[inputs_size] = {0};
  constexpr size_t outputs_size = 1;
  const int32_t outputs[outputs_size] = {3};

  constexpr size_t operator0_inputs_size = 2;
  const int32_t operator0_inputs[operator0_inputs_size] = {0, 1};
  constexpr size_t operator0_outputs_size = 1;
  const int32_t operator0_outputs[operator0_outputs_size] = {2};

  constexpr size_t operator1_inputs_size = 1;
  const int32_t operator1_inputs[operator1_inputs_size] = {2};
  constexpr size_t operator1_outputs_size = 1;
  const int32_t operator1_outputs[operator1_outputs_size] = {3};

  constexpr size_t operators_size = 2;
  const Offset<Operator> operators[operators_size] = {
          CreateOperator(builder, 0,
                  builder.CreateVector(operator0_inputs, operator0_inputs_size),
                  builder.CreateVector(operator0_outputs, operator0_outputs_size),
                  BuiltinOptions_NONE),
          CreateOperator(builder, 0,
                  builder.CreateVector(operator1_inputs, operator1_inputs_size),
                  builder.CreateVector(operator1_outputs, operator1_outputs_size),
                  BuiltinOptions_NONE)};
  constexpr size_t subgraphs_size = 1;
  const Offset<SubGraph> subgraphs[subgraphs_size] = {
          CreateSubGraph(builder, builder.CreateVector(tensors, tensors_size),
                         builder.CreateVector(inputs, inputs_size),
                         builder.CreateVector(outputs, outputs_size),
                         builder.CreateVector(operators, operators_size),
                         builder.CreateString("test_subgraph"))};
  constexpr size_t operator_codes_size = 2;
  const Offset<OperatorCode> operator_codes[operator_codes_size] = {
          CreateOperatorCodeDirect(builder, BuiltinOperator_CUSTOM, "mock_custom_add",
                                   0),
          CreateOperatorCodeDirect(builder, BuiltinOperator_CUSTOM, "mock_custom_id",
                                   0)};
  const Offset<Model> model_offset = CreateModel(
          builder, 0, builder.CreateVector(operator_codes, operator_codes_size),
          builder.CreateVector(subgraphs, subgraphs_size),
          builder.CreateString("test_model"),
          builder.CreateVector(buffers, buffers_size));
  FinishModelBuffer(builder, model_offset);
  void* model_pointer = builder.GetBufferPointer();
  const Model* model = flatbuffers::GetRoot<Model>(model_pointer);
  return model;
}

const Model* GetMockModel() {
  static Model* model = nullptr;
  if (!model) {
    model = const_cast<Model*>(BuildMockModel());
  }
  return model;
}

const Model* GetLargerMockModel() {
  static Model* model = nullptr;
  if (!model) {
    model = const_cast<Model*>(BuildLargerMockModel());
  }
  return model;
}

}  // namespace
}  // namespace tflite


#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICRO_TESTING_MICRO_TEST_H_
