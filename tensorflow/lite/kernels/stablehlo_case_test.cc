#include <stdint.h>

#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/kernels/subgraph_test_util.h"

namespace tflite {

using subgraph_test_util::CheckIntTensor;
using subgraph_test_util::CheckScalarStringTensor;
using subgraph_test_util::CheckStringTensor;
using subgraph_test_util::ControlFlowOpTest;
using subgraph_test_util::FillIntTensor;
using subgraph_test_util::FillScalarStringTensor;
using subgraph_test_util::FillQuantizedTensor;
using subgraph_test_util::CheckInt8Tensor;

namespace {

// A simple test that performs `ADD` if index is 0, and `MUL` otherwise.
class SimpleCaseTest : public ControlFlowOpTest {
 protected:
  void SetUp() override {
    AddSubgraphs(2);
    builder_->BuildAddSubgraph(interpreter_->subgraph(1));
    builder_->BuildMulSubgraph(interpreter_->subgraph(2));
    builder_->BuildCaseSubgraph(&interpreter_->primary_subgraph(),2);

    interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1});
    interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1,2});
    interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {1, 2});
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
    
    FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {5, 7});
    FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {1, 2});
  }
};

TEST_F(SimpleCaseTest, TestCaseMul) {
  interpreter_->typed_input_tensor<int>(0)[0] = 1;
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
   TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[0]);
  TfLiteTensor* output1 = interpreter_->tensor(interpreter_->subgraph(1)->outputs()[0]);
  CheckIntTensor(output, {1, 2}, {5, 14});
}

TEST_F(SimpleCaseTest, TestCaseAdd) {
  interpreter_->typed_input_tensor<int>(0)[0] = 0;
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output, {1, 2}, {6, 9});
}

TEST_F(SimpleCaseTest, TestCaseIndex0WithLargeInputsTwice) {
  const size_t kNumLargeTensors = 100000;
  interpreter_->ResizeInputTensor(interpreter_->inputs()[1],
                                  {kNumLargeTensors});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {1});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  const std::vector<int> input_vector(kNumLargeTensors, 1);
  interpreter_->typed_input_tensor<int>(0)[0] = 0;
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), input_vector);
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {9});
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[0]);
  const std::vector<int> expected(kNumLargeTensors, 10);
  CheckIntTensor(output, {kNumLargeTensors}, expected);

  // Second invocation.
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {19});
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  output = interpreter_->tensor(interpreter_->outputs()[0]);
  const std::vector<int> expected2(kNumLargeTensors, 20);
  CheckIntTensor(output, {kNumLargeTensors}, expected2);
}

TEST_F(SimpleCaseTest, TestCaseIndex1WithLargeInputsTwice) {
  const size_t kNumLargeTensors = 100000;
  interpreter_->ResizeInputTensor(interpreter_->inputs()[1],
                                  {kNumLargeTensors});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {1});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  const std::vector<int> input_vector(kNumLargeTensors, 1);
  interpreter_->typed_input_tensor<int>(0)[0] = 1;
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), input_vector);
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {0});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[0]);
  const std::vector<int> expected(kNumLargeTensors, 0);
  CheckIntTensor(output, {kNumLargeTensors}, expected);

  // Second invocation.
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {7});
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  output = interpreter_->tensor(interpreter_->outputs()[0]);
  const std::vector<int> expected2(kNumLargeTensors, 7);
  CheckIntTensor(output, {kNumLargeTensors}, expected2);
}

class SimpleCaseTestWithMultipleBranches : public ControlFlowOpTest {
 protected:
  void SetUp() override {
    AddSubgraphs(4);
    builder_->BuildAddSubgraph(interpreter_->subgraph(1));
    builder_->BuildMulSubgraph(interpreter_->subgraph(2));
    builder_->BuildMaximumSubgraph(interpreter_->subgraph(3));
    builder_->BuildMinimumSubgraph(interpreter_->subgraph(4));
    builder_->BuildCaseSubgraph(&interpreter_->primary_subgraph(),4);

    interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1});
    interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1,2});
    interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {1, 2});
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
    
    FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {5, 7});
    FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {1, 10});
  }
};

TEST_F(SimpleCaseTestWithMultipleBranches, TestCaseMax) {
  interpreter_->typed_input_tensor<int>(0)[0] = 2;
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
   TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[0]);
  TfLiteTensor* output1 = interpreter_->tensor(interpreter_->subgraph(1)->outputs()[0]);
  CheckIntTensor(output, {1, 2}, {5,10});
}

TEST_F(SimpleCaseTestWithMultipleBranches, TestCaseMin) {
  interpreter_->typed_input_tensor<int>(0)[0] = 3;
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
   TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[0]);
  TfLiteTensor* output1 = interpreter_->tensor(interpreter_->subgraph(1)->outputs()[0]);
  CheckIntTensor(output, {1, 2}, {1,7});
}

class SimpleQuantizedCaseTest : public ControlFlowOpTest {
 protected:
  void SetUp() override {
    AddSubgraphs(2);
    builder_->BuildAddSubgraph(interpreter_->subgraph(1));
    builder_->BuildMulSubgraph(interpreter_->subgraph(2));
    builder_->BuildCaseSubgraph(&interpreter_->primary_subgraph(),2);

    interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1});
    interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1,2});
    interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {1, 2});
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
    interpreter_->tensor(interpreter_->inputs()[1])->type = kTfLiteInt8;
    interpreter_->tensor(interpreter_->inputs()[2])->type = kTfLiteInt8;
    FillQuantizedTensor(interpreter_->tensor(interpreter_->inputs()[1]), {2, 4},0.5,0);
    FillQuantizedTensor(interpreter_->tensor(interpreter_->inputs()[2]), {1, 2},0.5,0);
  }
};

TEST_F(SimpleQuantizedCaseTest, TestQuantized) {
  interpreter_->typed_input_tensor<int>(0)[0] = 1;
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
   TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[0]);
   output->type = kTfLiteInt8;
   std::cout<<"tensor_type "<<interpreter_->tensor(interpreter_->inputs()[1])->type<<std::endl;
  CheckInt8Tensor(output, {1, 2}, {int8_t(8), int8_t(32)});
}

class CaseTest : public ControlFlowOpTest {};

TEST_F(CaseTest, TestInputIsOutput) {
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);
  builder_->BuildInputIsOutputSubgraph(interpreter_->subgraph(1));
  builder_->BuildInputIsOutputSubgraph(interpreter_->subgraph(2));
  builder_->BuildMultiInputCaseSubgraph(&interpreter_->primary_subgraph(),2,4);
  
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[3], {1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  interpreter_->typed_input_tensor<int>(0)[0] = 0;
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {1});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {1});
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[3]), {1});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  TfLiteTensor* output0 = interpreter_->tensor(interpreter_->outputs()[0]);
  CheckIntTensor(output0, {1}, {2});
  TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[1]);
  CheckIntTensor(output1, {1}, {2});

  interpreter_->typed_input_tensor<int>(0)[0] = false;
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  CheckIntTensor(output0, {1}, {2});
  CheckIntTensor(output1, {1}, {2});
}

TEST_F(CaseTest, TestStaticUnconsumedOutputs) {
  for (int dynamic_tensors : {0, 1}) {
    interpreter_ = std::make_unique<Interpreter>();
    AddSubgraphs(2);
    builder_->BuildInputIsOutputSubgraph(interpreter_->subgraph(1));
    builder_->BuildInputIsOutputSubgraph(interpreter_->subgraph(2));
    builder_->BuildMultiInputCaseSubgraphWithUnconsumedOutput(
        &interpreter_->primary_subgraph(), 2,4);

    InterpreterOptions options;
    if (dynamic_tensors) {
      options.OptimizeMemoryForLargeTensors(1);
      interpreter_->ApplyOptions(&options);
    }

    ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1}),
              kTfLiteOk);
    ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1}),
              kTfLiteOk);
    ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {1}),
              kTfLiteOk);
    ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[3], {1}),
              kTfLiteOk);
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
    interpreter_->typed_input_tensor<int>(0)[0] = 0;
    FillIntTensor(interpreter_->tensor(interpreter_->inputs()[1]), {1});
    FillIntTensor(interpreter_->tensor(interpreter_->inputs()[2]), {2});
    FillIntTensor(interpreter_->tensor(interpreter_->inputs()[3]), {2});

    ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
    TfLiteTensor* output0 = interpreter_->tensor(interpreter_->outputs()[0]);
    CheckIntTensor(output0, {1}, {2});
    TfLiteTensor* output1 = interpreter_->tensor(interpreter_->outputs()[1]);
    CheckIntTensor(output1, {1}, {4});

    ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[3], {2}),
              kTfLiteOk);
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
    FillIntTensor(interpreter_->tensor(interpreter_->inputs()[3]), {2, 2});

    ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
    CheckIntTensor(output1, {2}, {4, 4});
    ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
    interpreter_->typed_input_tensor<int>(0)[0] = 1;
    ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  }
}

TEST_F(CaseTest, TestCaseLoopWithDynamicTensor) {
  interpreter_ = std::make_unique<Interpreter>();
  AddSubgraphs(2);
  builder_->BuildBodySubgraphWithDynamicTensor(interpreter_->subgraph(1));
  builder_->BuildBodySubgraphWithDynamicTensor(interpreter_->subgraph(2));
  builder_->BuildCaseSubgraphWithDynamicTensor(&interpreter_->primary_subgraph(),2);

  interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[2], {});
  interpreter_->ResizeInputTensor(interpreter_->inputs()[3], {1});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  interpreter_->typed_input_tensor<int>(0)[0] = 1;
  FillScalarStringTensor(interpreter_->tensor(interpreter_->inputs()[1]), "A");
  FillScalarStringTensor(interpreter_->tensor(interpreter_->inputs()[2]), "A");
  FillIntTensor(interpreter_->tensor(interpreter_->inputs()[3]), {1});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* string_output1 =
      interpreter_->tensor(interpreter_->outputs()[0]);
  CheckScalarStringTensor(string_output1, "A");
  TfLiteTensor* string_output2 =
      interpreter_->tensor(interpreter_->outputs()[1]);
  CheckStringTensor(string_output2, {2}, {"A", "A"});
  TfLiteTensor* integer_output =
      interpreter_->tensor(interpreter_->outputs()[2]);
  CheckIntTensor(integer_output, {1}, {2});

  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
}

}
}
