/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "tensorflow/lite/array.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/model_building.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_options.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/subgraph_test_util.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/util.h"

using testing::Contains;
using testing::StrEq;
// Before C++20, a template function with explicit arguments cannot be found via
// ADL. We need to explicitly pull it.
//
// https://stackoverflow.com/a/65225344
using tflite::model_builder::NewInputs;

// Needs to be in the global namespace because GTest relies on ADL to find it.
void PrintTo(const TfLiteRegistration& reg, std::ostream* os) {
  if (os) {
    *os << "is a "
        << EnumNameBuiltinOperator(
               static_cast<tflite::BuiltinOperator>(reg.builtin_code))
        << " op registration";
    if (reg.custom_name) {
      *os << " called " << reg.custom_name;
    }
  }
}

namespace tflite {

class OpRegistrationMatcher {
 public:
  using is_gtest_matcher = void;

  explicit OpRegistrationMatcher(BuiltinOperator op_code,
                                 std::string custom_name = {})
      : op_code_(op_code), custom_name_(std::move(custom_name)) {}

  bool MatchAndExplain(const TfLiteRegistration& reg, std::ostream* os) const {
    bool result = reg.builtin_code == op_code_;
    if (!custom_name_.empty()) {
      result = result && (custom_name_ == reg.custom_name);
    }
    return result;
  }

  // Describes the property of a value matching this matcher.
  void DescribeTo(std::ostream* os) const {
    *os << "is a " << EnumNameBuiltinOperator(op_code_) << " op registration";
    if (!custom_name_.empty()) {
      *os << " called " << custom_name_;
    }
  }

  // Describes the property of a value NOT matching this matcher.
  void DescribeNegationTo(std::ostream* os) const {
    *os << "isn't a " << EnumNameBuiltinOperator(op_code_)
        << " op registration";
    if (!custom_name_.empty()) {
      *os << " called " << custom_name_;
    }
  }

 private:
  BuiltinOperator op_code_;
  std::string custom_name_;
};

OpRegistrationMatcher IsOp(BuiltinOperator op) {
  return OpRegistrationMatcher(op);
}

OpRegistrationMatcher IsDelegateOp(std::string name = {}) {
  return OpRegistrationMatcher(BuiltinOperator_DELEGATE, std::move(name));
}

OpRegistrationMatcher IsCustomOp(std::string name = {}) {
  return OpRegistrationMatcher(BuiltinOperator_CUSTOM, std::move(name));
}

namespace {
// Mocks the delegation process in order to test the different cases of
// inlining a StableHLO composite op.
//
// This delegate does not actually run the nodes that it replaces. It only
// keeps track of which nodes should have been evaluated.
class CompositeDelegate {
 protected:
  CompositeDelegate() = default;

 public:
  struct Deleter {
    void operator()(TfLiteDelegate* tflite_delegate) const {
      delete reinterpret_cast<CompositeDelegate*>(tflite_delegate->data_);
      delete tflite_delegate;
    }
  };

  // Builds the delegate pointer that should be passed to
  // `ModifyGraphWithDlegate`.
  static std::unique_ptr<TfLiteDelegate, Deleter> Build() {
    // This is safe. The constructor is protected so we can't call
    // make_unique.
    std::unique_ptr<CompositeDelegate> delegate(new CompositeDelegate());
    auto tflite_delegate = std::make_unique<TfLiteDelegate>(
        TfLiteDelegate{.data_ = delegate.get(),
                       .Prepare = PrepareDelegate,
                       .CopyFromBufferHandle = nullptr,
                       .CopyToBufferHandle = nullptr,
                       .FreeBufferHandle = nullptr,
                       .flags = 0,
                       .opaque_delegate_builder = nullptr});
    delegate.release();
    return std::unique_ptr<TfLiteDelegate, Deleter>(tflite_delegate.release());
  }

  // Declares the given names as composite ops supported by the delegate.
  void AddSupportedComposite(std::initializer_list<std::string> names) {
    supported_composite_ops_.insert(names);
  }

  // Declares the given ops as supported by the delegate.
  void AddSupportedBuiltinOp(std::initializer_list<BuiltinOperator> ops) {
    supported_ops_.insert(ops);
  }

  // Gets the delegate logs.
  const std::vector<std::string>& GetLogs() const { return log_; }

  // Checks if the given node and registration have been declared as supported
  // by the delegate.
  bool SupportsOp(const TfLiteNode& node, const TfLiteRegistration& reg) const {
    if (reg.builtin_code == kTfLiteBuiltinStablehloComposite) {
      const TfLiteStablehloCompositeParams& params =
          *reinterpret_cast<TfLiteStablehloCompositeParams*>(node.builtin_data);
      return supported_composite_ops_.count(params.name);
    }
    return supported_ops_.count(static_cast<BuiltinOperator>(reg.builtin_code));
  }

  // Retrieves the delegate object from the given delegate node.
  static CompositeDelegate& GetDelegate(TfLiteNode* node) {
    return *reinterpret_cast<CompositeDelegate*>(node->delegate->data_);
  }

  // Prepares the delegation process.
  //
  // WARNING: this is not the delegate node prepare function.
  static TfLiteStatus PrepareDelegate(TfLiteContext* context,
                                      TfLiteDelegate* delegate) {
    TfLiteIntArray* execution_plan;
    TF_LITE_ENSURE_STATUS(context->GetExecutionPlan(context, &execution_plan));
    TF_LITE_ENSURE(context, delegate->data_);
    CompositeDelegate& this_delegate =
        *reinterpret_cast<CompositeDelegate*>(delegate->data_);

    std::vector<int> nodes_to_delegate;

    for (int i = 0; i < execution_plan->size; ++i) {
      const int node_index = execution_plan->data[i];
      TfLiteNode* node;
      TfLiteRegistration* reg;
      TF_LITE_ENSURE_STATUS(
          context->GetNodeAndRegistration(context, node_index, &node, &reg));
      if (this_delegate.SupportsOp(*node, *reg)) {
        nodes_to_delegate.emplace_back(node_index);
      }
    }

    if (nodes_to_delegate.empty()) {
      return kTfLiteOk;
    }

    TfLiteArrayUniquePtr<int> ops_to_replace =
        BuildTfLiteArray(nodes_to_delegate);

    return context->ReplaceNodeSubsetsWithDelegateKernels(
        context, GetRegistration(), ops_to_replace.get(), delegate);
  }

  struct DelegateNodeData {
    std::vector<std::string> delegated_operators;
  };

  static DelegateNodeData& GetNodeData(TfLiteNode* node) {
    return *reinterpret_cast<DelegateNodeData*>(node->user_data);
  }

  // Initializes the delegate node.
  static void* Init(TfLiteContext* context, const char* buffer, size_t length) {
    const TfLiteDelegateParams& params =
        *reinterpret_cast<const TfLiteDelegateParams*>(buffer);
    auto data = std::make_unique<DelegateNodeData>();
    for (int i = 0; i < params.nodes_to_replace->size; ++i) {
      const int node_index = params.nodes_to_replace->data[i];
      TfLiteNode* node;
      TfLiteRegistration* reg;
      context->GetNodeAndRegistration(context, node_index, &node, &reg);
      data->delegated_operators.push_back(GetOpNameByRegistration(*reg));
    }
    return data.release();
  }

  // Cleans up the delegate node.
  static void Free(TfLiteContext* context, void* buffer) {
    delete reinterpret_cast<DelegateNodeData*>(buffer);
  }

  // Prepares the delegate node.
  static TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
    CompositeDelegate& delegate = GetDelegate(node);
    delegate.log_.emplace_back("Prepare delegate node.");
    return kTfLiteOk;
  }

  // Evaluates the delegate node.
  static TfLiteStatus Invoke(TfLiteContext* context, TfLiteNode* node) {
    CompositeDelegate& delegate = GetDelegate(node);
    DelegateNodeData& data = GetNodeData(node);
    std::string log_entry = "Invoke delegate node:";
    for (const std::string& op : data.delegated_operators) {
      log_entry += " " + op;
    }
    delegate.log_.emplace_back(std::move(log_entry));
    return kTfLiteOk;
  }

  // Returns the registration for the delegate nodes.
  static const TfLiteRegistration& GetRegistration() {
    static TfLiteRegistration reg{.init = Init,
                                  .free = Free,
                                  .prepare = Prepare,
                                  .invoke = Invoke,
                                  .builtin_code = BuiltinOperator_DELEGATE,
                                  .custom_name = "CompositeTestDelegate"};
    return reg;
  }

 private:
  std::set<BuiltinOperator> supported_ops_;
  std::set<std::string> supported_composite_ops_;
  std::vector<std::string> log_;
};

// Retrieves the delegate object from the type erased TFLite delegate object.
CompositeDelegate& GetDelegate(TfLiteDelegate& tflite_delegate) {
  return *reinterpret_cast<CompositeDelegate*>(tflite_delegate.data_);
}

class CompositeInliningTest : public testing::Test {
 public:
  CompositeInliningTest() {
    interpreter_options_.SetShloCompositeInlining(true);
  }

 protected:
  void BuildInterpreter(model_builder::ModelBuilder& model_builder) {
    model_builder.Build(interpreter_);
    // Enable inlining.
    interpreter_.ApplyOptions(&interpreter_options_);
  }

  template <class IndirectionVector>
  TfLiteTensor* GetTensorWithIndirection(int id,
                                         const IndirectionVector& tensor_map) {
    return interpreter_.tensor(tensor_map[id]);
  }

  TfLiteTensor* GetInputTensor(int id) {
    return GetTensorWithIndirection(id, interpreter_.inputs());
  }

  TfLiteTensor* GetOutputTensor(int id) {
    return GetTensorWithIndirection(id, interpreter_.outputs());
  }

  template <class T, class IndirectionVector>
  absl::Span<T> GetTensorDataWithIndirection(
      int id, const IndirectionVector& tensor_map) {
    TfLiteTensor* const tensor = GetTensorWithIndirection(id, tensor_map);
    const size_t size = NumElements(tensor);
    return absl::Span<T>(GetTensorData<T>(tensor), size);
  }

  template <class T>
  absl::Span<T> GetInputData(int id) {
    return GetTensorDataWithIndirection<T>(id, interpreter_.inputs());
  }

  template <class T>
  absl::Span<T> GetOutputData(int id) {
    return GetTensorDataWithIndirection<T>(id, interpreter_.outputs());
  }

  // Returns the op code of the idx-th operation in the interpreter_'s execution
  // plan.
  const TfLiteRegistration& GetOpRegistration(int idx) {
    const int node_idx = interpreter_.execution_plan()[idx];
    const auto& [node, reg] = *interpreter_.node_and_registration(node_idx);
    return reg;
  };

  Interpreter interpreter_;
  InterpreterOptions interpreter_options_;
};

TEST_F(CompositeInliningTest, CompositeDelegateWorks) {
  model_builder::ModelBuilder model_builder;
  auto primary_graph = NewGraph(model_builder);
  auto decomposition_graph = NewGraph(model_builder);
  {
    auto [in1, in2] = NewInputs<2>(decomposition_graph, kTfLiteInt32);
    auto add1 = Mul(in1, in2);
    MarkOutput(add1);
  }

  {
    auto [in1, in2] = NewInputs<2>(primary_graph, kTfLiteInt32);
    auto outputs =
        StableHLOComposite("test_composite", decomposition_graph, {in1, in2});
    MarkOutput(outputs[0]);
  }

  BuildInterpreter(model_builder);

  ASSERT_THAT(interpreter_.execution_plan().size(), 1);
  ASSERT_THAT(GetOpRegistration(0), IsOp(BuiltinOperator_STABLEHLO_COMPOSITE));

  auto delegate = CompositeDelegate::Build();
  CompositeDelegate& composite_delegate = GetDelegate(*delegate);
  composite_delegate.AddSupportedComposite({"test_composite"});

  interpreter_.ResizeInputTensor(interpreter_.inputs()[0], {2, 3});
  interpreter_.ResizeInputTensor(interpreter_.inputs()[1], {2, 3});

  interpreter_.ModifyGraphWithDelegate(std::move(delegate));

  ASSERT_THAT(interpreter_.execution_plan().size(), 1);
  ASSERT_THAT(GetOpRegistration(0), IsDelegateOp("CompositeTestDelegate"));

  ASSERT_EQ(interpreter_.AllocateTensors(), kTfLiteOk);

  subgraph_test_util::FillIntTensor(GetInputTensor(0), {1, 2, 3, 4, 5, 6});
  subgraph_test_util::FillIntTensor(GetInputTensor(1), {7, 8, 9, 10, 11, 12});

  ASSERT_EQ(interpreter_.Invoke(), kTfLiteOk);

  const TfLiteTensor* const output = GetOutputTensor(0);
  ASSERT_THAT(output, DimsAre({2, 3}));

  EXPECT_THAT(composite_delegate.GetLogs(),
              Contains(StrEq("Prepare delegate node.")));
  EXPECT_THAT(composite_delegate.GetLogs(),
              Contains(StrEq("Invoke delegate node: STABLEHLO_COMPOSITE")));
}

TEST_F(CompositeInliningTest, CompositeIsInlinedByDefault) {
  model_builder::ModelBuilder model_builder;
  auto primary_graph = NewGraph(model_builder);
  auto decomposition_graph = NewGraph(model_builder);

  {
    auto [in1, in2, in3] = NewInputs<3>(decomposition_graph, kTfLiteInt32);
    auto mul1 = Mul(in1, in2);
    auto add1 = Add(mul1, in3);
    MarkOutput(add1);
  }

  {
    auto [in1, in2, in3] = NewInputs<3>(primary_graph, kTfLiteInt32);
    auto primary_outputs = StableHLOComposite(
        "composite_fma", decomposition_graph, {in1, in2, in3});
    MarkOutput(primary_outputs[0]);
  }

  BuildInterpreter(model_builder);

  interpreter_.ResizeInputTensor(interpreter_.inputs()[0], {2, 3});
  interpreter_.ResizeInputTensor(interpreter_.inputs()[1], {2, 3});
  interpreter_.ResizeInputTensor(interpreter_.inputs()[2], {2, 3});

  // Before allocation, we have a composite op.
  ASSERT_THAT(GetOpRegistration(0), IsOp(BuiltinOperator_STABLEHLO_COMPOSITE));

  ASSERT_EQ(interpreter_.AllocateTensors(), kTfLiteOk);

  // After allocation, the subgraph has been inlined.
  ASSERT_EQ(interpreter_.execution_plan().size(), 2);
  ASSERT_THAT(GetOpRegistration(0), IsOp(BuiltinOperator_MUL));
  ASSERT_THAT(GetOpRegistration(1), IsOp(BuiltinOperator_ADD));

  subgraph_test_util::FillIntTensor(GetInputTensor(0), {1, 2, 3, 4, 5, 6});
  subgraph_test_util::FillIntTensor(GetInputTensor(1), {7, 8, 9, 10, 11, 12});
  subgraph_test_util::FillIntTensor(GetInputTensor(2),
                                    {13, 14, 15, 16, 17, 18});

  ASSERT_EQ(interpreter_.Invoke(), kTfLiteOk);

  // Check that the results are correct (everything is wired up correctly).
  EXPECT_THAT(GetOutputData<int>(0),
              testing::ElementsAre(20, 30, 42, 56, 72, 90));
}

TEST_F(CompositeInliningTest, DelegateHasPrecedenceOverInlining) {
  model_builder::ModelBuilder model_builder;
  auto primary_graph = NewGraph(model_builder);
  auto decomposition_graph = NewGraph(model_builder);

  {
    auto [in1, in2, in3] = NewInputs<3>(decomposition_graph, kTfLiteInt32);
    auto mul1 = Mul(in1, in2);
    auto add1 = Add(mul1, in3);
    MarkOutput(add1);
  }

  {
    auto [in1, in2, in3] = NewInputs<3>(primary_graph, kTfLiteInt32);
    auto primary_outputs = StableHLOComposite(
        "composite_fma", decomposition_graph, {in1, in2, in3});
    MarkOutput(primary_outputs[0]);
  }

  BuildInterpreter(model_builder);

  interpreter_.ResizeInputTensor(interpreter_.inputs()[0], {2, 3});
  interpreter_.ResizeInputTensor(interpreter_.inputs()[1], {2, 3});
  interpreter_.ResizeInputTensor(interpreter_.inputs()[2], {2, 3});

  // Before allocation, we have a composite op.
  ASSERT_EQ(interpreter_.execution_plan().size(), 1);
  ASSERT_THAT(GetOpRegistration(0), IsOp(BuiltinOperator_STABLEHLO_COMPOSITE));

  auto delegate = CompositeDelegate::Build();
  CompositeDelegate& composite_delegate = GetDelegate(*delegate);
  composite_delegate.AddSupportedComposite({"composite_fma"});
  interpreter_.ModifyGraphWithDelegate(std::move(delegate));

  ASSERT_EQ(interpreter_.AllocateTensors(), kTfLiteOk);

  // After allocation, the subgraph has been delegated.
  ASSERT_EQ(interpreter_.execution_plan().size(), 1);
  ASSERT_THAT(GetOpRegistration(0), IsOp(BuiltinOperator_DELEGATE));

  subgraph_test_util::FillIntTensor(GetInputTensor(0), {1, 2, 3, 4, 5, 6});
  subgraph_test_util::FillIntTensor(GetInputTensor(1), {7, 8, 9, 10, 11, 12});
  subgraph_test_util::FillIntTensor(GetInputTensor(2),
                                    {13, 14, 15, 16, 17, 18});

  ASSERT_EQ(interpreter_.Invoke(), kTfLiteOk);

  // Check that the composite node has been delegated.
  EXPECT_THAT(composite_delegate.GetLogs(),
              Contains(StrEq("Prepare delegate node.")));
  EXPECT_THAT(composite_delegate.GetLogs(),
              Contains(StrEq("Invoke delegate node: STABLEHLO_COMPOSITE")));
}

TEST_F(CompositeInliningTest, RecursiveInliningWorks) {
  model_builder::ModelBuilder model_builder;
  auto primary_graph = NewGraph(model_builder);
  auto decomposition_graph_with_composite = NewGraph(model_builder);
  auto supported_decomposition_graph = NewGraph(model_builder);

  {
    auto [in1, in2] = NewInputs<2>(supported_decomposition_graph, kTfLiteInt32);
    auto add = Add(in1, in2);
    auto positive = Abs(add);
    MarkOutput(positive);
  }

  {
    auto [in1, in2, in3] =
        NewInputs<3>(decomposition_graph_with_composite, kTfLiteInt32);
    auto mul = Mul(in1, in2);
    auto composite = StableHLOComposite(
        "add_abs", supported_decomposition_graph, {mul, in3});
    auto abs = Abs(composite[0]);
    MarkOutput(abs);
  }

  {
    auto [in1, in2, in3] = NewInputs<3>(primary_graph, kTfLiteInt32);
    auto composite = StableHLOComposite(
        "composite", decomposition_graph_with_composite, {in1, in2, in3});
    MarkOutput(composite[0]);
  }

  BuildInterpreter(model_builder);

  interpreter_.ResizeInputTensor(interpreter_.inputs()[0], {2, 3});
  interpreter_.ResizeInputTensor(interpreter_.inputs()[1], {2, 3});
  interpreter_.ResizeInputTensor(interpreter_.inputs()[2], {2, 3});

  ASSERT_EQ(interpreter_.AllocateTensors(), kTfLiteOk);

  // After allocation, the subgraph has been inlined.
  ASSERT_EQ(interpreter_.execution_plan().size(), 4);
  EXPECT_THAT(GetOpRegistration(0), IsOp(BuiltinOperator_MUL));
  EXPECT_THAT(GetOpRegistration(1), IsOp(BuiltinOperator_ADD));
  EXPECT_THAT(GetOpRegistration(2), IsOp(BuiltinOperator_ABS));
  EXPECT_THAT(GetOpRegistration(3), IsOp(BuiltinOperator_ABS));

  subgraph_test_util::FillIntTensor(GetInputTensor(0), {1, 2, 3, 4, 5, 6});
  subgraph_test_util::FillIntTensor(GetInputTensor(1), {7, 8, 9, 10, 11, 12});
  subgraph_test_util::FillIntTensor(GetInputTensor(2),
                                    {13, 14, 15, 16, 17, 18});

  ASSERT_EQ(interpreter_.Invoke(), kTfLiteOk);

  EXPECT_THAT(GetOutputData<int>(0),
              testing::ElementsAre(20, 30, 42, 56, 72, 90));
}

TEST_F(CompositeInliningTest, InliningSharedDecompositionWorks) {
  model_builder::ModelBuilder model_builder;
  auto primary_graph = NewGraph(model_builder);
  auto decomposition_graph = NewGraph(model_builder);

  {
    auto [in1, in2] = NewInputs<2>(decomposition_graph, kTfLiteInt32);
    auto add = Add(in1, in2);
    auto abs = Abs(add);
    MarkOutput(abs);
  }

  {
    auto [in1, in2] = NewInputs<2>(primary_graph, kTfLiteInt32);
    auto composite1 =
        StableHLOComposite("composite", decomposition_graph, {in1, in2});
    auto composite2 = StableHLOComposite("composite", decomposition_graph,
                                         {composite1[0], in2});
    MarkOutput(composite2[0]);
  }

  BuildInterpreter(model_builder);

  interpreter_.ResizeInputTensor(interpreter_.inputs()[0], {2, 2});
  interpreter_.ResizeInputTensor(interpreter_.inputs()[1], {2, 2});

  ASSERT_EQ(interpreter_.AllocateTensors(), kTfLiteOk);

  // After allocation, both composites ops have been inlined.
  ASSERT_EQ(interpreter_.execution_plan().size(), 4);
  EXPECT_THAT(GetOpRegistration(0), IsOp(BuiltinOperator_ADD));
  EXPECT_THAT(GetOpRegistration(1), IsOp(BuiltinOperator_ABS));
  EXPECT_THAT(GetOpRegistration(2), IsOp(BuiltinOperator_ADD));
  EXPECT_THAT(GetOpRegistration(3), IsOp(BuiltinOperator_ABS));

  subgraph_test_util::FillIntTensor(GetInputTensor(0), {1, 2, 3, 4});
  subgraph_test_util::FillIntTensor(GetInputTensor(1), {-4, -3, -2, -1});

  ASSERT_EQ(interpreter_.Invoke(), kTfLiteOk);

  EXPECT_THAT(GetOutputData<int>(0), testing::ElementsAre(1, 2, 1, 2));
}

TEST_F(CompositeInliningTest, InlinedOpsDecompositionIsDelegated) {
  model_builder::ModelBuilder model_builder;
  auto primary_graph = NewGraph(model_builder);
  auto decomposition_graph = NewGraph(model_builder);

  {
    auto [in1, in2] = NewInputs<2>(decomposition_graph, kTfLiteInt32);
    auto mul = Mul(in1, in2);
    MarkOutput(mul);
  }

  {
    auto [in1, in2] = NewInputs<2>(primary_graph, kTfLiteInt32);
    auto inter1 = Abs(in1);
    auto composite =
        StableHLOComposite("composite", decomposition_graph, {inter1, in2});
    auto out = Abs(composite[0]);
    MarkOutput(out);
  }

  BuildInterpreter(model_builder);

  ASSERT_EQ(interpreter_.execution_plan().size(), 3);
  EXPECT_THAT(GetOpRegistration(0), IsOp(BuiltinOperator_ABS));
  EXPECT_THAT(GetOpRegistration(1), IsOp(BuiltinOperator_STABLEHLO_COMPOSITE));
  EXPECT_THAT(GetOpRegistration(2), IsOp(BuiltinOperator_ABS));

  auto delegate = CompositeDelegate::Build();
  CompositeDelegate& composite_delegate = GetDelegate(*delegate);
  composite_delegate.AddSupportedBuiltinOp(
      {BuiltinOperator_ABS, BuiltinOperator_MUL});
  interpreter_.ModifyGraphWithDelegate(std::move(delegate));

  ASSERT_EQ(interpreter_.execution_plan().size(), 3);
  EXPECT_THAT(GetOpRegistration(0), IsOp(BuiltinOperator_DELEGATE));
  EXPECT_THAT(GetOpRegistration(1), IsOp(BuiltinOperator_STABLEHLO_COMPOSITE));
  EXPECT_THAT(GetOpRegistration(2), IsOp(BuiltinOperator_DELEGATE));

  ASSERT_EQ(interpreter_.AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter_.execution_plan().size(), 1);
  EXPECT_THAT(GetOpRegistration(0), IsOp(BuiltinOperator_DELEGATE));
}

TEST_F(CompositeInliningTest, MultipleDelegatesCanBeAppliedBeforeInlining) {
  model_builder::ModelBuilder model_builder;
  auto primary_graph = NewGraph(model_builder);
  auto decomposition_graph = NewGraph(model_builder);

  {
    auto [in1, in2] = NewInputs<2>(decomposition_graph, kTfLiteInt32);
    auto mul = Mul(in1, in2);
    MarkOutput(mul);
  }

  {
    auto [in1, in2] = NewInputs<2>(primary_graph, kTfLiteInt32);
    auto composite =
        StableHLOComposite("composite", decomposition_graph, {in1, in2});
    auto composite2 =
        StableHLOComposite("composite2", decomposition_graph, {in1, in2});
    auto add = Add(composite[0], composite2[0]);
    MarkOutput(add);
  }

  BuildInterpreter(model_builder);

  {  // This delegate only supports "composite".
    auto delegate = CompositeDelegate::Build();
    CompositeDelegate& composite_delegate = GetDelegate(*delegate);
    composite_delegate.AddSupportedComposite({"composite"});
    interpreter_.ModifyGraphWithDelegate(std::move(delegate));
  }

  {  // This delegate only supports "composite2".
    auto delegate = CompositeDelegate::Build();
    CompositeDelegate& composite_delegate = GetDelegate(*delegate);
    composite_delegate.AddSupportedComposite({"composite2"});
    interpreter_.ModifyGraphWithDelegate(std::move(delegate));
  }

  ASSERT_EQ(interpreter_.AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter_.execution_plan().size(), 3);
  EXPECT_THAT(GetOpRegistration(0), IsOp(BuiltinOperator_DELEGATE));
  EXPECT_THAT(GetOpRegistration(1), IsOp(BuiltinOperator_DELEGATE));
  EXPECT_THAT(GetOpRegistration(2), IsOp(BuiltinOperator_ADD));
}

// When supported and unsupported composite ops are used, the unsupported ones
// are inlined and the supported ones are delegated.
TEST_F(CompositeInliningTest, MixedSupportOfCompositeWorks) {
  model_builder::ModelBuilder model_builder;
  auto primary_graph = NewGraph(model_builder);
  auto decomposition_graph = NewGraph(model_builder);
  {
    auto in1 = NewInput(decomposition_graph, kTfLiteInt32);
    auto out = Abs(in1);
    MarkOutput(out);
  }

  {
    auto in1 = NewInput(primary_graph, kTfLiteInt32);
    auto c1 = StableHLOComposite("supported", decomposition_graph, {in1})[0];
    auto c2 = StableHLOComposite("unsupported", decomposition_graph, {c1})[0];
    auto c3 = StableHLOComposite("supported", decomposition_graph, {c2})[0];
    auto c4 = StableHLOComposite("unsupported", decomposition_graph, {c3})[0];
    MarkOutput(c4);
  }

  BuildInterpreter(model_builder);

  {
    auto delegate = CompositeDelegate::Build();
    CompositeDelegate& composite_delegate = GetDelegate(*delegate);
    composite_delegate.AddSupportedComposite({"supported"});
    interpreter_.ModifyGraphWithDelegate(std::move(delegate));
  }

  ASSERT_EQ(interpreter_.AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter_.execution_plan().size(), 4);
  EXPECT_THAT(GetOpRegistration(0), IsOp(BuiltinOperator_DELEGATE));
  EXPECT_THAT(GetOpRegistration(1), IsOp(BuiltinOperator_ABS));
  EXPECT_THAT(GetOpRegistration(2), IsOp(BuiltinOperator_DELEGATE));
  EXPECT_THAT(GetOpRegistration(3), IsOp(BuiltinOperator_ABS));
}

// Inlining multiple composites interleaved with other ops works correctly.
TEST_F(CompositeInliningTest, MixedInliningOfCompositeWorks) {
  model_builder::ModelBuilder model_builder;
  auto primary_graph = NewGraph(model_builder);
  auto decomposition_graph_1 = NewGraph(model_builder);
  auto decomposition_graph_2 = NewGraph(model_builder);
  {
    auto in1 = NewInput(decomposition_graph_1, kTfLiteInt32);
    auto in2 = NewInput(decomposition_graph_1, kTfLiteInt32);
    auto out = Add(in1, in2);
    MarkOutput(out);
  }
  {
    auto in1 = NewInput(decomposition_graph_2, kTfLiteInt32);
    auto in2 = NewInput(decomposition_graph_2, kTfLiteInt32);
    auto out = Add(in1, in2);
    MarkOutput(out);
  }

  {
    auto in1 = NewInput(primary_graph, kTfLiteInt32);
    auto in2 = NewInput(primary_graph, kTfLiteInt32);
    auto c1 = Add(in1, in2);
    auto c2 =
        StableHLOComposite("unsupported", decomposition_graph_1, {c1, in2})[0];
    auto c3 = Add(c2, in2);
    auto c4 =
        StableHLOComposite("unsupported", decomposition_graph_2, {c3, in2})[0];
    MarkOutput(c4);
  }

  BuildInterpreter(model_builder);

  interpreter_.ResizeInputTensor(interpreter_.inputs()[0], {1, 4});
  interpreter_.ResizeInputTensor(interpreter_.inputs()[1], {1, 4});

  ASSERT_EQ(interpreter_.AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter_.execution_plan().size(), 4);
  EXPECT_THAT(GetOpRegistration(0), IsOp(BuiltinOperator_ADD));
  EXPECT_THAT(GetOpRegistration(1), IsOp(BuiltinOperator_ADD));
  EXPECT_THAT(GetOpRegistration(2), IsOp(BuiltinOperator_ADD));
  EXPECT_THAT(GetOpRegistration(3), IsOp(BuiltinOperator_ADD));

  subgraph_test_util::FillIntTensor(GetInputTensor(0), {1, 2, 3, 4});
  subgraph_test_util::FillIntTensor(GetInputTensor(1), {1, 1, 1, 1});

  ASSERT_EQ(interpreter_.Invoke(), kTfLiteOk);

  EXPECT_THAT(GetOutputData<int>(0), testing::ElementsAre(5, 6, 7, 8));
}

// Some ops define a parameter structure but don't use it. Check that those
// don't break when we try to inline them.
TEST_F(CompositeInliningTest, InliningNoDataNodeWorks) {
  model_builder::ModelBuilder model_builder;
  auto primary_graph = NewGraph(model_builder);
  auto decomposition_graph = NewGraph(model_builder);
  {
    auto [in, perm] = NewInputs<2>(decomposition_graph, kTfLiteInt32);
    auto out = Transpose(in, perm);
    MarkOutput(out);
  }

  {
    auto [in, perm] = NewInputs<2>(primary_graph, kTfLiteInt32);
    auto c2 =
        StableHLOComposite("unsupported", decomposition_graph, {in, perm})[0];
    MarkOutput(c2);
  }

  BuildInterpreter(model_builder);

  ASSERT_EQ(interpreter_.AllocateTensors(), kTfLiteOk);

  EXPECT_THAT(GetOpRegistration(0), IsOp(BuiltinOperator_TRANSPOSE));
}

}  // namespace
}  // namespace tflite
