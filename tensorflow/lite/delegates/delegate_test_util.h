/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_DELEGATES_DELEGATE_TEST_UTIL_H_
#define TENSORFLOW_LITE_DELEGATES_DELEGATE_TEST_UTIL_H_

#include <stdint.h>

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "Eigen/Core"  // from @eigen_archive
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"

namespace tflite {
namespace delegates {
namespace test_utils {

// Build a kernel registration for a custom addition op that adds its two
// tensor inputs to produce a tensor output.
TfLiteRegistration AddOpRegistration();

class SimpleDelegate {
 public:
  // Create a simple implementation of a TfLiteDelegate. We use the C++ class
  // SimpleDelegate and it can produce a handle TfLiteDelegate that is
  // value-copyable and compatible with TfLite.
  //
  // Parameters:
  //   nodes: Indices of the graph nodes that the delegate will handle.
  //   fail_node_prepare: To simulate failure of Delegate node's Prepare().
  //   min_ops_per_subset: If >0, partitioning preview is used to choose only
  //     those subsets with min_ops_per_subset number of nodes.
  //   fail_node_invoke: To simulate failure of Delegate node's Invoke().
  //   automatic_shape_propagation: This assumes that the runtime will
  //     propagate shapes using the original execution plan.
  //   custom_op: If true, the graph nodes specified in the 'nodes' parameter
  //     should be custom ops with name "my_add"; if false, they should be
  //     the builtin ADD operator.
  //   set_output_tensor_dynamic: If True, this delegate sets output tensor to
  //     as dynamic during kernel Prepare.
  explicit SimpleDelegate(const std::vector<int>& nodes,
                          int64_t delegate_flags = kTfLiteDelegateFlagsNone,
                          bool fail_node_prepare = false,
                          int min_ops_per_subset = 0,
                          bool fail_node_invoke = false,
                          bool automatic_shape_propagation = false,
                          bool custom_op = true,
                          bool set_output_tensor_dynamic = false);

  static std::unique_ptr<SimpleDelegate> DelegateWithRuntimeShapePropagation(
      const std::vector<int>& nodes, int64_t delegate_flags,
      int min_ops_per_subset);

  static std::unique_ptr<SimpleDelegate> DelegateWithDynamicOutput(
      const std::vector<int>& nodes);

  TfLiteRegistration FakeFusedRegistration();

  TfLiteDelegate* get_tf_lite_delegate() { return &delegate_; }

  int min_ops_per_subset() { return min_ops_per_subset_; }

 private:
  std::vector<int> nodes_;
  TfLiteDelegate delegate_;
  bool fail_delegate_node_prepare_ = false;
  int min_ops_per_subset_ = 0;
  bool fail_delegate_node_invoke_ = false;
  bool automatic_shape_propagation_ = false;
  bool custom_op_ = true;
  bool set_output_tensor_dynamic_ = false;
};

// Base class for single/multiple delegate tests.
// Friend of Interpreter to access private methods.
class TestDelegation {
 public:
  virtual ~TestDelegation() = default;

  // Returns an empty interpreter that uses the same default delegates that are
  // normally enabled by default.
  static std::unique_ptr<impl::Interpreter>
  NewInterpreterWithDefaultDelegates() {
    auto interpreter = std::make_unique<impl::Interpreter>();
    interpreter->lazy_delegate_providers_ =
        tflite::ops::builtin::BuiltinOpResolver().GetDelegateCreators();
    return interpreter;
  }

 protected:
  TfLiteStatus RemoveAllDelegates() {
    return interpreter_->RemoveAllDelegates();
  }
  void SetMetadata(const std::map<std::string, std::string>& metadata) {
    interpreter_->SetMetadata(metadata);
  }

  virtual void SetUpSubgraph(Subgraph* subgraph);
  void AddSubgraphs(int subgraphs_to_add,
                    int* first_new_subgraph_index = nullptr);

  std::unique_ptr<impl::Interpreter> interpreter_;
};

// Tests scenarios involving a single delegate.
class TestDelegate : public TestDelegation, public ::testing::Test {
 protected:
  void SetUp() override;

  void TearDown() override;

  TfLiteBufferHandle last_allocated_handle_ = kTfLiteNullBufferHandle;

  TfLiteBufferHandle AllocateBufferHandle() { return ++last_allocated_handle_; }

  std::unique_ptr<SimpleDelegate> delegate_, delegate2_;
};

// Tests scenarios involving a single delegate and control edges.
// Subgraph 0 has the form
//
//         /---OP2---\
//        /           \
// >---OP0             OP3--->
//        \           /
//         \---OP1---/
//
// Delegating OP0, OP2 will generate an execution graph with a "super-node"
// {OP0->OP2}, which can be disabled by adding (in metadata) a control edge
// between OP1 and OP2:
//
//         /->-OP2---\
//        /     ^     \
// >---OP0      ^      OP3--->
//        \     ^     /
//         \---OP1---/
//
class TestDelegateWithControlEdges : public TestDelegate {
 protected:
  void SetUpSubgraph(Subgraph* subgraph) override;
};

// Tests scenarios involving two delegates, parametrized by the first & second
// delegate's flags.
class TestTwoDelegates
    : public TestDelegation,
      public ::testing::TestWithParam<
          std::pair<TfLiteDelegateFlags, TfLiteDelegateFlags>> {
 protected:
  void SetUp() override;

  void TearDown() override;

  std::unique_ptr<SimpleDelegate> delegate_, delegate2_;
};

// Tests delegate functionality related to FP16 graphs.
// Model architecture:
// 1->DEQ->2   4->DEQ->5   7->DEQ->8   10->DEQ->11
//         |           |           |            |
// 0----->ADD->3----->ADD->6----->MUL->9------>ADD-->12
// Input: 0, Output:12.
// All constants are 2, so the function is: (x + 2 + 2) * 2 + 2 = 2x + 10
//
// Delegate only supports ADD, so can have up to two delegated partitions.
// TODO(b/156707497): Add more cases here once we have landed CPU kernels
// supporting FP16.
class TestFP16Delegation : public ::testing::TestWithParam<int> {
 protected:
  void SetUp() override;

  void VerifyInvoke();

  void TearDown() override { interpreter_.reset(); }

 protected:
  class FP16Delegate {
   public:
    // Uses FP16GraphPartitionHelper to accept ADD nodes with fp16 input.
    explicit FP16Delegate(int num_delegated_subsets,
                          bool fail_node_prepare = false,
                          bool fail_node_invoke = false);

    TfLiteRegistration FakeFusedRegistration();

    TfLiteDelegate* get_tf_lite_delegate() { return &delegate_; }

    int num_delegated_subsets() { return num_delegated_subsets_; }

   private:
    TfLiteDelegate delegate_;
    int num_delegated_subsets_;
    bool fail_delegate_node_prepare_ = false;
    bool fail_delegate_node_invoke_ = false;
  };

  std::unique_ptr<impl::Interpreter> interpreter_;
  std::unique_ptr<FP16Delegate> delegate_;
  Eigen::half float16_const_;
};

}  // namespace test_utils
}  // namespace delegates
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_DELEGATE_TEST_UTIL_H_
