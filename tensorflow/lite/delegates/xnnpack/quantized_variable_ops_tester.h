/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_VARIABLE_OPS_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_VARIABLE_OPS_TESTER_H_

#include <functional>
#include <memory>
#include <numeric>
#include <vector>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace xnnpack {

std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
NewXnnPackDelegateSupportingVariableOps();

class QuantizedVariableOpsTester {
 public:
  void TestAssignThenRead(TfLiteDelegate *delegate) const;

  void TestAssignTwiceThenRead(TfLiteDelegate *delegate) const;

  void TestAssignThenReadUsingAnotherVarHandle(TfLiteDelegate *delegate) const;

  void TestTwoVarHandlesAssignThenRead(TfLiteDelegate *delegate) const;

  void TestTwoSubgraphsReadAssign(TfLiteDelegate *delegate) const;

  void TestTwoSubgraphsReadAssignOneVarHandle(TfLiteDelegate *delegate) const;

  void TestTwoSubgraphsReadAssignOneVarHandle2(TfLiteDelegate *delegate) const;

  // Creates a model with this subgraph:
  //   (initial_input) ----\
  //                        \
  //   VAR_HANDLE ----------> AV
  //        \
  //         \---------------> RV  -> (output)
  std::vector<char> CreateModelAssignThenRead() const;

  // Creates a model with this subgraph:
  // (initial_input) -------
  //                        \
  // VAR_HANDLE ----------> AV
  //  \      \
  //   \      \---------------> RV  -> (output)
  //    \
  //     \-----AV
  //           /
  // (default_input)
  std::vector<char> CreateModelAssignTwiceThenRead() const;

  // Creates a model with this subgraph:
  //   (initial_input) ----\
  //                        \
  //   VAR_HANDLE ----------> AV
  //
  //   VAR_HANDLE ---------------> RV  -> (output)
  //   Second VAR_HANDLE is a different operator object but refers to the same
  //   variable by name.
  std::vector<char> CreateModelAssignThenReadUsingAnotherVarHandle() const;

  // Creates a model with this subgraph:
  // (default_input) -------
  //                        \
  // VAR_HANDLE ----------> AV
  //         \
  //          \---------------> RV  -> (output1)
  //
  // (default_input) ---------------
  //                                 \
  // VAR_HANDLE (different one) -----AV
  //         \
  //          \---------------> RV  -> (output2)
  std::vector<char> CreateModelTwoVarHandlesAssignThenRead() const;

  // Creates a model with two subgraphs.
  // primary subgraph:
  // CALL_ONCE (secondary subgraph)
  // VAR_HANDLE1 ---> RV ---> (output)
  // VAR_HANDLE2 ---> RV ---> (output)
  //
  // secondary subgraph:
  // VAR_HANDLE2   ----- AV
  // buffer1 -----------/
  // VAR_HANDLE1   ----- AV
  // buffer2 -----------/
  // The var handles are defined in different orders.
  std::vector<char> CreateModelTwoSubgraphsReadAssign() const;

  // Creates a model with two subgraphs.
  // primary subgraph has 1 var handle.
  // CALL_ONCE (secondary subgraph)
  // VAR_HANDLE1 ---> RV ---> (output)
  //
  // secondary subgraph has 2 varhandle:
  // VAR_HANDLE2   ----- AV
  // buffer1 -----------/
  // VAR_HANDLE1   ----- AV
  // buffer2 -----------/
  // The expected output is buffer2.
  // The var handles are defined in different orders.
  std::vector<char> CreateModelTwoSubgraphsReadAssignOneVarHandle() const;

  // Similar to CreateModelTwoSubgraphsReadAssignOneVarHandle but with the first
  // subgraph reading var handle 1 and flipping the order of var handles in the
  // second subgraph.
  // Creates a model with two subgraphs.
  // primary subgraph has 1 var handle.
  // CALL_ONCE (secondary subgraph)
  // VAR_HANDLE2 ---> RV ---> (output)
  //
  // secondary subgraph has 2 varhandle:
  // VAR_HANDLE1   ----- AV
  // buffer1 -----------/
  // VAR_HANDLE2   ----- AV
  // buffer2 -----------/
  // The expected output is buffer2.
  // The var handles are defined in different orders.
  std::vector<char> CreateModelTwoSubgraphsReadAssignOneVarHandle2() const;

  inline QuantizedVariableOpsTester &NumInputs(size_t num_inputs) {
    num_inputs_ = num_inputs;
    return *this;
  }

  inline size_t NumInputs() const { return num_inputs_; }

  inline QuantizedVariableOpsTester &NumOutputs(size_t num_outputs) {
    num_outputs_ = num_outputs;
    return *this;
  }

  inline size_t NumOutputs() const { return num_outputs_; }

  inline size_t NumSubgraphs() const { return num_subgraphs_; }

  inline QuantizedVariableOpsTester &NumSubgraphs(size_t num_subgraphs) {
    num_subgraphs_ = num_subgraphs;
    return *this;
  }

  const std::vector<int32_t> &Shape() const { return shape_; }

  const std::vector<int32_t> &ResourceShape() const { return resource_shape_; }

  size_t OutputSize() const {
    return std::accumulate(Shape().begin(), Shape().end(), 1,
                           std::multiplies<int32_t>());
  }

  size_t InputSize() const {
    return std::accumulate(Shape().begin(), Shape().cend(), 1,
                           std::multiplies<int32_t>());
  }

  inline QuantizedVariableOpsTester &ZeroPoint(int32_t zero_point) {
    zero_point_ = zero_point;
    return *this;
  }

  inline int32_t ZeroPoint() const { return zero_point_; }

  inline QuantizedVariableOpsTester &Scale(float scale) {
    scale_ = scale;
    return *this;
  }

  inline float Scale() const { return scale_; }

  inline QuantizedVariableOpsTester &Unsigned(bool is_unsigned) {
    unsigned_ = is_unsigned;
    return *this;
  }

  inline bool Unsigned() const { return unsigned_; }

 private:
  template <class T>
  void Test(TfLiteDelegate *delegate, const std::vector<char> &buffer) const;

  std::vector<int32_t> shape_ = {1, 2, 2, 3};
  std::vector<int32_t> resource_shape_ = {1};
  size_t num_inputs_ = 0;
  size_t num_outputs_ = 0;
  size_t num_subgraphs_ = 1;
  int32_t zero_point_ = 2;
  float scale_ = 0.75f;
  bool unsigned_ = false;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZED_VARIABLE_OPS_TESTER_H_
