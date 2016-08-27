/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#ifndef THIRD_PARTY_TENSORFLOW_CORE_FRAMEWORK_SHAPE_INFERENCE_TESTUTIL_H_
#define THIRD_PARTY_TENSORFLOW_CORE_FRAMEWORK_SHAPE_INFERENCE_TESTUTIL_H_

#include <vector>
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/types.h"

// Contains utilities for writing tests for shape inference functions.

namespace tensorflow {

class NodeDef;
class Tensor;

struct ShapeInferenceTestOp {
  explicit ShapeInferenceTestOp(StringPiece name) : name(name.ToString()) {}
  string name;
  NodeDef node_def;
  std::vector<const Tensor*> input_tensors;
};

namespace shape_inference {

class ShapeInferenceTestutil {
 public:
  // Run shape inference for <op.name>, given inputs specified by <ins>
  // and returns an error if the inferred shape does not match expected_outs.
  //
  // <ins> is a semicolon separated list of shapes. Each shape is formatted
  // according to the formatting per
  // shape_inference::InferenceContext::InferenceContext.
  //
  // <expected_outs> is a semicolon separated list of shapes. Each shape is
  // formatted as one of:
  // * ? - an unknown shape, but not matching an input shape
  // * in0|in2|... - output shape must be the same as one of these input shapes.
  // * [1,?,d0_0|d0_1] - output shape is of known rank, with comma-separated
  //      dimension values.
  //      Each dimension value is one of:
  //      * a constant, which means that constant not equal to a specific input
  //      * ?, which means an unknown dim size not equal to a specific input
  //      * d0_0|d1_2, indicating that the dim size must be equal to one of
  //            the given input dimensions; the first number is the input # and
  //            the second is which dimension in that input it corresponds to.
  // <expected_outs> can be "e"; this is used to indicate that shape inference
  // should have failed.
  static Status InferShapes(ShapeInferenceTestOp op, const string& ins,
                            const string& expected_outs);

 private:
  ShapeInferenceTestutil() {}
};

}  // namespace shape_inference

#define INFER_OK(op, i, o)                                                    \
  EXPECT_EQ(                                                                  \
      "", ::tensorflow::shape_inference::ShapeInferenceTestutil::InferShapes( \
              op, i, o)                                                       \
              .error_message())
#define INFER_ERROR(error_substring, op, i)                                 \
  {                                                                         \
    string error_message =                                                  \
        ::tensorflow::shape_inference::ShapeInferenceTestutil::InferShapes( \
            op, i, "e")                                                     \
            .error_message();                                               \
    const string& substring = error_substring;                              \
    EXPECT_NE("", error_message);                                           \
    EXPECT_TRUE(StringPiece(error_message).contains(substring))             \
        << "Expected to see '" << substring << "' in '" << error_message    \
        << "'";                                                             \
  }

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_FRAMEWORK_SHAPE_INFERENCE_TESTUTIL_H_
