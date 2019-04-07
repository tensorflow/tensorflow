/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_FRAMEWORK_FUNCTION_TESTLIB_H_
#define TENSORFLOW_CORE_FRAMEWORK_FUNCTION_TESTLIB_H_

#include <string>

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace test {
namespace function {

// A helper class to make AttrSlice from initializer lists
class Attrs {
 public:
  Attrs(const std::initializer_list<  // NOLINT(runtime/explicit)
        std::pair<string, FunctionDefHelper::AttrValueWrapper>>& attrs) {
    for (const auto& aval : attrs) {
      map_.insert({aval.first, aval.second.proto});
    }
  }

  operator AttrSlice() { return AttrSlice(&map_); }  // NOLINT(runtime/explicit)

 private:
  AttrValueMap map_;
};

// Helper to construct a NodeDef.
NodeDef NDef(
    StringPiece name, StringPiece op, gtl::ArraySlice<string> inputs,
    gtl::ArraySlice<std::pair<string, FunctionDefHelper::AttrValueWrapper>>
        attrs = {},
    const string& device = "");

// Helper to construct a GraphDef proto.
GraphDef GDef(gtl::ArraySlice<NodeDef> nodes,
              gtl::ArraySlice<FunctionDef> funcs = {});

// For testing convenience, we provide a few simple functions that can
// be easily executed and tested.

// x:T -> x * 2.
FunctionDef XTimesTwo();

// x:T -> cpu(x * 2) + cpu(x * 3).
FunctionDef TwoDeviceTimesFive();

// x:T -> cpu(x * 2), gpu(x * 3).
FunctionDef TwoDeviceMult();

// cpu(x):T, gpu(y):T -> cpu(x * 2), gpu(y * 3).
FunctionDef TwoDeviceInputOutput();

// Function taking a list of Tensors as input.
FunctionDef FuncWithListInput();

// Function returning a list of Tensors as output.
FunctionDef FuncWithListOutput();

// x:T -> x + x.
FunctionDef XAddX();

// x:T -> x * 2, where x is int32.
FunctionDef XTimesTwoInt32();

// x:T -> (x * 2) * 2.
FunctionDef XTimesFour();

// x:T -> ((x * 2) * 2) * 2.
FunctionDef XTimes16();

// w:T, x:T, b:T -> MatMul(w, x) + b
FunctionDef WXPlusB();

// x:T -> x:T, T is a type which we automatically converts to a bool.
FunctionDef NonZero();

// x: T -> bool.
FunctionDef IsZero();

// x: T -> int64
FunctionDef RandomUniform();

// x:T, y:T -> y:T, x:T
FunctionDef Swap();

// x:T, y:T -> y:T, x:T, the body has no nodes.
FunctionDef EmptyBodySwap();

// x:float, y:resource -> y:resource, 2*x:float.
FunctionDef ResourceOutput();

// x:resource -> y:float.
FunctionDef ReadResourceVariable();

// Contains malformed control flow which can't be run by the executor.
FunctionDef InvalidControlFlow();

// x:T -> x <= N.
FunctionDef LessThanOrEqualToN(int64 N);

// x:T, y:T -> x+1, x*y
FunctionDef XPlusOneXTimesY();

// x:T, y:T -> x <= N
FunctionDef XYXLessThanOrEqualToN(int64 N);

// x:T -> y: TensorSliceDatasetOp::Dataset
FunctionDef MakeTensorSliceDataset();

void FunctionTestSchedClosure(std::function<void()> fn);

}  // end namespace function
}  // end namespace test
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_FUNCTION_TESTLIB_H_
