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

  Attrs(
      const std::vector<std::pair<string, FunctionDefHelper::AttrValueWrapper>>&
          attrs) {
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
    StringPiece name, StringPiece op, absl::Span<const string> inputs,
    absl::Span<const std::pair<string, FunctionDefHelper::AttrValueWrapper>>
        attrs = {},
    const string& device = "");

// Helper to construct a GraphDef proto.
GraphDef GDef(absl::Span<const NodeDef> nodes,
              absl::Span<const FunctionDef> funcs = {});

// For testing convenience, we provide a few simple functions that can
// be easily executed and tested.

// x: T -> x * 2.
FunctionDef XTimesTwo();
// Same as `XTimesTwo` above, but with the `x` input as a control dependency.
FunctionDef XTimesTwoWithControlInput();
// Same as `XTimesTwo` above, but with a `dummy` control output node.
FunctionDef XTimesTwoWithControlOutput();
// Same as `XTimesTwo` above, but with a dangling `FloorDiv` node.
FunctionDef XTimesTwoWithDanglingFloorDivNode();

// x: T -> cpu(x * 2) + cpu(x * 3).
FunctionDef TwoDeviceTimesFive();

// x: T -> cpu(x * 2), gpu(x * 3).
FunctionDef TwoDeviceMult();

// cpu(x): T, gpu(y): T -> cpu(x * 2), gpu(y * 3).
FunctionDef TwoDeviceInputOutput();

// Function taking a list of Tensors as input.
FunctionDef FuncWithListInput();

// Function returning a list of Tensors as output.
FunctionDef FuncWithListOutput();

// x: T -> x + x.
FunctionDef XAddX();

// x: T, y: T -> x + y.
FunctionDef XAddY();

// x: T -> x * 2, where x is int32.
FunctionDef XTimesTwoInt32();

// x: T -> (x * 2) * 2.
FunctionDef XTimesFour();

// x: T -> (x * 2) * 2, where x is int32
FunctionDef XTimesFourInt32();

// x: T -> ((x * 2) * 2) * 2.
FunctionDef XTimes16();

// w: T, x: T, b: T -> MatMul(w, x) + b
FunctionDef WXPlusB();

// x: T -> x: T, T is a type which we automatically converts to a bool.
FunctionDef NonZero();

// x: T -> bool.
FunctionDef IsZero();

// x: T -> int64
FunctionDef RandomUniform();

// x: T, y:T  -> y: T, x: T
FunctionDef Swap();

// x: T, y: T -> y: T, x: T, the body has no nodes.
FunctionDef EmptyBodySwap();

// x: float, y: resource -> y: resource, 2*x: float.
FunctionDef ResourceOutput();

// x: resource -> x: resource
FunctionDef ResourceIdentity();

// x: resource -> y: float.
FunctionDef ReadResourceVariable();

// Contains simple control flow returning the input via an Enter op.
FunctionDef ControlFlow();

// Contains malformed control flow which can't be run by the executor.
FunctionDef InvalidControlFlow();

// x: T -> x <= N.
FunctionDef LessThanOrEqualToN(int64_t N);

// x: T, y: T -> x + 1, x * y
FunctionDef XPlusOneXTimesY();

// x: T, y: T -> x <= N
FunctionDef XYXLessThanOrEqualToN(int64_t N);

// x: T -> bool
FunctionDef RandomUniformLess();

// start: int64, stop: int64, step: int64 -> y: RangeDatasetOp::Dataset
FunctionDef MakeRangeDataset();

// input_dataset: variant, batch_size: int64, drop_remainder: bool
// -> y: BatchDatasetV2::Dataset
FunctionDef MakeBatchDataset();

// input_dataset: variant, other_arguments: Targuments, f: func,
// Targuments: list(type), output_types: list(type), output_shapes: list(shape),
// use_inter_op_parallelism: bool, preserve_cardinality: bool
// -> y: MapDatasetOp::Dataset
FunctionDef MakeMapDataset(bool has_other_args);

// input_dataset: variant, count: int64 -> y: TakeDataset::Dataset
FunctionDef MakeTakeDataset();

// x: T -> y: TensorSliceDatasetOp::Dataset
FunctionDef MakeTensorSliceDataset();

// x: T -> y: T, idx: out_idx
FunctionDef Unique();

void FunctionTestSchedClosure(std::function<void()> fn);

}  // end namespace function
}  // end namespace test
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_FUNCTION_TESTLIB_H_
