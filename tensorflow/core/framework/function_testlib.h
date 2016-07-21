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

#ifndef TENSORFLOW_FRAMEWORK_FUNCTION_TESTLIB_H_
#define TENSORFLOW_FRAMEWORK_FUNCTION_TESTLIB_H_

#include <string>

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace test {
namespace function {

// Helper to construct a NodeDef.
NodeDef NDef(
    const string& name, const string& op, gtl::ArraySlice<string> inputs,
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

// x:T -> (x * 2) * 2.
FunctionDef XTimesFour();

// x:T -> ((x * 2) * 2) * 2.
FunctionDef XTimes16();

// w:T, x:T, b:T -> MatMul(w, x) + b
FunctionDef WXPlusB();

// x:T -> x:T, T is a type which we automatically converts to a bool.
FunctionDef NonZero();

// x:T, y:T -> y:T, x:T
FunctionDef Swap();

}  // end namespace function
}  // end namespace test
}  // end namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_FUNCTION_TESTLIB_H_
