#ifndef TENSORFLOW_FRAMEWORK_FUNCTION_TESTLIB_H_
#define TENSORFLOW_FRAMEWORK_FUNCTION_TESTLIB_H_

#include <string>

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/port.h"

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
