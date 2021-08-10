/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_UTILS_TEST_UTILS_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_UTILS_TEST_UTILS_H_

#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <map>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/tf2tensorrt/common/datavec.h"
#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/compiler/tf2tensorrt/convert/convert_nodes.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_engine_utils.h"
#include "tensorflow/core/framework/node_def.pb.h"  // NOLINT
#include "tensorflow/core/framework/tensor.pb.h"    // NOLINT
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "third_party/tensorrt/NvInfer.h"

namespace nvinfer1 {
// Stream printing functions for GTest.
// These must be in nvinfer1 namespace.

// Alias a useful type computation for nvinfer1::Dims types;
// legacy dims like nvinfer1::Dims2 inherit from nvinfer1::Dims.
template <typename T>
using enable_if_nvinfer_dims =
    std::enable_if<std::is_base_of<nvinfer1::Dims, T>::value, T>;

// Prints nvinfer1::Dims (and any sub-struct, for legacy Dim types)
// to ostream.
template <typename T, typename enable_if_nvinfer_dims<T>::type* = nullptr>
std::ostream& operator<<(std::ostream& os, const T& v) {
  os << "nvinfer1::Dims[";
  for (int i = 0; i < v.nbDims; i++) {
    os << (i > 0 ? ", " : "") << v.d[i] << "";
  }
  os << "]";
  return os;
}

// Print nvinfer1::INetworkDefinition* information to ostream
inline std::ostream& operator<<(std::ostream& os,
                                nvinfer1::INetworkDefinition* n) {
  auto print_layers = [](std::ostream& os, nvinfer1::INetworkDefinition* n) {
    for (int i = 0; i < n->getNbLayers(); i++) {
      os << " " << n->getLayer(i)->getName() << "\n";
    }
  };
  os << "nvinfer1::INetworkDefinition{\n";
  print_layers(os, n);
  os << "}";
  return os;
}

}  // namespace nvinfer1

// Matchers used in graph/node conversion testing we put under
// tensorrt::convert.
namespace tensorflow {
namespace tensorrt {
namespace convert {

// Node creation utilities

// helper to create node with given op, inputs, and attributes
NodeDef MakeNodeDef(const std::string& name, const std::string& op,
                    const std::vector<std::string>& inputs,
                    const std::map<std::string, AttrValue> attrs = {});

// create a constant node with given vector and shape as tensor
template <typename T>
NodeDef MakeConstNodeDef(const std::string& name, const std::vector<T>& vals,
                         const TensorShape& shape) {
  Scope s = Scope::NewRootScope();
  Tensor t = test::AsTensor<T>(vals, shape);
  auto const_op = ops::Const(s.WithOpName(name), t);
  return const_op.node()->def();
}

// constant node with 1d shape
template <typename T>
NodeDef MakeConstNodeDef(const std::string& name, const std::vector<T>& vals) {
  TensorShape shape;
  const std::vector<int32> shape_dims = {static_cast<int32>(vals.size())};
  TF_EXPECT_OK(TensorShapeUtils::MakeShape(shape_dims, &shape));
  return MakeConstNodeDef(name, vals, shape);
}

// nvinfer1:: type helpers

// Checks equality of two sets of dims
bool TrtDimsEquals(const nvinfer1::Dims& lhs, const nvinfer1::Dims& rhs);

// Creates an nvinfer1::Dims struct from the given vector.
nvinfer1::Dims CreateDims(const std::vector<int>& d);

// general GMock matchers
// A gmock matcher that check that elements of a float vector match to a given
// tolerance.
::testing::Matcher<std::vector<float>> ArrayFloatNear(
    const std::vector<float>& values, float max_abs_error = 1e-5,
    bool nan_sensitive = false);

// nvinfer1::Dims GMock matchers

// matches nvinfer1::Dims to initializer list or vector of ints
// "EXPECT_THAT(my_dims, DimsAreArray({1, 2, 3}))"
MATCHER_P(DimsAreArrayHelper, array_value,
          absl::StrFormat("%s [%s]", negation ? "are" : "are not",
                          ::testing::PrintToString(array_value))) {
  if (arg.nbDims != array_value.size()) return false;
  for (int i = 0; i < arg.nbDims; ++i) {
    if (arg.d[i] != array_value[i]) {
      return false;
    }
  }
  return true;
}
using DimsAreArray = DimsAreArrayHelperMatcherP<std::vector<int>>;

// nvinfer1::INetworkDefinition GMock matchers

// Check layer names are equal to initializer list or vector of strings
MATCHER_P(LayerNamesAreArrayHelper, array_value,
          absl::StrFormat("layer names %s [%s]", negation ? "are" : "are not",
                          ::testing::PrintToString(array_value))) {
  if (array_value.size() != arg->getNbLayers()) return false;
  for (int i = 0; i < arg->getNbLayers(); ++i) {
    if (arg->getLayer(i)->getName() == nullptr) {
      return false;
    }
  }
  return true;
}
using LayerNamesAreArray =
    LayerNamesAreArrayHelperMatcherP<std::vector<std::string>>;

// Check layer names in INetworkDefinition are all non-empty.
MATCHER(LayerNamesNonEmpty, "") {
  for (int i = 0; i < arg->getNbLayers(); ++i) {
    if (arg->getLayer(i)->getName() == nullptr) {
      return false;
    }
  }
  return true;
}

// GMock matchers for TRT_ShapedWeights
MATCHER_P2(ShapedWeightsHasDimsAndValues, dims, expected_values, "") {
  if (arg->shape_ != dims) {
    return false;
  }
  if (arg->count() != expected_values.size()) {
    return false;
  }
  using T = typename decltype(expected_values)::value_type;
  auto actual_values = reinterpret_cast<T*>(arg->GetValues());
  for (int i = 0; i < expected_values.size(); ++i) {
    if (expected_values[i] != actual_values[i]) {
      return false;
    }
  }
}

template <typename InCType, typename OutCType>
std::vector<OutCType> CastVector(
    const gtl::ArraySlice<InCType>& vals) {  // non-absl ok
  std::vector<OutCType> res(vals.size());
  std::transform(vals.begin(), vals.end(), res.begin(),
                 [](const InCType in_val) -> OutCType {
                   return static_cast<OutCType>(in_val);
                 });
  return res;
}

template <typename CType>
std::vector<CType> CreateVectorIota(int size, CType start_value = CType(0)) {
  std::vector<CType> res(size);
  std::iota(res.begin(), res.end(), start_value);
  return res;
}

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_UTILS_TEST_UTILS_H_
