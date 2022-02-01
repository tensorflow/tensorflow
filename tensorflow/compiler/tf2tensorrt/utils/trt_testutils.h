/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
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

namespace tensorflow {
namespace tensorrt {
namespace convert {
// Creates a node with the given op, inputs, and attributes.
NodeDef MakeNodeDef(const std::string& name, const std::string& op,
                    const std::vector<std::string>& inputs,
                    const std::map<std::string, AttrValue> attrs = {});

// Creates a constant node with the given name and values arranged in the given
// shape.
template <typename T>
NodeDef MakeConstNodeDef(const std::string& name, const std::vector<T>& vals,
                         const TensorShape& shape) {
  Scope s = Scope::NewRootScope();
  Tensor t = test::AsTensor<T>(vals, shape);
  auto const_op = ops::Const(s.WithOpName(name), t);
  return const_op.node()->def();
}

// Creates a constant node with the given name and values, assuming a 1-D shape.
template <typename T>
NodeDef MakeConstNodeDef(const std::string& name, const std::vector<T>& vals) {
  TensorShape shape;
  const std::vector<int32> shape_dims = {static_cast<int32>(vals.size())};
  TF_EXPECT_OK(TensorShapeUtils::MakeShape(shape_dims, &shape));
  return MakeConstNodeDef(name, vals, shape);
}

// Creates an nvinfer1::Dims struct from the given vector.
nvinfer1::Dims CreateDims(const std::vector<int>& d);

// A gmock matcher that check that elements of a float vector match to a given
// tolerance.
::testing::Matcher<std::vector<float>> ArrayFloatNear(
    const std::vector<float>& values, float max_abs_error = 1e-5,
    bool nan_sensitive = false);

// nvinfer1::Dims gMock matchers

// matches nvinfer1::Dims to initializer list or vector of ints
// Example: EXPECT_THAT(my_dims, DimsAreArray({1, 2, 3}))
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

// nvinfer1::INetworkDefinition gMock matchers

// Checks that layer names are equal to initializer list or vector of strings.
// Example: EXPECT_THAT(my_network, LayerNamesAreArray({"conv1", "conv2"}))
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

// Checks layer names are all non-empty.
MATCHER(LayerNamesNonEmpty, "") {
  for (int i = 0; i < arg->getNbLayers(); ++i) {
    if (arg->getLayer(i)->getName() == nullptr) {
      return false;
    }
  }
  return true;
}

// TRT_ShapedWeights gMock matchers.

// Checks that the weight dimensions are values are equal to the given values.
// Example: EXPECT_THAT(my_weights,
//                      ShapedWeightsHasDimsAndValues({1, 2},{1.0f, 2.0f}))
MATCHER_P2(ShapedWeightsHasDimsAndValuesHelper, dims_vec, expected_values, "") {
  DimsAdapter dims(dims_vec);
  if (arg.Shape() != dims) {
    return false;
  }
  if (arg.count() != expected_values.size()) {
    return false;
  }
  using T = typename decltype(expected_values)::value_type;
  const T* actual_values = arg.template GetPointer<T>();
  for (int i = 0; i < expected_values.size(); ++i) {
    if (expected_values[i] != actual_values[i]) {
      return false;
    }
  }
  return true;
}

template <typename T>
using ShapedWeightsHasDimsAndValues =
    ShapedWeightsHasDimsAndValuesHelperMatcherP2<std::vector<int>,
                                                 std::vector<T>>;

// std::vector convenience utilities.

// Creates a new vector by casting all values of the given InCType vector to
// OutCType.
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

// Creates a new vector of the given size and fills it with an increasing
// sequence starting from the given start_value using std::iota.
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
