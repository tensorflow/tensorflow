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

#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ragged_tensor_variant.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

#ifndef TENSORFLOW_CORE_KERNELS_RAGGED_TENSOR_TO_VARIANT_OP_TEST_H_
#define TENSORFLOW_CORE_KERNELS_RAGGED_TENSOR_TO_VARIANT_OP_TEST_H_

namespace tensorflow {

class RaggedTensorToVariantKernelTest : public ::tensorflow::OpsTestBase {
 protected:
  // Builds the tensorflow test graph for the RaggedTensorToVariant op, and
  // populates the `splits` input with the given values.
  template <typename VALUE_TYPE, typename SPLIT_TYPE>
  void BuildEncodeRaggedTensorGraph(
      const std::vector<std::vector<SPLIT_TYPE>>& ragged_splits,
      const TensorShape& ragged_values_shape,
      const std::vector<VALUE_TYPE>& ragged_values, const bool batched) {
    const auto values_dtype = DataTypeToEnum<VALUE_TYPE>::v();
    const auto splits_dtype = DataTypeToEnum<SPLIT_TYPE>::v();
    int64_t num_splits = ragged_splits.size();
    TF_ASSERT_OK(
        NodeDefBuilder("tested_op", "RaggedTensorToVariant")
            .Input(FakeInput(num_splits, splits_dtype))  // ragged_splits
            .Input(FakeInput(values_dtype))              // ragged_values
            .Attr("RAGGED_RANK", num_splits)
            .Attr("Tvalues", values_dtype)
            .Attr("Tsplits", splits_dtype)
            .Attr("batched_input", batched)
            .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
    for (const auto& splits : ragged_splits) {
      int64_t splits_size = splits.size();
      AddInputFromArray<SPLIT_TYPE>(TensorShape({splits_size}), splits);
    }
    AddInputFromArray<VALUE_TYPE>(ragged_values_shape, ragged_values);
  }

  template <typename VALUE_TYPE, typename SPLIT_TYPE>
  void BuildEncodeRaggedTensorGraph(
      const std::vector<std::vector<SPLIT_TYPE>>& ragged_splits,
      const TensorShape& ragged_values_shape, const VALUE_TYPE& ragged_values,
      const bool batched) {
    const auto values_dtype = DataTypeToEnum<VALUE_TYPE>::v();
    const auto splits_dtype = DataTypeToEnum<SPLIT_TYPE>::v();
    int64_t num_splits = ragged_splits.size();
    TF_ASSERT_OK(
        NodeDefBuilder("tested_op", "RaggedTensorToVariant")
            .Input(FakeInput(num_splits, splits_dtype))  // ragged_splits
            .Input(FakeInput(values_dtype))              // ragged_values
            .Attr("RAGGED_RANK", num_splits)
            .Attr("Tvalues", values_dtype)
            .Attr("Tsplits", splits_dtype)
            .Attr("batched_input", batched)
            .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
    for (const auto& splits : ragged_splits) {
      int64_t splits_size = splits.size();
      AddInputFromArray<SPLIT_TYPE>(TensorShape({splits_size}), splits);
    }
    AddInput<VALUE_TYPE>(ragged_values_shape,
                         [&ragged_values](int i) { return ragged_values; });
  }

  template <typename VALUE_TYPE, typename SPLIT_TYPE>
  RaggedTensorVariant CreateVariantFromRagged(
      const std::vector<std::vector<SPLIT_TYPE>>& ragged_splits,
      const TensorShape& ragged_values_shape,
      const std::vector<VALUE_TYPE>& ragged_values) {
    RaggedTensorVariant encoded;
    for (auto ragged_split : ragged_splits) {
      int splits_size = ragged_split.size();
      Tensor splits(DataTypeToEnum<SPLIT_TYPE>::v(),
                    TensorShape({splits_size}));
      test::FillValues<SPLIT_TYPE>(&splits, ragged_split);
      encoded.append_splits(splits);
    }
    Tensor values(DataTypeToEnum<VALUE_TYPE>::v(), ragged_values_shape);
    test::FillValues<VALUE_TYPE>(&values, ragged_values);
    encoded.set_values(values);
    return encoded;
  }

  template <typename VALUE_TYPE, typename SPLIT_TYPE>
  RaggedTensorVariant CreateVariantFromRagged(
      const std::vector<std::vector<SPLIT_TYPE>>& ragged_splits,
      const std::vector<VALUE_TYPE>& ragged_values) {
    int num_values = ragged_values.size();
    return CreateVariantFromRagged(ragged_splits, {num_values}, ragged_values);
  }

  template <typename VALUE_TYPE, typename SPLIT_TYPE>
  void ExpectRaggedTensorVariantEqual(const RaggedTensorVariant& expected,
                                      const RaggedTensorVariant& actual) {
    test::ExpectTensorEqual<VALUE_TYPE>(actual.values(), expected.values());
    EXPECT_EQ(actual.ragged_rank(), expected.ragged_rank());
    for (int i = 0; i < actual.ragged_rank(); ++i) {
      test::ExpectTensorEqual<SPLIT_TYPE>(actual.splits(i), expected.splits(i));
    }
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_RAGGED_TENSOR_TO_VARIANT_OP_TEST_H_
