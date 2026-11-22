/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

template <typename VALUE_TYPE>
struct ShapeAndValues {
  TensorShape shape;
  std::vector<VALUE_TYPE> values;
};

template <typename VALUE_TYPE>
ShapeAndValues<VALUE_TYPE> createVector(const std::vector<VALUE_TYPE>& values) {
  TensorShape shape({static_cast<int64_t>(values.size())});
  return {shape, values};
}

template <typename VALUE_TYPE>
ShapeAndValues<VALUE_TYPE> createScalar(const VALUE_TYPE& values) {
  TensorShape shape({});
  return {shape, {values}};
}

class RaggedTensorToTensorOpTest : public ::tensorflow::OpsTestBase {
 protected:
  void ClearTestInputs() { inputs_.clear(); }

  // Builds the tensorflow test graph for RaggedTensorToTensor.
  template <typename VALUE_TYPE, typename INDEX_TYPE>
  void BuildRaggedTensorToTensorGraph(
      const TensorShape& shape,
      const std::vector<std::string>& row_partition_types,
      const ShapeAndValues<VALUE_TYPE>& values,
      const ShapeAndValues<VALUE_TYPE>& default_value,
      const std::vector<ShapeAndValues<INDEX_TYPE>>& row_partition_tensors) {
    const auto& value_dtype = DataTypeToEnum<VALUE_TYPE>::v();
    const auto& index_dtype = DataTypeToEnum<INDEX_TYPE>::v();
    int num_row_partition_tensors = row_partition_tensors.size();
    TF_ASSERT_OK(
        NodeDefBuilder("tested_op", "RaggedTensorToTensor")
            .Attr("T", value_dtype)
            .Attr("Tindex", index_dtype)
            .Attr("num_row_partition_tensors", num_row_partition_tensors)
            .Attr("row_partition_types", row_partition_types)
            .Input(FakeInput(index_dtype))
            .Input(FakeInput(value_dtype))  // values
            .Input(FakeInput(value_dtype))  // default_value
            .Input(FakeInput(num_row_partition_tensors,
                             index_dtype))  // row_partition_tensors
            .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
    {
      std::vector<INDEX_TYPE> shape_as_vector;
      for (const auto& dim : shape.dim_sizes()) {
        shape_as_vector.push_back(dim);
      }
      ShapeAndValues<INDEX_TYPE> shape_as_tensor =
          createVector(shape_as_vector);
      AddInputFromArray<INDEX_TYPE>(shape_as_tensor.shape,
                                    shape_as_tensor.values);
    }
    AddInputFromArray<VALUE_TYPE>(values.shape, values.values);
    AddInputFromArray<VALUE_TYPE>(default_value.shape, default_value.values);

    for (const auto& row_partition_tensor : row_partition_tensors) {
      AddInputFromArray<INDEX_TYPE>(row_partition_tensor.shape,
                                    row_partition_tensor.values);
    }
  }
};

TEST_F(RaggedTensorToTensorOpTest, RaggedTensorToTensor) {
  // indices = [2, 1, 0, 3]
  // params = [[.1, .2, .3], [], [.4, .5, .6, .7], [.8, .9]]
  // params.shape = [4, None]
  BuildRaggedTensorToTensorGraph<float, int32_t>(
      TensorShape({4, 4}),                 // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS"},  // row_partition_types
      createVector<float>({.1, .2, .3, .4, .5, .6, .7, .8, .9}),  // values
      createScalar<float>(1.5),  // default_value
      {createScalar<int32_t>(4),
       createVector<int32_t>({0, 0, 0, 2, 2, 2, 2, 3, 3})}
      // row_partition_tensors
  );

  TF_ASSERT_OK(RunOpKernel());

  test::ExpectTensorNear<float>(
      *GetOutput(0),
      test::AsTensor<float>({.1, .2, .3, 1.5, 1.5, 1.5, 1.5, 1.5, .4, .5, .6,
                             .7, .8, .9, 1.5, 1.5},
                            TensorShape({4, 4})),
      0.01);
}

TEST_F(RaggedTensorToTensorOpTest, RaggedTensorToTensorRowSplits) {
  // indices = [2, 1, 0, 3]
  // params = [[.1, .2, .3], [], [.4, .5, .6, .7], [.8, .9]]
  BuildRaggedTensorToTensorGraph<float, int32_t>(
      TensorShape({4, 4}),  // shape
      {"ROW_SPLITS"},       // row_partition_types
      createVector<float>({.1, .2, .3, .4, .5, .6, .7, .8, .9}),  // values
      createScalar<float>(1.5),                 // default_value
      {createVector<int32_t>({0, 3, 3, 7, 9})}  // row_partition_tensors
  );

  TF_ASSERT_OK(RunOpKernel());

  test::ExpectTensorNear<float>(
      *GetOutput(0),
      test::AsTensor<float>({.1, .2, .3, 1.5, 1.5, 1.5, 1.5, 1.5, .4, .5, .6,
                             .7, .8, .9, 1.5, 1.5},
                            TensorShape({4, 4})),
      0.01);
}

TEST_F(RaggedTensorToTensorOpTest, RaggedTensorToTensor_3DParams) {
  // params = [
  //           [[]],
  //           [[.1, .2], [.3]],
  //           [],
  //           [[.4, .5], [.6, .7, .8]],
  //           [[.9]]
  //          ]
  BuildRaggedTensorToTensorGraph<float, int32_t>(
      TensorShape({5, 2, 3}),  // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS",
       "VALUE_ROWIDS"},  // row_partition_types
      createVector<float>({.1, .2, .3, .4, .5, .6, .7, .8, .9}),  // values
      createScalar<float>(1.5),  // default_value
      {
          createScalar<int32_t>(5),
          createVector<int32_t>({0, 1, 1, 3, 3, 4}),
          createVector<int32_t>({1, 1, 2, 3, 3, 4, 4, 4, 5}),
      }  // row_partition_tensors
  );
  TF_ASSERT_OK(RunOpKernel());

  // Expected = [
  //              [[1.5, 1.5, 1.5], [1.5, 1.5, 1.5]],
  //              [[.1, .2, 1.5], [.3, 1.5, 1.5]],
  //              [[1.5, 1.5, 1.5], [1.5, 1.5, 1.5]],
  //              [[.4, .5, 1.5], [.6, .7, .8]],
  //              [[.9, 1.5, 1.5], [1.5, 1.5, 1.5]]
  //            ]
  test::ExpectTensorNear<float>(
      *GetOutput(0),
      test::AsTensor<float>({1.5, 1.5, 1.5, 1.5, 1.5, 1.5, .1,  .2,  1.5, .3,
                             1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, .4,  .5,
                             1.5, .6,  .7,  .8,  .9,  1.5, 1.5, 1.5, 1.5, 1.5},
                            TensorShape({5, 2, 3})),
      0.1);
}

TEST_F(RaggedTensorToTensorOpTest, RaggedTensorToTensor_3DParamsRowSplits) {
  // params = [
  //           [[]],
  //           [[.1, .2], [.3]],
  //           [],
  //           [[.4, .5], [.6, .7, .8]],
  //           [[.9]]
  //          ]
  BuildRaggedTensorToTensorGraph<float, int32_t>(
      TensorShape({5, 2, 3}),        // shape
      {"ROW_SPLITS", "ROW_SPLITS"},  // row_partition_types
      createVector<float>({.1, .2, .3, .4, .5, .6, .7, .8, .9}),  // values
      createScalar<float>(1.5),  // default_value
      {
          createVector<int32_t>({0, 1, 3, 3, 5, 6}),
          createVector<int32_t>({0, 0, 2, 3, 5, 8, 9}),
      }  // row_partition_tensors
  );
  TF_ASSERT_OK(RunOpKernel());

  // Expected = [
  //              [[1.5, 1.5, 1.5], [1.5, 1.5, 1.5]],
  //              [[.1, .2, 1.5], [.3, 1.5, 1.5]],
  //              [[1.5, 1.5, 1.5], [1.5, 1.5, 1.5]],
  //              [[.4, .5, 1.5], [.6, .7, .8]],
  //              [[.9, 1.5, 1.5], [1.5, 1.5, 1.5]]
  //            ]
  test::ExpectTensorNear<float>(
      *GetOutput(0),
      test::AsTensor<float>({1.5, 1.5, 1.5, 1.5, 1.5, 1.5, .1,  .2,  1.5, .3,
                             1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, .4,  .5,
                             1.5, .6,  .7,  .8,  .9,  1.5, 1.5, 1.5, 1.5, 1.5},
                            TensorShape({5, 2, 3})),
      0.1);
}

// test_three_dimensional_ragged fails, want to try it at a lower level.
TEST_F(RaggedTensorToTensorOpTest, RaggedTensorToTensor_3DParamsRowSplits2) {
  // params = [
  //           [[0, 1, 2], []],
  //           [],
  //           [[3]]
  //          ]
  BuildRaggedTensorToTensorGraph<int64_t, int64_t>(
      TensorShape({3, 2, 3}),               // shape
      {"ROW_SPLITS", "ROW_SPLITS"},         // row_partition_types
      createVector<int64_t>({0, 1, 2, 3}),  // values
      createScalar<int64_t>(5),             // default_value
      {
          createVector<int64_t>({0, 2, 2, 3}),
          createVector<int64_t>({0, 3, 3, 4}),
      }  // row_partition_tensors
  );
  TF_ASSERT_OK(RunOpKernel());

  // Expected = [
  //              [[0, 1, 2], [5, 5, 5]],
  //              [[5, 5, 5], [5, 5, 5]],
  //              [[3, 5, 5], [5, 5, 5]]
  //            ]
  test::ExpectTensorEqual<int64_t>(
      *GetOutput(0), test::AsTensor<int64_t>(
                         {0, 1, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 5, 5, 5, 5, 5},
                         TensorShape({3, 2, 3})));
}

TEST_F(RaggedTensorToTensorOpTest, RaggedTensorToTensor_4DParams) {
  // Input:    [[],
  //            [
  //             [[1, 2], [3, 4], [5, 6]],
  //             [[7, 8]]
  //            ],
  //            [[]],
  //            []
  // ]
  // params.shape = [3, 2, 3, 2]
  BuildRaggedTensorToTensorGraph<int32_t, int32_t>(
      TensorShape({4, 2, 3, 2}),  // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS", "VALUE_ROWIDS",
       "VALUE_ROWIDS"},                                 // row_partition_types
      createVector<int32_t>({1, 2, 3, 4, 5, 6, 7, 8}),  // values
      createScalar<int32_t>(15),                        // default_value
      {createScalar<int32_t>(5), createVector<int32_t>({0, 1, 1}),
       createVector<int32_t>({1, 1, 1, 2}),
       createVector<int32_t>({0, 0, 1, 1, 2, 2, 3, 3})}
      // row_partition_tensors
  );

  TF_ASSERT_OK(RunOpKernel());
  // params = [
  //           [
  //             [[15,15],[15,15],[15,15]],
  //             [[15,15],[15,15],[15,15]],
  //           ],
  //           [
  //             [[1, 2], [3, 4], [5, 6]],
  //             [[7, 8], [15, 15], [15,15]],
  //           ],
  //             [[15,15],[15,15],[15,15]],
  //             [[15,15],[15,15],[15,15]],
  //           ],
  //             [[15,15],[15,15],[15,15]],
  //             [[15,15],[15,15],[15,15]],
  //           ]
  // params.shape = [3, 2, 3, 2]
  test::ExpectTensorEqual<int32_t>(
      *GetOutput(0),
      test::AsTensor<int32_t>(
          {15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 1,  2,  3,  4,
           5,  6,  7,  8,  15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
           15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15},
          TensorShape({4, 2, 3, 2})));
}

TEST_F(RaggedTensorToTensorOpTest, RaggedTensorToTensor_4DParamsRowSplit) {
  // Input:    [[],
  //            [
  //             [[1, 2], [3, 4], [5, 6]],
  //             [[7, 8]]
  //            ],
  //            [[]],
  //            []
  // ]
  // params.shape = [3, 2, 3, 2]
  BuildRaggedTensorToTensorGraph<int32_t, int32_t>(
      TensorShape({4, 2, 3, 2}),  // shape
      {"ROW_SPLITS", "ROW_SPLITS", "ROW_SPLITS"},
      // row_partition_types
      createVector<int32_t>({1, 2, 3, 4, 5, 6, 7, 8}),  // values
      createScalar<int32_t>(15),                        // default_value
      {createVector<int32_t>({0, 1, 3}), createVector<int32_t>({0, 0, 3, 4}),
       createVector<int32_t>({0, 2, 4, 6, 8})}  // row_partition_tensors
  );

  TF_ASSERT_OK(RunOpKernel());
  // params = [
  //           [
  //             [[15,15],[15,15],[15,15]],
  //             [[15,15],[15,15],[15,15]],
  //           ],
  //           [
  //             [[1, 2], [3, 4], [5, 6]],
  //             [[7, 8], [15, 15], [15,15]],
  //           ],
  //             [[15,15],[15,15],[15,15]],
  //             [[15,15],[15,15],[15,15]],
  //           ],
  //             [[15,15],[15,15],[15,15]],
  //             [[15,15],[15,15],[15,15]],
  //           ]
  // params.shape = [3, 2, 3, 2]
  test::ExpectTensorEqual<int32_t>(
      *GetOutput(0),
      test::AsTensor<int32_t>(
          {15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 1,  2,  3,  4,
           5,  6,  7,  8,  15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
           15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15},
          TensorShape({4, 2, 3, 2})));
}

TEST_F(RaggedTensorToTensorOpTest, RaggedTensorToTensorContractExpanded) {
  // params = [[.1, .2, .3], [], [.4, .5, .6, .7], [.8, .9]]
  BuildRaggedTensorToTensorGraph<float, int32_t>(
      TensorShape({3, 5}),                 // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS"},  // row_partition_types
      createVector<float>({.1, .2, .3, .4, .5, .6, .7, .8, .9}),  // values
      createScalar<float>(1.5),  // default_value
      {createScalar<int32_t>(4),
       createVector<int32_t>({0, 0, 0, 2, 2, 2, 2, 3, 3})}
      // row_partition_tensors
  );

  TF_ASSERT_OK(RunOpKernel());

  test::ExpectTensorNear<float>(
      *GetOutput(0),
      test::AsTensor<float>({.1, .2, .3, 1.5, 1.5,     //
                             1.5, 1.5, 1.5, 1.5, 1.5,  //
                             .4, .5, .6, .7, 1.5},     //
                            TensorShape({3, 5})),
      0.01);
}

// Adds a dense dimension.
TEST_F(RaggedTensorToTensorOpTest, RaggedTensorToTensorContractExpandedDense) {
  // params = [[.1, .2, .3], [], [.4, .5, .6, .7], [.8, .9]]
  BuildRaggedTensorToTensorGraph<float, int32_t>(
      TensorShape({3, 5, 2}),              // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS"},  // row_partition_types
      ShapeAndValues<float>{TensorShape({9, 2}),
                            {.1, 1.1, .2, 1.2, .3, 1.3, .4, 1.4, .5, 1.5, .6,
                             1.6, .7, 1.7, .8, 1.8, .9, 1.9}},  // values
      createScalar<float>(1.5),                                 // default_value
      {createScalar<int32_t>(4),
       createVector<int32_t>({0, 0, 0, 2, 2, 2, 2, 3, 3})}
      // row_partition_tensors
  );

  TF_ASSERT_OK(RunOpKernel());

  test::ExpectTensorNear<float>(
      *GetOutput(0),
      test::AsTensor<float>(
          {.1,  1.1, .2,  1.2, .3,  1.3, 1.5, 1.5, 1.5, 1.5,   //
           1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,   //
           .4,  1.4, .5,  1.5, .6,  1.6, .7,  1.7, 1.5, 1.5},  //
          TensorShape({3, 5, 2})),
      0.01);
}

TEST_F(RaggedTensorToTensorOpTest, RaggedTensorToTensorConstrained) {
  // params = [[.1, .2, .3],
  //           [],
  //           [.4, .5, .6, .7],
  //           [.8, .9]]
  // constrained to (3, 3)
  BuildRaggedTensorToTensorGraph<float, int32_t>(
      TensorShape({3, 3}),                 // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS"},  // row_partition_types
      createVector<float>({.1, .2, .3, .4, .5, .6, .7, .8, .9}),  // values
      createScalar<float>(1.5),  // default_value
      {createScalar<int32_t>(4),
       createVector<int32_t>({0, 0, 0, 2, 2, 2, 2, 3, 3})}
      // row_partition_tensors
  );

  TF_ASSERT_OK(RunOpKernel());

  test::ExpectTensorNear<float>(*GetOutput(0),
                                test::AsTensor<float>(
                                    {
                                        //
                                        .1, .2, .3,     //
                                        1.5, 1.5, 1.5,  //
                                        .4, .5, .6      //
                                    },
                                    TensorShape({3, 3})),
                                0.01);
}

TEST_F(RaggedTensorToTensorOpTest, RaggedTensorToTensor_3DParamsConstrained) {
  // params = [
  //           [[]],
  //           [[.1, .2], [.3]],
  //           [],
  //           [[.4, .5], [.6, .7, .8]],
  //           [[.9]]
  //          ]
  // params.shape = [5, None, None]
  BuildRaggedTensorToTensorGraph<float, int32_t>(
      TensorShape({4, 1, 2}),  // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS",
       "VALUE_ROWIDS"},  // row_partition_types
      createVector<float>({.1, .2, .3, .4, .5, .6, .7, .8, .9}),  // values
      createScalar<float>(1.5),  // default_value
      {
          createScalar<int32_t>(5),
          createVector<int32_t>({0, 1, 1, 3, 3, 4}),
          createVector<int32_t>({1, 1, 2, 3, 3, 4, 4, 4, 5}),
      }  // row_partition_tensors
  );
  TF_ASSERT_OK(RunOpKernel());

  // Expected = [
  //              [[1.5, 1.5]],
  //              [[.1, .2]],
  //              [[1.5, 1.5]],
  //              [[.4, .5]],
  //            ]
  test::ExpectTensorNear<float>(
      *GetOutput(0),
      test::AsTensor<float>({1.5, 1.5, .1, .2, 1.5, 1.5, .4, .5},
                            TensorShape({4, 1, 2})),
      0.01);
}

// Seg fault but removing this does not make the problem go away.
// This tests is labeled as flaky. Removing it to find out.
TEST_F(RaggedTensorToTensorOpTest, RaggedTensorToTensor_4DParamsConstrained) {
  // Input:    [[],
  //            [
  //             [[1, 2], [3, 4], [5, 6]],
  //             [[7, 8]]
  //            ],
  //            [[]],
  //            []
  // ]
  // params.shape = [3, 2, 3, 2]
  BuildRaggedTensorToTensorGraph<int32_t, int32_t>(
      TensorShape({2, 2, 2, 2}),  // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS", "VALUE_ROWIDS",
       "VALUE_ROWIDS"},                                 // row_partition_types
      createVector<int32_t>({1, 2, 3, 4, 5, 6, 7, 8}),  // values
      createScalar<int32_t>(15),                        // default_value
      {createScalar<int32_t>(5), createVector<int32_t>({0, 1, 1}),
       createVector<int32_t>({1, 1, 1, 2}),
       createVector<int32_t>({0, 0, 1, 1, 2, 2, 3, 3})}
      // row_partition_tensors
  );

  TF_ASSERT_OK(RunOpKernel());
  // params = [
  //           [
  //             [[15,15],[15,15]],
  //             [[15,15],[15,15]],
  //           ],
  //           [
  //             [[1, 2], [3, 4]],
  //             [[7, 8], [15, 15]],
  //           ],
  //          ]
  // params.shape = [3, 2, 3, 2]
  test::ExpectTensorEqual<int32_t>(*GetOutput(0),
                                   test::AsTensor<int32_t>(
                                       {
                                           15,
                                           15,
                                           15,
                                           15,  //
                                           15,
                                           15,
                                           15,
                                           15,  //
                                           1,
                                           2,
                                           3,
                                           4,  //
                                           7,
                                           8,
                                           15,
                                           15,  //
                                       },
                                       TensorShape({2, 2, 2, 2})));
}

TEST_F(RaggedTensorToTensorOpTest, ShapeWrongDimensions) {
  BuildRaggedTensorToTensorGraph<int32_t, int32_t>(
      TensorShape({10, 7, 10, 20}),  // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS",
       "VALUE_ROWIDS"},                     // row_partition_types
      createVector<int32_t>({1, 2, 3, 4}),  // values
      createScalar<int32_t>(15),            // default_value
      {createScalar<int32_t>(5), createVector<int32_t>({0, 1, 1}),
       createVector<int32_t>({1, 1, 1, 2})}  // row_partition_tensors
  );
  // Fails with an invalid argument.
  EXPECT_EQ(absl::IsInvalidArgument(RunOpKernel()), true);
}

TEST_F(RaggedTensorToTensorOpTest, ValuesFewerThanRowPartition) {
  BuildRaggedTensorToTensorGraph<float, int32_t>(
      TensorShape({4, 4}),                      // shape
      {"ROW_SPLITS"},                           // row_partition_types
      createVector<float>({1.0}),               // values
      createScalar<float>(1.5),                 // default_value
      {createVector<int32_t>({0, 3, 3, 7, 9})}  // row_partition_tensors
  );
  // Fails with an invalid argument because the row partitions indicate
  // a size that is larger than the values tensor.
  EXPECT_EQ(absl::IsInvalidArgument(RunOpKernel()), true);
}

TEST_F(RaggedTensorToTensorOpTest, ValidRowPartitionControl) {
  BuildRaggedTensorToTensorGraph<float, int32_t>(
      TensorShape({2, 2}),                      // shape
      {"ROW_SPLITS"},                           // row_partition_types
      createVector<float>({1.0f, 2.0f, 3.0f}),  // values
      createScalar<float>(0.0f),                // default_value
      {createVector<int32_t>({0, 2, 3})}        // row_partition_tensors
  );
  TF_ASSERT_OK(RunOpKernel());
  test::ExpectTensorNear<float>(
      *GetOutput(0),
      test::AsTensor<float>({1.0f, 2.0f, 3.0f, 0.0f}, TensorShape({2, 2})),
      0.01);

  ClearTestInputs();
  BuildRaggedTensorToTensorGraph<float, int64_t>(
      TensorShape({2, 2}),                      // shape
      {"ROW_SPLITS"},                           // row_partition_types
      createVector<float>({1.0f, 2.0f, 3.0f}),  // values
      createScalar<float>(0.0f),                // default_value
      {createVector<int64_t>({0, 2, 3})}        // row_partition_tensors
  );
  TF_ASSERT_OK(RunOpKernel());
  test::ExpectTensorNear<float>(
      *GetOutput(0),
      test::AsTensor<float>({1.0f, 2.0f, 3.0f, 0.0f}, TensorShape({2, 2})),
      0.01);
}

TEST_F(RaggedTensorToTensorOpTest, RowSplitNonZeroOriginRejected) {
  BuildRaggedTensorToTensorGraph<float, int32_t>(
      TensorShape({2, 2}),                // shape
      {"ROW_SPLITS"},                     // row_partition_types
      createVector<float>({1.0f, 2.0f}),  // values
      createScalar<float>(0.0f),          // default_value
      {createVector<int32_t>({2, 0, 2})}  // row_partition_tensors
  );
  absl::Status status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find("First value of row_split must be 0"),
            std::string::npos);

  ClearTestInputs();
  BuildRaggedTensorToTensorGraph<float, int64_t>(
      TensorShape({2, 2}),                // shape
      {"ROW_SPLITS"},                     // row_partition_types
      createVector<float>({1.0f, 2.0f}),  // values
      createScalar<float>(0.0f),          // default_value
      {createVector<int64_t>({2, 0, 2})}  // row_partition_tensors
  );
  status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find("First value of row_split must be 0"),
            std::string::npos);
}

TEST_F(RaggedTensorToTensorOpTest, RowSplitNonMonotonicRejected) {
  // Non-monotonic row_split with zero origin ([0, 2, 1]) must be rejected by
  // ValidateRowSplits before width/index subtraction.
  BuildRaggedTensorToTensorGraph<float, int32_t>(
      TensorShape({2, 2}),                // shape
      {"ROW_SPLITS"},                     // row_partition_types
      createVector<float>({1.0f, 2.0f}),  // values
      createScalar<float>(0.0f),          // default_value
      {createVector<int32_t>({0, 2, 1})}  // row_partition_tensors
  );
  absl::Status status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find("row_split must be monotonically increasing"),
            std::string::npos);

  // Zero-sized output shape still validates row_split ordering in Compute.
  ClearTestInputs();
  BuildRaggedTensorToTensorGraph<float, int32_t>(
      TensorShape({2, 0}),                // shape
      {"ROW_SPLITS"},                     // row_partition_types
      createVector<float>({}),            // values
      createScalar<float>(0.0f),          // default_value
      {createVector<int32_t>({0, 2, 1})}  // row_partition_tensors
  );
  status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find("row_split must be monotonically increasing"),
            std::string::npos);

  // The same two cases with int64 row splits.
  ClearTestInputs();
  BuildRaggedTensorToTensorGraph<float, int64_t>(
      TensorShape({2, 2}),                // shape
      {"ROW_SPLITS"},                     // row_partition_types
      createVector<float>({1.0f, 2.0f}),  // values
      createScalar<float>(0.0f),          // default_value
      {createVector<int64_t>({0, 2, 1})}  // row_partition_tensors
  );
  status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find("row_split must be monotonically increasing"),
            std::string::npos);

  ClearTestInputs();
  BuildRaggedTensorToTensorGraph<float, int64_t>(
      TensorShape({2, 0}),                // shape
      {"ROW_SPLITS"},                     // row_partition_types
      createVector<float>({}),            // values
      createScalar<float>(0.0f),          // default_value
      {createVector<int64_t>({0, 2, 1})}  // row_partition_tensors
  );
  status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find("row_split must be monotonically increasing"),
            std::string::npos);
}

TEST_F(RaggedTensorToTensorOpTest, EmptyRowSplitPartitionRejected) {
  BuildRaggedTensorToTensorGraph<float, int32_t>(
      TensorShape({0, 2, 2}),        // shape
      {"ROW_SPLITS", "ROW_SPLITS"},  // row_partition_types
      createVector<float>({}),       // values
      createScalar<float>(0.0f),     // default_value
      {createVector<int32_t>({0}), createVector<int32_t>({})}
      // row_partition_tensors
  );
  absl::Status status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find("Row split tensor must not be empty"),
            std::string::npos);

  ClearTestInputs();
  BuildRaggedTensorToTensorGraph<float, int32_t>(
      TensorShape({1, 2, 2}),        // shape
      {"ROW_SPLITS", "ROW_SPLITS"},  // row_partition_types
      createVector<float>({1.0f}),   // values
      createScalar<float>(0.0f),     // default_value
      {createVector<int32_t>({0, 1}), createVector<int32_t>({})}
      // row_partition_tensors
  );
  status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find("Row split tensor must not be empty"),
            std::string::npos);

  // The same two cases with int64 row splits.
  ClearTestInputs();
  BuildRaggedTensorToTensorGraph<float, int64_t>(
      TensorShape({0, 2, 2}),        // shape
      {"ROW_SPLITS", "ROW_SPLITS"},  // row_partition_types
      createVector<float>({}),       // values
      createScalar<float>(0.0f),     // default_value
      {createVector<int64_t>({0}), createVector<int64_t>({})}
      // row_partition_tensors
  );
  status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find("Row split tensor must not be empty"),
            std::string::npos);

  ClearTestInputs();
  BuildRaggedTensorToTensorGraph<float, int64_t>(
      TensorShape({1, 2, 2}),        // shape
      {"ROW_SPLITS", "ROW_SPLITS"},  // row_partition_types
      createVector<float>({1.0f}),   // values
      createScalar<float>(0.0f),     // default_value
      {createVector<int64_t>({0, 1}), createVector<int64_t>({})}
      // row_partition_tensors
  );
  status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find("Row split tensor must not be empty"),
            std::string::npos);
}

TEST_F(RaggedTensorToTensorOpTest, NegativeFirstValueRowIdRejected) {
  BuildRaggedTensorToTensorGraph<float, int32_t>(
      TensorShape({2, 2}),                 // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS"},  // row_partition_types
      createVector<float>({1.0f}),         // values
      createScalar<float>(0.0f),           // default_value
      {createScalar<int32_t>(2), createVector<int32_t>({-1})}
      // row_partition_tensors
  );
  absl::Status status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find("which is not in [0, 2)"), std::string::npos);

  ClearTestInputs();
  BuildRaggedTensorToTensorGraph<float, int64_t>(
      TensorShape({2, 2}),                 // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS"},  // row_partition_types
      createVector<float>({1.0f}),         // values
      createScalar<float>(0.0f),           // default_value
      {createScalar<int64_t>(2), createVector<int64_t>({-1})}
      // row_partition_tensors
  );
  status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find("which is not in [0, 2)"), std::string::npos);
}

TEST_F(RaggedTensorToTensorOpTest, InvalidFirstDimensionPartitionRejected) {
  // Negative FIRST_DIM_SIZE values must be rejected for both index types.
  BuildRaggedTensorToTensorGraph<float, int32_t>(
      TensorShape({0, 2}),                 // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS"},  // row_partition_types
      createVector<float>({1.0f}),         // values
      createScalar<float>(0.0f),           // default_value
      {createScalar<int32_t>(-1), createVector<int32_t>({0})}
      // row_partition_tensors
  );
  absl::Status status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find("First dimension size must be non-negative"),
            std::string::npos);

  ClearTestInputs();
  BuildRaggedTensorToTensorGraph<float, int64_t>(
      TensorShape({0, 2}),                 // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS"},  // row_partition_types
      createVector<float>({1.0f}),         // values
      createScalar<float>(0.0f),           // default_value
      {createScalar<int64_t>(-1), createVector<int64_t>({0})}
      // row_partition_tensors
  );
  status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find("First dimension size must be non-negative"),
            std::string::npos);

  // Non-vector first ROW_SPLITS tensors must be rejected for both index types.
  ClearTestInputs();
  BuildRaggedTensorToTensorGraph<float, int32_t>(
      TensorShape({1, 2}),        // shape
      {"ROW_SPLITS"},             // row_partition_types
      createVector<float>({}),    // values
      createScalar<float>(0.0f),  // default_value
      {ShapeAndValues<int32_t>{TensorShape({2, 1}), {0, 0}}}  // rank-2 tensor
  );
  status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find("ROW_SPLITS has invalid shape [2,1]"),
            std::string::npos);

  ClearTestInputs();
  BuildRaggedTensorToTensorGraph<float, int64_t>(
      TensorShape({1, 2}),        // shape
      {"ROW_SPLITS"},             // row_partition_types
      createVector<float>({}),    // values
      createScalar<float>(0.0f),  // default_value
      {ShapeAndValues<int64_t>{TensorShape({2, 1}), {0, 0}}}  // rank-2 tensor
  );
  status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find("ROW_SPLITS has invalid shape [2,1]"),
            std::string::npos);
}

TEST_F(RaggedTensorToTensorOpTest, EmptyFirstRowSplitPartitionRejected) {
  // Regression guard: reject an empty first ROW_SPLITS tensor before
  // GetFirstDimensionSize could compute dim_size(0) - 1.
  BuildRaggedTensorToTensorGraph<float, int32_t>(
      TensorShape({0, 2}),         // shape
      {"ROW_SPLITS"},              // row_partition_types
      createVector<float>({}),     // values
      createScalar<float>(0.0f),   // default_value
      {createVector<int32_t>({})}  // row_partition_tensors
  );
  absl::Status status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find("Invalid first partition input"),
            std::string::npos);

  ClearTestInputs();
  BuildRaggedTensorToTensorGraph<float, int64_t>(
      TensorShape({0, 2}),         // shape
      {"ROW_SPLITS"},              // row_partition_types
      createVector<float>({}),     // values
      createScalar<float>(0.0f),   // default_value
      {createVector<int64_t>({})}  // row_partition_tensors
  );
  status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find("Invalid first partition input"),
            std::string::npos);
}

TEST_F(RaggedTensorToTensorOpTest, RowSplitsValidatedBeforeFirstDimensionSize) {
  // Rank validation precedes GetFirstDimensionSize and row-split validation.
  BuildRaggedTensorToTensorGraph<float, int32_t>(
      TensorShape({1, 2}),        // shape
      {"ROW_SPLITS"},             // row_partition_types
      createVector<float>({}),    // values
      createScalar<float>(0.0f),  // default_value
      {ShapeAndValues<int32_t>{TensorShape({2, 1}), {2, 0}}}  // rank-2 tensor
  );
  absl::Status status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find("ROW_SPLITS has invalid shape [2,1]"),
            std::string::npos);

  ClearTestInputs();
  BuildRaggedTensorToTensorGraph<float, int64_t>(
      TensorShape({1, 2}),        // shape
      {"ROW_SPLITS"},             // row_partition_types
      createVector<float>({}),    // values
      createScalar<float>(0.0f),  // default_value
      {ShapeAndValues<int64_t>{TensorShape({2, 1}), {2, 0}}}  // rank-2 tensor
  );
  status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find("ROW_SPLITS has invalid shape [2,1]"),
            std::string::npos);
}

TEST_F(RaggedTensorToTensorOpTest, NonScalarFirstDimensionSizeRejected) {
  BuildRaggedTensorToTensorGraph<float, int32_t>(
      TensorShape({1, 1}),                 // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS"},  // row_partition_types
      createVector<float>({1.0f}),         // values
      createScalar<float>(0.0f),           // default_value
      {createVector<int32_t>({1, 1}), createVector<int32_t>({0})}
      // row_partition_tensors
  );
  absl::Status status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find("FIRST_DIM_SIZE has invalid shape [2]"),
            std::string::npos);

  ClearTestInputs();
  BuildRaggedTensorToTensorGraph<float, int64_t>(
      TensorShape({1, 1}),                 // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS"},  // row_partition_types
      createVector<float>({1.0f}),         // values
      createScalar<float>(0.0f),           // default_value
      {createVector<int64_t>({1, 1}), createVector<int64_t>({0})}
      // row_partition_tensors
  );
  status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find("FIRST_DIM_SIZE has invalid shape [2]"),
            std::string::npos);
}

TEST_F(RaggedTensorToTensorOpTest, InnerRowSplitRankRejected) {
  BuildRaggedTensorToTensorGraph<float, int32_t>(
      TensorShape({1, 1, 1}),        // shape
      {"ROW_SPLITS", "ROW_SPLITS"},  // row_partition_types
      createVector<float>({1.0f}),   // values
      createScalar<float>(0.0f),     // default_value
      {createVector<int32_t>({0, 1}),
       ShapeAndValues<int32_t>{TensorShape({2, 1}), {0, 1}}}
      // row_partition_tensors
  );
  absl::Status status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find(
                "Row partition tensor 1 of type ROW_SPLITS has invalid shape"),
            std::string::npos);

  ClearTestInputs();
  BuildRaggedTensorToTensorGraph<float, int64_t>(
      TensorShape({1, 1, 1}),        // shape
      {"ROW_SPLITS", "ROW_SPLITS"},  // row_partition_types
      createVector<float>({1.0f}),   // values
      createScalar<float>(0.0f),     // default_value
      {createVector<int64_t>({0, 1}),
       ShapeAndValues<int64_t>{TensorShape({2, 1}), {0, 1}}}
      // row_partition_tensors
  );
  status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find(
                "Row partition tensor 1 of type ROW_SPLITS has invalid shape"),
            std::string::npos);
}

TEST_F(RaggedTensorToTensorOpTest, ValueRowIdsRankRejected) {
  BuildRaggedTensorToTensorGraph<float, int32_t>(
      TensorShape({1, 1}),                 // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS"},  // row_partition_types
      createVector<float>({1.0f}),         // values
      createScalar<float>(0.0f),           // default_value
      {createScalar<int32_t>(1),
       ShapeAndValues<int32_t>{TensorShape({1, 1}), {0}}}
      // row_partition_tensors
  );
  absl::Status status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(
      status.message().find(
          "Row partition tensor 1 of type VALUE_ROWIDS has invalid shape"),
      std::string::npos);

  ClearTestInputs();
  BuildRaggedTensorToTensorGraph<float, int64_t>(
      TensorShape({1, 1}),                 // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS"},  // row_partition_types
      createVector<float>({1.0f}),         // values
      createScalar<float>(0.0f),           // default_value
      {createScalar<int64_t>(1),
       ShapeAndValues<int64_t>{TensorShape({1, 1}), {0}}}
      // row_partition_tensors
  );
  status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(
      status.message().find(
          "Row partition tensor 1 of type VALUE_ROWIDS has invalid shape"),
      std::string::npos);
}

TEST_F(RaggedTensorToTensorOpTest, ShortPartitionListRejected) {
  BuildRaggedTensorToTensorGraph<float, int32_t>(
      TensorShape({1, 1}),                 // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS"},  // row_partition_types
      createVector<float>({1.0f}),         // values
      createScalar<float>(0.0f),           // default_value
      {createScalar<int32_t>(1)}           // row_partition_tensors
  );
  absl::Status status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find(
                "Expected 2 row partition tensors and ragged rank >= 1, got 1"),
            std::string::npos);

  ClearTestInputs();
  BuildRaggedTensorToTensorGraph<float, int64_t>(
      TensorShape({1, 1}),                 // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS"},  // row_partition_types
      createVector<float>({1.0f}),         // values
      createScalar<float>(0.0f),           // default_value
      {createScalar<int64_t>(1)}           // row_partition_tensors
  );
  status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find(
                "Expected 2 row partition tensors and ragged rank >= 1, got 1"),
            std::string::npos);
}

TEST_F(RaggedTensorToTensorOpTest, ScalarValuesRejected) {
  // values is checked for rank >= 1 before Compute reads its first dimension.
  BuildRaggedTensorToTensorGraph<float, int32_t>(
      TensorShape({1, 1}),             // shape
      {"ROW_SPLITS"},                  // row_partition_types
      createScalar<float>(1.0f),       // values
      createScalar<float>(0.0f),       // default_value
      {createVector<int32_t>({0, 1})}  // row_partition_tensors
  );
  absl::Status status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(
      status.message().find("values must have rank at least 1, got shape []"),
      std::string::npos);

  ClearTestInputs();
  BuildRaggedTensorToTensorGraph<float, int64_t>(
      TensorShape({1, 1}),             // shape
      {"ROW_SPLITS"},                  // row_partition_types
      createScalar<float>(1.0f),       // values
      createScalar<float>(0.0f),       // default_value
      {createVector<int64_t>({0, 1})}  // row_partition_tensors
  );
  status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(
      status.message().find("values must have rank at least 1, got shape []"),
      std::string::npos);
}

TEST_F(RaggedTensorToTensorOpTest,
       RowSplitLastValueBoundedBeforeIndexCalculation) {
  // The output is empty, so index calculation never runs and only the early
  // bound can reject the split.
  BuildRaggedTensorToTensorGraph<float, int32_t>(
      TensorShape({0, 2}),             // shape
      {"ROW_SPLITS"},                  // row_partition_types
      createVector<float>({1.0f}),     // values
      createScalar<float>(0.0f),       // default_value
      {createVector<int32_t>({0, 2})}  // row_partition_tensors
  );
  absl::Status status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find(
                "Row partition size (2) exceeds the number of values (1)"),
            std::string::npos);

  ClearTestInputs();
  BuildRaggedTensorToTensorGraph<float, int64_t>(
      TensorShape({1, 1}),                            // shape
      {"ROW_SPLITS"},                                 // row_partition_types
      createVector<float>({}),                        // values
      createScalar<float>(0.0f),                      // default_value
      {createVector<int64_t>({0, int64_t{1} << 62})}  // row partitions
  );
  status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find("exceeds the number of values (0)"),
            std::string::npos);
}

TEST_F(RaggedTensorToTensorOpTest, NestedRowSplitLastValueBounded) {
  // The output is empty, so index calculation never runs and only the early
  // bound can reject the inner split.
  BuildRaggedTensorToTensorGraph<float, int32_t>(
      TensorShape({0, 1, 1}),        // shape
      {"ROW_SPLITS", "ROW_SPLITS"},  // row_partition_types
      createVector<float>({1.0f}),   // values
      createScalar<float>(0.0f),     // default_value
      {createVector<int32_t>({0, 1}), createVector<int32_t>({0, 2})}
      // row_partition_tensors
  );
  absl::Status status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find(
                "Row partition size (2) exceeds the number of values (1)"),
            std::string::npos);

  ClearTestInputs();
  BuildRaggedTensorToTensorGraph<float, int64_t>(
      TensorShape({1, 1, 1}),        // shape
      {"ROW_SPLITS", "ROW_SPLITS"},  // row_partition_types
      createVector<float>({}),       // values
      createScalar<float>(0.0f),     // default_value
      {createVector<int64_t>({0, 1}),
       createVector<int64_t>({0, int64_t{1} << 62})}  // row_partition_tensors
  );
  status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find("exceeds the number of values (0)"),
            std::string::npos);
}

TEST_F(RaggedTensorToTensorOpTest, NestedRowSplitPartitionSizesMustMatch) {
  BuildRaggedTensorToTensorGraph<float, int32_t>(
      TensorShape({1, 1, 1}),        // shape
      {"ROW_SPLITS", "ROW_SPLITS"},  // row_partition_types
      createVector<float>({1.0f}),   // values
      createScalar<float>(0.0f),     // default_value
      {createVector<int32_t>({0, 2}), createVector<int32_t>({0, 1})}
      // row_partition_tensors
  );
  absl::Status status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find(
                "Row partition size (2) does not match the next row partition"),
            std::string::npos);

  ClearTestInputs();
  BuildRaggedTensorToTensorGraph<float, int64_t>(
      TensorShape({1, 1, 1}),        // shape
      {"ROW_SPLITS", "ROW_SPLITS"},  // row_partition_types
      createVector<float>({1.0f}),   // values
      createScalar<float>(0.0f),     // default_value
      {createVector<int64_t>({0, 0}), createVector<int64_t>({0, 1})}
      // row_partition_tensors
  );
  status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find(
                "Row partition size (0) does not match the next row partition"),
            std::string::npos);
}

TEST_F(RaggedTensorToTensorOpTest, InnerRowSplitPartitionRejectedInCompute) {
  // A non-monotonic inner ROW_SPLITS partition is rejected by Compute before
  // CalculateOutputIndexRowSplit consumes it.
  BuildRaggedTensorToTensorGraph<float, int32_t>(
      TensorShape({2, 2, 2}),             // shape
      {"ROW_SPLITS", "ROW_SPLITS"},       // row_partition_types
      createVector<float>({1.0f, 2.0f}),  // values
      createScalar<float>(0.0f),          // default_value
      {createVector<int32_t>({0, 1, 2}), createVector<int32_t>({0, 2, 1})}
      // row_partition_tensors
  );
  absl::Status status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find(
                "row_split must be monotonically increasing, got 1 after 2"),
            std::string::npos);

  // A second ROW_SPLITS partition with a non-zero origin is rejected the same
  // way.
  ClearTestInputs();
  BuildRaggedTensorToTensorGraph<float, int32_t>(
      TensorShape({1, 2, 2}),             // shape
      {"ROW_SPLITS", "ROW_SPLITS"},       // row_partition_types
      createVector<float>({1.0f, 2.0f}),  // values
      createScalar<float>(0.0f),          // default_value
      {createVector<int32_t>({0, 2}), createVector<int32_t>({1, 1, 2})}
      // row_partition_tensors
  );
  status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find("First value of row_split must be 0, got 1"),
            std::string::npos);

  // The same two cases with int64 row splits.
  ClearTestInputs();
  BuildRaggedTensorToTensorGraph<float, int64_t>(
      TensorShape({2, 2, 2}),             // shape
      {"ROW_SPLITS", "ROW_SPLITS"},       // row_partition_types
      createVector<float>({1.0f, 2.0f}),  // values
      createScalar<float>(0.0f),          // default_value
      {createVector<int64_t>({0, 1, 2}), createVector<int64_t>({0, 2, 1})}
      // row_partition_tensors
  );
  status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find(
                "row_split must be monotonically increasing, got 1 after 2"),
            std::string::npos);

  ClearTestInputs();
  BuildRaggedTensorToTensorGraph<float, int64_t>(
      TensorShape({1, 2, 2}),             // shape
      {"ROW_SPLITS", "ROW_SPLITS"},       // row_partition_types
      createVector<float>({1.0f, 2.0f}),  // values
      createScalar<float>(0.0f),          // default_value
      {createVector<int64_t>({0, 2}), createVector<int64_t>({1, 1, 2})}
      // row_partition_tensors
  );
  status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find("First value of row_split must be 0, got 1"),
            std::string::npos);
}

TEST_F(RaggedTensorToTensorOpTest, FirstValueRowIdOutOfRangeRejected) {
  // A first value_rowid equal to the parent size is rejected for both index
  // types.
  BuildRaggedTensorToTensorGraph<float, int32_t>(
      TensorShape({2, 2}),                 // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS"},  // row_partition_types
      createVector<float>({1.0f}),         // values
      createScalar<float>(0.0f),           // default_value
      {createScalar<int32_t>(2), createVector<int32_t>({2})}
      // row_partition_tensors
  );
  absl::Status status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find("which is not in [0, 2)"), std::string::npos);

  ClearTestInputs();
  BuildRaggedTensorToTensorGraph<float, int64_t>(
      TensorShape({2, 2}),                 // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS"},  // row_partition_types
      createVector<float>({1.0f}),         // values
      createScalar<float>(0.0f),           // default_value
      {createScalar<int64_t>(2), createVector<int64_t>({2})}
      // row_partition_tensors
  );
  status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_NE(status.message().find("which is not in [0, 2)"), std::string::npos);
}

class RaggedTensorToTensorOpUnknownShapeTest
    : public ::tensorflow::OpsTestBase {
 protected:
  std::unique_ptr<ShapeInferenceTestOp> op_;
  void SetAttributes(const absl::Span<const std::string> row_partition_types,
                     int num_row_partition_tensors) {
    op_ = std::make_unique<ShapeInferenceTestOp>("RaggedTensorToTensor");
    SetAttrValue(row_partition_types,
                 &((*op_->node_def.mutable_attr())["row_partition_types"]));
    (*op_->node_def.mutable_attr())["num_row_partition_tensors"].set_i(
        num_row_partition_tensors);
  }
};

TEST_F(RaggedTensorToTensorOpUnknownShapeTest, ValueRowIDs) {
  SetAttributes(absl::Span<const std::string>{"FIRST_DIM_SIZE", "VALUE_ROWIDS"},
                2);

  INFER_OK(*op_, "?;?;?;?;?", "?");
  INFER_OK(*op_, "?;[6];[];[];[6]", "[?,?]");
  INFER_OK(*op_, "?;[6];?;[];[6]", "[?,?]");
  INFER_OK(*op_, "?;?;[];[];[6]", "?");
  INFER_OK(*op_, "?;[6];?;[];[6]", "[?,?]");
  INFER_OK(*op_, "?;[6,2];?;[];[6]", "[?,?,2]");
  INFER_OK(*op_, "?;[6,2];[2];[];[6]", "[?,?,2]");
  INFER_OK(*op_, "?;[6,2,7];[2,7];[];[6]", "[?,?,2,7]");
  INFER_ERROR(
      "default_value.shape=[3] and rt_input.flat_values.shape=[6,2] "
      "are incompatible",
      *op_, "?;[6,2];[3];[];[6]");
  INFER_ERROR(
      "default_value.shape=[2,2] and rt_input.flat_values.shape="
      "[6,2,1,2] are incompatible",
      *op_, "?;[6,2,1,2];[2,2];[];[6]");
  INFER_ERROR("must be a vector", *op_, "?;[6];[];[];[3,6]");
  INFER_ERROR("must be a scalar", *op_, "?;[6];[];[7];[3]");
}

TEST_F(RaggedTensorToTensorOpUnknownShapeTest, RowSplits) {
  // RaggedTensorToTensor(param_splits+, param_values, indices) -> [splits+,
  // values]
  SetAttributes(absl::Span<const std::string>{"ROW_SPLITS"}, 1);

  // value, default_value, ROW_SPLITS
  INFER_OK(*op_, "?;?;?;?", "?");
  INFER_OK(*op_, "?;[3];[];[6]", "[?,?]");
  INFER_OK(*op_, "?;?;?;?", "?");
  INFER_OK(*op_, "?;[3,2];[2];[6]", "[?,?,2]");
  INFER_OK(*op_, "?;[3,2,7];[2,7];[6]", "[?,?,2,7]");
  INFER_OK(*op_, "?;[3,2,7];[2,7];[6]", "[?,?,2,7]");
}

}  // namespace
}  // namespace tensorflow
