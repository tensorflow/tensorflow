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
  TensorShape shape({static_cast<int64>(values.size())});
  return {shape, values};
}

template <typename VALUE_TYPE>
ShapeAndValues<VALUE_TYPE> createScalar(const VALUE_TYPE& values) {
  TensorShape shape({});
  return {shape, {values}};
}

class RaggedTensorToTensorOpTest : public ::tensorflow::OpsTestBase {
 protected:
  // Builds the tensorflow test graph for RaggedTensorToTensor.
  template <typename VALUE_TYPE, typename INDEX_TYPE>
  void BuildRaggedTensorToTensorGraph(
      const TensorShape& shape, const std::vector<string>& row_partition_types,
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
  BuildRaggedTensorToTensorGraph<float, int32>(
      TensorShape({4, 4}),                 // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS"},  // row_partition_types
      createVector<float>({.1, .2, .3, .4, .5, .6, .7, .8, .9}),  // values
      createScalar<float>(1.5),  // default_value
      {createScalar<int32>(4), createVector<int32>({0, 0, 0, 2, 2, 2, 2, 3, 3})}
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
  BuildRaggedTensorToTensorGraph<float, int32>(
      TensorShape({4, 4}),  // shape
      {"ROW_SPLITS"},       // row_partition_types
      createVector<float>({.1, .2, .3, .4, .5, .6, .7, .8, .9}),  // values
      createScalar<float>(1.5),               // default_value
      {createVector<int32>({0, 3, 3, 7, 9})}  // row_partition_tensors
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
  BuildRaggedTensorToTensorGraph<float, int32>(
      TensorShape({5, 2, 3}),  // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS",
       "VALUE_ROWIDS"},  // row_partition_types
      createVector<float>({.1, .2, .3, .4, .5, .6, .7, .8, .9}),  // values
      createScalar<float>(1.5),  // default_value
      {
          createScalar<int32>(5),
          createVector<int32>({0, 1, 1, 3, 3, 4}),
          createVector<int32>({1, 1, 2, 3, 3, 4, 4, 4, 5}),
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
  BuildRaggedTensorToTensorGraph<float, int32>(
      TensorShape({5, 2, 3}),        // shape
      {"ROW_SPLITS", "ROW_SPLITS"},  // row_partition_types
      createVector<float>({.1, .2, .3, .4, .5, .6, .7, .8, .9}),  // values
      createScalar<float>(1.5),  // default_value
      {
          createVector<int32>({0, 1, 3, 3, 5, 6}),
          createVector<int32>({0, 0, 2, 3, 5, 8, 9}),
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
  BuildRaggedTensorToTensorGraph<int64, int64>(
      TensorShape({3, 2, 3}),             // shape
      {"ROW_SPLITS", "ROW_SPLITS"},       // row_partition_types
      createVector<int64>({0, 1, 2, 3}),  // values
      createScalar<int64>(5),             // default_value
      {
          createVector<int64>({0, 2, 2, 3}),
          createVector<int64>({0, 3, 3, 4}),
      }  // row_partition_tensors
  );
  TF_ASSERT_OK(RunOpKernel());

  // Expected = [
  //              [[0, 1, 2], [5, 5, 5]],
  //              [[5, 5, 5], [5, 5, 5]],
  //              [[3, 5, 5], [5, 5, 5]]
  //            ]
  test::ExpectTensorEqual<int64>(
      *GetOutput(0), test::AsTensor<int64>(
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
  BuildRaggedTensorToTensorGraph<int32, int32>(
      TensorShape({4, 2, 3, 2}),  // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS", "VALUE_ROWIDS",
       "VALUE_ROWIDS"},                               // row_partition_types
      createVector<int32>({1, 2, 3, 4, 5, 6, 7, 8}),  // values
      createScalar<int32>(15),                        // default_value
      {createScalar<int32>(5), createVector<int32>({0, 1, 1}),
       createVector<int32>({1, 1, 1, 2}),
       createVector<int32>({0, 0, 1, 1, 2, 2, 3, 3})}  // row_partition_tensors
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
  test::ExpectTensorEqual<int32>(
      *GetOutput(0),
      test::AsTensor<int32>(
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
  BuildRaggedTensorToTensorGraph<int32, int32>(
      TensorShape({4, 2, 3, 2}),  // shape
      {"ROW_SPLITS", "ROW_SPLITS", "ROW_SPLITS"},
      // row_partition_types
      createVector<int32>({1, 2, 3, 4, 5, 6, 7, 8}),  // values
      createScalar<int32>(15),                        // default_value
      {createVector<int32>({0, 1, 3}), createVector<int32>({0, 0, 3, 4}),
       createVector<int32>({0, 2, 4, 6, 8})}  // row_partition_tensors
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
  test::ExpectTensorEqual<int32>(
      *GetOutput(0),
      test::AsTensor<int32>(
          {15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 1,  2,  3,  4,
           5,  6,  7,  8,  15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
           15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15},
          TensorShape({4, 2, 3, 2})));
}

TEST_F(RaggedTensorToTensorOpTest, RaggedTensorToTensorContractExpanded) {
  // params = [[.1, .2, .3], [], [.4, .5, .6, .7], [.8, .9]]
  BuildRaggedTensorToTensorGraph<float, int32>(
      TensorShape({3, 5}),                 // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS"},  // row_partition_types
      createVector<float>({.1, .2, .3, .4, .5, .6, .7, .8, .9}),  // values
      createScalar<float>(1.5),  // default_value
      {createScalar<int32>(4), createVector<int32>({0, 0, 0, 2, 2, 2, 2, 3, 3})}
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
  BuildRaggedTensorToTensorGraph<float, int32>(
      TensorShape({3, 5, 2}),              // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS"},  // row_partition_types
      ShapeAndValues<float>{TensorShape({9, 2}),
                            {.1, 1.1, .2, 1.2, .3, 1.3, .4, 1.4, .5, 1.5, .6,
                             1.6, .7, 1.7, .8, 1.8, .9, 1.9}},  // values
      createScalar<float>(1.5),                                 // default_value
      {createScalar<int32>(4), createVector<int32>({0, 0, 0, 2, 2, 2, 2, 3, 3})}
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
  BuildRaggedTensorToTensorGraph<float, int32>(
      TensorShape({3, 3}),                 // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS"},  // row_partition_types
      createVector<float>({.1, .2, .3, .4, .5, .6, .7, .8, .9}),  // values
      createScalar<float>(1.5),  // default_value
      {createScalar<int32>(4), createVector<int32>({0, 0, 0, 2, 2, 2, 2, 3, 3})}
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
  BuildRaggedTensorToTensorGraph<float, int32>(
      TensorShape({4, 1, 2}),  // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS",
       "VALUE_ROWIDS"},  // row_partition_types
      createVector<float>({.1, .2, .3, .4, .5, .6, .7, .8, .9}),  // values
      createScalar<float>(1.5),  // default_value
      {
          createScalar<int32>(5),
          createVector<int32>({0, 1, 1, 3, 3, 4}),
          createVector<int32>({1, 1, 2, 3, 3, 4, 4, 4, 5}),
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
  BuildRaggedTensorToTensorGraph<int32, int32>(
      TensorShape({2, 2, 2, 2}),  // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS", "VALUE_ROWIDS",
       "VALUE_ROWIDS"},                               // row_partition_types
      createVector<int32>({1, 2, 3, 4, 5, 6, 7, 8}),  // values
      createScalar<int32>(15),                        // default_value
      {createScalar<int32>(5), createVector<int32>({0, 1, 1}),
       createVector<int32>({1, 1, 1, 2}),
       createVector<int32>({0, 0, 1, 1, 2, 2, 3, 3})}  // row_partition_tensors
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
  test::ExpectTensorEqual<int32>(*GetOutput(0), test::AsTensor<int32>(
                                                    {
                                                        15, 15, 15, 15,  //
                                                        15, 15, 15, 15,  //
                                                        1, 2, 3, 4,      //
                                                        7, 8, 15, 15,    //
                                                    },
                                                    TensorShape({2, 2, 2, 2})));
}

TEST_F(RaggedTensorToTensorOpTest, ShapeWrongDimensions) {
  BuildRaggedTensorToTensorGraph<int32, int32>(
      TensorShape({10, 7, 10, 20}),  // shape
      {"FIRST_DIM_SIZE", "VALUE_ROWIDS",
       "VALUE_ROWIDS"},                   // row_partition_types
      createVector<int32>({1, 2, 3, 4}),  // values
      createScalar<int32>(15),            // default_value
      {createScalar<int32>(5), createVector<int32>({0, 1, 1}),
       createVector<int32>({1, 1, 1, 2})}  // row_partition_tensors
  );
  // Fails with an invalid argument.
  EXPECT_EQ(RunOpKernel().code(), errors::Code::INVALID_ARGUMENT);
}

class RaggedTensorToTensorOpUnknownShapeTest
    : public ::tensorflow::OpsTestBase {
 protected:
  std::unique_ptr<ShapeInferenceTestOp> op_;
  void SetAttributes(const gtl::ArraySlice<string> row_partition_types,
                     int num_row_partition_tensors) {
    op_ = absl::make_unique<ShapeInferenceTestOp>("RaggedTensorToTensor");
    SetAttrValue(row_partition_types,
                 &((*op_->node_def.mutable_attr())["row_partition_types"]));
    (*op_->node_def.mutable_attr())["num_row_partition_tensors"].set_i(
        num_row_partition_tensors);
  }
};

TEST_F(RaggedTensorToTensorOpUnknownShapeTest, ValueRowIDs) {
  SetAttributes(gtl::ArraySlice<string>{"FIRST_DIM_SIZE", "VALUE_ROWIDS"}, 2);

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
  SetAttributes(gtl::ArraySlice<string>{"ROW_SPLITS"}, 1);

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
