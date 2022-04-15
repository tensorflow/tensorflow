/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/data/service/test_util.h"

#include <functional>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace data {
namespace testing {
namespace {

using ::tensorflow::test::AsScalar;
using ::tensorflow::test::function::GDef;
using ::tensorflow::test::function::NDef;

constexpr int64_t kShardHint = -1;
constexpr const char kTestdataDir[] =
    "tensorflow/core/data/service/testdata";
constexpr const char kInterleaveTextlineDatasetFile[] =
    "interleave_textline_dataset.pbtxt";

NodeDef GetMapNode(absl::string_view name, absl::string_view input_node_name,
                   absl::string_view function_name) {
  return NDef(
      name, /*op=*/"MapDataset", {std::string(input_node_name)},
      {{"f", FunctionDefHelper::FunctionRef(std::string(function_name))},
       {"Targuments", {}},
       {"output_shapes", gtl::ArraySlice<TensorShape>{TensorShape()}},
       {"output_types", gtl::ArraySlice<DataType>{DT_INT64}}});
}

FunctionDef XTimesX() {
  return FunctionDefHelper::Create(
      /*function_name=*/"XTimesX",
      /*in_def=*/{"x: int64"},
      /*out_def=*/{"y: int64"},
      /*attr_def=*/{},
      /*node_def=*/{{{"y"}, "Mul", {"x", "x"}, {{"T", DT_INT64}}}},
      /*ret_def=*/{{"y", "y:z:0"}});
}

Status CreateTestFiles(const std::vector<tstring>& filenames,
                       const std::vector<tstring>& contents) {
  if (filenames.size() != contents.size()) {
    return errors::InvalidArgument(
        "The number of files does not match with the contents.");
  }
  for (int i = 0; i < filenames.size(); ++i) {
    TF_RETURN_IF_ERROR(WriteDataToFile(filenames[i], contents[i].data()));
  }
  return Status::OK();
}
}  // namespace

DatasetDef RangeDataset(int64_t range) {
  DatasetDef dataset_def;
  *dataset_def.mutable_graph() = GDef(
      {NDef("start", "Const", /*inputs=*/{},
            {{"value", AsScalar<int64_t>(0)}, {"dtype", DT_INT64}}),
       NDef("stop", "Const", /*inputs=*/{},
            {{"value", AsScalar<int64_t>(range)}, {"dtype", DT_INT64}}),
       NDef("step", "Const", /*inputs=*/{},
            {{"value", AsScalar<int64_t>(1)}, {"dtype", DT_INT64}}),
       NDef("range", "RangeDataset", /*inputs=*/{"start", "stop", "step"},
            {{"output_shapes", gtl::ArraySlice<TensorShape>{TensorShape()}},
             {"output_types", gtl::ArraySlice<DataType>{DT_INT64}}}),
       NDef("dataset", "_Retval", /*inputs=*/{"range"},
            {{"T", DT_VARIANT}, {"index", 0}})},
      {});
  return dataset_def;
}

DatasetDef RangeSquareDataset(const int64_t range) {
  DatasetDef dataset_def;
  *dataset_def.mutable_graph() = GDef(
      {NDef("start", "Const", /*inputs=*/{},
            {{"value", AsScalar<int64_t>(0)}, {"dtype", DT_INT64}}),
       NDef("stop", "Const", /*inputs=*/{},
            {{"value", AsScalar<int64_t>(range)}, {"dtype", DT_INT64}}),
       NDef("step", "Const", /*inputs=*/{},
            {{"value", AsScalar<int64_t>(1)}, {"dtype", DT_INT64}}),
       NDef("range", "RangeDataset", /*inputs=*/{"start", "stop", "step"},
            {{"output_shapes", gtl::ArraySlice<TensorShape>{TensorShape()}},
             {"output_types", gtl::ArraySlice<DataType>{DT_INT64}}}),
       GetMapNode("map", "range", "XTimesX"),
       NDef("dataset", "_Retval", /*inputs=*/{"map"},
            {{"T", DT_VARIANT}, {"index", 0}})},
      {XTimesX()});
  return dataset_def;
}

DatasetDef RangeDatasetWithShardHint(const int64_t range) {
  DatasetDef dataset_def;
  *dataset_def.mutable_graph() = GDef(
      {NDef("start", "Const", /*inputs=*/{},
            {{"value", AsScalar<int64_t>(0)}, {"dtype", DT_INT64}}),
       NDef("stop", "Const", /*inputs=*/{},
            {{"value", AsScalar<int64_t>(range)}, {"dtype", DT_INT64}}),
       NDef("step", "Const", /*inputs=*/{},
            {{"value", AsScalar<int64_t>(1)}, {"dtype", DT_INT64}}),
       NDef("range", "RangeDataset", /*inputs=*/{"start", "stop", "step"},
            {{"output_shapes", gtl::ArraySlice<TensorShape>{TensorShape()}},
             {"output_types", gtl::ArraySlice<DataType>{DT_INT64}}}),
       NDef("num_shards", "Const", /*inputs=*/{},
            {{"value", AsScalar<int64_t>(kShardHint)}, {"dtype", DT_INT64}}),
       NDef("index", "Const", /*inputs=*/{},
            {{"value", AsScalar<int64_t>(kShardHint)}, {"dtype", DT_INT64}}),
       NDef("ShardDataset", "ShardDataset",
            /*inputs=*/{"range", "num_shards", "index"},
            {{"output_shapes", gtl::ArraySlice<TensorShape>{TensorShape()}},
             {"output_types", gtl::ArraySlice<DataType>{DT_INT64}}}),
       NDef("dataset", "_Retval", /*inputs=*/{"ShardDataset"},
            {{"T", DT_VARIANT}, {"index", 0}})},
      /*funcs=*/{});
  return dataset_def;
}

StatusOr<DatasetDef> InterleaveTextlineDataset(
    const std::vector<tstring>& filenames,
    const std::vector<tstring>& contents) {
  TF_RETURN_IF_ERROR(CreateTestFiles(filenames, contents));
  DatasetDef dataset;
  std::string graph_file =
      io::JoinPath(kTestdataDir, kInterleaveTextlineDatasetFile);
  TF_RETURN_IF_ERROR(
      ReadTextProto(Env::Default(), graph_file, dataset.mutable_graph()));

  Tensor filenames_tensor = test::AsTensor<tstring>(
      filenames, TensorShape({static_cast<int64_t>(filenames.size())}));
  filenames_tensor.AsProtoTensorContent(
      (*dataset.mutable_graph()->mutable_node(0)->mutable_attr())["value"]
          .mutable_tensor());
  return dataset;
}

Status WaitWhile(std::function<StatusOr<bool>()> f) {
  while (true) {
    TF_ASSIGN_OR_RETURN(bool result, f());
    if (!result) {
      return Status::OK();
    }
    Env::Default()->SleepForMicroseconds(10 * 1000);  // 10ms.
  }
}

}  // namespace testing
}  // namespace data
}  // namespace tensorflow
