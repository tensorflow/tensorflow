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

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/dataset_ops_internal.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/kernels/data/dataset_test_base.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace data {
namespace test_util {

using namespace tensorflow::ops;
using namespace tensorflow::ops::internal;

Status CreateMapDatasetGraphDef(GraphDef* graph_def) {
  Scope root = Scope::NewRootScope();
  auto start = Const(root, 0LL);
  auto stop = Const(root, 10LL);
  auto step = Const(root, 1LL);
  auto range = RangeDataset(root, start, stop, step, {DT_INT64},
                            {PartialTensorShape({})});

  auto map_fn = NameAttrList();
  map_fn.set_name("XAddX");
  auto dtype = AttrValue();
  dtype.set_type(DT_INT64);
  map_fn.mutable_attr()->insert({"T", dtype});
  auto other_args = InputList(std::initializer_list<Input>({}));
  auto map = MapDataset(root, range, other_args, map_fn, {DT_INT64},
                        {PartialTensorShape({1})});
  auto ret = _Retval(root.WithOpName("RetVal"), map, 0);

  TF_RETURN_IF_ERROR(root.ToGraphDef(graph_def));

  VersionDef* versions = graph_def->mutable_versions();
  versions->set_producer(TF_GRAPH_DEF_VERSION);
  versions->set_min_consumer(TF_GRAPH_DEF_VERSION_MIN_CONSUMER);
  *graph_def->mutable_library()->add_function() = test::function::XAddX();

  return Status::OK();
}

Status map_test_case(GraphDefTestCase* test_case) {
  GraphDef graph_def;
  TF_RETURN_IF_ERROR(CreateMapDatasetGraphDef(&graph_def));
  int num_elements = 10;
  std::vector<std::vector<Tensor>> outputs(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    outputs[i] = CreateTensors<int64>(TensorShape{}, {{i + i}});
  }
  *test_case = {"MapGraph", graph_def, outputs};
  return Status::OK();
}

}  // namespace test_util
}  // namespace data
}  // namespace tensorflow
