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

#include "tensorflow/core/kernels/data/dataset_utils.h"

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/kernels/data/dataset_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace data {
namespace {

using ::testing::UnorderedElementsAre;

string full_name(string key) { return FullName("Iterator:", key); }

TEST(DatasetUtilsTest, MatchesAnyVersion) {
  EXPECT_TRUE(MatchesAnyVersion("BatchDataset", "BatchDataset"));
  EXPECT_TRUE(MatchesAnyVersion("BatchDataset", "BatchDatasetV2"));
  EXPECT_TRUE(MatchesAnyVersion("BatchDataset", "BatchDatasetV3"));
  EXPECT_FALSE(MatchesAnyVersion("BatchDataset", "BatchDatasetXV3"));
  EXPECT_FALSE(MatchesAnyVersion("BatchDataset", "BatchV2Dataset"));
  EXPECT_FALSE(MatchesAnyVersion("BatchDataset", "PaddedBatchDataset"));
}

TEST(DatasetUtilsTest, VariantTensorDataRoundtrip) {
  VariantTensorDataWriter writer;
  TF_ASSERT_OK(writer.WriteScalar(full_name("Int64"), 24));
  Tensor input_tensor(DT_FLOAT, {1});
  input_tensor.flat<float>()(0) = 2.0f;
  TF_ASSERT_OK(writer.WriteTensor(full_name("Tensor"), input_tensor));
  std::vector<const VariantTensorData*> data;
  writer.GetData(&data);

  VariantTensorDataReader reader(data);
  int64 val_int64;
  TF_ASSERT_OK(reader.ReadScalar(full_name("Int64"), &val_int64));
  EXPECT_EQ(val_int64, 24);
  Tensor val_tensor;
  TF_ASSERT_OK(reader.ReadTensor(full_name("Tensor"), &val_tensor));
  EXPECT_EQ(input_tensor.NumElements(), val_tensor.NumElements());
  EXPECT_EQ(input_tensor.flat<float>()(0), val_tensor.flat<float>()(0));
}

TEST(DatasetUtilsTest, VariantTensorDataNonExistentKey) {
  VariantTensorData data;
  strings::StrAppend(&data.metadata_, "key1", "@@");
  data.tensors_.push_back(Tensor(DT_INT64, {1}));
  std::vector<const VariantTensorData*> reader_data;
  reader_data.push_back(&data);
  VariantTensorDataReader reader(reader_data);
  int64 val_int64;
  tstring val_string;
  Tensor val_tensor;
  EXPECT_EQ(error::NOT_FOUND,
            reader.ReadScalar(full_name("NonExistentKey"), &val_int64).code());
  EXPECT_EQ(error::NOT_FOUND,
            reader.ReadScalar(full_name("NonExistentKey"), &val_string).code());
  EXPECT_EQ(error::NOT_FOUND,
            reader.ReadTensor(full_name("NonExistentKey"), &val_tensor).code());
}

TEST(DatasetUtilsTest, VariantTensorDataRoundtripIteratorName) {
  VariantTensorDataWriter writer;
  TF_ASSERT_OK(writer.WriteScalar("Iterator", "Int64", 24));
  Tensor input_tensor(DT_FLOAT, {1});
  input_tensor.flat<float>()(0) = 2.0f;
  TF_ASSERT_OK(writer.WriteTensor("Iterator", "Tensor", input_tensor));
  std::vector<const VariantTensorData*> data;
  writer.GetData(&data);

  VariantTensorDataReader reader(data);
  int64 val_int64;
  TF_ASSERT_OK(reader.ReadScalar("Iterator", "Int64", &val_int64));
  EXPECT_EQ(val_int64, 24);
  Tensor val_tensor;
  TF_ASSERT_OK(reader.ReadTensor("Iterator", "Tensor", &val_tensor));
  EXPECT_EQ(input_tensor.NumElements(), val_tensor.NumElements());
  EXPECT_EQ(input_tensor.flat<float>()(0), val_tensor.flat<float>()(0));
}

TEST(DatasetUtilsTest, VariantTensorDataNonExistentKeyIteratorName) {
  VariantTensorData data;
  strings::StrAppend(&data.metadata_, "key1", "@@");
  data.tensors_.push_back(Tensor(DT_INT64, {1}));
  std::vector<const VariantTensorData*> reader_data;
  reader_data.push_back(&data);
  VariantTensorDataReader reader(reader_data);
  int64 val_int64;
  tstring val_string;
  Tensor val_tensor;
  EXPECT_EQ(error::NOT_FOUND,
            reader.ReadScalar("Iterator", "NonExistentKey", &val_int64).code());
  EXPECT_EQ(
      error::NOT_FOUND,
      reader.ReadScalar("Iterator", "NonExistentKey", &val_string).code());
  EXPECT_EQ(
      error::NOT_FOUND,
      reader.ReadTensor("Iterator", "NonExistentKey", &val_tensor).code());
}

TEST(DatasetUtilsTest, VariantTensorDataWriteAfterFlushing) {
  VariantTensorDataWriter writer;
  TF_ASSERT_OK(writer.WriteScalar(full_name("Int64"), 24));
  std::vector<const VariantTensorData*> data;
  writer.GetData(&data);
  Tensor input_tensor(DT_FLOAT, {1});
  input_tensor.flat<float>()(0) = 2.0f;
  EXPECT_EQ(error::FAILED_PRECONDITION,
            writer.WriteTensor(full_name("Tensor"), input_tensor).code());
}

TEST(DatasetUtilsTest, CheckpointElementsRoundTrip) {
  std::vector<std::vector<Tensor>> elements;
  elements.push_back(CreateTensors<int32>(TensorShape({3}), {{1, 2, 3}}));
  elements.push_back(CreateTensors<int32>(TensorShape({2}), {{4, 5}}));
  VariantTensorDataWriter writer;
  tstring test_prefix = full_name("test_prefix");
  TF_ASSERT_OK(WriteElementsToCheckpoint(&writer, test_prefix, elements));
  std::vector<const VariantTensorData*> data;
  writer.GetData(&data);

  VariantTensorDataReader reader(data);
  std::vector<std::vector<Tensor>> read_elements;
  TF_ASSERT_OK(
      ReadElementsFromCheckpoint(&reader, test_prefix, &read_elements));
  ASSERT_EQ(elements.size(), read_elements.size());
  for (int i = 0; i < elements.size(); ++i) {
    std::vector<Tensor>& original = elements[i];
    std::vector<Tensor>& read = read_elements[i];

    ASSERT_EQ(original.size(), read.size());
    for (int j = 0; j < original.size(); ++j) {
      EXPECT_EQ(original[j].NumElements(), read[j].NumElements());
      EXPECT_EQ(original[j].flat<int32>()(0), read[j].flat<int32>()(0));
    }
  }
}

TEST(DatasetUtilsTest, AddToFunctionLibrary) {
  auto make_fn_a = [](const string& fn_name) {
    return FunctionDefHelper::Create(
        /*function_name=*/fn_name,
        /*in_def=*/{"arg: int64"},
        /*out_def=*/{"ret: int64"},
        /*attr_def=*/{},
        /*node_def=*/{{{"node"}, "Identity", {"arg"}, {{"T", DT_INT64}}}},
        /*ret_def=*/{{"ret", "node:output:0"}});
  };

  auto make_fn_b = [](const string& fn_name) {
    return FunctionDefHelper::Create(
        /*function_name=*/fn_name,
        /*in_def=*/{"arg: int64"},
        /*out_def=*/{"ret: int64"},
        /*attr_def=*/{},
        /*node_def=*/
        {{{"node"}, "Identity", {"arg"}, {{"T", DT_INT64}}},
         {{"node2"}, "Identity", {"node:output:0"}, {{"T", DT_INT64}}}},
        /*ret_def=*/{{"ret", "node2:output:0"}});
  };

  FunctionDefLibrary fdef_base;
  *fdef_base.add_function() = make_fn_a("0");
  *fdef_base.add_function() = make_fn_a("1");
  *fdef_base.add_function() = make_fn_a("2");

  FunctionDefLibrary fdef_to_add;
  *fdef_to_add.add_function() = make_fn_b("0");  // Override
  *fdef_to_add.add_function() = make_fn_a("1");  // Do nothing
  *fdef_to_add.add_function() = make_fn_b("3");  // Add new function

  FunctionLibraryDefinition flib_0(OpRegistry::Global(), fdef_base);
  TF_ASSERT_OK(AddToFunctionLibrary(&flib_0, fdef_to_add));

  FunctionLibraryDefinition flib_1(OpRegistry::Global(), fdef_base);
  FunctionLibraryDefinition flib_to_add(OpRegistry::Global(), fdef_to_add);
  TF_ASSERT_OK(AddToFunctionLibrary(&flib_1, flib_to_add));

  for (const auto& flib : {flib_0, flib_1}) {
    EXPECT_TRUE(FunctionDefsEqual(*flib.Find("0"), make_fn_b("0")));
    EXPECT_TRUE(FunctionDefsEqual(*flib.Find("1"), make_fn_a("1")));
    EXPECT_TRUE(FunctionDefsEqual(*flib.Find("2"), make_fn_a("2")));
    EXPECT_TRUE(FunctionDefsEqual(*flib.Find("3"), make_fn_b("3")));
  }
}

TEST(DatasetUtilsTest, AddToFunctionLibraryWithConflictingSignatures) {
  FunctionDefLibrary fdef_base;
  *fdef_base.add_function() = FunctionDefHelper::Create(
      /*function_name=*/"0",
      /*in_def=*/{"arg: int64"},
      /*out_def=*/{"ret: int64"},
      /*attr_def=*/{},
      /*node_def=*/{},
      /*ret_def=*/{{"ret", "arg"}});

  FunctionDefLibrary fdef_to_add;
  *fdef_to_add.add_function() = FunctionDefHelper::Create(
      /*function_name=*/"0",
      /*in_def=*/{"arg: int64"},
      /*out_def=*/{"ret: int64", "ret2: int64"},
      /*attr_def=*/{},
      /*node_def=*/{},
      /*ret_def=*/{{"ret", "arg"}, {"ret2", "arg"}});

  FunctionLibraryDefinition flib_0(OpRegistry::Global(), fdef_base);
  Status s = AddToFunctionLibrary(&flib_0, fdef_to_add);
  EXPECT_EQ(error::Code::INVALID_ARGUMENT, s.code());
  EXPECT_EQ(
      "Cannot add function '0' because a different function with the same "
      "signature already exists.",
      s.error_message());

  FunctionLibraryDefinition flib_1(OpRegistry::Global(), fdef_base);
  FunctionLibraryDefinition flib_to_add(OpRegistry::Global(), fdef_to_add);
  s = AddToFunctionLibrary(&flib_1, flib_to_add);
  EXPECT_EQ(error::Code::INVALID_ARGUMENT, s.code());
  EXPECT_EQ(
      "Cannot add function '0' because a different function with the same "
      "signature already exists.",
      s.error_message());
}

TEST(DatasetUtilsTest, StripDevicePlacement) {
  FunctionDefLibrary flib;
  *flib.add_function() = FunctionDefHelper::Create(
      /*function_name=*/"0",
      /*in_def=*/{"arg: int64"},
      /*out_def=*/{"ret: int64"},
      /*attr_def=*/{},
      /*node_def=*/
      {{{"node"},
        "Identity",
        {"arg"},
        {{"T", DT_INT64}},
        /*dep=*/{},
        /*device=*/"device:CPU:0"}},
      /*ret_def=*/{{"ret", "arg"}});
  EXPECT_EQ(flib.function(0).node_def(0).device(), "device:CPU:0");
  StripDevicePlacement(&flib);
  EXPECT_EQ(flib.function(0).node_def(0).device(), "");
}

TEST(DatasetUtilsTest, RunnerWithMaxParallelism) {
  auto runner =
      RunnerWithMaxParallelism([](const std::function<void()> fn) { fn(); }, 2);
  auto fn = []() { ASSERT_EQ(GetPerThreadMaxParallelism(), 2); };
  runner(fn);
}

TEST(DatasetUtilsTest, ParseDeterminismPolicy) {
  DeterminismPolicy determinism;
  TF_ASSERT_OK(DeterminismPolicy::FromString("true", &determinism));
  EXPECT_TRUE(determinism.IsDeterministic());
  TF_ASSERT_OK(DeterminismPolicy::FromString("false", &determinism));
  EXPECT_TRUE(determinism.IsNondeterministic());
  TF_ASSERT_OK(DeterminismPolicy::FromString("default", &determinism));
  EXPECT_TRUE(determinism.IsDefault());
}

TEST(DatasetUtilsTest, DeterminismString) {
  for (auto s : {"true", "false", "default"}) {
    DeterminismPolicy determinism;
    TF_ASSERT_OK(DeterminismPolicy::FromString(s, &determinism));
    EXPECT_TRUE(s == determinism.String());
  }
}

TEST(DatasetUtilsTest, BoolConstructor) {
  EXPECT_TRUE(DeterminismPolicy(true).IsDeterministic());
  EXPECT_FALSE(DeterminismPolicy(true).IsNondeterministic());
  EXPECT_FALSE(DeterminismPolicy(true).IsDefault());

  EXPECT_TRUE(DeterminismPolicy(false).IsNondeterministic());
  EXPECT_FALSE(DeterminismPolicy(false).IsDeterministic());
  EXPECT_FALSE(DeterminismPolicy(false).IsDefault());
}

class SelectOptimizationsHashTest : public ::testing::TestWithParam<uint64> {};

TEST_P(SelectOptimizationsHashTest, DatasetUtils) {
  const uint64 hash_result = GetParam();
  string job_name = "job";
  auto hash_func = [hash_result](const string& str) { return hash_result; };
  absl::flat_hash_map<string, uint64> live_experiments = {
      {"exp1", 0},  {"exp2", 20}, {"exp3", 33}, {"exp4", 45},
      {"exp5", 67}, {"exp6", 88}, {"exp7", 100}};
  std::vector<tstring> optimizations_enabled, optimizations_disabled,
      optimizations_default;
  std::vector<tstring> optimizations = SelectOptimizations(
      job_name, live_experiments, optimizations_enabled, optimizations_disabled,
      optimizations_default, hash_func);

  int tested_times = 0;
  switch (hash_result) {
    case 0:
    case 100:
    case 200:
      tested_times++;
      EXPECT_THAT(optimizations, UnorderedElementsAre("exp2", "exp3", "exp4",
                                                      "exp5", "exp6", "exp7"));
      break;
    case 33:
    case 133:
      tested_times++;
      EXPECT_THAT(optimizations,
                  UnorderedElementsAre("exp4", "exp5", "exp6", "exp7"));
      break;
    case 67:
    case 167:
      tested_times++;
      EXPECT_THAT(optimizations, UnorderedElementsAre("exp6", "exp7"));
      break;
  }
  EXPECT_EQ(tested_times, 1);
}

INSTANTIATE_TEST_SUITE_P(Test, SelectOptimizationsHashTest,
                         ::testing::Values(0, 33, 67, 100, 133, 167, 200));

class SelectOptimizationsOptTest
    : public ::testing::TestWithParam<std::tuple<string, string>> {};

TEST_P(SelectOptimizationsOptTest, DatasetUtils) {
  const string opt_ins = std::get<0>(GetParam());
  const string opt_outs = std::get<1>(GetParam());
  if (!opt_ins.empty()) {
    setenv("TF_DATA_EXPERIMENT_OPT_IN", opt_ins.c_str(), 1);
  }
  if (!opt_outs.empty()) {
    setenv("TF_DATA_EXPERIMENT_OPT_OUT", opt_outs.c_str(), 1);
  }
  string job_name = "job";
  auto hash_func = [](const string& str) { return 50; };
  absl::flat_hash_map<string, uint64> live_experiments = {
      {"exp1", 0}, {"exp2", 25}, {"exp3", 50}, {"exp4", 75}, {"exp5", 100}};
  std::vector<tstring> optimizations_enabled, optimizations_disabled,
      optimizations_default;
  std::vector<tstring> optimizations = SelectOptimizations(
      job_name, live_experiments, optimizations_enabled, optimizations_disabled,
      optimizations_default, hash_func);

  int tested_times = 0;
  if (opt_outs == "all") {
    EXPECT_THAT(optimizations, UnorderedElementsAre());
    tested_times++;
  } else if (opt_outs.empty()) {
    if (opt_ins == "all") {
      EXPECT_THAT(optimizations,
                  UnorderedElementsAre("exp1", "exp2", "exp3", "exp4", "exp5"));
      tested_times++;
    } else if (opt_ins.empty()) {
      EXPECT_THAT(optimizations, UnorderedElementsAre("exp4", "exp5"));
      tested_times++;
    } else if (opt_ins == "exp2,exp4") {
      EXPECT_THAT(optimizations, UnorderedElementsAre("exp2", "exp4", "exp5"));
      tested_times++;
    }
  } else if (opt_outs == "exp1,exp5") {
    if (opt_ins == "all") {
      EXPECT_THAT(optimizations, UnorderedElementsAre("exp2", "exp3", "exp4"));
      tested_times++;
    } else if (opt_ins.empty()) {
      EXPECT_THAT(optimizations, UnorderedElementsAre("exp4"));
      tested_times++;
    } else if (opt_ins == "exp2,exp4") {
      EXPECT_THAT(optimizations, UnorderedElementsAre("exp2", "exp4"));
      tested_times++;
    }
  }
  EXPECT_EQ(tested_times, 1);

  if (!opt_ins.empty()) {
    unsetenv("TF_DATA_EXPERIMENT_OPT_IN");
  }
  if (!opt_outs.empty()) {
    unsetenv("TF_DATA_EXPERIMENT_OPT_OUT");
  }
}

INSTANTIATE_TEST_SUITE_P(
    Test, SelectOptimizationsOptTest,
    ::testing::Combine(::testing::Values("all", "", "exp2,exp4"),
                       ::testing::Values("all", "", "exp1,exp5")));

class SelectOptimizationsConflictTest
    : public ::testing::TestWithParam<std::tuple<string, string, uint64>> {};

TEST_P(SelectOptimizationsConflictTest, DatasetUtils) {
  const string opt_ins = std::get<0>(GetParam());
  const string opt_outs = std::get<1>(GetParam());
  const uint64 hash_result = std::get<2>(GetParam());
  if (!opt_ins.empty()) {
    setenv("TF_DATA_EXPERIMENT_OPT_IN", opt_ins.c_str(), 1);
  }
  if (!opt_outs.empty()) {
    setenv("TF_DATA_EXPERIMENT_OPT_OUT", opt_outs.c_str(), 1);
  }
  string job_name = "job";
  auto hash_func = [hash_result](const string& str) { return hash_result; };
  absl::flat_hash_map<string, uint64> live_experiments = {
      {"exp1", 20}, {"exp2", 30}, {"exp3", 40},
      {"exp4", 60}, {"exp5", 70}, {"exp6", 80}};
  std::vector<tstring> optimizations_enabled = {"exp1", "exp4"},
                       optimizations_disabled = {"exp2", "exp5"},
                       optimizations_default = {"exp3", "exp6"};
  std::vector<tstring> optimizations = SelectOptimizations(
      job_name, live_experiments, optimizations_enabled, optimizations_disabled,
      optimizations_default, hash_func);

  int tested_times = 0;
  if (opt_outs.empty()) {
    EXPECT_THAT(optimizations,
                UnorderedElementsAre("exp1", "exp3", "exp4", "exp6"));
    tested_times++;
  } else if (opt_outs == "exp1,exp3") {
    EXPECT_THAT(optimizations, UnorderedElementsAre("exp1", "exp4", "exp6"));
    tested_times++;
  }
  EXPECT_EQ(tested_times, 1);

  if (!opt_ins.empty()) {
    unsetenv("TF_DATA_EXPERIMENT_OPT_IN");
  }
  if (!opt_outs.empty()) {
    unsetenv("TF_DATA_EXPERIMENT_OPT_OUT");
  }
}

INSTANTIATE_TEST_SUITE_P(Test, SelectOptimizationsConflictTest,
                         ::testing::Combine(::testing::Values("", "exp2"),
                                            ::testing::Values("", "exp1,exp3"),
                                            ::testing::Values(10, 50, 90)));

class SelectOptimizationsJobTest
    : public ::testing::TestWithParam<std::tuple<string, string, string>> {};

TEST_P(SelectOptimizationsJobTest, DatasetUtils) {
  const string job_name = std::get<0>(GetParam());
  const string opt_ins = std::get<1>(GetParam());
  const string opt_outs = std::get<2>(GetParam());
  if (!opt_ins.empty()) {
    setenv("TF_DATA_EXPERIMENT_OPT_IN", opt_ins.c_str(), 1);
  }
  if (!opt_outs.empty()) {
    setenv("TF_DATA_EXPERIMENT_OPT_OUT", opt_outs.c_str(), 1);
  }
  std::vector<tstring> optimizations_enabled = {"exp4"}, optimizations_disabled,
                       optimizations_default = {"exp2"};
  absl::flat_hash_map<string, uint64> live_experiments = {
      {"exp1", 0}, {"exp2", 100}, {"exp3", 100}};
  auto hash_func = [](const string& str) { return Hash64(str); };
  std::vector<tstring> optimizations = SelectOptimizations(
      job_name, live_experiments, optimizations_enabled, optimizations_disabled,
      optimizations_default, hash_func);

  int tested_times = 0;
  if (job_name.empty()) {
    EXPECT_THAT(optimizations, UnorderedElementsAre("exp2", "exp4"));
    tested_times++;
  } else if (opt_ins.empty()) {
    if (opt_outs.empty()) {
      EXPECT_THAT(optimizations, UnorderedElementsAre("exp2", "exp3", "exp4"));
      tested_times++;
    } else if (opt_outs == "exp2,exp3") {
      EXPECT_THAT(optimizations, UnorderedElementsAre("exp4"));
      tested_times++;
    }
  } else if (opt_ins == "exp1") {
    if (opt_outs.empty()) {
      EXPECT_THAT(optimizations,
                  UnorderedElementsAre("exp1", "exp2", "exp3", "exp4"));
      tested_times++;
    } else if (opt_outs == "exp2,exp3") {
      EXPECT_THAT(optimizations, UnorderedElementsAre("exp1", "exp4"));
      tested_times++;
    }
  }
  EXPECT_EQ(tested_times, 1);

  if (!opt_ins.empty()) {
    unsetenv("TF_DATA_EXPERIMENT_OPT_IN");
  }
  if (!opt_outs.empty()) {
    unsetenv("TF_DATA_EXPERIMENT_OPT_OUT");
  }
}

INSTANTIATE_TEST_SUITE_P(Test, SelectOptimizationsJobTest,
                         ::testing::Combine(::testing::Values("", "job"),
                                            ::testing::Values("", "exp1"),
                                            ::testing::Values("",
                                                              "exp2,exp3")));

}  // namespace
}  // namespace data
}  // namespace tensorflow
