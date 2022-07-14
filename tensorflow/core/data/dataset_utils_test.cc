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

#include "tensorflow/core/data/dataset_utils.h"

#include <functional>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/data/serialization_utils.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/util/determinism_test_util.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace data {
namespace {

TEST(DatasetUtilsTest, MatchesAnyVersion) {
  EXPECT_TRUE(MatchesAnyVersion("BatchDataset", "BatchDataset"));
  EXPECT_TRUE(MatchesAnyVersion("BatchDataset", "BatchDatasetV2"));
  EXPECT_TRUE(MatchesAnyVersion("BatchDataset", "BatchDatasetV3"));
  EXPECT_FALSE(MatchesAnyVersion("BatchDataset", "BatchDatasetXV3"));
  EXPECT_FALSE(MatchesAnyVersion("BatchDataset", "BatchV2Dataset"));
  EXPECT_FALSE(MatchesAnyVersion("BatchDataset", "PaddedBatchDataset"));
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

REGISTER_DATASET_EXPERIMENT("test_only_experiment_0", 0);
REGISTER_DATASET_EXPERIMENT("test_only_experiment_1", 1);
REGISTER_DATASET_EXPERIMENT("test_only_experiment_5", 5);
REGISTER_DATASET_EXPERIMENT("test_only_experiment_10", 10);
REGISTER_DATASET_EXPERIMENT("test_only_experiment_50", 50);
REGISTER_DATASET_EXPERIMENT("test_only_experiment_99", 99);
REGISTER_DATASET_EXPERIMENT("test_only_experiment_100", 100);

struct GetExperimentsHashTestCase {
  uint64 hash;
  std::vector<string> expected_in;
  std::vector<string> expected_out;
};

class GetExperimentsHashTest
    : public ::testing::TestWithParam<GetExperimentsHashTestCase> {};

TEST_P(GetExperimentsHashTest, DatasetUtils) {
  const GetExperimentsHashTestCase test_case = GetParam();
  uint64 hash_result = test_case.hash;
  auto job_name = "job";
  auto hash_func = [hash_result](const string& str) { return hash_result; };
  auto experiments = GetExperiments(job_name, hash_func);

  absl::flat_hash_set<string> experiment_set(experiments.begin(),
                                             experiments.end());
  for (const auto& experiment : test_case.expected_in) {
    EXPECT_TRUE(experiment_set.find(experiment) != experiment_set.end())
        << "experiment=" << experiment << " hash=" << hash_result;
  }
  for (const auto& experiment : test_case.expected_out) {
    EXPECT_TRUE(experiment_set.find(experiment) == experiment_set.end())
        << "experiment=" << experiment << " hash=" << hash_result;
  }
}

INSTANTIATE_TEST_SUITE_P(
    Test, GetExperimentsHashTest,
    ::testing::Values<GetExperimentsHashTestCase>(
        GetExperimentsHashTestCase{
            /*hash=*/0,
            /*expected_in=*/
            {"test_only_experiment_1", "test_only_experiment_5",
             "test_only_experiment_10", "test_only_experiment_50",
             "test_only_experiment_99", "test_only_experiment_100"},
            /*expected_out=*/{"test_only_experiment_0"},
        },
        GetExperimentsHashTestCase{
            /*hash=*/5,
            /*expected_in=*/
            {"test_only_experiment_10", "test_only_experiment_50",
             "test_only_experiment_99", "test_only_experiment_100"},
            /*expected_out=*/
            {
                "test_only_experiment_0",
                "test_only_experiment_1",
                "test_only_experiment_5",
            },
        },
        GetExperimentsHashTestCase{
            /*hash=*/95,
            /*expected_in=*/
            {"test_only_experiment_99", "test_only_experiment_100"},
            /*expected_out=*/
            {"test_only_experiment_0", "test_only_experiment_1",
             "test_only_experiment_5", "test_only_experiment_10",
             "test_only_experiment_50"},
        },
        GetExperimentsHashTestCase{
            /*hash=*/99,
            /*expected_in=*/{"test_only_experiment_100"},
            /*expected_out=*/
            {"test_only_experiment_0", "test_only_experiment_1",
             "test_only_experiment_5", "test_only_experiment_10",
             "test_only_experiment_50", "test_only_experiment_99"},
        },
        GetExperimentsHashTestCase{
            /*hash=*/100,
            /*expected_in=*/
            {"test_only_experiment_1", "test_only_experiment_5",
             "test_only_experiment_10", "test_only_experiment_50",
             "test_only_experiment_99", "test_only_experiment_100"},
            /*expected_out=*/{"test_only_experiment_0"},
        },
        GetExperimentsHashTestCase{
            /*hash=*/105,
            /*expected_in=*/
            {"test_only_experiment_10", "test_only_experiment_50",
             "test_only_experiment_99", "test_only_experiment_100"},
            /*expected_out=*/
            {
                "test_only_experiment_0",
                "test_only_experiment_1",
                "test_only_experiment_5",
            },
        },
        GetExperimentsHashTestCase{
            /*hash=*/195,
            /*expected_in=*/
            {"test_only_experiment_99", "test_only_experiment_100"},
            /*expected_out=*/
            {"test_only_experiment_0", "test_only_experiment_1",
             "test_only_experiment_5", "test_only_experiment_10",
             "test_only_experiment_50"},
        }));

struct GetExperimentsOptTestCase {
  std::vector<string> opt_ins;
  std::vector<string> opt_outs;
  std::vector<string> expected_in;
  std::vector<string> expected_out;
};

class GetExperimentsOptTest
    : public ::testing::TestWithParam<GetExperimentsOptTestCase> {};

TEST_P(GetExperimentsOptTest, DatasetUtils) {
  const GetExperimentsOptTestCase test_case = GetParam();
  auto opt_ins = test_case.opt_ins;
  auto opt_outs = test_case.opt_outs;
  if (!opt_ins.empty()) {
    setenv("TF_DATA_EXPERIMENT_OPT_IN", str_util::Join(opt_ins, ",").c_str(),
           1);
  }
  if (!opt_outs.empty()) {
    setenv("TF_DATA_EXPERIMENT_OPT_OUT", str_util::Join(opt_outs, ",").c_str(),
           1);
  }
  auto job_name = "job";
  auto hash_func = [](const string& str) { return 0; };
  auto experiments = GetExperiments(job_name, hash_func);

  absl::flat_hash_set<string> experiment_set(experiments.begin(),
                                             experiments.end());
  for (const auto& experiment : test_case.expected_in) {
    EXPECT_TRUE(experiment_set.find(experiment) != experiment_set.end())
        << "experiment=" << experiment << " opt_ins={"
        << str_util::Join(opt_ins, ",") << "} opt_outs={"
        << str_util::Join(opt_outs, ",") << "}";
  }
  for (const auto& experiment : test_case.expected_out) {
    EXPECT_TRUE(experiment_set.find(experiment) == experiment_set.end())
        << "experiment=" << experiment << " opt_ins={"
        << str_util::Join(opt_ins, ",") << "} opt_outs={"
        << str_util::Join(opt_outs, ",") << "}";
  }

  if (!opt_ins.empty()) {
    unsetenv("TF_DATA_EXPERIMENT_OPT_IN");
  }
  if (!opt_outs.empty()) {
    unsetenv("TF_DATA_EXPERIMENT_OPT_OUT");
  }
}

INSTANTIATE_TEST_SUITE_P(
    Test, GetExperimentsOptTest,
    ::testing::Values<GetExperimentsOptTestCase>(
        GetExperimentsOptTestCase{
            /*opt_ins=*/{"all"},
            /*opt_outs=*/{"all"},
            /*expected_in=*/{},
            /*expected_out=*/
            {"test_only_experiment_0", "test_only_experiment_1",
             "test_only_experiment_5", "test_only_experiment_10",
             "test_only_experiment_50", "test_only_experiment_99",
             "test_only_experiment_100"}},
        GetExperimentsOptTestCase{
            /*opt_ins=*/{"all"},
            /*opt_outs=*/{},
            /*expected_in=*/
            {"test_only_experiment_0", "test_only_experiment_1",
             "test_only_experiment_5", "test_only_experiment_10",
             "test_only_experiment_50", "test_only_experiment_99",
             "test_only_experiment_100"},
            /*expected_out=*/{}},
        GetExperimentsOptTestCase{
            /*opt_ins=*/{"all"},
            /*opt_outs=*/{"test_only_experiment_1", "test_only_experiment_99"},
            /*expected_in=*/
            {"test_only_experiment_0", "test_only_experiment_5",
             "test_only_experiment_10", "test_only_experiment_50",
             "test_only_experiment_100"},
            /*expected_out=*/
            {"test_only_experiment_1", "test_only_experiment_99"}},
        GetExperimentsOptTestCase{
            /*opt_ins=*/{},
            /*opt_outs=*/{"all"},
            /*expected_in=*/{},
            /*expected_out=*/
            {"test_only_experiment_0", "test_only_experiment_1",
             "test_only_experiment_5", "test_only_experiment_10",
             "test_only_experiment_50", "test_only_experiment_99",
             "test_only_experiment_100"}},
        GetExperimentsOptTestCase{
            /*opt_ins=*/{},
            /*opt_outs=*/{},
            /*expected_in=*/
            {"test_only_experiment_1", "test_only_experiment_5",
             "test_only_experiment_10", "test_only_experiment_50",
             "test_only_experiment_99", "test_only_experiment_100"},
            /*expected_out=*/{"test_only_experiment_0"}},
        GetExperimentsOptTestCase{
            /*opt_ins=*/{},
            /*opt_outs=*/{"test_only_experiment_1", "test_only_experiment_99"},
            /*expected_in=*/
            {"test_only_experiment_5", "test_only_experiment_10",
             "test_only_experiment_50", "test_only_experiment_100"},
            /*expected_out=*/
            {"test_only_experiment_0", "test_only_experiment_1",
             "test_only_experiment_99"}},
        GetExperimentsOptTestCase{
            /*opt_ins=*/{"test_only_experiment_0", "test_only_experiment_100"},
            /*opt_outs=*/{"all"},
            /*expected_in=*/{},
            /*expected_out=*/
            {"test_only_experiment_0", "test_only_experiment_1",
             "test_only_experiment_5", "test_only_experiment_10",
             "test_only_experiment_50", "test_only_experiment_99",
             "test_only_experiment_100"}},
        GetExperimentsOptTestCase{
            /*opt_ins=*/{"test_only_experiment_0", "test_only_experiment_100"},
            /*opt_outs=*/{"all_except_opt_in"},
            /*expected_in=*/
            {"test_only_experiment_0", "test_only_experiment_100"},
            /*expected_out=*/
            {"test_only_experiment_1", "test_only_experiment_5",
             "test_only_experiment_10", "test_only_experiment_50",
             "test_only_experiment_99"}},
        GetExperimentsOptTestCase{
            /*opt_ins=*/{"test_only_experiment_0", "test_only_experiment_100"},
            /*opt_outs=*/{},
            /*expected_in=*/
            {"test_only_experiment_0", "test_only_experiment_1",
             "test_only_experiment_5", "test_only_experiment_10",
             "test_only_experiment_50", "test_only_experiment_99",
             "test_only_experiment_100"},
            /*expected_out=*/{}},
        GetExperimentsOptTestCase{
            /*opt_ins=*/{"test_only_experiment_0", "test_only_experiment_100"},
            /*opt_outs=*/{"test_only_experiment_1", "test_only_experiment_99"},
            /*expected_in=*/
            {"test_only_experiment_0", "test_only_experiment_5",
             "test_only_experiment_10", "test_only_experiment_50",
             "test_only_experiment_100"},
            /*expected_out=*/
            {"test_only_experiment_1", "test_only_experiment_99"}}));

struct GetExperimentsJobNameTestCase {
  string job_name;
  std::vector<string> expected_in;
  std::vector<string> expected_out;
};

class GetExperimentsJobNameTest
    : public ::testing::TestWithParam<GetExperimentsJobNameTestCase> {};

TEST_P(GetExperimentsJobNameTest, DatasetUtils) {
  const GetExperimentsJobNameTestCase test_case = GetParam();
  auto job_name = test_case.job_name;
  auto hash_func = [](const string& str) { return 0; };
  auto experiments = GetExperiments(job_name, hash_func);

  absl::flat_hash_set<string> experiment_set(experiments.begin(),
                                             experiments.end());
  for (const auto& experiment : test_case.expected_in) {
    EXPECT_TRUE(experiment_set.find(experiment) != experiment_set.end())
        << "experiment=" << experiment << " job_name=" << job_name;
  }
  for (const auto& experiment : test_case.expected_out) {
    EXPECT_TRUE(experiment_set.find(experiment) == experiment_set.end())
        << "experiment=" << experiment << " job_name=" << job_name;
  }
}

INSTANTIATE_TEST_SUITE_P(
    Test, GetExperimentsJobNameTest,
    ::testing::Values(GetExperimentsJobNameTestCase{
                          /*job_name=*/"",
                          /*expected_in=*/{},
                          /*expected_out=*/
                          {"test_only_experiment_0", "test_only_experiment_1",
                           "test_only_experiment_5", "test_only_experiment_10",
                           "test_only_experiment_50", "test_only_experiment_99",
                           "test_only_experiment_100"}},
                      GetExperimentsJobNameTestCase{
                          /*job_name=*/"job_name",
                          /*expected_in=*/
                          {"test_only_experiment_1", "test_only_experiment_5",
                           "test_only_experiment_10", "test_only_experiment_50",
                           "test_only_experiment_99",
                           "test_only_experiment_100"},
                          /*expected_out=*/{"test_only_experiment_0"}}));

struct GetOptimizationsTestCase {
  Options options;
  std::vector<string> expected_enabled;
  std::vector<string> expected_disabled;
  std::vector<string> expected_default;
};

// Tests the default.
GetOptimizationsTestCase GetOptimizationTestCase1() {
  return {
      /*options=*/Options(),
      /*expected_enabled=*/{},
      /*expected_disabled=*/{},
      /*expected_default=*/
      {"noop_elimination", "map_and_batch_fusion", "shuffle_and_repeat_fusion",
       "map_parallelization", "parallel_batch"}};
}

// Tests disabling application of default optimizations.
GetOptimizationsTestCase GetOptimizationTestCase2() {
  Options options;
  options.mutable_optimization_options()->set_apply_default_optimizations(
      false);
  return {options, /*expected_enabled=*/{}, /*expected_disabled=*/{},
          /*expected_default=*/{}};
}

// Tests explicitly enabling / disabling some default and non-default
// optimizations.
GetOptimizationsTestCase GetOptimizationTestCase3() {
  Options options;
  options.set_deterministic(false);
  options.mutable_optimization_options()->set_map_and_batch_fusion(true);
  options.mutable_optimization_options()->set_map_parallelization(false);
  options.mutable_optimization_options()->set_parallel_batch(false);
  return {options,
          /*expected_enabled=*/{"make_sloppy", "map_and_batch_fusion"},
          /*expected_disabled=*/{"parallel_batch", "map_parallelization"},
          /*expected_default=*/
          {"noop_elimination", "shuffle_and_repeat_fusion"}};
}

// Test enabling all / most available optimizations.
GetOptimizationsTestCase GetOptimizationTestCase4() {
  Options options;
  options.set_deterministic(false);
  options.mutable_optimization_options()->set_filter_fusion(true);
  options.mutable_optimization_options()->set_filter_parallelization(true);
  options.mutable_optimization_options()->set_map_and_batch_fusion(true);
  options.mutable_optimization_options()->set_map_and_filter_fusion(true);
  options.mutable_optimization_options()->set_map_fusion(true);
  options.mutable_optimization_options()->set_map_parallelization(true);
  options.mutable_optimization_options()->set_noop_elimination(true);
  options.mutable_optimization_options()->set_parallel_batch(true);
  options.mutable_optimization_options()->set_shuffle_and_repeat_fusion(true);
  options.mutable_optimization_options()->set_inject_prefetch(true);
  options.set_slack(true);
  return {options,
          /*expected_enabled=*/
          {"filter_fusion", "filter_parallelization", "make_sloppy",
           "map_and_batch_fusion", "map_and_filter_fusion", "map_fusion",
           "map_parallelization", "noop_elimination", "parallel_batch",
           "shuffle_and_repeat_fusion", "slack", "inject_prefetch"},
          /*expected_disabled=*/{},
          /*expected_default=*/{}};
}

class GetOptimizationsTest
    : public ::testing::TestWithParam<GetOptimizationsTestCase> {};

TEST_P(GetOptimizationsTest, DatasetUtils) {
  const GetOptimizationsTestCase test_case = GetParam();
  auto options = test_case.options;

  absl::flat_hash_set<tstring> actual_enabled, actual_disabled, actual_default;
  GetOptimizations(options, &actual_enabled, &actual_disabled, &actual_default);

  EXPECT_THAT(std::vector<string>(actual_enabled.begin(), actual_enabled.end()),
              ::testing::UnorderedElementsAreArray(test_case.expected_enabled));
  EXPECT_THAT(
      std::vector<string>(actual_disabled.begin(), actual_disabled.end()),
      ::testing::UnorderedElementsAreArray(test_case.expected_disabled));
  EXPECT_THAT(std::vector<string>(actual_default.begin(), actual_default.end()),
              ::testing::UnorderedElementsAreArray(test_case.expected_default));
}

INSTANTIATE_TEST_SUITE_P(Test, GetOptimizationsTest,
                         ::testing::Values(GetOptimizationTestCase1(),
                                           GetOptimizationTestCase2(),
                                           GetOptimizationTestCase3(),
                                           GetOptimizationTestCase4()));

TEST(DeterministicOpsTest, GetOptimizations) {
  test::DeterministicOpsScope det_scope;
  Options options;
  // options.deterministic should be ignored when deterministic ops are enabled.
  options.set_deterministic(false);
  absl::flat_hash_set<tstring> actual_enabled, actual_disabled, actual_default;
  GetOptimizations(options, &actual_enabled, &actual_disabled, &actual_default);
  EXPECT_THAT(std::vector<string>(actual_enabled.begin(), actual_enabled.end()),
              ::testing::UnorderedElementsAreArray({"make_deterministic"}));
  EXPECT_EQ(actual_disabled.size(), 0);
}

REGISTER_DATASET_EXPERIMENT("test_only_experiment", 42);

TEST(DatasetUtilsTest, DatasetExperimentRegistry) {
  auto experiments = DatasetExperimentRegistry::Experiments();
  EXPECT_TRUE(experiments.find("test_only_experiment") != experiments.end());
  EXPECT_TRUE(experiments.find("non_existing_experiment") == experiments.end());
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
