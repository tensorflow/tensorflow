/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include <dirent.h>
#include <string.h>
#include <fstream>
#include <vector>

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/event.pb.h"

namespace tensorflow {
namespace {

class DebugIdentityOpTest : public OpsTestBase {
 protected:
  Status Init(DataType input_type, const std::vector<string> debug_urls) {
    env_ = Env::Default();

    TF_CHECK_OK(NodeDefBuilder("op", "DebugIdentity")
                    .Input(FakeInput(input_type))
                    .Attr("tensor_name", "FakeTensor:0")
                    .Attr("debug_urls", debug_urls)
                    .Finalize(node_def()));
    return InitOp();
  }

  Status Init(DataType input_type) {
    std::vector<string> empty_debug_urls;
    return Init(input_type, empty_debug_urls);
  }

  Env* env_;
};

TEST_F(DebugIdentityOpTest, Int32Success_6) {
  TF_ASSERT_OK(Init(DT_INT32));
  AddInputFromArray<int32>(TensorShape({6}), {1, 2, 3, 4, 5, 6});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_INT32, TensorShape({6}));
  test::FillValues<int32>(&expected, {1, 2, 3, 4, 5, 6});
  // Verify the identity output
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));
}

TEST_F(DebugIdentityOpTest, Int32Success_6_FileURLs) {
  const int kNumDumpDirs = 3;

  const string tmp_dir = testing::TmpDir();

  std::vector<string> dump_roots;
  std::vector<string> debug_urls;
  for (int i = 0; i < kNumDumpDirs; ++i) {
    const string dump_root = strings::StrCat(tmp_dir, "_", i);
    dump_roots.push_back(dump_root);

    debug_urls.push_back(strings::StrCat("file://", dump_root));
  }

  uint64 wall_time = Env::Default()->NowMicros();

  TF_ASSERT_OK(Init(DT_INT32, debug_urls));
  AddInputFromArray<int32>(TensorShape({6}), {1, 2, 3, 4, 5, 6});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_INT32, TensorShape({6}));
  test::FillValues<int32>(&expected, {1, 2, 3, 4, 5, 6});
  // Verify the identity output
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));

  for (int i = 0; i < kNumDumpDirs; ++i) {
    ASSERT_TRUE(env_->FileExists(dump_roots[i]).ok());
    ASSERT_TRUE(env_->IsDirectory(dump_roots[i]).ok());

    DIR* dir = opendir(dump_roots[i].c_str());
    struct dirent* ent;
    int dump_files_found = 0;
    while ((ent = readdir(dir)) != NULL) {
      if (strcmp(ent->d_name, ".") && strcmp(ent->d_name, "..")) {
        dump_files_found++;

        // Try reading the file into a Event proto.
        const string dump_file_path =
            strings::StrCat(dump_roots[i], "/", ent->d_name);
        std::fstream ifs(dump_file_path, std::ios::in | std::ios::binary);
        Event event;
        event.ParseFromIstream(&ifs);
        ifs.close();

        ASSERT_GE(event.wall_time(), wall_time);
        ASSERT_EQ(1, event.summary().value().size());
        ASSERT_EQ(strings::StrCat("FakeTensor", ":", 0, ":", "DebugIdentity"),
                  event.summary().value(0).node_name());

        Tensor tensor_prime(DT_INT32);
        ASSERT_TRUE(tensor_prime.FromProto(event.summary().value(0).tensor()));

        // Verify tensor shape and value from the dump file.
        ASSERT_EQ(TensorShape({6}), tensor_prime.shape());

        for (int j = 0; j < 6; ++j) {
          ASSERT_EQ(j + 1, tensor_prime.flat<int32>()(j));
        }
      }
    }
    closedir(dir);

    ASSERT_EQ(1, dump_files_found);

    // Remove temporary dump directory and file.
    int64 undeleted_files = 0;
    int64 undeleted_dirs = 0;
    ASSERT_TRUE(env_->DeleteRecursively(dump_roots[i], &undeleted_files,
                                        &undeleted_dirs)
                    .ok());
    ASSERT_EQ(0, undeleted_files);
    ASSERT_EQ(0, undeleted_dirs);
  }
}

TEST_F(DebugIdentityOpTest, Int32Success_2_3) {
  TF_ASSERT_OK(Init(DT_INT32));
  AddInputFromArray<int32>(TensorShape({2, 3}), {1, 2, 3, 4, 5, 6});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_INT32, TensorShape({2, 3}));
  test::FillValues<int32>(&expected, {1, 2, 3, 4, 5, 6});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));
}

TEST_F(DebugIdentityOpTest, StringSuccess) {
  TF_ASSERT_OK(Init(DT_STRING));
  AddInputFromArray<string>(TensorShape({6}), {"A", "b", "C", "d", "E", "f"});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({6}));
  test::FillValues<string>(&expected, {"A", "b", "C", "d", "E", "f"});
  test::ExpectTensorEqual<string>(expected, *GetOutput(0));
}

// Tests for DebugNanCountOp
class DebugNanCountOpTest : public OpsTestBase {
 protected:
  Status Init(DataType input_type) {
    TF_CHECK_OK(NodeDefBuilder("op", "DebugNanCount")
                    .Input(FakeInput(input_type))
                    .Attr("tensor_name", "FakeTensor:0")
                    .Finalize(node_def()));
    return InitOp();
  }
};

TEST_F(DebugNanCountOpTest, Float_has_NaNs) {
  TF_ASSERT_OK(Init(DT_FLOAT));
  AddInputFromArray<float>(TensorShape({6}),
                           {1.1, std::numeric_limits<float>::quiet_NaN(), 3.3,
                            std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::quiet_NaN(), 6.6});
  TF_ASSERT_OK(RunOpKernel());

  // Verify the NaN-count debug signal
  Tensor expected_nan_count(allocator(), DT_INT64, TensorShape({1}));
  test::FillValues<int64>(&expected_nan_count, {3});
  test::ExpectTensorEqual<int64>(expected_nan_count, *GetOutput(0));
}

TEST_F(DebugNanCountOpTest, Float_no_NaNs) {
  TF_ASSERT_OK(Init(DT_FLOAT));
  AddInputFromArray<float>(
      TensorShape({6}),
      {1.1, 2.2, 3.3, std::numeric_limits<float>::infinity(), 5.5, 6.6});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_nan_count(allocator(), DT_INT64, TensorShape({1}));
  test::FillValues<int64>(&expected_nan_count, {0});
  test::ExpectTensorEqual<int64>(expected_nan_count, *GetOutput(0));
}

TEST_F(DebugNanCountOpTest, Double_has_NaNs) {
  TF_ASSERT_OK(Init(DT_DOUBLE));
  AddInputFromArray<double>(TensorShape({6}),
                            {1.1, std::numeric_limits<double>::quiet_NaN(), 3.3,
                             std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::quiet_NaN(), 6.6});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_nan_count(allocator(), DT_INT64, TensorShape({1}));
  test::FillValues<int64>(&expected_nan_count, {3});
  test::ExpectTensorEqual<int64>(expected_nan_count, *GetOutput(0));
}

TEST_F(DebugNanCountOpTest, Double_no_NaNs) {
  TF_ASSERT_OK(Init(DT_DOUBLE));
  AddInputFromArray<double>(
      TensorShape({6}),
      {1.1, 2.2, 3.3, std::numeric_limits<double>::infinity(), 5.5, 6.6});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_nan_count(allocator(), DT_INT64, TensorShape({1}));
  test::FillValues<int64>(&expected_nan_count, {0});
  test::ExpectTensorEqual<int64>(expected_nan_count, *GetOutput(0));
}

// Tests for DebugNumericSummaryOp
class DebugNumericSummaryOpTest : public OpsTestBase {
 protected:
  Status Init(DataType input_type) {
    TF_CHECK_OK(NodeDefBuilder("op", "DebugNumericSummary")
                    .Input(FakeInput(input_type))
                    .Attr("tensor_name", "FakeTensor:0")
                    .Finalize(node_def()));
    return InitOp();
  }
};

TEST_F(DebugNumericSummaryOpTest, Float_full_house) {
  TF_ASSERT_OK(Init(DT_FLOAT));
  AddInputFromArray<float>(
      TensorShape({18}),
      {std::numeric_limits<float>::quiet_NaN(),
       std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f, 0.0f, -1.0f, -3.0f,
       3.0f, 7.0f, -std::numeric_limits<float>::infinity(),
       -std::numeric_limits<float>::infinity(),
       std::numeric_limits<float>::infinity(),
       std::numeric_limits<float>::infinity(),
       std::numeric_limits<float>::infinity(),
       std::numeric_limits<float>::infinity(),
       std::numeric_limits<float>::infinity(),
       std::numeric_limits<float>::quiet_NaN(),
       std::numeric_limits<float>::quiet_NaN()});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_DOUBLE, TensorShape({12}));
  test::FillValues<double>(
      &expected,
      {1.0,              // Is initialized.
       18.0,             // Total element count.
       4.0,              // nan count.
       2.0,              // -inf count.
       2.0,              // negative number count (excluding -inf).
       3.0,              // zero count.
       2.0,              // positive number count (excluding +inf).
       5.0,              // +inf count.
       -3.0,             // minimum of non-inf and non-nan elements.
       7.0,              // maximum of non-inf and non-nan elements.
       0.85714285714,    // mean of non-inf and non-nan elements.
       8.97959183673});  // variance of non-inf and non-nan elements.

  test::ExpectTensorNear<double>(expected, *GetOutput(0), 1e-8);
}

TEST_F(DebugNumericSummaryOpTest, Double_full_house) {
  TF_ASSERT_OK(Init(DT_DOUBLE));
  AddInputFromArray<double>(
      TensorShape({18}),
      {std::numeric_limits<double>::quiet_NaN(),
       std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0, -1.0, -3.0, 3.0,
       7.0, -std::numeric_limits<double>::infinity(),
       -std::numeric_limits<double>::infinity(),
       std::numeric_limits<double>::infinity(),
       std::numeric_limits<double>::infinity(),
       std::numeric_limits<double>::infinity(),
       std::numeric_limits<double>::infinity(),
       std::numeric_limits<double>::infinity(),
       std::numeric_limits<double>::quiet_NaN(),
       std::numeric_limits<double>::quiet_NaN()});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_DOUBLE, TensorShape({12}));
  test::FillValues<double>(
      &expected,
      {1.0,              // Is initialized.
       18.0,             // Total element count.
       4.0,              // nan count.
       2.0,              // -inf count.
       2.0,              // negative count (excluding -inf).
       3.0,              // zero count.
       2.0,              // positive count (excluding +inf).
       5.0,              // +inf count.
       -3.0,             // minimum of non-inf and non-nan elements.
       7.0,              // maximum of non-inf and non-nan elements.
       0.85714285714,    // mean of non-inf and non-nan elements.
       8.97959183673});  // variance of non-inf and non-nan elements.

  test::ExpectTensorNear<double>(expected, *GetOutput(0), 1e-8);
}

TEST_F(DebugNumericSummaryOpTest, Float_only_valid_values) {
  TF_ASSERT_OK(Init(DT_FLOAT));
  AddInputFromArray<float>(TensorShape({2, 3}),
                           {0.0f, 0.0f, -1.0f, 3.0f, 3.0f, 7.0f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_DOUBLE, TensorShape({12}));
  test::FillValues<double>(
      &expected,
      {1.0,              // Is initialized.
       6.0,              // Total element count.
       0.0,              // nan count.
       0.0,              // -inf count.
       1.0,              // negative count (excluding -inf).
       2.0,              // zero count.
       3.0,              // positive count (excluding +inf).
       0.0,              // +inf count.
       -1.0,             // minimum of non-inf and non-nan elements.
       7.0,              // maximum of non-inf and non-nan elements.
       2.0,              // mean of non-inf and non-nan elements.
       7.33333333333});  // variance of non-inf and non-nan elements.

  test::ExpectTensorNear<double>(expected, *GetOutput(0), 1e-8);
}

TEST_F(DebugNumericSummaryOpTest, Float_all_Inf_or_NaN) {
  TF_ASSERT_OK(Init(DT_FLOAT));
  AddInputFromArray<float>(TensorShape({3, 3}),
                           {std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::quiet_NaN(),
                            -std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::quiet_NaN()});
  TF_ASSERT_OK(RunOpKernel());

  Tensor output_tensor = *GetOutput(0);
  const double* output = output_tensor.template flat<double>().data();

  Tensor expected(allocator(), DT_DOUBLE, TensorShape({12}));
  // Use ASSERT_NEAR below because test::ExpectTensorNear does not work with
  // NaNs.
  ASSERT_NEAR(1.0, output[0], 1e-8);  // Is initialized.
  ASSERT_NEAR(9.0, output[1], 1e-8);  // Total element count.
  ASSERT_NEAR(4.0, output[2], 1e-8);  // nan count.
  ASSERT_NEAR(2.0, output[3], 1e-8);  // -inf count.
  ASSERT_NEAR(0.0, output[4], 1e-8);  // negative count (excluding -inf).
  ASSERT_NEAR(0.0, output[5], 1e-8);  // zero count.
  ASSERT_NEAR(0.0, output[6], 1e-8);  // positive count (excluding +inf).
  ASSERT_NEAR(3.0, output[7], 1e-8);  // +inf count.
  // Due to the absence of any non-inf and non-nan values, the output of min,
  // max, mean and var are all degenerate.
  ASSERT_EQ(std::numeric_limits<float>::infinity(), output[8]);
  ASSERT_EQ(-std::numeric_limits<float>::infinity(), output[9]);
  ASSERT_TRUE(Eigen::numext::isnan(output[10]));
  ASSERT_TRUE(Eigen::numext::isnan(output[11]));
}

TEST_F(DebugNumericSummaryOpTest, Int16Success) {
  TF_ASSERT_OK(Init(DT_INT16));
  AddInputFromArray<int16>(TensorShape({4, 1}), {-1, -3, 3, 7});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_DOUBLE, TensorShape({12}));
  test::FillValues<double>(
      &expected,
      {1.0,      // Is initialized.
       4.0,      // Total element count.
       0.0,      // nan count.
       0.0,      // -inf count.
       2.0,      // negative count (excluding -inf).
       0.0,      // zero count.
       2.0,      // positive count (excluding +inf).
       0.0,      // +inf count.
       -3.0,     // minimum of non-inf and non-nan elements.
       7.0,      // maximum of non-inf and non-nan elements.
       1.5,      // mean of non-inf and non-nan elements.
       14.75});  // variance of non-inf and non-nan elements.

  test::ExpectTensorNear<double>(expected, *GetOutput(0), 1e-8);
}

TEST_F(DebugNumericSummaryOpTest, Int32Success) {
  TF_ASSERT_OK(Init(DT_INT32));
  AddInputFromArray<int32>(TensorShape({2, 3}), {0, 0, -1, 3, 3, 7});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_DOUBLE, TensorShape({12}));
  test::FillValues<double>(
      &expected,
      {1.0,              // Is initialized.
       6.0,              // Total element count.
       0.0,              // nan count.
       0.0,              // -inf count.
       1.0,              // negative count (excluding -inf).
       2.0,              // zero count.
       3.0,              // positive count (excluding +inf).
       0.0,              // +inf count.
       -1.0,             // minimum of non-inf and non-nan elements.
       7.0,              // maximum of non-inf and non-nan elements.
       2.0,              // mean of non-inf and non-nan elements.
       7.33333333333});  // variance of non-inf and non-nan elements.

  test::ExpectTensorNear<double>(expected, *GetOutput(0), 1e-8);
}

TEST_F(DebugNumericSummaryOpTest, Int64Success) {
  TF_ASSERT_OK(Init(DT_INT64));
  AddInputFromArray<int64>(TensorShape({2, 2, 2}), {0, 0, -1, 3, 3, 7, 0, 0});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_DOUBLE, TensorShape({12}));
  test::FillValues<double>(
      &expected,
      {1.0,     // Is initialized.
       8.0,     // Total element count.
       0.0,     // nan count.
       0.0,     // -inf count.
       1.0,     // negative count (excluding -inf).
       4.0,     // zero count.
       3.0,     // positive count (excluding +inf).
       0.0,     // +inf count.
       -1.0,    // minimum of non-inf and non-nan elements.
       7.0,     // maximum of non-inf and non-nan elements.
       1.5,     // mean of non-inf and non-nan elements.
       6.25});  // variance of non-inf and non-nan elements.

  test::ExpectTensorNear<double>(expected, *GetOutput(0), 1e-8);
}

TEST_F(DebugNumericSummaryOpTest, UInt8Success) {
  TF_ASSERT_OK(Init(DT_UINT8));
  AddInputFromArray<uint8>(TensorShape({1, 5}), {0, 10, 30, 30, 70});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_DOUBLE, TensorShape({12}));
  test::FillValues<double>(
      &expected,
      {1.0,      // Is initialized.
       5.0,      // Total element count.
       0.0,      // nan count.
       0.0,      // -inf count.
       0.0,      // negative count (excluding -inf).
       1.0,      // zero count.
       4.0,      // positive count (excluding +inf).
       0.0,      // +inf count.
       0.0,      // minimum of non-inf and non-nan elements.
       70.0,     // maximum of non-inf and non-nan elements.
       28.0,     // mean of non-inf and non-nan elements.
       576.0});  // variance of non-inf and non-nan elements.

  test::ExpectTensorNear<double>(expected, *GetOutput(0), 1e-8);
}

TEST_F(DebugNumericSummaryOpTest, BoolSuccess) {
  TF_ASSERT_OK(Init(DT_BOOL));
  AddInputFromArray<bool>(TensorShape({2, 3}), {0, 0, 1, 1, 1, 0});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_DOUBLE, TensorShape({12}));
  test::FillValues<double>(
      &expected,
      {1.0,     // Is initialized.
       6.0,     // Total element count.
       0.0,     // nan count.
       0.0,     // -inf count.
       0.0,     // negative count (excluding -inf).
       3.0,     // zero count.
       3.0,     // positive count (excluding +inf).
       0.0,     // +inf count.
       0.0,     // minimum of non-inf and non-nan elements.
       1.0,     // maximum of non-inf and non-nan elements.
       0.5,     // mean of non-inf and non-nan elements.
       0.25});  // variance of non-inf and non-nan elements.

  test::ExpectTensorNear<double>(expected, *GetOutput(0), 1e-8);
}

}  // namespace
}  // namespace tensorflow
