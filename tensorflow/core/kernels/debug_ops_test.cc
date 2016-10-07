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
    ASSERT_TRUE(env_->FileExists(dump_roots[i]));
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

}  // namespace
}  // namespace tensorflow
