/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <gtest/gtest.h>
#include "re2/re2.h"
#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/testing/parse_testdata.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/subprocess.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace {
bool FLAGS_ignore_known_bugs = true;
}  // namespace

namespace tflite {
namespace testing {

// TensorFlow system environment for file system called.
tensorflow::Env* env = tensorflow::Env::Default();

// List of tests that are expected to fail when
//   --test_arg=--ignore_known_bugs=false
// Key is a substring of the test name and value is a bug number.
// TODO(ahentz): make sure we clean this list up frequently.
std::map<string, string> kBrokenTests = {
    // Add doesn't support broadcasting.
    {R"(addd.*input_shape_1=\[1,3,4,3\],input_shape_2=\[3\])", "68500195"},
    {R"(muld.*input_shape_1=\[1,3,4,3\],input_shape_2=\[3\])", "68500195"},

    // Add only supports float32. (and "constant" tests use Add)
    {R"(addd.*int32)", "68808744"},
    {R"(constant.*int32)", "68808744"},
    {R"(mul.*int32)", "68808744"},

    // Toco or TFLite has a bug to deal with some constant functions with
    // more than 1 element.
    {R"(constant.*input_shape=\[(2|2,2,2,2)\])", "68721522"},

    // Pad only supports 4D float32 tensors.
    {R"(paddtype=.*,input_shape=\[.,.\],paddings=\[\[.,.\],\[.,.\]\])",
     "70527055"},
    {R"(padd.*int32)", "70527055"},

    // L2Norm only supports 4D tensors.
    {R"(l2normdim=.*,epsilon=.*,input_shape=\[.,.\])", "67963684"},
    {R"(l2normdim=.*,epsilon=.*,input_shape=\[.,.,.,.,.*\])", "67963684"},

    // SpaceToBatch only supports 4D tensors.
    {R"(space_to_batch_nd.*input_shape=\[1,4,4,4,1,1\])", "70848787"},

    // L2Norm only works for dim=-1.
    {R"(l2normdim=-2,epsilon=.*,input_shape=\[3,15,14,3\])", "67963812"},
    {R"(l2normdim=-2,epsilon=.*,input_shape=\[1,3,4,3\])", "67963812"},
    {R"(l2normdim=2,epsilon=.*,input_shape=\[3,15,14,3\])", "67963812"},
    {R"(l2normdim=2,epsilon=.*,input_shape=\[1,3,4,3\])", "67963812"},
    {R"(l2normdim=0,epsilon=.*,input_shape=\[3,15,14,3\])", "67963812"},
    {R"(l2normdim=0,epsilon=.*,input_shape=\[1,3,4,3\])", "67963812"},
    {R"(l2normdim=1,epsilon=.*,input_shape=\[3,15,14,3\])", "67963812"},
    {R"(l2normdim=1,epsilon=.*,input_shape=\[1,3,4,3\])", "67963812"},
    {R"(l2normdim=\[2,3\],epsilon=.*,input_shape=\[3,15,14,3\])", "67963812"},
    {R"(l2normdim=\[2,3\],epsilon=.*,input_shape=\[1,3,4,3\])", "67963812"},

    // ResizeBilinear looks completely incompatible with Tensorflow
    {R"(resize_bilinear)", "67964336"},

    // Transpose only supports 1D-4D input tensors.
    {R"(transposedtype=.*,input_shape=\[.,.,.,.,.\],perm=.*)", "71545879"},
};

// Allows test data to be unzipped into a temporary directory and makes
// sure those temporary directories are removed later.
class ZipEnvironment : public ::testing::Environment {
 public:
  ~ZipEnvironment() override {}

  // Delete all temporary directories on teardown.
  void TearDown() override {
    for (const auto& dir : temporary_directories_) {
      tensorflow::int64 undeleted_dirs, undeleted_files;
      TF_CHECK_OK(
          env->DeleteRecursively(dir, &undeleted_dirs, &undeleted_files));
    }
    temporary_directories_.clear();
  }

  // Unzip `zip` file into a new temporary directory  `out_dir`.
  tensorflow::Status UnZip(const string& zip, string* out_dir) {
    string dir;
    TF_CHECK_OK(MakeTemporaryDirectory(&dir));
    tensorflow::SubProcess proc;
    string unzip_binary =
        "/usr/bin/unzip";
    proc.SetProgram(unzip_binary, {"unzip", "-d", dir, zip});
    proc.SetChannelAction(tensorflow::CHAN_STDOUT, tensorflow::ACTION_PIPE);
    proc.SetChannelAction(tensorflow::CHAN_STDERR, tensorflow::ACTION_PIPE);
    if (!proc.Start())
      return tensorflow::Status(tensorflow::error::UNKNOWN,
                                "unzip couldn't start");
    string out, err;
    int status = proc.Communicate(nullptr, &out, &err);
    if (WEXITSTATUS(status) == 0) {
      *out_dir = dir;
      return tensorflow::Status::OK();
    } else {
      return tensorflow::Status(tensorflow::error::UNKNOWN, "unzip failed");
    }
  }

 private:
  // Make a temporary directory and return its name in `temporary`.
  tensorflow::Status MakeTemporaryDirectory(string* temporary) {
    if (env->LocalTempFilename(temporary)) {
      TF_CHECK_OK(env->CreateDir(*temporary));
      temporary_directories_.push_back(*temporary);
      return tensorflow::Status::OK();
    }
    return tensorflow::Status(tensorflow::error::UNKNOWN,
                              "make temporary directory failed");
  }

  std::vector<string> temporary_directories_;
};

// Return the singleton zip_environment.
ZipEnvironment* zip_environment() {
  static ZipEnvironment* env = new ZipEnvironment;
  return env;
}

// Read the manifest.txt out of the unarchived zip file. Specifically
// `original_file` is the original zip file for error messages. `dir` is
// the temporary directory where the zip file has been unarchived and
// `test_paths` is the list of test prefixes that were in the manifest.
// Note, it is an error for a manifest to contain no tests.
tensorflow::Status ReadManifest(const string& original_file, const string& dir,
                                std::vector<string>* test_paths) {
  // Read the newline delimited list of entries in the manifest.
  std::ifstream manifest_fp(dir + "/manifest.txt");
  string manifest((std::istreambuf_iterator<char>(manifest_fp)),
                  std::istreambuf_iterator<char>());
  size_t pos = 0;
  int added = 0;
  while (true) {
    size_t end_pos = manifest.find("\n", pos);
    if (end_pos == string::npos) break;
    string filename = manifest.substr(pos, end_pos - pos);
    test_paths->push_back(dir + "/" + filename);
    pos = end_pos + 1;
    added += 1;
  }
  if (!added) {
    string message = "Test had no examples: " + original_file;
    return tensorflow::Status(tensorflow::error::UNKNOWN, message.c_str());
  }
  return tensorflow::Status::OK();
}

// Get a list of tests from a zip file `zip_file_name`.
std::vector<string> UnarchiveZipAndFindTestNames(const string& zip_file_name) {
  string zip_file = ::tensorflow::testing::TensorFlowSrcRoot() +
                    "/contrib/lite/testing/optest/" + zip_file_name;
  string decompress_tmp_dir;
  TF_CHECK_OK(zip_environment()->UnZip(zip_file, &decompress_tmp_dir));
  std::vector<string> stuff;
  TF_CHECK_OK(ReadManifest(zip_file, decompress_tmp_dir, &stuff));
  return stuff;
}

class OpsTest : public ::testing::TestWithParam<string> {};

TEST_P(OpsTest, RunStuff) {
  string test_path = GetParam();
  string tflite_file = test_path + ".bin";
  string tflite_examples = test_path + ".inputs";
  string test_name = test_path.substr(test_path.find_last_of('/'));

  auto model = tflite::FlatBufferModel::BuildFromFile(tflite_file.c_str());
  std::unique_ptr<tflite::Interpreter> interpreter;

  tflite::ops::builtin::BuiltinOpResolver builtins;
  ASSERT_EQ(tflite::InterpreterBuilder(*model, builtins)(&interpreter),
            kTfLiteOk);

  std::vector<tflite::testing::Example> examples;
  ASSERT_EQ(tflite::testing::ParseExamples(tflite_examples.c_str(), &examples),
            kTfLiteOk);

  string bug_number;
  for (const auto& p : kBrokenTests) {
    if (RE2::PartialMatch(test_name, p.first)) {
      bug_number = p.second;
    }
  }

  for (const auto& example : examples) {
    ASSERT_EQ(interpreter->inputs().size(), example.inputs.size());
    auto result = [&]() {
      TF_LITE_ENSURE_STATUS(FeedExample(interpreter.get(), example));
      TF_LITE_ENSURE_STATUS(interpreter->Invoke());
      TF_LITE_ENSURE_STATUS(CheckOutputs(interpreter.get(), example));
      return kTfLiteOk;
    }();

    if (bug_number.empty()) {
      ASSERT_EQ(result, kTfLiteOk);
    } else {
      if (FLAGS_ignore_known_bugs) {
        ASSERT_EQ(result, kTfLiteError)
            << "Not failing as expected due to http://b/" << bug_number;
      } else {
        ASSERT_EQ(result, kTfLiteOk)
            << "Possibly due to http://b/" << bug_number;
      }
    }
  }
}

// Instantiate a test. This assumes `zip_base`.zip is a declared data file
// of this test.
#define INSTANTIATE_TESTS(zip_base) \
  INSTANTIATE_TEST_CASE_P(          \
      zip_base, OpsTest,            \
      ::testing::ValuesIn(UnarchiveZipAndFindTestNames(#zip_base ".zip")));

INSTANTIATE_TESTS(add)
INSTANTIATE_TESTS(avg_pool)
INSTANTIATE_TESTS(space_to_batch_nd)
INSTANTIATE_TESTS(batch_to_space_nd)
INSTANTIATE_TESTS(concat)
// TODO(b/71642435) re-enable this test
// INSTANTIATE_TESTS(constant)
INSTANTIATE_TESTS(control_dep)
INSTANTIATE_TESTS(conv)
INSTANTIATE_TESTS(depthwiseconv)
INSTANTIATE_TESTS(fully_connected)
INSTANTIATE_TESTS(fused_batch_norm)
INSTANTIATE_TESTS(gather)
INSTANTIATE_TESTS(global_batch_norm)
INSTANTIATE_TESTS(l2norm)
INSTANTIATE_TESTS(l2_pool)
INSTANTIATE_TESTS(local_response_norm)
INSTANTIATE_TESTS(max_pool)
INSTANTIATE_TESTS(mul)
INSTANTIATE_TESTS(pad)
INSTANTIATE_TESTS(relu)
INSTANTIATE_TESTS(relu1)
INSTANTIATE_TESTS(relu6)
INSTANTIATE_TESTS(reshape)
INSTANTIATE_TESTS(resize_bilinear)
INSTANTIATE_TESTS(sigmoid)
INSTANTIATE_TESTS(softmax)
INSTANTIATE_TESTS(space_to_depth)
INSTANTIATE_TESTS(transpose)

}  // namespace testing
}  // namespace tflite

int main(int argc, char** argv) {
  ::testing::AddGlobalTestEnvironment(tflite::testing::zip_environment());

  std::vector<tensorflow::Flag> flags = {tensorflow::Flag(
      "ignore_known_bugs", &FLAGS_ignore_known_bugs,
      "If a particular model is affected by a known bug, the "
      "corresponding test should expect the outputs to not match.")};
  bool success = tensorflow::Flags::Parse(&argc, argv, flags);
  if (!success || (argc == 2 && !strcmp(argv[1], "--helpfull"))) {
    fprintf(stderr, "%s", tensorflow::Flags::Usage(argv[0], flags).c_str());
    return 1;
  }

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
