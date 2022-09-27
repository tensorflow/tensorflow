/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include <string>
#include <vector>

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

namespace tensorflow {
namespace {

void WriteCheckpoint(const string& prefix, gtl::ArraySlice<string> names,
                     gtl::ArraySlice<Tensor> tensors) {
  BundleWriter writer(Env::Default(), prefix);
  ASSERT_TRUE(names.size() == tensors.size());
  for (size_t i = 0; i < names.size(); ++i) {
    TF_ASSERT_OK(writer.Add(names[i], tensors[i]));
  }
  TF_ASSERT_OK(writer.Finish());
}

template <typename T>
Tensor Constant(T v, TensorShape shape) {
  Tensor ret(DataTypeToEnum<T>::value, shape);
  ret.flat<T>().setConstant(v);
  return ret;
}

class MergeV2CheckpointsOpTest : public OpsTestBase {
 protected:
  void MakeOp(bool delete_old_dirs, bool allow_missing_files) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "MergeV2Checkpoints")
                     .Input(FakeInput())  // checkpoint_prefixes
                     .Input(FakeInput())  // destination_prefix
                     .Attr("delete_old_dirs", delete_old_dirs)
                     .Attr("allow_missing_files", allow_missing_files)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }

  void RunMergeTest(bool delete_old_dirs, bool allow_missing_files) {
    // Writes two checkpoints.
    const std::vector<string> prefixes = {
        io::JoinPath(testing::TmpDir(), "worker0/ckpt0"),
        io::JoinPath(testing::TmpDir(), "worker1/ckpt1"),
        io::JoinPath(testing::TmpDir(), "merged/ckpt") /* merged prefix */};
    // In a different directory, to exercise "delete_old_dirs".
    const string& kMergedPrefix = prefixes[2];

    // Only write this particular checkpoint if we do not allow missing files.
    if (!allow_missing_files) {
      WriteCheckpoint(prefixes[0], {"tensor0"},
                      {Constant<float>(0, TensorShape({10}))});
    }

    WriteCheckpoint(prefixes[1], {"tensor1", "tensor2"},
                    {Constant<int64_t>(1, TensorShape({1, 16, 18})),
                     Constant<bool>(true, TensorShape({}))});
    // Now merges.
    MakeOp(delete_old_dirs, allow_missing_files);
    // Add checkpoint_prefixes.
    AddInput<tstring>(TensorShape({2}),
                      [&prefixes](int i) -> tstring { return prefixes[i]; });
    // Add destination_prefix.
    AddInput<tstring>(TensorShape({}), [kMergedPrefix](int unused) -> tstring {
      return kMergedPrefix;
    });
    TF_ASSERT_OK(RunOpKernel());

    // Check that the merged checkpoint file is properly written.
    BundleReader reader(Env::Default(), kMergedPrefix);
    TF_EXPECT_OK(reader.status());

    if (!allow_missing_files) {
      // We expect this only when all checkpoint files must exist.
      {
        Tensor val0;
        TF_EXPECT_OK(reader.Lookup("tensor0", &val0));
        test::ExpectTensorEqual<float>(Constant<float>(0, TensorShape({10})),
                                       val0);
      }
    }
    // We expect to find all saved tensors.
    {
      Tensor val1;
      TF_EXPECT_OK(reader.Lookup("tensor1", &val1));
      test::ExpectTensorEqual<int64_t>(
          Constant<int64_t>(1, TensorShape({1, 16, 18})), val1);
    }
    {
      Tensor val2;
      TF_EXPECT_OK(reader.Lookup("tensor2", &val2));
      test::ExpectTensorEqual<bool>(Constant<bool>(true, TensorShape({})),
                                    val2);
    }

    // Exercises "delete_old_dirs".
    for (int i = 0; i < 2; ++i) {
      // If we allow missing files, the first checkpoint file did not exist.
      if (allow_missing_files && i == 0) continue;
      int directory_found =
          Env::Default()->IsDirectory(string(io::Dirname(prefixes[i]))).code();
      if (delete_old_dirs) {
        EXPECT_EQ(error::NOT_FOUND, directory_found);
      } else {
        EXPECT_EQ(error::OK, directory_found);
      }
    }
  }
};

TEST_F(MergeV2CheckpointsOpTest, MergeNoDeleteNoMissingFiles) {
  RunMergeTest(false /* don't delete old dirs */,
               false /* don't allow missing files */);
}

TEST_F(MergeV2CheckpointsOpTest, MergeAndDeleteNoMissingFiles) {
  RunMergeTest(true /* delete old dirs */,
               false /* don't allow missing files */);
}

TEST_F(MergeV2CheckpointsOpTest, MergeNoDeleteWithMissingFiles) {
  RunMergeTest(false /* don't delete old dirs */,
               true /* allow missing files */);
}

TEST_F(MergeV2CheckpointsOpTest, MergeAndDeleteWithMissingFiles) {
  RunMergeTest(true /* delete old dirs */, true /* allow missing files */);
}

}  // namespace
}  // namespace tensorflow
