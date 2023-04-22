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
#include "tensorflow/core/kernels/immutable_constant_op.h"

#include <algorithm>
#include <tuple>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/null_file_system.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace {
// A safe alignment that equal to memmapped page alignment on many modern
// architectures.
constexpr size_t kTestAlignment = 4096;
constexpr size_t kTestTensorSize = 4;
constexpr size_t kTestTensorSizeBytes = kTestTensorSize * sizeof(float);

// A test ReadOnlyMemoryRegion implementation.
class TestReadOnlyMemoryRegion : public ReadOnlyMemoryRegion {
 public:
  TestReadOnlyMemoryRegion() = delete;
  explicit TestReadOnlyMemoryRegion(uint64 length)
      : memptr_(cpu_allocator()->AllocateRaw(kTestAlignment, length)),
        length_(length) {}
  ~TestReadOnlyMemoryRegion() override {
    cpu_allocator()->DeallocateRaw(memptr_);
  }
  const void* data() override { return memptr_; }
  float* GetWritableDataStart() { return reinterpret_cast<float*>(memptr_); }
  uint64 length() override { return length_; }

 protected:
  void* memptr_;
  uint64 length_;
};

// A mock file system and environment class that creates ReadOnlyMemoryRegion
// from allocated memory.
class TestFileSystem : public NullFileSystem {
 public:
  ~TestFileSystem() override = default;

  // import non-transactional method from the base class
  using NullFileSystem::NewReadOnlyMemoryRegionFromFile;

  Status NewReadOnlyMemoryRegionFromFile(
      const string& fname, TransactionToken* token,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) override {
    float val = 0;
    StringPiece scheme, host, path;
    io::ParseURI(fname, &scheme, &host, &path);
    // For the tests create in-memory regions with float values equal to the
    // region name.
    if (path == "/2") {
      val = 2.0f;
    } else if (path == "/3") {
      val = 3.0f;
    } else {
      val = 0.0f;
    }

    auto region = new TestReadOnlyMemoryRegion(kTestTensorSizeBytes);
    std::fill_n(region->GetWritableDataStart(), kTestTensorSize, val);
    result->reset(region);
    return Status::OK();
  }
};

REGISTER_FILE_SYSTEM("test", TestFileSystem);

struct ImmutableConstantOpTest {};

TEST(ImmutableConstantOpTest, Simple) {
  const TensorShape kTestTensorShape({4, 1});
  const TensorShape kTestTensorShapeT({1, 4});
  auto root = Scope::NewRootScope().ExitOnError();
  auto node1 =
      ops::ImmutableConst(root, DT_FLOAT, kTestTensorShape, "test:///2");
  auto node2 =
      ops::ImmutableConst(root, DT_FLOAT, kTestTensorShapeT, "test:///3");
  auto result = ops::MatMul(root, node1, node2);
  GraphDef graph_def;
  TF_ASSERT_OK(root.ToGraphDef(&graph_def));
  SessionOptions session_options;
  session_options.env = Env::Default();
  session_options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(OptimizerOptions::L0);
  std::unique_ptr<Session> session(NewSession(session_options));
  ASSERT_TRUE(session != nullptr) << "Failed to create session";
  TF_ASSERT_OK(session->Create(graph_def)) << "Can't create test graph";
  std::vector<Tensor> outputs;
  TF_ASSERT_OK(session->Run({}, {result.node()->name() + ":0"}, {}, &outputs));
  ASSERT_EQ(outputs.size(), 1);
  EXPECT_EQ(outputs.front().flat<float>()(0), 2.0f * 3.0f);
  EXPECT_EQ(outputs.front().flat<float>()(1), 2.0f * 3.0f);
  EXPECT_EQ(outputs.front().flat<float>()(2), 2.0f * 3.0f);
  EXPECT_EQ(outputs.front().flat<float>()(kTestTensorSize - 1), 2.0f * 3.0f);
}

// Creates a test graph with two immutable_const tensors and a simple math
// operation, one of nodes has wrong size, check that error properly reported.

TEST(ImmutableConstantOpTest, ExecutionError) {
  const TensorShape kBadTensorShape({40, 100});
  const TensorShape kTestTensorShapeT({1, 4});

  auto root = Scope::DisabledShapeInferenceScope().ExitOnError();
  auto node1 =
      ops::ImmutableConst(root, DT_FLOAT, kBadTensorShape, "test:///2");
  auto node2 =
      ops::ImmutableConst(root, DT_FLOAT, kTestTensorShapeT, "test:///3");
  auto result = ops::MatMul(root, node1, node2);
  GraphDef graph_def;
  TF_ASSERT_OK(root.ToGraphDef(&graph_def));
  SessionOptions session_options;
  session_options.env = Env::Default();
  std::unique_ptr<Session> session(NewSession(session_options));
  ASSERT_TRUE(session != nullptr) << "Failed to create session";
  TF_ASSERT_OK(session->Create(graph_def)) << "Can't create test graph";
  std::vector<Tensor> outputs;
  // Check that the run returned error.
  EXPECT_EQ(
      session->Run({}, {result.node()->name() + ":0"}, {}, &outputs).code(),
      error::INTERNAL);
}

Status CreateTempFile(Env* env, float value, uint64 size, string* filename) {
  const string dir = testing::TmpDir();
  *filename = io::JoinPath(dir, strings::StrCat("file_", value));
  std::unique_ptr<WritableFile> file;
  TF_RETURN_IF_ERROR(env->NewWritableFile(*filename, &file));
  for (uint64 i = 0; i < size; ++i) {
    StringPiece sp(static_cast<char*>(static_cast<void*>(&value)),
                   sizeof(value));
    TF_RETURN_IF_ERROR(file->Append(sp));
  }
  TF_RETURN_IF_ERROR(file->Close());
  return Status::OK();
}

TEST(ImmutableConstantOpTest, FromFile) {
  const TensorShape kFileTensorShape({1000, 1});
  Env* env = Env::Default();
  auto root = Scope::NewRootScope().ExitOnError();

  string two_file, three_file;
  TF_ASSERT_OK(CreateTempFile(env, 2.0f, 1000, &two_file));
  TF_ASSERT_OK(CreateTempFile(env, 3.0f, 1000, &three_file));
  auto node1 = ops::ImmutableConst(root, DT_FLOAT, kFileTensorShape, two_file);
  auto node2 =
      ops::ImmutableConst(root, DT_FLOAT, kFileTensorShape, three_file);
  auto result = ops::MatMul(root, node1, node2, ops::MatMul::TransposeB(true));

  GraphDef graph_def;
  TF_ASSERT_OK(root.ToGraphDef(&graph_def));
  SessionOptions session_options;
  session_options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(OptimizerOptions::L0);
  std::unique_ptr<Session> session(NewSession(session_options));
  ASSERT_TRUE(session != nullptr) << "Failed to create session";
  TF_ASSERT_OK(session->Create(graph_def)) << "Can't create test graph";
  std::vector<Tensor> outputs;
  TF_ASSERT_OK(session->Run({}, {result.node()->name() + ":0"}, {}, &outputs));
  ASSERT_EQ(outputs.size(), 1);
  EXPECT_EQ(outputs.front().flat<float>()(0), 2.0f * 3.0f);
  EXPECT_EQ(outputs.front().flat<float>()(1), 2.0f * 3.0f);
  EXPECT_EQ(outputs.front().flat<float>()(2), 2.0f * 3.0f);
}

}  // namespace
}  // namespace tensorflow
