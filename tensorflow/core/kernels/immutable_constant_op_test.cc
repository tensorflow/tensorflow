/* Copyright 2016 Google Inc. All Rights Reserved.

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
  Status NewReadOnlyMemoryRegionFromFile(
      const string& fname, ReadOnlyMemoryRegion** result) override {
    float val = 0;
    // For the tests create in-memory regions with float values equal to the
    // first letter of the region name.
    switch (GetNameFromURI(fname).front()) {
      case '2':
        val = 2.0f;
        break;
      case '3':
        val = 3.0f;
        break;
      default:
        val = 0.0f;
        break;
    }

    auto region = new TestReadOnlyMemoryRegion(kTestTensorSizeBytes);
    std::fill_n(region->GetWritableDataStart(), kTestTensorSize, val);
    *result = region;
    return Status::OK();
  }
};

REGISTER_FILE_SYSTEM("test", TestFileSystem);

struct ImmutableConstantOpTest {};

TEST(ImmutableConstantOpTest, Simple) {
  const TensorShape kTestTensorShape({4, 1});
  const TensorShape kTestTensorShapeT({1, 4});
  GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
  Node* node1 =
      ops::ImmutableConst(DT_FLOAT, kTestTensorShape, "test://2", b.opts());
  Node* node2 =
      ops::ImmutableConst(DT_FLOAT, kTestTensorShapeT, "test://3", b.opts());
  Node* result = ops::MatMul(node1, node2, b.opts());
  GraphDef graph_def;
  TF_ASSERT_OK(b.ToGraphDef(&graph_def));
  SessionOptions session_options;
  session_options.env = Env::Default();
  session_options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(OptimizerOptions_Level_L0);
  std::unique_ptr<Session> session(NewSession(session_options));
  ASSERT_TRUE(session != nullptr) << "Failed to create session";
  TF_ASSERT_OK(session->Create(graph_def)) << "Can't create test graph";
  std::vector<Tensor> outputs;
  TF_ASSERT_OK(session->Run({}, {result->name() + ":0"}, {}, &outputs));
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
  GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
  Node* node1 =
      ops::ImmutableConst(DT_FLOAT, kBadTensorShape, "test://2", b.opts());
  Node* node2 =
      ops::ImmutableConst(DT_FLOAT, kTestTensorShapeT, "test://3", b.opts());
  Node* result = ops::MatMul(node1, node2, b.opts());
  GraphDef graph_def;
  TF_ASSERT_OK(b.ToGraphDef(&graph_def));
  SessionOptions session_options;
  session_options.env = Env::Default();
  std::unique_ptr<Session> session(NewSession(session_options));
  ASSERT_TRUE(session != nullptr) << "Failed to create session";
  TF_ASSERT_OK(session->Create(graph_def)) << "Can't create test graph";
  std::vector<Tensor> outputs;
  // Check that the run returned error.
  EXPECT_EQ(session->Run({}, {result->name() + ":0"}, {}, &outputs).code(),
            error::INTERNAL);
}

Status CreateTempFile(Env* env, float value, uint64 size, string* filename) {
  const string dir = testing::TmpDir();
  *filename = io::JoinPath(dir, strings::StrCat("file_", value));
  WritableFile* file;
  TF_RETURN_IF_ERROR(env->NewWritableFile(*filename, &file));
  std::unique_ptr<WritableFile> file_unique_ptr(file);
  for (uint64 i = 0; i < size; ++i) {
    StringPiece sp;
    sp.set(&value, sizeof(value));
    TF_RETURN_IF_ERROR(file->Append(sp));
  }
  TF_RETURN_IF_ERROR(file->Close());
  return Status::OK();
}

TEST(ImmutableConstantOpTest, FromFile) {
  const TensorShape kFileTensorShape({1000, 1});
  Env* env = Env::Default();
  GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
  string two_file, three_file;
  TF_ASSERT_OK(CreateTempFile(env, 2.0f, 1000, &two_file));
  TF_ASSERT_OK(CreateTempFile(env, 3.0f, 1000, &three_file));
  Node* node1 =
      ops::ImmutableConst(DT_FLOAT, kFileTensorShape, two_file, b.opts());
  Node* node2 =
      ops::ImmutableConst(DT_FLOAT, kFileTensorShape, three_file, b.opts());
  Node* result =
      ops::MatMul(node1, node2, b.opts().WithAttr("transpose_b", true));

  GraphDef graph_def;
  TF_ASSERT_OK(b.ToGraphDef(&graph_def));
  SessionOptions session_options;
  session_options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(OptimizerOptions_Level_L0);
  std::unique_ptr<Session> session(NewSession(session_options));
  ASSERT_TRUE(session != nullptr) << "Failed to create session";
  TF_ASSERT_OK(session->Create(graph_def)) << "Can't create test graph";
  std::vector<Tensor> outputs;
  TF_ASSERT_OK(session->Run({}, {result->name() + ":0"}, {}, &outputs));
  ASSERT_EQ(outputs.size(), 1);
  EXPECT_EQ(outputs.front().flat<float>()(0), 2.0f * 3.0f);
  EXPECT_EQ(outputs.front().flat<float>()(1), 2.0f * 3.0f);
  EXPECT_EQ(outputs.front().flat<float>()(2), 2.0f * 3.0f);
}

}  // namespace
}  // namespace tensorflow
