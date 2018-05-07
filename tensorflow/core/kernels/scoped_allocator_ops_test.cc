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

#include <vector>

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/common_runtime/scoped_allocator.h"
#include "tensorflow/core/common_runtime/scoped_allocator_mgr.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class ScopedAllocatorOpTest : public OpsTestBase {
 protected:
  void MakeOp(const gtl::ArraySlice<TensorShape>& shapes, DataType dtype,
              const string& name, int32 id, int32 expected_call_count) {
    TF_EXPECT_OK(NodeDefBuilder("scoped_allocator_op", "_ScopedAllocator")
                     .Attr("T", dtype)
                     .Attr("shapes", shapes)
                     .Attr("sa_name", name)
                     .Attr("id", id)
                     .Attr("expected_call_count", expected_call_count)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
    TF_ASSERT_OK(RunOpKernel());

    // Allocate and Deallocate the tensors so that memory is not leaked
    AllocatorAttributes attr;
    Allocator* allocator;
    for (size_t i = 0; i < shapes.size(); i++) {
      attr.scope_id = id + i + 1;
      allocator = device_->GetScopedAllocator(attr, context_->step_id());
      Tensor temp(allocator, dtype, shapes[i]);
    }
  }
};

TEST_F(ScopedAllocatorOpTest, Simple) {
  MakeOp({TensorShape({8})}, DT_FLOAT, "test", 120, 1);
  MakeOp({TensorShape({32, 32})}, DT_DOUBLE, "test1", 130, 1);
  MakeOp({TensorShape({64}), TensorShape({3, 3}), TensorShape({5, 5, 5})},
         DT_HALF, "test2", 140, 3);
  MakeOp({TensorShape({512}), TensorShape({64, 8})}, DT_UINT32, "test3", 150,
         2);
}

// PrepOp is common to ConcatOp tests and SplitOpTests.
// It allocates a backing tensor that is large enough to hold all slices defined
// by fields, creates ScopedAllocatorInstances for each field, allocates the
// tensors, and assigns them as inputs to the op.
// We won't use the AddInput* suite of functions from ops_testutil.h because
// they allocate new tensors for each input.  We need to mimic what a
// ScopedAllocator would do.
void PrepOp(DataType dtype, int32 id,
            const std::vector<TensorShape>& fields_shapes,
            std::vector<ScopedAllocator::Field>* fields,
            Tensor** backing_tensor, Allocator* allocator,
            ScopedAllocatorMgr* sam, const string& op_name,
            std::vector<Tensor>* tensors,
            gtl::InlinedVector<TensorValue, 4>* inputs,
            const DataTypeVector& input_types) {
  ScopedAllocatorMgr::PopulateFields(id, fields_shapes, dtype, fields);
  // We don't simply allocate a tensor with shape as backing_tensor_shape,
  // because we need to account for padding in the fields.  We actually need a
  // tensor of size at least (fields[-1].offset + fields[-1].bytes).
  size_t num_bytes = fields->back().offset + fields->back().bytes;
  int32_t num_elements = num_bytes / DataTypeSize(dtype);
  CHECK_EQ(num_bytes % DataTypeSize(dtype), 0);

  *backing_tensor = new Tensor(allocator, dtype, {num_elements});
  int64 step_id = 10;
  Status s = sam->AddScopedAllocator(**backing_tensor, step_id, id,
                                     "sa_" + op_name + "_test", *fields,
                                     fields_shapes.size());
  TF_ASSERT_OK(s);

  ScopedAllocatorContainer* sac = sam->GetContainer(step_id);
  std::vector<ScopedAllocatorInstance*> sa_instances(fields_shapes.size(),
                                                     nullptr);
  for (size_t i = 0; i < fields_shapes.size(); i++) {
    sa_instances[i] = sac->GetInstance(id + i + 1);
    tensors->push_back(Tensor(sa_instances[i], dtype, fields_shapes[i]));
  }
  // Now add the tensor as an input to ScopedAllocator<op_name>Op.
  // Order matters here, so first add the backing tensor, then the slices.
  inputs->reserve(1 + tensors->size());
  CHECK_GT(input_types.size(), inputs->size());
  CHECK_EQ(input_types[inputs->size()], dtype);
  inputs->push_back({nullptr, *backing_tensor});
  for (size_t i = 0; i < tensors->size(); i++) {
    CHECK_EQ(input_types[inputs->size()], dtype);
    inputs->push_back({nullptr, &((*tensors)[i])});
  }
}

class ScopedAllocatorConcatOpTest : public OpsTestBase {
 protected:
  void BuildNodeDef(const TensorShape& shape, DataType dtype,
                    const string& name, int32 id, int32 num_tensors) {
    TF_EXPECT_OK(
        NodeDefBuilder("scoped_allocator_concat_op", "_ScopedAllocatorConcat")
            .Attr("shape", shape)
            .Attr("T", dtype)
            .Attr("N", num_tensors)
            .Attr("sa_name", name)
            .Attr("id", id)
            .Input(FakeInput(dtype))               // backing tensor
            .Input(FakeInput(num_tensors, dtype))  // list of tensors
            .Finalize(node_def()));
    shape_ = shape;
    reshape_ = false;
  }

  void BuildNodeDefWithReshape(const TensorShape& shape, DataType dtype,
                               bool reshape, const string& name, int32 id,
                               int32 num_tensors) {
    TF_EXPECT_OK(
        NodeDefBuilder("scoped_allocator_concat_op", "_ScopedAllocatorConcat")
            .Attr("shape", shape)
            .Attr("T", dtype)
            .Attr("reshape", reshape)
            .Attr("N", num_tensors)
            .Attr("sa_name", name)
            .Attr("id", id)
            .Input(FakeInput(dtype))               // backing tensor
            .Input(FakeInput(num_tensors, dtype))  // list of tensors
            .Finalize(node_def()));
    shape_ = shape;
    reshape_ = reshape;
  }

  void MakeOp(const TensorShape& shape, DataType dtype, bool reshape,
              const string& name, int32 id, int32 num_tensors) {
    BuildNodeDefWithReshape(shape, dtype, reshape, name, id, num_tensors);
    TF_EXPECT_OK(InitOp());
  }

  void ExecOp(DataType dtype, int32 id,
              const std::vector<TensorShape>& fields_shapes) {
    Tensor* backing_tensor = nullptr;
    std::vector<Tensor> tensors;
    std::vector<ScopedAllocator::Field> fields;
    PrepOp(dtype, id, fields_shapes, &fields, &backing_tensor, allocator(),
           device_->GetScopedAllocatorMgr(), "concat", &tensors, &inputs_,
           input_types_);

    TF_ASSERT_OK(RunOpKernel());

    // Check input and output are same tensor.
    const Tensor& input = context_->input(0);
    OpOutputList output_list;
    Status s = context_->output_list("output", &output_list);
    TF_ASSERT_OK(s);
    const Tensor& output = *(output_list[0]);
    CHECK_EQ(DMAHelper::base(&input), DMAHelper::base(&output));
    CHECK_EQ(input.dtype(), output.dtype());
    CHECK_EQ(input.NumElements(), output.NumElements());
    if (reshape_) {
      CHECK_EQ(shape_, output.shape());
    } else {
      TensorShape expected_shape({input.NumElements()});
      CHECK_EQ(expected_shape, output.shape());
    }

    // Free the backing tensor which was allocated in PrepOp.
    delete backing_tensor;
  }

 private:
  TensorShape shape_;
  bool reshape_;
};

TEST_F(ScopedAllocatorConcatOpTest, Success1) {
  MakeOp({32}, DT_FLOAT, false, "test", 120, 2);
  ExecOp(DT_FLOAT, 120, {{16}, {16}});
}

TEST_F(ScopedAllocatorConcatOpTest, Success2) {
  MakeOp({2, 2, 2}, DT_DOUBLE, false, "test", 120, 2);
  ExecOp(DT_DOUBLE, 120, {{2, 2}, {2, 2}});
}

TEST_F(ScopedAllocatorConcatOpTest, Success3) {
  MakeOp({3, 3, 3}, DT_HALF, false, "test", 120, 3);
  ExecOp(DT_HALF, 120, {{3, 3}, {3, 3}, {3, 3}});
}

TEST_F(ScopedAllocatorConcatOpTest, Reshape) {
  MakeOp({2, 2, 2}, DT_DOUBLE, true, "test", 120, 2);
  ExecOp(DT_DOUBLE, 120, {{2, 2}, {2, 2}});
}

TEST_F(ScopedAllocatorConcatOpTest, NoReshapeAttr) {
  BuildNodeDef({3, 4, 4}, DT_HALF, "test", 120, 3);
  TF_EXPECT_OK(InitOp());
  ExecOp(DT_HALF, 120, {{4, 4}, {4, 4}, {4, 4}});
}

TEST_F(ScopedAllocatorConcatOpTest, FailDtypeCheck) {
  MakeOp({8}, DT_FLOAT, false, "test", 120, 2);
  EXPECT_DEATH(ExecOp(DT_DOUBLE, 120, {{4}, {4}}), "");
}

TEST_F(ScopedAllocatorConcatOpTest, FailNumElementsCheck) {
  MakeOp({32}, DT_FLOAT, false, "test", 120, 2);
  AddInputFromArray<float>({8}, {0, 1, 2, 3, 4, 5, 6, 7});
  AddInputFromArray<float>({4}, {0, 1, 2, 3});
  AddInputFromArray<float>({4}, {4, 5, 6, 7});
  Status s = RunOpKernel();
  EXPECT_EQ(s.code(), error::INVALID_ARGUMENT);
}

// This test should fail because the backing tensor and the input tensors are
// unrelated, i.e. the inputs are not slices of the backing tensor.
TEST_F(ScopedAllocatorConcatOpTest, FailBounds) {
  MakeOp({8}, DT_DOUBLE, false, "test", 120, 2);
  AddInputFromArray<double>({8}, {0, 1, 2, 3, 4, 5, 6, 7});
  AddInputFromArray<double>({4}, {0, 1, 2, 3});
  AddInputFromArray<double>({4}, {4, 5, 6, 7});
  Status s = RunOpKernel();
  EXPECT_EQ(s.code(), error::INVALID_ARGUMENT);
}

class ScopedAllocatorSplitOpTest : public OpsTestBase {
 protected:
  void BuildNodeDef(const TensorShape& shape, DataType dtype,
                    const string& name, int32 id, int32 num_tensors) {
    TF_EXPECT_OK(
        NodeDefBuilder("scoped_allocator_split_op", "_ScopedAllocatorSplit")
            .Attr("T", dtype)
            .Attr("N", num_tensors)
            .Attr("sa_name", name)
            .Attr("id", id)
            .Input(FakeInput(dtype))  // backing tensor and input
            .Input(
                FakeInput(num_tensors, dtype))  // list of subtensors to forward
            .Finalize(node_def()));
  }

  void MakeOp(const TensorShape& shape, DataType dtype, const string& name,
              int32 id, int32 num_tensors) {
    BuildNodeDef(shape, dtype, name, id, num_tensors);
    TF_EXPECT_OK(InitOp());
  }

  // Similar to ConcatOpTest, we add inputs that are allocated from
  // ScopedAllocator so that the memory lines up nicely.
  void ExecOp(DataType dtype, int32 id,
              const std::vector<TensorShape>& fields_shapes) {
    Tensor* backing_tensor = nullptr;
    std::vector<Tensor> tensors;
    std::vector<ScopedAllocator::Field> fields;
    PrepOp(dtype, id, fields_shapes, &fields, &backing_tensor, allocator(),
           device_->GetScopedAllocatorMgr(), "split", &tensors, &inputs_,
           input_types_);

    TF_ASSERT_OK(RunOpKernel());

    // Check that outputs are slices of backing tensor.
    const Tensor& input = context_->input(0);
    const void* lower_limit = DMAHelper::base(&input);
    const char* lower_limit_c =
        static_cast<const char*>(lower_limit);  // for pointer arithmetic
    OpOutputList output_list;
    Status s = context_->output_list("output", &output_list);
    TF_ASSERT_OK(s);
    for (int i = 0; i < output_list.size(); i++) {
      const Tensor& output = *(output_list[i]);
      const void* expected_base =
          static_cast<const void*>(lower_limit_c + fields[i].offset);
      CHECK_EQ(output.dtype(), input.dtype());
      CHECK_EQ(expected_base, DMAHelper::base(&output));
      CHECK_EQ(output.NumElements(), fields_shapes[i].num_elements());
    }

    // Free the backing tensor which was allocated in PrepOp.
    delete backing_tensor;
  }
};

TEST_F(ScopedAllocatorSplitOpTest, Success1) {
  MakeOp({32}, DT_FLOAT, "test", 120, 2);
  ExecOp(DT_FLOAT, 120, {{16}, {16}});
}

TEST_F(ScopedAllocatorSplitOpTest, Success2) {
  MakeOp({2, 2, 2}, DT_DOUBLE, "test", 120, 2);
  ExecOp(DT_DOUBLE, 120, {{2, 2}, {2, 2}});
}

TEST_F(ScopedAllocatorSplitOpTest, Success3) {
  MakeOp({3, 3, 3}, DT_HALF, "test", 120, 3);
  ExecOp(DT_HALF, 120, {{3, 3}, {3, 3}, {3, 3}});
}

TEST_F(ScopedAllocatorSplitOpTest, FailNLessThan2) {
  BuildNodeDef({4, 4}, DT_FLOAT, "test", 120, 1);
  Status s = InitOp();
  EXPECT_EQ(s.code(), error::INVALID_ARGUMENT);
}

TEST_F(ScopedAllocatorSplitOpTest, FailDtypeCheck) {
  MakeOp({8}, DT_FLOAT, "test", 120, 2);
  EXPECT_DEATH(ExecOp(DT_HALF, 120, {{4}, {4}}), "");
}

TEST_F(ScopedAllocatorSplitOpTest, FailBounds) {
  MakeOp({8}, DT_DOUBLE, "test", 120, 2);
  AddInputFromArray<double>({8}, {0, 1, 2, 3, 4, 5, 6, 7});
  AddInputFromArray<double>({4}, {0, 1, 2, 3});
  AddInputFromArray<double>({4}, {4, 5, 6, 7});
  Status s = RunOpKernel();
  EXPECT_EQ(s.code(), error::INVALID_ARGUMENT);
}

}  // end namespace tensorflow
