/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");

You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Tests kernels of lookup ops.

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/lookup_interface.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/lookup_table_op.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

// This is a mock op we test against, in lieu of the real AnonymousHashTable op.
// They are very similar. The only difference between this op and
// AnonymousHashTable is that the former flips the global variable `alive` in
// its destructor to tell the outside world that it has been deleted.
REGISTER_OP("MockAnonymousHashTable")
    .Output("table_handle: resource")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetIsStateful();

static bool alive = false;

template <class K, class V>
class MockHashTable : public lookup::HashTable<K, V> {
 public:
  MockHashTable(OpKernelContext* ctx, OpKernel* kernel)
      : lookup::HashTable<K, V>(ctx, kernel) {
    alive = true;
  }
  ~MockHashTable() override { alive = false; }
};

typedef int32 key_dtype;
typedef int32 value_dtype;

REGISTER_KERNEL_BUILDER(
    Name("MockAnonymousHashTable")
        .Device(DEVICE_CPU)
        .TypeConstraint<key_dtype>("key_dtype")
        .TypeConstraint<value_dtype>("value_dtype"),
    AnonymousLookupTableOp<MockHashTable<key_dtype, value_dtype>, key_dtype,
                           value_dtype>);

class LookupOpsTest : public OpsTestBase {};

TEST_F(LookupOpsTest, AnonymousHashTable_RefCounting) {
  TF_ASSERT_OK(
      NodeDefBuilder("mock_anonymous_hash_table", "MockAnonymousHashTable")
          .Attr("key_dtype", DataTypeToEnum<key_dtype>::v())
          .Attr("value_dtype", DataTypeToEnum<value_dtype>::v())
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  alive = false;

  // Feed and run
  TF_ASSERT_OK(RunOpKernel());
  EXPECT_TRUE(alive);

  // Check the output.
  Tensor* output = GetOutput(0);
  ResourceHandle& output_handle = output->scalar<ResourceHandle>()();
  EXPECT_TRUE(output_handle.IsRefCounting());
  ResourceBase* base = output_handle.resource().get();
  EXPECT_TRUE(base);
  EXPECT_EQ(base->RefCount(), 1);
  auto resource_or = output_handle.GetResource<lookup::LookupInterface>();
  TF_EXPECT_OK(resource_or.status());
  if (resource_or.ok()) {
    auto mock = resource_or.ValueOrDie();
    EXPECT_TRUE(mock);
    EXPECT_EQ(base->RefCount(), 1);  // GetResource won't increase ref-count
  }

  // context_->outputs_ holds the last ref to the output tensor (i.e. the
  // resource handle)
  context_.reset();
  // Now that all resource handles are gone, the resource should have been
  // deleted automatically.
  EXPECT_FALSE(alive);
}

}  // namespace
}  // namespace tensorflow
