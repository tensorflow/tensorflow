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
#include "tensorflow/core/common_runtime/scoped_allocator_mgr.h"

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/scoped_allocator.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class ScopedAllocatorMgrTest : public ::testing::Test {
 public:
  ScopedAllocatorMgrTest() : sam_("CPU0") {}

  void InitTensor() {
    backing_tensor_ = Tensor(cpu_allocator(), DT_FLOAT, backing_tensor_shape_);
  }

  void PopulateFields() {
    ScopedAllocatorMgr::PopulateFields(scope_id_, fields_shapes_, DT_FLOAT,
                                       &fields_);
  }

  Status AddScopedAllocator(int expected_use_count, int scope_id) {
    VLOG(2) << "Adding ScopedAllocator step_id " << step_id_ << " scope_id "
            << scope_id_ << " #fields " << fields_.size()
            << " expected_use_count " << expected_use_count;
    return sam_.AddScopedAllocator(backing_tensor_, step_id_, scope_id,
                                   "tensor_shape_599", fields_,
                                   expected_use_count);
  }

  Status PrepScopedAllocatorMgr(int expected_use_count) {
    InitTensor();
    PopulateFields();
    return AddScopedAllocator(expected_use_count, scope_id_);
  }

  void SaveInstances(int num_instances) {
    sa_instances_.clear();
    sa_instances_.resize(num_instances);
    ScopedAllocatorContainer* sac = sam_.GetContainer(step_id_);
    for (int i = 0; i < num_instances; i++) {
      sa_instances_[i] = sac->GetInstance(scope_id_ + 1 + i);
    }
  }

  // For the specific case when the backing tensor is of shape
  // {512 + 9 + 512 + 16} and the fields_shapes are {{512}, {3,3}, {2, 256}}
  // This method computes the padding between the second and third slice of the
  // backing tensor.  This example is reused across multiple tests.
  int AlignmentPadding() {
    int alignment_padding =
        (Allocator::kAllocatorAlignment -
         (521 * sizeof(float)) % Allocator::kAllocatorAlignment) %
        Allocator::kAllocatorAlignment;
    return alignment_padding;
  }

  // Debug
  void PrintShapes() {
    VLOG(2) << "tensor_shape=" << backing_tensor_shape_.DebugString();
    for (int i = 0; i < fields_shapes_.size(); i++) {
      VLOG(2) << "fields_shapes[" << i
              << "]=" << fields_shapes_[i].DebugString();
    }
  }

 protected:
  TensorShape backing_tensor_shape_;
  Tensor backing_tensor_;
  std::vector<TensorShape> fields_shapes_;
  std::vector<ScopedAllocator::Field> fields_;
  ScopedAllocatorMgr sam_;
  const int step_id_ = 101;
  const int scope_id_ = 599;
  std::vector<ScopedAllocatorInstance*> sa_instances_;
};

TEST_F(ScopedAllocatorMgrTest, ContainerAllocation) {
  ScopedAllocatorContainer* sac_101 = sam_.GetContainer(101);
  EXPECT_TRUE(sac_101 != nullptr);
  ScopedAllocatorContainer* sac_201 = sam_.GetContainer(201);
  EXPECT_TRUE(sac_201 != nullptr);
  EXPECT_NE(sac_101, sac_201);
  ScopedAllocatorContainer* also_sac_101 = sam_.GetContainer(101);
  EXPECT_EQ(sac_101, also_sac_101);
  sam_.Cleanup(101);
  // 201 should be cleaned up by the destructor.
}

TEST_F(ScopedAllocatorMgrTest, PopulateFields) {
  backing_tensor_shape_ = TensorShape({512 + 9 + 512 + 16});
  fields_shapes_ = std::vector<TensorShape>({{512}, {3, 3}, {2, 256}});
  InitTensor();
  PopulateFields();
  EXPECT_EQ(0, fields_[0].offset);
  EXPECT_EQ(512 * sizeof(float), fields_[0].bytes_requested);
  EXPECT_EQ(scope_id_ + 1, fields_[0].scope_id);
  EXPECT_EQ(512 * sizeof(float), fields_[1].offset);
  EXPECT_EQ(9 * sizeof(float), fields_[1].bytes_requested);
  EXPECT_EQ(scope_id_ + 2, fields_[1].scope_id);
  EXPECT_EQ(521 * sizeof(float) + AlignmentPadding(), fields_[2].offset);
  EXPECT_EQ(512 * sizeof(float), fields_[2].bytes_requested);
  EXPECT_EQ(scope_id_ + 3, fields_[2].scope_id);
}

TEST_F(ScopedAllocatorMgrTest, ContainerAddAllocator) {
  backing_tensor_shape_ = TensorShape({1024});
  fields_shapes_ = std::vector<TensorShape>({{512}, {512}});
  Status s = PrepScopedAllocatorMgr(2);
  EXPECT_TRUE(s.ok());
  // Need to call Allocate and Deallocate in order to use up the expected uses
  // for this allocator.  Save the instances for now.
  SaveInstances(fields_shapes_.size());

  s = AddScopedAllocator(2, scope_id_);
  EXPECT_FALSE(s.ok());
  fields_[0].scope_id = scope_id_ + 1;
  s = AddScopedAllocator(2, scope_id_ + 3);
  EXPECT_FALSE(s.ok());

  // Cleanup the instances by invoking allocate and deallocate.
  void* ptr0 =
      sa_instances_[0]->AllocateRaw(0 /* alignment */, 512 * sizeof(float));
  void* ptr1 =
      sa_instances_[1]->AllocateRaw(0 /* alignment */, 512 * sizeof(float));
  sa_instances_[0]->DeallocateRaw(ptr0);
  sa_instances_[1]->DeallocateRaw(ptr1);
}

TEST_F(ScopedAllocatorMgrTest, AllocatorSuccess) {
  ScopedAllocatorContainer* sac = sam_.GetContainer(step_id_);
  ScopedAllocator* other = sac->GetAllocator(scope_id_);
  EXPECT_EQ(other, nullptr);
  backing_tensor_shape_ = TensorShape({512 + 9 + 512 + 16});
  fields_shapes_ = std::vector<TensorShape>({{512}, {3, 3}, {2, 256}});
  Status s = PrepScopedAllocatorMgr(3);
  other = sac->GetAllocator(scope_id_);

  ScopedAllocatorInstance* inst0 = sac->GetInstance(scope_id_ + 1);
  char* ptr0 = static_cast<char*>(inst0->AllocateRaw(0, 512 * sizeof(float)));
  const char* base =
      static_cast<const char*>(DMAHelper::base(&backing_tensor_));
  EXPECT_EQ(ptr0, base);

  ScopedAllocatorInstance* inst1 = sac->GetInstance(scope_id_ + 2);
  char* ptr1 = static_cast<char*>(inst1->AllocateRaw(0, 9 * sizeof(float)));
  EXPECT_EQ(ptr1, ptr0 + (512 * sizeof(float)));

  ScopedAllocatorInstance* inst2 = sac->GetInstance(scope_id_ + 3);
  char* ptr2 = static_cast<char*>(inst2->AllocateRaw(0, 512 * sizeof(float)));
  EXPECT_EQ(ptr2, ptr1 + AlignmentPadding() + (9 * sizeof(float)));

  // At this point the scopes should be gone from the container
  EXPECT_EQ(nullptr, sac->GetAllocator(scope_id_));

  // The ScopedAllocatorInstances automatically delete when their memory
  // is returned and they are out of table.
  inst0->DeallocateRaw(ptr0);
  inst1->DeallocateRaw(ptr1);
  inst2->DeallocateRaw(ptr2);
}

// ScopedAllocator initialization should fail because backing_tensor is not
// large enough to hold all the fields
TEST_F(ScopedAllocatorMgrTest, AllocatorInitFail) {
  backing_tensor_shape_ = TensorShape({8});
  InitTensor();
  fields_.resize(1);
  fields_[0].scope_id = scope_id_ + 1;
  fields_[0].offset = 0;
  fields_[0].bytes_requested =
      backing_tensor_shape_.num_elements() * 2 * sizeof(float);
  // fields[0].offset + fields[0].bytes_requested is larger than the size of the
  // backing tensor, so this check should fail
  EXPECT_DEATH(Status s = AddScopedAllocator(1, scope_id_), "");
}

// ScopedAllocator allocation should fail because we called more times than
// expected, or we deallocated a non-existent pointer, or we requested more
// or less than the exact size of an instance buffer.
TEST_F(ScopedAllocatorMgrTest, AllocatorFail) {
  backing_tensor_shape_ = TensorShape({1024});
  fields_shapes_ = std::vector<TensorShape>({{512}, {512}});
  Status s = PrepScopedAllocatorMgr(2);
  EXPECT_TRUE(s.ok());
  // Save instances so that we can explicitly delete later on.  In normal
  // operation the instances will be automatically deleted after single use, but
  // in this test we are invoking the ScopedAllocator's Alloc/Dealloc interface,
  // so we need to explicitly delete the instances to avoid a memleak.
  SaveInstances(fields_shapes_.size());

  char* ptr0 =
      static_cast<char*>(sa_instances_[0]->AllocateRaw(0, 512 * sizeof(float)));
  VLOG(2) << "Should fail because we deallocate ptr="
          << static_cast<void*>(ptr0 + 8) << " which we never allocated.";
  EXPECT_DEATH(sa_instances_[0]->DeallocateRaw(ptr0 + 8), "");
  VLOG(2) << "Should fail because we allocate smaller than the size of the "
          << "field.";
  EXPECT_EQ(nullptr, sa_instances_[1]->AllocateRaw(0, 256 * sizeof(float)));
  VLOG(2) << "Should fail because we allocate larger than the size of the "
          << "field.";
  EXPECT_EQ(nullptr, sa_instances_[1]->AllocateRaw(0, 1024 * sizeof(float)));
  void* ptr1 = sa_instances_[1]->AllocateRaw(0, 512 * sizeof(float));
  VLOG(2) << "Should fail because we exceed expected_use_count.";
  EXPECT_EQ(nullptr, sa_instances_[0]->AllocateRaw(0, 512 * sizeof(float)));
  sa_instances_[0]->DeallocateRaw(ptr0);
  sa_instances_[1]->DeallocateRaw(ptr1);
}

}  // namespace
}  // namespace tensorflow
