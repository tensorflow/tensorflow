#!/usr/bin/env python3
"""
Test cases to validate the GatherV2 axis parameter validation fix.
This script tests various invalid axis scenarios to ensure proper error handling.
"""

import os
import sys
import tempfile

def create_test_cpp():
    """Create a C++ test file to test the gather operation fix."""
    
    test_content = '''
#include <gtest/gtest.h>
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class GatherV2AxisValidationTest : public OpsTestBase {};

TEST_F(GatherV2AxisValidationTest, TestInvalidPositiveAxis) {
  // Test case from the original bug report: axis=9 for 2D tensor
  ASSERT_OK(NodeDefBuilder("gather_op", "GatherV2")
                .Input(FakeInput(DT_INT32))  // params
                .Input(FakeInput(DT_INT32))  // indices 
                .Input(FakeInput(DT_INT32))  // axis
                .Finalize(node_def()));
  ASSERT_OK(InitOp());

  // Create a 2D params tensor
  AddInputFromArray<int32_t>(TensorShape({2, 3}), {1, 2, 3, 4, 5, 6});
  // Create indices tensor
  AddInputFromArray<int32_t>(TensorShape({2}), {0, 1});
  // Create axis tensor with invalid value 9 (out of bounds for 2D tensor)
  AddInputFromArray<int32_t>(TensorShape({}), {9});

  // This should fail with InvalidArgumentError instead of CUDA memory error
  auto status = RunOpKernel();
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_TRUE(absl::StrContains(status.message(), "axis"));
  EXPECT_TRUE(absl::StrContains(status.message(), "out of bounds"));
}

TEST_F(GatherV2AxisValidationTest, TestInvalidNegativeAxis) {
  ASSERT_OK(NodeDefBuilder("gather_op", "GatherV2")
                .Input(FakeInput(DT_INT32))
                .Input(FakeInput(DT_INT32)) 
                .Input(FakeInput(DT_INT32))
                .Finalize(node_def()));
  ASSERT_OK(InitOp());

  // Create a 2D params tensor 
  AddInputFromArray<int32_t>(TensorShape({2, 3}), {1, 2, 3, 4, 5, 6});
  // Create indices tensor
  AddInputFromArray<int32_t>(TensorShape({2}), {0, 1});
  // Create axis tensor with invalid negative value -5 (too negative for 2D tensor)
  AddInputFromArray<int32_t>(TensorShape({}), {-5});

  // This should fail with InvalidArgumentError 
  auto status = RunOpKernel();
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_TRUE(absl::StrContains(status.message(), "axis"));
  EXPECT_TRUE(absl::StrContains(status.message(), "out of bounds"));
}

TEST_F(GatherV2AxisValidationTest, TestValidPositiveAxis) {
  ASSERT_OK(NodeDefBuilder("gather_op", "GatherV2")
                .Input(FakeInput(DT_INT32))
                .Input(FakeInput(DT_INT32))
                .Input(FakeInput(DT_INT32))
                .Finalize(node_def()));
  ASSERT_OK(InitOp());

  // Create a 2D params tensor
  AddInputFromArray<int32_t>(TensorShape({2, 3}), {1, 2, 3, 4, 5, 6});
  // Create indices tensor
  AddInputFromArray<int32_t>(TensorShape({2}), {0, 1});
  // Create axis tensor with valid value 1 (valid for 2D tensor)
  AddInputFromArray<int32_t>(TensorShape({}), {1});

  // This should succeed
  ASSERT_OK(RunOpKernel());
  
  // Verify output tensor shape and values
  Tensor expected(allocator(), DT_INT32, TensorShape({2, 2}));
  test::FillValues<int32_t>(&expected, {1, 2, 4, 5});
  test::ExpectTensorEqual<int32_t>(expected, *GetOutput(0));
}

TEST_F(GatherV2AxisValidationTest, TestValidNegativeAxis) {
  ASSERT_OK(NodeDefBuilder("gather_op", "GatherV2")
                .Input(FakeInput(DT_INT32))
                .Input(FakeInput(DT_INT32))
                .Input(FakeInput(DT_INT32))
                .Finalize(node_def()));
  ASSERT_OK(InitOp());

  // Create a 2D params tensor
  AddInputFromArray<int32_t>(TensorShape({2, 3}), {1, 2, 3, 4, 5, 6});
  // Create indices tensor  
  AddInputFromArray<int32_t>(TensorShape({2}), {0, 1});
  // Create axis tensor with valid negative value -1 (equivalent to axis=1 for 2D tensor)
  AddInputFromArray<int32_t>(TensorShape({}), {-1});

  // This should succeed
  ASSERT_OK(RunOpKernel());
  
  // Verify output tensor shape and values 
  Tensor expected(allocator(), DT_INT32, TensorShape({2, 2}));
  test::FillValues<int32_t>(&expected, {1, 2, 4, 5});
  test::ExpectTensorEqual<int32_t>(expected, *GetOutput(0));
}

TEST_F(GatherV2AxisValidationTest, TestAxisBoundaryValues) {
  ASSERT_OK(NodeDefBuilder("gather_op", "GatherV2")
                .Input(FakeInput(DT_INT32))
                .Input(FakeInput(DT_INT32))
                .Input(FakeInput(DT_INT32))
                .Finalize(node_def()));
  ASSERT_OK(InitOp());

  // Create a 3D params tensor for more comprehensive testing
  AddInputFromArray<int32_t>(TensorShape({2, 2, 2}), {1, 2, 3, 4, 5, 6, 7, 8});
  // Create indices tensor
  AddInputFromArray<int32_t>(TensorShape({1}), {0});
  // Test axis=3 which is exactly equal to params.dims() (should fail)
  AddInputFromArray<int32_t>(TensorShape({}), {3});

  auto status = RunOpKernel();
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_TRUE(absl::StrContains(status.message(), "axis"));
}

}  // namespace
}  // namespace tensorflow
'''
    
    return test_content

def main():
    # Create the test file
    test_content = create_test_cpp()
    
    # Write to a test file
    test_file = "/workspaces/tensorflow/tensorflow/core/kernels/gather_op_axis_validation_test.cc"
    with open(test_file, 'w') as f:
        f.write(test_content)
    
    print(f"✓ Created test file: {test_file}")
    print("\nTest cases cover:")
    print("1. Invalid positive axis (axis=9 for 2D tensor)")
    print("2. Invalid negative axis (axis=-5 for 2D tensor)")  
    print("3. Valid positive axis (axis=1 for 2D tensor)")
    print("4. Valid negative axis (axis=-1 for 2D tensor)")
    print("5. Boundary case (axis=params.dims())")
    print("\nThe tests verify that invalid axis values now produce proper")
    print("InvalidArgumentError instead of CUDA memory initialization failures.")

if __name__ == "__main__":
    main()
