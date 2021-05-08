/* Copyright 2020 Elphel, Inc.

SPDX-License-Identifier: Apache-2.0
==============================================================================*/

package org.tensorflow;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.fail;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class GPUTensorTest {

  @Test
  public void allocateInGPU() {

    long[] valid_shape = new long[]{256};
    long[] invalid_shape = new long[]{255};

    // not supposed to fail
    allocateInGPUMemory(valid_shape,DataType.UINT8,true);
    allocateInGPUMemory(valid_shape,DataType.INT32,true);
    allocateInGPUMemory(valid_shape,DataType.FLOAT,true);

    // NullPointerException fail is expected
    allocateInGPUMemory(  valid_shape,DataType.DOUBLE,false);
    allocateInGPUMemory(  valid_shape,DataType.INT64,false);
    allocateInGPUMemory(invalid_shape,DataType.UINT8,false);
    allocateInGPUMemory(invalid_shape,DataType.INT32,false);
    allocateInGPUMemory(invalid_shape,DataType.FLOAT,false);

  }

  private static void allocateInGPUMemory(long[] shape, DataType dtype, boolean failOnException){
    try(
      Tensor<?> t = Tensor.createGPU(shape,dtype)
    ){
      assertTrue(t.isGPUTensor());
    } catch (NullPointerException e){
      if (failOnException){
        fail("Failed to check GPU allocation of the tensor");
      }else{
        // expected exception.
      }
    }
  }

  @Test
  public void checkCPUTensorAllocation() {
    try(
      Tensor<Integer> t = Tensors.create(new int[] {1, 2, 3, 4});
    ){
      assertFalse(t.isGPUTensor());
    }
  }

}