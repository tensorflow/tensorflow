// Copyright 2026 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
// RUN: tfrt_fallback_translate -mlir-to-bef %s | tf_bef_executor --work_queue_type=mstd 2>&1 | FileCheck %s

// This test is to verify the fix for the bug in converting tensorhandle to
// fallback tensor. The TensorHandle should not be moved to the output,
// otherwise, the input will be set to null and cannot be referenced again.
// This bug happens if the AsyncTensor is unavailable during the conversion.

func.func @test_convert_tensorhandle_to_fallback_tensor() {
  %us = tfrt.constant.i32 1000

  // Given an available TensorHandle with an unavailable AsyncTensor, which will
  // become available after 1000 ms.
  // CHECK: Created TensorHandle (test string)
  %1 = "tfrt_fallback_test.create_tensorhandle_with_delayed_async_tensor"(%us) : (i32) -> !corert.tensorhandle

  %2 = "tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor"(%1) {_tfrt_cost = 1 : i64, device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (!corert.tensorhandle) -> !tfrt_fallback.tf_tensor

  // ... when `tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor` is
  //     invoked twice with the same input,
  // ... then the test should not crash.
  %3 = "tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor"(%1) {_tfrt_cost = 1 : i64, device = "/job:localhost/replica:0/task:0/device:CPU:1"} : (!corert.tensorhandle) -> !tfrt_fallback.tf_tensor

  // CHECK: Slept for 1000 microseconds
  // CHECK: Marked AsyncTensor available.

  tfrt.return
}
