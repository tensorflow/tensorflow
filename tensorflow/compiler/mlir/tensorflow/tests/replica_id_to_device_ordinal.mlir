// Copyright 2026 Google Inc. All Rights Reserved.
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
// RUN: tf-opt -split-input-file -verify-diagnostics %s -tf-replica-id-to-device-ordinal | FileCheck %s


// Tests device ordinal is set correctly for multiple devices.
// CHECK-LABEL: func @device_ordinal_attr_added_multiple_devices
module attributes {tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0", "/job:worker/replica:0/task:0/device:TPU:1"]} {
  func.func @device_ordinal_attr_added_multiple_devices() {
    tf_executor.graph {
      %0 = tf_executor.island {
        "tf_device.launch"() ({
          %1 = "tf.opA"() : () -> tensor<!tf_type.string>
          "tf.EnqueueTPUEmbeddingArbitraryTensorBatch"(%1){_xla_replica_id = 0 : i64, device_ordinal = -1 : i64} : (tensor<!tf_type.string>) -> ()
          tf_device.return
        }) {device = "/job:worker/replica:0/task:0/device:CPU:0"} : () -> ()
        "tf_device.launch"() ({
          %1 = "tf.opA"() : () -> tensor<!tf_type.string>
          "tf.EnqueueTPUEmbeddingArbitraryTensorBatch"(%1){_xla_replica_id = 1 : i64, device_ordinal = -1 : i64} : (tensor<!tf_type.string>) -> ()
          tf_device.return
        }) {device = "/job:worker/replica:0/task:0/device:CPU:0"} : () -> ()
        "tf_device.launch"() ({
          %1 = "tf.opA"() : () -> tensor<!tf_type.string>
          "tf.EnqueueTPUEmbeddingArbitraryTensorBatch"(%1){_xla_replica_id = 2 : i64, device_ordinal = -1 : i64} : (tensor<!tf_type.string>) -> ()
          tf_device.return
        }) {device = "/job:worker/replica:0/task:0/device:CPU:0"} : () -> ()
        "tf_device.launch"() ({
          %1 = "tf.opA"() : () -> tensor<!tf_type.string>
          "tf.EnqueueTPUEmbeddingArbitraryTensorBatch"(%1){_xla_replica_id = 3 : i64, device_ordinal = -1 : i64} : (tensor<!tf_type.string>) -> ()
          tf_device.return
        }) {device = "/job:worker/replica:0/task:0/device:CPU:0"} : () -> ()
        tf_executor.yield
      }
      tf_executor.fetch
    }
    func.return
  }

  // CHECK:      tf_executor.island
  // CHECK:      "tf.EnqueueTPUEmbeddingArbitraryTensorBatch"
  // CHECK-SAME:   device_ordinal = 0
  // CHECK:      "tf.EnqueueTPUEmbeddingArbitraryTensorBatch"
  // CHECK-SAME:   device_ordinal = 1
  // CHECK:      "tf.EnqueueTPUEmbeddingArbitraryTensorBatch"
  // CHECK-SAME:   device_ordinal = 0
  // CHECK:      "tf.EnqueueTPUEmbeddingArbitraryTensorBatch"
  // CHECK-SAME:   device_ordinal = 1
}

// -----


// Tests device ordinal is set correctly for single device.
// CHECK-LABEL: func @device_ordinal_attr_added_single_device
module attributes {tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func.func @device_ordinal_attr_added_single_device() {
    tf_executor.graph {
      %0 = tf_executor.island {
        "tf_device.launch"() ({
          %1 = "tf.opA"() : () -> tensor<!tf_type.string>
          "tf.EnqueueTPUEmbeddingArbitraryTensorBatch"(%1){_xla_replica_id = 0 : i64, device_ordinal = -1 : i64} : (tensor<!tf_type.string>) -> ()
          tf_device.return
        }) {device = "/job:worker/replica:0/task:0/device:CPU:0"} : () -> ()
        tf_executor.yield
      }
      tf_executor.fetch
    }
    func.return
  }
  // CHECK:      tf_executor.island
  // CHECK:      "tf.EnqueueTPUEmbeddingArbitraryTensorBatch"
  // CHECK-SAME:   device_ordinal = 0
}

// -----

// Tests device ordinal is not set for no tpu device.
// CHECK-LABEL: func @device_ordinal_attr_not_added_no_tpu_device
module attributes {tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0"]} {
  func.func @device_ordinal_attr_not_added_no_tpu_device() {
    tf_executor.graph {
      %0 = tf_executor.island {
        "tf_device.launch"() ({
          %1 = "tf.opA"() : () -> tensor<!tf_type.string>
          "tf.EnqueueTPUEmbeddingArbitraryTensorBatch"(%1){_xla_replica_id = 0 : i64, device_ordinal = -1 : i64} : (tensor<!tf_type.string>) -> ()
          tf_device.return
        }) {device = "/job:worker/replica:0/task:0/device:CPU:0"} : () -> ()
        tf_executor.yield
      }
      tf_executor.fetch
    }
    func.return
  }
  // CHECK:      tf_executor.island
  // CHECK:      "tf.EnqueueTPUEmbeddingArbitraryTensorBatch"
  // CHECK-SAME:   device_ordinal = -1
}

// -----

