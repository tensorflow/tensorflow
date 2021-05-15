/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

include "mlir/Pass/PassBase.td"

// TF device dialect passes.

def ResourceOpLiftingPass : Pass<"tf-resource-op-lifting", "ModuleOp"> {
  let summary = "Lifting resource operations out of device computation";
  let description = [{
    This pass lifts resource variable operations outside of device computation.
    This is useful because a lot of accelerator devices can not interact with
    resource variables directly..

    Here is a simple example in TensorFlow where a device doubles the value of a
    TensorFlow resource variable and returns new value:

    ```mlir
    %resource_handle = "tf.VarHandleOp"()
    %1 = "tf_device.cluster"() ( {
      %init_value = "tf.ReadVariableOp"(%resource_handle)
      "tf.AssignAddVariableOp"(%resource_handle, %init_value)
      %new_value = "tf.ReadVariableOp"(%resource_handle)
      tf_device.return %new_value
    })
    ```

    After this pass, the computation would become:

    ```mlir
    %resource_handle = "tf.VarHandleOp"()
    %init_value = "tf.ReadVariableOp"(%resource_handle)
    %1:2 = "tf_device.cluster"() ( {
      %new_value = "tf.AddV2"(%init_value, %init_value)
      tf_device.return %new_value, %new_value
    })
    "tf.AssignVariableOp"(%resource_handle, %1#1)
    ```

    You can see that there are a few main changes applied:
    1) All the resource variable reads and writes are now outside of
       tf_device.cluster op.
    2) Instead of taking resource handles as input, this device computation now
       takes snapshotted values of that device.
    3) Some resource load operations are eliminated with store-load forwarding.
    4) Updated values to resource are appended to `tf_device.return` and used by
       external resource store operations so that resources are still updated
       after the computation.

    If the cluster body contains functional control flow, the pass first lifts
    the loads/stores in the body/cond/branch functions to the cluster body, then
    performs the above lifting. E.g.,

    ```mlir
    func @cluster_with_loop() -> () {
      %0 = "tf.VarHandleOp"() ...
      "tf_device.cluster"() ( {
         %1 = "tf.While"(%0) {body = @while_body, cond = @while_cond}
         tf_device.return
      })
      return
    }
    func @while_body(%arg0: tensor<*x!tf.resource<tensor<f32>>>) {
     %constant = "tf.Const"() ...
     "tf.AssignVariableOp"(%arg0, %constant)
     return %arg0
    }
    func @while_cond(%arg0: tensor<*x!tf.resource<tensor<f32>>>) {
      %read = "tf.ReadVariableOp"(%arg0)
      return %read
    }
    ```

    will be transformed to:

    ```mlir
    func @cluster_with_loop() {
      %0 = "tf.VarHandleOp"() ...
      %1 = "tf.ReadVariableOp"(%0)
      %2 = "tf_device.cluster"() ( {
        %3 = "tf.While"(%1) {body = @while_body, cond = @while_cond}
        tf_device.return %3 : tensor<f32>
      }) : () -> tensor<f32>
      "tf.AssignVariableOp"(%0, %2)
      return
    }
    func @while_body(%arg0: tensor<f32>) {
      %0 = "tf.Const"() ...
      return %0 : tensor<f32>
    }
    func @while_cond(%arg0: tensor<f32>) {
      return %arg0
    }
    ```
  }];

  let constructor = "TFDevice::CreateResourceOpLiftingPass()";
}

def ResourceOpLiftingForMainFunctionPass :
    Pass<"tf-resource-op-lifting-for-main-function", "ModuleOp"> {
  let summary = "Lifting resource operations out of control flow statements "
    "for the main function";
  let constructor = "TFDevice::CreateResourceOpLiftingForMainFunctionPass()";
}

def AnnotateParameterReplicationPass :
    Pass<"tf-annotate-parameter-replication", "ModuleOp"> {
  let summary = "Annotate whether a ClusterFuncOp's parameters have the same data across replicas.";
  let constructor = "TFDevice::CreateAnnotateParameterReplicationPass()";
}
