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
    func @while_body(%arg0: tensor<*x!tf_type.resource<tensor<f32>>>) {
     %constant = "tf.Const"() ...
     "tf.AssignVariableOp"(%arg0, %constant)
     return %arg0
    }
    func @while_cond(%arg0: tensor<*x!tf_type.resource<tensor<f32>>>) {
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

def DeviceAttributeToLaunchPass : FunctionPass<"tf-device-attribute-to-launch"> {
  let summary = "Wraps each TF op which has a non-empty device attribute in a tf_device.launch.";

  let description = [{
    This pass wraps TF ops which have a non-empty device attribute in a tf_device.lauch with
    the same device attribute.

    For example, the following:

    ```mlir
    func @single_op_launch() {
      %a = "tf.opA"() {device = "CPU:0"} : () -> tensor<i1>
      return %a
    }
    ```

    will be transformed into:

    ```mlir
    func @single_op_launch() {
      %1 = tf_device.launch() ( {
        %a = "tf.opA"() : () -> tensor<i1>
        tf_device.return %a
      }) {device = "CPU:0"} : () -> tensor<i1>
      return %1
    }
    ```
  }];

  let constructor = "TFDevice::CreateDeviceAttributeToLaunchPass()";
}

def HostLaunchToOutsideCompiledPass : Pass<"tf-device-host-launch-to-outside-compiled", "ModuleOp"> {
  let summary = "Converts each op wrapped in launch op with host device assignnment to op with _xla_outside_compiled attribute.";

  let description = [{
    This pass takes ops wrapped in a tf_device.launch op with host device
    assignment extracts them from launch and adds an `_xla_outside_compilation`
    attribute. This is the inverse of OutsideCompiledToHostLaunchPass.

    A simple example:

    ```mlir
      "tf_device.cluster"() ( {
        "tf.A"()
        "tf_device.launch"() {
          "tf.B"()
          tf_device.return
        } {device = "TPU_REPLICATED_HOST"} : () -> ()
        "tf.C"()
        tf_device.return
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []}
    ```

    Would become the following ops (unimportant attribute, type are omitted):

    ```mlir
      "tf_device.cluster"() ( {
        "tf.A"()
        "tf.B"() {_xla_outside_compilation = "cluster1"}
        "tf.C"()
        tf_device.return
      }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []}
    ```
  }];

  let constructor = "TFDevice::CreateHostLaunchToOutsideCompiledPass()";
}

def DecomposeResourceOpsPass : FunctionPass<"tf-device-decompose-resource-ops"> {
  let summary = "Decompose composite resource variable operations into primitive Read/AssignVariableOp and raw computation.";

  let description = [{
    A pass that decomposes composite resource operations into primitive ones like
    ReadVariableOp, AssignVariableOp and other computations to facilitate
    transformations like resource op lifting.

    For example:

    ```mlir
    tf.AssignAddVariableOp(%res, %0)
    ```

    Becomes

    ```mlir
    %res_val = tf.ReadVariableOp(%res)
    %1 = tf.AddV2(%res_val, %0)
    tf.AssignVariableOp(%res, %1)
    ```

    NOTE: This pass does not support `use_locking=true` for a lot of resource
    operations. So decomposition may not be correct outside of backends like XLA,
    which automatically locks all resource variables.
  }];

  let constructor = "TFDevice::CreateDecomposeResourceOpsPass()";
}

def DecomposeResourceOpsInClusterPass : Pass<"tf-device-decompose-resource-ops-in-cluster", "ModuleOp"> {
  let summary = "Decompose composite resource variable operations into primitive Read/AssignVariableOp and raw computation within device cluster and reachable functions.";

  let description = [{
    This is a variation of DecomposeResourceOps pass. It operates on a module and
    only decomposes ops within a device cluster (tf_device.cluster op) and any
    functions reachable from the cluster.
  }];

  let constructor = "TFDevice::CreateDecomposeResourceOpsInClusterPass()";
}

def LaunchToDeviceAttributePass : FunctionPass<"tf-launch-to-device-attribute"> {
  let summary = "Hoists and annotates device launch inner ops with associated device attribute.";

  let description = [{
    This pass hoists a `tf_device.launch` body and assigns a `device` attribute
    to each TensorFlow dialect op in the body based on the `device` attribute on
    the `tf_device.launch`. If a TensorFlow dialect op already has a device
    attribute, that attribute will be overwritten with the `tf_device.launch`
    device.

    For example:

    ```mlir
      %island:5 = tf_executor.island {
        %a = "tf.opA"() : () -> tensor<i1>
        %launch:2 = "tf_device.launch"() ( {
          %b = "tf.opB"() : () -> tensor<i32>
          %c = "tf.opC"() : () -> tensor<f32>
          tf_device.return %c, %b : tensor<f32>, tensor<i32>
        }) {device = "CPU:0"} : () -> (tensor<f32>, tensor<i32>)
        %d = "tf.opD"() : () -> tensor<i1>
        tf_executor.yield %a, %launch#0, %launch#1, %d :
                          tensor<i1>, tensor<f32>, tensor<i32>, tensor<i1>
      }
    ```

    Will be transformed into:

    ```mlir
      %island:5 = tf_executor.island {
        %a = "tf.opA"() : () -> tensor<i1>
        %b = "tf.opB"() {device = "CPU:0"} : () -> tensor<i32>
        %c = "tf.opC"() {device = "CPU:0"} : () -> tensor<f32>
        %d = "tf.opD"() : () -> tensor<i1>
        tf_executor.yield %a, %c, %b, %d :
                          tensor<i1>, tensor<f32>, tensor<i32>, tensor<i1>
      }
    ```
  }];

  let constructor = "TFDevice::CreateLaunchToDeviceAttributePass()";
}
