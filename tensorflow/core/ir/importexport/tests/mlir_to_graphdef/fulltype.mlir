// RUN: tfg-translate -mlir-to-graphdef %s | FileCheck %s

// Verifies integer fulltype correctly exported.

// CHECK: experimental_type {
// CHECK-NEXT: i: 0
// CHECK: }

tfg.graph #tf_type.version<producer = 527, min_consumer = 12> {
  %VarHandleOp, %ctl = VarHandleOp device("/job:localhost/replica:0/task:0/device:CPU:0") name("Variable") {_class = ["loc:@Variable"], _output_shapes = [#tf_type.shape<*>], allowed_devices = [], container = "", dtype = !tf_type.string, shape = #tf_type.shape<>, shared_name = "VOriabAle"} : () -> (tensor<*x!tf_type.resource>)
  %Placeholder, %ctl_0 = Placeholder device("/job:localhost/replica:0/task:0/device:CPU:0") name("asset_path_initializer") {_output_shapes = [#tf_type.func<@__inference__traced_restore_52, {}>], dtype = !tf_type.string, shape = #tf_type.shape<>} : () -> (tensor<!tf_type.string>)
  %ctl_1 = AssignVariableOp(%VarHandleOp, %Placeholder) device("/job:localhost/replica:0/task:0/device:CPU:0") name("Variable/Assign") {dtype = !tf_type.string, validate_shape = false} : tensor<*x!tf_type.resource>, tensor<!tf_type.string>
  %ctl_2 = NoOp [%ctl_1] device("/device:CPU:0") name("NoOp") {_mlir_fulltype = #tf_type.full_type<unset 0 : i64>}
}

