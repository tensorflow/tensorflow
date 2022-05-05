// RUN: tfg-translate -mlir-to-graphdef %s | FileCheck %s

// Verifies no default value attributes are exported

// CHECK-NOT: container
// CHECK-NOT: shared_name
// CHECK-NOT: validate_shape

tfg.graph #tf_type.version<producer = 527, min_consumer = 12> {
  %VarHandleOp, %ctl = VarHandleOp device("/job:localhost/replica:0/task:0/device:CPU:0") name("Variable") {_class = ["loc:@Variable"], allowed_devices = [], container = "", dtype = !tf_type.string, shape = #tf_type.shape<>, shared_name = ""} : () -> (tensor<*x!tf_type.resource>)
  %Placeholder, %ctl_0 = Placeholder device("/job:localhost/replica:0/task:0/device:CPU:0") name("asset_path_initializer") {dtype = !tf_type.string, shape = #tf_type.shape<>} : () -> (tensor<!tf_type.string>)
  %ctl_1 = AssignVariableOp(%VarHandleOp, %Placeholder) device("/job:localhost/replica:0/task:0/device:CPU:0") name("Variable/Assign") {allowed_devices = [], dtype = !tf_type.string, validate_shape = false} : tensor<*x!tf_type.resource>, tensor<!tf_type.string>
}
