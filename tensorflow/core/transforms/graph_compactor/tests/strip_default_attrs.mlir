// RUN: tfg-transforms-opt --tfg-strip-default-attrs %s | FileCheck %s

// CHECK-LABEL: tfg.graph
tfg.graph #tf_type.version<producer = 1, min_consumer = 1> {
  // CHECK: VarHandleOp
  // CHECK-NOT: container
  // CHECK-NOT: shared_name
  // CHECK-NOT: allowed_devices
  // CHECK: dtype
  // CHECK: shape
  %Var, %ctl = VarHandleOp {
    container = "", shared_name = "", dtype = i32, 
    shape = #tf_type.shape<4x4>, allowed_devices = []
  } : () -> (tensor<*x!tf_type.resource>)
}

// CHECK-LABEL: tfg.graph
tfg.graph #tf_type.version<producer = 1, min_consumer = 1> {
  // CHECK: VarHandleOp
  // CHECK: container
  // CHECK-NOT: allowed_devices
  // CHECK: dtype
  // CHECK: shape
  // CHECK: shared_name
  %Var, %ctl = VarHandleOp {
    container = "foo", shared_name = "bar", dtype = i32, 
    shape = #tf_type.shape<4x4>, allowed_devices = []
  } : () -> (tensor<*x!tf_type.resource>)
}