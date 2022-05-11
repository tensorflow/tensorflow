// RUN: tfg-opt-no-passes %s --split-input-file --verify-diagnostics

// Exercise some basic custom syntax

// -----

// expected-error @+1 {{invalid kind of attribute specified}}
tfg.graph {
}

// -----

// expected-error @+1 {{expects a non-empty block}}
tfg.func @foo(%arg0 : tensor<10xf32>) -> (tensor<10xf32>) {
}

// -----

// expected-error @+1 {{expects body to be terminated with a tfg.return, but got: tfg.Op}}
tfg.func @foo(%arg0 : tensor<10xf32>) -> (tensor<10xf32>) {
  %ctl = "tfg.Op"() : () -> (!tf_type.control)
}

// -----

tfg.func @foo(%arg0 : tensor<10xf32>) {
// expected-error @+1 {{found non-control input in position #1 after control input in position #0}}
  %ctl = "tfg.Op"(%arg0.ctl, %arg0) : (!tf_type.control, tensor<10xf32>) -> (!tf_type.control)
  tfg.return : () -> ()
}

// -----

tfg.func @foo() {
// expected-error @+1 {{found non-control result in position #1 after control result in position #0}}
  %res:2, %ctl = "tfg.Op"() : () -> (!tf_type.control, tensor<10xf32>, !tf_type.control)
  tfg.return : () -> ()
}

// -----

tfg.graph  #tf_type.version<producer = 42, min_consumer = 33> {
// expected-error @+1 {{found non-control result in position #1 after control result in position #0}}
  %res:2, %ctl = "tfg.Op"() : () -> (!tf_type.control, tensor<10xf32>, !tf_type.control)
}

// -----

tfg.func @test_control_ret_attrs() {
  %ctl = NoOp
  %ctl_0 = NoOp
// expected-error @+1 {{expected as many control result attributes as there are control operands}}
  "tfg.return"(%ctl, %ctl_0) {control_ret_attrs = [{tfg.name = "foo"}]} : (!tf_type.control, !tf_type.control) -> ()
}
