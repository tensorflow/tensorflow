// RUN: tfg-opt-no-passes %s --split-input-file --verify-diagnostics

// Exercise some basic custom syntax

// -----

// expected-error @+2 {{invalid kind of attribute specified}}
// expected-error @+1 {{expected a version attribute}}
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
