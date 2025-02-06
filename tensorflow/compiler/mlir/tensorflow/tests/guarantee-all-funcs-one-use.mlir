// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-guarantee-all-funcs-one-use | FileCheck %s

// -----
// Basic test.
// CHECK-LABEL: func @f
func.func @f() {
  // CHECK: call @g() : () -> ()
  // CHECK: call @[[NEWG:.+]]() : () -> ()
  func.call @g() : () -> ()
  func.call @g() : () -> ()
  func.return
}

// CHECK: func @g()
// CHECK: func private @[[NEWG]]()
func.func @g() {
  func.return
}

// -----
// Transitive callees.
// CHECK-LABEL: func @f
// 2 copies of @g
// CHECK-DAG: func @g{{.*}}
// CHECK-DAG: func private @g{{.*}}
// 4 copies of @h
// CHECK-DAG: func @h{{.*}}
// CHECK-DAG: func private @h{{.*}}
// CHECK-DAG: func private @h{{.*}}
// CHECK-DAG: func private @h{{.*}}
func.func @f() {
  func.call @g() : () -> ()
  func.call @g() : () -> ()
  func.return
}

func.func @g() {
  func.call @h() : () -> ()
  func.call @h() : () -> ()
  func.return
}

func.func @h() {
  func.return
}

// -----
// Handle error case of infinite recursion.
// expected-error @+1 {{recursive call graph cannot be transformed}}
module {
  func.func private @f() {
    func.call @f() : () -> ()
    func.call @f() : () -> ()
    func.return
  }
}

// -----
// Handle error case of infinite recursion with mutually recursive ops.
// expected-error @+1 {{recursive call graph cannot be transformed}}
module {
  func.func private @f() {
    func.call @g() : () -> ()
    func.return
  }
  func.func private @g() {
    func.call @f() : () -> ()
    func.return
  }
}

// -----
// Test stateful and stateless partitioned calls.
// CHECK-LABEL: func @f
func.func @f() {
  // CHECK: "tf.PartitionedCall"() <{config = "",  config_proto = "", executor_type = "", f = @g}> : () -> ()
  "tf.PartitionedCall"() {config = "",  config_proto = "", executor_type = "", f = @g} : () -> ()
  // CHECK: "tf.StatefulPartitionedCall"() <{config = "",  config_proto = "", executor_type = "", f = @[[NEWG:.+]]}> : () -> ()
  "tf.StatefulPartitionedCall"() {config = "",  config_proto = "", executor_type = "", f = @g} : () -> ()
  func.return
}

// CHECK: func.func @g()
// CHECK: func.func private @[[NEWG]]()
func.func @g() {
  func.return
}

