// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-guarantee-all-funcs-one-use | FileCheck %s

// -----
// Basic test.
// CHECK-LABEL: func @f
func @f() {
  // CHECK: call @g() : () -> ()
  // CHECK: call @[[NEWG:.+]]() : () -> ()
  func.call @g() : () -> ()
  func.call @g() : () -> ()
  return
}

// CHECK: func @g()
// CHECK: func private @[[NEWG]]()
func @g() {
  return
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
func @f() {
  func.call @g() : () -> ()
  func.call @g() : () -> ()
  return
}

func @g() {
  func.call @h() : () -> ()
  func.call @h() : () -> ()
  return
}

func @h() {
  return
}

// -----
// Handle error case of infinite recursion.
// expected-error @+1 {{reached cloning limit}}
func private @f() {
  func.call @f() : () -> ()
  func.call @f() : () -> ()
  return
}
