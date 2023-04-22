// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-guarantee-all-funcs-one-use | FileCheck %s

// -----
// Basic test.
// CHECK-LABEL: func @f
func @f() {
  // CHECK: call @g() : () -> ()
  // CHECK: call @[[NEWG:.+]]() : () -> ()
  call @g() : () -> ()
  call @g() : () -> ()
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
  call @g() : () -> ()
  call @g() : () -> ()
  return
}

func @g() {
  call @h() : () -> ()
  call @h() : () -> ()
  return
}

func @h() {
  return
}

// -----
// Handle error case of infinite recursion.
// expected-error @+1 {{reached cloning limit}}
func private @f() {
  call @f() : () -> ()
  call @f() : () -> ()
  return
}
