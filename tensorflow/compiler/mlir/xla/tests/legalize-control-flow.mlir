// RUN: tf-opt -xla-legalize-control-flow %s -o - | FileCheck %s

// CHECK-LABEL: func @cond(%arg0: tensor<i64>) -> tensor<i1> {
func @cond(%arg0: tensor<i64>) -> tensor<i1> {
  // CHECK-NEXT: %0 = "xla_hlo.compare"(%arg0, %arg0) {comparison_direction = "LT", name = "compare.2"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %0 = "xla_hlo.compare"(%arg0, %arg0) {comparison_direction = "LT", name = "compare.2"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  // CHECK-NEXT: return %0 : tensor<i1>
  return %0 : tensor<i1>
}

// CHECK-LABEL: func @loop(%arg0: tensor<i64>) -> tensor<i64> {
func @loop(%arg0: tensor<i64>) -> tensor<i64> {
  // CHECK-NEXT: %0 = xla_hlo.add %arg0, %arg0 {name = "compare.0"} : tensor<i64>
  %0 = "xla_hlo.add"(%arg0, %arg0) {name = "compare.0"} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  // CHECK-NEXT: return %0 : tensor<i64>
  return %0 : tensor<i64>
}

// CHECK-LABEL: func @main(%arg0: tensor<i64>) -> tensor<i64> {
func @main(%arg0: tensor<i64>) -> tensor<i64> {
  // CHECK-NEXT:   br ^bb1(%arg0 : tensor<i64>)
  // CHECK-NEXT: b1(%0: tensor<i64>):
  // CHECK-NEXT:   %1 = call @cond(%0) : (tensor<i64>) -> tensor<i1>
  // CHECK-NEXT:   %2 = extract_element %1[] : tensor<i1>
  // CHECK-NEXT:   cond_br %2, ^bb2(%0 : tensor<i64>), ^bb3(%0 : tensor<i64>)
  // CHECK-NEXT: b2(%3: tensor<i64>):	// pred: ^bb1
  // CHECK-NEXT:   %4 = call @loop(%3) : (tensor<i64>) -> tensor<i64>
  // CHECK-NEXT:   br ^bb1(%4 : tensor<i64>)
  // CHECK-NEXT: b3(%5: tensor<i64>):	// pred: ^bb1
  %0 = "xla_hlo.while"(%arg0) {body = @loop, cond = @cond} : (tensor<i64>) -> tensor<i64>
  // CHECK-NEXT:   return %5 : tensor<i64>
  return %0 : tensor<i64>
}
