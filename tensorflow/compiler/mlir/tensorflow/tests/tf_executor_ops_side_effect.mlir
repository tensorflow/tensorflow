// RUN: tf-opt %s -cse  | FileCheck %s
// Checks that CSE runs without generating invalid IR and doesn't CSE ops like
// NextIteration.Source and NextIteration.Sink.
// CHECK-LABEL: func @main
// CHECK: tf_executor.NextIteration.Source
// CHECK: tf_executor.NextIteration.Source
// CHECK: tf_executor.NextIteration.Sink
// CHECK: tf_executor.NextIteration.Sink
func.func @main() -> (tensor<1xi32>, tensor<1xi32>) {
  %0, %1 = tf_executor.graph {
    %output_1, %token_1, %control_1 = tf_executor.NextIteration.Source : tensor<1xi32> {T = i32, device = ""}
    %output_2, %token_2, %control_2 = tf_executor.NextIteration.Source : tensor<1xi32> {T = i32, device = ""}
    tf_executor.NextIteration.Sink [%token_1] %output_1 : tensor<1xi32> {T = i32, device = ""}
    tf_executor.NextIteration.Sink [%token_2] %output_2 : tensor<1xi32> {T = i32, device = ""}
    tf_executor.fetch %output_1, %output_2 : tensor<1xi32>, tensor<1xi32>
  }
  func.return %0, %1 : tensor<1xi32>, tensor<1xi32>
}

