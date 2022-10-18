// RUN: not tf-mlir-translate -split-input-file -mlir-hlo-to-hlo-text %s 2>&1 | FileCheck %s

// CHECK: conversion requires module with `main`
func.func @non_main() {
  %0 = "mhlo.constant"() {value = dense_resource<__elided__> : tensor<4xf32>} : () -> tensor<4xf32>
  func.return
}
