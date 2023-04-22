// RUN: not tf-mlir-translate -split-input-file -mlir-hlo-to-hlo-text %s 2>&1 | FileCheck %s

// CHECK: conversion requires module with `main`
func @non_main() {
  %0 = "mhlo.constant"() {value = opaque<"mhlo", "0x0123456789ABCDEF"> : tensor<4xf32>} : () -> tensor<4xf32>
  return
}
