// RUN: mlir-opt %s -verify-diagnostics -mlir-print-op-on-diagnostic

// This file tests the functionality of 'mlir-print-op-on-diagnostic'.

// expected-error@below {{invalid to use 'test.invalid_attr'}}
// expected-note@below {{see current operation: "module"()}}
module attributes {test.invalid_attr} {}
