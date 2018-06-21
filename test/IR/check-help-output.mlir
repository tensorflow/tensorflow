// RUN: %mlir-opt --help | grep 'MLIR modular optimizer'
// RUN: %mlir-opt --help | FileCheck %s
//
// CHECK: OVERVIEW: MLIR modular optimizer driver

extfunc @test()

