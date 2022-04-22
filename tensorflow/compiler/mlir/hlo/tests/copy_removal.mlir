// RUN: mlir-hlo-opt -copy-removal -allow-unregistered-dialect -split-input-file %s | FileCheck %s

//CHECK-LABEL: @parameter_not_removed
func.func @parameter_not_removed(%in : memref<42xf32>) -> memref<42xf32> {
  // CHECK: memref.copy
  %0 = memref.alloc() : memref<42xf32>
  memref.copy %in, %0 : memref<42xf32> to memref<42xf32>
  memref.dealloc %in : memref<42xf32>
  func.return %0 : memref<42xf32>
}

// -----

// CHECK-LABEL: block_local_removed
func.func @block_local_removed() {
  // CHECK-NOT: memref.copy
  %0 = memref.alloc() : memref<42xf32>
  %1 = memref.alloc() : memref<42xf32>
  memref.copy %0, %1 : memref<42xf32> to memref<42xf32>
  memref.dealloc %0 : memref<42xf32>
  "use.use"(%1) : (memref<42xf32>) -> ()
  memref.dealloc %1 : memref<42xf32>
  func.return
}

// -----

// CHECK-LABEL: conflicting_use
func.func @conflicting_use() {
  // CHECK: memref.copy
  %0 = memref.alloc() : memref<42xf32>
  %1 = memref.alloc() : memref<42xf32>
  memref.copy %0, %1 : memref<42xf32> to memref<42xf32>
  "use.use"(%1) : (memref<42xf32>) -> ()
  memref.dealloc %0 : memref<42xf32>
  "use.use"(%0) : (memref<42xf32>) -> ()
  memref.dealloc %1 : memref<42xf32>
  func.return
}

// -----

// CHECK-LABEL: incompatible_maps
func.func @incompatible_maps() {
  // CHECK: memref.copy
  %0 = memref.alloc() : memref<42xf32>
  %1 = memref.alloc() : memref<21xf32>
  %2 = memref.reinterpret_cast %0 to offset: [0], sizes: [21], strides: [2]
      : memref<42xf32> to memref<21xf32, offset: 0, strides: [2]>
  memref.copy %2, %1 : memref<21xf32, offset: 0, strides: [2]> to memref<21xf32>
  memref.dealloc %0 : memref<42xf32>
  "use.use"(%1) : (memref<21xf32>) -> ()
  memref.dealloc %1 : memref<21xf32>
  func.return
}

// -----

// CHECK-LABEL: compatible_maps
func.func @compatible_maps() {
  // CHECK-NOT: memref.copy
  %0 = memref.alloc() : memref<42xf32>
  %1 = memref.alloc() : memref<42xf32>
  %2 = memref.reinterpret_cast %0 to offset: [0], sizes: [21], strides: [2]
      : memref<42xf32> to memref<21xf32, offset: 0, strides: [2]>
  %3 = memref.reinterpret_cast %1 to offset: [0], sizes: [21], strides: [2]
      : memref<42xf32> to memref<21xf32, offset: 0, strides: [2]>
  memref.copy %2, %3 : memref<21xf32, offset: 0, strides: [2]>
      to memref<21xf32, offset: 0, strides: [2]>
  memref.dealloc %0 : memref<42xf32>
  "use.use"(%3) : (memref<21xf32, offset: 0, strides: [2]>) -> ()
  memref.dealloc %1 : memref<42xf32>
  func.return
}

// -----

// CHECK-LABEL: conflicting_alias_use
func.func @conflicting_alias_use() {
  // CHECK: memref.copy
  %0 = memref.alloc() : memref<42xf32>
  %1 = memref.alloc() : memref<42xf32>
  %2 = memref.reinterpret_cast %0 to offset: [0], sizes: [21], strides: [2]
      : memref<42xf32> to memref<21xf32, offset: 0, strides: [2]>
  %3 = memref.reinterpret_cast %1 to offset: [0], sizes: [21], strides: [2]
      : memref<42xf32> to memref<21xf32, offset: 0, strides: [2]>
  memref.copy %2, %3 : memref<21xf32, offset: 0, strides: [2]>
      to memref<21xf32, offset: 0, strides: [2]>
  "use.use"(%0) : (memref<42xf32>) -> ()
  memref.dealloc %0 : memref<42xf32>
  "use.use"(%3) : (memref<21xf32, offset: 0, strides: [2]>) -> ()
  memref.dealloc %1 : memref<42xf32>
  func.return
}
