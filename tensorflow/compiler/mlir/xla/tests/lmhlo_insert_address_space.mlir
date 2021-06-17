// mlir-hlo-opt --insert-address-space %s | FileCheck %s

func @main() {
  // CHECK: memref.alloc({{.*}}) : memref<?x?xf32, "device_type_2df32">
  // CHECK: memref.alloc({{.*}}) : memref<?x?xf32, "device_type_2df32">
  // CHECK: memref.alloc({{.*}}) : memref<?x?xf32, "device_type_2df32">
  // CHECK: memref.alloc({{.*}}) : memref<?x?xf32, "device_type_2df32">
  // CHECK: call @fusion_memref({{.*}}) : (memref<?x?xf32, "device_type_2df32">, memref<?x?xf32, "device_type_2df32">, memref<?x?xf32, "device_type_2df32">, memref<?x?xf32, "device_type_2df32">) -> ()
  %c4 = constant 1024 : index
  %c5 = constant 1024 : index
  %arg0 = memref.alloc(%c4, %c5) {mhlo_place_type = "device"} : memref<?x?xf32>
  %arg1 = memref.alloc(%c4, %c5) {mhlo_place_type = "device"} : memref<?x?xf32>
  %arg2 = memref.alloc(%c4, %c5) {mhlo_place_type = "device"} : memref<?x?xf32>
  %arg3 = memref.alloc(%c4, %c5) {mhlo_place_type = "device"} : memref<?x?xf32>
  call @fusion_memref(%arg0, %arg1, %arg2, %arg3) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  return
}

// CHECK-LABEL: func @fusion_memref
func @fusion_memref(%input1: memref<?x?xf32>, %input2: memref<?x?xf32>, %input3: memref<?x?xf32>, %out: memref<?x?xf32>) -> () {
  "lmhlo.fusion"() ( {
    // CHECK: memref.tensor_load {{.*}} : memref<?x?xf32, "device_type_2df32">
    // CHECK: memref.tensor_load {{.*}} : memref<?x?xf32, "device_type_2df32">
    // CHECK: mhlo.add {{.*}} {name = "add"} : tensor<?x?xf32>
    // CHECK: memref.tensor_load {{.*}} : memref<?x?xf32, "device_type_2df32">
    // CHECK: mhlo.multiply {{.*}} {name = "multiply"} : tensor<?x?xf32>
    // CHECK: memref.tensor_store {{.*}} : memref<?x?xf32, "device_type_2df32">

    %0 = memref.tensor_load %input1 : memref<?x?xf32>
    %1 = memref.tensor_load %input2 : memref<?x?xf32>
    %2 = "mhlo.add"(%0, %1) {name = "add"} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    %3 = memref.tensor_load %input3 : memref<?x?xf32>
    %4 = "mhlo.multiply"(%2, %3) {name = "multiply"} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    memref.tensor_store %4, %out : memref<?x?xf32>
    "lmhlo.terminator"() : () -> ()
  } ) : () -> ()
  return
}