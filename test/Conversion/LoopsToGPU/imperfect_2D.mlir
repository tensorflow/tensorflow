// RUN: mlir-opt -convert-loop-op-to-gpu -gpu-num-workgroups=2,2 -gpu-workgroup-size=32,4 %s | FileCheck %s

module {
  // arg2 = arg0 * transpose(arg1) ; with intermediate buffer and tile size passed as argument
  // CHECK: func {{@.*}}([[ARG0:%.*]]: memref<?x?xf32>, [[ARG1:%.*]]: memref<?x?xf32>, [[ARG2:%.*]]: memref<?x?xf32>, [[ARG3:%.*]]: index, [[ARG4:%.*]]: index)
  func @foo(%arg0: memref<?x?xf32>, %arg1 : memref<?x?xf32>, %arg2 : memref<?x?xf32>, %arg3 : index, %arg4 : index) {
    %0 = dim %arg0, 0 : memref<?x?xf32>
    %1 = dim %arg0, 1 : memref<?x?xf32>
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    // CHECK: gpu.launch blocks([[ARG5:%.*]], [[ARG6:%.*]], [[ARG7:%.*]]) in ([[ARG11:%.*]] = {{%.*}}, [[ARG12:%.*]] = {{%.*}}, [[ARG13:%.*]] = {{%.*}}) threads([[ARG8:%.*]], [[ARG9:%.*]], [[ARG10:%.*]]) in ([[ARG14:%.*]] = {{%.*}}, [[ARG15:%.*]] = {{%.*}}, [[ARG16:%.*]] = {{%.*}}) args([[ARG17:%.*]] = [[ARG3]], [[ARG18:%.*]] = [[ARG4]], [[ARG19:%.*]] = [[ARG1]], [[ARG20:%.*]] = {{%.*}}, {{%.*}} = {{%.*}}, [[ARG22:%.*]] = [[ARG0]], [[ARG23:%.*]] = [[ARG2]]
    // CHECK: [[TEMP1:%.*]] = muli [[ARG17]], [[ARG6]] : index
    // CHECK: [[BLOCKLOOPYLB:%.*]] = addi {{%.*}}, [[TEMP1]] : index
    // CHECK: [[BLOCKLOOPYSTEP:%.*]] = muli [[ARG17]], [[ARG12]] : index
    // CHECK: loop.for [[BLOCKLOOPYIV:%.*]] = [[BLOCKLOOPYLB]] to {{%.*}} step [[BLOCKLOOPYSTEP]]
    loop.for %iv1 = %c0 to %0 step %arg3 {

      // CHECK: [[TEMP2:%.*]] = muli [[ARG18]], [[ARG5]] : index
      // CHECK: [[BLOCKLOOPXLB:%.*]] = addi  {{%.*}}, [[TEMP2]] : index
      // CHECK: [[BLOCKLOOPXSTEP:%.*]] = muli [[ARG18]], [[ARG11]] : index
      // CHECK: loop.for [[BLOCKLOOPXIV:%.*]] = [[BLOCKLOOPXLB]] to {{%.*}} step [[BLOCKLOOPXSTEP]]

      loop.for %iv2 = %c0 to %1 step %arg4 {

        // TODO: This is effectively shared memory. Lower it to a
        // shared memory.
        %2 = alloc(%arg3, %arg4) : memref<?x?xf32>

        // Load transpose tile
        // CHECK: [[TEMP3:%.*]] = muli [[ARG20]], [[ARG9:%.*]] : index
        // CHECK: [[THREADLOOP1YLB:%.*]] = addi {{%.*}}, [[TEMP3]] : index
        // CHECK: [[THREADLOOP1YSTEP:%.*]] = muli [[ARG20]], [[ARG15]] : index
        // CHECK: loop.for [[THREADLOOP1YIV:%.*]] = [[THREADLOOP1YLB]] to {{%.*}} step [[THREADLOOP1YSTEP]]
        loop.for %iv3 = %c0 to %arg3 step %c1 {
          // CHECK: [[TEMP4:%.*]] = muli [[ARG20]], [[ARG8]] : index
          // CHECK: [[THREADLOOP1XLB:%.*]] = addi {{%.*}}, [[TEMP4]] : index
          // CHECK: [[THREADLOOP1XSTEP:%.*]] = muli [[ARG20]], [[ARG14]] : index
          // CHECK: loop.for [[THREADLOOP1XIV:%.*]] = [[THREADLOOP1XLB]] to {{%.*}} step [[THREADLOOP1XSTEP]]
          loop.for %iv4 = %c1 to %arg4 step %c1 {
            // CHECK: [[INDEX2:%.*]] = addi [[BLOCKLOOPYIV]], [[THREADLOOP1YIV]] : index
            %10 = addi %iv1, %iv3 : index
            // CHECK: [[INDEX1:%.*]] = addi [[BLOCKLOOPXIV]], [[THREADLOOP1XIV]] : index
            %11 = addi %iv2, %iv4 : index
            // CHECK: [[VAL1:%.*]] = load [[ARG19]]{{\[}}[[INDEX1]], [[INDEX2]]{{\]}} : memref<?x?xf32>
            %12 = load %arg1[%11, %10] : memref<?x?xf32>
            // CHECK: store [[VAL1]], [[SCRATCHSPACE:%.*]]{{\[}}[[THREADLOOP1XIV]], [[THREADLOOP1YIV]]{{\]}} : memref<?x?xf32>
            store %12, %2[%iv4, %iv3] : memref<?x?xf32>
          }
        }

        // TODO: There needs to be a sync here for correctness, but
        // testing only loop partitioning for now.

        // CHECK: [[TEMP5:%.*]] = muli [[ARG20]], [[ARG9]] : index
        // CHECK: [[THREADLOOP2YLB:%.*]] = addi {{%.*}}, [[TEMP5]] : index
        // CHECK: [[THREADLOOP2YSTEP:%.*]] = muli [[ARG20]], [[ARG15]] : index
        // CHECK: loop.for [[THREADLOOP2YIV:%.*]] = [[THREADLOOP2YLB]] to {{%.*}} step [[THREADLOOP2YSTEP]]
        loop.for %iv3 = %c0 to %arg3 step %c1 {
          // CHECK: [[TEMP6:%.*]] = muli [[ARG20]], [[ARG8]] : index
          // CHECK: [[THREADLOOP2XLB:%.*]] = addi {{%.*}}, [[TEMP6]] : index
          // CHECK: [[THREADLOOP2XSTEP:%.*]] = muli [[ARG20]], [[ARG14]] : index
          // CHECK: loop.for [[THREADLOOP2XIV:%.*]] = [[THREADLOOP2XLB]] to {{%.*}} step [[THREADLOOP2XSTEP]]
          loop.for %iv4 = %c1 to %arg4 step %c1 {
            // CHECK: [[INDEX3:%.*]] = addi [[BLOCKLOOPYIV]], [[THREADLOOP2YIV]] : index
            %13 = addi %iv1, %iv3 : index
            // CHECK: [[INDEX4:%.*]] = addi [[BLOCKLOOPXIV]], [[THREADLOOP2XIV]] : index
            %14 = addi %iv2, %iv4 : index
            // CHECK: {{%.*}} = load [[SCRATCHSPACE]]{{\[}}[[THREADLOOP2XIV]], [[THREADLOOP2YIV]]{{\]}} : memref<?x?xf32>
            %15 = load %2[%iv4, %iv3] : memref<?x?xf32>
            // CHECK: {{%.*}} = load [[ARG22]]{{\[}}[[INDEX3]], [[INDEX4]]{{\]}}
            %16 = load %arg0[%13, %14] : memref<?x?xf32>
            %17 = mulf %15, %16 : f32
            // CHECK: store {{%.*}}, [[ARG23]]{{\[}}[[INDEX3]], [[INDEX4]]{{\]}}
            store %17, %arg2[%13, %14] : memref<?x?xf32>
          }
        }

        dealloc %2 : memref<?x?xf32>
      }
    }
    return
  }
}