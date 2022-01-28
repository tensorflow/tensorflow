// RUN: kernel-gen-opt %s --embed-memref-prints | FileCheck %s

func @print_memrefs(
    %ctx: !tf_framework.op_kernel_context, %input: memref<*xf16>)
    -> memref<*xf16> attributes {tf_entry} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %rank = memref.rank %input : memref<*xf16>
  %shape = memref.alloca(%rank) : memref<?xindex>
  scf.for %i = %c0 to %rank step %c1 {
    %dim = memref.dim %input, %i : memref<*xf16>
    memref.store %dim, %shape[%i] : memref<?xindex>
  }

  %c9000 = arith.constant 9000 : index
  %num_elem = memref.alloca() : memref<1xindex>
  memref.store %c9000, %num_elem[%c0] : memref<1xindex>
  %flat_input = memref.reshape %input(%num_elem)
    : (memref<*xf16>, memref<1xindex>) -> memref<?xf16>

  %flat_output = tf_framework.alloc(%ctx, %c9000) : memref<?xf16>
  %output = memref.reshape %flat_output(%shape)
    : (memref<?xf16>, memref<?xindex>) -> memref<*xf16>
  return %output : memref<*xf16>
}

// CHECK:   func private @print_memref_i64(memref<*xi64>)

// CHECK-LABEL: func @print_memrefs

// CHECK: [[SHAPE:%.*]] = memref.alloca({{%.*}}) : memref<?xindex>
// CHECK: scf.for
// CHECK: [[NUM_ELEM:%.*]] = memref.alloca() : memref<1xindex>
// CHECK: store {{%.*}}, [[NUM_ELEM]]

// CHECK: [[NUM_ELEM_I64:%.*]] = arith.index_cast [[NUM_ELEM]]
// CHECK-SAME: : memref<1xindex> to memref<1xi64>
// CHECK-NEXT: [[UNRANKED_NUM_ELEM:%.*]] = memref.cast [[NUM_ELEM_I64]]
// CHECK-NEXT: call @print_memref_i64([[UNRANKED_NUM_ELEM]])

// CHECK: memref.reshape
// CHECK: tf_framework.alloc

// CHECK: [[SHAPE_I64:%.*]] = arith.index_cast [[SHAPE]]
// CHECK-SAME: : memref<?xindex> to memref<?xi64>
// CHECK-NEXT: [[UNRANKED_SHAPE:%.*]] = memref.cast [[SHAPE_I64]]
// CHECK-NEXT: call @print_memref_i64([[UNRANKED_SHAPE]])
// CHECK: memref.reshape
