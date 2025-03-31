// RUN: triton-opt %s -test-print-alignment -split-input-file -o /dev/null

tt.func @cast() {
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = 1}}
  %cst = arith.constant 1 : i32
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = 1}}
  %0 = arith.extsi %cst : i32 to i64
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [128], constant_value = 1}}
  %cst_tensor = arith.constant dense<1> : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [128], constant_value = 1}}
  // Bitcast preserves axis info for same-width types.
  %1 = tt.bitcast %cst_tensor : tensor<128xi32> -> tensor<128xf32>
  tt.return
}

// -----

tt.func @add() {
  // expeted-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [128], constant_value = 1}}
  %1 = arith.constant dense<1> : tensor<128xi32>
  // expeted-remark @below {{contiguity = [128], divisibility = [1], constancy = [1], constant_value = <none>}}
  %2 = arith.addi %0, %1 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [128], constant_value = 127}}
  %3 = arith.constant dense<127> : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [128], constancy = [128], constant_value = 128}}
  %4 = arith.addi %1, %3 : tensor<128xi32>
  tt.return
}

// -----

tt.func @addptr(%arg0: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<i64> {tt.divisibility = 16 : i32}) {
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = 1}}
  %cst1 = arith.constant 1 : i32
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %0 = tt.addptr %arg0, %cst1 : !tt.ptr<i1>, i32
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %1 = tt.addptr %arg1, %cst1 : !tt.ptr<i8>, i32
  // expeted-remark @below {{contiguity = [1], divisibility = [2], constancy = [1], constant_value = <none>}}
  %2 = tt.addptr %arg2, %cst1 : !tt.ptr<i16>, i32
  // expeted-remark @below {{contiguity = [1], divisibility = [4], constancy = [1], constant_value = <none>}}
  %3 = tt.addptr %arg3, %cst1 : !tt.ptr<i32>, i32
  // expeted-remark @below {{contiguity = [1], divisibility = [8], constancy = [1], constant_value = <none>}}
  %4 = tt.addptr %arg4, %cst1 : !tt.ptr<i64>, i32
  // expeted-remark @below {{contiguity = [1], divisibility = [4], constancy = [1], constant_value = 4}}
  %cst4 = arith.constant 4 : i32
  // expeted-remark @below {{contiguity = [1], divisibility = [4], constancy = [1], constant_value = <none>}}
  %5 = tt.addptr %arg0, %cst4 : !tt.ptr<i1>, i32
  // expeted-remark @below {{contiguity = [1], divisibility = [4], constancy = [1], constant_value = <none>}}
  %6 = tt.addptr %arg1, %cst4 : !tt.ptr<i8>, i32
  // expeted-remark @below {{contiguity = [1], divisibility = [8], constancy = [1], constant_value = <none>}}
  %7 = tt.addptr %arg2, %cst4 : !tt.ptr<i16>, i32
  // expeted-remark @below {{contiguity = [1], divisibility = [16], constancy = [1], constant_value = <none>}}
  %8 = tt.addptr %arg3, %cst4 : !tt.ptr<i32>, i32
  // expeted-remark @below {{contiguity = [1], divisibility = [16], constancy = [1], constant_value = <none>}}
  %9 = tt.addptr %arg4, %cst4 : !tt.ptr<i64>, i32
  // expeted-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %10 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1, 128], divisibility = [1, 1073741824], constancy = [1, 1], constant_value = <none>}}
  %11 = tt.expand_dims %10 {axis = 0: i32} : tensor<128xi32> -> tensor<1x128xi32>
  // expeted-remark @below {{contiguity = [1, 128], divisibility = [1, 1073741824], constancy = [128, 1], constant_value = <none>}}
  %12 = tt.broadcast %11 : tensor<1x128xi32> -> tensor<128x128xi32>
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 128], constant_value = <none>}}
  %13 = tt.splat %arg0 : !tt.ptr<i1> -> tensor<128x128x!tt.ptr<i1>>
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 128], constant_value = <none>}}
  %14 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<128x128x!tt.ptr<i8>>
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 128], constant_value = <none>}}
  %15 = tt.splat %arg2 : !tt.ptr<i16> -> tensor<128x128x!tt.ptr<i16>>
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 128], constant_value = <none>}}
  %16 = tt.splat %arg3 : !tt.ptr<i32> -> tensor<128x128x!tt.ptr<i32>>
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 128], constant_value = <none>}}
  %17 = tt.splat %arg4 : !tt.ptr<i64> -> tensor<128x128x!tt.ptr<i64>>
  // expeted-remark @below {{contiguity = [1, 128], divisibility = [1, 16], constancy = [128, 1], constant_value = <none>}}
  %18 = tt.addptr %13, %12 : tensor<128x128x!tt.ptr<i1>>, tensor<128x128xi32>
  // expeted-remark @below {{contiguity = [1, 128], divisibility = [1, 16], constancy = [128, 1], constant_value = <none>}}
  %19 = tt.addptr %14, %12 : tensor<128x128x!tt.ptr<i8>>, tensor<128x128xi32>
  // expeted-remark @below {{contiguity = [1, 128], divisibility = [2, 16], constancy = [128, 1], constant_value = <none>}}
  %20 = tt.addptr %15, %12 : tensor<128x128x!tt.ptr<i16>>, tensor<128x128xi32>
  // expeted-remark @below {{contiguity = [1, 128], divisibility = [4, 16], constancy = [128, 1], constant_value = <none>}}
  %21 = tt.addptr %16, %12 : tensor<128x128x!tt.ptr<i32>>, tensor<128x128xi32>
  // expeted-remark @below {{contiguity = [1, 128], divisibility = [8, 16], constancy = [128, 1], constant_value = <none>}}
  %22 = tt.addptr %17, %12 : tensor<128x128x!tt.ptr<i64>>, tensor<128x128xi32>
  tt.return
}

// -----

tt.func @sub() {
  // expeted-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [128], constant_value = 1}}
  %1 = arith.constant dense<1> : tensor<128xi32>
  // expeted-remark @below {{contiguity = [128], divisibility = [1], constancy = [1], constant_value = <none>}}
  %2 = arith.subi %0, %1 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %3 = arith.subi %1, %0 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [128], constant_value = 129}}
  %4 = arith.constant dense<129> : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [128], constancy = [128], constant_value = 128}}
  %5 = arith.subi %4, %1 : tensor<128xi32>
  tt.return
}

// -----

tt.func @mul(%arg0: i64 {tt.divisibility = 16 : i32}) {
  // expeted-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [128], constant_value = 1}}
  %1 = arith.constant dense<1> : tensor<128xi32>
  // expeted-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %2 = arith.muli %0, %1 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [128], constancy = [128], constant_value = 128}}
  %3 = arith.constant dense<128> : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [128], constancy = [128], constant_value = 128}}
  %4 = arith.muli %3, %1 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [2], constancy = [128], constant_value = 2}}
  %5 = arith.constant dense<2> : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [256], constancy = [128], constant_value = 256}}
  %6 = arith.muli %4, %5 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [4611686018427387904], constancy = [1], constant_value = 4611686018427387904}}
  %7 = arith.constant 4611686018427387904: i64
  // expeted-remark @below {{contiguity = [1], divisibility = [4611686018427387904], constancy = [1], constant_value = <none>}}
  %8 = arith.muli %arg0, %7 : i64
  tt.return
}

// -----

tt.func @div() {
  // expeted-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [128], constant_value = 1}}
  %1 = arith.constant dense<1> : tensor<128xi32>
  // expeted-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %2 = arith.divsi %0, %1 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %3 = arith.divui %1, %0 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [64], constancy = [128], constant_value = 64}}
  %4 = arith.constant dense<64> : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [64], constant_value = <none>}}
  %5 = arith.divsi %0, %4 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %6 = arith.divsi %4, %0 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [64], constancy = [128], constant_value = 64}}
  %7 = arith.divsi %4, %1 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [2], constancy = [128], constant_value = 66}}
  %8 = arith.constant dense<66> : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [2], constant_value = <none>}}
  %9 = arith.divui %0, %8 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [128], divisibility = [8192], constancy = [1], constant_value = <none>}}
  %10 = tt.make_range {end = 8320 : i32, start = 8192 : i32} : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [64], constant_value = <none>}}
  %11 = arith.divsi %10, %4 : tensor<128xi32>
  tt.return
}


// -----

tt.func @rem() {
  // expeted-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [128], constant_value = 1}}
  %1 = arith.constant dense<1> : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [4611686018427387904], constancy = [128], constant_value = 0}}
  %2 = arith.remsi %0, %1 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %3 = arith.remui %1, %0 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [64], constancy = [128], constant_value = 64}}
  %4 = arith.constant dense<64> : tensor<128xi32>
  // expeted-remark @below {{contiguity = [64], divisibility = [64], constancy = [1], constant_value = <none>}}
  %5 = arith.remsi %0, %4 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [64], constancy = [1], constant_value = <none>}}
  %6 = arith.remsi %4, %0 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [2], constancy = [128], constant_value = 66}}
  %7 = arith.constant dense<66> : tensor<128xi32>
  // expeted-remark @below {{contiguity = [2], divisibility = [2], constancy = [1], constant_value = <none>}}
  %8 = arith.remui %0, %7 : tensor<128xi32>
  tt.return
}

// -----

tt.func @expanddims() {
  // expeted-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [2], constancy = [128], constant_value = 2}}
  %1 = arith.constant dense<2> : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [2], constancy = [1], constant_value = <none>}}
  %2 = arith.muli %0, %1 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [2, 2], constancy = [1, 1], constant_value = <none>}}
  %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
  tt.return
}

// -----

tt.func @broadcast() {
  // expeted-remark @below {{contiguity = [1], divisibility = [64], constancy = [128], constant_value = 64}}
  %0 = arith.constant dense<64> : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [64, 64], constancy = [128, 1], constant_value = 64}}
  %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [64, 64], constancy = [128, 128], constant_value = 64}}
  %2 = tt.broadcast %1 : tensor<128x1xi32> -> tensor<128x128xi32>
  tt.return
}

// -----

tt.func @splat(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 128], constant_value = <none>}}
  %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
  tt.return
}

// -----

tt.func @cmp_all_contiguous() {
  // expeted-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [4611686018427387904], constancy = [128], constant_value = 0}}
  %1 = arith.constant dense<0> : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %2 = arith.cmpi eq, %0, %1 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %3 = arith.cmpi ne, %0, %1 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [128], constant_value = <none>}}
  %4 = arith.cmpi slt, %0, %1 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %5 = arith.cmpi sle, %0, %1 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [128], constant_value = <none>}}
  %6 = arith.cmpi sge, %0, %1 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %7 = arith.cmpi sgt, %0, %1 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %8 = arith.cmpi eq, %1, %0 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %9 = arith.cmpi ne, %1, %0 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %10 = arith.cmpi slt, %1, %0 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [128], constant_value = <none>}}
  %11 = arith.cmpi sle, %1, %0 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %12 = arith.cmpi sge, %1, %0 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [128], constant_value = <none>}}
  %13 = arith.cmpi sgt, %1, %0 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [8], constancy = [128], constant_value = 8}}
  %14 = arith.constant dense<8> : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [8], constant_value = <none>}}
  %15 = arith.cmpi sgt, %14, %0 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [128], constant_value = 1}}
  %16 = arith.cmpi sgt, %14, %1 : tensor<128xi32>
  tt.return
}

tt.func @cmp_partial_contiguous() {
  // expeted-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [8], constancy = [128], constant_value = 8}}
  %1 = arith.constant dense<8> : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [32], constancy = [128], constant_value = 32}}
  %3 = arith.constant dense<32> : tensor<128xi32>
  // expeted-remark @below {{contiguity = [32], divisibility = [32], constancy = [1], constant_value = <none>}}
  %4 = arith.remsi %0, %3 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %5 = arith.cmpi eq, %4, %1 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %6 = arith.cmpi ne, %4, %1 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [8], constant_value = <none>}}
  %7 = arith.cmpi slt, %4, %1 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %8 = arith.cmpi sle, %4, %1 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [8], constant_value = <none>}}
  %9 = arith.cmpi sge, %4, %1 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %10 = arith.cmpi sgt, %4, %1 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %11 = arith.cmpi eq, %1, %4 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %12 = arith.cmpi ne, %1, %4 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %13 = arith.cmpi slt, %1, %4 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [8], constant_value = <none>}}
  %14 = arith.cmpi sle, %1, %4 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %15 = arith.cmpi sge, %1, %4 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [8], constant_value = <none>}}
  %16 = arith.cmpi sgt, %1, %4 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [16], constancy = [128], constant_value = 48}}
  %17 = arith.constant dense<48> : tensor<128xi32>
  // expeted-remark @below {{contiguity = [16], divisibility = [16], constancy = [1], constant_value = <none>}}
  %18 = arith.remsi %0, %17 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %19 = arith.cmpi eq, %18, %3 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %20 = arith.cmpi ne, %18, %3 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [16], constant_value = <none>}}
  %21 = arith.cmpi slt, %18, %3 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %22 = arith.cmpi sle, %18, %3 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [16], constant_value = <none>}}
  %23 = arith.cmpi sge, %18, %3 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %24 = arith.cmpi sgt, %18, %3 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %25 = arith.cmpi eq, %3, %18 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %26 = arith.cmpi ne, %3, %18 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %27 = arith.cmpi slt, %3, %18 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [16], constant_value = <none>}}
  %28 = arith.cmpi sle, %3, %18 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %29 = arith.cmpi sge, %3, %18 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [16], constant_value = <none}}
  %30 = arith.cmpi sgt, %3, %18 : tensor<128xi32>
  tt.return
}

// -----

tt.func @logic() {
  // expeted-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [64], constancy = [128], constant_value = 64}}
  %1 = arith.constant dense<64> : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [64], constant_value = <none>}}
  %2 = arith.divsi %0, %1 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [8], constancy = [128], constant_value = 8}}
  %3 = arith.constant dense<8> : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [8], constant_value = <none>}}
  %4 = arith.divsi %0, %3 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %5 = arith.andi %0, %1 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %6 = arith.ori %0, %1 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %7 = arith.xori %0, %1 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [8], constant_value = <none>}}
  %8 = arith.andi %2, %4 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [8], constant_value = <none>}}
  %9 = arith.ori %2, %4 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [8], constant_value = <none>}}
  %10 = arith.xori %2, %4 : tensor<128xi32>
  tt.return
}

// -----

tt.func @select(%arg0 : i1, %arg1 : tensor<4xi1>) {
  // expeted-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [4611686018427387904], constancy = [128], constant_value = 0}}
  %1 = arith.constant dense<0> : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %2 = arith.cmpi eq, %0, %1 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [128], constant_value = <none>}}
  %3 = arith.cmpi slt, %0, %1 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [4611686018427387904], constancy = [1], constant_value = 0}}
  %4 = arith.constant 0 : i1
  // expeted-remark @below {{contiguity = [1], divisibility = [4611686018427387904], constancy = [128], constant_value = 0}}
  %7 = tt.splat %4 : i1 -> tensor<128xi1>
  // expeted-remark @below {{contiguity = [1], divisibility = [4611686018427387904], constancy = [128], constant_value = 0}}
  %5 = arith.select %4, %3, %7 : tensor<128xi1>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %8 = arith.select %7, %3, %2 : tensor<128xi1>, tensor<128xi1>
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [1, 1], constancy = [1, 1], constant_value = <none>}}
  %9 = tt.expand_dims %2 {axis = 1 : i32} : tensor<128xi1> -> tensor<128x1xi1>
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [1, 1], constancy = [128, 1], constant_value = <none>}}
  %10 = tt.expand_dims %3 {axis = 1 : i32} : tensor<128xi1> -> tensor<128x1xi1>
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [1, 1], constancy = [1, 1], constant_value = <none>}}
  %11 = arith.select %arg0, %9, %10 : tensor<128x1xi1>
  // expeted-remark @below {{contiguity = [1], divisibility = [4], constancy = [4], constant_value = 4}}
  %cst = arith.constant dense<4> : tensor<4xi32>
  // expeted-remark @below {{contiguity = [4], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %12 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [4], constancy = [1], constant_value = <none>}}
  %13 = arith.muli %12, %cst : tensor<4xi32>
  // expeted-remark @below {{contiguity = [4], divisibility = [16], constancy = [1], constant_value = <none>}}
  %14 = tt.make_range {end = 20 : i32, start = 16 : i32} : tensor<4xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %15 = arith.select %arg1, %12, %13 : tensor<4xi1>, tensor<4xi32>
  tt.return
}

// -----

tt.func @shift(%arg0: i32 {tt.divisibility = 4 : i32}) {
  // expeted-remark @below {{contiguity = [1], divisibility = [4], constancy = [128], constant_value = <none>}}
  %s = tt.splat %arg0 : i32 -> tensor<128xi32>
  // expeted-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [8], constancy = [128], constant_value = 8}}
  %1 = arith.constant dense<8> : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [4], constancy = [128], constant_value = 4}}
  %2 = arith.constant dense<4> : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [256], constancy = [1], constant_value = <none>}}
  %3 = arith.shli %0, %1 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %4 = arith.shrsi %0, %2 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [128], constancy = [128], constant_value = 128}}
  %5 = arith.shli %1, %2 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [8], constancy = [128], constant_value = <none>}}
  %6 = arith.shli %1, %s : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %7 = arith.shrsi %0, %s : tensor<128xi32>
  tt.return
}

// -----

tt.func @max_min() {
  // expeted-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // expeted-remark @below {{contiguity = [128], divisibility = [64], constancy = [1], constant_value = <none>}}
  %1 = tt.make_range {end = 192 : i32, start = 64 : i32} : tensor<128xi32>
  // expeted-remark @below {{contiguity = [128], divisibility = [64], constancy = [1], constant_value = <none>}}
  %2 = arith.maxsi %0, %1 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [128], divisibility = [64], constancy = [1], constant_value = <none>}}
  %3 = arith.minsi %0, %1 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [8], constancy = [128], constant_value = 8}}
  %4 = arith.constant dense<8> : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [4], constancy = [128], constant_value = 4}}
  %5 = arith.constant dense<4> : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = 8}}
  %6 = arith.maxsi %4, %5 : tensor<128xi32>
  tt.return
}

// -----

tt.func @if(%i1 : i1) {
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [64, 64], constancy = [128, 32], constant_value = 64}}
  %cst_64 = arith.constant dense<64> : tensor<128x32xi32>
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [1, 1], constancy = [128, 32], constant_value = 1}}
  %cst_1 = arith.constant dense<1> : tensor<128x32xi32>
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [64, 64], constancy = [128, 32], constant_value = 64}}
  %a = arith.muli %cst_64, %cst_1 : tensor<128x32xi32>
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [1, 1], constancy = [128, 32], constant_value = <none>}}
  %ret = scf.if %i1 -> tensor<128x32xi32> {
    scf.yield %a : tensor<128x32xi32>
  } else {
    scf.yield %cst_1 : tensor<128x32xi32>
  }
  tt.return
}

// -----

tt.func @for() {
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [4611686018427387904, 4611686018427387904], constancy = [128, 32], constant_value = 0}}
  %a_init = arith.constant dense<0> : tensor<128x32xi32>
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [1, 1], constancy = [128, 32], constant_value = 1}}
  %b_init = arith.constant dense<1> : tensor<128x32xi32>
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [4, 4], constancy = [128, 32], constant_value = 4}}
  %c_init = arith.constant dense<4> : tensor<128x32xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [128], constancy = [1], constant_value = 128}}
  %ub = arith.constant 128 : i32
  // expeted-remark @below {{contiguity = [1], divisibility = [4611686018427387904], constancy = [1], constant_value = 0}}
  %lb = arith.constant 0 : i32
  // expeted-remark @below {{contiguity = [1], divisibility = [16], constancy = [1], constant_value = 16}}
  %step = arith.constant 16 : i32
  %a, %b, %c = scf.for %iv = %lb to %ub step %step iter_args(%a = %a_init, %b = %b_init, %c = %c_init) -> (tensor<128x32xi32>, tensor<128x32xi32>, tensor<128x32xi32>) : i32 {
    // expeted-remark @below {{contiguity = [1], divisibility = [16], constancy = [1], constant_value = <none>}}
    %t = arith.addi %iv, %lb : i32
    // expeted-remark @below {{contiguity = [1, 1], divisibility = [1, 1], constancy = [128, 32], constant_value = <none>}}
    // expeted-remark @below {{contiguity = [1, 1], divisibility = [1, 1], constancy = [128, 32], constant_value = <none>}}
    // expeted-remark @below {{contiguity = [1, 1], divisibility = [4, 4], constancy = [128, 32], constant_value = 4}}
    scf.yield %b, %a, %c : tensor<128x32xi32>, tensor<128x32xi32>, tensor<128x32xi32>
  }
  tt.return
}

// -----

tt.func @for_dynamic(%lb: i32 {tt.divisibility = 16 : i32}, %step: i32 {tt.divisibility = 8 : i32}, %ub: i32) {
  // expeted-remark @below {{contiguity = [1], divisibility = [4611686018427387904], constancy = [1], constant_value = 0}}
  %c0 = arith.constant 0 : i32
  scf.for %iv = %lb to %ub step %step : i32 {
    // expeted-remark @below {{contiguity = [1], divisibility = [8], constancy = [1], constant_value = <none>}}
    %t = arith.addi %iv, %c0 : i32
  }
  tt.return
}

// -----

tt.func @for_if(%i1: i1, %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
  // expeted-remark @below {{contiguity = [1], divisibility = [4611686018427387904], constancy = [1], constant_value = 0}}
  %c0_i32 = arith.constant 0 : i32
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = 1}}
  %c1_i32 = arith.constant 1 : i32
  // expeted-remark @below {{contiguity = [1], divisibility = [2], constancy = [1], constant_value = 10}}
  %c10_i32 = arith.constant 10 : i32
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [64, 64], constancy = [128, 64], constant_value = 64}}
  %cst = arith.constant dense<64> : tensor<128x64xi32>
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 64], constant_value = <none>}}
  %1 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>>
  %2 = scf.for %arg9 = %c0_i32 to %c10_i32 step %c1_i32 iter_args(%arg1 = %1) -> (tensor<128x64x!tt.ptr<f16>>): i32 {
    // expeted-remark @below {{scf.if}}
    // expeted-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 64], constant_value = <none>}}
    %3 = scf.if %i1 -> (tensor<128x64x!tt.ptr<f16>>) {
      scf.yield %arg1 : tensor<128x64x!tt.ptr<f16>>
    } else {
      scf.yield %arg1 : tensor<128x64x!tt.ptr<f16>>
    }
    // expeted-remark @below {{tt.addptr}}
    // expeted-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 64], constant_value = <none>}}
    %4 = tt.addptr %3, %cst : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
    // expeted-remark @below {{scf.for}}
    // expeted-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 64], constant_value = <none>}}
    scf.yield %1 : tensor<128x64x!tt.ptr<f16>>
  }
  tt.return
}

// -----

tt.func @for_if_for(%i1: i1, %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 8 : i32}) {
  // expeted-remark @below {{contiguity = [1], divisibility = [4611686018427387904], constancy = [1], constant_value = 0}}
  %c0_i32 = arith.constant 0 : i32
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = 1}}
  %c1_i32 = arith.constant 1 : i32
  // expeted-remark @below {{contiguity = [1], divisibility = [2], constancy = [1], constant_value = 10}}
  %c10_i32 = arith.constant 10 : i32
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [64, 64], constancy = [128, 64], constant_value = 64}}
  %cst = arith.constant dense<64> : tensor<128x64xi32>
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 64], constant_value = <none>}}
  %1 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>>
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [8, 8], constancy = [128, 64], constant_value = <none>}}
  %2 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>>
  // expeted-remark @below {{scf.for}}
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [8, 8], constancy = [128, 64], constant_value = <none>}}
  // expeted-remark @below {{scf.if}}
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [8, 8], constancy = [128, 64], constant_value = <none>}}
  // expeted-remark @below {{tt.addptr}}
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [8, 8], constancy = [128, 64], constant_value = <none>}}
  // expeted-remark @below {{scf.for}}
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 64], constant_value = <none>}}
  %3 = scf.for %arg9 = %c0_i32 to %c10_i32 step %c1_i32 iter_args(%arg2 = %1) -> (tensor<128x64x!tt.ptr<f16>>) : i32 {
    %4 = scf.if %i1 -> (tensor<128x64x!tt.ptr<f16>>) {
      %5 = scf.for %arg10 = %c0_i32 to %c10_i32 step %c1_i32 iter_args(%arg3 = %2) -> (tensor<128x64x!tt.ptr<f16>>) : i32 {
        scf.yield %arg3 : tensor<128x64x!tt.ptr<f16>>
      }
      scf.yield %5 : tensor<128x64x!tt.ptr<f16>>
    } else {
      scf.yield %arg2 : tensor<128x64x!tt.ptr<f16>>
    }
    %6 = tt.addptr %4, %cst : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
    scf.yield %1 : tensor<128x64x!tt.ptr<f16>>
  }
  tt.return
}

// -----

tt.func @permute_2d(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) {
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [1, 1], constancy = [128, 128], constant_value = 1}}
  %cst = arith.constant dense<true> : tensor<128x128xi1>
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [1, 1], constancy = [1, 1], constant_value = <none>}}
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
  // expeted-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // expeted-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // expeted-remark @below {{contiguity = [128, 1], divisibility = [1073741824, 1], constancy = [1, 1], constant_value = <none>}}
  %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 1], constant_value = <none>}}
  %3 = tt.splat %arg1 : i32 -> tensor<128x1xi32>
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [1, 1], constant_value = <none>}}
  %4 = arith.muli %2, %3 : tensor<128x1xi32>
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 1], constant_value = <none>}}
  %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>>
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [1, 1], constant_value = <none>}}
  %6 = tt.addptr %5, %4 : tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>
  // expeted-remark @below {{contiguity = [1, 128], divisibility = [1, 1073741824], constancy = [1, 1], constant_value = <none>}}
  %7 = tt.expand_dims %1 {axis = 0 : i32}: tensor<128xi32> -> tensor<1x128xi32>
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [1, 128], constant_value = <none>}}
  %8 = tt.broadcast %6 : tensor<128x1x!tt.ptr<f32>> -> tensor<128x128x!tt.ptr<f32>>
  // expeted-remark @below {{contiguity = [1, 128], divisibility = [1, 1073741824], constancy = [128, 1], constant_value = <none>}}
  %9 = tt.broadcast %7 : tensor<1x128xi32> -> tensor<128x128xi32>
  // expeted-remark @below {{contiguity = [1, 128], divisibility = [4, 16], constancy = [1, 1], constant_value = <none>}}
  %10 = tt.addptr %8, %9 : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
  // expeted-remark @below {{contiguity = [128, 1], divisibility = [1073741824, 1], constancy = [1, 1], constant_value = <none>}}
  %11 = tt.expand_dims %0 {axis = 1 : i32}: tensor<128xi32> -> tensor<128x1xi32>
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 1], constant_value = <none>}}
  %12 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>>
  // expeted-remark @below {{contiguity = [128, 1], divisibility = [16, 4], constancy = [1, 1], constant_value = <none>}}
  %13 = tt.addptr %12, %11 : tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>
  // expeted-remark @below {{contiguity = [1, 128], divisibility = [1, 1073741824], constancy = [1, 1], constant_value = <none>}}
  %14 = tt.expand_dims %1 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [1, 128], constant_value = <none>}}
  %15 = tt.splat %arg3 : i32 -> tensor<1x128xi32>
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [1, 1], constant_value = <none>}}
  %16 = arith.muli %14, %15 : tensor<1x128xi32>
  // expeted-remark @below {{contiguity = [128, 1], divisibility = [16, 4], constancy = [1, 128], constant_value = <none>}}
  %17 = tt.broadcast %13 : tensor<128x1x!tt.ptr<f32>> -> tensor<128x128x!tt.ptr<f32>>
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 1], constant_value = <none>}}
  %18 = tt.broadcast %16 : tensor<1x128xi32> -> tensor<128x128xi32>
  // expeted-remark @below {{contiguity = [128, 1], divisibility = [16, 4], constancy = [1, 1], constant_value = <none>}}
  %19 = tt.addptr %17, %18 : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [1, 1], constancy = [1, 1], constant_value = <none>}}
  %20 = tt.load %10, %cst, %cst_0 : tensor<128x128x!tt.ptr<f32>>
  tt.store %19, %20, %cst : tensor<128x128x!tt.ptr<f32>>
  tt.return
}

// -----

tt.func @load_constancy(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 1 : i32}) {
  // expeted-remark @below {{divisibility = [16]}}
  %sixteen = arith.constant dense<16> : tensor<1024xi32>
  // expeted-remark @below {{divisibility = [8]}}
  %eight = arith.constant dense<8> : tensor<1024xi32>
  // expeted-remark @below {{contiguity = [1024], divisibility = [1073741824], constancy = [1]}}
  %1 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  // expeted-remark @below {{constancy = [16]}}
  %2 = arith.divsi %1, %sixteen : tensor<1024xi32>
  // expeted-remark @below {{constancy = [1024]}}
  %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
  // expeted-remark @below {{constancy = [1024]}}
  %4 = tt.splat %arg1 : i32 -> tensor<1024xi32>
  // expeted-remark @below {{constancy = [8]}}
  %5 = arith.divsi %1, %eight : tensor<1024xi32>
  // expeted-remark @below {{constancy = [8]}}
  %6 = arith.cmpi slt, %5, %4 : tensor<1024xi32>
  // expeted-remark @below {{constancy = [16]}}
  %7 = tt.addptr %3, %2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
  // expeted-remark @below {{constancy = [16]}}
  %8 = tt.load %7 : tensor<1024x!tt.ptr<f32>>
  // expeted-remark @below {{constancy = [8]}}
  %9 = tt.load %7, %6 : tensor<1024x!tt.ptr<f32>>
  tt.return
}

// -----

// This is a tiny test for verifying StoreOp-related alignment, It simply store a constant to a buffer.
tt.func @store_constant_align(%addr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n: i32 {tt.divisibility = 16 : i32}) {
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %pid = tt.get_program_id x : i32
  // expeted-remark @below {{contiguity = [1], divisibility = [128], constancy = [1], constant_value = 128}}
  %c128_i32 = arith.constant 128 : i32
  // expeted-remark @below {{contiguity = [1], divisibility = [128], constancy = [1], constant_value = <none>}}
  %1 = arith.muli %pid, %c128_i32 : i32
  // expeted-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
 // expeted-remark @below {{contiguity = [1], divisibility = [128], constancy = [128], constant_value = <none>}}
  %3 = tt.splat %1 : i32 -> tensor<128xi32>
 // expeted-remark @below {{contiguity = [128], divisibility = [128], constancy = [1], constant_value = <none>}}
  %4 = arith.addi %3, %2 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [16], constancy = [128], constant_value = <none>}}
  %5 = tt.splat %addr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  // expeted-remark @below {{contiguity = [128], divisibility = [16], constancy = [1], constant_value = <none>}}
  %6 = tt.addptr %5, %4 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [16], constancy = [128], constant_value = <none>}}
  %9 = tt.splat %n : i32 -> tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [16], constant_value = <none>}}
  %mask = arith.cmpi slt, %4, %9 : tensor<128xi32>
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %cst = arith.constant dense<0.0> : tensor<128xf32>
  tt.store %5, %cst, %mask : tensor<128x!tt.ptr<f32>>
  tt.return
}

// -----

// This IR is dumped from vecadd test.
// Note, the hint {tt.divisibility = 16 : i32} for %n_elements affects the alignment of mask.
tt.func @vecadd_mask_align_16(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n_elements: i32 {tt.divisibility = 16 : i32}) {
  %c64_i32 = arith.constant 64 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c64_i32 : i32
  %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %3 = tt.splat %1 : i32 -> tensor<64xi32>
  %4 = arith.addi %3, %2 : tensor<64xi32>
  %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  %6 = tt.addptr %5, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
  %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  %8 = tt.addptr %7, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
  %9 = tt.splat %n_elements : i32 -> tensor<64xi32>
  // expeted-remark @below {{arith.cmpi slt, %{{.*}} => contiguity = [1], divisibility = [1], constancy = [16], constant_value = <none>}}
  %mask = arith.cmpi slt, %4, %9 : tensor<64xi32>
  %11 = tt.load %6, %mask : tensor<64x!tt.ptr<f32>>
  %12 = tt.load %8, %mask : tensor<64x!tt.ptr<f32>>
  %13 = arith.addf %11, %12 : tensor<64xf32>
  %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  // expeted-remark @below {{tt.addptr %{{.*}} => contiguity = [64], divisibility = [16], constancy = [1], constant_value = <none>}}
  %15 = tt.addptr %14, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
  tt.store %15, %13, %mask : tensor<64x!tt.ptr<f32>>
  tt.return
}

// -----

// This IR is dumped from vecadd test.
// Note, there is no divisibility hint for %n_elements, Triton should assume its divisibility to be 1 by default.
tt.func @vecadd_mask_align_1(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n_elements: i32) {
  %c64_i32 = arith.constant 64 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c64_i32 : i32
  %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %3 = tt.splat %1 : i32 -> tensor<64xi32>
  %4 = arith.addi %3, %2 : tensor<64xi32>
  %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  %6 = tt.addptr %5, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
  %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  %8 = tt.addptr %7, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
  %9 = tt.splat %n_elements : i32 -> tensor<64xi32>
  // expeted-remark @below {{arith.cmpi slt, %{{.*}} => contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>}}
  %10 = arith.cmpi slt, %4, %9 : tensor<64xi32>
  %11 = tt.load %6, %10 : tensor<64x!tt.ptr<f32>>
  %12 = tt.load %8, %10 : tensor<64x!tt.ptr<f32>>
  %13 = arith.addf %11, %12 : tensor<64xf32>
  %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  %15 = tt.addptr %14, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
  tt.store %15, %13, %10 : tensor<64x!tt.ptr<f32>>
  tt.return
}

// -----

module {

// We don't use function cloning here, so the alignment info is the gcd of all call sites.
tt.func @addptr_hints(%arg0: !tt.ptr<i32>) {
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = 1}}
  %cst1 = arith.constant 1 : i32
  // expeted-remark @below {{contiguity = [1], divisibility = [4], constancy = [1], constant_value = <none>}}
  %1 = tt.addptr %arg0, %cst1 : !tt.ptr<i32>, i32
  // expeted-remark @below {{contiguity = [1], divisibility = [4], constancy = [1], constant_value = 4}}
  %cst4 = arith.constant 4 : i32
  // expeted-remark @below {{contiguity = [1], divisibility = [4], constancy = [1], constant_value = <none>}}
  %2 = tt.addptr %arg0, %cst4 : !tt.ptr<i32>, i32
  // expeted-remark @below {{contiguity = [1], divisibility = [16], constancy = [1], constant_value = 16}}
  %cst16 = arith.constant 16 : i32
  // expeted-remark @below {{contiguity = [1], divisibility = [4], constancy = [1], constant_value = <none>}}
  %3 = tt.addptr %arg0, %cst4 : !tt.ptr<i32>, i32
  tt.return
}

tt.func @kernel_div16(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
  tt.call @addptr_hints(%arg0) : (!tt.ptr<i32>) -> ()
  tt.return
}

tt.func @kernel_div8(%arg0: !tt.ptr<i32> {tt.divisibility = 8 : i32}) {
  tt.call @addptr_hints(%arg0) : (!tt.ptr<i32>) -> ()
  tt.return
}

tt.func @kernel_div4(%arg0: !tt.ptr<i32> {tt.divisibility = 4 : i32}) {
  tt.call @addptr_hints(%arg0) : (!tt.ptr<i32>) -> ()
  tt.return
}

}

// -----

module {

// We don't use function cloning here, so the alignment info is the gcd of all call sites.
tt.func @mul(%arg0: i32) {
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = 1}}
  %cst1 = arith.constant 1 : i32
  // expeted-remark @below {{contiguity = [1], divisibility = [4], constancy = [1], constant_value = <none>}}
  %1 = arith.muli %arg0, %cst1 : i32
  tt.return
}

tt.func @bar(%arg0: i32) {
  tt.call @mul(%arg0) : (i32) -> ()
  tt.return
}

tt.func @foo(%arg0: i32) {
  tt.call @mul(%arg0) : (i32) -> ()
  tt.return
}

tt.func @call_graph(%arg0: i32) {
  // expeted-remark @below {{contiguity = [1], divisibility = [4], constancy = [1], constant_value = 12}}
  %cst12 = arith.constant 12 : i32
  // expeted-remark @below {{contiguity = [1], divisibility = [4], constancy = [1], constant_value = <none>}}
  %0 = arith.muli %arg0, %cst12 : i32
  tt.call @foo(%0) : (i32) -> ()
  // expeted-remark @below {{contiguity = [1], divisibility = [8], constancy = [1], constant_value = 8}}
  %cst8 = arith.constant 8 : i32
  // expeted-remark @below {{contiguity = [1], divisibility = [8], constancy = [1], constant_value = <none>}}
  %1 = arith.muli %arg0, %cst8 : i32
  tt.call @bar(%1) : (i32) -> ()
  tt.return
}

}

// -----

tt.func @tensor_ptr(%arg0: !tt.ptr<tensor<64x16xi32>, 1>) {
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [1, 1], constancy = [1, 1], constant_value = <none>}}
  %0 = tt.load %arg0 : !tt.ptr<tensor<64x16xi32>, 1>
  tt.return
}


// -----

tt.func public @chained_for(%8: tensor<128x64x!tt.ptr<bf16>> {tt.divisibility = 16 : i32}) {
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [1, 1], constancy = [1, 1], constant_value = <none>}}
  %cst = arith.constant dense<0.000000e+00> : tensor<128x64xbf16>
  // expeted-remark @below {{contiguity = [1], divisibility = [16], constancy = [1], constant_value = 16}}
  %c16_i32 = arith.constant 16 : i32
  // expeted-remark @below {{contiguity = [1], divisibility = [1], constancy = [1], constant_value = 1}}
  %c1_i32 = arith.constant 1 : i32
  // expeted-remark @below {{contiguity = [1], divisibility = [4611686018427387904], constancy = [1], constant_value = 0}}
  %c0_i32 = arith.constant 0 : i32
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [64, 64], constancy = [128, 64], constant_value = 64}}
  %cst_0 = arith.constant dense<64> : tensor<128x64xi32>
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [1, 1], constant_value = <none>}}
  %9 = scf.for %arg7 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg8 = %8) -> (tensor<128x64x!tt.ptr<bf16>>)  : i32 {
    %11 = tt.addptr %arg8, %cst_0 : tensor<128x64x!tt.ptr<bf16>>, tensor<128x64xi32>
    scf.yield %11 : tensor<128x64x!tt.ptr<bf16>>
  }
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [1, 1], constant_value = <none>}}
  // expeted-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [1, 1], constant_value = <none>}}
  %10 = scf.for %arg7 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg8 = %9) -> (tensor<128x64x!tt.ptr<bf16>>)  : i32 {
    tt.store %arg8, %cst : tensor<128x64x!tt.ptr<bf16>>
    %11 = tt.addptr %arg8, %cst_0 : tensor<128x64x!tt.ptr<bf16>>, tensor<128x64xi32>
    scf.yield %11 : tensor<128x64x!tt.ptr<bf16>>
  }
  tt.return
}

// -----

module {
  tt.func @int_min_does_not_underflow_in_analysis() -> i64 {
    // expeted-remark @below {{divisibility = [4611686018427387904]}}
    %int_min = arith.constant -9223372036854775808 : i64
    tt.return %int_min : i64
  }
}
