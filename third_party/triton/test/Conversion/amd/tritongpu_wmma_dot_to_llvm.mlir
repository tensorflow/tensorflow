// RUN: triton-opt %s --split-input-file --convert-triton-amdgpu-to-llvm=arch=gfx1100 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma1 = #ttg.amd_wmma<{version = 1, warpsPerCTA = [2, 2]}>
#mma2 = #ttg.amd_wmma<{version = 2, warpsPerCTA = [2, 2]}>
#mma2_transposed = #ttg.amd_wmma<{version = 2, warpsPerCTA = [2, 2], isTranspose = true}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  //  CHECK-LABEL: wmma1_dot_operand
  tt.func @wmma1_dot_operand(%arg0: !ttg.memdesc<64x64xf16, #shared, #smem>) {
    // 2 CTA * 4 rep * load_per_thread_per_instr
    // CHECK-COUNT-8: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<16xf16>
    %0 = ttg.local_load %arg0 : !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>>
    // CHECK-COUNT-128: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<1xf16>
    %1 = ttg.local_load %arg0 : !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>>
    tt.return
  }

  //  CHECK-LABEL: wmma2_dot_operand
  tt.func @wmma2_dot_operand(%arg0: !ttg.memdesc<64x64xf16, #shared, #smem>) {
    // 2 CTA * 4 rep * load_per_thread_per_instr
    // CHECK-COUNT-8: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<8xf16>
    %0 = ttg.local_load %arg0 : !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma2, kWidth = 8}>>
    // CHECK-COUNT-64: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<1xf16>
    %1 = ttg.local_load %arg0 : !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma2, kWidth = 8}>>
    tt.return
  }

  //  CHECK-LABEL: wmma1_dot_f16
  tt.func @wmma1_dot_f16(%arg0: tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>>, %arg1: tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>>, %arg2: tensor<16x16xf16, #mma1>) {
    // CHECK-COUNT-32: llvm.extractvalue %{{.*}} : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16)>
    // CHECK-COUNT-8: llvm.extractvalue %{{.*}} : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16)>
    // CHECK: llvm.mlir.undef : vector<16xf16>
    // CHECK-COUNT-8: llvm.insertelement {{.*}} : vector<16xf16>
    // CHECK: wmma.f16.16x16x16.f16{{.*}} : (vector<16xf16>, vector<16xf16>, vector<16xf16>, i1) -> vector<16xf16>
    %0 = tt.dot %arg0, %arg1, %arg2, inputPrecision = ieee : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>> -> tensor<16x16xf16, #mma1>
    // CHECK-COUNT-8: llvm.extractelement {{.*}} : vector<16xf16>
    // CHECK: llvm.mlir.undef : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16)>
    // CHECK-COUNT-8: llvm.insertvalue {{.*}} : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16)>
    tt.return
  }

  //  CHECK-LABEL: wmma1_dot_bf16
  tt.func @wmma1_dot_bf16(%arg0: tensor<16x16xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>>, %arg1: tensor<16x16xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>>, %arg2: tensor<16x16xbf16, #mma1>) {
    // CHECK-COUNT-16: llvm.extractvalue %{{.*}} : !llvm.struct<(bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16)>
    // CHECK: llvm.bitcast %{{.*}} : vector<16xbf16> to vector<16xi16>
    // CHECK-COUNT-16: llvm.extractvalue %{{.*}} : !llvm.struct<(bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16)>
    // CHECK: llvm.bitcast %{{.*}} : vector<16xbf16> to vector<16xi16>
    // CHECK-COUNT-8: llvm.extractvalue %{{.*}} : !llvm.struct<(bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16)>
    // CHECK: llvm.mlir.undef : vector<16xbf16>
    // CHECK-COUNT-8: llvm.insertelement {{.*}} : vector<16xbf16>
    // CHECK: wmma.bf16.16x16x16.bf16{{.*}} : (vector<16xi16>, vector<16xi16>, vector<16xbf16>, i1) -> vector<16xbf16>
    %0 = tt.dot %arg0, %arg1, %arg2, inputPrecision = ieee : tensor<16x16xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>> * tensor<16x16xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>> -> tensor<16x16xbf16, #mma1>
    tt.return
  }

  //  CHECK-LABEL: wmma1_dot_f16_tied
  tt.func @wmma1_dot_f16_tied(%arg0: tensor<64x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>>, %arg1: tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>>, %arg2: tensor<64x16xf16, #mma1>) {
    // CHECK-COUNT-32: llvm.extractvalue %{{.*}} : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16)>
    // CHECK-COUNT-8: llvm.extractvalue %{{.*}} : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16)>
    // CHECK: llvm.mlir.undef : vector<16xf16>
    // CHECK-COUNT-16: llvm.insertelement {{.*}} : vector<16xf16>
    // CHECK-COUNT-2: wmma.f16.16x16x16.f16.tied{{.*}} : (vector<16xf16>, vector<16xf16>, vector<16xf16>, i1) -> vector<16xf16>
    %0 = tt.dot %arg0, %arg1, %arg2, inputPrecision = ieee : tensor<64x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>> -> tensor<64x16xf16, #mma1>
    // CHECK-COUNT-8: llvm.extractelement {{.*}} : vector<16xf16>
    // CHECK: llvm.mlir.undef : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16)>
    // CHECK-COUNT-8: llvm.insertvalue {{.*}} : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16)>
    tt.return
  }

  //  CHECK-LABEL: wmma1_dot_bf16_tied
  tt.func @wmma1_dot_bf16_tied(%arg0: tensor<64x16xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>>, %arg1: tensor<16x16xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>>, %arg2: tensor<64x16xbf16, #mma1>) {
    // CHECK-COUNT-32: llvm.extractvalue %{{.*}} : !llvm.struct<(bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16)>
    // CHECK-COUNT-8: llvm.extractvalue %{{.*}} : !llvm.struct<(bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16)>
    // CHECK: llvm.mlir.undef : vector<16xbf16>
    // CHECK-COUNT-16: llvm.insertelement {{.*}} : vector<16xbf16>
    // CHECK-COUNT-2: wmma.bf16.16x16x16.bf16.tied{{.*}} : (vector<16xi16>, vector<16xi16>, vector<16xbf16>, i1) -> vector<16xbf16>
    %0 = tt.dot %arg0, %arg1, %arg2, inputPrecision = ieee : tensor<64x16xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>> * tensor<16x16xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>> -> tensor<64x16xbf16, #mma1>
    // CHECK-COUNT-8: llvm.extractelement {{.*}} : vector<16xbf16>
    // CHECK: llvm.mlir.undef : !llvm.struct<(bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16)>
    // CHECK-COUNT-8: llvm.insertvalue {{.*}} : !llvm.struct<(bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16)>
    tt.return
  }

  //  CHECK-LABEL: wmma1_dot_int8_32
  tt.func @wmma1_dot_int8_32(%arg0: tensor<16x16xi8, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>>, %arg1: tensor<16x16xi8, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>>, %arg2: tensor<16x16xi32, #mma1>) {
    // CHECK-COUNT-16: llvm.extractvalue %{{.*}} : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK-COUNT-16: llvm.insertelement {{.*}} : vector<16xi8>
    // CHECK: llvm.bitcast %{{.*}} : vector<16xi8> to vector<4xi32>
    // CHECK-COUNT-16: llvm.extractvalue %{{.*}} : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK-COUNT-16: llvm.insertelement {{.*}} : vector<16xi8>
    // CHECK: llvm.bitcast %{{.*}} : vector<16xi8> to vector<4xi32>
    // CHECK-COUNT-8: llvm.extractvalue %{{.*}} : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>
    // CHECK: wmma.i32.16x16x16.iu8{{.*}} : (i1, vector<4xi32>, i1, vector<4xi32>, vector<8xi32>, i1) -> vector<8xi32>
    %0 = tt.dot %arg0, %arg1, %arg2 {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<16x16xi8, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>> * tensor<16x16xi8, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>> -> tensor<16x16xi32, #mma1>
    // CHECK-COUNT-8: llvm.insertvalue {{.*}} : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>
    tt.return
  }

  //  CHECK-LABEL: wmma1_dot_int4_32
  tt.func @wmma1_dot_int4_32(%arg0: tensor<16x16xi4, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>>, %arg1: tensor<16x16xi4, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>>, %arg2: tensor<16x16xi32, #mma1>) {
    // CHECK-COUNT-16: llvm.extractvalue %{{.*}} : !llvm.struct<(i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4)>
    // CHECK-COUNT-16: llvm.insertelement {{.*}} : vector<16xi4>
    // CHECK: llvm.bitcast %{{.*}} : vector<16xi4> to vector<2xi32>
    // CHECK-COUNT-16: llvm.extractvalue %{{.*}} : !llvm.struct<(i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4)>
    // CHECK-COUNT-16: llvm.insertelement {{.*}} : vector<16xi4>
    // CHECK: llvm.bitcast %{{.*}} : vector<16xi4> to vector<2xi32>
    // CHECK-COUNT-8: llvm.extractvalue %{{.*}} : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>
    // CHECK: wmma.i32.16x16x16.iu4{{.*}} : (i1, vector<2xi32>, i1, vector<2xi32>, vector<8xi32>, i1) -> vector<8xi32>
    %0 = tt.dot %arg0, %arg1, %arg2 {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<16x16xi4, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>> * tensor<16x16xi4, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>> -> tensor<16x16xi32, #mma1>
    // CHECK-COUNT-8: llvm.insertvalue {{.*}} : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>
    tt.return
  }

  //  CHECK-LABEL: wmma2_dot
  tt.func @wmma2_dot(%arg0: tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma2, kWidth = 8}>>, %arg1: tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma2, kWidth = 8}>>, %arg2: tensor<16x16xf16, #mma2>) {
    // CHECK-COUNT-8: llvm.extractvalue %{{.*}} : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16)>
    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8: llvm.extractvalue %{{.*}} : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16)>
    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8: llvm.extractvalue %{{.*}} : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16)>
    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK: llvm.call_intrinsic "llvm.amdgcn.wmma.f16.16x16x16.f16.v8f16.v8f16"{{.*}} : (vector<8xf16>, vector<8xf16>, vector<8xf16>, i1) -> vector<8xf16>
    %0 = tt.dot %arg0, %arg1, %arg2, inputPrecision = ieee : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma2, kWidth = 8}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma2, kWidth = 8}>> -> tensor<16x16xf16, #mma2>
    // CHECK-COUNT-8: llvm.extractelement {{.*}} : vector<8xf16>
    // CHECK: llvm.mlir.undef : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16)>
    // CHECK-COUNT-8: llvm.insertvalue {{.*}} : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16)>
    tt.return
  }

  // CHECK-LABEL: wmma2_transposed_dot
  tt.func @wmma2_transposed_dot(%arg0: tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma2_transposed, kWidth = 8}>>, %arg1: tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma2_transposed, kWidth = 8}>>, %arg2: tensor<16x16xf16, #mma2_transposed>) {
    // CHECK: llvm.call_intrinsic "llvm.amdgcn.wmma.f16.16x16x16.f16.v8f16.v8f16"{{.*}} : (vector<8xf16>, vector<8xf16>, vector<8xf16>, i1) -> vector<8xf16>
    %0 = tt.dot %arg0, %arg1, %arg2, inputPrecision = ieee : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma2_transposed, kWidth = 8}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma2_transposed, kWidth = 8}>> -> tensor<16x16xf16, #mma2_transposed>
    tt.return
  }

  //  CHECK-LABEL: blocked_to_wmma1
  tt.func @blocked_to_wmma1(%arg0: tensor<128x16xi32, #blocked>) {
    // CHECK-COUNT-16: llvm.extractvalue {{.*}} : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    // CHECK-COUNT-32: llvm.insertvalue {{.*}} : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    %0 = ttg.convert_layout %arg0 {allocation.offset = 0 : i32} : tensor<128x16xi32, #blocked> -> tensor<128x16xi32, #mma1>
    tt.return
  }

  //  CHECK-LABEL: slice_blocked_to_wmma1
  tt.func @slice_blocked_to_wmma1(%arg0: tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>) {
    // CHECK-COUNT-16: llvm.extractvalue {{.*}} : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    // CHECK-COUNT-1: llvm.insertvalue {{.*}} : !llvm.struct<(i32)>
    %0 = ttg.convert_layout %arg0 {allocation.offset = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<16xi32, #ttg.slice<{dim = 0, parent = #mma1}>>
    tt.return
  }

  //  CHECK-LABEL: wmma1_to_blocked
  tt.func @wmma1_to_blocked(%arg0: tensor<128x16xi32, #mma1>) {
    // CHECK-COUNT-32: llvm.extractvalue {{.*}} : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    // CHECK-COUNT-16: llvm.insertvalue {{.*}} : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    %0 = ttg.convert_layout %arg0 {allocation.offset = 0 : i32} : tensor<128x16xi32, #mma1> -> tensor<128x16xi32, #blocked>
    tt.return
  }

  //  CHECK-LABEL: slice_wmma1_to_blocked
  tt.func @slice_wmma1_to_blocked(%arg0: tensor<16xi32, #ttg.slice<{dim = 0, parent = #mma1}>>) {
    // CHECK-COUNT-1: llvm.extractvalue {{.*}} : !llvm.struct<(i32)>
    // CHECK-COUNT-16: llvm.insertvalue {{.*}} : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    %0 = ttg.convert_layout %arg0 {allocation.offset = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #mma1}>> -> tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    tt.return
  }

  //  CHECK-LABEL: blocked_to_wmma2
  tt.func @blocked_to_wmma2(%arg0: tensor<128x16xi32, #blocked>) {
    // CHECK-COUNT-16: llvm.extractvalue {{.*}} : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    // CHECK-COUNT-32: llvm.insertvalue {{.*}} : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    %0 = ttg.convert_layout %arg0 {allocation.offset = 0 : i32} : tensor<128x16xi32, #blocked> -> tensor<128x16xi32, #mma2>
    tt.return
  }

  //  CHECK-LABEL: slice_blocked_to_wmma2
  tt.func @slice_blocked_to_wmma2(%arg0: tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>) {
    // CHECK-COUNT-16: llvm.extractvalue {{.*}} : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    // CHECK-COUNT-1: llvm.insertvalue {{.*}} : !llvm.struct<(i32)>
    %0 = ttg.convert_layout %arg0 {allocation.offset = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<16xi32, #ttg.slice<{dim = 0, parent = #mma2}>>
    tt.return
  }

  //  CHECK-LABEL: wmma2_to_blocked
  tt.func @wmma2_to_blocked(%arg0: tensor<128x16xi32, #mma2>) {
    // CHECK-COUNT-32: llvm.extractvalue {{.*}} : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    // CHECK-COUNT-16: llvm.insertvalue {{.*}} : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    %0 = ttg.convert_layout %arg0 {allocation.offset = 0 : i32} : tensor<128x16xi32, #mma2> -> tensor<128x16xi32, #blocked>
    tt.return
  }

  //  CHECK-LABEL: slice_wmma2_to_blocked
  tt.func @slice_wmma2_to_blocked(%arg0: tensor<16xi32, #ttg.slice<{dim = 0, parent = #mma2}>>) {
    // CHECK-COUNT-1: llvm.extractvalue {{.*}} : !llvm.struct<(i32)>
    // CHECK-COUNT-16: llvm.insertvalue {{.*}} : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    %0 = ttg.convert_layout %arg0 {allocation.offset = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #mma2}>> -> tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}>
#mma1 = #ttg.amd_wmma<{version = 1, warpsPerCTA = [2, 1, 4]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: wmma_dot_operand3d
  tt.func @wmma_dot_operand3d(%arg0: !ttg.memdesc<4x16x32xf16, #shared, #smem>) {
    // CHECK-COUNT-4: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<16xf16>
    %0 = ttg.local_load %arg0 : !ttg.memdesc<4x16x32xf16, #shared, #smem> -> tensor<4x16x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>>
    // CHECK-COUNT-32: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<1xf16>
    %1 = ttg.local_load %arg0 : !ttg.memdesc<4x16x32xf16, #shared, #smem> -> tensor<4x16x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>>
    tt.return
  }

  // CHECK-LABEL: wmma_dot3d
  tt.func @wmma_dot3d(%arg0: tensor<2x16x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>>, %arg1: tensor<2x32x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>>, %arg2: tensor<2x16x16xf16, #mma1>) {
    // CHECK-COUNT-32: llvm.extractvalue %arg0
    // CHECK-COUNT-32: llvm.insertelement
    // CHECK-COUNT-32: llvm.extractvalue %arg1
    // CHECK-COUNT-32: llvm.insertelement
    // CHECK-COUNT-8: llvm.extractvalue %arg2
    // CHECK-COUNT-8: llvm.insertelement
    // CHECK-COUNT-2: wmma.f16.16x16x16.f16{{.*}} : (vector<16xf16>, vector<16xf16>, vector<16xf16>, i1) -> vector<16xf16>
    %0 = tt.dot %arg0, %arg1, %arg2, inputPrecision = ieee : tensor<2x16x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>> * tensor<2x32x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>> -> tensor<2x16x16xf16, #mma1>
    // CHECK-COUNT-8: llvm.extractelement
    // CHECK-COUNT-8: llvm.insertvalue
    tt.return
  }
}
