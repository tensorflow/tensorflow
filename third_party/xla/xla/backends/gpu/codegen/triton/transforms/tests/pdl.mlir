// RUN: xla-opt %s -split-input-file -xla-gpu-insert-pdl \
// RUN:   | FileCheck %s --check-prefix=INSERT
// RUN: xla-opt %s -split-input-file -xla-gpu-insert-pdl \
// RUN:   -triton-xla-extract-insert-to-triton \
// RUN:   -xla-lower-pdl-wait \
// RUN:   | FileCheck %s --check-prefix=LOWER
// RUN: xla-opt %s -split-input-file -xla-gpu-insert-pdl \
// RUN:   -triton-xla-extract-insert-to-triton="allow_tma=1 num_stages=3" \
// RUN:   -xla-lower-pdl-wait \
// RUN:   | FileCheck %s --check-prefix=LOWER-TMA

func.func @basic(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) {
  %extracted_tensor = triton_xla.extract from %arg0
      as memref<128xbf16, #xtile.layout<[0]>>
      [0] [16] [1] : tensor<16xbf16>
  triton_xla.insert %extracted_tensor into %arg1
      as memref<256xbf16, #xtile.layout<[0]>>
      [0] [16] [1] : tensor<16xbf16>
  func.return
}

// INSERT-LABEL:  func.func @basic(
// INSERT:        xla_gpu.pdl_wait
// INSERT:        %[[EXTRACTED:.*]] = triton_xla.extract from %arg0
// INSERT:        triton_xla.insert %[[EXTRACTED]] into %arg1
// INSERT:        return

// LOWER-LABEL:   tt.func @basic(
// LOWER:         nvvm.griddepcontrol wait
// LOWER-NOT:     nvvm.griddepcontrol wait
// LOWER:         tt.load
// LOWER:         tt.store
// LOWER:         tt.return

// LOWER-TMA-LABEL:   tt.func @basic(
// LOWER-TMA-SAME:    %arg0: !tt.tensordesc<16xbf16>
// LOWER-TMA-SAME:    %arg1: !tt.tensordesc<16xbf16>
// LOWER-TMA:         nvvm.griddepcontrol wait
// LOWER-TMA-NOT:     nvvm.griddepcontrol wait
// LOWER-TMA:         %[[LOAD:.*]] = tt.descriptor_load %arg0
// LOWER-TMA:         tt.descriptor_store %arg1[{{.*}}], %[[LOAD]]
// LOWER-TMA-NOT:     tt.load
// LOWER-TMA-NOT:     tt.store
// LOWER-TMA:         tt.return
