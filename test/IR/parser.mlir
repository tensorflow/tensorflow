; TODO(andydavis) Resolve relative path issue w.r.t invoking mlir-opt in RUN
; statements (perhaps through using lit config substitutions).
;
; RUN: %S/../../mlir-opt %s -o - | FileCheck %s


; CHECK: extfunc @foo(i32, i64) -> f32
extfunc @foo(i32, i64) -> f32

; CHECK: extfunc @bar()
extfunc @bar() -> ()

; CHECK: extfunc @baz() -> (i1, int, f32)
extfunc @baz() -> (i1, int, f32)

; CHECK: extfunc @missingReturn()
extfunc @missingReturn()


; CHECK: extfunc @vectors(vector<1xf32>, vector<2x4xf32>)
extfunc @vectors(vector<1 x f32>, vector<2x4xf32>)

; CHECK: extfunc @tensors(i1, i1, i1, i1)
extfunc @tensors(tensor<?? f32>, tensor<?? vector<2x4xf32>>,
                 tensor<1x?x4x?x?xint>, tensor<i8>)

; CHECK: extfunc @memrefs(i1, i1)
extfunc @memrefs(memref<1x?x4x?x?xint>, memref<i8>)

; CHECK: extfunc @functions((i1, i1) -> (), () -> ())
extfunc @functions((memref<1x?x4x?x?xint>, memref<i8>) -> (), ()->())


; CHECK-LABEL: cfgfunc @simpleCFG() {
cfgfunc @simpleCFG() {
bb42:       ; CHECK: bb0:
  return    ; CHECK: return
}           ; CHECK: }

; CHECK-LABEL: cfgfunc @multiblock() -> i32 {
cfgfunc @multiblock() -> i32 {
bb0:         ; CHECK: bb0:
  return     ; CHECK: return
bb4:         ; CHECK: bb1:
  return     ; CHECK: return
}            ; CHECK: }
