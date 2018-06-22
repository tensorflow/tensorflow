; TODO(andydavis) Resolve relative path issue w.r.t invoking mlir-opt in RUN
; statements (perhaps through using lit config substitutions).
;
; RUN: %S/../../mlir-opt %s -o - | FileCheck %s


; CHECK: extfunc @foo()
extfunc @foo(i32, i64) -> f32

; CHECK: extfunc @bar()
extfunc @bar() -> ()

; CHECK: extfunc @baz()
extfunc @baz() -> (i1, int, f32)

; CHECK: extfunc @missingReturn()
extfunc @missingReturn()


; CHECK: extfunc @vectors()
extfunc @vectors(vector<1 x f32>, vector<2x4xf32>)

; CHECK: extfunc @tensors()
extfunc @tensors(tensor<?? f32>, tensor<?? vector<2x4xf32>>,
                 tensor<1x?x4x?x?xint>, tensor<i8>)

; CHECK: extfunc @memrefs()
extfunc @memrefs(memref<1x?x4x?x?xint>, memref<i8>)
