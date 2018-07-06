; TODO(andydavis) Resolve relative path issue w.r.t invoking mlir-opt in RUN
; statements (perhaps through using lit config substitutions).
;
; RUN: %S/../../mlir-opt %s -o - | FileCheck %s


; CHECK: extfunc @foo(i32, i64) -> f32
extfunc @foo(i32, i64) -> f32

; CHECK: extfunc @bar()
extfunc @bar() -> ()

; CHECK: extfunc @baz() -> (i1, affineint, f32)
extfunc @baz() -> (i1, affineint, f32)

; CHECK: extfunc @missingReturn()
extfunc @missingReturn()

; CHECK: extfunc @int_types(i1, i2, i4, i7, i87) -> (i1, affineint, i19)
extfunc @int_types(i1, i2, i4, i7, i87) -> (i1, affineint, i19)


; CHECK: extfunc @vectors(vector<1xf32>, vector<2x4xf32>)
extfunc @vectors(vector<1 x f32>, vector<2x4xf32>)

; CHECK: extfunc @tensors(tensor<??f32>, tensor<??vector<2x4xf32>>, tensor<1x?x4x?x?xaffineint>, tensor<i8>)
extfunc @tensors(tensor<?? f32>, tensor<?? vector<2x4xf32>>,
                 tensor<1x?x4x?x?xaffineint>, tensor<i8>)

; CHECK: extfunc @memrefs(i1, i1)
extfunc @memrefs(memref<1x?x4x?x?xaffineint>, memref<i8>)

; CHECK: extfunc @functions((i1, i1) -> (), () -> ())
extfunc @functions((memref<1x?x4x?x?xaffineint>, memref<i8>) -> (), ()->())


; CHECK-LABEL: cfgfunc @simpleCFG() {
cfgfunc @simpleCFG() {
bb42:       ; CHECK: bb0:
  "foo"()   ; CHECK: "foo"()
  "bar"()   ; CHECK: "bar"()
  return    ; CHECK: return
}           ; CHECK: }

; CHECK-LABEL: cfgfunc @multiblock() -> i32 {
cfgfunc @multiblock() -> i32 {
bb0:         ; CHECK: bb0:
  return     ; CHECK:   return
bb1:         ; CHECK: bb1:
  br bb4     ; CHECK:   br bb3
bb2:         ; CHECK: bb2:
  br bb2     ; CHECK:   br bb2
bb4:         ; CHECK: bb3:
  return     ; CHECK:   return
}            ; CHECK: }

; CHECK-LABEL: mlfunc @simpleMLF() {
mlfunc @simpleMLF() {
  return     ; CHECK:  return
}            ; CHECK: }

; CHECK-LABEL: mlfunc @loops() {
mlfunc @loops() {
  for {      ; CHECK:   for {
    for {    ; CHECK:     for {
    }        ; CHECK:     }
  }          ; CHECK:   }
  return     ; CHECK:   return
}            ; CHECK: }

; CHECK-LABEL: mlfunc @ifstmt() {
mlfunc @ifstmt() {
  for {          ; CHECK   for {
    if {         ; CHECK     if {
    } else if {  ; CHECK     } else if {
    } else {     ; CHECK     } else {
    }            ; CHECK     }
  }              ; CHECK   }
  return         ; CHECK   return
}                ; CHECK }

; CHECK-LABEL: cfgfunc @attributes() {
cfgfunc @attributes() {
bb42:       ; CHECK: bb0:

  ; CHECK: "foo"()
  "foo"(){}

  ; CHECK: "foo"(){a: 1, b: -423, c: [true, false]}
  "foo"(){a: 1, b: -423, c: [true, false] }

  ; CHECK: "foo"(){cfgfunc: [], i123: 7, if: "foo"}
  "foo"(){if: "foo", cfgfunc: [], i123: 7}

  return
}

; CHECK-LABEL: cfgfunc @standard_instrs() {
cfgfunc @standard_instrs() {
bb42:       ; CHECK: bb0:
  ; CHECK: dim xxx, 2 : sometype
  "dim"(){index: 2}

  ; CHECK: addf xx, yy : sometype
  "addf"()
  return
}
