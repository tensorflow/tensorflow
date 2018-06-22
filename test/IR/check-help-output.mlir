; TODO(andydavis) Resolve relative path issue w.r.t invoking mlir-opt in RUN
; statements (perhaps through using lit config substitutions).
;
; RUN: %S/../../mlir-opt --help | FileCheck %s
;
; CHECK: OVERVIEW: MLIR modular optimizer driver

