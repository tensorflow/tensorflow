// RUN: mlir-opt %s -verify

// expected-error@+1{{llvm.noalias argument attribute of non boolean type}}
func @invalid_noalias(%arg0: !llvm<"i32"> {llvm.noalias: 3}) {
  "llvm.return"() : () -> ()
}
