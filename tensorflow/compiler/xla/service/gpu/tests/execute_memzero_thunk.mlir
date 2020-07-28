// RUN: xla-thunks-opt %s | FileCheck --color --dump-input=fail %s

func @main( %execute_params: !llvm<"i8*"> ) {
  // CHECK: "xla_thunks.execute_memzero_thunk"
  // CHECK-SAME: {allocation_index = 0 : i64, offset = 128 : i64, size = 1024 : i64}
  // CHECK-SAME: (!llvm<"i8*">) -> (i1, !llvm<"i8*">)
  %ok, %error_message =
      "xla_thunks.execute_memzero_thunk"( %execute_params )
          { allocation_slice = { allocation_index = 0
                               , offset = 128
                               , size = 1024 } }
          : (!llvm<"i8*">) -> (i1, !llvm<"i8*">)
  return
}

