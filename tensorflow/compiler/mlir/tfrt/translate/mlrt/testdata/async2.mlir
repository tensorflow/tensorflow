func.func @add_i32(%arg0: i32, %arg1: i32) -> i32 {
  %0 = "test_mlbc.add.i32"(%arg0, %arg1) : (i32, i32) -> i32
  func.return %0 : i32
}

func.func @main(%arg0: i32, %arg1: i32) -> i32 {
  %c1 = "test_mlbc.add.i32"(%arg0, %arg1) : (i32, i32) -> i32
 
  %handle = "mlrt.async"(%c1, %c1) {callee = @add_i32} : (i32, i32) -> !mlrt.async_handle

  "mlrt.await_handle"(%handle) : (!mlrt.async_handle) -> () 

  %c2 = "test_mlbc.add.i32"(%c1, %c1) : (i32, i32) -> i32

  func.return %c2 : i32
}