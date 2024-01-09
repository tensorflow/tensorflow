func.func @add_i32_10(%c0: i32) -> (i32, i32, i32) {
  %c1 = "test_mlbc.add.i32"(%c0, %c0) : (i32, i32) -> i32
  %c2 = "test_mlbc.sub.i32"(%c1, %c0) : (i32, i32) -> i32
  %c3 = "test_mlbc.add.i32"(%c2, %c0) : (i32, i32) -> i32
  %c4 = "test_mlbc.sub.i32"(%c3, %c0) : (i32, i32) -> i32
  %c5 = "test_mlbc.add.i32"(%c4, %c0) : (i32, i32) -> i32
  %c6 = "test_mlbc.sub.i32"(%c5, %c0) : (i32, i32) -> i32
  %c7 = "test_mlbc.add.i32"(%c6, %c0) : (i32, i32) -> i32
  %c8 = "test_mlbc.sub.i32"(%c7, %c0) : (i32, i32) -> i32
  %c9 = "test_mlbc.add.i32"(%c8, %c0) : (i32, i32) -> i32
  %c10, %c11, %c12 = call @add_i32_10(%c9) : (i32) -> (i32, i32, i32)
  func.return %c0, %c10, %c10 : i32, i32, i32
}
