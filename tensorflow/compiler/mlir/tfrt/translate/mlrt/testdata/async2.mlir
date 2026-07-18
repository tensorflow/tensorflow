// Copyright 2026 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

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