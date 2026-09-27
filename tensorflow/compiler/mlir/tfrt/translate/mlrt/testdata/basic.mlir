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
