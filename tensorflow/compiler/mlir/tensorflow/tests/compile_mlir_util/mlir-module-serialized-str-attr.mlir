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
// RUN: tf-mlir-translate -mlir-tf-mlir-to-str-attr -mlir-print-local-scope %s | FileCheck %s

module attributes {tf.versions = {producer = 888 : i32}} {
  func.func @main(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.Identity"(%arg0) : (tensor<?xi32>) -> tensor<?xi32> loc(unknown)
    func.return %0 : tensor<?xi32> loc(unknown)
  } loc(unknown)
} loc(unknown)

// CHECK: module attributes {tf.versions = {producer = 888 : i32}} {\0A func.func @main(%arg0: tensor<?xi32> loc({{.*}})) -> tensor<?xi32> {\0A %0 = \22tf.Identity\22(%arg0) : (tensor<?xi32>) -> tensor<?xi32> loc(unknown)\0A return %0 : tensor<?xi32> loc(unknown)\0A } loc(unknown)\0A} loc(unknown)"
