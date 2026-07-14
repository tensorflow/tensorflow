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
// RUN: not tf-mlir-translate -mlir-tf-str-attr-to-mlir %s 2>&1 | FileCheck %s

"builtin.totally @invalid MLIR module {here} <-"

// CHECK: could not parse MLIR module-:1:1: error: custom op 'builtin.totally' is unknown
