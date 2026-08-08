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
// RUN: tfg-transforms-opt %s --tfg-consolidate-attrs --tfg-prepare-attrs-export | FileCheck %s

// CHECK-LABEL: tfg.func generic @generic_func
// CHECK-SAME: !tf_type.tensor {tf._output_shapes = [#tf_type.shape<4>]}
tfg.func generic @generic_func(%arg0: !tf_type.tensor {tf._output_shapes = [#tf_type.shape<4>]})
  // CHECK-NEXT: !tf_type.tensor {tf._output_shapes = [#tf_type.shape<4>]}
  -> (!tf_type.tensor {tf._output_shapes = [#tf_type.shape<4>]})
    // CHECK-NEXT: attributes {tf._input_shapes = [#tf_type.shape<4>]}
    attributes {tf._input_shapes = [#tf_type.shape<4>]} {
  // CHECK-NEXT: A {_output_shapes = [#tf_type.shape<4>]}
  %A, %ctlA = A {_output_shapes = [#tf_type.shape<4>]} : () -> (!tf_type.tensor)
  return(%A) : !tf_type.tensor
}
