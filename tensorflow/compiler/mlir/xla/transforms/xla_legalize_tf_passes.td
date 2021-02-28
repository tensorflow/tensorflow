/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Declare passes used in xla_legalize_tf.

include "mlir/Pass/PassBase.td"

def LegalizeTfTypesPass : Pass<"xla-legalize-tf-types"> {
  let summary = "Replace TensorFlow types with types that are legal in the MHLO dialect";

  let description = [{
The TF dialect uses some TF types that are illegal in the MHLO dialect and
some generic types that are legal in MHLO. This pass legalizes TF types into
types that are legal in MHLO. Rewrites here should run before TF to MHLO op
legalizations are run.

Specifically, this pass replaces each quantized integer type with the
corresponding ordinary types. For example, `TF::Qint8Type` is replaced with `i8`
everywhere it occurs. Types that are replaced are `TF::Qint8Type`,
`TF::Qint16Type`, `TF::Qint32Type`, `TF::Quint8Type`, and `TF::Quint16Type`.
  }];

  let constructor = "::mlir::mhlo::CreateLegalizeTfTypesPass()";
}
