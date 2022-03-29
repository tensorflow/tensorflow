/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_DTENSOR_MLIR_DTENSOR_LOCATION_H_
#define TENSORFLOW_DTENSOR_MLIR_DTENSOR_LOCATION_H_

#include <string>

#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

// mlir::Location utilities for DTensor. `DTensorLocation` augments a location
// object with the current file and line _of the C++ code creating an
// operation_. This simplifies tracking down the creator of an invalid operation
// while debugging.
namespace tensorflow {
namespace dtensor {

mlir::Location DTensorLocation(mlir::Location loc, llvm::StringRef file,
                               unsigned int line);

mlir::Location DTensorLocation(mlir::Operation* op, llvm::StringRef file,
                               unsigned int line);

// Creates a string from a location of the following format:
//    >> pass_file_1:line1:col1
//    >> pass_file_2:line2:col2
//
// DTensor location format overloads the filename value to encode pass
// information.
//   original_file
//    >> pass_file_1:line1:col1
//    >> pass_file_2:line2:col2
//   original_line:original_col
std::string DTensorLocationToString(mlir::Location loc);

}  // namespace dtensor
}  // namespace tensorflow

#define DT_LOC(loc) \
  ::tensorflow::dtensor::DTensorLocation(loc, __FILE__, __LINE__)

#endif  // TENSORFLOW_DTENSOR_MLIR_DTENSOR_LOCATION_H_
