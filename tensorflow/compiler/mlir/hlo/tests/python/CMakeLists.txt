#
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

add_custom_target(check-mlir-hlo-python)

function(add_mhlo_python_test test_name file_name)
add_custom_target(${test_name}
  ${Python3_EXECUTABLE} ${file_name}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS
    MLIRHLOPythonModules
)
set_target_properties(${test_name}
  PROPERTIES ENVIRONMENT
  "PYTHONPATH=${MLIR_HLO_BINARY_DIR}/python_packages/mlir_hlo/mlir:$ENV{PYTHONPATH}"
)
add_dependencies(check-mlir-hlo-python ${test_name})
endfunction()

add_mhlo_python_test(python-attributes attributes.py)
add_mhlo_python_test(python-smoketest smoketest.py)
add_mhlo_python_test(python-types types.py)

add_dependencies(check-mlir-hlo check-mlir-hlo-python)
