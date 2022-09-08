# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
# Copyright 2022 The StableHLO Authors.
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
add_custom_target(check-stablehlo-python)

function(add_stablehlo_python_test test_name file_name)
add_custom_target(${test_name}
  PYTHONPATH=${STABLEHLO_BINARY_DIR}/python_packages/stablehlo:$ENV{PYTHONPATH} ${Python3_EXECUTABLE} ${file_name}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS
    StablehloUnifiedPythonModules
)
add_dependencies(check-stablehlo-python ${test_name})
endfunction()

add_stablehlo_python_test(stablehlo-python-smoketest smoketest.py)

add_dependencies(check-stablehlo check-stablehlo-python)
