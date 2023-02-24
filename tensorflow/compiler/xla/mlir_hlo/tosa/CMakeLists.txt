# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(check-mhlo-tosa)

add_subdirectory(transforms)
add_subdirectory(tests)

add_mlir_pdll_library(MHLOTOSAPDLLPatternsIncGen
  transforms/legalize_mhlo/legalize_mhlo.pdll
  transforms/legalize_mhlo/legalize_mhlo.pdll.h.inc
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib

        MLIRTosaDialect
        MhloRegisterDialects
        MhloRegisterDialects

        MHLOTOSATransforms
        )
add_llvm_executable(mhlo-tosa-opt mhlo_tosa_opt.cc
)
llvm_update_compile_flags(mhlo-tosa-opt)
add_dependencies(mhlo-tosa-opt MHLOTOSATransformsPassIncGen)
target_link_libraries(mhlo-tosa-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(mhlo-tosa-opt)
