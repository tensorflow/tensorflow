# Copyright 2022	 The TensorFlow Authors. All Rights Reserved.
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
configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(MHLO_TOSA_TEST_DEPENDS
        FileCheck count not
        mhlo-tosa-opt
)

add_lit_testsuite(check-mhlo-tosa-lit "Running the mhlo-tosa regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${MHLO_TOSA_TEST_DEPENDS}
        )
set_target_properties(check-mhlo-tosa-lit PROPERTIES FOLDER "Tests")

add_lit_testsuites(MHLO_TOSA_OPT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${MHLO_TOSA_TEST_DEPENDS})

add_dependencies(check-mhlo-tosa check-mhlo-tosa-lit)
