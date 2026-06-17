cmake_minimum_required(VERSION 3.14)
project(TestAOT)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_definitions(-DTF_PIP_INTEGRATION_TEST)

set(TENSORFLOW_PACKAGE_PATH "/usr/local/lib/python3.9/dist-packages/tensorflow"
    CACHE STRING "Full path of tensorflow python package")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_executable(aot_compiled_test
        aot_compiled_test.cc
        aot_compiled_vars_and_arithmetic.o
        aot_compiled_vars_and_arithmetic_frozen.o
        aot_compiled_x_matmul_y_small.o
        aot_compiled_x_matmul_y_large.o
        aot_compiled_x_matmul_y_large_multithreaded.o
        aot_compiled_x_plus_y.o
        )

# The paths assume the setup performed by run_integration_test.sh.
target_include_directories(aot_compiled_test PUBLIC
        "/tmp/generated_models"
        "${TENSORFLOW_PACKAGE_PATH}/include"
        "/tmp/tf_third_party")

add_subdirectory("${TENSORFLOW_PACKAGE_PATH}/xla_aot_runtime_src"
        ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/tf_runtime)

target_link_libraries(aot_compiled_test GTest::gtest_main tf_xla_runtime)

include(GoogleTest)
gtest_discover_tests(aot_compiled_test)
