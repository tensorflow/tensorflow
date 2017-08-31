# GRPC global cmake file
# This currently builds C and C++ code.
# This file has been automatically generated from a template file.
# Please look at the templates directory instead.
# This file can be regenerated from the template by running
# tools/buildgen/generate_projects.sh
#
# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



cmake_minimum_required(VERSION 2.8)

set(PACKAGE_NAME      "grpc")
set(PACKAGE_VERSION   "1.5.0-dev")
set(PACKAGE_STRING    "${PACKAGE_NAME} ${PACKAGE_VERSION}")
set(PACKAGE_TARNAME   "${PACKAGE_NAME}-${PACKAGE_VERSION}")
set(PACKAGE_BUGREPORT "https://github.com/grpc/grpc/issues/")
project(${PACKAGE_NAME} C CXX)

set(gRPC_INSTALL_BINDIR "${CMAKE_INSTALL_PREFIX}/bin" CACHE PATH "Installation directory for executables")
set(gRPC_INSTALL_LIBDIR "${CMAKE_INSTALL_PREFIX}/lib" CACHE PATH "Installation directory for libraries")
set(gRPC_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_PREFIX}/include" CACHE PATH "Installation directory for headers")
set(gRPC_INSTALL_CMAKEDIR "${CMAKE_INSTALL_PREFIX}/lib/cmake/${PACKAGE_NAME}" CACHE PATH "Installation directory for cmake config files")

# Options
option(gRPC_BUILD_TESTS "Build tests" OFF)

set(gRPC_INSTALL_default ON)
if (NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  # Disable gRPC_INSTALL by default if building as a submodule
  set(gRPC_INSTALL_default OFF)
endif()
set(gRPC_INSTALL ${gRPC_INSTALL_default} CACHE BOOL
    "Generate installation target: gRPC_ZLIB_PROVIDER, gRPC_CARES_PROVIDER, gRPC_SSL_PROVIDER and gRPC_PROTOBUF_PROVIDER must all be \"package\"")

set(gRPC_ZLIB_PROVIDER "module" CACHE STRING "Provider of zlib library")
set_property(CACHE gRPC_ZLIB_PROVIDER PROPERTY STRINGS "module" "package")

set(gRPC_CARES_PROVIDER "module" CACHE STRING "Provider of c-ares library")
set_property(CACHE gRPC_CARES_PROVIDER PROPERTY STRINGS "module" "package")

set(gRPC_SSL_PROVIDER "module" CACHE STRING "Provider of ssl library")
set_property(CACHE gRPC_SSL_PROVIDER PROPERTY STRINGS "module" "package")

set(gRPC_PROTOBUF_PROVIDER "module" CACHE STRING "Provider of protobuf library")
set_property(CACHE gRPC_PROTOBUF_PROVIDER PROPERTY STRINGS "module" "package")

set(gRPC_PROTOBUF_PACKAGE_TYPE "" CACHE STRING "Algorithm for searching protobuf package")
set_property(CACHE gRPC_PROTOBUF_PACKAGE_TYPE PROPERTY STRINGS "CONFIG" "MODULE")

set(gRPC_GFLAGS_PROVIDER "module" CACHE STRING "Provider of gflags library")
set_property(CACHE gRPC_GFLAGS_PROVIDER PROPERTY STRINGS "module" "package")

set(gRPC_BENCHMARK_PROVIDER "module" CACHE STRING "Provider of benchmark library")
set_property(CACHE gRPC_BENCHMARK_PROVIDER PROPERTY STRINGS "module" "package")

set(gRPC_USE_PROTO_LITE OFF CACHE BOOL "Use the protobuf-lite library")

if(UNIX)
  if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(_gRPC_PLATFORM_LINUX ON)
  elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(_gRPC_PLATFORM_MAC ON)
  else()
    set(_gRPC_PLATFORM_POSIX ON)
  endif()
endif()
if(WIN32)
  set(_gRPC_PLATFORM_WINDOWS ON)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

if (MSVC)
  include(cmake/msvc_static_runtime.cmake)
  add_definitions(-D_WIN32_WINNT=0x600 -D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_WARNINGS -D_WINSOCK_DEPRECATED_NO_WARNINGS)
  # needed to compile protobuf
  add_definitions(/wd4065 /wd4506)
  # TODO(jtattermusch): revisit C4267 occurrences throughout the code
  add_definitions(/wd4267)
endif()

if (gRPC_USE_PROTO_LITE)
  set(_gRPC_PROTOBUF_LIBRARY_NAME "libprotobuf-lite")
  add_definitions("-DGRPC_USE_PROTO_LITE")
else()
  set(_gRPC_PROTOBUF_LIBRARY_NAME "libprotobuf")
endif()

if("${gRPC_ZLIB_PROVIDER}" STREQUAL "module")
  if(NOT ZLIB_ROOT_DIR)
    set(ZLIB_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/zlib)
  endif()
  set(ZLIB_INCLUDE_DIR "${ZLIB_ROOT_DIR}")
  if(EXISTS "${ZLIB_ROOT_DIR}/CMakeLists.txt")
      # TODO(jtattermusch): workaround for https://github.com/madler/zlib/issues/218
      include_directories(${ZLIB_INCLUDE_DIR})

      add_subdirectory(${ZLIB_ROOT_DIR} third_party/zlib)
      if(TARGET zlibstatic)
          set(_gRPC_ZLIB_LIBRARIES zlibstatic)
      endif()
  else()
      message(WARNING "gRPC_ZLIB_PROVIDER is \"module\" but ZLIB_ROOT_DIR is wrong")
  endif()
  if(gRPC_INSTALL)
    message(WARNING "gRPC_INSTALL will be forced to FALSE because gRPC_ZLIB_PROVIDER is \"module\"")
    set(gRPC_INSTALL FALSE)
  endif()
elseif("${gRPC_ZLIB_PROVIDER}" STREQUAL "package")
  find_package(ZLIB)
  if(TARGET ZLIB::ZLIB)
    set(_gRPC_ZLIB_LIBRARIES ZLIB::ZLIB)
  endif()
  set(_gRPC_FIND_ZLIB "if(NOT ZLIB_FOUND)\n  find_package(ZLIB)\nendif()")
endif()

if("${gRPC_CARES_PROVIDER}" STREQUAL "module")
  if(NOT CARES_ROOT_DIR)
    set(CARES_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/c-ares)
  endif()
  string(TOLOWER ${CMAKE_SYSTEM_NAME} CARES_SYSTEM_NAME)
  set(CARES_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cares/cares")
  set(CARES_BUILD_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cares")
  set(CARES_PLATFORM_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cares/config_${CARES_SYSTEM_NAME}")
  if(EXISTS "${CARES_ROOT_DIR}/CMakeLists.txt")
    if("${CARES_SYSTEM_NAME}" MATCHES "windows")
      add_definitions(-DCARES_STATICLIB=1)
      add_definitions(-DWIN32_LEAN_AND_MEAN=1)
    else()
      add_definitions(-DHAVE_CONFIG_H=1)
      add_definitions(-D_GNU_SOURCE=1)
    endif()
    add_subdirectory(src/c-ares third_party/cares)
    if(TARGET cares)
        set(_gRPC_CARES_LIBRARIES cares)
    endif()
  else()
    message(WARNING "gRPC_CARES_PROVIDER is \"module\" but CARES_ROOT_DIR is wrong")
  endif()
  if(gRPC_INSTALL)
    message(WARNING "gRPC_INSTALL will be forced to FALSE because gRPC_CARES_PROVIDER is \"module\"")
    set(gRPC_INSTALL FALSE)
  endif()
elseif("${gRPC_CARES_PROVIDER}" STREQUAL "package")
  find_package(c-ares CONFIG)
  if(TARGET c-ares::cares)
    set(_gRPC_CARES_LIBRARIES c-ares::cares)
  endif()
  set(_gRPC_FIND_CARES "if(NOT c-ares_FOUND)\n  find_package(c-ares CONFIG)\nendif()")
endif()

if("${gRPC_PROTOBUF_PROVIDER}" STREQUAL "module")
  # Building the protobuf tests require gmock what is not part of a standard protobuf checkout.
  # Disable them unless they are explicitly requested from the cmake command line (when we assume
  # gmock is downloaded to the right location inside protobuf).
  if(NOT protobuf_BUILD_TESTS)
    set(protobuf_BUILD_TESTS OFF CACHE BOOL "Build protobuf tests")
  endif()
  # Disable building protobuf with zlib. Building protobuf with zlib breaks
  # the build if zlib is not installed on the system.
  if(NOT protobuf_WITH_ZLIB)
    set(protobuf_WITH_ZLIB OFF CACHE BOOL "Build protobuf with zlib.")
  endif()
  if(NOT PROTOBUF_ROOT_DIR)
    set(PROTOBUF_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/protobuf)
  endif()
  set(PROTOBUF_WELLKNOWN_IMPORT_DIR ${PROTOBUF_ROOT_DIR}/src)
  if(EXISTS "${PROTOBUF_ROOT_DIR}/cmake/CMakeLists.txt")
    set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "Link static runtime libraries")
    add_subdirectory(${PROTOBUF_ROOT_DIR}/cmake third_party/protobuf)
    if(TARGET ${_gRPC_PROTOBUF_LIBRARY_NAME})
      set(_gRPC_PROTOBUF_LIBRARIES ${_gRPC_PROTOBUF_LIBRARY_NAME})
    endif()
    if(TARGET libprotoc)
      set(_gRPC_PROTOBUF_PROTOC_LIBRARIES libprotoc)
    endif()
    if(TARGET protoc)
      set(_gRPC_PROTOBUF_PROTOC protoc)
    endif()
  else()
      message(WARNING "gRPC_PROTOBUF_PROVIDER is \"module\" but PROTOBUF_ROOT_DIR is wrong")
  endif()
  if(gRPC_INSTALL)
    message(WARNING "gRPC_INSTALL will be forced to FALSE because gRPC_PROTOBUF_PROVIDER is \"module\"")
    set(gRPC_INSTALL FALSE)
  endif()
elseif("${gRPC_PROTOBUF_PROVIDER}" STREQUAL "package")
  find_package(Protobuf ${gRPC_PROTOBUF_PACKAGE_TYPE})
  if(Protobuf_FOUND OR PROTOBUF_FOUND)
    if(TARGET protobuf::${_gRPC_PROTOBUF_LIBRARY_NAME})
      set(_gRPC_PROTOBUF_LIBRARIES protobuf::${_gRPC_PROTOBUF_LIBRARY_NAME})
    else()
      set(_gRPC_PROTOBUF_LIBRARIES ${PROTOBUF_LIBRARIES})
    endif()
    if(TARGET protobuf::libprotoc)
      set(_gRPC_PROTOBUF_PROTOC_LIBRARIES protobuf::libprotoc)
    else()
      set(_gRPC_PROTOBUF_PROTOC_LIBRARIES ${PROTOBUF_PROTOC_LIBRARIES})
    endif()
    if(TARGET protobuf::protoc)
      set(_gRPC_PROTOBUF_PROTOC protobuf::protoc)
    else()
      set(_gRPC_PROTOBUF_PROTOC ${PROTOBUF_PROTOC_EXECUTABLE})
    endif()
    set(_gRPC_FIND_PROTOBUF "if(NOT Protobuf_FOUND AND NOT PROTOBUF_FOUND)\n  find_package(Protobuf ${gRPC_PROTOBUF_PACKAGE_TYPE})\nendif()")
  endif()
  if(PROTOBUF_FOUND)
    include_directories(${PROTOBUF_INCLUDE_DIRS})
  endif()
  set(PROTOBUF_WELLKNOWN_IMPORT_DIR /usr/local/include)
endif()

if("${gRPC_SSL_PROVIDER}" STREQUAL "module")
  if(NOT BORINGSSL_ROOT_DIR)
    set(BORINGSSL_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/boringssl)
  endif()
  if(EXISTS "${BORINGSSL_ROOT_DIR}/CMakeLists.txt")
    set(OPENSSL_NO_ASM ON)  # make boringssl buildable with Visual Studio
    add_subdirectory(${BORINGSSL_ROOT_DIR} third_party/boringssl)
    if(TARGET ssl)
      set(_gRPC_SSL_LIBRARIES ssl)
    endif()
  else()
      message(WARNING "gRPC_SSL_PROVIDER is \"module\" but BORINGSSL_ROOT_DIR is wrong")
  endif()
  if(gRPC_INSTALL)
    message(WARNING "gRPC_INSTALL will be forced to FALSE because gRPC_SSL_PROVIDER is \"module\"")
    set(gRPC_INSTALL FALSE)
  endif()
elseif("${gRPC_SSL_PROVIDER}" STREQUAL "package")
  find_package(OpenSSL)
  if(TARGET OpenSSL::SSL)
    set(_gRPC_SSL_LIBRARIES OpenSSL::SSL)
  endif()
  set(_gRPC_FIND_SSL "if(NOT OpenSSL_FOUND)\n  find_package(OpenSSL)\nendif()")
endif()

if("${gRPC_GFLAGS_PROVIDER}" STREQUAL "module")
  if(NOT GFLAGS_ROOT_DIR)
    set(GFLAGS_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gflags)
  endif()
  if(EXISTS "${GFLAGS_ROOT_DIR}/CMakeLists.txt")
      add_subdirectory(${GFLAGS_ROOT_DIR} third_party/gflags)
      if(TARGET gflags_static)
          set(_gRPC_GFLAGS_LIBRARIES gflags_static)
      endif()
  else()
      message(WARNING "gRPC_GFLAGS_PROVIDER is \"module\" but GFLAGS_ROOT_DIR is wrong")
  endif()
elseif("${gRPC_GFLAGS_PROVIDER}" STREQUAL "package")
  find_package(gflags)
  if(TARGET gflags::gflags)
    set(_gRPC_GFLAGS_LIBRARIES gflags::gflags)
  endif()
  set(_gRPC_FIND_GFLAGS "if(NOT gflags_FOUND)\n  find_package(gflags)\nendif()")
endif()

if("${gRPC_BENCHMARK_PROVIDER}" STREQUAL "module")
  if(NOT BENCHMARK_ROOT_DIR)
    set(BENCHMARK_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/benchmark)
  endif()
  if(EXISTS "${BENCHMARK_ROOT_DIR}/CMakeLists.txt")
      add_subdirectory(${BENCHMARK_ROOT_DIR} third_party/benchmark)
      if(TARGET benchmark)
          set(_gRPC_BENCHMARK_LIBRARIES benchmark)
      endif()
  else()
      message(WARNING "gRPC_BENCHMARK_PROVIDER is \"module\" but BENCHMARK_ROOT_DIR is wrong")
  endif()
elseif("${gRPC_BENCHMARK_PROVIDER}" STREQUAL "package")
  find_package(benchmark)
  if(TARGET benchmark::benchmark)
    set(_gRPC_BENCHMARK_LIBRARIES benchmark::benchmark)
  endif()
  set(_gRPC_FIND_BENCHMARK "if(NOT benchmark_FOUND)\n  find_package(benchmark)\nendif()")
endif()

if(NOT MSVC)
  set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -std=c99")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

if(_gRPC_PLATFORM_MAC)
  set(_gRPC_ALLTARGETS_LIBRARIES ${CMAKE_DL_LIBS} m pthread)
elseif(UNIX)
  set(_gRPC_ALLTARGETS_LIBRARIES ${CMAKE_DL_LIBS} rt m pthread)
endif()

if(WIN32 AND MSVC)
  set(_gRPC_BASELIB_LIBRARIES wsock32 ws2_32)
endif()

# Create directory for generated .proto files
set(_gRPC_PROTO_GENS_DIR ${CMAKE_BINARY_DIR}/gens)
file(MAKE_DIRECTORY ${_gRPC_PROTO_GENS_DIR})

#  protobuf_generate_grpc_cpp
#  --------------------------
#
#   Add custom commands to process ``.proto`` files to C++ using protoc and
#   GRPC plugin::
#
#     protobuf_generate_grpc_cpp [<ARGN>...]
#
#   ``ARGN``
#     ``.proto`` files
#
function(protobuf_generate_grpc_cpp)
  if(NOT ARGN)
    message(SEND_ERROR "Error: PROTOBUF_GENERATE_GRPC_CPP() called without any proto files")
    return()
  endif()

  set(_protobuf_include_path -I . -I ${PROTOBUF_WELLKNOWN_IMPORT_DIR})
  foreach(FIL ${ARGN})
    get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
    get_filename_component(FIL_WE ${FIL} NAME_WE)
    file(RELATIVE_PATH REL_FIL ${CMAKE_CURRENT_SOURCE_DIR} ${ABS_FIL})
    get_filename_component(REL_DIR ${REL_FIL} DIRECTORY)
    set(RELFIL_WE "${REL_DIR}/${FIL_WE}")

    add_custom_command(
      OUTPUT "${_gRPC_PROTO_GENS_DIR}/${RELFIL_WE}.grpc.pb.cc"
             "${_gRPC_PROTO_GENS_DIR}/${RELFIL_WE}.grpc.pb.h"
             "${_gRPC_PROTO_GENS_DIR}/${RELFIL_WE}_mock.grpc.pb.h"
             "${_gRPC_PROTO_GENS_DIR}/${RELFIL_WE}.pb.cc"
             "${_gRPC_PROTO_GENS_DIR}/${RELFIL_WE}.pb.h"
      COMMAND $<TARGET_FILE:${_gRPC_PROTOBUF_PROTOC}>
      ARGS --grpc_out=generate_mock_code=true:${_gRPC_PROTO_GENS_DIR}
           --cpp_out=${_gRPC_PROTO_GENS_DIR}
           --plugin=protoc-gen-grpc=$<TARGET_FILE:grpc_cpp_plugin>
           ${_protobuf_include_path}
           ${REL_FIL}
      DEPENDS ${ABS_FIL} ${_gRPC_PROTOBUF_PROTOC} grpc_cpp_plugin
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Running gRPC C++ protocol buffer compiler on ${FIL}"
      VERBATIM)

      set_source_files_properties("${_gRPC_PROTO_GENS_DIR}/${RELFIL_WE}.grpc.pb.cc" "${_gRPC_PROTO_GENS_DIR}/${RELFIL_WE}.grpc.pb.h"  "${_gRPC_PROTO_GENS_DIR}/${RELFIL_WE}_mock.grpc.pb.h" "${_gRPC_PROTO_GENS_DIR}/${RELFIL_WE}.pb.cc" "${_gRPC_PROTO_GENS_DIR}/${RELFIL_WE}.pb.h" PROPERTIES GENERATED TRUE)
  endforeach()
endfunction()

add_custom_target(plugins
  DEPENDS
  grpc_cpp_plugin
  grpc_csharp_plugin
  grpc_node_plugin
  grpc_objective_c_plugin
  grpc_php_plugin
  grpc_python_plugin
  grpc_ruby_plugin
)

add_custom_target(tools_c
  DEPENDS
  check_epollexclusive
  gen_hpack_tables
  gen_legal_metadata_characters
  gen_percent_encoding_tables
  grpc_create_jwt
  grpc_print_google_default_creds_token
  grpc_verify_jwt
)

add_custom_target(tools_cxx
  DEPENDS
)

add_custom_target(tools
  DEPENDS tools_c tools_cxx)

if (gRPC_BUILD_TESTS)
add_custom_target(buildtests_c)
add_dependencies(buildtests_c alarm_test)
add_dependencies(buildtests_c algorithm_test)
add_dependencies(buildtests_c alloc_test)
add_dependencies(buildtests_c alpn_test)
add_dependencies(buildtests_c arena_test)
add_dependencies(buildtests_c bad_server_response_test)
add_dependencies(buildtests_c bdp_estimator_test)
add_dependencies(buildtests_c bin_decoder_test)
add_dependencies(buildtests_c bin_encoder_test)
add_dependencies(buildtests_c census_context_test)
add_dependencies(buildtests_c census_intrusive_hash_map_test)
add_dependencies(buildtests_c census_resource_test)
add_dependencies(buildtests_c census_trace_context_test)
add_dependencies(buildtests_c channel_create_test)
add_dependencies(buildtests_c chttp2_hpack_encoder_test)
add_dependencies(buildtests_c chttp2_stream_map_test)
add_dependencies(buildtests_c chttp2_varint_test)
add_dependencies(buildtests_c combiner_test)
add_dependencies(buildtests_c compression_test)
add_dependencies(buildtests_c concurrent_connectivity_test)
add_dependencies(buildtests_c connection_refused_test)
add_dependencies(buildtests_c dns_resolver_connectivity_test)
add_dependencies(buildtests_c dns_resolver_test)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_c dualstack_socket_test)
endif()
add_dependencies(buildtests_c endpoint_pair_test)
add_dependencies(buildtests_c error_test)
if(_gRPC_PLATFORM_LINUX)
add_dependencies(buildtests_c ev_epollsig_linux_test)
endif()
add_dependencies(buildtests_c fake_resolver_test)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_c fd_conservation_posix_test)
endif()
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_c fd_posix_test)
endif()
add_dependencies(buildtests_c fling_client)
add_dependencies(buildtests_c fling_server)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_c fling_stream_test)
endif()
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_c fling_test)
endif()
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_c goaway_server_test)
endif()
add_dependencies(buildtests_c gpr_avl_test)
add_dependencies(buildtests_c gpr_backoff_test)
add_dependencies(buildtests_c gpr_cmdline_test)
add_dependencies(buildtests_c gpr_cpu_test)
add_dependencies(buildtests_c gpr_env_test)
add_dependencies(buildtests_c gpr_histogram_test)
add_dependencies(buildtests_c gpr_host_port_test)
add_dependencies(buildtests_c gpr_log_test)
add_dependencies(buildtests_c gpr_mpscq_test)
add_dependencies(buildtests_c gpr_spinlock_test)
add_dependencies(buildtests_c gpr_stack_lockfree_test)
add_dependencies(buildtests_c gpr_string_test)
add_dependencies(buildtests_c gpr_sync_test)
add_dependencies(buildtests_c gpr_thd_test)
add_dependencies(buildtests_c gpr_time_test)
add_dependencies(buildtests_c gpr_tls_test)
add_dependencies(buildtests_c gpr_useful_test)
add_dependencies(buildtests_c grpc_auth_context_test)
add_dependencies(buildtests_c grpc_b64_test)
add_dependencies(buildtests_c grpc_byte_buffer_reader_test)
add_dependencies(buildtests_c grpc_channel_args_test)
add_dependencies(buildtests_c grpc_channel_stack_test)
add_dependencies(buildtests_c grpc_completion_queue_test)
add_dependencies(buildtests_c grpc_completion_queue_threading_test)
add_dependencies(buildtests_c grpc_credentials_test)
add_dependencies(buildtests_c grpc_fetch_oauth2)
add_dependencies(buildtests_c grpc_invalid_channel_args_test)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_c grpc_json_token_test)
endif()
add_dependencies(buildtests_c grpc_jwt_verifier_test)
add_dependencies(buildtests_c grpc_security_connector_test)
if(_gRPC_PLATFORM_LINUX)
add_dependencies(buildtests_c handshake_client)
endif()
if(_gRPC_PLATFORM_LINUX)
add_dependencies(buildtests_c handshake_server)
endif()
add_dependencies(buildtests_c hpack_parser_test)
add_dependencies(buildtests_c hpack_table_test)
add_dependencies(buildtests_c http_parser_test)
add_dependencies(buildtests_c httpcli_format_request_test)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_c httpcli_test)
endif()
if(_gRPC_PLATFORM_LINUX)
add_dependencies(buildtests_c httpscli_test)
endif()
add_dependencies(buildtests_c init_test)
add_dependencies(buildtests_c invalid_call_argument_test)
add_dependencies(buildtests_c json_rewrite)
add_dependencies(buildtests_c json_rewrite_test)
add_dependencies(buildtests_c json_stream_error_test)
add_dependencies(buildtests_c json_test)
add_dependencies(buildtests_c lame_client_test)
add_dependencies(buildtests_c lb_policies_test)
add_dependencies(buildtests_c load_file_test)
add_dependencies(buildtests_c memory_profile_client)
add_dependencies(buildtests_c memory_profile_server)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_c memory_profile_test)
endif()
add_dependencies(buildtests_c message_compress_test)
add_dependencies(buildtests_c minimal_stack_is_minimal_test)
add_dependencies(buildtests_c mlog_test)
add_dependencies(buildtests_c multiple_server_queues_test)
add_dependencies(buildtests_c murmur_hash_test)
add_dependencies(buildtests_c no_server_test)
add_dependencies(buildtests_c num_external_connectivity_watchers_test)
add_dependencies(buildtests_c parse_address_test)
add_dependencies(buildtests_c percent_encoding_test)
if(_gRPC_PLATFORM_LINUX)
add_dependencies(buildtests_c pollset_set_test)
endif()
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_c resolve_address_posix_test)
endif()
add_dependencies(buildtests_c resolve_address_test)
add_dependencies(buildtests_c resource_quota_test)
add_dependencies(buildtests_c secure_channel_create_test)
add_dependencies(buildtests_c secure_endpoint_test)
add_dependencies(buildtests_c sequential_connectivity_test)
add_dependencies(buildtests_c server_chttp2_test)
add_dependencies(buildtests_c server_test)
add_dependencies(buildtests_c slice_buffer_test)
add_dependencies(buildtests_c slice_hash_table_test)
add_dependencies(buildtests_c slice_string_helpers_test)
add_dependencies(buildtests_c slice_test)
add_dependencies(buildtests_c sockaddr_resolver_test)
add_dependencies(buildtests_c sockaddr_utils_test)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_c socket_utils_test)
endif()
add_dependencies(buildtests_c status_conversion_test)
add_dependencies(buildtests_c stream_compression_test)
add_dependencies(buildtests_c stream_owned_slice_test)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_c tcp_client_posix_test)
endif()
add_dependencies(buildtests_c tcp_client_uv_test)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_c tcp_posix_test)
endif()
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_c tcp_server_posix_test)
endif()
add_dependencies(buildtests_c tcp_server_uv_test)
add_dependencies(buildtests_c time_averaged_stats_test)
add_dependencies(buildtests_c timeout_encoding_test)
add_dependencies(buildtests_c timer_heap_test)
add_dependencies(buildtests_c timer_list_test)
add_dependencies(buildtests_c transport_connectivity_state_test)
add_dependencies(buildtests_c transport_metadata_test)
add_dependencies(buildtests_c transport_pid_controller_test)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_c transport_security_test)
endif()
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_c udp_server_test)
endif()
add_dependencies(buildtests_c uri_parser_test)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_c wakeup_fd_cv_test)
endif()
add_dependencies(buildtests_c public_headers_must_be_c89)
add_dependencies(buildtests_c badreq_bad_client_test)
add_dependencies(buildtests_c connection_prefix_bad_client_test)
add_dependencies(buildtests_c head_of_line_blocking_bad_client_test)
add_dependencies(buildtests_c headers_bad_client_test)
add_dependencies(buildtests_c initial_settings_frame_bad_client_test)
add_dependencies(buildtests_c large_metadata_bad_client_test)
add_dependencies(buildtests_c server_registered_method_bad_client_test)
add_dependencies(buildtests_c simple_request_bad_client_test)
add_dependencies(buildtests_c unknown_frame_bad_client_test)
add_dependencies(buildtests_c window_overflow_bad_client_test)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_c bad_ssl_cert_server)
endif()
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_c bad_ssl_cert_test)
endif()
add_dependencies(buildtests_c h2_census_test)
add_dependencies(buildtests_c h2_compress_test)
add_dependencies(buildtests_c h2_fakesec_test)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_c h2_fd_test)
endif()
add_dependencies(buildtests_c h2_full_test)
if(_gRPC_PLATFORM_LINUX)
add_dependencies(buildtests_c h2_full+pipe_test)
endif()
add_dependencies(buildtests_c h2_full+trace_test)
add_dependencies(buildtests_c h2_full+workarounds_test)
add_dependencies(buildtests_c h2_http_proxy_test)
add_dependencies(buildtests_c h2_load_reporting_test)
add_dependencies(buildtests_c h2_oauth2_test)
add_dependencies(buildtests_c h2_proxy_test)
add_dependencies(buildtests_c h2_sockpair_test)
add_dependencies(buildtests_c h2_sockpair+trace_test)
add_dependencies(buildtests_c h2_sockpair_1byte_test)
add_dependencies(buildtests_c h2_ssl_test)
add_dependencies(buildtests_c h2_ssl_cert_test)
add_dependencies(buildtests_c h2_ssl_proxy_test)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_c h2_uds_test)
endif()
add_dependencies(buildtests_c inproc_test)
add_dependencies(buildtests_c h2_census_nosec_test)
add_dependencies(buildtests_c h2_compress_nosec_test)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_c h2_fd_nosec_test)
endif()
add_dependencies(buildtests_c h2_full_nosec_test)
if(_gRPC_PLATFORM_LINUX)
add_dependencies(buildtests_c h2_full+pipe_nosec_test)
endif()
add_dependencies(buildtests_c h2_full+trace_nosec_test)
add_dependencies(buildtests_c h2_full+workarounds_nosec_test)
add_dependencies(buildtests_c h2_http_proxy_nosec_test)
add_dependencies(buildtests_c h2_load_reporting_nosec_test)
add_dependencies(buildtests_c h2_proxy_nosec_test)
add_dependencies(buildtests_c h2_sockpair_nosec_test)
add_dependencies(buildtests_c h2_sockpair+trace_nosec_test)
add_dependencies(buildtests_c h2_sockpair_1byte_nosec_test)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_c h2_uds_nosec_test)
endif()
add_dependencies(buildtests_c inproc_nosec_test)
add_dependencies(buildtests_c api_fuzzer_one_entry)
add_dependencies(buildtests_c client_fuzzer_one_entry)
add_dependencies(buildtests_c hpack_parser_fuzzer_test_one_entry)
add_dependencies(buildtests_c http_request_fuzzer_test_one_entry)
add_dependencies(buildtests_c http_response_fuzzer_test_one_entry)
add_dependencies(buildtests_c json_fuzzer_test_one_entry)
add_dependencies(buildtests_c nanopb_fuzzer_response_test_one_entry)
add_dependencies(buildtests_c nanopb_fuzzer_serverlist_test_one_entry)
add_dependencies(buildtests_c percent_decode_fuzzer_one_entry)
add_dependencies(buildtests_c percent_encode_fuzzer_one_entry)
add_dependencies(buildtests_c server_fuzzer_one_entry)
add_dependencies(buildtests_c ssl_server_fuzzer_one_entry)
add_dependencies(buildtests_c uri_fuzzer_test_one_entry)

add_custom_target(buildtests_cxx)
add_dependencies(buildtests_cxx alarm_cpp_test)
add_dependencies(buildtests_cxx async_end2end_test)
add_dependencies(buildtests_cxx auth_property_iterator_test)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_cxx bm_arena)
endif()
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_cxx bm_call_create)
endif()
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_cxx bm_chttp2_hpack)
endif()
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_cxx bm_chttp2_transport)
endif()
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_cxx bm_closure)
endif()
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_cxx bm_cq)
endif()
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_cxx bm_cq_multiple_threads)
endif()
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_cxx bm_error)
endif()
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_cxx bm_fullstack_streaming_ping_pong)
endif()
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_cxx bm_fullstack_streaming_pump)
endif()
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_cxx bm_fullstack_trickle)
endif()
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_cxx bm_fullstack_unary_ping_pong)
endif()
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_cxx bm_metadata)
endif()
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_cxx bm_pollset)
endif()
add_dependencies(buildtests_cxx channel_arguments_test)
add_dependencies(buildtests_cxx channel_filter_test)
add_dependencies(buildtests_cxx cli_call_test)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_cxx client_crash_test)
endif()
add_dependencies(buildtests_cxx client_crash_test_server)
add_dependencies(buildtests_cxx client_lb_end2end_test)
add_dependencies(buildtests_cxx codegen_test_full)
add_dependencies(buildtests_cxx codegen_test_minimal)
add_dependencies(buildtests_cxx credentials_test)
add_dependencies(buildtests_cxx cxx_byte_buffer_test)
add_dependencies(buildtests_cxx cxx_slice_test)
add_dependencies(buildtests_cxx cxx_string_ref_test)
add_dependencies(buildtests_cxx cxx_time_test)
add_dependencies(buildtests_cxx end2end_test)
add_dependencies(buildtests_cxx error_details_test)
add_dependencies(buildtests_cxx filter_end2end_test)
add_dependencies(buildtests_cxx generic_end2end_test)
add_dependencies(buildtests_cxx golden_file_test)
add_dependencies(buildtests_cxx grpc_cli)
add_dependencies(buildtests_cxx grpc_tool_test)
add_dependencies(buildtests_cxx grpclb_api_test)
add_dependencies(buildtests_cxx grpclb_end2end_test)
add_dependencies(buildtests_cxx grpclb_test)
add_dependencies(buildtests_cxx health_service_end2end_test)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_cxx http2_client)
endif()
add_dependencies(buildtests_cxx hybrid_end2end_test)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_cxx interop_client)
endif()
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_cxx interop_server)
endif()
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_cxx interop_test)
endif()
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_cxx json_run_localhost)
endif()
add_dependencies(buildtests_cxx memory_test)
add_dependencies(buildtests_cxx metrics_client)
add_dependencies(buildtests_cxx mock_test)
add_dependencies(buildtests_cxx noop-benchmark)
add_dependencies(buildtests_cxx proto_server_reflection_test)
add_dependencies(buildtests_cxx proto_utils_test)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_cxx qps_interarrival_test)
endif()
add_dependencies(buildtests_cxx qps_json_driver)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_cxx qps_openloop_test)
endif()
add_dependencies(buildtests_cxx qps_worker)
add_dependencies(buildtests_cxx reconnect_interop_client)
add_dependencies(buildtests_cxx reconnect_interop_server)
add_dependencies(buildtests_cxx secure_auth_context_test)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_cxx secure_sync_unary_ping_pong_test)
endif()
add_dependencies(buildtests_cxx server_builder_plugin_test)
add_dependencies(buildtests_cxx server_builder_test)
add_dependencies(buildtests_cxx server_context_test_spouse_test)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_cxx server_crash_test)
endif()
add_dependencies(buildtests_cxx server_crash_test_client)
add_dependencies(buildtests_cxx server_request_call_test)
add_dependencies(buildtests_cxx shutdown_test)
add_dependencies(buildtests_cxx status_test)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_cxx streaming_throughput_test)
endif()
add_dependencies(buildtests_cxx stress_test)
add_dependencies(buildtests_cxx thread_manager_test)
add_dependencies(buildtests_cxx thread_stress_test)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)
add_dependencies(buildtests_cxx writes_per_rpc_test)
endif()

add_custom_target(buildtests
  DEPENDS buildtests_c buildtests_cxx)
endif (gRPC_BUILD_TESTS)


add_library(gpr
  src/core/lib/profiling/basic_timers.c
  src/core/lib/profiling/stap_timers.c
  src/core/lib/support/alloc.c
  src/core/lib/support/arena.c
  src/core/lib/support/atm.c
  src/core/lib/support/avl.c
  src/core/lib/support/backoff.c
  src/core/lib/support/cmdline.c
  src/core/lib/support/cpu_iphone.c
  src/core/lib/support/cpu_linux.c
  src/core/lib/support/cpu_posix.c
  src/core/lib/support/cpu_windows.c
  src/core/lib/support/env_linux.c
  src/core/lib/support/env_posix.c
  src/core/lib/support/env_windows.c
  src/core/lib/support/histogram.c
  src/core/lib/support/host_port.c
  src/core/lib/support/log.c
  src/core/lib/support/log_android.c
  src/core/lib/support/log_linux.c
  src/core/lib/support/log_posix.c
  src/core/lib/support/log_windows.c
  src/core/lib/support/mpscq.c
  src/core/lib/support/murmur_hash.c
  src/core/lib/support/stack_lockfree.c
  src/core/lib/support/string.c
  src/core/lib/support/string_posix.c
  src/core/lib/support/string_util_windows.c
  src/core/lib/support/string_windows.c
  src/core/lib/support/subprocess_posix.c
  src/core/lib/support/subprocess_windows.c
  src/core/lib/support/sync.c
  src/core/lib/support/sync_posix.c
  src/core/lib/support/sync_windows.c
  src/core/lib/support/thd.c
  src/core/lib/support/thd_posix.c
  src/core/lib/support/thd_windows.c
  src/core/lib/support/time.c
  src/core/lib/support/time_posix.c
  src/core/lib/support/time_precise.c
  src/core/lib/support/time_windows.c
  src/core/lib/support/tls_pthread.c
  src/core/lib/support/tmpfile_msys.c
  src/core/lib/support/tmpfile_posix.c
  src/core/lib/support/tmpfile_windows.c
  src/core/lib/support/wrap_memcpy.c
)

if(WIN32 AND MSVC)
  set_target_properties(gpr PROPERTIES COMPILE_PDB_NAME "gpr"
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  if (gRPC_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/gpr.pdb
      DESTINATION ${gRPC_INSTALL_LIBDIR} OPTIONAL
    )
  endif()
endif()


target_include_directories(gpr
  PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${ZLIB_INCLUDE_DIR}
  PRIVATE ${BENCHMARK}/include
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(gpr
  ${_gRPC_ALLTARGETS_LIBRARIES}
)

foreach(_hdr
  include/grpc/support/alloc.h
  include/grpc/support/atm.h
  include/grpc/support/atm_gcc_atomic.h
  include/grpc/support/atm_gcc_sync.h
  include/grpc/support/atm_windows.h
  include/grpc/support/avl.h
  include/grpc/support/cmdline.h
  include/grpc/support/cpu.h
  include/grpc/support/histogram.h
  include/grpc/support/host_port.h
  include/grpc/support/log.h
  include/grpc/support/log_windows.h
  include/grpc/support/port_platform.h
  include/grpc/support/string_util.h
  include/grpc/support/subprocess.h
  include/grpc/support/sync.h
  include/grpc/support/sync_generic.h
  include/grpc/support/sync_posix.h
  include/grpc/support/sync_windows.h
  include/grpc/support/thd.h
  include/grpc/support/time.h
  include/grpc/support/tls.h
  include/grpc/support/tls_gcc.h
  include/grpc/support/tls_msvc.h
  include/grpc/support/tls_pthread.h
  include/grpc/support/useful.h
  include/grpc/impl/codegen/atm.h
  include/grpc/impl/codegen/atm_gcc_atomic.h
  include/grpc/impl/codegen/atm_gcc_sync.h
  include/grpc/impl/codegen/atm_windows.h
  include/grpc/impl/codegen/gpr_slice.h
  include/grpc/impl/codegen/gpr_types.h
  include/grpc/impl/codegen/port_platform.h
  include/grpc/impl/codegen/sync.h
  include/grpc/impl/codegen/sync_generic.h
  include/grpc/impl/codegen/sync_posix.h
  include/grpc/impl/codegen/sync_windows.h
)
  string(REPLACE "include/" "" _path ${_hdr})
  get_filename_component(_path ${_path} PATH)
  install(FILES ${_hdr}
    DESTINATION "${gRPC_INSTALL_INCLUDEDIR}/${_path}"
  )
endforeach()


if (gRPC_INSTALL)
  install(TARGETS gpr EXPORT gRPCTargets
    RUNTIME DESTINATION ${gRPC_INSTALL_BINDIR}
    LIBRARY DESTINATION ${gRPC_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${gRPC_INSTALL_LIBDIR}
  )
endif()

if (gRPC_BUILD_TESTS)

add_library(gpr_test_util
  test/core/util/test_config.c
)

if(WIN32 AND MSVC)
  set_target_properties(gpr_test_util PROPERTIES COMPILE_PDB_NAME "gpr_test_util"
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  if (gRPC_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/gpr_test_util.pdb
      DESTINATION ${gRPC_INSTALL_LIBDIR} OPTIONAL
    )
  endif()
endif()


target_include_directories(gpr_test_util
  PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${ZLIB_INCLUDE_DIR}
  PRIVATE ${BENCHMARK}/include
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(gpr_test_util
  ${_gRPC_ALLTARGETS_LIBRARIES}
  gpr
)


endif (gRPC_BUILD_TESTS)

add_library(grpc
  src/core/lib/surface/init.c
  src/core/lib/channel/channel_args.c
  src/core/lib/channel/channel_stack.c
  src/core/lib/channel/channel_stack_builder.c
  src/core/lib/channel/connected_channel.c
  src/core/lib/channel/handshaker.c
  src/core/lib/channel/handshaker_factory.c
  src/core/lib/channel/handshaker_registry.c
  src/core/lib/compression/compression.c
  src/core/lib/compression/message_compress.c
  src/core/lib/compression/stream_compression.c
  src/core/lib/http/format_request.c
  src/core/lib/http/httpcli.c
  src/core/lib/http/parser.c
  src/core/lib/iomgr/closure.c
  src/core/lib/iomgr/combiner.c
  src/core/lib/iomgr/endpoint.c
  src/core/lib/iomgr/endpoint_pair_posix.c
  src/core/lib/iomgr/endpoint_pair_uv.c
  src/core/lib/iomgr/endpoint_pair_windows.c
  src/core/lib/iomgr/error.c
  src/core/lib/iomgr/ev_epoll1_linux.c
  src/core/lib/iomgr/ev_epoll_limited_pollers_linux.c
  src/core/lib/iomgr/ev_epoll_thread_pool_linux.c
  src/core/lib/iomgr/ev_epollex_linux.c
  src/core/lib/iomgr/ev_epollsig_linux.c
  src/core/lib/iomgr/ev_poll_posix.c
  src/core/lib/iomgr/ev_posix.c
  src/core/lib/iomgr/ev_windows.c
  src/core/lib/iomgr/exec_ctx.c
  src/core/lib/iomgr/executor.c
  src/core/lib/iomgr/iocp_windows.c
  src/core/lib/iomgr/iomgr.c
  src/core/lib/iomgr/iomgr_posix.c
  src/core/lib/iomgr/iomgr_uv.c
  src/core/lib/iomgr/iomgr_windows.c
  src/core/lib/iomgr/is_epollexclusive_available.c
  src/core/lib/iomgr/load_file.c
  src/core/lib/iomgr/lockfree_event.c
  src/core/lib/iomgr/network_status_tracker.c
  src/core/lib/iomgr/polling_entity.c
  src/core/lib/iomgr/pollset_set_uv.c
  src/core/lib/iomgr/pollset_set_windows.c
  src/core/lib/iomgr/pollset_uv.c
  src/core/lib/iomgr/pollset_windows.c
  src/core/lib/iomgr/resolve_address_posix.c
  src/core/lib/iomgr/resolve_address_uv.c
  src/core/lib/iomgr/resolve_address_windows.c
  src/core/lib/iomgr/resource_quota.c
  src/core/lib/iomgr/sockaddr_utils.c
  src/core/lib/iomgr/socket_factory_posix.c
  src/core/lib/iomgr/socket_mutator.c
  src/core/lib/iomgr/socket_utils_common_posix.c
  src/core/lib/iomgr/socket_utils_linux.c
  src/core/lib/iomgr/socket_utils_posix.c
  src/core/lib/iomgr/socket_utils_uv.c
  src/core/lib/iomgr/socket_utils_windows.c
  src/core/lib/iomgr/socket_windows.c
  src/core/lib/iomgr/tcp_client_posix.c
  src/core/lib/iomgr/tcp_client_uv.c
  src/core/lib/iomgr/tcp_client_windows.c
  src/core/lib/iomgr/tcp_posix.c
  src/core/lib/iomgr/tcp_server_posix.c
  src/core/lib/iomgr/tcp_server_utils_posix_common.c
  src/core/lib/iomgr/tcp_server_utils_posix_ifaddrs.c
  src/core/lib/iomgr/tcp_server_utils_posix_noifaddrs.c
  src/core/lib/iomgr/tcp_server_uv.c
  src/core/lib/iomgr/tcp_server_windows.c
  src/core/lib/iomgr/tcp_uv.c
  src/core/lib/iomgr/tcp_windows.c
  src/core/lib/iomgr/time_averaged_stats.c
  src/core/lib/iomgr/timer_generic.c
  src/core/lib/iomgr/timer_heap.c
  src/core/lib/iomgr/timer_manager.c
  src/core/lib/iomgr/timer_uv.c
  src/core/lib/iomgr/udp_server.c
  src/core/lib/iomgr/unix_sockets_posix.c
  src/core/lib/iomgr/unix_sockets_posix_noop.c
  src/core/lib/iomgr/wakeup_fd_cv.c
  src/core/lib/iomgr/wakeup_fd_eventfd.c
  src/core/lib/iomgr/wakeup_fd_nospecial.c
  src/core/lib/iomgr/wakeup_fd_pipe.c
  src/core/lib/iomgr/wakeup_fd_posix.c
  src/core/lib/json/json.c
  src/core/lib/json/json_reader.c
  src/core/lib/json/json_string.c
  src/core/lib/json/json_writer.c
  src/core/lib/slice/b64.c
  src/core/lib/slice/percent_encoding.c
  src/core/lib/slice/slice.c
  src/core/lib/slice/slice_buffer.c
  src/core/lib/slice/slice_hash_table.c
  src/core/lib/slice/slice_intern.c
  src/core/lib/slice/slice_string_helpers.c
  src/core/lib/surface/alarm.c
  src/core/lib/surface/api_trace.c
  src/core/lib/surface/byte_buffer.c
  src/core/lib/surface/byte_buffer_reader.c
  src/core/lib/surface/call.c
  src/core/lib/surface/call_details.c
  src/core/lib/surface/call_log_batch.c
  src/core/lib/surface/channel.c
  src/core/lib/surface/channel_init.c
  src/core/lib/surface/channel_ping.c
  src/core/lib/surface/channel_stack_type.c
  src/core/lib/surface/completion_queue.c
  src/core/lib/surface/completion_queue_factory.c
  src/core/lib/surface/event_string.c
  src/core/lib/surface/lame_client.cc
  src/core/lib/surface/metadata_array.c
  src/core/lib/surface/server.c
  src/core/lib/surface/validate_metadata.c
  src/core/lib/surface/version.c
  src/core/lib/transport/bdp_estimator.c
  src/core/lib/transport/byte_stream.c
  src/core/lib/transport/connectivity_state.c
  src/core/lib/transport/error_utils.c
  src/core/lib/transport/metadata.c
  src/core/lib/transport/metadata_batch.c
  src/core/lib/transport/pid_controller.c
  src/core/lib/transport/service_config.c
  src/core/lib/transport/static_metadata.c
  src/core/lib/transport/status_conversion.c
  src/core/lib/transport/timeout_encoding.c
  src/core/lib/transport/transport.c
  src/core/lib/transport/transport_op_string.c
  src/core/lib/debug/trace.c
  src/core/ext/transport/chttp2/server/secure/server_secure_chttp2.c
  src/core/ext/transport/chttp2/transport/bin_decoder.c
  src/core/ext/transport/chttp2/transport/bin_encoder.c
  src/core/ext/transport/chttp2/transport/chttp2_plugin.c
  src/core/ext/transport/chttp2/transport/chttp2_transport.c
  src/core/ext/transport/chttp2/transport/frame_data.c
  src/core/ext/transport/chttp2/transport/frame_goaway.c
  src/core/ext/transport/chttp2/transport/frame_ping.c
  src/core/ext/transport/chttp2/transport/frame_rst_stream.c
  src/core/ext/transport/chttp2/transport/frame_settings.c
  src/core/ext/transport/chttp2/transport/frame_window_update.c
  src/core/ext/transport/chttp2/transport/hpack_encoder.c
  src/core/ext/transport/chttp2/transport/hpack_parser.c
  src/core/ext/transport/chttp2/transport/hpack_table.c
  src/core/ext/transport/chttp2/transport/http2_settings.c
  src/core/ext/transport/chttp2/transport/huffsyms.c
  src/core/ext/transport/chttp2/transport/incoming_metadata.c
  src/core/ext/transport/chttp2/transport/parsing.c
  src/core/ext/transport/chttp2/transport/stream_lists.c
  src/core/ext/transport/chttp2/transport/stream_map.c
  src/core/ext/transport/chttp2/transport/varint.c
  src/core/ext/transport/chttp2/transport/writing.c
  src/core/ext/transport/chttp2/alpn/alpn.c
  src/core/ext/filters/http/client/http_client_filter.c
  src/core/ext/filters/http/http_filters_plugin.c
  src/core/ext/filters/http/message_compress/message_compress_filter.c
  src/core/ext/filters/http/server/http_server_filter.c
  src/core/lib/http/httpcli_security_connector.c
  src/core/lib/security/context/security_context.c
  src/core/lib/security/credentials/composite/composite_credentials.c
  src/core/lib/security/credentials/credentials.c
  src/core/lib/security/credentials/credentials_metadata.c
  src/core/lib/security/credentials/fake/fake_credentials.c
  src/core/lib/security/credentials/google_default/credentials_generic.c
  src/core/lib/security/credentials/google_default/google_default_credentials.c
  src/core/lib/security/credentials/iam/iam_credentials.c
  src/core/lib/security/credentials/jwt/json_token.c
  src/core/lib/security/credentials/jwt/jwt_credentials.c
  src/core/lib/security/credentials/jwt/jwt_verifier.c
  src/core/lib/security/credentials/oauth2/oauth2_credentials.c
  src/core/lib/security/credentials/plugin/plugin_credentials.c
  src/core/lib/security/credentials/ssl/ssl_credentials.c
  src/core/lib/security/transport/client_auth_filter.c
  src/core/lib/security/transport/lb_targets_info.c
  src/core/lib/security/transport/secure_endpoint.c
  src/core/lib/security/transport/security_connector.c
  src/core/lib/security/transport/security_handshaker.c
  src/core/lib/security/transport/server_auth_filter.c
  src/core/lib/security/transport/tsi_error.c
  src/core/lib/security/util/json_util.c
  src/core/lib/surface/init_secure.c
  src/core/tsi/fake_transport_security.c
  src/core/tsi/gts_transport_security.c
  src/core/tsi/ssl_transport_security.c
  src/core/tsi/transport_security.c
  src/core/tsi/transport_security_adapter.c
  src/core/ext/transport/chttp2/server/chttp2_server.c
  src/core/ext/transport/chttp2/client/secure/secure_channel_create.c
  src/core/ext/filters/client_channel/channel_connectivity.c
  src/core/ext/filters/client_channel/client_channel.c
  src/core/ext/filters/client_channel/client_channel_factory.c
  src/core/ext/filters/client_channel/client_channel_plugin.c
  src/core/ext/filters/client_channel/connector.c
  src/core/ext/filters/client_channel/http_connect_handshaker.c
  src/core/ext/filters/client_channel/http_proxy.c
  src/core/ext/filters/client_channel/lb_policy.c
  src/core/ext/filters/client_channel/lb_policy_factory.c
  src/core/ext/filters/client_channel/lb_policy_registry.c
  src/core/ext/filters/client_channel/parse_address.c
  src/core/ext/filters/client_channel/proxy_mapper.c
  src/core/ext/filters/client_channel/proxy_mapper_registry.c
  src/core/ext/filters/client_channel/resolver.c
  src/core/ext/filters/client_channel/resolver_factory.c
  src/core/ext/filters/client_channel/resolver_registry.c
  src/core/ext/filters/client_channel/retry_throttle.c
  src/core/ext/filters/client_channel/subchannel.c
  src/core/ext/filters/client_channel/subchannel_index.c
  src/core/ext/filters/client_channel/uri_parser.c
  src/core/ext/filters/deadline/deadline_filter.c
  src/core/ext/transport/chttp2/client/chttp2_connector.c
  src/core/ext/transport/chttp2/server/insecure/server_chttp2.c
  src/core/ext/transport/chttp2/server/insecure/server_chttp2_posix.c
  src/core/ext/transport/chttp2/client/insecure/channel_create.c
  src/core/ext/transport/chttp2/client/insecure/channel_create_posix.c
  src/core/ext/transport/inproc/inproc_plugin.c
  src/core/ext/transport/inproc/inproc_transport.c
  src/core/ext/filters/client_channel/lb_policy/grpclb/client_load_reporting_filter.c
  src/core/ext/filters/client_channel/lb_policy/grpclb/grpclb.c
  src/core/ext/filters/client_channel/lb_policy/grpclb/grpclb_channel_secure.c
  src/core/ext/filters/client_channel/lb_policy/grpclb/grpclb_client_stats.c
  src/core/ext/filters/client_channel/lb_policy/grpclb/load_balancer_api.c
  src/core/ext/filters/client_channel/lb_policy/grpclb/proto/grpc/lb/v1/load_balancer.pb.c
  third_party/nanopb/pb_common.c
  third_party/nanopb/pb_decode.c
  third_party/nanopb/pb_encode.c
  src/core/ext/filters/client_channel/resolver/fake/fake_resolver.c
  src/core/ext/filters/client_channel/lb_policy/pick_first/pick_first.c
  src/core/ext/filters/client_channel/lb_policy/round_robin/round_robin.c
  src/core/ext/filters/client_channel/resolver/dns/c_ares/dns_resolver_ares.c
  src/core/ext/filters/client_channel/resolver/dns/c_ares/grpc_ares_ev_driver_posix.c
  src/core/ext/filters/client_channel/resolver/dns/c_ares/grpc_ares_wrapper.c
  src/core/ext/filters/client_channel/resolver/dns/c_ares/grpc_ares_wrapper_fallback.c
  src/core/ext/filters/client_channel/resolver/dns/native/dns_resolver.c
  src/core/ext/filters/client_channel/resolver/sockaddr/sockaddr_resolver.c
  src/core/ext/filters/load_reporting/load_reporting.c
  src/core/ext/filters/load_reporting/load_reporting_filter.c
  src/core/ext/census/base_resources.c
  src/core/ext/census/context.c
  src/core/ext/census/gen/census.pb.c
  src/core/ext/census/gen/trace_context.pb.c
  src/core/ext/census/grpc_context.c
  src/core/ext/census/grpc_filter.c
  src/core/ext/census/grpc_plugin.c
  src/core/ext/census/initialize.c
  src/core/ext/census/intrusive_hash_map.c
  src/core/ext/census/mlog.c
  src/core/ext/census/operation.c
  src/core/ext/census/placeholders.c
  src/core/ext/census/resource.c
  src/core/ext/census/trace_context.c
  src/core/ext/census/tracing.c
  src/core/ext/filters/max_age/max_age_filter.c
  src/core/ext/filters/message_size/message_size_filter.c
  src/core/ext/filters/workarounds/workaround_cronet_compression_filter.c
  src/core/ext/filters/workarounds/workaround_utils.c
  src/core/plugin_registry/grpc_plugin_registry.c
)

if(WIN32 AND MSVC)
  set_target_properties(grpc PROPERTIES COMPILE_PDB_NAME "grpc"
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  if (gRPC_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/grpc.pdb
      DESTINATION ${gRPC_INSTALL_LIBDIR} OPTIONAL
    )
  endif()
endif()


target_include_directories(grpc
  PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${ZLIB_INCLUDE_DIR}
  PRIVATE ${BENCHMARK}/include
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(grpc
  ${_gRPC_BASELIB_LIBRARIES}
  ${_gRPC_SSL_LIBRARIES}
  ${_gRPC_ZLIB_LIBRARIES}
  ${_gRPC_CARES_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  gpr
)

foreach(_hdr
  include/grpc/byte_buffer.h
  include/grpc/byte_buffer_reader.h
  include/grpc/compression.h
  include/grpc/grpc.h
  include/grpc/grpc_posix.h
  include/grpc/grpc_security_constants.h
  include/grpc/load_reporting.h
  include/grpc/slice.h
  include/grpc/slice_buffer.h
  include/grpc/status.h
  include/grpc/support/workaround_list.h
  include/grpc/impl/codegen/byte_buffer_reader.h
  include/grpc/impl/codegen/compression_types.h
  include/grpc/impl/codegen/connectivity_state.h
  include/grpc/impl/codegen/exec_ctx_fwd.h
  include/grpc/impl/codegen/grpc_types.h
  include/grpc/impl/codegen/propagation_bits.h
  include/grpc/impl/codegen/slice.h
  include/grpc/impl/codegen/status.h
  include/grpc/impl/codegen/atm.h
  include/grpc/impl/codegen/atm_gcc_atomic.h
  include/grpc/impl/codegen/atm_gcc_sync.h
  include/grpc/impl/codegen/atm_windows.h
  include/grpc/impl/codegen/gpr_slice.h
  include/grpc/impl/codegen/gpr_types.h
  include/grpc/impl/codegen/port_platform.h
  include/grpc/impl/codegen/sync.h
  include/grpc/impl/codegen/sync_generic.h
  include/grpc/impl/codegen/sync_posix.h
  include/grpc/impl/codegen/sync_windows.h
  include/grpc/grpc_security.h
  include/grpc/census.h
)
  string(REPLACE "include/" "" _path ${_hdr})
  get_filename_component(_path ${_path} PATH)
  install(FILES ${_hdr}
    DESTINATION "${gRPC_INSTALL_INCLUDEDIR}/${_path}"
  )
endforeach()


if (gRPC_INSTALL)
  install(TARGETS grpc EXPORT gRPCTargets
    RUNTIME DESTINATION ${gRPC_INSTALL_BINDIR}
    LIBRARY DESTINATION ${gRPC_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${gRPC_INSTALL_LIBDIR}
  )
endif()


add_library(grpc_cronet
  src/core/lib/surface/init.c
  src/core/lib/channel/channel_args.c
  src/core/lib/channel/channel_stack.c
  src/core/lib/channel/channel_stack_builder.c
  src/core/lib/channel/connected_channel.c
  src/core/lib/channel/handshaker.c
  src/core/lib/channel/handshaker_factory.c
  src/core/lib/channel/handshaker_registry.c
  src/core/lib/compression/compression.c
  src/core/lib/compression/message_compress.c
  src/core/lib/compression/stream_compression.c
  src/core/lib/http/format_request.c
  src/core/lib/http/httpcli.c
  src/core/lib/http/parser.c
  src/core/lib/iomgr/closure.c
  src/core/lib/iomgr/combiner.c
  src/core/lib/iomgr/endpoint.c
  src/core/lib/iomgr/endpoint_pair_posix.c
  src/core/lib/iomgr/endpoint_pair_uv.c
  src/core/lib/iomgr/endpoint_pair_windows.c
  src/core/lib/iomgr/error.c
  src/core/lib/iomgr/ev_epoll1_linux.c
  src/core/lib/iomgr/ev_epoll_limited_pollers_linux.c
  src/core/lib/iomgr/ev_epoll_thread_pool_linux.c
  src/core/lib/iomgr/ev_epollex_linux.c
  src/core/lib/iomgr/ev_epollsig_linux.c
  src/core/lib/iomgr/ev_poll_posix.c
  src/core/lib/iomgr/ev_posix.c
  src/core/lib/iomgr/ev_windows.c
  src/core/lib/iomgr/exec_ctx.c
  src/core/lib/iomgr/executor.c
  src/core/lib/iomgr/iocp_windows.c
  src/core/lib/iomgr/iomgr.c
  src/core/lib/iomgr/iomgr_posix.c
  src/core/lib/iomgr/iomgr_uv.c
  src/core/lib/iomgr/iomgr_windows.c
  src/core/lib/iomgr/is_epollexclusive_available.c
  src/core/lib/iomgr/load_file.c
  src/core/lib/iomgr/lockfree_event.c
  src/core/lib/iomgr/network_status_tracker.c
  src/core/lib/iomgr/polling_entity.c
  src/core/lib/iomgr/pollset_set_uv.c
  src/core/lib/iomgr/pollset_set_windows.c
  src/core/lib/iomgr/pollset_uv.c
  src/core/lib/iomgr/pollset_windows.c
  src/core/lib/iomgr/resolve_address_posix.c
  src/core/lib/iomgr/resolve_address_uv.c
  src/core/lib/iomgr/resolve_address_windows.c
  src/core/lib/iomgr/resource_quota.c
  src/core/lib/iomgr/sockaddr_utils.c
  src/core/lib/iomgr/socket_factory_posix.c
  src/core/lib/iomgr/socket_mutator.c
  src/core/lib/iomgr/socket_utils_common_posix.c
  src/core/lib/iomgr/socket_utils_linux.c
  src/core/lib/iomgr/socket_utils_posix.c
  src/core/lib/iomgr/socket_utils_uv.c
  src/core/lib/iomgr/socket_utils_windows.c
  src/core/lib/iomgr/socket_windows.c
  src/core/lib/iomgr/tcp_client_posix.c
  src/core/lib/iomgr/tcp_client_uv.c
  src/core/lib/iomgr/tcp_client_windows.c
  src/core/lib/iomgr/tcp_posix.c
  src/core/lib/iomgr/tcp_server_posix.c
  src/core/lib/iomgr/tcp_server_utils_posix_common.c
  src/core/lib/iomgr/tcp_server_utils_posix_ifaddrs.c
  src/core/lib/iomgr/tcp_server_utils_posix_noifaddrs.c
  src/core/lib/iomgr/tcp_server_uv.c
  src/core/lib/iomgr/tcp_server_windows.c
  src/core/lib/iomgr/tcp_uv.c
  src/core/lib/iomgr/tcp_windows.c
  src/core/lib/iomgr/time_averaged_stats.c
  src/core/lib/iomgr/timer_generic.c
  src/core/lib/iomgr/timer_heap.c
  src/core/lib/iomgr/timer_manager.c
  src/core/lib/iomgr/timer_uv.c
  src/core/lib/iomgr/udp_server.c
  src/core/lib/iomgr/unix_sockets_posix.c
  src/core/lib/iomgr/unix_sockets_posix_noop.c
  src/core/lib/iomgr/wakeup_fd_cv.c
  src/core/lib/iomgr/wakeup_fd_eventfd.c
  src/core/lib/iomgr/wakeup_fd_nospecial.c
  src/core/lib/iomgr/wakeup_fd_pipe.c
  src/core/lib/iomgr/wakeup_fd_posix.c
  src/core/lib/json/json.c
  src/core/lib/json/json_reader.c
  src/core/lib/json/json_string.c
  src/core/lib/json/json_writer.c
  src/core/lib/slice/b64.c
  src/core/lib/slice/percent_encoding.c
  src/core/lib/slice/slice.c
  src/core/lib/slice/slice_buffer.c
  src/core/lib/slice/slice_hash_table.c
  src/core/lib/slice/slice_intern.c
  src/core/lib/slice/slice_string_helpers.c
  src/core/lib/surface/alarm.c
  src/core/lib/surface/api_trace.c
  src/core/lib/surface/byte_buffer.c
  src/core/lib/surface/byte_buffer_reader.c
  src/core/lib/surface/call.c
  src/core/lib/surface/call_details.c
  src/core/lib/surface/call_log_batch.c
  src/core/lib/surface/channel.c
  src/core/lib/surface/channel_init.c
  src/core/lib/surface/channel_ping.c
  src/core/lib/surface/channel_stack_type.c
  src/core/lib/surface/completion_queue.c
  src/core/lib/surface/completion_queue_factory.c
  src/core/lib/surface/event_string.c
  src/core/lib/surface/lame_client.cc
  src/core/lib/surface/metadata_array.c
  src/core/lib/surface/server.c
  src/core/lib/surface/validate_metadata.c
  src/core/lib/surface/version.c
  src/core/lib/transport/bdp_estimator.c
  src/core/lib/transport/byte_stream.c
  src/core/lib/transport/connectivity_state.c
  src/core/lib/transport/error_utils.c
  src/core/lib/transport/metadata.c
  src/core/lib/transport/metadata_batch.c
  src/core/lib/transport/pid_controller.c
  src/core/lib/transport/service_config.c
  src/core/lib/transport/static_metadata.c
  src/core/lib/transport/status_conversion.c
  src/core/lib/transport/timeout_encoding.c
  src/core/lib/transport/transport.c
  src/core/lib/transport/transport_op_string.c
  src/core/lib/debug/trace.c
  src/core/ext/transport/cronet/client/secure/cronet_channel_create.c
  src/core/ext/transport/cronet/transport/cronet_api_dummy.c
  src/core/ext/transport/cronet/transport/cronet_transport.c
  src/core/ext/transport/chttp2/client/secure/secure_channel_create.c
  src/core/ext/transport/chttp2/transport/bin_decoder.c
  src/core/ext/transport/chttp2/transport/bin_encoder.c
  src/core/ext/transport/chttp2/transport/chttp2_plugin.c
  src/core/ext/transport/chttp2/transport/chttp2_transport.c
  src/core/ext/transport/chttp2/transport/frame_data.c
  src/core/ext/transport/chttp2/transport/frame_goaway.c
  src/core/ext/transport/chttp2/transport/frame_ping.c
  src/core/ext/transport/chttp2/transport/frame_rst_stream.c
  src/core/ext/transport/chttp2/transport/frame_settings.c
  src/core/ext/transport/chttp2/transport/frame_window_update.c
  src/core/ext/transport/chttp2/transport/hpack_encoder.c
  src/core/ext/transport/chttp2/transport/hpack_parser.c
  src/core/ext/transport/chttp2/transport/hpack_table.c
  src/core/ext/transport/chttp2/transport/http2_settings.c
  src/core/ext/transport/chttp2/transport/huffsyms.c
  src/core/ext/transport/chttp2/transport/incoming_metadata.c
  src/core/ext/transport/chttp2/transport/parsing.c
  src/core/ext/transport/chttp2/transport/stream_lists.c
  src/core/ext/transport/chttp2/transport/stream_map.c
  src/core/ext/transport/chttp2/transport/varint.c
  src/core/ext/transport/chttp2/transport/writing.c
  src/core/ext/transport/chttp2/alpn/alpn.c
  src/core/ext/filters/http/client/http_client_filter.c
  src/core/ext/filters/http/http_filters_plugin.c
  src/core/ext/filters/http/message_compress/message_compress_filter.c
  src/core/ext/filters/http/server/http_server_filter.c
  src/core/ext/filters/client_channel/channel_connectivity.c
  src/core/ext/filters/client_channel/client_channel.c
  src/core/ext/filters/client_channel/client_channel_factory.c
  src/core/ext/filters/client_channel/client_channel_plugin.c
  src/core/ext/filters/client_channel/connector.c
  src/core/ext/filters/client_channel/http_connect_handshaker.c
  src/core/ext/filters/client_channel/http_proxy.c
  src/core/ext/filters/client_channel/lb_policy.c
  src/core/ext/filters/client_channel/lb_policy_factory.c
  src/core/ext/filters/client_channel/lb_policy_registry.c
  src/core/ext/filters/client_channel/parse_address.c
  src/core/ext/filters/client_channel/proxy_mapper.c
  src/core/ext/filters/client_channel/proxy_mapper_registry.c
  src/core/ext/filters/client_channel/resolver.c
  src/core/ext/filters/client_channel/resolver_factory.c
  src/core/ext/filters/client_channel/resolver_registry.c
  src/core/ext/filters/client_channel/retry_throttle.c
  src/core/ext/filters/client_channel/subchannel.c
  src/core/ext/filters/client_channel/subchannel_index.c
  src/core/ext/filters/client_channel/uri_parser.c
  src/core/ext/filters/deadline/deadline_filter.c
  src/core/lib/http/httpcli_security_connector.c
  src/core/lib/security/context/security_context.c
  src/core/lib/security/credentials/composite/composite_credentials.c
  src/core/lib/security/credentials/credentials.c
  src/core/lib/security/credentials/credentials_metadata.c
  src/core/lib/security/credentials/fake/fake_credentials.c
  src/core/lib/security/credentials/google_default/credentials_generic.c
  src/core/lib/security/credentials/google_default/google_default_credentials.c
  src/core/lib/security/credentials/iam/iam_credentials.c
  src/core/lib/security/credentials/jwt/json_token.c
  src/core/lib/security/credentials/jwt/jwt_credentials.c
  src/core/lib/security/credentials/jwt/jwt_verifier.c
  src/core/lib/security/credentials/oauth2/oauth2_credentials.c
  src/core/lib/security/credentials/plugin/plugin_credentials.c
  src/core/lib/security/credentials/ssl/ssl_credentials.c
  src/core/lib/security/transport/client_auth_filter.c
  src/core/lib/security/transport/lb_targets_info.c
  src/core/lib/security/transport/secure_endpoint.c
  src/core/lib/security/transport/security_connector.c
  src/core/lib/security/transport/security_handshaker.c
  src/core/lib/security/transport/server_auth_filter.c
  src/core/lib/security/transport/tsi_error.c
  src/core/lib/security/util/json_util.c
  src/core/lib/surface/init_secure.c
  src/core/tsi/fake_transport_security.c
  src/core/tsi/gts_transport_security.c
  src/core/tsi/ssl_transport_security.c
  src/core/tsi/transport_security.c
  src/core/tsi/transport_security_adapter.c
  src/core/ext/transport/chttp2/client/chttp2_connector.c
  src/core/ext/filters/load_reporting/load_reporting.c
  src/core/ext/filters/load_reporting/load_reporting_filter.c
  src/core/plugin_registry/grpc_cronet_plugin_registry.c
)

if(WIN32 AND MSVC)
  set_target_properties(grpc_cronet PROPERTIES COMPILE_PDB_NAME "grpc_cronet"
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  if (gRPC_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/grpc_cronet.pdb
      DESTINATION ${gRPC_INSTALL_LIBDIR} OPTIONAL
    )
  endif()
endif()


target_include_directories(grpc_cronet
  PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${ZLIB_INCLUDE_DIR}
  PRIVATE ${BENCHMARK}/include
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(grpc_cronet
  ${_gRPC_BASELIB_LIBRARIES}
  ${_gRPC_SSL_LIBRARIES}
  ${_gRPC_ZLIB_LIBRARIES}
  ${_gRPC_CARES_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  gpr
)

foreach(_hdr
  include/grpc/byte_buffer.h
  include/grpc/byte_buffer_reader.h
  include/grpc/compression.h
  include/grpc/grpc.h
  include/grpc/grpc_posix.h
  include/grpc/grpc_security_constants.h
  include/grpc/load_reporting.h
  include/grpc/slice.h
  include/grpc/slice_buffer.h
  include/grpc/status.h
  include/grpc/support/workaround_list.h
  include/grpc/impl/codegen/byte_buffer_reader.h
  include/grpc/impl/codegen/compression_types.h
  include/grpc/impl/codegen/connectivity_state.h
  include/grpc/impl/codegen/exec_ctx_fwd.h
  include/grpc/impl/codegen/grpc_types.h
  include/grpc/impl/codegen/propagation_bits.h
  include/grpc/impl/codegen/slice.h
  include/grpc/impl/codegen/status.h
  include/grpc/impl/codegen/atm.h
  include/grpc/impl/codegen/atm_gcc_atomic.h
  include/grpc/impl/codegen/atm_gcc_sync.h
  include/grpc/impl/codegen/atm_windows.h
  include/grpc/impl/codegen/gpr_slice.h
  include/grpc/impl/codegen/gpr_types.h
  include/grpc/impl/codegen/port_platform.h
  include/grpc/impl/codegen/sync.h
  include/grpc/impl/codegen/sync_generic.h
  include/grpc/impl/codegen/sync_posix.h
  include/grpc/impl/codegen/sync_windows.h
  include/grpc/grpc_cronet.h
  include/grpc/grpc_security.h
)
  string(REPLACE "include/" "" _path ${_hdr})
  get_filename_component(_path ${_path} PATH)
  install(FILES ${_hdr}
    DESTINATION "${gRPC_INSTALL_INCLUDEDIR}/${_path}"
  )
endforeach()


if (gRPC_INSTALL)
  install(TARGETS grpc_cronet EXPORT gRPCTargets
    RUNTIME DESTINATION ${gRPC_INSTALL_BINDIR}
    LIBRARY DESTINATION ${gRPC_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${gRPC_INSTALL_LIBDIR}
  )
endif()

if (gRPC_BUILD_TESTS)

add_library(grpc_test_util
  test/core/end2end/data/client_certs.c
  test/core/end2end/data/server1_cert.c
  test/core/end2end/data/server1_key.c
  test/core/end2end/data/test_root_cert.c
  test/core/security/oauth2_utils.c
  src/core/ext/filters/client_channel/resolver/fake/fake_resolver.c
  test/core/end2end/cq_verifier.c
  test/core/end2end/fixtures/http_proxy_fixture.c
  test/core/end2end/fixtures/proxy.c
  test/core/iomgr/endpoint_tests.c
  test/core/util/debugger_macros.c
  test/core/util/grpc_profiler.c
  test/core/util/memory_counters.c
  test/core/util/mock_endpoint.c
  test/core/util/parse_hexstring.c
  test/core/util/passthru_endpoint.c
  test/core/util/port.c
  test/core/util/port_server_client.c
  test/core/util/slice_splitter.c
  test/core/util/trickle_endpoint.c
  src/core/lib/channel/channel_args.c
  src/core/lib/channel/channel_stack.c
  src/core/lib/channel/channel_stack_builder.c
  src/core/lib/channel/connected_channel.c
  src/core/lib/channel/handshaker.c
  src/core/lib/channel/handshaker_factory.c
  src/core/lib/channel/handshaker_registry.c
  src/core/lib/compression/compression.c
  src/core/lib/compression/message_compress.c
  src/core/lib/compression/stream_compression.c
  src/core/lib/http/format_request.c
  src/core/lib/http/httpcli.c
  src/core/lib/http/parser.c
  src/core/lib/iomgr/closure.c
  src/core/lib/iomgr/combiner.c
  src/core/lib/iomgr/endpoint.c
  src/core/lib/iomgr/endpoint_pair_posix.c
  src/core/lib/iomgr/endpoint_pair_uv.c
  src/core/lib/iomgr/endpoint_pair_windows.c
  src/core/lib/iomgr/error.c
  src/core/lib/iomgr/ev_epoll1_linux.c
  src/core/lib/iomgr/ev_epoll_limited_pollers_linux.c
  src/core/lib/iomgr/ev_epoll_thread_pool_linux.c
  src/core/lib/iomgr/ev_epollex_linux.c
  src/core/lib/iomgr/ev_epollsig_linux.c
  src/core/lib/iomgr/ev_poll_posix.c
  src/core/lib/iomgr/ev_posix.c
  src/core/lib/iomgr/ev_windows.c
  src/core/lib/iomgr/exec_ctx.c
  src/core/lib/iomgr/executor.c
  src/core/lib/iomgr/iocp_windows.c
  src/core/lib/iomgr/iomgr.c
  src/core/lib/iomgr/iomgr_posix.c
  src/core/lib/iomgr/iomgr_uv.c
  src/core/lib/iomgr/iomgr_windows.c
  src/core/lib/iomgr/is_epollexclusive_available.c
  src/core/lib/iomgr/load_file.c
  src/core/lib/iomgr/lockfree_event.c
  src/core/lib/iomgr/network_status_tracker.c
  src/core/lib/iomgr/polling_entity.c
  src/core/lib/iomgr/pollset_set_uv.c
  src/core/lib/iomgr/pollset_set_windows.c
  src/core/lib/iomgr/pollset_uv.c
  src/core/lib/iomgr/pollset_windows.c
  src/core/lib/iomgr/resolve_address_posix.c
  src/core/lib/iomgr/resolve_address_uv.c
  src/core/lib/iomgr/resolve_address_windows.c
  src/core/lib/iomgr/resource_quota.c
  src/core/lib/iomgr/sockaddr_utils.c
  src/core/lib/iomgr/socket_factory_posix.c
  src/core/lib/iomgr/socket_mutator.c
  src/core/lib/iomgr/socket_utils_common_posix.c
  src/core/lib/iomgr/socket_utils_linux.c
  src/core/lib/iomgr/socket_utils_posix.c
  src/core/lib/iomgr/socket_utils_uv.c
  src/core/lib/iomgr/socket_utils_windows.c
  src/core/lib/iomgr/socket_windows.c
  src/core/lib/iomgr/tcp_client_posix.c
  src/core/lib/iomgr/tcp_client_uv.c
  src/core/lib/iomgr/tcp_client_windows.c
  src/core/lib/iomgr/tcp_posix.c
  src/core/lib/iomgr/tcp_server_posix.c
  src/core/lib/iomgr/tcp_server_utils_posix_common.c
  src/core/lib/iomgr/tcp_server_utils_posix_ifaddrs.c
  src/core/lib/iomgr/tcp_server_utils_posix_noifaddrs.c
  src/core/lib/iomgr/tcp_server_uv.c
  src/core/lib/iomgr/tcp_server_windows.c
  src/core/lib/iomgr/tcp_uv.c
  src/core/lib/iomgr/tcp_windows.c
  src/core/lib/iomgr/time_averaged_stats.c
  src/core/lib/iomgr/timer_generic.c
  src/core/lib/iomgr/timer_heap.c
  src/core/lib/iomgr/timer_manager.c
  src/core/lib/iomgr/timer_uv.c
  src/core/lib/iomgr/udp_server.c
  src/core/lib/iomgr/unix_sockets_posix.c
  src/core/lib/iomgr/unix_sockets_posix_noop.c
  src/core/lib/iomgr/wakeup_fd_cv.c
  src/core/lib/iomgr/wakeup_fd_eventfd.c
  src/core/lib/iomgr/wakeup_fd_nospecial.c
  src/core/lib/iomgr/wakeup_fd_pipe.c
  src/core/lib/iomgr/wakeup_fd_posix.c
  src/core/lib/json/json.c
  src/core/lib/json/json_reader.c
  src/core/lib/json/json_string.c
  src/core/lib/json/json_writer.c
  src/core/lib/slice/b64.c
  src/core/lib/slice/percent_encoding.c
  src/core/lib/slice/slice.c
  src/core/lib/slice/slice_buffer.c
  src/core/lib/slice/slice_hash_table.c
  src/core/lib/slice/slice_intern.c
  src/core/lib/slice/slice_string_helpers.c
  src/core/lib/surface/alarm.c
  src/core/lib/surface/api_trace.c
  src/core/lib/surface/byte_buffer.c
  src/core/lib/surface/byte_buffer_reader.c
  src/core/lib/surface/call.c
  src/core/lib/surface/call_details.c
  src/core/lib/surface/call_log_batch.c
  src/core/lib/surface/channel.c
  src/core/lib/surface/channel_init.c
  src/core/lib/surface/channel_ping.c
  src/core/lib/surface/channel_stack_type.c
  src/core/lib/surface/completion_queue.c
  src/core/lib/surface/completion_queue_factory.c
  src/core/lib/surface/event_string.c
  src/core/lib/surface/lame_client.cc
  src/core/lib/surface/metadata_array.c
  src/core/lib/surface/server.c
  src/core/lib/surface/validate_metadata.c
  src/core/lib/surface/version.c
  src/core/lib/transport/bdp_estimator.c
  src/core/lib/transport/byte_stream.c
  src/core/lib/transport/connectivity_state.c
  src/core/lib/transport/error_utils.c
  src/core/lib/transport/metadata.c
  src/core/lib/transport/metadata_batch.c
  src/core/lib/transport/pid_controller.c
  src/core/lib/transport/service_config.c
  src/core/lib/transport/static_metadata.c
  src/core/lib/transport/status_conversion.c
  src/core/lib/transport/timeout_encoding.c
  src/core/lib/transport/transport.c
  src/core/lib/transport/transport_op_string.c
  src/core/lib/debug/trace.c
)

if(WIN32 AND MSVC)
  set_target_properties(grpc_test_util PROPERTIES COMPILE_PDB_NAME "grpc_test_util"
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  if (gRPC_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/grpc_test_util.pdb
      DESTINATION ${gRPC_INSTALL_LIBDIR} OPTIONAL
    )
  endif()
endif()


target_include_directories(grpc_test_util
  PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${ZLIB_INCLUDE_DIR}
  PRIVATE ${BENCHMARK}/include
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(grpc_test_util
  ${_gRPC_ALLTARGETS_LIBRARIES}
  gpr_test_util
  gpr
  grpc
)

foreach(_hdr
  include/grpc/byte_buffer.h
  include/grpc/byte_buffer_reader.h
  include/grpc/compression.h
  include/grpc/grpc.h
  include/grpc/grpc_posix.h
  include/grpc/grpc_security_constants.h
  include/grpc/load_reporting.h
  include/grpc/slice.h
  include/grpc/slice_buffer.h
  include/grpc/status.h
  include/grpc/support/workaround_list.h
  include/grpc/impl/codegen/byte_buffer_reader.h
  include/grpc/impl/codegen/compression_types.h
  include/grpc/impl/codegen/connectivity_state.h
  include/grpc/impl/codegen/exec_ctx_fwd.h
  include/grpc/impl/codegen/grpc_types.h
  include/grpc/impl/codegen/propagation_bits.h
  include/grpc/impl/codegen/slice.h
  include/grpc/impl/codegen/status.h
  include/grpc/impl/codegen/atm.h
  include/grpc/impl/codegen/atm_gcc_atomic.h
  include/grpc/impl/codegen/atm_gcc_sync.h
  include/grpc/impl/codegen/atm_windows.h
  include/grpc/impl/codegen/gpr_slice.h
  include/grpc/impl/codegen/gpr_types.h
  include/grpc/impl/codegen/port_platform.h
  include/grpc/impl/codegen/sync.h
  include/grpc/impl/codegen/sync_generic.h
  include/grpc/impl/codegen/sync_posix.h
  include/grpc/impl/codegen/sync_windows.h
)
  string(REPLACE "include/" "" _path ${_hdr})
  get_filename_component(_path ${_path} PATH)
  install(FILES ${_hdr}
    DESTINATION "${gRPC_INSTALL_INCLUDEDIR}/${_path}"
  )
endforeach()

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_library(grpc_test_util_unsecure
  src/core/ext/filters/client_channel/resolver/fake/fake_resolver.c
  test/core/end2end/cq_verifier.c
  test/core/end2end/fixtures/http_proxy_fixture.c
  test/core/end2end/fixtures/proxy.c
  test/core/iomgr/endpoint_tests.c
  test/core/util/debugger_macros.c
  test/core/util/grpc_profiler.c
  test/core/util/memory_counters.c
  test/core/util/mock_endpoint.c
  test/core/util/parse_hexstring.c
  test/core/util/passthru_endpoint.c
  test/core/util/port.c
  test/core/util/port_server_client.c
  test/core/util/slice_splitter.c
  test/core/util/trickle_endpoint.c
)

if(WIN32 AND MSVC)
  set_target_properties(grpc_test_util_unsecure PROPERTIES COMPILE_PDB_NAME "grpc_test_util_unsecure"
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  if (gRPC_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/grpc_test_util_unsecure.pdb
      DESTINATION ${gRPC_INSTALL_LIBDIR} OPTIONAL
    )
  endif()
endif()


target_include_directories(grpc_test_util_unsecure
  PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${ZLIB_INCLUDE_DIR}
  PRIVATE ${BENCHMARK}/include
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(grpc_test_util_unsecure
  ${_gRPC_ALLTARGETS_LIBRARIES}
  gpr
  gpr_test_util
  grpc_unsecure
  grpc
)


endif (gRPC_BUILD_TESTS)

add_library(grpc_unsecure
  src/core/lib/surface/init.c
  src/core/lib/surface/init_unsecure.c
  src/core/lib/channel/channel_args.c
  src/core/lib/channel/channel_stack.c
  src/core/lib/channel/channel_stack_builder.c
  src/core/lib/channel/connected_channel.c
  src/core/lib/channel/handshaker.c
  src/core/lib/channel/handshaker_factory.c
  src/core/lib/channel/handshaker_registry.c
  src/core/lib/compression/compression.c
  src/core/lib/compression/message_compress.c
  src/core/lib/compression/stream_compression.c
  src/core/lib/http/format_request.c
  src/core/lib/http/httpcli.c
  src/core/lib/http/parser.c
  src/core/lib/iomgr/closure.c
  src/core/lib/iomgr/combiner.c
  src/core/lib/iomgr/endpoint.c
  src/core/lib/iomgr/endpoint_pair_posix.c
  src/core/lib/iomgr/endpoint_pair_uv.c
  src/core/lib/iomgr/endpoint_pair_windows.c
  src/core/lib/iomgr/error.c
  src/core/lib/iomgr/ev_epoll1_linux.c
  src/core/lib/iomgr/ev_epoll_limited_pollers_linux.c
  src/core/lib/iomgr/ev_epoll_thread_pool_linux.c
  src/core/lib/iomgr/ev_epollex_linux.c
  src/core/lib/iomgr/ev_epollsig_linux.c
  src/core/lib/iomgr/ev_poll_posix.c
  src/core/lib/iomgr/ev_posix.c
  src/core/lib/iomgr/ev_windows.c
  src/core/lib/iomgr/exec_ctx.c
  src/core/lib/iomgr/executor.c
  src/core/lib/iomgr/iocp_windows.c
  src/core/lib/iomgr/iomgr.c
  src/core/lib/iomgr/iomgr_posix.c
  src/core/lib/iomgr/iomgr_uv.c
  src/core/lib/iomgr/iomgr_windows.c
  src/core/lib/iomgr/is_epollexclusive_available.c
  src/core/lib/iomgr/load_file.c
  src/core/lib/iomgr/lockfree_event.c
  src/core/lib/iomgr/network_status_tracker.c
  src/core/lib/iomgr/polling_entity.c
  src/core/lib/iomgr/pollset_set_uv.c
  src/core/lib/iomgr/pollset_set_windows.c
  src/core/lib/iomgr/pollset_uv.c
  src/core/lib/iomgr/pollset_windows.c
  src/core/lib/iomgr/resolve_address_posix.c
  src/core/lib/iomgr/resolve_address_uv.c
  src/core/lib/iomgr/resolve_address_windows.c
  src/core/lib/iomgr/resource_quota.c
  src/core/lib/iomgr/sockaddr_utils.c
  src/core/lib/iomgr/socket_factory_posix.c
  src/core/lib/iomgr/socket_mutator.c
  src/core/lib/iomgr/socket_utils_common_posix.c
  src/core/lib/iomgr/socket_utils_linux.c
  src/core/lib/iomgr/socket_utils_posix.c
  src/core/lib/iomgr/socket_utils_uv.c
  src/core/lib/iomgr/socket_utils_windows.c
  src/core/lib/iomgr/socket_windows.c
  src/core/lib/iomgr/tcp_client_posix.c
  src/core/lib/iomgr/tcp_client_uv.c
  src/core/lib/iomgr/tcp_client_windows.c
  src/core/lib/iomgr/tcp_posix.c
  src/core/lib/iomgr/tcp_server_posix.c
  src/core/lib/iomgr/tcp_server_utils_posix_common.c
  src/core/lib/iomgr/tcp_server_utils_posix_ifaddrs.c
  src/core/lib/iomgr/tcp_server_utils_posix_noifaddrs.c
  src/core/lib/iomgr/tcp_server_uv.c
  src/core/lib/iomgr/tcp_server_windows.c
  src/core/lib/iomgr/tcp_uv.c
  src/core/lib/iomgr/tcp_windows.c
  src/core/lib/iomgr/time_averaged_stats.c
  src/core/lib/iomgr/timer_generic.c
  src/core/lib/iomgr/timer_heap.c
  src/core/lib/iomgr/timer_manager.c
  src/core/lib/iomgr/timer_uv.c
  src/core/lib/iomgr/udp_server.c
  src/core/lib/iomgr/unix_sockets_posix.c
  src/core/lib/iomgr/unix_sockets_posix_noop.c
  src/core/lib/iomgr/wakeup_fd_cv.c
  src/core/lib/iomgr/wakeup_fd_eventfd.c
  src/core/lib/iomgr/wakeup_fd_nospecial.c
  src/core/lib/iomgr/wakeup_fd_pipe.c
  src/core/lib/iomgr/wakeup_fd_posix.c
  src/core/lib/json/json.c
  src/core/lib/json/json_reader.c
  src/core/lib/json/json_string.c
  src/core/lib/json/json_writer.c
  src/core/lib/slice/b64.c
  src/core/lib/slice/percent_encoding.c
  src/core/lib/slice/slice.c
  src/core/lib/slice/slice_buffer.c
  src/core/lib/slice/slice_hash_table.c
  src/core/lib/slice/slice_intern.c
  src/core/lib/slice/slice_string_helpers.c
  src/core/lib/surface/alarm.c
  src/core/lib/surface/api_trace.c
  src/core/lib/surface/byte_buffer.c
  src/core/lib/surface/byte_buffer_reader.c
  src/core/lib/surface/call.c
  src/core/lib/surface/call_details.c
  src/core/lib/surface/call_log_batch.c
  src/core/lib/surface/channel.c
  src/core/lib/surface/channel_init.c
  src/core/lib/surface/channel_ping.c
  src/core/lib/surface/channel_stack_type.c
  src/core/lib/surface/completion_queue.c
  src/core/lib/surface/completion_queue_factory.c
  src/core/lib/surface/event_string.c
  src/core/lib/surface/lame_client.cc
  src/core/lib/surface/metadata_array.c
  src/core/lib/surface/server.c
  src/core/lib/surface/validate_metadata.c
  src/core/lib/surface/version.c
  src/core/lib/transport/bdp_estimator.c
  src/core/lib/transport/byte_stream.c
  src/core/lib/transport/connectivity_state.c
  src/core/lib/transport/error_utils.c
  src/core/lib/transport/metadata.c
  src/core/lib/transport/metadata_batch.c
  src/core/lib/transport/pid_controller.c
  src/core/lib/transport/service_config.c
  src/core/lib/transport/static_metadata.c
  src/core/lib/transport/status_conversion.c
  src/core/lib/transport/timeout_encoding.c
  src/core/lib/transport/transport.c
  src/core/lib/transport/transport_op_string.c
  src/core/lib/debug/trace.c
  src/core/ext/transport/chttp2/server/insecure/server_chttp2.c
  src/core/ext/transport/chttp2/server/insecure/server_chttp2_posix.c
  src/core/ext/transport/chttp2/transport/bin_decoder.c
  src/core/ext/transport/chttp2/transport/bin_encoder.c
  src/core/ext/transport/chttp2/transport/chttp2_plugin.c
  src/core/ext/transport/chttp2/transport/chttp2_transport.c
  src/core/ext/transport/chttp2/transport/frame_data.c
  src/core/ext/transport/chttp2/transport/frame_goaway.c
  src/core/ext/transport/chttp2/transport/frame_ping.c
  src/core/ext/transport/chttp2/transport/frame_rst_stream.c
  src/core/ext/transport/chttp2/transport/frame_settings.c
  src/core/ext/transport/chttp2/transport/frame_window_update.c
  src/core/ext/transport/chttp2/transport/hpack_encoder.c
  src/core/ext/transport/chttp2/transport/hpack_parser.c
  src/core/ext/transport/chttp2/transport/hpack_table.c
  src/core/ext/transport/chttp2/transport/http2_settings.c
  src/core/ext/transport/chttp2/transport/huffsyms.c
  src/core/ext/transport/chttp2/transport/incoming_metadata.c
  src/core/ext/transport/chttp2/transport/parsing.c
  src/core/ext/transport/chttp2/transport/stream_lists.c
  src/core/ext/transport/chttp2/transport/stream_map.c
  src/core/ext/transport/chttp2/transport/varint.c
  src/core/ext/transport/chttp2/transport/writing.c
  src/core/ext/transport/chttp2/alpn/alpn.c
  src/core/ext/filters/http/client/http_client_filter.c
  src/core/ext/filters/http/http_filters_plugin.c
  src/core/ext/filters/http/message_compress/message_compress_filter.c
  src/core/ext/filters/http/server/http_server_filter.c
  src/core/ext/transport/chttp2/server/chttp2_server.c
  src/core/ext/transport/chttp2/client/insecure/channel_create.c
  src/core/ext/transport/chttp2/client/insecure/channel_create_posix.c
  src/core/ext/transport/chttp2/client/chttp2_connector.c
  src/core/ext/filters/client_channel/channel_connectivity.c
  src/core/ext/filters/client_channel/client_channel.c
  src/core/ext/filters/client_channel/client_channel_factory.c
  src/core/ext/filters/client_channel/client_channel_plugin.c
  src/core/ext/filters/client_channel/connector.c
  src/core/ext/filters/client_channel/http_connect_handshaker.c
  src/core/ext/filters/client_channel/http_proxy.c
  src/core/ext/filters/client_channel/lb_policy.c
  src/core/ext/filters/client_channel/lb_policy_factory.c
  src/core/ext/filters/client_channel/lb_policy_registry.c
  src/core/ext/filters/client_channel/parse_address.c
  src/core/ext/filters/client_channel/proxy_mapper.c
  src/core/ext/filters/client_channel/proxy_mapper_registry.c
  src/core/ext/filters/client_channel/resolver.c
  src/core/ext/filters/client_channel/resolver_factory.c
  src/core/ext/filters/client_channel/resolver_registry.c
  src/core/ext/filters/client_channel/retry_throttle.c
  src/core/ext/filters/client_channel/subchannel.c
  src/core/ext/filters/client_channel/subchannel_index.c
  src/core/ext/filters/client_channel/uri_parser.c
  src/core/ext/filters/deadline/deadline_filter.c
  src/core/ext/transport/inproc/inproc_plugin.c
  src/core/ext/transport/inproc/inproc_transport.c
  src/core/ext/filters/client_channel/resolver/dns/c_ares/dns_resolver_ares.c
  src/core/ext/filters/client_channel/resolver/dns/c_ares/grpc_ares_ev_driver_posix.c
  src/core/ext/filters/client_channel/resolver/dns/c_ares/grpc_ares_wrapper.c
  src/core/ext/filters/client_channel/resolver/dns/c_ares/grpc_ares_wrapper_fallback.c
  src/core/ext/filters/client_channel/resolver/dns/native/dns_resolver.c
  src/core/ext/filters/client_channel/resolver/sockaddr/sockaddr_resolver.c
  src/core/ext/filters/client_channel/resolver/fake/fake_resolver.c
  src/core/ext/filters/load_reporting/load_reporting.c
  src/core/ext/filters/load_reporting/load_reporting_filter.c
  src/core/ext/filters/client_channel/lb_policy/grpclb/client_load_reporting_filter.c
  src/core/ext/filters/client_channel/lb_policy/grpclb/grpclb.c
  src/core/ext/filters/client_channel/lb_policy/grpclb/grpclb_channel.c
  src/core/ext/filters/client_channel/lb_policy/grpclb/grpclb_client_stats.c
  src/core/ext/filters/client_channel/lb_policy/grpclb/load_balancer_api.c
  src/core/ext/filters/client_channel/lb_policy/grpclb/proto/grpc/lb/v1/load_balancer.pb.c
  third_party/nanopb/pb_common.c
  third_party/nanopb/pb_decode.c
  third_party/nanopb/pb_encode.c
  src/core/ext/filters/client_channel/lb_policy/pick_first/pick_first.c
  src/core/ext/filters/client_channel/lb_policy/round_robin/round_robin.c
  src/core/ext/census/base_resources.c
  src/core/ext/census/context.c
  src/core/ext/census/gen/census.pb.c
  src/core/ext/census/gen/trace_context.pb.c
  src/core/ext/census/grpc_context.c
  src/core/ext/census/grpc_filter.c
  src/core/ext/census/grpc_plugin.c
  src/core/ext/census/initialize.c
  src/core/ext/census/intrusive_hash_map.c
  src/core/ext/census/mlog.c
  src/core/ext/census/operation.c
  src/core/ext/census/placeholders.c
  src/core/ext/census/resource.c
  src/core/ext/census/trace_context.c
  src/core/ext/census/tracing.c
  src/core/ext/filters/max_age/max_age_filter.c
  src/core/ext/filters/message_size/message_size_filter.c
  src/core/ext/filters/workarounds/workaround_cronet_compression_filter.c
  src/core/ext/filters/workarounds/workaround_utils.c
  src/core/plugin_registry/grpc_unsecure_plugin_registry.c
)

if(WIN32 AND MSVC)
  set_target_properties(grpc_unsecure PROPERTIES COMPILE_PDB_NAME "grpc_unsecure"
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  if (gRPC_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/grpc_unsecure.pdb
      DESTINATION ${gRPC_INSTALL_LIBDIR} OPTIONAL
    )
  endif()
endif()


target_include_directories(grpc_unsecure
  PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${ZLIB_INCLUDE_DIR}
  PRIVATE ${BENCHMARK}/include
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(grpc_unsecure
  ${_gRPC_BASELIB_LIBRARIES}
  ${_gRPC_ZLIB_LIBRARIES}
  ${_gRPC_CARES_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  gpr
)

foreach(_hdr
  include/grpc/byte_buffer.h
  include/grpc/byte_buffer_reader.h
  include/grpc/compression.h
  include/grpc/grpc.h
  include/grpc/grpc_posix.h
  include/grpc/grpc_security_constants.h
  include/grpc/load_reporting.h
  include/grpc/slice.h
  include/grpc/slice_buffer.h
  include/grpc/status.h
  include/grpc/support/workaround_list.h
  include/grpc/impl/codegen/byte_buffer_reader.h
  include/grpc/impl/codegen/compression_types.h
  include/grpc/impl/codegen/connectivity_state.h
  include/grpc/impl/codegen/exec_ctx_fwd.h
  include/grpc/impl/codegen/grpc_types.h
  include/grpc/impl/codegen/propagation_bits.h
  include/grpc/impl/codegen/slice.h
  include/grpc/impl/codegen/status.h
  include/grpc/impl/codegen/atm.h
  include/grpc/impl/codegen/atm_gcc_atomic.h
  include/grpc/impl/codegen/atm_gcc_sync.h
  include/grpc/impl/codegen/atm_windows.h
  include/grpc/impl/codegen/gpr_slice.h
  include/grpc/impl/codegen/gpr_types.h
  include/grpc/impl/codegen/port_platform.h
  include/grpc/impl/codegen/sync.h
  include/grpc/impl/codegen/sync_generic.h
  include/grpc/impl/codegen/sync_posix.h
  include/grpc/impl/codegen/sync_windows.h
  include/grpc/census.h
)
  string(REPLACE "include/" "" _path ${_hdr})
  get_filename_component(_path ${_path} PATH)
  install(FILES ${_hdr}
    DESTINATION "${gRPC_INSTALL_INCLUDEDIR}/${_path}"
  )
endforeach()


if (gRPC_INSTALL)
  install(TARGETS grpc_unsecure EXPORT gRPCTargets
    RUNTIME DESTINATION ${gRPC_INSTALL_BINDIR}
    LIBRARY DESTINATION ${gRPC_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${gRPC_INSTALL_LIBDIR}
  )
endif()

if (gRPC_BUILD_TESTS)

add_library(reconnect_server
  test/core/util/reconnect_server.c
)

if(WIN32 AND MSVC)
  set_target_properties(reconnect_server PROPERTIES COMPILE_PDB_NAME "reconnect_server"
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  if (gRPC_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/reconnect_server.pdb
      DESTINATION ${gRPC_INSTALL_LIBDIR} OPTIONAL
    )
  endif()
endif()


target_include_directories(reconnect_server
  PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${ZLIB_INCLUDE_DIR}
  PRIVATE ${BENCHMARK}/include
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(reconnect_server
  ${_gRPC_ALLTARGETS_LIBRARIES}
  test_tcp_server
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)


endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_library(test_tcp_server
  test/core/util/test_tcp_server.c
)

if(WIN32 AND MSVC)
  set_target_properties(test_tcp_server PROPERTIES COMPILE_PDB_NAME "test_tcp_server"
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  if (gRPC_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/test_tcp_server.pdb
      DESTINATION ${gRPC_INSTALL_LIBDIR} OPTIONAL
    )
  endif()
endif()


target_include_directories(test_tcp_server
  PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${ZLIB_INCLUDE_DIR}
  PRIVATE ${BENCHMARK}/include
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(test_tcp_server
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)


endif (gRPC_BUILD_TESTS)

add_library(grpc++
  src/cpp/client/insecure_credentials.cc
  src/cpp/client/secure_credentials.cc
  src/cpp/common/auth_property_iterator.cc
  src/cpp/common/secure_auth_context.cc
  src/cpp/common/secure_channel_arguments.cc
  src/cpp/common/secure_create_auth_context.cc
  src/cpp/server/insecure_server_credentials.cc
  src/cpp/server/secure_server_credentials.cc
  src/cpp/client/channel_cc.cc
  src/cpp/client/client_context.cc
  src/cpp/client/create_channel.cc
  src/cpp/client/create_channel_internal.cc
  src/cpp/client/create_channel_posix.cc
  src/cpp/client/credentials_cc.cc
  src/cpp/client/generic_stub.cc
  src/cpp/common/channel_arguments.cc
  src/cpp/common/channel_filter.cc
  src/cpp/common/completion_queue_cc.cc
  src/cpp/common/core_codegen.cc
  src/cpp/common/resource_quota_cc.cc
  src/cpp/common/rpc_method.cc
  src/cpp/common/version_cc.cc
  src/cpp/server/async_generic_service.cc
  src/cpp/server/channel_argument_option.cc
  src/cpp/server/create_default_thread_pool.cc
  src/cpp/server/dynamic_thread_pool.cc
  src/cpp/server/health/default_health_check_service.cc
  src/cpp/server/health/health.pb.c
  src/cpp/server/health/health_check_service.cc
  src/cpp/server/health/health_check_service_server_builder_option.cc
  src/cpp/server/server_builder.cc
  src/cpp/server/server_cc.cc
  src/cpp/server/server_context.cc
  src/cpp/server/server_credentials.cc
  src/cpp/server/server_posix.cc
  src/cpp/thread_manager/thread_manager.cc
  src/cpp/util/byte_buffer_cc.cc
  src/cpp/util/slice_cc.cc
  src/cpp/util/status.cc
  src/cpp/util/string_ref.cc
  src/cpp/util/time_cc.cc
  third_party/nanopb/pb_common.c
  third_party/nanopb/pb_decode.c
  third_party/nanopb/pb_encode.c
  src/cpp/codegen/codegen_init.cc
)

if(WIN32 AND MSVC)
  set_target_properties(grpc++ PROPERTIES COMPILE_PDB_NAME "grpc++"
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  if (gRPC_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/grpc++.pdb
      DESTINATION ${gRPC_INSTALL_LIBDIR} OPTIONAL
    )
  endif()
endif()


target_include_directories(grpc++
  PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${ZLIB_INCLUDE_DIR}
  PRIVATE ${BENCHMARK}/include
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(grpc++
  ${_gRPC_BASELIB_LIBRARIES}
  ${_gRPC_SSL_LIBRARIES}
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc
)

foreach(_hdr
  include/grpc++/alarm.h
  include/grpc++/channel.h
  include/grpc++/client_context.h
  include/grpc++/completion_queue.h
  include/grpc++/create_channel.h
  include/grpc++/create_channel_posix.h
  include/grpc++/ext/health_check_service_server_builder_option.h
  include/grpc++/generic/async_generic_service.h
  include/grpc++/generic/generic_stub.h
  include/grpc++/grpc++.h
  include/grpc++/health_check_service_interface.h
  include/grpc++/impl/call.h
  include/grpc++/impl/channel_argument_option.h
  include/grpc++/impl/client_unary_call.h
  include/grpc++/impl/codegen/core_codegen.h
  include/grpc++/impl/grpc_library.h
  include/grpc++/impl/method_handler_impl.h
  include/grpc++/impl/rpc_method.h
  include/grpc++/impl/rpc_service_method.h
  include/grpc++/impl/serialization_traits.h
  include/grpc++/impl/server_builder_option.h
  include/grpc++/impl/server_builder_plugin.h
  include/grpc++/impl/server_initializer.h
  include/grpc++/impl/service_type.h
  include/grpc++/resource_quota.h
  include/grpc++/security/auth_context.h
  include/grpc++/security/auth_metadata_processor.h
  include/grpc++/security/credentials.h
  include/grpc++/security/server_credentials.h
  include/grpc++/server.h
  include/grpc++/server_builder.h
  include/grpc++/server_context.h
  include/grpc++/server_posix.h
  include/grpc++/support/async_stream.h
  include/grpc++/support/async_unary_call.h
  include/grpc++/support/byte_buffer.h
  include/grpc++/support/channel_arguments.h
  include/grpc++/support/config.h
  include/grpc++/support/slice.h
  include/grpc++/support/status.h
  include/grpc++/support/status_code_enum.h
  include/grpc++/support/string_ref.h
  include/grpc++/support/stub_options.h
  include/grpc++/support/sync_stream.h
  include/grpc++/support/time.h
  include/grpc++/impl/codegen/async_stream.h
  include/grpc++/impl/codegen/async_unary_call.h
  include/grpc++/impl/codegen/call.h
  include/grpc++/impl/codegen/call_hook.h
  include/grpc++/impl/codegen/channel_interface.h
  include/grpc++/impl/codegen/client_context.h
  include/grpc++/impl/codegen/client_unary_call.h
  include/grpc++/impl/codegen/completion_queue.h
  include/grpc++/impl/codegen/completion_queue_tag.h
  include/grpc++/impl/codegen/config.h
  include/grpc++/impl/codegen/core_codegen_interface.h
  include/grpc++/impl/codegen/create_auth_context.h
  include/grpc++/impl/codegen/grpc_library.h
  include/grpc++/impl/codegen/metadata_map.h
  include/grpc++/impl/codegen/method_handler_impl.h
  include/grpc++/impl/codegen/rpc_method.h
  include/grpc++/impl/codegen/rpc_service_method.h
  include/grpc++/impl/codegen/security/auth_context.h
  include/grpc++/impl/codegen/serialization_traits.h
  include/grpc++/impl/codegen/server_context.h
  include/grpc++/impl/codegen/server_interface.h
  include/grpc++/impl/codegen/service_type.h
  include/grpc++/impl/codegen/slice.h
  include/grpc++/impl/codegen/status.h
  include/grpc++/impl/codegen/status_code_enum.h
  include/grpc++/impl/codegen/string_ref.h
  include/grpc++/impl/codegen/stub_options.h
  include/grpc++/impl/codegen/sync_stream.h
  include/grpc++/impl/codegen/time.h
  include/grpc/impl/codegen/byte_buffer_reader.h
  include/grpc/impl/codegen/compression_types.h
  include/grpc/impl/codegen/connectivity_state.h
  include/grpc/impl/codegen/exec_ctx_fwd.h
  include/grpc/impl/codegen/grpc_types.h
  include/grpc/impl/codegen/propagation_bits.h
  include/grpc/impl/codegen/slice.h
  include/grpc/impl/codegen/status.h
  include/grpc/impl/codegen/atm.h
  include/grpc/impl/codegen/atm_gcc_atomic.h
  include/grpc/impl/codegen/atm_gcc_sync.h
  include/grpc/impl/codegen/atm_windows.h
  include/grpc/impl/codegen/gpr_slice.h
  include/grpc/impl/codegen/gpr_types.h
  include/grpc/impl/codegen/port_platform.h
  include/grpc/impl/codegen/sync.h
  include/grpc/impl/codegen/sync_generic.h
  include/grpc/impl/codegen/sync_posix.h
  include/grpc/impl/codegen/sync_windows.h
  include/grpc++/impl/codegen/proto_utils.h
  include/grpc++/impl/codegen/config_protobuf.h
)
  string(REPLACE "include/" "" _path ${_hdr})
  get_filename_component(_path ${_path} PATH)
  install(FILES ${_hdr}
    DESTINATION "${gRPC_INSTALL_INCLUDEDIR}/${_path}"
  )
endforeach()


if (gRPC_INSTALL)
  install(TARGETS grpc++ EXPORT gRPCTargets
    RUNTIME DESTINATION ${gRPC_INSTALL_BINDIR}
    LIBRARY DESTINATION ${gRPC_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${gRPC_INSTALL_LIBDIR}
  )
endif()


add_library(grpc++_cronet
  src/cpp/client/cronet_credentials.cc
  src/cpp/client/insecure_credentials.cc
  src/cpp/common/insecure_create_auth_context.cc
  src/cpp/server/insecure_server_credentials.cc
  src/cpp/client/channel_cc.cc
  src/cpp/client/client_context.cc
  src/cpp/client/create_channel.cc
  src/cpp/client/create_channel_internal.cc
  src/cpp/client/create_channel_posix.cc
  src/cpp/client/credentials_cc.cc
  src/cpp/client/generic_stub.cc
  src/cpp/common/channel_arguments.cc
  src/cpp/common/channel_filter.cc
  src/cpp/common/completion_queue_cc.cc
  src/cpp/common/core_codegen.cc
  src/cpp/common/resource_quota_cc.cc
  src/cpp/common/rpc_method.cc
  src/cpp/common/version_cc.cc
  src/cpp/server/async_generic_service.cc
  src/cpp/server/channel_argument_option.cc
  src/cpp/server/create_default_thread_pool.cc
  src/cpp/server/dynamic_thread_pool.cc
  src/cpp/server/health/default_health_check_service.cc
  src/cpp/server/health/health.pb.c
  src/cpp/server/health/health_check_service.cc
  src/cpp/server/health/health_check_service_server_builder_option.cc
  src/cpp/server/server_builder.cc
  src/cpp/server/server_cc.cc
  src/cpp/server/server_context.cc
  src/cpp/server/server_credentials.cc
  src/cpp/server/server_posix.cc
  src/cpp/thread_manager/thread_manager.cc
  src/cpp/util/byte_buffer_cc.cc
  src/cpp/util/slice_cc.cc
  src/cpp/util/status.cc
  src/cpp/util/string_ref.cc
  src/cpp/util/time_cc.cc
  third_party/nanopb/pb_common.c
  third_party/nanopb/pb_decode.c
  third_party/nanopb/pb_encode.c
  src/cpp/codegen/codegen_init.cc
  src/core/ext/transport/chttp2/client/insecure/channel_create.c
  src/core/ext/transport/chttp2/client/insecure/channel_create_posix.c
  src/core/ext/transport/chttp2/client/chttp2_connector.c
  src/core/ext/transport/chttp2/transport/bin_decoder.c
  src/core/ext/transport/chttp2/transport/bin_encoder.c
  src/core/ext/transport/chttp2/transport/chttp2_plugin.c
  src/core/ext/transport/chttp2/transport/chttp2_transport.c
  src/core/ext/transport/chttp2/transport/frame_data.c
  src/core/ext/transport/chttp2/transport/frame_goaway.c
  src/core/ext/transport/chttp2/transport/frame_ping.c
  src/core/ext/transport/chttp2/transport/frame_rst_stream.c
  src/core/ext/transport/chttp2/transport/frame_settings.c
  src/core/ext/transport/chttp2/transport/frame_window_update.c
  src/core/ext/transport/chttp2/transport/hpack_encoder.c
  src/core/ext/transport/chttp2/transport/hpack_parser.c
  src/core/ext/transport/chttp2/transport/hpack_table.c
  src/core/ext/transport/chttp2/transport/http2_settings.c
  src/core/ext/transport/chttp2/transport/huffsyms.c
  src/core/ext/transport/chttp2/transport/incoming_metadata.c
  src/core/ext/transport/chttp2/transport/parsing.c
  src/core/ext/transport/chttp2/transport/stream_lists.c
  src/core/ext/transport/chttp2/transport/stream_map.c
  src/core/ext/transport/chttp2/transport/varint.c
  src/core/ext/transport/chttp2/transport/writing.c
  src/core/lib/channel/channel_args.c
  src/core/lib/channel/channel_stack.c
  src/core/lib/channel/channel_stack_builder.c
  src/core/lib/channel/connected_channel.c
  src/core/lib/channel/handshaker.c
  src/core/lib/channel/handshaker_factory.c
  src/core/lib/channel/handshaker_registry.c
  src/core/lib/compression/compression.c
  src/core/lib/compression/message_compress.c
  src/core/lib/compression/stream_compression.c
  src/core/lib/http/format_request.c
  src/core/lib/http/httpcli.c
  src/core/lib/http/parser.c
  src/core/lib/iomgr/closure.c
  src/core/lib/iomgr/combiner.c
  src/core/lib/iomgr/endpoint.c
  src/core/lib/iomgr/endpoint_pair_posix.c
  src/core/lib/iomgr/endpoint_pair_uv.c
  src/core/lib/iomgr/endpoint_pair_windows.c
  src/core/lib/iomgr/error.c
  src/core/lib/iomgr/ev_epoll1_linux.c
  src/core/lib/iomgr/ev_epoll_limited_pollers_linux.c
  src/core/lib/iomgr/ev_epoll_thread_pool_linux.c
  src/core/lib/iomgr/ev_epollex_linux.c
  src/core/lib/iomgr/ev_epollsig_linux.c
  src/core/lib/iomgr/ev_poll_posix.c
  src/core/lib/iomgr/ev_posix.c
  src/core/lib/iomgr/ev_windows.c
  src/core/lib/iomgr/exec_ctx.c
  src/core/lib/iomgr/executor.c
  src/core/lib/iomgr/iocp_windows.c
  src/core/lib/iomgr/iomgr.c
  src/core/lib/iomgr/iomgr_posix.c
  src/core/lib/iomgr/iomgr_uv.c
  src/core/lib/iomgr/iomgr_windows.c
  src/core/lib/iomgr/is_epollexclusive_available.c
  src/core/lib/iomgr/load_file.c
  src/core/lib/iomgr/lockfree_event.c
  src/core/lib/iomgr/network_status_tracker.c
  src/core/lib/iomgr/polling_entity.c
  src/core/lib/iomgr/pollset_set_uv.c
  src/core/lib/iomgr/pollset_set_windows.c
  src/core/lib/iomgr/pollset_uv.c
  src/core/lib/iomgr/pollset_windows.c
  src/core/lib/iomgr/resolve_address_posix.c
  src/core/lib/iomgr/resolve_address_uv.c
  src/core/lib/iomgr/resolve_address_windows.c
  src/core/lib/iomgr/resource_quota.c
  src/core/lib/iomgr/sockaddr_utils.c
  src/core/lib/iomgr/socket_factory_posix.c
  src/core/lib/iomgr/socket_mutator.c
  src/core/lib/iomgr/socket_utils_common_posix.c
  src/core/lib/iomgr/socket_utils_linux.c
  src/core/lib/iomgr/socket_utils_posix.c
  src/core/lib/iomgr/socket_utils_uv.c
  src/core/lib/iomgr/socket_utils_windows.c
  src/core/lib/iomgr/socket_windows.c
  src/core/lib/iomgr/tcp_client_posix.c
  src/core/lib/iomgr/tcp_client_uv.c
  src/core/lib/iomgr/tcp_client_windows.c
  src/core/lib/iomgr/tcp_posix.c
  src/core/lib/iomgr/tcp_server_posix.c
  src/core/lib/iomgr/tcp_server_utils_posix_common.c
  src/core/lib/iomgr/tcp_server_utils_posix_ifaddrs.c
  src/core/lib/iomgr/tcp_server_utils_posix_noifaddrs.c
  src/core/lib/iomgr/tcp_server_uv.c
  src/core/lib/iomgr/tcp_server_windows.c
  src/core/lib/iomgr/tcp_uv.c
  src/core/lib/iomgr/tcp_windows.c
  src/core/lib/iomgr/time_averaged_stats.c
  src/core/lib/iomgr/timer_generic.c
  src/core/lib/iomgr/timer_heap.c
  src/core/lib/iomgr/timer_manager.c
  src/core/lib/iomgr/timer_uv.c
  src/core/lib/iomgr/udp_server.c
  src/core/lib/iomgr/unix_sockets_posix.c
  src/core/lib/iomgr/unix_sockets_posix_noop.c
  src/core/lib/iomgr/wakeup_fd_cv.c
  src/core/lib/iomgr/wakeup_fd_eventfd.c
  src/core/lib/iomgr/wakeup_fd_nospecial.c
  src/core/lib/iomgr/wakeup_fd_pipe.c
  src/core/lib/iomgr/wakeup_fd_posix.c
  src/core/lib/json/json.c
  src/core/lib/json/json_reader.c
  src/core/lib/json/json_string.c
  src/core/lib/json/json_writer.c
  src/core/lib/slice/b64.c
  src/core/lib/slice/percent_encoding.c
  src/core/lib/slice/slice.c
  src/core/lib/slice/slice_buffer.c
  src/core/lib/slice/slice_hash_table.c
  src/core/lib/slice/slice_intern.c
  src/core/lib/slice/slice_string_helpers.c
  src/core/lib/surface/alarm.c
  src/core/lib/surface/api_trace.c
  src/core/lib/surface/byte_buffer.c
  src/core/lib/surface/byte_buffer_reader.c
  src/core/lib/surface/call.c
  src/core/lib/surface/call_details.c
  src/core/lib/surface/call_log_batch.c
  src/core/lib/surface/channel.c
  src/core/lib/surface/channel_init.c
  src/core/lib/surface/channel_ping.c
  src/core/lib/surface/channel_stack_type.c
  src/core/lib/surface/completion_queue.c
  src/core/lib/surface/completion_queue_factory.c
  src/core/lib/surface/event_string.c
  src/core/lib/surface/lame_client.cc
  src/core/lib/surface/metadata_array.c
  src/core/lib/surface/server.c
  src/core/lib/surface/validate_metadata.c
  src/core/lib/surface/version.c
  src/core/lib/transport/bdp_estimator.c
  src/core/lib/transport/byte_stream.c
  src/core/lib/transport/connectivity_state.c
  src/core/lib/transport/error_utils.c
  src/core/lib/transport/metadata.c
  src/core/lib/transport/metadata_batch.c
  src/core/lib/transport/pid_controller.c
  src/core/lib/transport/service_config.c
  src/core/lib/transport/static_metadata.c
  src/core/lib/transport/status_conversion.c
  src/core/lib/transport/timeout_encoding.c
  src/core/lib/transport/transport.c
  src/core/lib/transport/transport_op_string.c
  src/core/lib/debug/trace.c
  src/core/ext/transport/chttp2/alpn/alpn.c
  src/core/ext/filters/http/client/http_client_filter.c
  src/core/ext/filters/http/http_filters_plugin.c
  src/core/ext/filters/http/message_compress/message_compress_filter.c
  src/core/ext/filters/http/server/http_server_filter.c
  src/core/ext/filters/client_channel/channel_connectivity.c
  src/core/ext/filters/client_channel/client_channel.c
  src/core/ext/filters/client_channel/client_channel_factory.c
  src/core/ext/filters/client_channel/client_channel_plugin.c
  src/core/ext/filters/client_channel/connector.c
  src/core/ext/filters/client_channel/http_connect_handshaker.c
  src/core/ext/filters/client_channel/http_proxy.c
  src/core/ext/filters/client_channel/lb_policy.c
  src/core/ext/filters/client_channel/lb_policy_factory.c
  src/core/ext/filters/client_channel/lb_policy_registry.c
  src/core/ext/filters/client_channel/parse_address.c
  src/core/ext/filters/client_channel/proxy_mapper.c
  src/core/ext/filters/client_channel/proxy_mapper_registry.c
  src/core/ext/filters/client_channel/resolver.c
  src/core/ext/filters/client_channel/resolver_factory.c
  src/core/ext/filters/client_channel/resolver_registry.c
  src/core/ext/filters/client_channel/retry_throttle.c
  src/core/ext/filters/client_channel/subchannel.c
  src/core/ext/filters/client_channel/subchannel_index.c
  src/core/ext/filters/client_channel/uri_parser.c
  src/core/ext/filters/deadline/deadline_filter.c
  src/core/ext/transport/chttp2/server/insecure/server_chttp2.c
  src/core/ext/transport/chttp2/server/insecure/server_chttp2_posix.c
  src/core/ext/transport/chttp2/server/chttp2_server.c
  src/core/ext/census/base_resources.c
  src/core/ext/census/context.c
  src/core/ext/census/gen/census.pb.c
  src/core/ext/census/gen/trace_context.pb.c
  src/core/ext/census/grpc_context.c
  src/core/ext/census/grpc_filter.c
  src/core/ext/census/grpc_plugin.c
  src/core/ext/census/initialize.c
  src/core/ext/census/intrusive_hash_map.c
  src/core/ext/census/mlog.c
  src/core/ext/census/operation.c
  src/core/ext/census/placeholders.c
  src/core/ext/census/resource.c
  src/core/ext/census/trace_context.c
  src/core/ext/census/tracing.c
)

if(WIN32 AND MSVC)
  set_target_properties(grpc++_cronet PROPERTIES COMPILE_PDB_NAME "grpc++_cronet"
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  if (gRPC_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/grpc++_cronet.pdb
      DESTINATION ${gRPC_INSTALL_LIBDIR} OPTIONAL
    )
  endif()
endif()


target_include_directories(grpc++_cronet
  PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${ZLIB_INCLUDE_DIR}
  PRIVATE ${BENCHMARK}/include
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(grpc++_cronet
  ${_gRPC_BASELIB_LIBRARIES}
  ${_gRPC_SSL_LIBRARIES}
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  gpr
  grpc_cronet
  grpc
)

foreach(_hdr
  include/grpc++/alarm.h
  include/grpc++/channel.h
  include/grpc++/client_context.h
  include/grpc++/completion_queue.h
  include/grpc++/create_channel.h
  include/grpc++/create_channel_posix.h
  include/grpc++/ext/health_check_service_server_builder_option.h
  include/grpc++/generic/async_generic_service.h
  include/grpc++/generic/generic_stub.h
  include/grpc++/grpc++.h
  include/grpc++/health_check_service_interface.h
  include/grpc++/impl/call.h
  include/grpc++/impl/channel_argument_option.h
  include/grpc++/impl/client_unary_call.h
  include/grpc++/impl/codegen/core_codegen.h
  include/grpc++/impl/grpc_library.h
  include/grpc++/impl/method_handler_impl.h
  include/grpc++/impl/rpc_method.h
  include/grpc++/impl/rpc_service_method.h
  include/grpc++/impl/serialization_traits.h
  include/grpc++/impl/server_builder_option.h
  include/grpc++/impl/server_builder_plugin.h
  include/grpc++/impl/server_initializer.h
  include/grpc++/impl/service_type.h
  include/grpc++/resource_quota.h
  include/grpc++/security/auth_context.h
  include/grpc++/security/auth_metadata_processor.h
  include/grpc++/security/credentials.h
  include/grpc++/security/server_credentials.h
  include/grpc++/server.h
  include/grpc++/server_builder.h
  include/grpc++/server_context.h
  include/grpc++/server_posix.h
  include/grpc++/support/async_stream.h
  include/grpc++/support/async_unary_call.h
  include/grpc++/support/byte_buffer.h
  include/grpc++/support/channel_arguments.h
  include/grpc++/support/config.h
  include/grpc++/support/slice.h
  include/grpc++/support/status.h
  include/grpc++/support/status_code_enum.h
  include/grpc++/support/string_ref.h
  include/grpc++/support/stub_options.h
  include/grpc++/support/sync_stream.h
  include/grpc++/support/time.h
  include/grpc++/impl/codegen/async_stream.h
  include/grpc++/impl/codegen/async_unary_call.h
  include/grpc++/impl/codegen/call.h
  include/grpc++/impl/codegen/call_hook.h
  include/grpc++/impl/codegen/channel_interface.h
  include/grpc++/impl/codegen/client_context.h
  include/grpc++/impl/codegen/client_unary_call.h
  include/grpc++/impl/codegen/completion_queue.h
  include/grpc++/impl/codegen/completion_queue_tag.h
  include/grpc++/impl/codegen/config.h
  include/grpc++/impl/codegen/core_codegen_interface.h
  include/grpc++/impl/codegen/create_auth_context.h
  include/grpc++/impl/codegen/grpc_library.h
  include/grpc++/impl/codegen/metadata_map.h
  include/grpc++/impl/codegen/method_handler_impl.h
  include/grpc++/impl/codegen/rpc_method.h
  include/grpc++/impl/codegen/rpc_service_method.h
  include/grpc++/impl/codegen/security/auth_context.h
  include/grpc++/impl/codegen/serialization_traits.h
  include/grpc++/impl/codegen/server_context.h
  include/grpc++/impl/codegen/server_interface.h
  include/grpc++/impl/codegen/service_type.h
  include/grpc++/impl/codegen/slice.h
  include/grpc++/impl/codegen/status.h
  include/grpc++/impl/codegen/status_code_enum.h
  include/grpc++/impl/codegen/string_ref.h
  include/grpc++/impl/codegen/stub_options.h
  include/grpc++/impl/codegen/sync_stream.h
  include/grpc++/impl/codegen/time.h
  include/grpc/impl/codegen/byte_buffer_reader.h
  include/grpc/impl/codegen/compression_types.h
  include/grpc/impl/codegen/connectivity_state.h
  include/grpc/impl/codegen/exec_ctx_fwd.h
  include/grpc/impl/codegen/grpc_types.h
  include/grpc/impl/codegen/propagation_bits.h
  include/grpc/impl/codegen/slice.h
  include/grpc/impl/codegen/status.h
  include/grpc/impl/codegen/atm.h
  include/grpc/impl/codegen/atm_gcc_atomic.h
  include/grpc/impl/codegen/atm_gcc_sync.h
  include/grpc/impl/codegen/atm_windows.h
  include/grpc/impl/codegen/gpr_slice.h
  include/grpc/impl/codegen/gpr_types.h
  include/grpc/impl/codegen/port_platform.h
  include/grpc/impl/codegen/sync.h
  include/grpc/impl/codegen/sync_generic.h
  include/grpc/impl/codegen/sync_posix.h
  include/grpc/impl/codegen/sync_windows.h
  include/grpc/byte_buffer.h
  include/grpc/byte_buffer_reader.h
  include/grpc/compression.h
  include/grpc/grpc.h
  include/grpc/grpc_posix.h
  include/grpc/grpc_security_constants.h
  include/grpc/load_reporting.h
  include/grpc/slice.h
  include/grpc/slice_buffer.h
  include/grpc/status.h
  include/grpc/support/workaround_list.h
  include/grpc/census.h
)
  string(REPLACE "include/" "" _path ${_hdr})
  get_filename_component(_path ${_path} PATH)
  install(FILES ${_hdr}
    DESTINATION "${gRPC_INSTALL_INCLUDEDIR}/${_path}"
  )
endforeach()


if (gRPC_INSTALL)
  install(TARGETS grpc++_cronet EXPORT gRPCTargets
    RUNTIME DESTINATION ${gRPC_INSTALL_BINDIR}
    LIBRARY DESTINATION ${gRPC_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${gRPC_INSTALL_LIBDIR}
  )
endif()


add_library(grpc++_error_details
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/status/status.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/status/status.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/status/status.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/status/status.grpc.pb.h
  src/cpp/util/error_details.cc
)

if(WIN32 AND MSVC)
  set_target_properties(grpc++_error_details PROPERTIES COMPILE_PDB_NAME "grpc++_error_details"
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  if (gRPC_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/grpc++_error_details.pdb
      DESTINATION ${gRPC_INSTALL_LIBDIR} OPTIONAL
    )
  endif()
endif()

protobuf_generate_grpc_cpp(
  src/proto/grpc/status/status.proto
)

target_include_directories(grpc++_error_details
  PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${ZLIB_INCLUDE_DIR}
  PRIVATE ${BENCHMARK}/include
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(grpc++_error_details
  ${_gRPC_BASELIB_LIBRARIES}
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++
)

foreach(_hdr
  include/grpc++/support/error_details.h
)
  string(REPLACE "include/" "" _path ${_hdr})
  get_filename_component(_path ${_path} PATH)
  install(FILES ${_hdr}
    DESTINATION "${gRPC_INSTALL_INCLUDEDIR}/${_path}"
  )
endforeach()


if (gRPC_INSTALL)
  install(TARGETS grpc++_error_details EXPORT gRPCTargets
    RUNTIME DESTINATION ${gRPC_INSTALL_BINDIR}
    LIBRARY DESTINATION ${gRPC_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${gRPC_INSTALL_LIBDIR}
  )
endif()

if (gRPC_BUILD_TESTS)

add_library(grpc++_proto_reflection_desc_db
  test/cpp/util/proto_reflection_descriptor_database.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/reflection/v1alpha/reflection.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/reflection/v1alpha/reflection.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/reflection/v1alpha/reflection.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/reflection/v1alpha/reflection.grpc.pb.h
)

if(WIN32 AND MSVC)
  set_target_properties(grpc++_proto_reflection_desc_db PROPERTIES COMPILE_PDB_NAME "grpc++_proto_reflection_desc_db"
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  if (gRPC_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/grpc++_proto_reflection_desc_db.pdb
      DESTINATION ${gRPC_INSTALL_LIBDIR} OPTIONAL
    )
  endif()
endif()

protobuf_generate_grpc_cpp(
  src/proto/grpc/reflection/v1alpha/reflection.proto
)

target_include_directories(grpc++_proto_reflection_desc_db
  PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${ZLIB_INCLUDE_DIR}
  PRIVATE ${BENCHMARK}/include
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(grpc++_proto_reflection_desc_db
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++
  grpc
)

foreach(_hdr
  include/grpc++/impl/codegen/config_protobuf.h
)
  string(REPLACE "include/" "" _path ${_hdr})
  get_filename_component(_path ${_path} PATH)
  install(FILES ${_hdr}
    DESTINATION "${gRPC_INSTALL_INCLUDEDIR}/${_path}"
  )
endforeach()

endif (gRPC_BUILD_TESTS)

add_library(grpc++_reflection
  src/cpp/ext/proto_server_reflection.cc
  src/cpp/ext/proto_server_reflection_plugin.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/reflection/v1alpha/reflection.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/reflection/v1alpha/reflection.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/reflection/v1alpha/reflection.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/reflection/v1alpha/reflection.grpc.pb.h
)

if(WIN32 AND MSVC)
  set_target_properties(grpc++_reflection PROPERTIES COMPILE_PDB_NAME "grpc++_reflection"
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  if (gRPC_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/grpc++_reflection.pdb
      DESTINATION ${gRPC_INSTALL_LIBDIR} OPTIONAL
    )
  endif()
endif()

protobuf_generate_grpc_cpp(
  src/proto/grpc/reflection/v1alpha/reflection.proto
)

target_include_directories(grpc++_reflection
  PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${ZLIB_INCLUDE_DIR}
  PRIVATE ${BENCHMARK}/include
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(grpc++_reflection
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++
  grpc
)

foreach(_hdr
  include/grpc++/ext/proto_server_reflection_plugin.h
)
  string(REPLACE "include/" "" _path ${_hdr})
  get_filename_component(_path ${_path} PATH)
  install(FILES ${_hdr}
    DESTINATION "${gRPC_INSTALL_INCLUDEDIR}/${_path}"
  )
endforeach()


if (gRPC_INSTALL)
  install(TARGETS grpc++_reflection EXPORT gRPCTargets
    RUNTIME DESTINATION ${gRPC_INSTALL_BINDIR}
    LIBRARY DESTINATION ${gRPC_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${gRPC_INSTALL_LIBDIR}
  )
endif()

if (gRPC_BUILD_TESTS)

add_library(grpc++_test_config
  test/cpp/util/test_config_cc.cc
)

if(WIN32 AND MSVC)
  set_target_properties(grpc++_test_config PROPERTIES COMPILE_PDB_NAME "grpc++_test_config"
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  if (gRPC_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/grpc++_test_config.pdb
      DESTINATION ${gRPC_INSTALL_LIBDIR} OPTIONAL
    )
  endif()
endif()


target_include_directories(grpc++_test_config
  PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${ZLIB_INCLUDE_DIR}
  PRIVATE ${BENCHMARK}/include
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(grpc++_test_config
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
)


endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_library(grpc++_test_util
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/health/v1/health.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/health/v1/health.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/health/v1/health.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/health/v1/health.grpc.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo_messages.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo_messages.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo_messages.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo_messages.grpc.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo.grpc.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo_mock.grpc.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/duplicate/echo_duplicate.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/duplicate/echo_duplicate.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/duplicate/echo_duplicate.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/duplicate/echo_duplicate.grpc.pb.h
  test/cpp/end2end/test_service_impl.cc
  test/cpp/util/byte_buffer_proto_helper.cc
  test/cpp/util/create_test_channel.cc
  test/cpp/util/string_ref_helper.cc
  test/cpp/util/subprocess.cc
  test/cpp/util/test_credentials_provider.cc
  src/cpp/codegen/codegen_init.cc
)

if(WIN32 AND MSVC)
  set_target_properties(grpc++_test_util PROPERTIES COMPILE_PDB_NAME "grpc++_test_util"
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  if (gRPC_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/grpc++_test_util.pdb
      DESTINATION ${gRPC_INSTALL_LIBDIR} OPTIONAL
    )
  endif()
endif()

protobuf_generate_grpc_cpp(
  src/proto/grpc/health/v1/health.proto
)
protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/echo_messages.proto
)
protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/echo.proto
)
protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/duplicate/echo_duplicate.proto
)

target_include_directories(grpc++_test_util
  PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${ZLIB_INCLUDE_DIR}
  PRIVATE ${BENCHMARK}/include
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(grpc++_test_util
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++
  grpc_test_util
  grpc
)

foreach(_hdr
  include/grpc++/impl/codegen/async_stream.h
  include/grpc++/impl/codegen/async_unary_call.h
  include/grpc++/impl/codegen/call.h
  include/grpc++/impl/codegen/call_hook.h
  include/grpc++/impl/codegen/channel_interface.h
  include/grpc++/impl/codegen/client_context.h
  include/grpc++/impl/codegen/client_unary_call.h
  include/grpc++/impl/codegen/completion_queue.h
  include/grpc++/impl/codegen/completion_queue_tag.h
  include/grpc++/impl/codegen/config.h
  include/grpc++/impl/codegen/core_codegen_interface.h
  include/grpc++/impl/codegen/create_auth_context.h
  include/grpc++/impl/codegen/grpc_library.h
  include/grpc++/impl/codegen/metadata_map.h
  include/grpc++/impl/codegen/method_handler_impl.h
  include/grpc++/impl/codegen/rpc_method.h
  include/grpc++/impl/codegen/rpc_service_method.h
  include/grpc++/impl/codegen/security/auth_context.h
  include/grpc++/impl/codegen/serialization_traits.h
  include/grpc++/impl/codegen/server_context.h
  include/grpc++/impl/codegen/server_interface.h
  include/grpc++/impl/codegen/service_type.h
  include/grpc++/impl/codegen/slice.h
  include/grpc++/impl/codegen/status.h
  include/grpc++/impl/codegen/status_code_enum.h
  include/grpc++/impl/codegen/string_ref.h
  include/grpc++/impl/codegen/stub_options.h
  include/grpc++/impl/codegen/sync_stream.h
  include/grpc++/impl/codegen/time.h
  include/grpc/impl/codegen/byte_buffer_reader.h
  include/grpc/impl/codegen/compression_types.h
  include/grpc/impl/codegen/connectivity_state.h
  include/grpc/impl/codegen/exec_ctx_fwd.h
  include/grpc/impl/codegen/grpc_types.h
  include/grpc/impl/codegen/propagation_bits.h
  include/grpc/impl/codegen/slice.h
  include/grpc/impl/codegen/status.h
  include/grpc/impl/codegen/atm.h
  include/grpc/impl/codegen/atm_gcc_atomic.h
  include/grpc/impl/codegen/atm_gcc_sync.h
  include/grpc/impl/codegen/atm_windows.h
  include/grpc/impl/codegen/gpr_slice.h
  include/grpc/impl/codegen/gpr_types.h
  include/grpc/impl/codegen/port_platform.h
  include/grpc/impl/codegen/sync.h
  include/grpc/impl/codegen/sync_generic.h
  include/grpc/impl/codegen/sync_posix.h
  include/grpc/impl/codegen/sync_windows.h
  include/grpc++/impl/codegen/proto_utils.h
  include/grpc++/impl/codegen/config_protobuf.h
)
  string(REPLACE "include/" "" _path ${_hdr})
  get_filename_component(_path ${_path} PATH)
  install(FILES ${_hdr}
    DESTINATION "${gRPC_INSTALL_INCLUDEDIR}/${_path}"
  )
endforeach()

endif (gRPC_BUILD_TESTS)

add_library(grpc++_unsecure
  src/cpp/client/insecure_credentials.cc
  src/cpp/common/insecure_create_auth_context.cc
  src/cpp/server/insecure_server_credentials.cc
  src/cpp/client/channel_cc.cc
  src/cpp/client/client_context.cc
  src/cpp/client/create_channel.cc
  src/cpp/client/create_channel_internal.cc
  src/cpp/client/create_channel_posix.cc
  src/cpp/client/credentials_cc.cc
  src/cpp/client/generic_stub.cc
  src/cpp/common/channel_arguments.cc
  src/cpp/common/channel_filter.cc
  src/cpp/common/completion_queue_cc.cc
  src/cpp/common/core_codegen.cc
  src/cpp/common/resource_quota_cc.cc
  src/cpp/common/rpc_method.cc
  src/cpp/common/version_cc.cc
  src/cpp/server/async_generic_service.cc
  src/cpp/server/channel_argument_option.cc
  src/cpp/server/create_default_thread_pool.cc
  src/cpp/server/dynamic_thread_pool.cc
  src/cpp/server/health/default_health_check_service.cc
  src/cpp/server/health/health.pb.c
  src/cpp/server/health/health_check_service.cc
  src/cpp/server/health/health_check_service_server_builder_option.cc
  src/cpp/server/server_builder.cc
  src/cpp/server/server_cc.cc
  src/cpp/server/server_context.cc
  src/cpp/server/server_credentials.cc
  src/cpp/server/server_posix.cc
  src/cpp/thread_manager/thread_manager.cc
  src/cpp/util/byte_buffer_cc.cc
  src/cpp/util/slice_cc.cc
  src/cpp/util/status.cc
  src/cpp/util/string_ref.cc
  src/cpp/util/time_cc.cc
  third_party/nanopb/pb_common.c
  third_party/nanopb/pb_decode.c
  third_party/nanopb/pb_encode.c
  src/cpp/codegen/codegen_init.cc
)

if(WIN32 AND MSVC)
  set_target_properties(grpc++_unsecure PROPERTIES COMPILE_PDB_NAME "grpc++_unsecure"
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  if (gRPC_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/grpc++_unsecure.pdb
      DESTINATION ${gRPC_INSTALL_LIBDIR} OPTIONAL
    )
  endif()
endif()


target_include_directories(grpc++_unsecure
  PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${ZLIB_INCLUDE_DIR}
  PRIVATE ${BENCHMARK}/include
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(grpc++_unsecure
  ${_gRPC_BASELIB_LIBRARIES}
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  gpr
  grpc_unsecure
)

foreach(_hdr
  include/grpc++/alarm.h
  include/grpc++/channel.h
  include/grpc++/client_context.h
  include/grpc++/completion_queue.h
  include/grpc++/create_channel.h
  include/grpc++/create_channel_posix.h
  include/grpc++/ext/health_check_service_server_builder_option.h
  include/grpc++/generic/async_generic_service.h
  include/grpc++/generic/generic_stub.h
  include/grpc++/grpc++.h
  include/grpc++/health_check_service_interface.h
  include/grpc++/impl/call.h
  include/grpc++/impl/channel_argument_option.h
  include/grpc++/impl/client_unary_call.h
  include/grpc++/impl/codegen/core_codegen.h
  include/grpc++/impl/grpc_library.h
  include/grpc++/impl/method_handler_impl.h
  include/grpc++/impl/rpc_method.h
  include/grpc++/impl/rpc_service_method.h
  include/grpc++/impl/serialization_traits.h
  include/grpc++/impl/server_builder_option.h
  include/grpc++/impl/server_builder_plugin.h
  include/grpc++/impl/server_initializer.h
  include/grpc++/impl/service_type.h
  include/grpc++/resource_quota.h
  include/grpc++/security/auth_context.h
  include/grpc++/security/auth_metadata_processor.h
  include/grpc++/security/credentials.h
  include/grpc++/security/server_credentials.h
  include/grpc++/server.h
  include/grpc++/server_builder.h
  include/grpc++/server_context.h
  include/grpc++/server_posix.h
  include/grpc++/support/async_stream.h
  include/grpc++/support/async_unary_call.h
  include/grpc++/support/byte_buffer.h
  include/grpc++/support/channel_arguments.h
  include/grpc++/support/config.h
  include/grpc++/support/slice.h
  include/grpc++/support/status.h
  include/grpc++/support/status_code_enum.h
  include/grpc++/support/string_ref.h
  include/grpc++/support/stub_options.h
  include/grpc++/support/sync_stream.h
  include/grpc++/support/time.h
  include/grpc++/impl/codegen/async_stream.h
  include/grpc++/impl/codegen/async_unary_call.h
  include/grpc++/impl/codegen/call.h
  include/grpc++/impl/codegen/call_hook.h
  include/grpc++/impl/codegen/channel_interface.h
  include/grpc++/impl/codegen/client_context.h
  include/grpc++/impl/codegen/client_unary_call.h
  include/grpc++/impl/codegen/completion_queue.h
  include/grpc++/impl/codegen/completion_queue_tag.h
  include/grpc++/impl/codegen/config.h
  include/grpc++/impl/codegen/core_codegen_interface.h
  include/grpc++/impl/codegen/create_auth_context.h
  include/grpc++/impl/codegen/grpc_library.h
  include/grpc++/impl/codegen/metadata_map.h
  include/grpc++/impl/codegen/method_handler_impl.h
  include/grpc++/impl/codegen/rpc_method.h
  include/grpc++/impl/codegen/rpc_service_method.h
  include/grpc++/impl/codegen/security/auth_context.h
  include/grpc++/impl/codegen/serialization_traits.h
  include/grpc++/impl/codegen/server_context.h
  include/grpc++/impl/codegen/server_interface.h
  include/grpc++/impl/codegen/service_type.h
  include/grpc++/impl/codegen/slice.h
  include/grpc++/impl/codegen/status.h
  include/grpc++/impl/codegen/status_code_enum.h
  include/grpc++/impl/codegen/string_ref.h
  include/grpc++/impl/codegen/stub_options.h
  include/grpc++/impl/codegen/sync_stream.h
  include/grpc++/impl/codegen/time.h
  include/grpc/impl/codegen/byte_buffer_reader.h
  include/grpc/impl/codegen/compression_types.h
  include/grpc/impl/codegen/connectivity_state.h
  include/grpc/impl/codegen/exec_ctx_fwd.h
  include/grpc/impl/codegen/grpc_types.h
  include/grpc/impl/codegen/propagation_bits.h
  include/grpc/impl/codegen/slice.h
  include/grpc/impl/codegen/status.h
  include/grpc/impl/codegen/atm.h
  include/grpc/impl/codegen/atm_gcc_atomic.h
  include/grpc/impl/codegen/atm_gcc_sync.h
  include/grpc/impl/codegen/atm_windows.h
  include/grpc/impl/codegen/gpr_slice.h
  include/grpc/impl/codegen/gpr_types.h
  include/grpc/impl/codegen/port_platform.h
  include/grpc/impl/codegen/sync.h
  include/grpc/impl/codegen/sync_generic.h
  include/grpc/impl/codegen/sync_posix.h
  include/grpc/impl/codegen/sync_windows.h
)
  string(REPLACE "include/" "" _path ${_hdr})
  get_filename_component(_path ${_path} PATH)
  install(FILES ${_hdr}
    DESTINATION "${gRPC_INSTALL_INCLUDEDIR}/${_path}"
  )
endforeach()


if (gRPC_INSTALL)
  install(TARGETS grpc++_unsecure EXPORT gRPCTargets
    RUNTIME DESTINATION ${gRPC_INSTALL_BINDIR}
    LIBRARY DESTINATION ${gRPC_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${gRPC_INSTALL_LIBDIR}
  )
endif()

if (gRPC_BUILD_TESTS)

add_library(grpc_benchmark
  test/cpp/microbenchmarks/helpers.cc
)

if(WIN32 AND MSVC)
  set_target_properties(grpc_benchmark PROPERTIES COMPILE_PDB_NAME "grpc_benchmark"
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  if (gRPC_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/grpc_benchmark.pdb
      DESTINATION ${gRPC_INSTALL_LIBDIR} OPTIONAL
    )
  endif()
endif()


target_include_directories(grpc_benchmark
  PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${ZLIB_INCLUDE_DIR}
  PRIVATE ${BENCHMARK}/include
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(grpc_benchmark
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  benchmark
  grpc++
  grpc_test_util
  grpc
  ${_gRPC_GFLAGS_LIBRARIES}
)


endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_library(grpc_cli_libs
  test/cpp/util/cli_call.cc
  test/cpp/util/cli_credentials.cc
  test/cpp/util/grpc_tool.cc
  test/cpp/util/proto_file_parser.cc
  test/cpp/util/service_describer.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/reflection/v1alpha/reflection.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/reflection/v1alpha/reflection.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/reflection/v1alpha/reflection.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/reflection/v1alpha/reflection.grpc.pb.h
)

if(WIN32 AND MSVC)
  set_target_properties(grpc_cli_libs PROPERTIES COMPILE_PDB_NAME "grpc_cli_libs"
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  if (gRPC_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/grpc_cli_libs.pdb
      DESTINATION ${gRPC_INSTALL_LIBDIR} OPTIONAL
    )
  endif()
endif()

protobuf_generate_grpc_cpp(
  src/proto/grpc/reflection/v1alpha/reflection.proto
)

target_include_directories(grpc_cli_libs
  PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${ZLIB_INCLUDE_DIR}
  PRIVATE ${BENCHMARK}/include
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(grpc_cli_libs
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++_proto_reflection_desc_db
  grpc++
  grpc
)

foreach(_hdr
  include/grpc++/impl/codegen/config_protobuf.h
)
  string(REPLACE "include/" "" _path ${_hdr})
  get_filename_component(_path ${_path} PATH)
  install(FILES ${_hdr}
    DESTINATION "${gRPC_INSTALL_INCLUDEDIR}/${_path}"
  )
endforeach()

endif (gRPC_BUILD_TESTS)

add_library(grpc_plugin_support
  src/compiler/cpp_generator.cc
  src/compiler/csharp_generator.cc
  src/compiler/node_generator.cc
  src/compiler/objective_c_generator.cc
  src/compiler/php_generator.cc
  src/compiler/python_generator.cc
  src/compiler/ruby_generator.cc
)

if(WIN32 AND MSVC)
  set_target_properties(grpc_plugin_support PROPERTIES COMPILE_PDB_NAME "grpc_plugin_support"
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  if (gRPC_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/grpc_plugin_support.pdb
      DESTINATION ${gRPC_INSTALL_LIBDIR} OPTIONAL
    )
  endif()
endif()


target_include_directories(grpc_plugin_support
  PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${ZLIB_INCLUDE_DIR}
  PRIVATE ${BENCHMARK}/include
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(grpc_plugin_support
  ${_gRPC_PROTOBUF_PROTOC_LIBRARIES}
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
)

foreach(_hdr
  include/grpc++/impl/codegen/config_protobuf.h
)
  string(REPLACE "include/" "" _path ${_hdr})
  get_filename_component(_path ${_path} PATH)
  install(FILES ${_hdr}
    DESTINATION "${gRPC_INSTALL_INCLUDEDIR}/${_path}"
  )
endforeach()


if (gRPC_INSTALL)
  install(TARGETS grpc_plugin_support EXPORT gRPCTargets
    RUNTIME DESTINATION ${gRPC_INSTALL_BINDIR}
    LIBRARY DESTINATION ${gRPC_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${gRPC_INSTALL_LIBDIR}
  )
endif()

if (gRPC_BUILD_TESTS)

add_library(http2_client_main
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/empty.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/empty.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/empty.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/empty.grpc.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.grpc.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/test.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/test.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/test.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/test.grpc.pb.h
  test/cpp/interop/http2_client.cc
)

if(WIN32 AND MSVC)
  set_target_properties(http2_client_main PROPERTIES COMPILE_PDB_NAME "http2_client_main"
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  if (gRPC_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/http2_client_main.pdb
      DESTINATION ${gRPC_INSTALL_LIBDIR} OPTIONAL
    )
  endif()
endif()

protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/empty.proto
)
protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/messages.proto
)
protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/test.proto
)

target_include_directories(http2_client_main
  PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${ZLIB_INCLUDE_DIR}
  PRIVATE ${BENCHMARK}/include
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(http2_client_main
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  grpc++_test_config
)


endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_library(interop_client_helper
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.grpc.pb.h
  test/cpp/interop/client_helper.cc
)

if(WIN32 AND MSVC)
  set_target_properties(interop_client_helper PROPERTIES COMPILE_PDB_NAME "interop_client_helper"
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  if (gRPC_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/interop_client_helper.pdb
      DESTINATION ${gRPC_INSTALL_LIBDIR} OPTIONAL
    )
  endif()
endif()

protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/messages.proto
)

target_include_directories(interop_client_helper
  PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${ZLIB_INCLUDE_DIR}
  PRIVATE ${BENCHMARK}/include
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(interop_client_helper
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr
)


endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_library(interop_client_main
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/empty.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/empty.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/empty.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/empty.grpc.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.grpc.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/test.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/test.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/test.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/test.grpc.pb.h
  test/cpp/interop/client.cc
  test/cpp/interop/interop_client.cc
)

if(WIN32 AND MSVC)
  set_target_properties(interop_client_main PROPERTIES COMPILE_PDB_NAME "interop_client_main"
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  if (gRPC_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/interop_client_main.pdb
      DESTINATION ${gRPC_INSTALL_LIBDIR} OPTIONAL
    )
  endif()
endif()

protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/empty.proto
)
protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/messages.proto
)
protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/test.proto
)

target_include_directories(interop_client_main
  PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${ZLIB_INCLUDE_DIR}
  PRIVATE ${BENCHMARK}/include
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(interop_client_main
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  interop_client_helper
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  grpc++_test_config
)


endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_library(interop_server_helper
  test/cpp/interop/server_helper.cc
)

if(WIN32 AND MSVC)
  set_target_properties(interop_server_helper PROPERTIES COMPILE_PDB_NAME "interop_server_helper"
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  if (gRPC_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/interop_server_helper.pdb
      DESTINATION ${gRPC_INSTALL_LIBDIR} OPTIONAL
    )
  endif()
endif()


target_include_directories(interop_server_helper
  PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${ZLIB_INCLUDE_DIR}
  PRIVATE ${BENCHMARK}/include
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(interop_server_helper
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr
)


endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_library(interop_server_lib
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/empty.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/empty.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/empty.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/empty.grpc.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.grpc.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/test.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/test.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/test.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/test.grpc.pb.h
  test/cpp/interop/interop_server.cc
)

if(WIN32 AND MSVC)
  set_target_properties(interop_server_lib PROPERTIES COMPILE_PDB_NAME "interop_server_lib"
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  if (gRPC_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/interop_server_lib.pdb
      DESTINATION ${gRPC_INSTALL_LIBDIR} OPTIONAL
    )
  endif()
endif()

protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/empty.proto
)
protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/messages.proto
)
protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/test.proto
)

target_include_directories(interop_server_lib
  PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${ZLIB_INCLUDE_DIR}
  PRIVATE ${BENCHMARK}/include
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(interop_server_lib
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  interop_server_helper
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  grpc++_test_config
)


endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_library(interop_server_main
  test/cpp/interop/interop_server_bootstrap.cc
)

if(WIN32 AND MSVC)
  set_target_properties(interop_server_main PROPERTIES COMPILE_PDB_NAME "interop_server_main"
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  if (gRPC_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/interop_server_main.pdb
      DESTINATION ${gRPC_INSTALL_LIBDIR} OPTIONAL
    )
  endif()
endif()


target_include_directories(interop_server_main
  PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${ZLIB_INCLUDE_DIR}
  PRIVATE ${BENCHMARK}/include
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(interop_server_main
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  interop_server_lib
)


endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_library(qps
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.grpc.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/payloads.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/payloads.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/payloads.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/payloads.grpc.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/stats.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/stats.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/stats.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/stats.grpc.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/control.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/control.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/control.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/control.grpc.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/services.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/services.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/services.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/services.grpc.pb.h
  test/cpp/qps/benchmark_config.cc
  test/cpp/qps/client_async.cc
  test/cpp/qps/client_sync.cc
  test/cpp/qps/driver.cc
  test/cpp/qps/parse_json.cc
  test/cpp/qps/qps_worker.cc
  test/cpp/qps/report.cc
  test/cpp/qps/server_async.cc
  test/cpp/qps/server_sync.cc
  test/cpp/qps/usage_timer.cc
)

if(WIN32 AND MSVC)
  set_target_properties(qps PROPERTIES COMPILE_PDB_NAME "qps"
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  if (gRPC_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/qps.pdb
      DESTINATION ${gRPC_INSTALL_LIBDIR} OPTIONAL
    )
  endif()
endif()

protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/messages.proto
)
protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/payloads.proto
)
protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/stats.proto
)
protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/control.proto
)
protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/services.proto
)

target_include_directories(qps
  PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${ZLIB_INCLUDE_DIR}
  PRIVATE ${BENCHMARK}/include
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(qps
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc++_test_util
  grpc++
  grpc
)


endif (gRPC_BUILD_TESTS)

add_library(grpc_csharp_ext SHARED
  src/csharp/ext/grpc_csharp_ext.c
)

if(WIN32 AND MSVC)
  set_target_properties(grpc_csharp_ext PROPERTIES COMPILE_PDB_NAME "grpc_csharp_ext"
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  if (gRPC_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/grpc_csharp_ext.pdb
      DESTINATION ${gRPC_INSTALL_LIBDIR} OPTIONAL
    )
  endif()
endif()


target_include_directories(grpc_csharp_ext
  PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${ZLIB_INCLUDE_DIR}
  PRIVATE ${BENCHMARK}/include
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(grpc_csharp_ext
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc
  gpr
)



if (gRPC_INSTALL)
  install(TARGETS grpc_csharp_ext EXPORT gRPCTargets
    RUNTIME DESTINATION ${gRPC_INSTALL_BINDIR}
    LIBRARY DESTINATION ${gRPC_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${gRPC_INSTALL_LIBDIR}
  )
endif()

if (gRPC_BUILD_TESTS)

add_library(ares
  third_party/cares/cares/ares__close_sockets.c
  third_party/cares/cares/ares__get_hostent.c
  third_party/cares/cares/ares__read_line.c
  third_party/cares/cares/ares__timeval.c
  third_party/cares/cares/ares_cancel.c
  third_party/cares/cares/ares_create_query.c
  third_party/cares/cares/ares_data.c
  third_party/cares/cares/ares_destroy.c
  third_party/cares/cares/ares_expand_name.c
  third_party/cares/cares/ares_expand_string.c
  third_party/cares/cares/ares_fds.c
  third_party/cares/cares/ares_free_hostent.c
  third_party/cares/cares/ares_free_string.c
  third_party/cares/cares/ares_getenv.c
  third_party/cares/cares/ares_gethostbyaddr.c
  third_party/cares/cares/ares_gethostbyname.c
  third_party/cares/cares/ares_getnameinfo.c
  third_party/cares/cares/ares_getopt.c
  third_party/cares/cares/ares_getsock.c
  third_party/cares/cares/ares_init.c
  third_party/cares/cares/ares_library_init.c
  third_party/cares/cares/ares_llist.c
  third_party/cares/cares/ares_mkquery.c
  third_party/cares/cares/ares_nowarn.c
  third_party/cares/cares/ares_options.c
  third_party/cares/cares/ares_parse_a_reply.c
  third_party/cares/cares/ares_parse_aaaa_reply.c
  third_party/cares/cares/ares_parse_mx_reply.c
  third_party/cares/cares/ares_parse_naptr_reply.c
  third_party/cares/cares/ares_parse_ns_reply.c
  third_party/cares/cares/ares_parse_ptr_reply.c
  third_party/cares/cares/ares_parse_soa_reply.c
  third_party/cares/cares/ares_parse_srv_reply.c
  third_party/cares/cares/ares_parse_txt_reply.c
  third_party/cares/cares/ares_platform.c
  third_party/cares/cares/ares_process.c
  third_party/cares/cares/ares_query.c
  third_party/cares/cares/ares_search.c
  third_party/cares/cares/ares_send.c
  third_party/cares/cares/ares_strcasecmp.c
  third_party/cares/cares/ares_strdup.c
  third_party/cares/cares/ares_strerror.c
  third_party/cares/cares/ares_timeout.c
  third_party/cares/cares/ares_version.c
  third_party/cares/cares/ares_writev.c
  third_party/cares/cares/bitncmp.c
  third_party/cares/cares/inet_net_pton.c
  third_party/cares/cares/inet_ntop.c
  third_party/cares/cares/windows_port.c
)

if(WIN32 AND MSVC)
  set_target_properties(ares PROPERTIES COMPILE_PDB_NAME "ares"
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  if (gRPC_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ares.pdb
      DESTINATION ${gRPC_INSTALL_LIBDIR} OPTIONAL
    )
  endif()
endif()


target_include_directories(ares
  PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${ZLIB_INCLUDE_DIR}
  PRIVATE ${BENCHMARK}/include
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(ares
  ${_gRPC_SSL_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
)


endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_library(bad_client_test
  test/core/bad_client/bad_client.c
)

if(WIN32 AND MSVC)
  set_target_properties(bad_client_test PROPERTIES COMPILE_PDB_NAME "bad_client_test"
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  if (gRPC_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/bad_client_test.pdb
      DESTINATION ${gRPC_INSTALL_LIBDIR} OPTIONAL
    )
  endif()
endif()


target_include_directories(bad_client_test
  PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${ZLIB_INCLUDE_DIR}
  PRIVATE ${BENCHMARK}/include
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(bad_client_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util_unsecure
  grpc_unsecure
  gpr_test_util
  gpr
)


endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_library(bad_ssl_test_server
  test/core/bad_ssl/server_common.c
)

if(WIN32 AND MSVC)
  set_target_properties(bad_ssl_test_server PROPERTIES COMPILE_PDB_NAME "bad_ssl_test_server"
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  if (gRPC_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/bad_ssl_test_server.pdb
      DESTINATION ${gRPC_INSTALL_LIBDIR} OPTIONAL
    )
  endif()
endif()


target_include_directories(bad_ssl_test_server
  PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${ZLIB_INCLUDE_DIR}
  PRIVATE ${BENCHMARK}/include
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(bad_ssl_test_server
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)


endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_library(end2end_tests
  test/core/end2end/end2end_tests.c
  test/core/end2end/end2end_test_utils.c
  test/core/end2end/tests/authority_not_supported.c
  test/core/end2end/tests/bad_hostname.c
  test/core/end2end/tests/bad_ping.c
  test/core/end2end/tests/binary_metadata.c
  test/core/end2end/tests/call_creds.c
  test/core/end2end/tests/cancel_after_accept.c
  test/core/end2end/tests/cancel_after_client_done.c
  test/core/end2end/tests/cancel_after_invoke.c
  test/core/end2end/tests/cancel_after_round_trip.c
  test/core/end2end/tests/cancel_before_invoke.c
  test/core/end2end/tests/cancel_in_a_vacuum.c
  test/core/end2end/tests/cancel_with_status.c
  test/core/end2end/tests/compressed_payload.c
  test/core/end2end/tests/connectivity.c
  test/core/end2end/tests/default_host.c
  test/core/end2end/tests/disappearing_server.c
  test/core/end2end/tests/empty_batch.c
  test/core/end2end/tests/filter_call_init_fails.c
  test/core/end2end/tests/filter_causes_close.c
  test/core/end2end/tests/filter_latency.c
  test/core/end2end/tests/graceful_server_shutdown.c
  test/core/end2end/tests/high_initial_seqno.c
  test/core/end2end/tests/hpack_size.c
  test/core/end2end/tests/idempotent_request.c
  test/core/end2end/tests/invoke_large_request.c
  test/core/end2end/tests/keepalive_timeout.c
  test/core/end2end/tests/large_metadata.c
  test/core/end2end/tests/load_reporting_hook.c
  test/core/end2end/tests/max_concurrent_streams.c
  test/core/end2end/tests/max_connection_age.c
  test/core/end2end/tests/max_connection_idle.c
  test/core/end2end/tests/max_message_length.c
  test/core/end2end/tests/negative_deadline.c
  test/core/end2end/tests/network_status_change.c
  test/core/end2end/tests/no_logging.c
  test/core/end2end/tests/no_op.c
  test/core/end2end/tests/payload.c
  test/core/end2end/tests/ping.c
  test/core/end2end/tests/ping_pong_streaming.c
  test/core/end2end/tests/proxy_auth.c
  test/core/end2end/tests/registered_call.c
  test/core/end2end/tests/request_with_flags.c
  test/core/end2end/tests/request_with_payload.c
  test/core/end2end/tests/resource_quota_server.c
  test/core/end2end/tests/server_finishes_request.c
  test/core/end2end/tests/shutdown_finishes_calls.c
  test/core/end2end/tests/shutdown_finishes_tags.c
  test/core/end2end/tests/simple_cacheable_request.c
  test/core/end2end/tests/simple_delayed_request.c
  test/core/end2end/tests/simple_metadata.c
  test/core/end2end/tests/simple_request.c
  test/core/end2end/tests/streaming_error_response.c
  test/core/end2end/tests/trailing_metadata.c
  test/core/end2end/tests/workaround_cronet_compression.c
  test/core/end2end/tests/write_buffering.c
  test/core/end2end/tests/write_buffering_at_end.c
)

if(WIN32 AND MSVC)
  set_target_properties(end2end_tests PROPERTIES COMPILE_PDB_NAME "end2end_tests"
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  if (gRPC_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/end2end_tests.pdb
      DESTINATION ${gRPC_INSTALL_LIBDIR} OPTIONAL
    )
  endif()
endif()


target_include_directories(end2end_tests
  PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${ZLIB_INCLUDE_DIR}
  PRIVATE ${BENCHMARK}/include
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(end2end_tests
  ${_gRPC_SSL_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)


endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_library(end2end_nosec_tests
  test/core/end2end/end2end_nosec_tests.c
  test/core/end2end/end2end_test_utils.c
  test/core/end2end/tests/authority_not_supported.c
  test/core/end2end/tests/bad_hostname.c
  test/core/end2end/tests/bad_ping.c
  test/core/end2end/tests/binary_metadata.c
  test/core/end2end/tests/cancel_after_accept.c
  test/core/end2end/tests/cancel_after_client_done.c
  test/core/end2end/tests/cancel_after_invoke.c
  test/core/end2end/tests/cancel_after_round_trip.c
  test/core/end2end/tests/cancel_before_invoke.c
  test/core/end2end/tests/cancel_in_a_vacuum.c
  test/core/end2end/tests/cancel_with_status.c
  test/core/end2end/tests/compressed_payload.c
  test/core/end2end/tests/connectivity.c
  test/core/end2end/tests/default_host.c
  test/core/end2end/tests/disappearing_server.c
  test/core/end2end/tests/empty_batch.c
  test/core/end2end/tests/filter_call_init_fails.c
  test/core/end2end/tests/filter_causes_close.c
  test/core/end2end/tests/filter_latency.c
  test/core/end2end/tests/graceful_server_shutdown.c
  test/core/end2end/tests/high_initial_seqno.c
  test/core/end2end/tests/hpack_size.c
  test/core/end2end/tests/idempotent_request.c
  test/core/end2end/tests/invoke_large_request.c
  test/core/end2end/tests/keepalive_timeout.c
  test/core/end2end/tests/large_metadata.c
  test/core/end2end/tests/load_reporting_hook.c
  test/core/end2end/tests/max_concurrent_streams.c
  test/core/end2end/tests/max_connection_age.c
  test/core/end2end/tests/max_connection_idle.c
  test/core/end2end/tests/max_message_length.c
  test/core/end2end/tests/negative_deadline.c
  test/core/end2end/tests/network_status_change.c
  test/core/end2end/tests/no_logging.c
  test/core/end2end/tests/no_op.c
  test/core/end2end/tests/payload.c
  test/core/end2end/tests/ping.c
  test/core/end2end/tests/ping_pong_streaming.c
  test/core/end2end/tests/proxy_auth.c
  test/core/end2end/tests/registered_call.c
  test/core/end2end/tests/request_with_flags.c
  test/core/end2end/tests/request_with_payload.c
  test/core/end2end/tests/resource_quota_server.c
  test/core/end2end/tests/server_finishes_request.c
  test/core/end2end/tests/shutdown_finishes_calls.c
  test/core/end2end/tests/shutdown_finishes_tags.c
  test/core/end2end/tests/simple_cacheable_request.c
  test/core/end2end/tests/simple_delayed_request.c
  test/core/end2end/tests/simple_metadata.c
  test/core/end2end/tests/simple_request.c
  test/core/end2end/tests/streaming_error_response.c
  test/core/end2end/tests/trailing_metadata.c
  test/core/end2end/tests/workaround_cronet_compression.c
  test/core/end2end/tests/write_buffering.c
  test/core/end2end/tests/write_buffering_at_end.c
)

if(WIN32 AND MSVC)
  set_target_properties(end2end_nosec_tests PROPERTIES COMPILE_PDB_NAME "end2end_nosec_tests"
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  if (gRPC_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/end2end_nosec_tests.pdb
      DESTINATION ${gRPC_INSTALL_LIBDIR} OPTIONAL
    )
  endif()
endif()


target_include_directories(end2end_nosec_tests
  PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${ZLIB_INCLUDE_DIR}
  PRIVATE ${BENCHMARK}/include
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(end2end_nosec_tests
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util_unsecure
  grpc_unsecure
  gpr_test_util
  gpr
)


endif (gRPC_BUILD_TESTS)

if (gRPC_BUILD_TESTS)

add_executable(alarm_test
  test/core/surface/alarm_test.c
)


target_include_directories(alarm_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(alarm_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(algorithm_test
  test/core/compression/algorithm_test.c
)


target_include_directories(algorithm_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(algorithm_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(alloc_test
  test/core/support/alloc_test.c
)


target_include_directories(alloc_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(alloc_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(alpn_test
  test/core/transport/chttp2/alpn_test.c
)


target_include_directories(alpn_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(alpn_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(arena_test
  test/core/support/arena_test.c
)


target_include_directories(arena_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(arena_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(bad_server_response_test
  test/core/end2end/bad_server_response_test.c
)


target_include_directories(bad_server_response_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(bad_server_response_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  test_tcp_server
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(bdp_estimator_test
  test/core/transport/bdp_estimator_test.c
)


target_include_directories(bdp_estimator_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(bdp_estimator_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(bin_decoder_test
  test/core/transport/chttp2/bin_decoder_test.c
)


target_include_directories(bin_decoder_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(bin_decoder_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(bin_encoder_test
  test/core/transport/chttp2/bin_encoder_test.c
)


target_include_directories(bin_encoder_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(bin_encoder_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(census_context_test
  test/core/census/context_test.c
)


target_include_directories(census_context_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(census_context_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(census_intrusive_hash_map_test
  test/core/census/intrusive_hash_map_test.c
)


target_include_directories(census_intrusive_hash_map_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(census_intrusive_hash_map_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(census_resource_test
  test/core/census/resource_test.c
)


target_include_directories(census_resource_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(census_resource_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(census_trace_context_test
  test/core/census/trace_context_test.c
)


target_include_directories(census_trace_context_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(census_trace_context_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(channel_create_test
  test/core/surface/channel_create_test.c
)


target_include_directories(channel_create_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(channel_create_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)

add_executable(check_epollexclusive
  test/build/check_epollexclusive.c
)


target_include_directories(check_epollexclusive
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(check_epollexclusive
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc
  gpr
)


if (gRPC_INSTALL)
  install(TARGETS check_epollexclusive EXPORT gRPCTargets
    RUNTIME DESTINATION ${gRPC_INSTALL_BINDIR}
    LIBRARY DESTINATION ${gRPC_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${gRPC_INSTALL_LIBDIR}
  )
endif()

if (gRPC_BUILD_TESTS)

add_executable(chttp2_hpack_encoder_test
  test/core/transport/chttp2/hpack_encoder_test.c
)


target_include_directories(chttp2_hpack_encoder_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(chttp2_hpack_encoder_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(chttp2_stream_map_test
  test/core/transport/chttp2/stream_map_test.c
)


target_include_directories(chttp2_stream_map_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(chttp2_stream_map_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(chttp2_varint_test
  test/core/transport/chttp2/varint_test.c
)


target_include_directories(chttp2_varint_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(chttp2_varint_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(combiner_test
  test/core/iomgr/combiner_test.c
)


target_include_directories(combiner_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(combiner_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(compression_test
  test/core/compression/compression_test.c
)


target_include_directories(compression_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(compression_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(concurrent_connectivity_test
  test/core/surface/concurrent_connectivity_test.c
)


target_include_directories(concurrent_connectivity_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(concurrent_connectivity_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(connection_refused_test
  test/core/end2end/connection_refused_test.c
)


target_include_directories(connection_refused_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(connection_refused_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(dns_resolver_connectivity_test
  test/core/client_channel/resolvers/dns_resolver_connectivity_test.c
)


target_include_directories(dns_resolver_connectivity_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(dns_resolver_connectivity_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(dns_resolver_test
  test/core/client_channel/resolvers/dns_resolver_test.c
)


target_include_directories(dns_resolver_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(dns_resolver_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(dualstack_socket_test
  test/core/end2end/dualstack_socket_test.c
)


target_include_directories(dualstack_socket_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(dualstack_socket_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(endpoint_pair_test
  test/core/iomgr/endpoint_pair_test.c
)


target_include_directories(endpoint_pair_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(endpoint_pair_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(error_test
  test/core/iomgr/error_test.c
)


target_include_directories(error_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(error_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX)

add_executable(ev_epollsig_linux_test
  test/core/iomgr/ev_epollsig_linux_test.c
)


target_include_directories(ev_epollsig_linux_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(ev_epollsig_linux_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(fake_resolver_test
  test/core/client_channel/resolvers/fake_resolver_test.c
)


target_include_directories(fake_resolver_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(fake_resolver_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(fd_conservation_posix_test
  test/core/iomgr/fd_conservation_posix_test.c
)


target_include_directories(fd_conservation_posix_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(fd_conservation_posix_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(fd_posix_test
  test/core/iomgr/fd_posix_test.c
)


target_include_directories(fd_posix_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(fd_posix_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(fling_client
  test/core/fling/client.c
)


target_include_directories(fling_client
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(fling_client
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(fling_server
  test/core/fling/server.c
)


target_include_directories(fling_server
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(fling_server
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(fling_stream_test
  test/core/fling/fling_stream_test.c
)


target_include_directories(fling_stream_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(fling_stream_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(fling_test
  test/core/fling/fling_test.c
)


target_include_directories(fling_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(fling_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif()
endif (gRPC_BUILD_TESTS)

add_executable(gen_hpack_tables
  tools/codegen/core/gen_hpack_tables.c
)


target_include_directories(gen_hpack_tables
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(gen_hpack_tables
  ${_gRPC_ALLTARGETS_LIBRARIES}
  gpr
  grpc
)


if (gRPC_INSTALL)
  install(TARGETS gen_hpack_tables EXPORT gRPCTargets
    RUNTIME DESTINATION ${gRPC_INSTALL_BINDIR}
    LIBRARY DESTINATION ${gRPC_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${gRPC_INSTALL_LIBDIR}
  )
endif()


add_executable(gen_legal_metadata_characters
  tools/codegen/core/gen_legal_metadata_characters.c
)


target_include_directories(gen_legal_metadata_characters
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(gen_legal_metadata_characters
  ${_gRPC_ALLTARGETS_LIBRARIES}
)


if (gRPC_INSTALL)
  install(TARGETS gen_legal_metadata_characters EXPORT gRPCTargets
    RUNTIME DESTINATION ${gRPC_INSTALL_BINDIR}
    LIBRARY DESTINATION ${gRPC_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${gRPC_INSTALL_LIBDIR}
  )
endif()


add_executable(gen_percent_encoding_tables
  tools/codegen/core/gen_percent_encoding_tables.c
)


target_include_directories(gen_percent_encoding_tables
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(gen_percent_encoding_tables
  ${_gRPC_ALLTARGETS_LIBRARIES}
)


if (gRPC_INSTALL)
  install(TARGETS gen_percent_encoding_tables EXPORT gRPCTargets
    RUNTIME DESTINATION ${gRPC_INSTALL_BINDIR}
    LIBRARY DESTINATION ${gRPC_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${gRPC_INSTALL_LIBDIR}
  )
endif()

if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(goaway_server_test
  test/core/end2end/goaway_server_test.c
)


target_include_directories(goaway_server_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(goaway_server_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(gpr_avl_test
  test/core/support/avl_test.c
)


target_include_directories(gpr_avl_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(gpr_avl_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(gpr_backoff_test
  test/core/support/backoff_test.c
)


target_include_directories(gpr_backoff_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(gpr_backoff_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(gpr_cmdline_test
  test/core/support/cmdline_test.c
)


target_include_directories(gpr_cmdline_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(gpr_cmdline_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(gpr_cpu_test
  test/core/support/cpu_test.c
)


target_include_directories(gpr_cpu_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(gpr_cpu_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(gpr_env_test
  test/core/support/env_test.c
)


target_include_directories(gpr_env_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(gpr_env_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(gpr_histogram_test
  test/core/support/histogram_test.c
)


target_include_directories(gpr_histogram_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(gpr_histogram_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(gpr_host_port_test
  test/core/support/host_port_test.c
)


target_include_directories(gpr_host_port_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(gpr_host_port_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(gpr_log_test
  test/core/support/log_test.c
)


target_include_directories(gpr_log_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(gpr_log_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(gpr_mpscq_test
  test/core/support/mpscq_test.c
)


target_include_directories(gpr_mpscq_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(gpr_mpscq_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(gpr_spinlock_test
  test/core/support/spinlock_test.c
)


target_include_directories(gpr_spinlock_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(gpr_spinlock_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(gpr_stack_lockfree_test
  test/core/support/stack_lockfree_test.c
)


target_include_directories(gpr_stack_lockfree_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(gpr_stack_lockfree_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(gpr_string_test
  test/core/support/string_test.c
)


target_include_directories(gpr_string_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(gpr_string_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(gpr_sync_test
  test/core/support/sync_test.c
)


target_include_directories(gpr_sync_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(gpr_sync_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(gpr_thd_test
  test/core/support/thd_test.c
)


target_include_directories(gpr_thd_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(gpr_thd_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(gpr_time_test
  test/core/support/time_test.c
)


target_include_directories(gpr_time_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(gpr_time_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(gpr_tls_test
  test/core/support/tls_test.c
)


target_include_directories(gpr_tls_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(gpr_tls_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(gpr_useful_test
  test/core/support/useful_test.c
)


target_include_directories(gpr_useful_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(gpr_useful_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(grpc_auth_context_test
  test/core/security/auth_context_test.c
)


target_include_directories(grpc_auth_context_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(grpc_auth_context_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(grpc_b64_test
  test/core/slice/b64_test.c
)


target_include_directories(grpc_b64_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(grpc_b64_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(grpc_byte_buffer_reader_test
  test/core/surface/byte_buffer_reader_test.c
)


target_include_directories(grpc_byte_buffer_reader_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(grpc_byte_buffer_reader_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(grpc_channel_args_test
  test/core/channel/channel_args_test.c
)


target_include_directories(grpc_channel_args_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(grpc_channel_args_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(grpc_channel_stack_test
  test/core/channel/channel_stack_test.c
)


target_include_directories(grpc_channel_stack_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(grpc_channel_stack_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(grpc_completion_queue_test
  test/core/surface/completion_queue_test.c
)


target_include_directories(grpc_completion_queue_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(grpc_completion_queue_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(grpc_completion_queue_threading_test
  test/core/surface/completion_queue_threading_test.c
)


target_include_directories(grpc_completion_queue_threading_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(grpc_completion_queue_threading_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)

add_executable(grpc_create_jwt
  test/core/security/create_jwt.c
)


target_include_directories(grpc_create_jwt
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(grpc_create_jwt
  ${_gRPC_SSL_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc
  gpr
)


if (gRPC_INSTALL)
  install(TARGETS grpc_create_jwt EXPORT gRPCTargets
    RUNTIME DESTINATION ${gRPC_INSTALL_BINDIR}
    LIBRARY DESTINATION ${gRPC_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${gRPC_INSTALL_LIBDIR}
  )
endif()

if (gRPC_BUILD_TESTS)

add_executable(grpc_credentials_test
  test/core/security/credentials_test.c
)


target_include_directories(grpc_credentials_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(grpc_credentials_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(grpc_fetch_oauth2
  test/core/security/fetch_oauth2.c
)


target_include_directories(grpc_fetch_oauth2
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(grpc_fetch_oauth2
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(grpc_invalid_channel_args_test
  test/core/surface/invalid_channel_args_test.c
)


target_include_directories(grpc_invalid_channel_args_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(grpc_invalid_channel_args_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(grpc_json_token_test
  test/core/security/json_token_test.c
)


target_include_directories(grpc_json_token_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(grpc_json_token_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(grpc_jwt_verifier_test
  test/core/security/jwt_verifier_test.c
)


target_include_directories(grpc_jwt_verifier_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(grpc_jwt_verifier_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)

add_executable(grpc_print_google_default_creds_token
  test/core/security/print_google_default_creds_token.c
)


target_include_directories(grpc_print_google_default_creds_token
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(grpc_print_google_default_creds_token
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc
  gpr
)


if (gRPC_INSTALL)
  install(TARGETS grpc_print_google_default_creds_token EXPORT gRPCTargets
    RUNTIME DESTINATION ${gRPC_INSTALL_BINDIR}
    LIBRARY DESTINATION ${gRPC_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${gRPC_INSTALL_LIBDIR}
  )
endif()

if (gRPC_BUILD_TESTS)

add_executable(grpc_security_connector_test
  test/core/security/security_connector_test.c
)


target_include_directories(grpc_security_connector_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(grpc_security_connector_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)

add_executable(grpc_verify_jwt
  test/core/security/verify_jwt.c
)


target_include_directories(grpc_verify_jwt
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(grpc_verify_jwt
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc
  gpr
)


if (gRPC_INSTALL)
  install(TARGETS grpc_verify_jwt EXPORT gRPCTargets
    RUNTIME DESTINATION ${gRPC_INSTALL_BINDIR}
    LIBRARY DESTINATION ${gRPC_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${gRPC_INSTALL_LIBDIR}
  )
endif()

if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX)

add_executable(handshake_client
  test/core/handshake/client_ssl.c
)


target_include_directories(handshake_client
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(handshake_client
  ${_gRPC_SSL_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX)

add_executable(handshake_server
  test/core/handshake/server_ssl.c
)


target_include_directories(handshake_server
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(handshake_server
  ${_gRPC_SSL_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(hpack_parser_test
  test/core/transport/chttp2/hpack_parser_test.c
)


target_include_directories(hpack_parser_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(hpack_parser_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(hpack_table_test
  test/core/transport/chttp2/hpack_table_test.c
)


target_include_directories(hpack_table_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(hpack_table_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(http_parser_test
  test/core/http/parser_test.c
)


target_include_directories(http_parser_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(http_parser_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(httpcli_format_request_test
  test/core/http/format_request_test.c
)


target_include_directories(httpcli_format_request_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(httpcli_format_request_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(httpcli_test
  test/core/http/httpcli_test.c
)


target_include_directories(httpcli_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(httpcli_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX)

add_executable(httpscli_test
  test/core/http/httpscli_test.c
)


target_include_directories(httpscli_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(httpscli_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(init_test
  test/core/surface/init_test.c
)


target_include_directories(init_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(init_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(invalid_call_argument_test
  test/core/end2end/invalid_call_argument_test.c
)


target_include_directories(invalid_call_argument_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(invalid_call_argument_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(json_rewrite
  test/core/json/json_rewrite.c
)


target_include_directories(json_rewrite
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(json_rewrite
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(json_rewrite_test
  test/core/json/json_rewrite_test.c
)


target_include_directories(json_rewrite_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(json_rewrite_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(json_stream_error_test
  test/core/json/json_stream_error_test.c
)


target_include_directories(json_stream_error_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(json_stream_error_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(json_test
  test/core/json/json_test.c
)


target_include_directories(json_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(json_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(lame_client_test
  test/core/surface/lame_client_test.c
)


target_include_directories(lame_client_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(lame_client_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(lb_policies_test
  test/core/client_channel/lb_policies_test.c
)


target_include_directories(lb_policies_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(lb_policies_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(load_file_test
  test/core/iomgr/load_file_test.c
)


target_include_directories(load_file_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(load_file_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(memory_profile_client
  test/core/memory_usage/client.c
)


target_include_directories(memory_profile_client
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(memory_profile_client
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(memory_profile_server
  test/core/memory_usage/server.c
)


target_include_directories(memory_profile_server
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(memory_profile_server
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(memory_profile_test
  test/core/memory_usage/memory_usage_test.c
)


target_include_directories(memory_profile_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(memory_profile_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(message_compress_test
  test/core/compression/message_compress_test.c
)


target_include_directories(message_compress_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(message_compress_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(minimal_stack_is_minimal_test
  test/core/channel/minimal_stack_is_minimal_test.c
)


target_include_directories(minimal_stack_is_minimal_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(minimal_stack_is_minimal_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(mlog_test
  test/core/census/mlog_test.c
)


target_include_directories(mlog_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(mlog_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(multiple_server_queues_test
  test/core/end2end/multiple_server_queues_test.c
)


target_include_directories(multiple_server_queues_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(multiple_server_queues_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(murmur_hash_test
  test/core/support/murmur_hash_test.c
)


target_include_directories(murmur_hash_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(murmur_hash_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(no_server_test
  test/core/end2end/no_server_test.c
)


target_include_directories(no_server_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(no_server_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(num_external_connectivity_watchers_test
  test/core/surface/num_external_connectivity_watchers_test.c
)


target_include_directories(num_external_connectivity_watchers_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(num_external_connectivity_watchers_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(parse_address_test
  test/core/client_channel/parse_address_test.c
)


target_include_directories(parse_address_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(parse_address_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(percent_encoding_test
  test/core/slice/percent_encoding_test.c
)


target_include_directories(percent_encoding_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(percent_encoding_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX)

add_executable(pollset_set_test
  test/core/iomgr/pollset_set_test.c
)


target_include_directories(pollset_set_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(pollset_set_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(resolve_address_posix_test
  test/core/iomgr/resolve_address_posix_test.c
)


target_include_directories(resolve_address_posix_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(resolve_address_posix_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(resolve_address_test
  test/core/iomgr/resolve_address_test.c
)


target_include_directories(resolve_address_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(resolve_address_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(resource_quota_test
  test/core/iomgr/resource_quota_test.c
)


target_include_directories(resource_quota_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(resource_quota_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(secure_channel_create_test
  test/core/surface/secure_channel_create_test.c
)


target_include_directories(secure_channel_create_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(secure_channel_create_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(secure_endpoint_test
  test/core/security/secure_endpoint_test.c
)


target_include_directories(secure_endpoint_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(secure_endpoint_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(sequential_connectivity_test
  test/core/surface/sequential_connectivity_test.c
)


target_include_directories(sequential_connectivity_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(sequential_connectivity_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(server_chttp2_test
  test/core/surface/server_chttp2_test.c
)


target_include_directories(server_chttp2_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(server_chttp2_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(server_test
  test/core/surface/server_test.c
)


target_include_directories(server_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(server_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(slice_buffer_test
  test/core/slice/slice_buffer_test.c
)


target_include_directories(slice_buffer_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(slice_buffer_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(slice_hash_table_test
  test/core/slice/slice_hash_table_test.c
)


target_include_directories(slice_hash_table_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(slice_hash_table_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(slice_string_helpers_test
  test/core/slice/slice_string_helpers_test.c
)


target_include_directories(slice_string_helpers_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(slice_string_helpers_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(slice_test
  test/core/slice/slice_test.c
)


target_include_directories(slice_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(slice_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(sockaddr_resolver_test
  test/core/client_channel/resolvers/sockaddr_resolver_test.c
)


target_include_directories(sockaddr_resolver_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(sockaddr_resolver_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(sockaddr_utils_test
  test/core/iomgr/sockaddr_utils_test.c
)


target_include_directories(sockaddr_utils_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(sockaddr_utils_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(socket_utils_test
  test/core/iomgr/socket_utils_test.c
)


target_include_directories(socket_utils_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(socket_utils_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(status_conversion_test
  test/core/transport/status_conversion_test.c
)


target_include_directories(status_conversion_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(status_conversion_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(stream_compression_test
  test/core/compression/stream_compression_test.c
)


target_include_directories(stream_compression_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(stream_compression_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(stream_owned_slice_test
  test/core/transport/stream_owned_slice_test.c
)


target_include_directories(stream_owned_slice_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(stream_owned_slice_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(tcp_client_posix_test
  test/core/iomgr/tcp_client_posix_test.c
)


target_include_directories(tcp_client_posix_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(tcp_client_posix_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(tcp_client_uv_test
  test/core/iomgr/tcp_client_uv_test.c
)


target_include_directories(tcp_client_uv_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(tcp_client_uv_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(tcp_posix_test
  test/core/iomgr/tcp_posix_test.c
)


target_include_directories(tcp_posix_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(tcp_posix_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(tcp_server_posix_test
  test/core/iomgr/tcp_server_posix_test.c
)


target_include_directories(tcp_server_posix_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(tcp_server_posix_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(tcp_server_uv_test
  test/core/iomgr/tcp_server_uv_test.c
)


target_include_directories(tcp_server_uv_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(tcp_server_uv_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(time_averaged_stats_test
  test/core/iomgr/time_averaged_stats_test.c
)


target_include_directories(time_averaged_stats_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(time_averaged_stats_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(timeout_encoding_test
  test/core/transport/timeout_encoding_test.c
)


target_include_directories(timeout_encoding_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(timeout_encoding_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(timer_heap_test
  test/core/iomgr/timer_heap_test.c
)


target_include_directories(timer_heap_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(timer_heap_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(timer_list_test
  test/core/iomgr/timer_list_test.c
)


target_include_directories(timer_list_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(timer_list_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(transport_connectivity_state_test
  test/core/transport/connectivity_state_test.c
)


target_include_directories(transport_connectivity_state_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(transport_connectivity_state_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(transport_metadata_test
  test/core/transport/metadata_test.c
)


target_include_directories(transport_metadata_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(transport_metadata_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(transport_pid_controller_test
  test/core/transport/pid_controller_test.c
)


target_include_directories(transport_pid_controller_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(transport_pid_controller_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(transport_security_test
  test/core/tsi/transport_security_test.c
)


target_include_directories(transport_security_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(transport_security_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(udp_server_test
  test/core/iomgr/udp_server_test.c
)


target_include_directories(udp_server_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(udp_server_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(uri_parser_test
  test/core/client_channel/uri_parser_test.c
)


target_include_directories(uri_parser_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(uri_parser_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(wakeup_fd_cv_test
  test/core/iomgr/wakeup_fd_cv_test.c
)


target_include_directories(wakeup_fd_cv_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(wakeup_fd_cv_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(alarm_cpp_test
  test/cpp/common/alarm_cpp_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(alarm_cpp_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(alarm_cpp_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(async_end2end_test
  test/cpp/end2end/async_end2end_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(async_end2end_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(async_end2end_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(auth_property_iterator_test
  test/cpp/common/auth_property_iterator_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(auth_property_iterator_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(auth_property_iterator_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(bm_arena
  test/cpp/microbenchmarks/bm_arena.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(bm_arena
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(bm_arena
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_benchmark
  benchmark
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(bm_call_create
  test/cpp/microbenchmarks/bm_call_create.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(bm_call_create
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(bm_call_create
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_benchmark
  benchmark
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(bm_chttp2_hpack
  test/cpp/microbenchmarks/bm_chttp2_hpack.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(bm_chttp2_hpack
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(bm_chttp2_hpack
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_benchmark
  benchmark
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(bm_chttp2_transport
  test/cpp/microbenchmarks/bm_chttp2_transport.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(bm_chttp2_transport
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(bm_chttp2_transport
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_benchmark
  benchmark
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(bm_closure
  test/cpp/microbenchmarks/bm_closure.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(bm_closure
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(bm_closure
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_benchmark
  benchmark
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(bm_cq
  test/cpp/microbenchmarks/bm_cq.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(bm_cq
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(bm_cq
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_benchmark
  benchmark
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(bm_cq_multiple_threads
  test/cpp/microbenchmarks/bm_cq_multiple_threads.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(bm_cq_multiple_threads
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(bm_cq_multiple_threads
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_benchmark
  benchmark
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(bm_error
  test/cpp/microbenchmarks/bm_error.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(bm_error
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(bm_error
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_benchmark
  benchmark
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(bm_fullstack_streaming_ping_pong
  test/cpp/microbenchmarks/bm_fullstack_streaming_ping_pong.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(bm_fullstack_streaming_ping_pong
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(bm_fullstack_streaming_ping_pong
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_benchmark
  benchmark
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(bm_fullstack_streaming_pump
  test/cpp/microbenchmarks/bm_fullstack_streaming_pump.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(bm_fullstack_streaming_pump
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(bm_fullstack_streaming_pump
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_benchmark
  benchmark
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(bm_fullstack_trickle
  test/cpp/microbenchmarks/bm_fullstack_trickle.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(bm_fullstack_trickle
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(bm_fullstack_trickle
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_benchmark
  benchmark
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(bm_fullstack_unary_ping_pong
  test/cpp/microbenchmarks/bm_fullstack_unary_ping_pong.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(bm_fullstack_unary_ping_pong
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(bm_fullstack_unary_ping_pong
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_benchmark
  benchmark
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(bm_metadata
  test/cpp/microbenchmarks/bm_metadata.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(bm_metadata
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(bm_metadata
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_benchmark
  benchmark
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(bm_pollset
  test/cpp/microbenchmarks/bm_pollset.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(bm_pollset
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(bm_pollset
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_benchmark
  benchmark
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(channel_arguments_test
  test/cpp/common/channel_arguments_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(channel_arguments_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(channel_arguments_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++
  grpc
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(channel_filter_test
  test/cpp/common/channel_filter_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(channel_filter_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(channel_filter_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++
  grpc
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(cli_call_test
  test/cpp/util/cli_call_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(cli_call_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(cli_call_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_cli_libs
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(client_crash_test
  test/cpp/end2end/client_crash_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(client_crash_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(client_crash_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(client_crash_test_server
  test/cpp/end2end/client_crash_test_server.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(client_crash_test_server
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(client_crash_test_server
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(client_lb_end2end_test
  test/cpp/end2end/client_lb_end2end_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(client_lb_end2end_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(client_lb_end2end_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(codegen_test_full
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/control.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/control.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/control.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/control.grpc.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.grpc.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/payloads.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/payloads.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/payloads.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/payloads.grpc.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/services.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/services.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/services.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/services.grpc.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/stats.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/stats.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/stats.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/stats.grpc.pb.h
  test/cpp/codegen/codegen_test_full.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)

protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/control.proto
)
protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/messages.proto
)
protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/payloads.proto
)
protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/services.proto
)
protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/stats.proto
)

target_include_directories(codegen_test_full
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(codegen_test_full
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++
  grpc
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(codegen_test_minimal
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/control.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/control.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/control.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/control.grpc.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.grpc.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/payloads.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/payloads.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/payloads.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/payloads.grpc.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/services.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/services.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/services.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/services.grpc.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/stats.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/stats.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/stats.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/stats.grpc.pb.h
  test/cpp/codegen/codegen_test_minimal.cc
  src/cpp/codegen/codegen_init.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)

protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/control.proto
)
protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/messages.proto
)
protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/payloads.proto
)
protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/services.proto
)
protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/stats.proto
)

target_include_directories(codegen_test_minimal
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(codegen_test_minimal
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(credentials_test
  test/cpp/client/credentials_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(credentials_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(credentials_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++
  grpc
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(cxx_byte_buffer_test
  test/cpp/util/byte_buffer_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(cxx_byte_buffer_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(cxx_byte_buffer_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(cxx_slice_test
  test/cpp/util/slice_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(cxx_slice_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(cxx_slice_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(cxx_string_ref_test
  test/cpp/util/string_ref_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(cxx_string_ref_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(cxx_string_ref_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++
  grpc
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(cxx_time_test
  test/cpp/util/time_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(cxx_time_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(cxx_time_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(end2end_test
  test/cpp/end2end/end2end_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(end2end_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(end2end_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(error_details_test
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo_messages.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo_messages.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo_messages.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo_messages.grpc.pb.h
  test/cpp/util/error_details_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)

protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/echo_messages.proto
)

target_include_directories(error_details_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(error_details_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++_error_details
  grpc++
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(filter_end2end_test
  test/cpp/end2end/filter_end2end_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(filter_end2end_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(filter_end2end_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(generic_end2end_test
  test/cpp/end2end/generic_end2end_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(generic_end2end_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(generic_end2end_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(golden_file_test
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/compiler_test.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/compiler_test.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/compiler_test.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/compiler_test.grpc.pb.h
  test/cpp/codegen/golden_file_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)

protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/compiler_test.proto
)

target_include_directories(golden_file_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(golden_file_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++
  grpc
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(grpc_cli
  test/cpp/util/grpc_cli.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(grpc_cli
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(grpc_cli
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_cli_libs
  grpc++_proto_reflection_desc_db
  grpc++
  grpc
  gpr
  grpc++_test_config
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)

add_executable(grpc_cpp_plugin
  src/compiler/cpp_plugin.cc
)


target_include_directories(grpc_cpp_plugin
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(grpc_cpp_plugin
  ${_gRPC_PROTOBUF_PROTOC_LIBRARIES}
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_plugin_support
)


if (gRPC_INSTALL)
  install(TARGETS grpc_cpp_plugin EXPORT gRPCTargets
    RUNTIME DESTINATION ${gRPC_INSTALL_BINDIR}
    LIBRARY DESTINATION ${gRPC_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${gRPC_INSTALL_LIBDIR}
  )
endif()


add_executable(grpc_csharp_plugin
  src/compiler/csharp_plugin.cc
)


target_include_directories(grpc_csharp_plugin
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(grpc_csharp_plugin
  ${_gRPC_PROTOBUF_PROTOC_LIBRARIES}
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_plugin_support
)


if (gRPC_INSTALL)
  install(TARGETS grpc_csharp_plugin EXPORT gRPCTargets
    RUNTIME DESTINATION ${gRPC_INSTALL_BINDIR}
    LIBRARY DESTINATION ${gRPC_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${gRPC_INSTALL_LIBDIR}
  )
endif()


add_executable(grpc_node_plugin
  src/compiler/node_plugin.cc
)


target_include_directories(grpc_node_plugin
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(grpc_node_plugin
  ${_gRPC_PROTOBUF_PROTOC_LIBRARIES}
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_plugin_support
)


if (gRPC_INSTALL)
  install(TARGETS grpc_node_plugin EXPORT gRPCTargets
    RUNTIME DESTINATION ${gRPC_INSTALL_BINDIR}
    LIBRARY DESTINATION ${gRPC_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${gRPC_INSTALL_LIBDIR}
  )
endif()


add_executable(grpc_objective_c_plugin
  src/compiler/objective_c_plugin.cc
)


target_include_directories(grpc_objective_c_plugin
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(grpc_objective_c_plugin
  ${_gRPC_PROTOBUF_PROTOC_LIBRARIES}
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_plugin_support
)


if (gRPC_INSTALL)
  install(TARGETS grpc_objective_c_plugin EXPORT gRPCTargets
    RUNTIME DESTINATION ${gRPC_INSTALL_BINDIR}
    LIBRARY DESTINATION ${gRPC_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${gRPC_INSTALL_LIBDIR}
  )
endif()


add_executable(grpc_php_plugin
  src/compiler/php_plugin.cc
)


target_include_directories(grpc_php_plugin
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(grpc_php_plugin
  ${_gRPC_PROTOBUF_PROTOC_LIBRARIES}
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_plugin_support
)


if (gRPC_INSTALL)
  install(TARGETS grpc_php_plugin EXPORT gRPCTargets
    RUNTIME DESTINATION ${gRPC_INSTALL_BINDIR}
    LIBRARY DESTINATION ${gRPC_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${gRPC_INSTALL_LIBDIR}
  )
endif()


add_executable(grpc_python_plugin
  src/compiler/python_plugin.cc
)


target_include_directories(grpc_python_plugin
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(grpc_python_plugin
  ${_gRPC_PROTOBUF_PROTOC_LIBRARIES}
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_plugin_support
)


if (gRPC_INSTALL)
  install(TARGETS grpc_python_plugin EXPORT gRPCTargets
    RUNTIME DESTINATION ${gRPC_INSTALL_BINDIR}
    LIBRARY DESTINATION ${gRPC_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${gRPC_INSTALL_LIBDIR}
  )
endif()


add_executable(grpc_ruby_plugin
  src/compiler/ruby_plugin.cc
)


target_include_directories(grpc_ruby_plugin
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(grpc_ruby_plugin
  ${_gRPC_PROTOBUF_PROTOC_LIBRARIES}
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_plugin_support
)


if (gRPC_INSTALL)
  install(TARGETS grpc_ruby_plugin EXPORT gRPCTargets
    RUNTIME DESTINATION ${gRPC_INSTALL_BINDIR}
    LIBRARY DESTINATION ${gRPC_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${gRPC_INSTALL_LIBDIR}
  )
endif()

if (gRPC_BUILD_TESTS)

add_executable(grpc_tool_test
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo.grpc.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo_messages.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo_messages.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo_messages.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo_messages.grpc.pb.h
  test/cpp/util/grpc_tool_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)

protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/echo.proto
)
protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/echo_messages.proto
)

target_include_directories(grpc_tool_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(grpc_tool_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_cli_libs
  grpc++_proto_reflection_desc_db
  grpc++_reflection
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(grpclb_api_test
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/lb/v1/load_balancer.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/lb/v1/load_balancer.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/lb/v1/load_balancer.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/lb/v1/load_balancer.grpc.pb.h
  test/cpp/grpclb/grpclb_api_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)

protobuf_generate_grpc_cpp(
  src/proto/grpc/lb/v1/load_balancer.proto
)

target_include_directories(grpclb_api_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(grpclb_api_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(grpclb_end2end_test
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/lb/v1/load_balancer.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/lb/v1/load_balancer.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/lb/v1/load_balancer.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/lb/v1/load_balancer.grpc.pb.h
  test/cpp/end2end/grpclb_end2end_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)

protobuf_generate_grpc_cpp(
  src/proto/grpc/lb/v1/load_balancer.proto
)

target_include_directories(grpclb_end2end_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(grpclb_end2end_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(grpclb_test
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/lb/v1/load_balancer.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/lb/v1/load_balancer.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/lb/v1/load_balancer.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/lb/v1/load_balancer.grpc.pb.h
  test/cpp/grpclb/grpclb_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)

protobuf_generate_grpc_cpp(
  src/proto/grpc/lb/v1/load_balancer.proto
)

target_include_directories(grpclb_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(grpclb_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(health_service_end2end_test
  test/cpp/end2end/health_service_end2end_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(health_service_end2end_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(health_service_end2end_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(http2_client
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(http2_client
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(http2_client
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  http2_client_main
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  grpc++_test_config
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(hybrid_end2end_test
  test/cpp/end2end/hybrid_end2end_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(hybrid_end2end_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(hybrid_end2end_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(interop_client
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(interop_client
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(interop_client
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  interop_client_main
  interop_client_helper
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  grpc++_test_config
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(interop_server
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(interop_server
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(interop_server
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  interop_server_main
  interop_server_helper
  interop_server_lib
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  grpc++_test_config
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(interop_test
  test/cpp/interop/interop_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(interop_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(interop_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
  grpc++_test_config
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(json_run_localhost
  test/cpp/qps/json_run_localhost.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(json_run_localhost
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(json_run_localhost
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  grpc++_test_config
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(memory_test
  test/core/support/memory_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(memory_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(memory_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(metrics_client
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/metrics.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/metrics.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/metrics.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/metrics.grpc.pb.h
  test/cpp/interop/metrics_client.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)

protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/metrics.proto
)

target_include_directories(metrics_client
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(metrics_client
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++
  grpc
  gpr
  grpc++_test_config
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(mock_test
  test/cpp/end2end/mock_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(mock_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(mock_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(noop-benchmark
  test/cpp/microbenchmarks/noop-benchmark.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(noop-benchmark
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(noop-benchmark
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  benchmark
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(proto_server_reflection_test
  test/cpp/end2end/proto_server_reflection_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(proto_server_reflection_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(proto_server_reflection_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++_proto_reflection_desc_db
  grpc++_reflection
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(proto_utils_test
  test/cpp/codegen/proto_utils_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(proto_utils_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(proto_utils_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++
  grpc
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(qps_interarrival_test
  test/cpp/qps/qps_interarrival_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(qps_interarrival_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(qps_interarrival_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  qps
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  grpc++_test_config
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(qps_json_driver
  test/cpp/qps/qps_json_driver.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(qps_json_driver
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(qps_json_driver
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  qps
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  grpc++_test_config
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(qps_openloop_test
  test/cpp/qps/qps_openloop_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(qps_openloop_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(qps_openloop_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  qps
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  grpc++_test_config
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(qps_worker
  test/cpp/qps/worker.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(qps_worker
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(qps_worker
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  qps
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  grpc++_test_config
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(reconnect_interop_client
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/empty.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/empty.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/empty.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/empty.grpc.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.grpc.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/test.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/test.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/test.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/test.grpc.pb.h
  test/cpp/interop/reconnect_interop_client.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)

protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/empty.proto
)
protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/messages.proto
)
protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/test.proto
)

target_include_directories(reconnect_interop_client
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(reconnect_interop_client
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  grpc++_test_config
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(reconnect_interop_server
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/empty.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/empty.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/empty.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/empty.grpc.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.grpc.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/test.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/test.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/test.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/test.grpc.pb.h
  test/cpp/interop/reconnect_interop_server.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)

protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/empty.proto
)
protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/messages.proto
)
protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/test.proto
)

target_include_directories(reconnect_interop_server
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(reconnect_interop_server
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  reconnect_server
  test_tcp_server
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  grpc++_test_config
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(secure_auth_context_test
  test/cpp/common/secure_auth_context_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(secure_auth_context_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(secure_auth_context_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(secure_sync_unary_ping_pong_test
  test/cpp/qps/secure_sync_unary_ping_pong_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(secure_sync_unary_ping_pong_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(secure_sync_unary_ping_pong_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  qps
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  grpc++_test_config
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(server_builder_plugin_test
  test/cpp/end2end/server_builder_plugin_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(server_builder_plugin_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(server_builder_plugin_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(server_builder_test
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo_messages.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo_messages.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo_messages.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo_messages.grpc.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo.grpc.pb.h
  test/cpp/server/server_builder_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)

protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/echo_messages.proto
)
protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/echo.proto
)

target_include_directories(server_builder_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(server_builder_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++_test_util
  grpc_test_util
  gpr_test_util
  grpc++
  grpc
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(server_context_test_spouse_test
  test/cpp/test/server_context_test_spouse_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(server_context_test_spouse_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(server_context_test_spouse_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(server_crash_test
  test/cpp/end2end/server_crash_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(server_crash_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(server_crash_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(server_crash_test_client
  test/cpp/end2end/server_crash_test_client.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(server_crash_test_client
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(server_crash_test_client
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(server_request_call_test
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo_messages.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo_messages.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo_messages.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo_messages.grpc.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/echo.grpc.pb.h
  test/cpp/server/server_request_call_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)

protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/echo_messages.proto
)
protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/echo.proto
)

target_include_directories(server_request_call_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(server_request_call_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++_test_util
  grpc_test_util
  gpr_test_util
  grpc++
  grpc
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(shutdown_test
  test/cpp/end2end/shutdown_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(shutdown_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(shutdown_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(status_test
  test/cpp/util/status_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(status_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(status_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(streaming_throughput_test
  test/cpp/end2end/streaming_throughput_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(streaming_throughput_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(streaming_throughput_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(stress_test
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/empty.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/empty.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/empty.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/empty.grpc.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/messages.grpc.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/metrics.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/metrics.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/metrics.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/metrics.grpc.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/test.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/test.grpc.pb.cc
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/test.pb.h
  ${_gRPC_PROTO_GENS_DIR}/src/proto/grpc/testing/test.grpc.pb.h
  test/cpp/interop/interop_client.cc
  test/cpp/interop/stress_interop_client.cc
  test/cpp/interop/stress_test.cc
  test/cpp/util/metrics_server.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)

protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/empty.proto
)
protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/messages.proto
)
protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/metrics.proto
)
protobuf_generate_grpc_cpp(
  src/proto/grpc/testing/test.proto
)

target_include_directories(stress_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(stress_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  grpc++_test_config
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(thread_manager_test
  test/cpp/thread_manager/thread_manager_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(thread_manager_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(thread_manager_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++
  grpc
  gpr
  grpc++_test_config
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(thread_stress_test
  test/cpp/end2end/thread_stress_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(thread_stress_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(thread_stress_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(writes_per_rpc_test
  test/cpp/performance/writes_per_rpc_test.cc
  third_party/googletest/googletest/src/gtest-all.cc
  third_party/googletest/googlemock/src/gmock-all.cc
)


target_include_directories(writes_per_rpc_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
  PRIVATE third_party/googletest/googletest/include
  PRIVATE third_party/googletest/googletest
  PRIVATE third_party/googletest/googlemock/include
  PRIVATE third_party/googletest/googlemock
  PRIVATE ${_gRPC_PROTO_GENS_DIR}
)

target_link_libraries(writes_per_rpc_test
  ${_gRPC_PROTOBUF_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc++_test_util
  grpc_test_util
  grpc++
  grpc
  gpr_test_util
  gpr
  ${_gRPC_GFLAGS_LIBRARIES}
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(public_headers_must_be_c89
  test/core/surface/public_headers_must_be_c89.c
)


target_include_directories(public_headers_must_be_c89
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(public_headers_must_be_c89
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(badreq_bad_client_test
  test/core/bad_client/tests/badreq.c
)


target_include_directories(badreq_bad_client_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(badreq_bad_client_test
  ${_gRPC_SSL_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  bad_client_test
  grpc_test_util_unsecure
  grpc_unsecure
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(connection_prefix_bad_client_test
  test/core/bad_client/tests/connection_prefix.c
)


target_include_directories(connection_prefix_bad_client_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(connection_prefix_bad_client_test
  ${_gRPC_SSL_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  bad_client_test
  grpc_test_util_unsecure
  grpc_unsecure
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(head_of_line_blocking_bad_client_test
  test/core/bad_client/tests/head_of_line_blocking.c
)


target_include_directories(head_of_line_blocking_bad_client_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(head_of_line_blocking_bad_client_test
  ${_gRPC_SSL_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  bad_client_test
  grpc_test_util_unsecure
  grpc_unsecure
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(headers_bad_client_test
  test/core/bad_client/tests/headers.c
)


target_include_directories(headers_bad_client_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(headers_bad_client_test
  ${_gRPC_SSL_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  bad_client_test
  grpc_test_util_unsecure
  grpc_unsecure
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(initial_settings_frame_bad_client_test
  test/core/bad_client/tests/initial_settings_frame.c
)


target_include_directories(initial_settings_frame_bad_client_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(initial_settings_frame_bad_client_test
  ${_gRPC_SSL_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  bad_client_test
  grpc_test_util_unsecure
  grpc_unsecure
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(large_metadata_bad_client_test
  test/core/bad_client/tests/large_metadata.c
)


target_include_directories(large_metadata_bad_client_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(large_metadata_bad_client_test
  ${_gRPC_SSL_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  bad_client_test
  grpc_test_util_unsecure
  grpc_unsecure
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(server_registered_method_bad_client_test
  test/core/bad_client/tests/server_registered_method.c
)


target_include_directories(server_registered_method_bad_client_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(server_registered_method_bad_client_test
  ${_gRPC_SSL_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  bad_client_test
  grpc_test_util_unsecure
  grpc_unsecure
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(simple_request_bad_client_test
  test/core/bad_client/tests/simple_request.c
)


target_include_directories(simple_request_bad_client_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(simple_request_bad_client_test
  ${_gRPC_SSL_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  bad_client_test
  grpc_test_util_unsecure
  grpc_unsecure
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(unknown_frame_bad_client_test
  test/core/bad_client/tests/unknown_frame.c
)


target_include_directories(unknown_frame_bad_client_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(unknown_frame_bad_client_test
  ${_gRPC_SSL_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  bad_client_test
  grpc_test_util_unsecure
  grpc_unsecure
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(window_overflow_bad_client_test
  test/core/bad_client/tests/window_overflow.c
)


target_include_directories(window_overflow_bad_client_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(window_overflow_bad_client_test
  ${_gRPC_SSL_LIBRARIES}
  ${_gRPC_ALLTARGETS_LIBRARIES}
  bad_client_test
  grpc_test_util_unsecure
  grpc_unsecure
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(bad_ssl_cert_server
  test/core/bad_ssl/servers/cert.c
)


target_include_directories(bad_ssl_cert_server
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(bad_ssl_cert_server
  ${_gRPC_ALLTARGETS_LIBRARIES}
  bad_ssl_test_server
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(bad_ssl_cert_test
  test/core/bad_ssl/bad_ssl_test.c
)


target_include_directories(bad_ssl_cert_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(bad_ssl_cert_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(h2_census_test
  test/core/end2end/fixtures/h2_census.c
)


target_include_directories(h2_census_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(h2_census_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  end2end_tests
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(h2_compress_test
  test/core/end2end/fixtures/h2_compress.c
)


target_include_directories(h2_compress_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(h2_compress_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  end2end_tests
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(h2_fakesec_test
  test/core/end2end/fixtures/h2_fakesec.c
)


target_include_directories(h2_fakesec_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(h2_fakesec_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  end2end_tests
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(h2_fd_test
  test/core/end2end/fixtures/h2_fd.c
)


target_include_directories(h2_fd_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(h2_fd_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  end2end_tests
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(h2_full_test
  test/core/end2end/fixtures/h2_full.c
)


target_include_directories(h2_full_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(h2_full_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  end2end_tests
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX)

add_executable(h2_full+pipe_test
  test/core/end2end/fixtures/h2_full+pipe.c
)


target_include_directories(h2_full+pipe_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(h2_full+pipe_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  end2end_tests
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(h2_full+trace_test
  test/core/end2end/fixtures/h2_full+trace.c
)


target_include_directories(h2_full+trace_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(h2_full+trace_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  end2end_tests
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(h2_full+workarounds_test
  test/core/end2end/fixtures/h2_full+workarounds.c
)


target_include_directories(h2_full+workarounds_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(h2_full+workarounds_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  end2end_tests
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(h2_http_proxy_test
  test/core/end2end/fixtures/h2_http_proxy.c
)


target_include_directories(h2_http_proxy_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(h2_http_proxy_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  end2end_tests
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(h2_load_reporting_test
  test/core/end2end/fixtures/h2_load_reporting.c
)


target_include_directories(h2_load_reporting_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(h2_load_reporting_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  end2end_tests
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(h2_oauth2_test
  test/core/end2end/fixtures/h2_oauth2.c
)


target_include_directories(h2_oauth2_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(h2_oauth2_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  end2end_tests
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(h2_proxy_test
  test/core/end2end/fixtures/h2_proxy.c
)


target_include_directories(h2_proxy_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(h2_proxy_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  end2end_tests
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(h2_sockpair_test
  test/core/end2end/fixtures/h2_sockpair.c
)


target_include_directories(h2_sockpair_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(h2_sockpair_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  end2end_tests
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(h2_sockpair+trace_test
  test/core/end2end/fixtures/h2_sockpair+trace.c
)


target_include_directories(h2_sockpair+trace_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(h2_sockpair+trace_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  end2end_tests
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(h2_sockpair_1byte_test
  test/core/end2end/fixtures/h2_sockpair_1byte.c
)


target_include_directories(h2_sockpair_1byte_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(h2_sockpair_1byte_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  end2end_tests
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(h2_ssl_test
  test/core/end2end/fixtures/h2_ssl.c
)


target_include_directories(h2_ssl_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(h2_ssl_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  end2end_tests
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(h2_ssl_cert_test
  test/core/end2end/fixtures/h2_ssl_cert.c
)


target_include_directories(h2_ssl_cert_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(h2_ssl_cert_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  end2end_tests
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(h2_ssl_proxy_test
  test/core/end2end/fixtures/h2_ssl_proxy.c
)


target_include_directories(h2_ssl_proxy_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(h2_ssl_proxy_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  end2end_tests
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(h2_uds_test
  test/core/end2end/fixtures/h2_uds.c
)


target_include_directories(h2_uds_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(h2_uds_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  end2end_tests
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(inproc_test
  test/core/end2end/fixtures/inproc.c
)


target_include_directories(inproc_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(inproc_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  end2end_tests
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(h2_census_nosec_test
  test/core/end2end/fixtures/h2_census.c
)


target_include_directories(h2_census_nosec_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(h2_census_nosec_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  end2end_nosec_tests
  grpc_test_util_unsecure
  grpc_unsecure
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(h2_compress_nosec_test
  test/core/end2end/fixtures/h2_compress.c
)


target_include_directories(h2_compress_nosec_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(h2_compress_nosec_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  end2end_nosec_tests
  grpc_test_util_unsecure
  grpc_unsecure
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(h2_fd_nosec_test
  test/core/end2end/fixtures/h2_fd.c
)


target_include_directories(h2_fd_nosec_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(h2_fd_nosec_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  end2end_nosec_tests
  grpc_test_util_unsecure
  grpc_unsecure
  gpr_test_util
  gpr
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(h2_full_nosec_test
  test/core/end2end/fixtures/h2_full.c
)


target_include_directories(h2_full_nosec_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(h2_full_nosec_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  end2end_nosec_tests
  grpc_test_util_unsecure
  grpc_unsecure
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX)

add_executable(h2_full+pipe_nosec_test
  test/core/end2end/fixtures/h2_full+pipe.c
)


target_include_directories(h2_full+pipe_nosec_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(h2_full+pipe_nosec_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  end2end_nosec_tests
  grpc_test_util_unsecure
  grpc_unsecure
  gpr_test_util
  gpr
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(h2_full+trace_nosec_test
  test/core/end2end/fixtures/h2_full+trace.c
)


target_include_directories(h2_full+trace_nosec_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(h2_full+trace_nosec_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  end2end_nosec_tests
  grpc_test_util_unsecure
  grpc_unsecure
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(h2_full+workarounds_nosec_test
  test/core/end2end/fixtures/h2_full+workarounds.c
)


target_include_directories(h2_full+workarounds_nosec_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(h2_full+workarounds_nosec_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  end2end_nosec_tests
  grpc_test_util_unsecure
  grpc_unsecure
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(h2_http_proxy_nosec_test
  test/core/end2end/fixtures/h2_http_proxy.c
)


target_include_directories(h2_http_proxy_nosec_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(h2_http_proxy_nosec_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  end2end_nosec_tests
  grpc_test_util_unsecure
  grpc_unsecure
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(h2_load_reporting_nosec_test
  test/core/end2end/fixtures/h2_load_reporting.c
)


target_include_directories(h2_load_reporting_nosec_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(h2_load_reporting_nosec_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  end2end_nosec_tests
  grpc_test_util_unsecure
  grpc_unsecure
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(h2_proxy_nosec_test
  test/core/end2end/fixtures/h2_proxy.c
)


target_include_directories(h2_proxy_nosec_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(h2_proxy_nosec_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  end2end_nosec_tests
  grpc_test_util_unsecure
  grpc_unsecure
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(h2_sockpair_nosec_test
  test/core/end2end/fixtures/h2_sockpair.c
)


target_include_directories(h2_sockpair_nosec_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(h2_sockpair_nosec_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  end2end_nosec_tests
  grpc_test_util_unsecure
  grpc_unsecure
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(h2_sockpair+trace_nosec_test
  test/core/end2end/fixtures/h2_sockpair+trace.c
)


target_include_directories(h2_sockpair+trace_nosec_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(h2_sockpair+trace_nosec_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  end2end_nosec_tests
  grpc_test_util_unsecure
  grpc_unsecure
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(h2_sockpair_1byte_nosec_test
  test/core/end2end/fixtures/h2_sockpair_1byte.c
)


target_include_directories(h2_sockpair_1byte_nosec_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(h2_sockpair_1byte_nosec_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  end2end_nosec_tests
  grpc_test_util_unsecure
  grpc_unsecure
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)
if(_gRPC_PLATFORM_LINUX OR _gRPC_PLATFORM_MAC OR _gRPC_PLATFORM_POSIX)

add_executable(h2_uds_nosec_test
  test/core/end2end/fixtures/h2_uds.c
)


target_include_directories(h2_uds_nosec_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(h2_uds_nosec_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  end2end_nosec_tests
  grpc_test_util_unsecure
  grpc_unsecure
  gpr_test_util
  gpr
)

endif()
endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(inproc_nosec_test
  test/core/end2end/fixtures/inproc.c
)


target_include_directories(inproc_nosec_test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(inproc_nosec_test
  ${_gRPC_ALLTARGETS_LIBRARIES}
  end2end_nosec_tests
  grpc_test_util_unsecure
  grpc_unsecure
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(api_fuzzer_one_entry
  test/core/end2end/fuzzers/api_fuzzer.c
  test/core/util/one_corpus_entry_fuzzer.c
)


target_include_directories(api_fuzzer_one_entry
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(api_fuzzer_one_entry
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(client_fuzzer_one_entry
  test/core/end2end/fuzzers/client_fuzzer.c
  test/core/util/one_corpus_entry_fuzzer.c
)


target_include_directories(client_fuzzer_one_entry
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(client_fuzzer_one_entry
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(hpack_parser_fuzzer_test_one_entry
  test/core/transport/chttp2/hpack_parser_fuzzer_test.c
  test/core/util/one_corpus_entry_fuzzer.c
)


target_include_directories(hpack_parser_fuzzer_test_one_entry
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(hpack_parser_fuzzer_test_one_entry
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(http_request_fuzzer_test_one_entry
  test/core/http/request_fuzzer.c
  test/core/util/one_corpus_entry_fuzzer.c
)


target_include_directories(http_request_fuzzer_test_one_entry
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(http_request_fuzzer_test_one_entry
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(http_response_fuzzer_test_one_entry
  test/core/http/response_fuzzer.c
  test/core/util/one_corpus_entry_fuzzer.c
)


target_include_directories(http_response_fuzzer_test_one_entry
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(http_response_fuzzer_test_one_entry
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(json_fuzzer_test_one_entry
  test/core/json/fuzzer.c
  test/core/util/one_corpus_entry_fuzzer.c
)


target_include_directories(json_fuzzer_test_one_entry
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(json_fuzzer_test_one_entry
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(nanopb_fuzzer_response_test_one_entry
  test/core/nanopb/fuzzer_response.c
  test/core/util/one_corpus_entry_fuzzer.c
)


target_include_directories(nanopb_fuzzer_response_test_one_entry
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(nanopb_fuzzer_response_test_one_entry
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(nanopb_fuzzer_serverlist_test_one_entry
  test/core/nanopb/fuzzer_serverlist.c
  test/core/util/one_corpus_entry_fuzzer.c
)


target_include_directories(nanopb_fuzzer_serverlist_test_one_entry
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(nanopb_fuzzer_serverlist_test_one_entry
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(percent_decode_fuzzer_one_entry
  test/core/slice/percent_decode_fuzzer.c
  test/core/util/one_corpus_entry_fuzzer.c
)


target_include_directories(percent_decode_fuzzer_one_entry
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(percent_decode_fuzzer_one_entry
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(percent_encode_fuzzer_one_entry
  test/core/slice/percent_encode_fuzzer.c
  test/core/util/one_corpus_entry_fuzzer.c
)


target_include_directories(percent_encode_fuzzer_one_entry
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(percent_encode_fuzzer_one_entry
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(server_fuzzer_one_entry
  test/core/end2end/fuzzers/server_fuzzer.c
  test/core/util/one_corpus_entry_fuzzer.c
)


target_include_directories(server_fuzzer_one_entry
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(server_fuzzer_one_entry
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(ssl_server_fuzzer_one_entry
  test/core/security/ssl_server_fuzzer.c
  test/core/util/one_corpus_entry_fuzzer.c
)


target_include_directories(ssl_server_fuzzer_one_entry
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(ssl_server_fuzzer_one_entry
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)
if (gRPC_BUILD_TESTS)

add_executable(uri_fuzzer_test_one_entry
  test/core/client_channel/uri_fuzzer_test.c
  test/core/util/one_corpus_entry_fuzzer.c
)


target_include_directories(uri_fuzzer_test_one_entry
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${BORINGSSL_ROOT_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${BENCHMARK_ROOT_DIR}/include
  PRIVATE ${ZLIB_ROOT_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib
  PRIVATE ${CARES_BUILD_INCLUDE_DIR}
  PRIVATE ${CARES_INCLUDE_DIR}
  PRIVATE ${CARES_PLATFORM_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cares/cares
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/third_party/gflags/include
)

target_link_libraries(uri_fuzzer_test_one_entry
  ${_gRPC_ALLTARGETS_LIBRARIES}
  grpc_test_util
  grpc
  gpr_test_util
  gpr
)

endif (gRPC_BUILD_TESTS)







if (gRPC_INSTALL)
  install(EXPORT gRPCTargets
    DESTINATION ${gRPC_INSTALL_CMAKEDIR}
    NAMESPACE gRPC::
  )
endif()

foreach(_config gRPCConfig gRPCConfigVersion)
  configure_file(tools/cmake/${_config}.cmake.in
    ${_config}.cmake @ONLY)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${_config}.cmake
    DESTINATION ${gRPC_INSTALL_CMAKEDIR}
  )
endforeach()
