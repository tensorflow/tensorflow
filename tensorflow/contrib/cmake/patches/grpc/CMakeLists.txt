# GRPC global cmake file, modified for the TensorFlow build system.
# This currently builds C and C++ code.

# This file is based on the CMakeLists.txt available from here:
# https://github.com/grpc/grpc/blob/3bc78cd0b5bd784a235c01612d634b1ec5f8fb97/CMakeLists.txt
# with modifications to remove dependencies on SSL, and to reuse
# previously compiled libprotobuf.
#
# Copyright 2016, Google Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following disclaimer
# in the documentation and/or other materials provided with the
# distribution.
#     * Neither the name of Google Inc. nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
cmake_minimum_required(VERSION 2.8)

set(PACKAGE_NAME      "grpc")
set(PACKAGE_VERSION   "1.0.0-pre2-tensorflow")
set(PACKAGE_STRING    "${PACKAGE_NAME} ${PACKAGE_VERSION}")
set(PACKAGE_TARNAME   "${PACKAGE_NAME}-${PACKAGE_VERSION}")
set(PACKAGE_BUGREPORT "https://github.com/grpc/grpc/issues/")
project(${PACKAGE_NAME} C CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -std=c11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if(WIN32)
  add_definitions(-D_WIN32_WINNT=0x0A00)
  find_package(ZLIB REQUIRED)
endif(WIN32)

add_library(gpr
  src/core/lib/profiling/basic_timers.c
  src/core/lib/profiling/stap_timers.c
  src/core/lib/support/alloc.c
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
  src/core/lib/support/murmur_hash.c
  src/core/lib/support/slice.c
  src/core/lib/support/slice_buffer.c
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

target_include_directories(gpr
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${PROTOBUF_INCLUDE_DIRS}
  PRIVATE ${ZLIB_INCLUDE_DIRS}
)

add_library(grpc_unsecure
  src/core/lib/surface/init.c
  src/core/lib/surface/init_unsecure.c
  src/core/lib/channel/channel_args.c
  src/core/lib/channel/channel_stack.c
  src/core/lib/channel/channel_stack_builder.c
  src/core/lib/channel/compress_filter.c
  src/core/lib/channel/connected_channel.c
  src/core/lib/channel/http_client_filter.c
  src/core/lib/channel/http_server_filter.c
  src/core/lib/compression/compression.c
  src/core/lib/compression/message_compress.c
  src/core/lib/debug/trace.c
  src/core/lib/http/format_request.c
  src/core/lib/http/httpcli.c
  src/core/lib/http/parser.c
  src/core/lib/iomgr/closure.c
  src/core/lib/iomgr/endpoint.c
  src/core/lib/iomgr/endpoint_pair_posix.c
  src/core/lib/iomgr/endpoint_pair_windows.c
  src/core/lib/iomgr/error.c
  src/core/lib/iomgr/ev_epoll_linux.c
  src/core/lib/iomgr/ev_poll_and_epoll_posix.c
  src/core/lib/iomgr/ev_poll_posix.c
  src/core/lib/iomgr/ev_posix.c
  src/core/lib/iomgr/exec_ctx.c
  src/core/lib/iomgr/executor.c
  src/core/lib/iomgr/iocp_windows.c
  src/core/lib/iomgr/iomgr.c
  src/core/lib/iomgr/iomgr_posix.c
  src/core/lib/iomgr/iomgr_windows.c
  src/core/lib/iomgr/load_file.c
  src/core/lib/iomgr/network_status_tracker.c
  src/core/lib/iomgr/polling_entity.c
  src/core/lib/iomgr/pollset_set_windows.c
  src/core/lib/iomgr/pollset_windows.c
  src/core/lib/iomgr/resolve_address_posix.c
  src/core/lib/iomgr/resolve_address_windows.c
  src/core/lib/iomgr/sockaddr_utils.c
  src/core/lib/iomgr/socket_utils_common_posix.c
  src/core/lib/iomgr/socket_utils_linux.c
  src/core/lib/iomgr/socket_utils_posix.c
  src/core/lib/iomgr/socket_windows.c
  src/core/lib/iomgr/tcp_client_posix.c
  src/core/lib/iomgr/tcp_client_windows.c
  src/core/lib/iomgr/tcp_posix.c
  src/core/lib/iomgr/tcp_server_posix.c
  src/core/lib/iomgr/tcp_server_windows.c
  src/core/lib/iomgr/tcp_windows.c
  src/core/lib/iomgr/time_averaged_stats.c
  src/core/lib/iomgr/timer.c
  src/core/lib/iomgr/timer_heap.c
  src/core/lib/iomgr/udp_server.c
  src/core/lib/iomgr/unix_sockets_posix.c
  src/core/lib/iomgr/unix_sockets_posix_noop.c
  src/core/lib/iomgr/wakeup_fd_eventfd.c
  src/core/lib/iomgr/wakeup_fd_nospecial.c
  src/core/lib/iomgr/wakeup_fd_pipe.c
  src/core/lib/iomgr/wakeup_fd_posix.c
  src/core/lib/iomgr/workqueue_posix.c
  src/core/lib/iomgr/workqueue_windows.c
  src/core/lib/json/json.c
  src/core/lib/json/json_reader.c
  src/core/lib/json/json_string.c
  src/core/lib/json/json_writer.c
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
  src/core/lib/surface/event_string.c
  src/core/lib/surface/lame_client.c
  src/core/lib/surface/metadata_array.c
  src/core/lib/surface/server.c
  src/core/lib/surface/validate_metadata.c
  src/core/lib/surface/version.c
  src/core/lib/transport/byte_stream.c
  src/core/lib/transport/connectivity_state.c
  src/core/lib/transport/metadata.c
  src/core/lib/transport/metadata_batch.c
  src/core/lib/transport/static_metadata.c
  src/core/lib/transport/transport.c
  src/core/lib/transport/transport_op_string.c
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
  src/core/ext/transport/chttp2/transport/huffsyms.c
  src/core/ext/transport/chttp2/transport/incoming_metadata.c
  src/core/ext/transport/chttp2/transport/parsing.c
  src/core/ext/transport/chttp2/transport/status_conversion.c
  src/core/ext/transport/chttp2/transport/stream_lists.c
  src/core/ext/transport/chttp2/transport/stream_map.c
  src/core/ext/transport/chttp2/transport/timeout_encoding.c
  src/core/ext/transport/chttp2/transport/varint.c
  src/core/ext/transport/chttp2/transport/writing.c
  src/core/ext/transport/chttp2/alpn/alpn.c
  src/core/ext/transport/chttp2/client/insecure/channel_create.c
  src/core/ext/transport/chttp2/client/insecure/channel_create_posix.c
  src/core/ext/client_config/channel_connectivity.c
  src/core/ext/client_config/client_channel.c
  src/core/ext/client_config/client_channel_factory.c
  src/core/ext/client_config/client_config.c
  src/core/ext/client_config/client_config_plugin.c
  src/core/ext/client_config/connector.c
  src/core/ext/client_config/default_initial_connect_string.c
  src/core/ext/client_config/initial_connect_string.c
  src/core/ext/client_config/lb_policy.c
  src/core/ext/client_config/lb_policy_factory.c
  src/core/ext/client_config/lb_policy_registry.c
  src/core/ext/client_config/parse_address.c
  src/core/ext/client_config/resolver.c
  src/core/ext/client_config/resolver_factory.c
  src/core/ext/client_config/resolver_registry.c
  src/core/ext/client_config/subchannel.c
  src/core/ext/client_config/subchannel_call_holder.c
  src/core/ext/client_config/subchannel_index.c
  src/core/ext/client_config/uri_parser.c
  src/core/ext/resolver/dns/native/dns_resolver.c
  src/core/ext/resolver/sockaddr/sockaddr_resolver.c
  src/core/ext/load_reporting/load_reporting.c
  src/core/ext/load_reporting/load_reporting_filter.c
  src/core/ext/lb_policy/grpclb/load_balancer_api.c
  src/core/ext/lb_policy/grpclb/proto/grpc/lb/v1/load_balancer.pb.c
  third_party/nanopb/pb_common.c
  third_party/nanopb/pb_decode.c
  third_party/nanopb/pb_encode.c
  src/core/ext/lb_policy/pick_first/pick_first.c
  src/core/ext/lb_policy/round_robin/round_robin.c
  src/core/ext/census/context.c
  src/core/ext/census/gen/census.pb.c
  src/core/ext/census/grpc_context.c
  src/core/ext/census/grpc_filter.c
  src/core/ext/census/grpc_plugin.c
  src/core/ext/census/initialize.c
  src/core/ext/census/mlog.c
  src/core/ext/census/operation.c
  src/core/ext/census/placeholders.c
  src/core/ext/census/tracing.c
  src/core/plugin_registry/grpc_unsecure_plugin_registry.c
)

target_include_directories(grpc_unsecure
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${PROTOBUF_ROOT_DIR}/src
  PRIVATE ${ZLIB_INCLUDE_DIRS}
)

target_link_libraries(grpc_unsecure
  gpr
)

add_library(grpc++_unsecure
  src/cpp/common/insecure_create_auth_context.cc
  src/cpp/client/channel.cc
  src/cpp/client/client_context.cc
  src/cpp/client/create_channel.cc
  src/cpp/client/create_channel_internal.cc
  src/cpp/client/create_channel_posix.cc
  src/cpp/client/credentials.cc
  src/cpp/client/generic_stub.cc
  src/cpp/client/insecure_credentials.cc
  src/cpp/common/channel_arguments.cc
  src/cpp/common/completion_queue.cc
  src/cpp/common/core_codegen.cc
  src/cpp/common/rpc_method.cc
  src/cpp/server/async_generic_service.cc
  src/cpp/server/create_default_thread_pool.cc
  src/cpp/server/dynamic_thread_pool.cc
  src/cpp/server/insecure_server_credentials.cc
  src/cpp/server/server.cc
  src/cpp/server/server_builder.cc
  src/cpp/server/server_context.cc
  src/cpp/server/server_credentials.cc
  src/cpp/server/server_posix.cc
  src/cpp/util/byte_buffer.cc
  src/cpp/util/slice.cc
  src/cpp/util/status.cc
  src/cpp/util/string_ref.cc
  src/cpp/util/time.cc
  src/cpp/codegen/codegen_init.cc
)

target_include_directories(grpc++_unsecure
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${PROTOBUF_INCLUDE_DIRS}
  PRIVATE ${ZLIB_INCLUDE_DIRS}
)

target_link_libraries(grpc++_unsecure
  ${PROTOBUF_LIBRARIES}
  gpr
  grpc_unsecure
)
