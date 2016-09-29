set(tf_tools_proto_text_src_dir "${tensorflow_source_dir}/tensorflow/tools/proto_text")

file(GLOB tf_tools_srcs
    "${tf_tools_proto_text_src_dir}/gen_proto_text_functions.cc"
    "${tf_tools_proto_text_src_dir}/gen_proto_text_functions_lib.h"
    "${tf_tools_proto_text_src_dir}/gen_proto_text_functions_lib.cc"
)

set(proto_text "proto_text")

add_executable(${proto_text}
    ${tf_tools_srcs}
    $<TARGET_OBJECTS:tf_core_lib>
)

target_include_directories(${proto_text} PUBLIC
    ${tensorflow_source_dir}
)

# TODO(mrry): Cut down the dependencies of this tool.
target_link_libraries(${proto_text} PUBLIC
    ${CMAKE_THREAD_LIBS_INIT}
    ${PROTOBUF_LIBRARIES}
    ${gif_STATIC_LIBRARIES}
    ${jpeg_STATIC_LIBRARIES}
    ${png_STATIC_LIBRARIES}
    ${ZLIB_LIBRARIES}
    ${jsoncpp_STATIC_LIBRARIES}
    ${CMAKE_DL_LIBS}
    )
if(tensorflow_ENABLE_SSL_SUPPORT)
  target_link_libraries(${proto_text} PUBLIC ${boringssl_STATIC_LIBRARIES})
endif()


add_dependencies(${proto_text}
    tf_core_lib
    protobuf
)

target_compile_options(${proto_text} PRIVATE
    -fno-exceptions
    -DEIGEN_AVOID_STL_ARRAY
)

# C++11
target_compile_features(${proto_text} PRIVATE
    cxx_rvalue_references
)
