########################################################
# tf_cc_framework library
########################################################
set(tf_cc_framework_srcs
    "${tensorflow_source_dir}/tensorflow/cc/framework/ops.h"
    "${tensorflow_source_dir}/tensorflow/cc/framework/ops.cc"
    "${tensorflow_source_dir}/tensorflow/cc/framework/scope.h"
    "${tensorflow_source_dir}/tensorflow/cc/framework/scope.cc"
)

add_library(tf_cc_framework OBJECT ${tf_cc_framework_srcs})

add_dependencies(tf_cc_framework tf_core_framework)

target_include_directories(tf_cc_framework PRIVATE
    ${tensorflow_source_dir}
    ${eigen_INCLUDE_DIRS}
)

target_compile_options(tf_cc_framework PRIVATE
    -fno-exceptions
    -DEIGEN_AVOID_STL_ARRAY
)

# C++11
target_compile_features(tf_cc_framework PRIVATE
    cxx_rvalue_references
)

########################################################
# tf_cc_op_gen_main library
########################################################
set(tf_cc_op_gen_main_srcs
    "${tensorflow_source_dir}/tensorflow/cc/framework/cc_op_gen.cc"
    "${tensorflow_source_dir}/tensorflow/cc/framework/cc_op_gen_main.cc"
    "${tensorflow_source_dir}/tensorflow/cc/framework/cc_op_gen.h"
)

add_library(tf_cc_op_gen_main OBJECT ${tf_cc_op_gen_main_srcs})

add_dependencies(tf_cc_op_gen_main tf_core_framework)

target_include_directories(tf_cc_op_gen_main PRIVATE
    ${tensorflow_source_dir}
    ${eigen_INCLUDE_DIRS}
)

#target_link_libraries(tf_cc_op_gen_main
#    ${CMAKE_THREAD_LIBS_INIT}
#    ${PROTOBUF_LIBRARIES}
#    tf_protos_cc
#    tf_core_lib
#    tf_core_framework
#)

target_compile_options(tf_cc_op_gen_main PRIVATE
    -fno-exceptions
    -DEIGEN_AVOID_STL_ARRAY
)

# C++11
target_compile_features(tf_cc_op_gen_main PRIVATE
    cxx_rvalue_references
)

########################################################
# tf_gen_op_wrapper_cc executables
########################################################

#
#  # Run the op generator.
#  if name == "sendrecv_ops":
#    include_internal = "1"
#  else:
#    include_internal = "0"
#  native.genrule(
#      name=name + "_genrule",
#      outs=[out_ops_file + ".h", out_ops_file + ".cc"],
#      tools=[":" + tool],
#      cmd=("$(location :" + tool + ") $(location :" + out_ops_file + ".h) " +
#           "$(location :" + out_ops_file + ".cc) " + include_internal))



#def tf_gen_op_wrappers_cc(name,
#                          op_lib_names=[],
#                          other_srcs=[],
#                          other_hdrs=[],
#                          pkg=""):
#  subsrcs = other_srcs
#  subhdrs = other_hdrs
#  for n in op_lib_names:
#    tf_gen_op_wrapper_cc(n, "ops/" + n, pkg=pkg)
#    subsrcs += ["ops/" + n + ".cc"]
#    subhdrs += ["ops/" + n + ".h"]
#
#  native.cc_library(name=name,
#                    srcs=subsrcs,
#                    hdrs=subhdrs,
#                    deps=["//tensorflow/core:core_cpu"],
#                    copts=tf_copts(),
#                    alwayslink=1,)

# create directory for ops generated files
set(cc_ops_target_dir ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/cc/ops)

add_custom_target(create_cc_ops_header_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${cc_ops_target_dir}
)

set(tf_cc_ops_generated_files)

set(tf_cc_op_lib_names
    ${tf_op_lib_names}
    "user_ops"
)
foreach(tf_cc_op_lib_name ${tf_cc_op_lib_names})
    #tf_gen_op_wrapper_cc(name, out_ops_file, pkg=""):
    #  # Construct an op generator binary for these ops.
    #  tool = out_ops_file + "_gen_cc"  #example ops/array_ops_gen_cc
    #  native.cc_binary(
    #      name = tool,
    #      copts = tf_copts(),
    #      linkopts = ["-lm"],
    #      linkstatic = 1,   # Faster to link this one-time-use binary dynamically
    #      deps = (["//tensorflow/cc:cc_op_gen_main",
    #               pkg + ":" + name + "_op_lib"])
    #  )
 
    # Using <TARGET_OBJECTS:...> to work around an issue where no ops were
    # registered (static initializers dropped by the linker because the ops
    # are not used explicitly in the *_gen_cc executables).
    add_executable(${tf_cc_op_lib_name}_gen_cc
        $<TARGET_OBJECTS:tf_cc_op_gen_main>
        $<TARGET_OBJECTS:tf_${tf_cc_op_lib_name}>
        $<TARGET_OBJECTS:tf_core_lib>
        $<TARGET_OBJECTS:tf_core_framework>
    )

    target_include_directories(${tf_cc_op_lib_name}_gen_cc PRIVATE
        ${tensorflow_source_dir}
        ${eigen_INCLUDE_DIRS}
    )

    find_package(ZLIB REQUIRED)

    target_link_libraries(${tf_cc_op_lib_name}_gen_cc PRIVATE
        ${CMAKE_THREAD_LIBS_INIT}
        ${PROTOBUF_LIBRARIES}
        tf_protos_cc
        re2_lib
        ${gif_STATIC_LIBRARIES}
        ${jpeg_STATIC_LIBRARIES}
        ${png_STATIC_LIBRARIES}
        ${ZLIB_LIBRARIES}
        ${jsoncpp_STATIC_LIBRARIES}
        ${boringssl_STATIC_LIBRARIES}
        ${CMAKE_DL_LIBS}
    )

    target_compile_options(${tf_cc_op_lib_name}_gen_cc PRIVATE
        -fno-exceptions
        -DEIGEN_AVOID_STL_ARRAY
        -lm
    )

    # C++11
    target_compile_features(${tf_cc_op_lib_name}_gen_cc PRIVATE
        cxx_rvalue_references
    )

    set(cc_ops_include_internal 0)
    if(${tf_cc_op_lib_name} STREQUAL "sendrecv_ops")
        set(cc_ops_include_internal 1)
    endif()

    add_custom_command(
        OUTPUT ${cc_ops_target_dir}/${tf_cc_op_lib_name}.h
               ${cc_ops_target_dir}/${tf_cc_op_lib_name}.cc
        COMMAND ${tf_cc_op_lib_name}_gen_cc ${cc_ops_target_dir}/${tf_cc_op_lib_name}.h ${cc_ops_target_dir}/${tf_cc_op_lib_name}.cc ${cc_ops_include_internal}
        DEPENDS ${tf_cc_op_lib_name}_gen_cc create_cc_ops_header_dir
    )
    
    list(APPEND tf_cc_ops_generated_files ${cc_ops_target_dir}/${tf_cc_op_lib_name}.h)
    list(APPEND tf_cc_ops_generated_files ${cc_ops_target_dir}/${tf_cc_op_lib_name}.cc)
endforeach()


########################################################
# tf_cc_ops library
########################################################
add_library(tf_cc_ops OBJECT
    ${tf_cc_ops_generated_files}
    "${tensorflow_source_dir}/tensorflow/cc/ops/const_op.h"
    "${tensorflow_source_dir}/tensorflow/cc/ops/const_op.cc"
    "${tensorflow_source_dir}/tensorflow/cc/ops/standard_ops.h"
)

target_include_directories(tf_cc_ops PRIVATE
    ${tensorflow_source_dir}
    ${eigen_INCLUDE_DIRS}
)

#target_link_libraries(tf_cc_ops
#    ${CMAKE_THREAD_LIBS_INIT}
#    ${PROTOBUF_LIBRARIES}
#    tf_protos_cc
#    tf_core_lib
#    tf_core_cpu
#    tf_models_word2vec_ops
#)

target_compile_options(tf_cc_ops PRIVATE
    -fno-exceptions
    -DEIGEN_AVOID_STL_ARRAY
)

# C++11
target_compile_features(tf_cc_ops PRIVATE
    cxx_rvalue_references
)


#tf_gen_op_wrappers_cc(
#    name = "cc_ops",
#    op_lib_names = [
#        ...
#    ],
#    other_hdrs = [
#        "ops/const_op.h",
#        "ops/standard_ops.h",
#    ],
#    other_srcs = [
#        "ops/const_op.cc",
#    ] + glob(["ops/*_grad.cc"]),
#    pkg = "//tensorflow/core",
#)
