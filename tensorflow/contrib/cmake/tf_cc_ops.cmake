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

########################################################
# tf_gen_op_wrapper_cc executables
########################################################

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
    # Using <TARGET_OBJECTS:...> to work around an issue where no ops were
    # registered (static initializers dropped by the linker because the ops
    # are not used explicitly in the *_gen_cc executables).
    add_executable(${tf_cc_op_lib_name}_gen_cc
        $<TARGET_OBJECTS:tf_cc_op_gen_main>
        $<TARGET_OBJECTS:tf_${tf_cc_op_lib_name}>
        $<TARGET_OBJECTS:tf_core_lib>
        $<TARGET_OBJECTS:tf_core_framework>
    )

    target_link_libraries(${tf_cc_op_lib_name}_gen_cc PRIVATE
        tf_protos_cc
        ${tensorflow_EXTERNAL_LIBRARIES}
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

########################################################
# tf_cc library
########################################################
file(GLOB_RECURSE tf_cc_srcs
    "${tensorflow_source_dir}/tensorflow/cc/client/*.h"
    "${tensorflow_source_dir}/tensorflow/cc/client/*.cc"
    "${tensorflow_source_dir}/tensorflow/cc/gradients/*.h"
    "${tensorflow_source_dir}/tensorflow/cc/gradients/*.cc"
    "${tensorflow_source_dir}/tensorflow/cc/saved_model/*.h"
    "${tensorflow_source_dir}/tensorflow/cc/saved_model/*.cc"
    "${tensorflow_source_dir}/tensorflow/cc/training/*.h"
    "${tensorflow_source_dir}/tensorflow/cc/training/*.cc"
)

set(tf_cc_srcs
    ${tf_cc_srcs}
    "${tensorflow_source_dir}/tensorflow/cc/framework/grad_op_registry.h"
    "${tensorflow_source_dir}/tensorflow/cc/framework/grad_op_registry.cc"
    "${tensorflow_source_dir}/tensorflow/cc/framework/gradient_checker.h"
    "${tensorflow_source_dir}/tensorflow/cc/framework/gradient_checker.cc"
    "${tensorflow_source_dir}/tensorflow/cc/framework/gradients.h"
    "${tensorflow_source_dir}/tensorflow/cc/framework/gradients.cc"
)

file(GLOB_RECURSE tf_cc_test_srcs
    "${tensorflow_source_dir}/tensorflow/cc/*test*.cc"
)

list(REMOVE_ITEM tf_cc_srcs ${tf_cc_test_srcs})

add_library(tf_cc OBJECT ${tf_cc_srcs})
add_dependencies(tf_cc tf_cc_framework tf_cc_ops)
