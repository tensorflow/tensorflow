include (ExternalProject)

set(tensorboard_dependencies)
add_custom_target(tensorboard_copy_dependencies)

function(tb_new_http_archive)
  cmake_parse_arguments(_TB "" "NAME;URL" "FILES" ${ARGN})
  ExternalProject_Add(${_TB_NAME}
    PREFIX ${_TB_NAME}
    URL ${_TB_URL}
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
  )

  set(src_dir "${CMAKE_CURRENT_BINARY_DIR}/${_TB_NAME}/src/${_TB_NAME}")
  set(dst_dir "${CMAKE_CURRENT_BINARY_DIR}/tensorboard_external/${_TB_NAME}")

  foreach(src_file ${_TB_FILES})
    add_custom_command(
      TARGET tensorboard_copy_dependencies PRE_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${src_dir}/${src_file} ${dst_dir}/${src_file}
    )
  endforeach()
  
  set(tensorboard_dependencies ${tensorboard_dependencies} ${_TB_NAME} PARENT_SCOPE)
endfunction()

function(tb_http_file)
  cmake_parse_arguments(_TB "" "NAME;URL" "" ${ARGN})
  get_filename_component(src_file ${_TB_URL} NAME)
  file(DOWNLOAD ${_TB_URL} "${DOWNLOAD_LOCATION}/${src_file}")
  
  set(src_dir "${DOWNLOAD_LOCATION}")
  set(dst_dir "${CMAKE_CURRENT_BINARY_DIR}/tensorboard_external/${_TB_NAME}/file")
  
  add_custom_command(
    TARGET tensorboard_copy_dependencies PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${src_dir}/${src_file} ${dst_dir}/${src_file}
  )
  
  add_custom_target(${_TB_NAME} DEPENDS ${src_dir}/${src_file})
  set(tensorboard_dependencies ${tensorboard_dependencies} ${_TB_NAME} PARENT_SCOPE)
endfunction()

include(external/tensorboard_deps.cmake)

add_dependencies(tensorboard_copy_dependencies ${tensorboard_dependencies})
