# MLIR project.
set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/projects/mlir/include ) # --src-root
set(MLIR_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include ) # --includedir
set(MLIR_TABLEGEN_EXE mlir-tblgen)

set(MLIR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MLIR_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# TODO: Temporary, remove when no longer needed.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

function(mlir_tablegen ofn)
  tablegen(MLIR ${ARGV} "-I${MLIR_MAIN_SRC_DIR}" "-I${MLIR_INCLUDE_DIR}")
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
      PARENT_SCOPE)
endfunction()

# TODO: This is to handle the current static registration, but should be
# factored out a bit.
function(whole_archive_link target)
  if("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
    set(link_flags "-L${CMAKE_BINARY_DIR}/lib ")
    FOREACH(LIB ${ARGN})
      string(CONCAT link_flags ${link_flags} "-Wl,-force_load ${CMAKE_BINARY_DIR}/lib/lib${LIB}.a ")
    ENDFOREACH(LIB)
  else()
    set(link_flags "-L${CMAKE_BINARY_DIR}/lib -Wl,--whole-archive,")
    FOREACH(LIB ${ARGN})
      string(CONCAT link_flags ${link_flags} "-l${LIB},")
    ENDFOREACH(LIB)
    string(CONCAT link_flags ${link_flags} "--no-whole-archive")
  endif()
  set_target_properties(${target} PROPERTIES LINK_FLAGS ${link_flags})
endfunction(whole_archive_link)

include_directories( "include")
include_directories( ${MLIR_INCLUDE_DIR})

add_subdirectory(include/mlir)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(unittests)
add_subdirectory(test)

if( LLVM_INCLUDE_EXAMPLES )
  add_subdirectory(examples)
endif()
