# Generate a list of CMake library targets so that other CMake projects can
# link against them. LLVM calls its version of this file LLVMExports.cmake, but
# the usual CMake convention is ${Project}Targets.cmake.
# This configuration has been partly adapted from MLIR project's
# (llvm-project:mlir/cmake/modules).

include(LLVMDistributionSupport)

set(MHLO_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/mlir-hlo)
set(mhlo_cmake_builddir "${CMAKE_BINARY_DIR}/${MHLO_INSTALL_PACKAGE_DIR}")

# Keep this in sync with llvm/cmake/CMakeLists.txt!
set(LLVM_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/llvm)
set(llvm_cmake_builddir "${LLVM_BINARY_DIR}/${LLVM_INSTALL_PACKAGE_DIR}")

# Since we just use MLIR's cmake functions (add_mlir_library, etc.),
# MLIR_EXPORTS would have the necessary targets to export.
get_property(MLIR_EXPORTS GLOBAL PROPERTY MLIR_EXPORTS)
export(TARGETS ${MLIR_EXPORTS} FILE ${mhlo_cmake_builddir}/MHLOTargets.cmake)

# Generate MlirConfig.cmake for the build tree.
set(MHLO_CONFIG_CMAKE_DIR "${mhlo_cmake_builddir}")
set(MHLO_CONFIG_LLVM_CMAKE_DIR "${llvm_cmake_builddir}")
set(MHLO_CONFIG_INCLUDE_EXPORTS "include(\"\${MHLO_CMAKE_DIR}/MHLOTargets.cmake\")")
set(MHLO_CONFIG_INCLUDE_DIRS
  "${MLIR_HLO_MAIN_INCLUDE_DIR}"
  "${MLIR_HLO_GEN_INCLUDE_DIR}"
  )
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/MHLOConfig.cmake.in
  ${mhlo_cmake_builddir}/MHLOConfig.cmake
  @ONLY)
# Unset the variables now that the file is configured.
set(MHLO_CONFIG_CMAKE_DIR)
set(MHLO_CONFIG_LLVM_CMAKE_DIR)
set(MHLO_CONFIG_INCLUDE_DIRS)

# Generate MHLOConfig.cmake for the install tree.
set(MHLO_CONFIG_CODE "
# Compute the installation prefix from this MHLOConfig.cmake file location.
get_filename_component(MHLO_INSTALL_PREFIX \"\${CMAKE_CURRENT_LIST_FILE}\" PATH)")
# Construct the proper number of get_filename_component(... PATH)
# calls to compute the installation prefix.
string(REGEX REPLACE "/" ";" _count "${MHLO_INSTALL_PACKAGE_DIR}")
foreach(p ${_count})
  set(MHLO_CONFIG_CODE "${MHLO_CONFIG_CODE}
get_filename_component(MHLO_INSTALL_PREFIX \"\${MHLO_INSTALL_PREFIX}\" PATH)")
endforeach(p)
set(MHLO_CONFIG_CMAKE_DIR "\${MHLO_INSTALL_PREFIX}/${MHLO_INSTALL_PACKAGE_DIR}")
set(MHLO_CONFIG_LLVM_CMAKE_DIR "\${MHLO_INSTALL_PREFIX}/${LLVM_INSTALL_PACKAGE_DIR}")
get_config_exports_includes(MHLO MHLO_CONFIG_INCLUDE_EXPORTS)
set(MHLO_CONFIG_INCLUDE_DIRS
  "\${MHLO_INSTALL_PREFIX}/include"
  )
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/MHLOConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/MHLOConfig.cmake
  @ONLY)
# Unset the variables now that the file is configured.
set(MHLO_CONFIG_CODE)
set(MHLO_CONFIG_CMAKE_DIR)
set(MHLO_CONFIG_LLVM_CMAKE_DIR)
set(MHLO_CONFIG_INCLUDE_DIRS)
