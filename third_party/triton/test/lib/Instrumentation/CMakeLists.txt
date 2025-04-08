set(GPU_INSTRUMENTATION_PASSES
	GPUInstrumentationTestLib
    )

set(GPUInstrumentationTestLib_SOURCES
    GPUHello.cpp
    )


foreach( plugin ${GPU_INSTRUMENTATION_PASSES} )
    add_library(
      ${plugin}
      SHARED
      ${${plugin}_SOURCES}
      )

    target_link_libraries(
      ${plugin}
      PRIVATE
      LLVMCore
      "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>"
      )
    # CMAKE_LIBRARY_OUTPUT_DIRECTORY is only set during the Python
    # build. It is empty if building directly from the root
    # CMakeLists.txt file. Therefore if not building from Python just
    # use the default CMake shared lib path otherwise this causes a hard
    # build error
    if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set_target_properties(${plugin} PROPERTIES
          LIBRARY_OUTPUT_DIRECTORY
      "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../instrumentation")
    endif(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)

    # This is set to -fvisibility=hidden in the top level CMake file
    # which causes the llvmGetPassPluginInfo symbol to be hidden and
    # an "entry point not found" error. Reset it just for this target
    if(NOT MSVC)
      target_compile_options(${plugin} PRIVATE -fvisibility=default)
    endif()
endforeach()
