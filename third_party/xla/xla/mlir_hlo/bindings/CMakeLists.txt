add_subdirectory(c)

if(MHLO_ENABLE_BINDINGS_PYTHON)
  add_subdirectory(python)
endif()
