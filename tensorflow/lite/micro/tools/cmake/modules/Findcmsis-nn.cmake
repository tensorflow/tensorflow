
include(cmsis-nn)
if(cmsis-nn_POPULATED)
  set(CMSIS-NN_FOUND TRUE CACHE BOOL "Found CMSIS-NN")
  get_target_property(CMSIS-NN_INCLUDE_DIRS cmsis-nn INCLUDE_DIRECTORIES)
  set(CMSIS-NN_INCLUDE_DIRS ${CMSIS-NN_INCLUDE_DIRS} CACHE STRING
    "CMSIS-NN include dirs"
  )
endif()

