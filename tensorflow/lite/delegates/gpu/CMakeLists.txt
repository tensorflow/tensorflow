set(SRCS
	metal_delegate.mm
	metal_delegate.h
)

include_directories(${CMAKE_BINARY_DIR}/abseil-cpp)
include_directories(${CMAKE_BINARY_DIR}/flatbuffers/include)

add_library(metal_delegate STATIC ${SRCS})

target_link_libraries(metal_delegate)

add_subdirectory("metal")

