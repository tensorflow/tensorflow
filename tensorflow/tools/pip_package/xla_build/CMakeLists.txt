cmake_minimum_required(VERSION 3.5)

file(GLOB_RECURSE TF_RUNTIME_SRC "*.cc")
add_library(tf_xla_runtime_objects OBJECT
	${TF_RUNTIME_SRC}
)

target_include_directories(tf_xla_runtime_objects PRIVATE ../include)
target_compile_options(tf_xla_runtime_objects PRIVATE
  -ftemplate-backtrace-limit=0
  -Wno-ignored-attributes
  -Wno-deprecated-copy
  -Wno-cast-qual
  -Wno-sign-compare
)

add_library(tf_xla_runtime STATIC
  $<TARGET_OBJECTS:tf_xla_runtime_objects>
)
