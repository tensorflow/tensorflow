include_directories(Linalg1/)
include_directories(Linalg1/include/)
include_directories(Linalg2/include/)
include_directories(Linalg3/include/)
include_directories(Linalg4/include/)

add_custom_target(Linalg)
set_target_properties(Linalg PROPERTIES FOLDER Examples)
add_dependencies(Linalg
  linalg-conversion-3
  linalg-example-2
  linalg-example-3
  linalg-example-4
  linalg-execution-3
  )

add_subdirectory(Linalg1)
add_subdirectory(Linalg2)
add_subdirectory(Linalg3)
add_subdirectory(Linalg4)
