add_subdirectory(lib)

set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

set(LLVM_OPTIONAL_SOURCES Conversion.cpp Example.cpp)

add_llvm_example(linalg-conversion-1
  Conversion.cpp
  )

add_llvm_example(linalg-example-1
  Example.cpp
  )

target_link_libraries(linalg-example-1
  PRIVATE
    Linalg1DialectConstruction
    Linalg1
    )

whole_archive_link(linalg-example-1
  MLIRStandardOps
  )

target_link_libraries(linalg-conversion-1
  PRIVATE
    Linalg1DialectConstruction
    Linalg1
    )

whole_archive_link(linalg-conversion-1
  MLIRStandardOps
  )
