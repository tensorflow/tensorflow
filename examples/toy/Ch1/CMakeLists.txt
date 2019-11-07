set(LLVM_LINK_COMPONENTS
  Support
  )

add_toy_chapter(toyc-ch1
  toyc.cpp
  parser/AST.cpp
  )
include_directories(include/)
target_link_libraries(toyc-ch1
  PRIVATE
    MLIRSupport)
