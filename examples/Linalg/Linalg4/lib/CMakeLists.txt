add_llvm_library(Linalg4
  Transforms.cpp
  )

target_link_libraries(Linalg4
  PUBLIC
    MLIRAnalysis
    MLIRDialect
    MLIREDSC
    MLIRIR
    MLIRLLVMIR
    MLIRParser
    MLIRPass
    MLIRTransforms
    Linalg3
    )
