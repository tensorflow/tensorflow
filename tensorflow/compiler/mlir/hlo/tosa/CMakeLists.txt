set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib

        MhloRegisterDialects
        TosaRegisterDialects
        AllMhloPasses
        MLIRHLOAnalysis
        )
add_llvm_executable(mhlo-tosa-opt mhlo_tosa_opt.cpp
  DEPENDS
        MLIRMhloPassIncGen
)
llvm_update_compile_flags(mhlo-tosa-opt)
target_link_libraries(mhlo-tosa-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(mhlo-tosa-opt)

