add_triton_library(TritonProtonToLLVM
    RecordOpToLLVM.cpp

    LINK_LIBS PUBLIC
    ProtonIR
)
