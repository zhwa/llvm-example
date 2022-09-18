add_mlir_dialect(Test test)

set(LLVM_TARGET_DEFINITIONS Test.td)

include_directories(${PROJECT_BINARY_DIR})

add_mlir_dialect_library(MLIRTest
        TestDialect.cpp
        Ops.cpp

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport)