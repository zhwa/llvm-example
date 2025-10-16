project(linalg-gemm-example)

# This example demonstrates MLIR's Linalg dialect for GEMM operations

add_executable(${PROJECT_NAME}
  gemm_example.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}
)

# Get all MLIR dialect libraries
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${llvm_libs}
    ${dialect_libs}
    MLIRIR
    MLIRAnalysis
    MLIRArithDialect
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRMemRefDialect
    MLIRParser
    MLIRPass
    MLIRTensorDialect
    MLIRTransforms
    MLIRSupport
)
