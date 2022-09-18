project(CrashMapping)

add_subdirectory(dialect)

add_executable(${PROJECT_NAME} main.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/dialect)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${llvm_libs}
    ${dialect_libs}
    ${conversion_libs}
    MLIRTest
    MLIRAnalysis
    MLIRCallInterfaces
    MLIRCastInterfaces
    MLIRExecutionEngine
    MLIRIR
    MLIRLLVMCommonConversion
    MLIRLLVMToLLVMIRTranslation
    MLIRMemRef
    MLIRParser
    MLIRPass
    MLIRSideEffectInterfaces
    MLIRTargetLLVMIRExport
    MLIRTransforms)