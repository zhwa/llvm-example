project(toyc-ch2simple)

add_subdirectory(include)

add_executable(${PROJECT_NAME}
  toyc.cpp
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/parser
    ${CMAKE_CURRENT_SOURCE_DIR}/mlir)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${llvm_libs}
    MLIRAnalysis
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRTransforms)

add_dependencies(${PROJECT_NAME} ToyCh2SimpleOpsIncGen)

# Copy sample code
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/codegen.toy $<TARGET_FILE_DIR:${PROJECT_NAME}>)