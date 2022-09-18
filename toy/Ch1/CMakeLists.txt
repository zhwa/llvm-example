project(toyc-ch1)

add_executable(${PROJECT_NAME}
  toyc.cpp
  parser/AST.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/parser)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${llvm_libs}
    MLIRSupport)

# Copy sample code
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/ast.toy $<TARGET_FILE_DIR:${PROJECT_NAME}>)