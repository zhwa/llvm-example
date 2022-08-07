project(ch5-pass)

add_library(${PROJECT_NAME} MODULE main.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE ${LLVM_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${llvm_libs})