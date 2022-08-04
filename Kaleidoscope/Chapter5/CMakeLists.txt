project(Kaleidoscope-Ch5)

add_executable(${PROJECT_NAME} toy.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE ${llvm_libs})