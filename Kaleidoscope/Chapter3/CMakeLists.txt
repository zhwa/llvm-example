project(Kaleidoscope-Ch3)

add_executable(${PROJECT_NAME} toy.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE ${llvm_libs})