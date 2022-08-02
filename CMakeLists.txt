cmake_minimum_required(VERSION 3.20)
project(llvm-example)

set(USE_FOLDERS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
cmake_path(APPEND CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" "bin") # Use the same output directory for the whole project

set(CMAKE_TOOLCHAIN_FILE D:/tools/vcpkg/scripts/buildsystems/vcpkg.cmake)

# VS specific flags
add_compile_options(/sdl) # Enable Additional Security Checks
add_compile_options(/EHsc) # Enable exceptions
add_compile_options(/W4) # Warning level
add_compile_options(/guard:cf) # Enable Control Flow Guard

# Set up LLVM env
find_package(LLVM CONFIG REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(HandleLLVMOptions)
add_definitions(${LLVM_DEFINITIONS})

# Find the libraries that correspond to the LLVM components that we wish to use
llvm_map_components_to_libnames(llvm_libs Support Core IRReader ExecutionEngine Object OrcJIT native)


add_subdirectory(chapter3)