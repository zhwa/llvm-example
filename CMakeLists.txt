cmake_minimum_required(VERSION 3.20)
project(llvm-example)

set(USE_FOLDERS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
cmake_path(APPEND CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" "bin") # Use the same output directory for the whole project

# VS specific flags.
add_compile_options(/sdl) # Enable Additional Security Checks
add_compile_options(/EHsc) # Enable exceptions
add_compile_options(/W4) # Warning level
add_compile_options(/guard:cf) # Enable Control Flow Guard

# Get LLVM and Clang
find_package(LLVM CONFIG REQUIRED)
find_package(Clang REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
# Use `add_llvm_...` functions.
include(AddLLVM)
# Set compiler flags.
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})

add_definitions(${LLVM_DEFINITIONS})
add_definitions(${CLANG_DEFINITIONS})

# Set up the LLVM and Clang libraries.
llvm_map_components_to_libnames(llvm_libs Analysis Support Core IRReader ExecutionEngine Object OrcJIT native InstCombine RuntimeDyld ScalarOpts)
SET(clang_libs libclang clangFrontendTool clangFrontend clangDriver clangSerialization clangCodeGen clangParse clangSema clangRewriteFrontend clangRewrite clangAnalysis clangEdit clangAST clangLex clangBasic clangTooling)

# Examples from <<Getting Started with LLVM Core Libraries>>
add_subdirectory(LLVMCoreLibExamples)

# Official example: Kaleidoscope
add_subdirectory(Kaleidoscope)