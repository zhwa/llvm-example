project(sqlgen)

add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

set(_SOURCE_FILES
    Error.cpp
    sqlgen.cpp
    SQLEmitter.cpp
    SQLInsertEmitter.cpp
    SQLQueryEmitter.cpp
    SQLTableEmitter.cpp
    )

set(LLVM_LINK_COMPONENTS
  Support
  TableGen
  )

add_llvm_executable(sqlgen
  ${_SOURCE_FILES})

set(FILECHECK_PATH "" CACHE FILEPATH "Path to FileCheck.")

if (NOT FILECHECK_PATH)
  # Try to find FileCheck from LLVM first
  find_program(FILECHECK_EXE
    FileCheck
    NO_CACHE)

  if (NOT FILECHECK_EXE)
    # Then try to find the python version, filecheck
    find_program(FILECHECK_EXE
      filecheck
      NO_CACHE)
  endif()

  if (FILECHECK_EXE)
    message(STATUS "Using ${FILECHECK_EXE}")
    set(FILECHECK_PATH ${FILECHECK_EXE})
  endif()
endif()

add_subdirectory(test)

file(GLOB test_files ${CMAKE_CURRENT_SOURCE_DIR}/test/*.td)
file(COPY ${test_files} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE})