# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# We need to know an absolute path to our repo root to do things like referencing ./LICENSE.txt file.
set(AZ_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/..")

macro(az_vcpkg_integrate)
  message("Vcpkg integrate step.")

  # AUTO CMAKE_TOOLCHAIN_FILE:
  #   User can call `cmake -DCMAKE_TOOLCHAIN_FILE="path_to_the_toolchain"` as the most specific scenario.
  #   As the last alternative (default case), Azure SDK will automatically clone VCPKG folder and set toolchain from there.
  if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    message("CMAKE_TOOLCHAIN_FILE is not defined. Define it for the user.")
    # Set AZURE_SDK_DISABLE_AUTO_VCPKG env var to avoid Azure SDK from cloning and setting VCPKG automatically
    # This option delegate package's dependencies installation to user.
    if(NOT DEFINED ENV{AZURE_SDK_DISABLE_AUTO_VCPKG})
      message("AZURE_SDK_DISABLE_AUTO_VCPKG is not defined. Fetch a local copy of vcpkg.")
      # GET VCPKG FROM SOURCE
      include(FetchContent)
      FetchContent_Declare(vcpkg
          GIT_REPOSITORY      https://github.com/microsoft/vcpkg.git)
      FetchContent_GetProperties(vcpkg)
      # make sure to pull vcpkg only once.
      if(NOT vcpkg_POPULATED) 
          FetchContent_Populate(vcpkg)
      endif()
      # use the vcpkg source path 
      set(CMAKE_TOOLCHAIN_FILE "${vcpkg_SOURCE_DIR}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
    endif()
  endif()

  # enable triplet customization
  if(DEFINED ENV{VCPKG_DEFAULT_TRIPLET} AND NOT DEFINED VCPKG_TARGET_TRIPLET)
    set(VCPKG_TARGET_TRIPLET "$ENV{VCPKG_DEFAULT_TRIPLET}" CACHE STRING "")
  endif()
  message("Vcpkg integrate step - DONE.")
endmacro()