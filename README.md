# llvm-example

This repo uses vcpkg to get LLVM headers and libraries. The official tutorial can be found at https://vcpkg.io/en/docs/users/selecting-library-features.html.

In this project, vcpkg is assumed to be present at D:/tools/vcpkg. Update [CMakePresets.json](./CMakePresets.json) if a different path is used.

## Set Up LLVM

Install LLVM, Clang and MLIR with:

```sh
vcpkg install llvm[mlir] --triplet=x64-windows
```

After a while, LLVM binaries would be installed at ```D:\tools\vcpkg\installed\x64-windows\tools\llvm```, and the corresponding headers and libs can be referred
with:

```json
"cacheVariables": {
  "CMAKE_TOOLCHAIN_FILE": "D:/tools/vcpkg/scripts/buildsystems/vcpkg.cmake"
},
```
in [CMakePresets.json](./CMakePresets.json).

## CMake Variables

The LLVM official doc has an instruction at https://llvm.org/docs/CMake.html. Several LLVM related variables and functions have been predefined.

No Clang version is available, but it can be easily created like

```cmake
FUNCTION(clang_map_components_to_libnames extra_clang_libs)
    FOREACH(l ${CLANG_LIBS})
        FIND_LIBRARY(LIB_${l} NAMES ${l} HINTS ${LLVM_LIBRARY_DIRS} )
        MARK_AS_ADVANCED(LIB_${l})
        LIST(APPEND clang_libs ${LIB_${l}})
    ENDFOREACH(l)

    SET(${extra_clang_libs} ${clang_libs} PARENT_SCOPE)
ENDFUNCTION()
```

## Examples

Most examples in this project are based on code pieces in *Getting Started with LLVM Core Libraries*.