add_custom_target(Kaleidoscope)
set_target_properties(Kaleidoscope PROPERTIES FOLDER Examples)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(BuildingAJIT)
add_subdirectory(Chapter2)
add_subdirectory(Chapter3)
add_subdirectory(Chapter4)
add_subdirectory(Chapter5)
add_subdirectory(Chapter6)
add_subdirectory(Chapter7)
add_subdirectory(Chapter8)
add_subdirectory(Chapter9)