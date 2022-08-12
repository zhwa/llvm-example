project(SkeletonPass)

add_library(${PROJECT_NAME} SHARED Skeleton.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE ${llvm_libs})