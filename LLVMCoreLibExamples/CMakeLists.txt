add_custom_target(LLVMCoreLibExamples)
set_target_properties(LLVMCoreLibExamples PROPERTIES FOLDER Examples)

add_subdirectory(chapter3)
add_subdirectory(chapter4-diag)
add_subdirectory(chapter4-ast)
add_subdirectory(chapter4-full)
add_subdirectory(chapter5-ir)
add_subdirectory(chapter5-pass)