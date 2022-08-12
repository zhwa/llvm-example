# llvm-pass-skeleton

This example is based on sample code from https://github.com/sampsyo/llvm-pass-skeleton. A detailed tutorial can be found at https://www.cs.cornell.edu/~asampson/blog/llvm.html.

LLVM's official document is also a good reference: https://llvm.org/docs/WritingAnLLVMPass.html.

Based on an old Q&A at here: https://stackoverflow.com/a/41980033, currently Windows isn't supported. In order to use it on Windows, LLVM itself has to be rebuilt
with flag ```LLVM_EXPORT_SYMBOLS_FOR_PLUGINS``` set to ```ON```. More discussion can be found at https://discourse.llvm.org/t/how-to-create-pass-independently-on-windows/474/8.