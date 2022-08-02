#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/Error.h"
#include <iostream>
#include <memory>

static llvm::cl::opt<std::string> FileName(llvm::cl::Positional, llvm::cl::desc("Bitcode file"), llvm::cl::Required);

int main(int argc, char** argv)
{
    llvm::cl::ParseCommandLineOptions(argc, argv, "LLVM hello world.\n");
    llvm::LLVMContext context;

    auto mb = llvm::MemoryBuffer::getFile(FileName);
    if (!mb)
    {
        std::cerr << "Failed to read the input file.\n";
        return -1;
    }
    llvm::MemoryBufferRef buffer{ *mb.get().get() };
    auto m = llvm::parseBitcodeFile(buffer, context);
    if (!m)
    {
        std::cerr << "Failed to parse the input file.'n";
        return -1;
    }

    llvm::raw_os_ostream O(std::cout);
    for (auto i = m.get()->getFunctionList().begin(); i != m.get()->getFunctionList().end(); ++i)
    {
        if (!i->isDeclaration())
        {
            O << i->getName() << " has " << i->size() << " basic block(s).\n";
        }
    }
}