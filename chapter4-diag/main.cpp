extern "C" {
#include "clang-c/Index.h"
}
#include "llvm/Support/CommandLine.h"
#include <iostream>

static llvm::cl::opt<std::string> FileName(llvm::cl::Positional,
    llvm::cl::desc("Input file"),
    llvm::cl::Required);

int main(int argc, char** argv) {
    llvm::cl::ParseCommandLineOptions(argc, argv, "Diagnostics Example");
    CXIndex index = clang_createIndex(0, 0);
    const char* args[] = {"-I/user/include", "-I."};
    auto translationUnit = clang_parseTranslationUnit(
        index, FileName.c_str(), args, 2, nullptr, 0, CXTranslationUnit_None);
    auto diagnosticCount = clang_getNumDiagnostics(translationUnit);
    for (unsigned int i = 0; i < diagnosticCount; i++) {
        auto diagnostic = clang_getDiagnostic(translationUnit, i);
        auto category = clang_getDiagnosticCategoryText(diagnostic);
        auto message = clang_getDiagnosticSpelling(diagnostic);
        auto severity = clang_getDiagnosticSeverity(diagnostic);
        auto loc = clang_getDiagnosticLocation(diagnostic);
        CXString fName;
        unsigned line = 0, col = 0;
        clang_getPresumedLocation(loc, &fName, &line, &col);
        std::cout << "Severity: " << severity
            << " File: " << clang_getCString(fName) << " Line: " << line
            << " Col: " << col << " Category: \""
            << clang_getCString(category)
            << "\" Message: " << clang_getCString(message) << std::endl;
        clang_disposeString(fName);
        clang_disposeString(message);
        clang_disposeString(category);
        clang_disposeDiagnostic(diagnostic);
    }
    clang_disposeTranslationUnit(translationUnit);
    clang_disposeIndex(index);

    return 0;
}