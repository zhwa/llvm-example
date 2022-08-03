#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Host.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Parse/Parser.h"
#include "clang/Parse/ParseAST.h"
#include <iostream>
#include <memory>

static llvm::cl::opt<std::string> FileName(llvm::cl::Positional,
    llvm::cl::desc("Input file"),
    llvm::cl::Required);

int main(int argc, char** argv) {
    llvm::cl::ParseCommandLineOptions(argc, argv, "My simple front end\n");
    clang::CompilerInstance CI;
    clang::DiagnosticOptions diagnosticOptions;
    CI.createDiagnostics();

    std::shared_ptr<clang::TargetOptions> PTO = std::make_shared<clang::TargetOptions>();
    PTO->Triple = llvm::sys::getDefaultTargetTriple();
    auto* PTI = clang::TargetInfo::CreateTargetInfo(CI.getDiagnostics(), PTO);
    CI.setTarget(PTI);
    CI.createFileManager();
    CI.createSourceManager(CI.getFileManager());
    CI.createPreprocessor(clang::TranslationUnitKind::TU_Complete);
    CI.getPreprocessorOpts().UsePredefines = false;

    // Add all of the include directories to the preprocessor.
    auto& headerSearchOption = CI.getHeaderSearchOpts();
    headerSearchOption.AddPath("C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.32.31326\\include", clang::frontend::IncludeDirGroup::CXXSystem, true, false);
    clang::InitializePreprocessor(CI.getPreprocessor(), CI.getPreprocessorOpts(), CI.getPCHContainerReader(), CI.getFrontendOpts());

    auto astConsumer = clang::CreateASTPrinter(nullptr, "");
    CI.setASTConsumer(std::move(astConsumer));
    CI.createASTContext();
    CI.createSema(clang::TranslationUnitKind::TU_Complete, nullptr);
    const auto pFile = CI.getFileManager().getFile(FileName);
    if (!pFile) {
        std::cerr << "File not found: " << FileName << std::endl;
        return 1;
    }

    // create an id and set the id: suggestion from https://stackoverflow.com/a/40429976.
    const auto id = CI.getSourceManager().createFileID(pFile.get(), {}, clang::SrcMgr::CharacteristicKind::C_User);
    CI.getSourceManager().setMainFileID(id);
    CI.getDiagnosticClient().BeginSourceFile(CI.getLangOpts(), nullptr);
    clang::ParseAST(CI.getSema());

    // Print AST statistics
    CI.getASTContext().PrintStats();
    CI.getASTContext().Idents.PrintStats();

    return 0;
}