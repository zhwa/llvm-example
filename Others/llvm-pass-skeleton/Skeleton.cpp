#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

namespace {
    struct SkeletonPass : public llvm::FunctionPass
    {
        static char ID;
        SkeletonPass() : llvm::FunctionPass(ID) {}

        virtual bool runOnFunction(llvm::Function& F)
        {
            llvm::errs() << "I saw a function called " << F.getName() << "!\n";
            return false;
        }
    };
}

char SkeletonPass::ID = 0;

// Register our class SkeletonPass, giving it a command line argument “skeleton”, and a name "Skeleton Pass".
// The last two arguments describe its behavior: if a pass walks CFG without modifying it then the third argument is set to true;
// if a pass is an analysis pass, for example dominator tree pass, then true is supplied as the fourth argument.
static llvm::RegisterPass<SkeletonPass> RegisterSkeletonClass("skeleton", "Skeleton Pass", false /* Only looks at CFG */, false /* Analysis Pass */);

// If we want to register the pass as a step of an existing pipeline, some extension points are provided,
// e.g. PassManagerBuilder::EP_EarlyAsPossible to apply our pass before any optimization,
// or PassManagerBuilder::EP_FullLinkTimeOptimizationLast to apply it after Link Time Optimizations.
static llvm::RegisterStandardPasses RegisterSkeletonPass(
    llvm::PassManagerBuilder::EP_EarlyAsPossible,
    [](const llvm::PassManagerBuilder& Builder, llvm::legacy::PassManagerBase& PM) { PM.add(new SkeletonPass()); });