#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

namespace {
    struct SkeletonPass : public llvm::FunctionPass
    {
        static char ID;
        SkeletonPass() : llvm::FunctionPass(ID) {}

        virtual bool runOnFunction(llvm::Function& F)
        {
            // Get function architecture. Corresponds to sampsyo's container branch.
            // Print function name and body.
            llvm::errs() << "I saw a function called " << F.getName() << "!\n";
            llvm::errs() << "Function body:\n";
            F.print(llvm::errs());

            // Iterate blocks and instructions.
            for (const auto& block : F)
            {
                llvm::errs() << "Basic block:\n";
                block.print(llvm::errs());

                for (const auto& instruction : block)
                {
                    llvm::errs() << "Instructions:\n";
                    instruction.print(llvm::errs(), true);
                    llvm::errs() << "\n";
                }
            }

            // Make the pass do something mildly interesting: change the code when you find them. Corresponds to mutate branch.
            // Replace the first binary operator(+, -, etc.) in every function with a multiply.
            for (auto& block : F)
            {
                for (auto& instruction : block)
                {
                    // dyn_cast will return nullptr if instruction isn't a llvm::BinaryOperator.
                    if (auto* op = llvm::dyn_cast<llvm::BinaryOperator>(&instruction))
                    {
                        // Insert at the point where the instruction 'op' appears.
                        llvm::IRBuilder<> builder(op);

                        // Make a multiple with the same operands as 'op'.
                        llvm::Value* lhs = op->getOperand(0);
                        llvm::Value* rhs = op->getOperand(1);
                        llvm::Value* mul = builder.CreateMul(lhs, rhs);

                        // Everywhere the old instruction was used as an operand, use new multiply instruction instead.
                        for (auto& user : op->uses())
                        {
                            llvm::User* u = user.getUser();
                            u->setOperand(user.getOperandNo(), mul);
                        }

                        // We modified the code.
                        return true;

                    }
                }
            }

            // Get the function to call from runtime library. Corresponds to rtlib branch.
            llvm::LLVMContext& ctx = F.getContext();
            std::vector<llvm::Type*> paramTypes = {llvm::Type::getInt32Ty(ctx)};
            llvm::Type* retType = llvm::Type::getVoidTy(ctx);
            llvm::FunctionType* logFuncType = llvm::FunctionType::get(retType, paramTypes, false);
            llvm::FunctionCallee logFunc = F.getParent()->getOrInsertFunction("logop", logFuncType);

            for (auto& block : F)
            {
                for (auto& instruction : block)
                {
                    if (auto* op = llvm::dyn_cast<llvm::BinaryOperator>(&instruction))
                    {
                        // Insert * after* 'op'.
                        llvm::IRBuilder<> builder(op);
                        builder.SetInsertPoint(&block, ++builder.GetInsertPoint());

                        // Insert a call to our function.
                        llvm::Value* args[] = {op};
                        builder.CreateCall(logFunc, args);

                        return true;
                    }
                }
            }

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