#include <llvm/IR/IRBuilder.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Support/ToolOutputFile.h>

// CreateEntryBlockAlloca - Create an alloca instruction in the entry block of the function. This is used for mutable variables etc.
static llvm::AllocaInst* CreateEntryBlockAlloca(llvm::Function* TheFunction, llvm::LLVMContext& context, const std::string& VarName)
{
	llvm::IRBuilder<> TmpB(&TheFunction->getEntryBlock(), TheFunction->getEntryBlock().begin());
	return TmpB.CreateAlloca(llvm::Type::getInt32Ty(context), nullptr, VarName);
}

int main()
{
	llvm::LLVMContext context;
	const llvm::StringRef input_file = "sum.ll";
	const llvm::StringRef output_file = "./sum.bc";
	const llvm::StringRef layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128";
	const llvm::StringRef triple = "x86_64-PC-Win32";

	llvm::Module mod(input_file, context);
	mod.setDataLayout(layout);
	mod.setTargetTriple(triple);

	llvm::SmallVector<llvm::Type*, 2> FuncTyArgs;
	FuncTyArgs.push_back(llvm::IntegerType::get(mod.getContext(), 32));
	FuncTyArgs.push_back(llvm::IntegerType::get(mod.getContext(), 32));
	auto* FuncTy = llvm::FunctionType::get(llvm::IntegerType::get(mod.getContext(), 32), FuncTyArgs, false);

	auto* funcSum = llvm::Function::Create(FuncTy, llvm::GlobalValue::ExternalLinkage, "sum", mod);
	funcSum->setCallingConv(llvm::CallingConv::C);

	auto args = funcSum->arg_begin();
	auto* int32_a = args++;
	int32_a->setName("a");
	auto* int32_b = args++;
	int32_b->setName("b");

	auto* labelEntry = llvm::BasicBlock::Create(mod.getContext(), "entry", funcSum, 0);

	auto* ptrA = CreateEntryBlockAlloca(funcSum, context, "a.addr");
	ptrA->setAlignment(llvm::Align(4));
	auto* ptrB = CreateEntryBlockAlloca(funcSum, context, "b.addr");
	ptrB->setAlignment(llvm::Align(4));

	auto* st0 = new llvm::StoreInst(int32_a, ptrA, false, labelEntry);
	st0->setAlignment(llvm::Align(4));
	auto* st1 = new llvm::StoreInst(int32_b, ptrB, false, labelEntry);
	st1->setAlignment(llvm::Align(4));

	auto* ld0 = new llvm::LoadInst(llvm::IntegerType::get(mod.getContext(), 32), ptrA, "", false, labelEntry);
	ld0->setAlignment(llvm::Align(4));
	auto* ld1 = new llvm::LoadInst(llvm::IntegerType::get(mod.getContext(), 32), ptrB, "", false, labelEntry);
	ld1->setAlignment(llvm::Align(4));
	auto* addRes = llvm::BinaryOperator::Create(llvm::Instruction::Add, ld0, ld1, "add", labelEntry);
	llvm::ReturnInst::Create(mod.getContext(), addRes, labelEntry);

	llvm::verifyModule(mod);
	std::error_code err;
	llvm::sys::fs::OpenFlags flag{};
	llvm::ToolOutputFile output(output_file, err, flag);
	llvm::WriteBitcodeToFile(mod, output.os());
	output.keep();
	return 0;
}