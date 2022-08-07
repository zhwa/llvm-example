#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace
{
	class FnArgCnt : public llvm::FunctionPass
	{
	public:
		static char ID;
		FnArgCnt() : llvm::FunctionPass(ID) {}

		virtual bool runOnFunction(llvm::Function& F)
		{
			llvm::errs() << "FnArgCnt --- " << F.getName() << ": ";
			auto args = F.args();
			llvm::errs() << std::distance(args.begin(), args.end()) << '\n';
			return false;
		}
	};
}

char FnArgCnt::ID = 0;
static llvm::RegisterPass<FnArgCnt> X("fnargcnt", "Function Argument Count Pass", false, false);