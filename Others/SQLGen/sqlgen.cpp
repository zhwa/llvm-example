#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/WithColor.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "Error.h"
#include <iostream>

using namespace llvm;

namespace {
    enum ActionType {
        PrintRecords,
        PrintDetailedRecords,
        EmitSQL
    };
} // end anonymous namespace

static cl::opt<ActionType> Action(
    cl::desc("Actions to perform"),
    cl::values(
        clEnumValN(PrintRecords, "print-records", ""),
        clEnumValN(PrintDetailedRecords, "print-detailed-records", ""),
        clEnumValN(EmitSQL, "emit-sql", "")
    ),
    cl::init(EmitSQL));

Error emitSQL(raw_ostream& OS, RecordKeeper& Records);

bool SQLGenMain(raw_ostream& OS, RecordKeeper& Records)
{
    switch (Action)
    {
    case PrintRecords:
        OS << Records;
        break;
    case EmitSQL:
        if (auto E = emitSQL(OS, Records))
        {
            handleAllErrors(std::move(E),
                [](const SMLocError& E) { llvm::PrintError(E.Loc.getPointer(), E.getMessage()); },
                [](const ErrorInfoBase& E) { E.log(WithColor::error()); errs() << "\n"; });
            return true;
        }
        break;
    default:
        return true;
    }

    return false;
}

int main(int argc, char** argv)
{
    // Add an ugly default for convenient debugging: sqlgen.exe --emit-sql InsertRows.td
    std::vector<char*> params = {const_cast<char*>("sqlgen.exe"), const_cast<char*>("--emit-sql"), const_cast<char*>("InsertRows.td")};
    if (argc < 2)
    {
        argc = 3;
        argv = params.data();
    }
    cl::ParseCommandLineOptions(argc, argv);
    return llvm::TableGenMain(argv[0], &SQLGenMain);
}