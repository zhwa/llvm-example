#ifndef MARCO_DIALECTS_MODELICA_MODELICA_TD
#define MARCO_DIALECTS_MODELICA_MODELICA_TD

include "TestDialect.td"

class TestDialect_Op<string mnemonic, list<Trait> traits = []> :
    Op<Test_Dialect, mnemonic, traits>;

def Test_TestOp : TestDialect_Op<"neg", []>
{
    let summary = "Test op.";

    let description = [{
        Test op.
    }];

    let arguments = (ins I64:$typedOperand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $typedOperand attr-dict `:` type($typedOperand) `->` type($result)
    }];
}

#endif // MARCO_DIALECTS_MODELICA_MODELICA_TD
