# Chapter 4: Transformations and Rewriting

> *"Perfection is achieved, not when there is nothing more to add, but when there is nothing left to take away." - Antoine de Saint-Exupéry*

## Introduction

In Chapter 3, we learned how to *create* MLIR operations using TableGen. Now we'll learn how to *transform* them—how to make code better through optimization and simplification.

This chapter introduces one of MLIR's most powerful features: **pattern-based rewriting**. You'll learn to write transformations in two ways:

1. **C++ patterns** - Imperative, procedural, full control
2. **Declarative Rewrite Rules (DRR)** - Declarative, concise, elegant

By the end of this chapter, you'll be able to:
- Write pattern matchers that find code to optimize
- Implement rewrite rules that improve code
- Use both C++ and TableGen for transformations
- Understand the canonicalization framework
- See real optimizations in action

Let's make our Toy code faster and cleaner!

---

## 4.1 Why Transform IR?

### What Is a Transformation?

A transformation takes an IR and produces an equivalent but improved IR. "Improved" might mean:

- **Faster**: Fewer operations, better algorithms
- **Smaller**: Less memory usage, smaller code size
- **Simpler**: Easier to analyze or further optimize
- **More canonical**: Standard form that's easier to recognize

### Types of Transformations

**1. Optimizations**
Make code run faster:
```mlir
// Before
%0 = toy.constant dense<2.0> : tensor<f64>
%1 = toy.mul %x, %0 : tensor<2x3xf64>

// After (constant folding if possible)
%1 = toy.mul %x, %const : tensor<2x3xf64>
```

**2. Simplifications**
Remove redundancy:
```mlir
// Before
%0 = toy.transpose(%x) to tensor<3x2xf64>
%1 = toy.transpose(%0) to tensor<2x3xf64>

// After
// %1 is just %x (double transpose cancels out)
```

**3. Canonicalization**
Convert to standard form:
```mlir
// Before
%0 = toy.reshape(%x) to tensor<2x3xf64>
%1 = toy.reshape(%0) to tensor<2x3xf64>

// After (nested reshapes merged)
%0 = toy.reshape(%x) to tensor<2x3xf64>
```

**4. Lowering**
Convert to lower abstraction level (we'll do this in Chapters 6-7):
```mlir
// Before (high-level)
%0 = toy.mul %a, %b : tensor<2x3xf64>

// After (lower-level loops)
affine.for %i = 0 to 2 {
  affine.for %j = 0 to 3 {
    // element-wise multiply
  }
}
```

### The Cost of Not Optimizing

Without transformations, code generated from high-level constructs can be inefficient:

```toy
def foo(a) {
  var b = transpose(a);
  var c = transpose(b);
  return c;
}
```

**Without optimization:**
- Two transpose operations (memory allocations, data movement)
- Runtime cost: O(n²) for n×n matrix

**With optimization:**
- Recognize that `transpose(transpose(x))` = `x`
- Zero transpose operations
- Runtime cost: O(1)

---

## 4.2 Pattern Matching in MLIR

### The Pattern Matching Framework

MLIR provides infrastructure for finding and transforming IR patterns. Think of it like regular expressions, but for code instead of text.

**Pattern = Matcher + Rewriter**

```cpp
class MyPattern : public OpRewritePattern<MyOp> {
  LogicalResult matchAndRewrite(MyOp op, PatternRewriter &rewriter) const override {
    // 1. Match: Check if this op matches our pattern
    if (!matches(op))
      return failure();
    
    // 2. Rewrite: Transform the op
    rewriter.replaceOp(op, newOp);
    return success();
  }
};
```

### How Pattern Matching Works

The pattern matcher walks through your IR looking for operations that match your patterns:

```
IR: %0 = op1  →  [Check Pattern A] → No match
    %1 = op2  →  [Check Pattern A] → Match! → Apply transformation
    %2 = op3  →  [Check Pattern A] → No match
    %3 = op4  →  [Check Pattern B] → Match! → Apply transformation
```

### Pattern Benefits

Patterns have a **benefit** score that determines priority:

```cpp
SimplifyRedundantTranspose(MLIRContext *context)
    : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}
```

Higher benefit = higher priority. MLIR tries patterns in order of benefit.

**Why benefits matter:**
- Multiple patterns might match the same op
- Higher benefit patterns tried first
- Allows expressing "prefer this transformation"

### Greedy Rewriting

MLIR uses **greedy rewriting**: repeatedly apply patterns until no more match:

```
Initial IR
    ↓
Apply pattern 1 → Changes made
    ↓
Apply pattern 2 → Changes made
    ↓
Try pattern 1 again → No match
Try pattern 2 again → No match
    ↓
Done (fixed point reached)
```

This continues until the IR stops changing (reaches a **fixed point**).

---

## 4.3 Rewrite Patterns in C++

Let's write our first transformation in C++.

### Example: Simplify Redundant Transpose

The pattern: `transpose(transpose(x))` → `x`

**Why this optimization matters:**
- Transpose creates a temporary tensor
- Two transposes = wasted computation
- Mathematically, they cancel out

### Step 1: Inherit from OpRewritePattern

```cpp
struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
  SimplifyRedundantTranspose(mlir::MLIRContext *context)
      : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(TransposeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Implementation goes here
  }
};
```

**Key elements:**
- Inherit from `OpRewritePattern<OpType>`
- Constructor specifies benefit
- Override `matchAndRewrite()` method

### Step 2: Implement Match Logic

```cpp
mlir::LogicalResult
matchAndRewrite(TransposeOp op,
                mlir::PatternRewriter &rewriter) const override {
  // Get the input to this transpose
  mlir::Value transposeInput = op.getOperand();
  
  // Check if the input is defined by another transpose
  TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();

  // If input is not a transpose, this pattern doesn't match
  if (!transposeInputOp)
    return failure();

  // We found transpose(transpose(x))!
  // Continue to rewrite...
}
```

**Step-by-step:**
1. Get the operand (input) of current transpose
2. Check if that input is produced by another transpose
3. If not, pattern doesn't match → return `failure()`
4. If yes, we have a match → continue to rewrite

### Step 3: Implement Rewrite Logic

```cpp
  // We have transpose(transpose(x))
  // Replace the outer transpose with x
  rewriter.replaceOp(op, {transposeInputOp.getOperand()});
  return success();
}
```

**What `replaceOp` does:**
- Removes `op` from the IR
- Replaces all uses of `op`'s result with the new value
- Updates all references automatically

### Complete Implementation

Here's the full pattern from `toy/Ch3/mlir/ToyCombine.cpp`:

```cpp
/// This is an example of a C++ rewrite pattern for the TransposeOp.
/// It optimizes: transpose(transpose(x)) -> x
struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
  /// Register this pattern to match every toy.transpose in the IR.
  /// The "benefit" is used to order patterns by profitability.
  SimplifyRedundantTranspose(mlir::MLIRContext *context)
      : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}

  /// This method attempts to match and rewrite the pattern.
  mlir::LogicalResult
  matchAndRewrite(TransposeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Look through the input of the current transpose
    mlir::Value transposeInput = op.getOperand();
    TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();

    // Input defined by another transpose? If not, no match.
    if (!transposeInputOp)
      return failure();

    // Otherwise, we have a redundant transpose. Use the rewriter.
    rewriter.replaceOp(op, {transposeInputOp.getOperand()});
    return success();
  }
};
```

### Visualizing the Transformation

**Before:**
```mlir
%0 = ... : tensor<2x3xf64>
%1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
%2 = toy.transpose(%1 : tensor<3x2xf64>) to tensor<2x3xf64>
%3 = toy.print %2 : tensor<2x3xf64>
```

**After applying SimplifyRedundantTranspose:**
```mlir
%0 = ... : tensor<2x3xf64>
// %1 and %2 removed
%3 = toy.print %0 : tensor<2x3xf64>
```

The pattern recognized that `%2 = transpose(transpose(%0))` and replaced all uses of `%2` with `%0`.

### The PatternRewriter API

The `PatternRewriter` provides methods for modifying IR:

```cpp
// Replace an operation
rewriter.replaceOp(oldOp, newValues);

// Erase an operation
rewriter.eraseOp(op);

// Replace with a new operation
auto newOp = rewriter.create<NewOp>(location, operands);
rewriter.replaceOp(oldOp, newOp.getResults());

// Modify in-place
rewriter.updateRootInPlace(op, [&] {
  op.setOperand(0, newValue);
});
```

---

## 4.4 Declarative Rewrite Rules (DRR)

Writing patterns in C++ is powerful but verbose. For simple patterns, MLIR offers **Declarative Rewrite Rules (DRR)** - pattern matching in TableGen!

### Why DRR?

Compare the same transformation:

**C++ version (10+ lines):**
```cpp
struct SimplifyRedundantTranspose : public OpRewritePattern<TransposeOp> {
  SimplifyRedundantTranspose(MLIRContext *context)
      : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}
  
  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter &rewriter) const override {
    Value transposeInput = op.getOperand();
    TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();
    if (!transposeInputOp)
      return failure();
    rewriter.replaceOp(op, {transposeInputOp.getOperand()});
    return success();
  }
};
```

**DRR version (1 line!):**
```tablegen
def : Pat<(TransposeOp (TransposeOp $arg)), (replaceWithValue $arg)>;
```

**Benefits of DRR:**
- ✅ Concise (often 1-5 lines vs 10-50 lines)
- ✅ Declarative (what to match, not how)
- ✅ Less error-prone
- ✅ Easier to read and maintain
- ✅ Automatically generates C++ code

**When to use C++ patterns:**
- Complex matching logic
- Need to compute new values
- Require control flow
- Access runtime information

**When to use DRR:**
- Simple structural patterns
- Direct replacements
- Common transformations

---

## 4.5 DRR Syntax and Patterns

Let's learn DRR step by step.

### Basic DRR Structure

```tablegen
def PatternName : Pat<
  (SourcePattern),    // What to match
  (ResultPattern)     // What to replace with
>;
```

Or more concisely:
```tablegen
def : Pat<(SourcePattern), (ResultPattern)>;
```

The `def :` means "anonymous definition" (no name needed).

### Example 1: Reshape Optimization

**Pattern:** `reshape(reshape(x))` → `reshape(x)`

```tablegen
def ReshapeReshapeOptPattern : Pat<
  (ReshapeOp(ReshapeOp $arg)),
  (ReshapeOp $arg)
>;
```

**Breaking it down:**

```tablegen
Pat<
  (ReshapeOp(ReshapeOp $arg)),  // Match: outer reshape of inner reshape
  //        ^^^^^^^^^^^^ inner reshape
  //^^^^^^^^^^^^^^^^^^^^ outer reshape
  
  (ReshapeOp $arg)              // Replace: single reshape with original arg
  //         ^^^^ the input to the inner reshape
>
```

**Variables:**
- `$arg` is a variable that captures the input to the inner reshape
- Same variable in result pattern uses the captured value

**Generated MLIR transformation:**
```mlir
// Before
%0 = ... : tensor<6xf64>
%1 = toy.reshape(%0 : tensor<6xf64>) to tensor<2x3xf64>
%2 = toy.reshape(%1 : tensor<2x3xf64>) to tensor<2x3xf64>

// After
%0 = ... : tensor<6xf64>
%1 = toy.reshape(%0 : tensor<6xf64>) to tensor<2x3xf64>
```

### Example 2: Using Native Code Calls

Sometimes you need to compute something. DRR allows **native code calls** - inline C++:

```tablegen
// Define a helper function
def ReshapeConstant :
  NativeCodeCall<"$0.reshape(::llvm::cast<ShapedType>($1.getType()))">;

// Use it in a pattern
def FoldConstantReshapeOptPattern : Pat<
  (ReshapeOp:$res (ConstantOp $arg)),
  (ConstantOp (ReshapeConstant $arg, $res))
>;
```

**Breaking it down:**

```tablegen
// Match: reshape of a constant
(ReshapeOp:$res (ConstantOp $arg))
//        ^^^^                    capture the reshape op as $res
//                          ^^^^  capture the constant value as $arg

// Replace: constant with reshaped value
(ConstantOp (ReshapeConstant $arg, $res))
//          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ call C++ code to reshape the constant
```

**The NativeCodeCall:**
```tablegen
NativeCodeCall<"$0.reshape(::llvm::cast<ShapedType>($1.getType()))">
//             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//             C++ code with $0 and $1 as placeholders
```

- `$0` = first argument (the constant attribute)
- `$1` = second argument (the reshape op)
- Returns a new reshaped constant attribute

**What this does:**
```mlir
// Before
%0 = toy.constant dense<[1, 2, 3, 4, 5, 6]> : tensor<6xf64>
%1 = toy.reshape(%0 : tensor<6xf64>) to tensor<2x3xf64>

// After (constant folded with reshape)
%1 = toy.constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xf64>
```

The reshape is eliminated by reshaping the constant data at compile time!

### Example 3: Patterns with Constraints

Sometimes transformations are only valid under certain conditions:

```tablegen
// Define a constraint
def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;

// Use it in a pattern
def RedundantReshapeOptPattern : Pat<
  (ReshapeOp:$res $arg),
  (replaceWithValue $arg),
  [(TypesAreIdentical $res, $arg)]  // Only apply if types match
>;
```

**Breaking it down:**

```tablegen
Pat<
  (ReshapeOp:$res $arg),           // Match: any reshape
  (replaceWithValue $arg),         // Replace: with the input value
  [(TypesAreIdentical $res, $arg)] // Constraint: only if types match
>
```

**The Constraint:**
```tablegen
Constraint<CPred<"$0.getType() == $1.getType()">>
//         ^^^^^ "C Predicate" - C++ boolean expression
```

**What this does:**
```mlir
// Before (types are identical)
%0 = ... : tensor<2x3xf64>
%1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>  // No-op!

// After (reshape removed)
%0 = ... : tensor<2x3xf64>
// %1 is replaced with %0
```

If types don't match, the pattern doesn't apply (it's actually doing something).

### The Complete DRR File

Here's `toy/Ch3/mlir/ToyCombine.td`:

```tablegen
#ifndef TOY_COMBINE
#define TOY_COMBINE

include "mlir/IR/PatternBase.td"
include "toy/Ops.td"

//===----------------------------------------------------------------------===//
// Basic Pattern-Match and Rewrite
//===----------------------------------------------------------------------===//

// Reshape(Reshape(x)) = Reshape(x)
def ReshapeReshapeOptPattern : Pat<
  (ReshapeOp(ReshapeOp $arg)),
  (ReshapeOp $arg)
>;

//===----------------------------------------------------------------------===//
// Pattern-Match and Rewrite using Native Code Call
//===----------------------------------------------------------------------===//

// Reshape(Constant(x)) = x'
def ReshapeConstant :
  NativeCodeCall<"$0.reshape(::llvm::cast<ShapedType>($1.getType()))">;
  
def FoldConstantReshapeOptPattern : Pat<
  (ReshapeOp:$res (ConstantOp $arg)),
  (ConstantOp (ReshapeConstant $arg, $res))
>;

//===----------------------------------------------------------------------===//
// Pattern-Match and Rewrite with Constraints
//===----------------------------------------------------------------------===//

// Reshape(x) = x, where input and output shapes are identical
def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;

def RedundantReshapeOptPattern : Pat<
  (ReshapeOp:$res $arg),
  (replaceWithValue $arg),
  [(TypesAreIdentical $res, $arg)]
>;

#endif // TOY_COMBINE
```

**Three patterns, three techniques:**
1. Simple structural matching
2. Native code calls for computation
3. Constraints for conditional rewriting

---

## 4.6 Canonicalization Framework

### What Is Canonicalization?

**Canonicalization** means converting to a standard ("canonical") form. It's like grammar rules for code.

**Example:**
```mlir
// Multiple non-canonical forms
%a = toy.add %x, %y
%b = toy.add %y, %x   // Same thing, different order
%c = toy.add %x, %x   // Could be simplified differently

// Canonical form (consistent)
%a = toy.add %x, %y   // Always smaller SSA value first
```

### Why Canonicalize?

1. **Easier to recognize patterns**: If code is in standard form, patterns match more reliably
2. **Enable other optimizations**: Canonical form reveals optimization opportunities
3. **Simplify analysis**: Consistent structure makes reasoning easier

### Registering Canonicalization Patterns

MLIR has built-in support for canonicalization. You register patterns with operations:

```cpp
/// Register patterns for TransposeOp
void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  // Add the C++ pattern
  results.add<SimplifyRedundantTranspose>(context);
}

/// Register patterns for ReshapeOp
void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  // Add the DRR patterns (generated from TableGen)
  results.add<ReshapeReshapeOptPattern,
              RedundantReshapeOptPattern,
              FoldConstantReshapeOptPattern>(context);
}
```

**How this works:**
1. Each operation can have canonicalization patterns
2. Patterns are registered via `getCanonicalizationPatterns()`
3. The canonicalizer pass collects all patterns
4. Applies them in a greedy fashion

### The Canonicalizer Pass

MLIR provides a built-in **canonicalizer pass** that:
1. Collects all registered canonicalization patterns
2. Applies them repeatedly until nothing changes
3. Guarantees reaching a fixed point

**Using the canonicalizer:**

```cpp
mlir::PassManager pm(module.get()->getName());

// Add the canonicalizer pass
pm.addNestedPass<mlir::toy::FuncOp>(mlir::createCanonicalizerPass());

// Run the pass pipeline
if (mlir::failed(pm.run(*module)))
  return failure();
```

**What happens:**
```
Initial MLIR
    ↓
[Canonicalizer Pass]
    ↓
Collect all patterns from all operations
    ↓
Apply patterns greedily
    ↓
Repeat until fixed point
    ↓
Optimized MLIR
```

---

## 4.7 Complete Example: Seeing Transformations

Let's trace a complete example from input to optimized output.

### Input: Toy Source Code

```toy
def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b = a;
  var c = transpose(transpose(b));
  var d = reshape(c);
  var e = reshape(d);
  print(e);
}
```

### Without Optimization

Generate MLIR without the `-opt` flag:

```powershell
toyc-ch3 -emit=mlir test.toy
```

**Output:**
```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
    %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
    %2 = toy.transpose(%1 : tensor<3x2xf64>) to tensor<2x3xf64>
    %3 = toy.reshape(%2 : tensor<2x3xf64>) to tensor<2x3xf64>
    %4 = toy.reshape(%3 : tensor<2x3xf64>) to tensor<2x3xf64>
    toy.print %4 : tensor<2x3xf64>
    toy.return
  }
}
```

**Problems:**
- Redundant transposes (`%1` and `%2`)
- Redundant reshapes (`%3` and `%4`)
- Lots of unnecessary operations

### With Optimization

Now run with the `-opt` flag:

```powershell
toyc-ch3 -emit=mlir -opt test.toy
```

**Output:**
```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
    toy.print %0 : tensor<2x3xf64>
    toy.return
  }
}
```

**Optimizations applied:**

1. **SimplifyRedundantTranspose**: `transpose(transpose(%0))` → `%0`
   - Eliminated `%1` and `%2`

2. **RedundantReshapeOptPattern**: `reshape(x)` where input/output types match → `x`
   - Eliminated `%3` and `%4`

**Result**: From 6 operations down to 2 operations!

### Transformation Trace

Let's see step-by-step how patterns applied:

**Initial:**
```mlir
%0 = constant
%1 = transpose(%0)
%2 = transpose(%1)
%3 = reshape(%2)
%4 = reshape(%3)
print %4
```

**After SimplifyRedundantTranspose:**
```mlir
%0 = constant
// %1 removed
// %2 replaced with %0
%3 = reshape(%0)
%4 = reshape(%3)
print %4
```

**After RedundantReshapeOptPattern (first application):**
```mlir
%0 = constant
// %3 replaced with %0 (types identical)
%4 = reshape(%0)
print %4
```

**After RedundantReshapeOptPattern (second application):**
```mlir
%0 = constant
// %4 replaced with %0 (types identical)
print %0
```

**Final:**
```mlir
%0 = constant
print %0
```

---

## 4.8 Building and Running Chapter 3

Let's see this in action!

### Building

```powershell
# From repo root
cmake --build build --target toyc-ch3
```

### Creating Test Files

Create `test.toy`:

```toy
def main() {
  # Create a matrix
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  
  # Do redundant operations
  var b = transpose(transpose(a));
  var c = reshape(b);
  var d = reshape(c);
  
  print(d);
}
```

### Running Without Optimization

```powershell
cd toy\Ch3
..\..\build\toy\Ch3\toyc-ch3.exe -emit=mlir test.toy
```

**Expected output:**
```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
    %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
    %2 = toy.transpose(%1 : tensor<3x2xf64>) to tensor<2x3xf64>
    %3 = toy.reshape(%2 : tensor<2x3xf64>) to tensor<2x3xf64>
    %4 = toy.reshape(%3 : tensor<2x3xf64>) to tensor<2x3xf64>
    toy.print %4 : tensor<2x3xf64>
    toy.return
  }
}
```

Count the operations: 7 (constant, 2 transposes, 2 reshapes, print, return)

### Running With Optimization

```powershell
..\..\build\toy\Ch3\toyc-ch3.exe -emit=mlir -opt test.toy
```

**Expected output:**
```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
    toy.print %0 : tensor<2x3xf64>
    toy.return
  }
}
```

Count the operations: 3 (constant, print, return)

**That's a 57% reduction in operations!**

### Experiment: More Complex Cases

Try this test case:

```toy
def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b = [[1, 2, 3], [4, 5, 6]];
  var c = reshape(a);
  var d = reshape(b);
  var e = c + d;
  print(e);
}
```

Questions:
- Which reshapes get eliminated?
- Why are some kept?
- What does the optimized IR look like?

---

## 4.9 Advanced Topics

### Pattern Ordering and Conflicts

What if multiple patterns match?

**Example:**
```cpp
// Pattern A: transpose(transpose(x)) -> x
// Pattern B: transpose(constant) -> constant'

// What if we have transpose(transpose(constant))?
```

**MLIR's solution:**
1. Try patterns in benefit order (highest first)
2. First successful match wins
3. After rewrite, re-check all patterns

**Best practice:** Assign benefits carefully:
- Higher benefit for more specific patterns
- Lower benefit for general patterns

### Preventing Infinite Loops

What if pattern A creates code that matches pattern B, and B creates code matching A?

```cpp
// Bad: A creates B's input, B creates A's input
Pattern A: x -> y
Pattern B: y -> x
```

**MLIR prevents this:**
- Tracks work list of changed operations
- Won't revisit unchanged operations
- Detects non-progress and stops

**Best practice:**
- Patterns should make progress toward canonicalization
- Should reduce operation count or complexity
- Shouldn't undo other patterns' work

### Debugging Patterns

When patterns don't work as expected:

**1. Add debug output:**
```cpp
LogicalResult matchAndRewrite(...) const override {
  llvm::errs() << "Trying pattern on: " << op << "\n";
  
  if (!matches(op)) {
    llvm::errs() << "  -> No match\n";
    return failure();
  }
  
  llvm::errs() << "  -> Matched! Rewriting...\n";
  // rewrite logic
}
```

**2. Use MLIR's built-in debugging:**
```powershell
toyc-ch3 -emit=mlir -opt test.toy --debug-only=canonicalize
```

**3. Check generated code:**
Look at `build/toy/Ch3/ToyCombine.inc` to see what TableGen generated.

---

## 4.10 Design Philosophy: When to Transform

Not all possible transformations should be done. Here are guidelines:

### Always Canonical

These are always good:
- Remove dead code
- Eliminate redundant operations
- Simplify to equivalent simpler forms

### Context-Dependent

These depend on context:
- Loop unrolling (code size vs speed)
- Inlining (compilation time vs runtime)
- Vectorization (works on SIMD, not on all targets)

### Never Harmful

Canonicalization should:
- ✅ Make IR simpler
- ✅ Enable other optimizations
- ✅ Be repeatable (idempotent)
- ❌ Not hurt code quality
- ❌ Not prevent other optimizations

**Rule of thumb:** If uncertain, make it a separate optimization pass, not a canonicalization pattern.

---

## Summary

Let's recap what we've learned:

### Key Concepts

1. **Transformations Improve Code**
   - Optimizations for performance
   - Simplifications for clarity
   - Canonicalization for consistency

2. **Pattern Matching**
   - Find code that matches patterns
   - Replace with better equivalent
   - Greedy application until fixed point

3. **Two Ways to Write Patterns**
   - **C++**: Imperative, powerful, verbose
   - **DRR**: Declarative, concise, elegant

4. **DRR Syntax**
   - `Pat<(match), (replace)>`
   - Variables: `$arg`
   - Native code calls: `NativeCodeCall<...>`
   - Constraints: `Constraint<CPred<...>>`

5. **Canonicalization Framework**
   - Standard form for IR
   - Registered per-operation
   - Built-in pass system
   - Automatic fixed-point iteration

6. **Real Impact**
   - 50%+ operation reduction possible
   - Enables further optimizations
   - Makes IR easier to analyze

### What We Built

- C++ pattern for redundant transpose elimination
- DRR patterns for reshape optimizations
- Integration with canonicalization framework
- Working optimizer via `-opt` flag

---

## What's Next

In **Chapter 5**, we'll explore **interfaces** - MLIR's way of writing generic algorithms that work across operations. We'll implement:

- Shape inference interfaces
- Generic shape inference pass
- Type propagation through the IR

This will prepare us for **Chapter 6** where we'll do our first **lowering** - transforming Toy operations into lower-level loops and memory operations.

---

## Exercises

### Exercise 1: Write a C++ Pattern

Implement a pattern that simplifies `add(x, x)` to `mul(x, 2)` (if 2x is considered better than add).

**Skeleton:**
```cpp
struct SimplifyDoubleAdd : public OpRewritePattern<AddOp> {
  SimplifyDoubleAdd(MLIRContext *context)
      : OpRewritePattern<AddOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: Check if both operands are the same
    // TODO: Create constant 2.0
    // TODO: Create MulOp
    // TODO: Replace AddOp with MulOp
  }
};
```

### Exercise 2: Write a DRR Pattern

Write a DRR pattern that removes redundant adds: `add(x, 0)` → `x`

**Hint:** You'll need to:
1. Define a constraint that checks if operand is zero
2. Write a Pat that matches `add(x, zero)`
3. Use `replaceWithValue` to return `x`

### Exercise 3: Analyze Optimization

Given this code:
```toy
def foo(a) {
  var b = transpose(a);
  var c = transpose(b);
  var d = reshape(c);
  var e = d * c;
  return e;
}
```

1. What patterns would apply?
2. In what order?
3. What would the final optimized IR look like?

Draw the transformation steps.

### Exercise 4: Debug a Pattern

This pattern doesn't work. Why?

```tablegen
def BadPattern : Pat<
  (MulOp $x, (ConstantOp $zero)),
  (ConstantOp $zero),
  [(IsZero $zero)]
>;
```

**Hint:** Think about what `$zero` captures.

<details>
<summary>Answer</summary>

`$zero` captures the ConstantOp operation itself, not its value. You can't check if an operation is zero. You need to check the constant's value attribute instead, which requires a more complex constraint or a native code call.
</details>

---

## Further Reading

### MLIR Documentation

- **Pattern Rewriting**: [https://mlir.llvm.org/docs/PatternRewriter/](https://mlir.llvm.org/docs/PatternRewriter/)
- **Declarative Rewrite Rules**: [https://mlir.llvm.org/docs/DeclarativeRewrites/](https://mlir.llvm.org/docs/DeclarativeRewrites/)
- **Canonicalization**: [https://mlir.llvm.org/docs/Canonicalization/](https://mlir.llvm.org/docs/Canonicalization/)

### Papers

- **Rewrite Rule Inference Using Equality Saturation** (PLDI 2009)
  - Theory behind pattern matching and rewriting
  
- **Equality Saturation: A New Approach to Optimization** (POPL 2011)
  - Advanced rewriting techniques

### LLVM Resources

- **LLVM InstCombine**: Similar pattern matching for LLVM IR
  - [https://llvm.org/docs/Passes.html#instcombine](https://llvm.org/docs/Passes.html#instcombine)

---

## Reflection Questions

Before moving to Chapter 5, consider:

1. **Pattern granularity**
   - Should you have many small patterns or few large ones?
   - What are the trade-offs?

2. **Soundness vs. completeness**
   - Is it better to miss some optimizations (false negatives)?
   - Or to occasionally apply incorrect transformations (false positives)?
   - How can you test patterns are correct?

3. **Human-written vs. auto-generated patterns**
   - Could patterns be learned from examples?
   - What would that look like?
   - What are the challenges?

These questions touch on active research areas in compiler optimization!

---

**Next up:** Chapter 5, where we learn about interfaces and implement shape inference! 🔍
