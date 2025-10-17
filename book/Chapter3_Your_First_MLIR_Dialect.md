# Chapter 3: Your First MLIR Dialect

> *"The secret to getting ahead is getting started." - Mark Twain*

## Introduction

Welcome to the heart of MLIR! In Chapter 2, we built a traditional compiler frontend with a lexer, parser, and AST. Now we're ready to take the leap into MLIR territory.

This chapter is pivotal because it introduces two of MLIR's most fundamental concepts:

1. **Dialects** - Custom vocabularies for your domain
2. **TableGen** - The domain-specific language for defining operations

**Fair warning**: TableGen can feel mysterious at first. It's a language for generating code, using syntax that might seem strange. But here's the key insight: **TableGen is just a template system**. Think of it as a smart way to avoid writing repetitive boilerplate code.

By the end of this chapter, you'll:
- Understand what dialects are and why they matter
- Read and write TableGen definitions confidently
- Convert AST to MLIR using MLIRGen
- See your Toy programs represented in MLIR

Let's demystify MLIR together!

---

## 3.1 What Is a Dialect?

### The Vocabulary Metaphor

Imagine you're a translator working with multiple languages. Each language has:
- Its own vocabulary (words)
- Its own grammar rules (syntax)
- Its own idioms (patterns)

MLIR dialects work the same way. A **dialect** is a namespace that contains:
- **Operations** (the vocabulary)
- **Types** (what kinds of data exist)
- **Attributes** (compile-time constants)
- **Constraints** (what's legal)

### Why Do We Need Dialects?

Remember from Chapter 1: different domains need different abstractions. MLIR solves this by letting you define custom dialects.

**Example dialects in MLIR:**

| Dialect | Purpose | Example Operation |
|---------|---------|------------------|
| `toy` | Our tutorial language | `toy.mul %a, %b` |
| `arith` | Arithmetic operations | `arith.addi %x, %y` |
| `linalg` | Linear algebra | `linalg.matmul` |
| `affine` | Structured loops | `affine.for %i = 0 to 10` |
| `scf` | Control flow | `scf.if %cond` |
| `llvm` | LLVM IR in MLIR | `llvm.add %a, %b` |

Each dialect speaks its own "language" but they can all interoperate within MLIR.

### Dialects Are Namespaces

In code, operations are prefixed with their dialect name:

```mlir
// Toy dialect operations
%0 = toy.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf64>
%1 = toy.transpose(%0 : tensor<2x2xf64>) to tensor<2x2xf64>

// Arithmetic dialect operations
%c = arith.constant 42 : i32
%sum = arith.addi %c, %c : i32

// Function dialect operations
func.func @main() {
  func.return
}
```

The `toy.`, `arith.`, and `func.` prefixes tell you which dialect each operation belongs to.

---

## 3.2 The Anatomy of an Operation

Before we define operations, let's understand what an operation *is*.

### Operations Are Instructions

In MLIR, **everything** is an operation. An operation is like an instruction that:
- Takes inputs (operands)
- Produces outputs (results)
- Has properties (attributes)
- Contains code (regions, optionally)

```mlir
%result = toy.add %lhs, %rhs : tensor<2x3xf64>
```

Let's break this down:

```
%result              →  Name of the result (SSA value)
=                    →  Assignment
toy.add              →  Operation name (dialect.mnemonic)
%lhs, %rhs           →  Operands (inputs)
: tensor<2x3xf64>    →  Type information
```

### The Four Key Components

**1. Operands (Inputs)**

Values that flow *into* the operation:

```mlir
%sum = toy.add %a, %b : tensor<2x3xf64>
//             ↑   ↑
//          operands
```

**2. Results (Outputs)**

Values produced *by* the operation:

```mlir
%result = toy.transpose(%input : tensor<2x3xf64>) to tensor<3x2xf64>
//↑
//result
```

**3. Attributes (Compile-Time Constants)**

Data attached to the operation that's known at compile time:

```mlir
%0 = toy.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf64>
//                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                   attribute (the actual data)
```

**4. Regions (Nested Code)**

Some operations contain other operations:

```mlir
toy.func @main() {
  // This is a region containing other operations
  %0 = toy.constant dense<1.0> : tensor<f64>
  toy.return
}
```

### SSA Form: Static Single Assignment

MLIR uses SSA form, which means:
- Each value is assigned exactly once
- Values are immutable once created
- Every use has exactly one definition

**Not SSA (bad):**
```
x = 1
x = x + 2    // Reassigning x
x = x * 3    // Reassigning x again
```

**SSA (good):**
```mlir
%x1 = arith.constant 1 : i32
%x2 = arith.addi %x1, %c2 : i32
%x3 = arith.muli %x2, %c3 : i32
```

Each value (`%x1`, `%x2`, `%x3`) is defined once and never changes. This makes optimization and analysis much easier.

---

## 3.3 Introducing TableGen

Now for the part that might seem intimidating: **TableGen**. Let's demystify it.

### What Is TableGen?

TableGen is a **domain-specific language** (DSL) for describing structured data. It's used throughout LLVM and MLIR to:
- Define operations
- Specify type systems
- Describe transformation patterns
- Generate C++ code automatically

**Think of TableGen as:**
- A template system (like Jinja or Mustache, but for C++ code)
- A way to avoid repetitive boilerplate
- A declarative alternative to manual coding

### Why TableGen?

Imagine defining an operation manually in C++. You'd need to write:
- Class definition (boilerplate)
- Constructor (boilerplate)
- Accessors for operands (boilerplate)
- Accessors for results (boilerplate)
- Parsing code (tedious)
- Printing code (tedious)
- Verification code (error-prone)

That's hundreds of lines of repetitive code per operation. **TableGen generates all of that automatically** from a concise description.

### TableGen vs C++: An Example

**Manual C++ approach (what you don't want to write):**

```cpp
// Dozens of lines of boilerplate per operation...
class AddOp : public Op<AddOp, OpTrait::OneResult, OpTrait::TwoOperands> {
public:
  static StringRef getOperationName() { return "toy.add"; }
  
  Value getLhs() { return getOperand(0); }
  Value getRhs() { return getOperand(1); }
  
  static ParseResult parse(OpAsmParser &parser, OperationState &result) {
    // 20+ lines of parsing code...
  }
  
  void print(OpAsmPrinter &p) {
    // 10+ lines of printing code...
  }
  
  LogicalResult verify() {
    // Verification logic...
  }
  
  // Plus builders, traits, interfaces, etc.
};
```

**TableGen approach (what you actually write):**

```tablegen
def AddOp : Toy_Op<"add"> {
  let summary = "element-wise addition operation";
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);
  let hasCustomAssemblyFormat = 1;
}
```

**8 lines instead of 100+**. The rest is generated automatically.

---

## 3.4 TableGen Language Basics

Let's learn the essential TableGen syntax.

### 1. Classes vs Definitions

TableGen has two main constructs:

**`class`** - A template or pattern (abstract)

```tablegen
class Toy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;
```

Think of this like a C++ class template. It defines a pattern that can be instantiated.

**`def`** - A concrete instance

```tablegen
def AddOp : Toy_Op<"add"> {
  // Concrete definition
}
```

This creates an actual operation called `AddOp` by instantiating the `Toy_Op` template.

**Analogy:**
- `class` is like a cookie cutter (template)
- `def` is like a cookie (concrete instance)

### 2. Template Parameters

Classes can take parameters:

```tablegen
class Toy_Op<string mnemonic, list<Trait> traits = []>
//           ^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^
//           parameter 1       parameter 2 with default value
```

When you instantiate it:

```tablegen
def ConstantOp : Toy_Op<"constant", [Pure]> {
  //                     ^^^^^^^^^^^  ^^^^^^
  //                     mnemonic     traits
}
```

### 3. The `let` Binding

`let` assigns values to fields:

```tablegen
def AddOp : Toy_Op<"add"> {
  let summary = "element-wise addition operation";
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);
}
```

Each `let` sets a property of the operation.

### 4. DAG Syntax (Directed Acyclic Graph)

The `(...)` syntax creates a DAG (Directed Acyclic Graph):

```tablegen
let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
//              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//              This is a DAG
```

**DAG structure:**
```
(operator arg1:$name1, arg2:$name2, ...)
```

- `ins` = operator (means "inputs")
- `F64Tensor` = type
- `:$lhs` = name (the `$` prefix indicates this is a variable)

**Common operators:**
- `ins` - Input operands
- `outs` - Output results
- `attr` - Attributes

### 5. Inheritance

Definitions can inherit from classes:

```tablegen
def AddOp : Toy_Op<"add"> { ... }
//          ^^^^^^^^^^^^
//          Inheriting from Toy_Op
```

This is like C++ inheritance. `AddOp` gets all the properties of `Toy_Op`, plus whatever you add.

### 6. String Interpolation

You can include variables in strings:

```tablegen
class MyOp<string name> {
  let description = [{
    This operation is named }] # name # [{
  }];
}
```

The `#` operator concatenates strings.

---

## 3.5 Defining the Toy Dialect

Now let's look at real code. Open `toy/Ch2/include/toy/Ops.td`.

### Dialect Definition

First, we define the dialect itself:

```tablegen
// Provide a definition of the 'toy' dialect in the ODS framework
def Toy_Dialect : Dialect {
  let name = "toy";
  let cppNamespace = "::mlir::toy";
}
```

**What this means:**
- Creates a dialect named "toy"
- Generated C++ code will be in the `::mlir::toy` namespace
- Operations will be prefixed with `toy.` (like `toy.add`)

**ODS = Operation Definition Specification** - That's the framework for defining operations in TableGen.

### Base Operation Class

Next, we define a base class for all Toy operations:

```tablegen
// Base class for toy dialect operations
class Toy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;
```

**Breaking it down:**

```tablegen
class Toy_Op<              // Define a class template
  string mnemonic,         // Parameter 1: operation name
  list<Trait> traits = []  // Parameter 2: optional traits (defaults to empty)
> :
  Op<                      // Inherit from the base Op class
    Toy_Dialect,           // This op belongs to Toy dialect
    mnemonic,              // Pass through the mnemonic
    traits                 // Pass through the traits
  >;
```

Now every operation can inherit from `Toy_Op` instead of writing out all this stuff.

---

## 3.6 Defining Operations in TableGen

Let's examine several operations to understand the patterns.

### Example 1: ConstantOp

```tablegen
def ConstantOp : Toy_Op<"constant", [Pure]> {
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                        : tensor<2x3xf64>
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins F64ElementsAttr:$value);

  // The constant operation returns a single value of TensorType.
  let results = (outs F64Tensor);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Add custom build methods
  let builders = [
    OpBuilder<(ins "DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,
    OpBuilder<(ins "double":$value)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}
```

**Let's decode this:**

#### Summary and Description

```tablegen
let summary = "constant";
let description = [{ ... }];
```

- `summary`: One-line description
- `description`: Multi-line documentation (note the `[{ }]` syntax for multi-line strings)

This generates documentation automatically!

#### Arguments

```tablegen
let arguments = (ins F64ElementsAttr:$value);
```

- `ins` = inputs (even though it's an attribute, not an operand)
- `F64ElementsAttr` = type (a dense array of f64 values)
- `:$value` = name of this argument

You can access it in generated C++ as `getValue()`.

#### Results

```tablegen
let results = (outs F64Tensor);
```

- `outs` = outputs
- `F64Tensor` = type
- No name needed (there's only one result)

Access in C++ as `getResult()`.

#### Traits

```tablegen
def ConstantOp : Toy_Op<"constant", [Pure]> {
```

The `[Pure]` trait means:
- This operation has no side effects
- It can be eliminated if its result is unused (dead code elimination)
- It's safe to move or duplicate

#### Custom Builders

```tablegen
let builders = [
  OpBuilder<(ins "DenseElementsAttr":$value), [{
    build($_builder, $_state, value.getType(), value);
  }]>,
  OpBuilder<(ins "double":$value)>
];
```

This generates C++ methods for constructing the operation:

```cpp
// Generated code:
ConstantOp::build(OpBuilder &builder, OperationState &state, 
                  DenseElementsAttr value) {
  build(builder, state, value.getType(), value);
}

// Can be used as:
builder.create<ConstantOp>(loc, myAttribute);
```

### Example 2: AddOp (Binary Operation)

```tablegen
def AddOp : Toy_Op<"add"> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}
```

**Key differences:**
- **Two operands**: `$lhs` and `$rhs` (left-hand side, right-hand side)
- **No traits**: Traits list is optional
- **Custom assembly**: We'll implement parsing/printing manually

**Generated accessors:**

```cpp
Value AddOp::getLhs() { return getOperand(0); }
Value AddOp::getRhs() { return getOperand(1); }
```

### Example 3: TransposeOp

```tablegen
def TransposeOp : Toy_Op<"transpose"> {
  let summary = "transpose operation";

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)
  }];

  let builders = [
    OpBuilder<(ins "Value":$input)>
  ];

  let hasVerifier = 1;
}
```

**New feature: Declarative assembly format:**

```tablegen
let assemblyFormat = [{
  `(` $input `:` type($input) `)` attr-dict `to` type(results)
}];
```

This defines how the operation looks in MLIR text:

```mlir
%1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
```

- `` `(` `` = literal left parenthesis
- `$input` = the input operand
- `:` = literal colon
- `type($input)` = print the type of input
- `attr-dict` = any attributes (if present)
- `` `to` `` = literal "to"
- `type(results)` = print the result type

**This declarative format automatically generates parsing and printing code!**

### Example 4: ReturnOp

```tablegen
def ReturnOp : Toy_Op<"return", [Pure, HasParent<"FuncOp">, Terminator]> {
  let summary = "return operation";
  
  let arguments = (ins Variadic<F64Tensor>:$input);

  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
  ];

  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  let hasVerifier = 1;
}
```

**New concepts:**

#### Multiple Traits

```tablegen
[Pure, HasParent<"FuncOp">, Terminator]
```

- `Pure`: No side effects
- `HasParent<"FuncOp">`: Must be inside a FuncOp
- `Terminator`: Ends a block (no operations can follow it)

#### Variadic Operands

```tablegen
let arguments = (ins Variadic<F64Tensor>:$input);
```

`Variadic` means "zero or more operands":
- `toy.return` - no operands
- `toy.return %0 : tensor<f64>` - one operand

#### Optional Assembly Syntax

```tablegen
let assemblyFormat = "($input^ `:` type($input))? attr-dict ";
//                    ^                          ^
//                    begin optional           end optional
```

The `?` makes the preceding group optional. The `^` indicates to only show this if `$input` is present.

#### Extra C++ Methods

```tablegen
let extraClassDeclaration = [{
  bool hasOperand() { return getNumOperands() != 0; }
}];
```

This adds custom C++ methods to the generated class.

---

## 3.7 How TableGen Generates C++ Code

Let's see what happens when you build the project.

### The Build Process

```
Ops.td (TableGen)
    ↓
[mlir-tblgen] (Code generator)
    ↓
Ops.h.inc (Generated C++ header)
Ops.cpp.inc (Generated C++ implementation)
    ↓
[C++ Compiler]
    ↓
Binary
```

### Generated Code (Conceptual)

From this TableGen:

```tablegen
def AddOp : Toy_Op<"add"> {
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);
}
```

TableGen generates (simplified):

```cpp
class AddOp : public Op<AddOp, /* traits */> {
public:
  // Operation name
  static constexpr StringLiteral getOperationName() {
    return "toy.add";
  }
  
  // Accessors for operands
  Value getLhs() { return getOperand(0); }
  Value getRhs() { return getOperand(1); }
  
  // Accessor for result
  Value getResult() { return (*this)->getResult(0); }
  
  // Builder methods
  static void build(OpBuilder &builder, OperationState &state,
                    Value lhs, Value rhs);
  
  // Parsing and printing
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  
  // Verification
  LogicalResult verify();
};
```

**You write 8 lines of TableGen. You get 50+ lines of correct, tested C++ code.**

### Including Generated Code

In `Dialect.cpp`, you include the generated code:

```cpp
#include "toy/Dialect.h"

// Include the generated dialect implementation
#include "toy/Dialect.cpp.inc"

void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "toy/Ops.cpp.inc"
  >();
}
```

The `#include "toy/Ops.cpp.inc"` brings in all the generated operation definitions.

---

## 3.8 Understanding Builders and Code Generation

Let's dive deeper into what TableGen generates vs what you write yourself, and clarify the role of builders.

### The Three Layers of Operation Creation

When you define an operation in TableGen, there are three layers to understand:

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: TableGen Definition (.td file)                     │
│ → What: High-level declaration (like a .h header)           │
│ → Who writes it: You                                        │
│ → Purpose: Describe the operation's structure               │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: Generated C++ Code (.inc files)                    │
│ → What: Operation class implementation                      │
│ → Who writes it: mlir-tblgen (automatic)                    │
│ → Purpose: Boilerplate accessors, builders, verification    │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: Lowering/Transformation Logic (.cpp files)         │
│ → What: How to transform or execute the operation           │
│ → Who writes it: You                                        │
│ → Purpose: Define semantics and behavior                    │
└─────────────────────────────────────────────────────────────┘
```

### What Does a Builder Do?

**A builder is a convenience constructor** for creating operations. Think of it as syntactic sugar that makes your C++ code cleaner.

#### Without Custom Builders

Every operation automatically gets a **default builder** that takes all inputs explicitly:

```cpp
// Default builder signature (auto-generated):
static void build(OpBuilder &builder, OperationState &state,
                  Type resultType,         // Result type
                  Value operand1,          // First operand
                  Value operand2);         // Second operand

// Usage (verbose):
Type resultType = lhs.getType();
builder.create<AddOp>(loc, resultType, lhs, rhs);
```

#### With Custom Builders

You can add custom builders to make common usage patterns easier:

```tablegen
def AddOp : Toy_Op<"add"> {
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);
  
  // Custom builder that infers result type from lhs
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}
```

**Generated code:**
```cpp
// Custom builder (generated from TableGen)
static void build(OpBuilder &builder, OperationState &state,
                  Value lhs, Value rhs) {
  // Automatically infer result type from lhs
  state.addTypes(lhs.getType());
  state.addOperands({lhs, rhs});
}

// Usage (clean):
builder.create<AddOp>(loc, lhs, rhs);  // Much simpler!
```

### Real Example: ConstantOp Builders

The `ConstantOp` has **two builders** for different use cases:

```tablegen
def ConstantOp : Toy_Op<"constant", [Pure]> {
  let arguments = (ins F64ElementsAttr:$value);
  let results = (outs F64Tensor);
  
  let builders = [
    // Builder 1: For creating constants from dense arrays
    OpBuilder<(ins "DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,
    
    // Builder 2: For creating scalar constants from double
    OpBuilder<(ins "double":$value)>
  ];
}
```

**How they're used in MLIRGen.cpp:**

```cpp
// Builder 1: Creating a tensor constant
mlir::Value mlirGen(LiteralExprAST &lit) {
  auto dataAttribute = DenseElementsAttr::get(dataType, data);
  // Uses builder 1: (DenseElementsAttr)
  return builder.create<ConstantOp>(loc(lit.loc()), type, dataAttribute);
}

// Builder 2: Creating a scalar constant  
mlir::Value mlirGen(NumberExprAST &num) {
  // Uses builder 2: (double)
  return builder.create<ConstantOp>(loc(num.loc()), num.getValue());
}
```

**Same operation, two convenient creation methods!**

### Do You Need Custom Builders?

**No, they're optional!** Let's see an operation without custom builders:

```tablegen
def PrintOp : Toy_Op<"print"> {
  let summary = "print operation";
  let arguments = (ins F64Tensor:$input);
  // NO custom builders defined!
}
```

This still works perfectly:

```cpp
// Uses the auto-generated default builder
builder.create<PrintOp>(loc, input);
```

TableGen automatically generates a builder that takes the operands.

### When to Add Custom Builders

Add custom builders when you want:

1. **Type Inference**: Automatically deduce result types
   ```tablegen
   let builders = [
     OpBuilder<(ins "Value":$input)>  // Infer result from input
   ];
   ```

2. **Multiple Input Formats**: Different ways to construct the same operation
   ```tablegen
   let builders = [
     OpBuilder<(ins "DenseElementsAttr":$value)>,  // From attribute
     OpBuilder<(ins "double":$value)>               // From scalar
   ];
   ```

3. **Default Arguments**: Provide sensible defaults
   ```tablegen
   let builders = [
     OpBuilder<(ins "Value":$input,
                    CArg<"bool", "false">:$transpose)>
   ];
   ```

4. **Convenience**: Simplify common patterns
   ```tablegen
   let builders = [
     // Complex setup hidden in builder body
     OpBuilder<(ins "ArrayRef<int64_t>":$shape), [{
       auto type = RankedTensorType::get(shape, builder.getF64Type());
       build($_builder, $_state, type);
     }]>
   ];
   ```

### The Complete Picture: What Generates What

Let's trace through the complete build process for `AddOp`:

**Step 1: You write TableGen** (`Ops.td`):
```tablegen
def AddOp : Toy_Op<"add"> {
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}
```

**Step 2: Build system runs mlir-tblgen** (during CMake build):
```cmake
mlir_tablegen(Ops.h.inc -gen-op-decls)      # Generate declarations
mlir_tablegen(Ops.cpp.inc -gen-op-defs)     # Generate implementations
```

**Step 3: Generated `Ops.h.inc`** (simplified):
```cpp
class AddOp : public Op<AddOp, ...> {
public:
  // Operation name
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("toy.add");
  }
  
  // Accessor methods for operands
  Value getLhs() { return getOperand(0); }
  Value getRhs() { return getOperand(1); }
  
  // Accessor for result
  Value getResult() { return getOperation()->getResult(0); }
  
  // Builder declarations
  static void build(OpBuilder &odsBuilder, OperationState &odsState,
                    Value lhs, Value rhs);
  
  // Parser and printer
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  
  // Verification
  LogicalResult verify();
};
```

**Step 4: Generated `Ops.cpp.inc`** (simplified):
```cpp
// Custom builder implementation
void AddOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                  Value lhs, Value rhs) {
  odsState.addOperands({lhs, rhs});
  odsState.addTypes({lhs.getType()});  // Infer result type
}

// Verification implementation
LogicalResult AddOp::verify() {
  // Check that operands and results have compatible types
  if (getLhs().getType() != getRhs().getType())
    return emitOpError("operand types must match");
  // ... more checks ...
  return success();
}

// ... parser and printer implementations ...
```

**Step 5: Your hand-written code** (`Dialect.cpp`, `LowerToAffineLoops.cpp`, etc.):
```cpp
// Using the operation in MLIRGen
Value result = builder.create<AddOp>(loc, lhs, rhs);

// Lowering the operation (YOU write this)
struct AddOpLowering : public OpRewritePattern<AddOp> {
  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const final {
    // Convert toy.add to affine.for loops with arith.addf
    // THIS is where the actual computation logic goes!
    // ...
    return success();
  }
};
```

### Key Insight: TableGen vs Behavior

**TableGen defines STRUCTURE, not BEHAVIOR:**

| What | Where | Who Writes It |
|------|-------|---------------|
| **Operation declaration** | `Ops.td` | You (TableGen) |
| **Generated accessors** | `Ops.cpp.inc` | mlir-tblgen (automatic) |
| **Generated builders** | `Ops.cpp.inc` | mlir-tblgen (automatic) |
| **Lowering logic** | `LowerToAffineLoops.cpp` | **You (C++ code!)** |
| **Optimization patterns** | `ToyCombine.cpp` | **You (C++ code!)** |
| **Execution semantics** | Eventually LLVM IR | Through lowering passes |

**The operation at MLIR IR level is just a data structure**. It doesn't "do" anything until you:
1. Lower it to executable operations
2. Generate LLVM IR
3. JIT compile and run

For `ConstantOp`, the "computation" is trivial—it just represents constant data. But for `AddOp`, the actual addition happens much later during lowering (Chapter 6) when you write code like:

```cpp
// This is where the actual "add" happens - YOU write this!
Value sum = rewriter.create<arith::AddFOp>(loc, elementA, elementB);
```

### Summary

- ✅ **Builders are optional convenience constructors**
- ✅ **Every operation gets a default builder automatically**
- ✅ **Custom builders make common patterns easier**
- ✅ **TableGen generates structure (classes, accessors, builders)**
- ✅ **You write behavior (lowering, optimization, semantics)**
- ✅ **Operations are data structures until lowered to executable code**

With this understanding, you can now read any `.td` file and understand what code it generates and what code you still need to write!

---

## 3.9 MLIR File Organization: The Standard Pattern

Before we dive into using operations, let's understand the **standard file organization** for MLIR dialects. This is important because you'll see this pattern everywhere in MLIR code.

### The Three-File Pattern

Every MLIR dialect typically follows this structure:

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Ops.td (TableGen)                                            │
│    → DECLARE operations                                         │
│    → Define structure, types, traits                            │
│    → "Here's WHAT operations exist"                             │
└───────────────────────┬─────────────────────────────────────────┘
                        ↓ mlir-tblgen generates
┌─────────────────────────────────────────────────────────────────┐
│ 2. Ops.h.inc + Ops.cpp.inc (Generated)                          │
│    → Operation class skeletons                                  │
│    → Default accessors                                          │
│    → Declaration stubs for custom methods                       │
└───────────────────────┬─────────────────────────────────────────┘
                        ↓ you implement
┌─────────────────────────────────────────────────────────────────┐
│ 3. Dialect.cpp (Hand-Written)                                   │
│    → IMPLEMENT custom builders                                  │
│    → IMPLEMENT parsers/printers                                 │
│    → IMPLEMENT verifiers                                        │
│    → Register dialect and operations                            │
│    → "Here's HOW operations work"                               │
└─────────────────────────┬───────────────────────────────────────┘
                        ↓ used by
┌─────────────────────────────────────────────────────────────────┐
│ 4. MLIRGen.cpp (Hand-Written)                                   │
│    → USE builder.create<Op>()                                   │
│    → Convert source language → MLIR                             │
│    → "Here's how to CREATE operations"                          │
└─────────────────────────────────────────────────────────────────┘
```

### In the Toy Tutorial

Let's look at the actual file structure for Chapter 2:

```
toy/Ch2/
├── include/toy/
│   ├── Ops.td              # TableGen definitions
│   ├── Dialect.h           # Dialect header
│   └── MLIRGen.h           # MLIRGen interface
│
└── mlir/
    ├── Dialect.cpp         # Operation implementations
    └── MLIRGen.cpp         # AST-to-MLIR conversion
```

### What Goes in Each File?

#### **`Ops.td`** - Define the Contract

```tablegen
def ConstantOp : Toy_Op<"constant", [Pure]> {
  let summary = "constant";
  let arguments = (ins F64ElementsAttr:$value);
  let results = (outs F64Tensor);
  
  // Declare that we have custom implementations
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  
  // Declare custom builders
  let builders = [
    OpBuilder<(ins "DenseElementsAttr":$value)>,
    OpBuilder<(ins "double":$value)>
  ];
}
```

**Purpose**: Declare what operations exist and their structure.

#### **`Dialect.cpp`** - Implement the Operations

```cpp
//===----------------------------------------------------------------------===//
// Custom Builder Implementation
//===----------------------------------------------------------------------===//

void ConstantOp::build(OpBuilder &builder, OperationState &state,
                       double value) {
  // Implementation of the builder declared in TableGen
  auto dataType = RankedTensorType::get({}, builder.getF64Type());
  auto dataAttribute = DenseElementsAttr::get(dataType, value);
  ConstantOp::build(builder, state, dataType, dataAttribute);
}

//===----------------------------------------------------------------------===//
// Custom Parser Implementation
//===----------------------------------------------------------------------===//

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  // Implementation: how to read "toy.constant dense<...>"
  DenseElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(value, "value", result.attributes))
    return failure();
  result.addTypes(value.getType());
  return success();
}

//===----------------------------------------------------------------------===//
// Custom Printer Implementation
//===----------------------------------------------------------------------===//

void ConstantOp::print(OpAsmPrinter &printer) {
  // Implementation: how to write "toy.constant dense<...>"
  printer << " ";
  printer.printOptionalAttrDict((*this)->getAttrs(), {"value"});
  printer << getValue();
}

//===----------------------------------------------------------------------===//
// Verifier Implementation
//===----------------------------------------------------------------------===//

LogicalResult ConstantOp::verify() {
  // Implementation: semantic checks
  auto resultType = llvm::dyn_cast<RankedTensorType>(getResult().getType());
  if (!resultType)
    return emitOpError("result must be a ranked tensor");
  
  // More validation...
  return success();
}

//===----------------------------------------------------------------------===//
// Dialect Registration
//===----------------------------------------------------------------------===//

void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "toy/Ops.cpp.inc"
  >();
}
```

**Purpose**: Implement the behavior declared in TableGen.

#### **`MLIRGen.cpp`** - Use the Operations

```cpp
class MLIRGenImpl {
  mlir::Value mlirGen(NumberExprAST &num) {
    // USE the custom builder we implemented in Dialect.cpp
    return builder.create<ConstantOp>(loc(num.loc()), num.getValue());
  }
  
  mlir::Value mlirGen(LiteralExprAST &lit) {
    auto dataAttribute = DenseElementsAttr::get(dataType, data);
    // USE the other custom builder
    return builder.create<ConstantOp>(loc(lit.loc()), type, dataAttribute);
  }
  
  mlir::Value mlirGen(BinaryExprAST &binop) {
    Value lhs = mlirGen(*binop.getLHS());
    Value rhs = mlirGen(*binop.getRHS());
    
    if (binop.getOp() == '+')
      return builder.create<AddOp>(loc(binop.loc()), lhs, rhs);
    
    return builder.create<MulOp>(loc(binop.loc()), lhs, rhs);
  }
};
```

**Purpose**: Generate MLIR by creating operations from AST.

### The Conceptual Division

Think of it this way:

| File | Role | Analogy |
|------|------|---------|
| **`Ops.td`** | **Declaration** | Interface (`.h` file) |
| **`Dialect.cpp`** | **Implementation** | Implementation (`.cpp` file) |
| **`MLIRGen.cpp`** | **Usage** | Client code that uses the API |

### Why This Separation?

This organization provides clear **separation of concerns**:

1. **`Ops.td`**: High-level specification
   - Easy to understand operation structure
   - Generates boilerplate automatically
   - Single source of truth

2. **`Dialect.cpp`**: Operation semantics
   - Custom behavior that can't be auto-generated
   - Parsing, printing, verification
   - Builder implementations with complex logic

3. **`MLIRGen.cpp`**: Language frontend
   - Converts your source language to MLIR
   - Can be replaced for different source languages
   - Independent of operation implementation details

### Real-World Example: Arith Dialect

Let's look at MLIR's built-in `arith` dialect structure:

```
mlir/lib/Dialect/Arith/
├── IR/
│   ├── ArithDialect.td         # Dialect definition
│   ├── ArithOps.td             # Operation definitions (TableGen)
│   ├── ArithOps.cpp            # Operation implementations
│   └── ArithDialect.cpp        # Dialect registration
│
└── Transforms/
    ├── ExpandOps.cpp           # Expand complex ops to simple ones
    ├── IntRangeOptimizations.cpp
    └── ...
```

Notice: No "Gen" file! Why? Because `arith` is a **target dialect**, not a source language. Operations are created by:
- Other dialects lowering to arith
- Transformation passes
- Optimization passes

### Scaling to Larger Projects

As projects grow, you'll see expanded organization:

```
MyDialect/
├── IR/                         # Core definitions
│   ├── MyDialect.td           # Dialect definition
│   ├── MyOps.td               # Operation definitions
│   ├── MyTypes.td             # Custom types
│   ├── MyAttributes.td        # Custom attributes
│   ├── MyDialect.cpp          # Dialect implementation
│   ├── MyOps.cpp              # Operation implementations
│   └── MyTypes.cpp            # Type implementations
│
├── Transforms/                 # Optimization passes
│   ├── Canonicalize.cpp       # Canonicalization patterns
│   ├── MyPass.cpp             # Custom transformation passes
│   └── Passes.td              # Pass definitions
│
├── Conversion/                 # Lowering to other dialects
│   ├── MyToStd.cpp           # Lower to standard dialect
│   ├── MyToLLVM.cpp          # Lower to LLVM dialect
│   └── ConversionPatterns.td  # Lowering patterns
│
└── Frontend/                   # Language-specific (if applicable)
    └── MyLangGen.cpp          # Source language → MyDialect
```

### Common Patterns You'll See

#### Pattern 1: Dialect with Frontend (like Toy)
```
Ops.td → Dialect.cpp → MLIRGen.cpp → Your Compiler
         (implement)   (use)
```

#### Pattern 2: Intermediate Dialect (like Affine, SCF)
```
Ops.td → Dialect.cpp → Lowering passes create these ops
         (implement)   (from higher dialects)
```

#### Pattern 3: Target Dialect (like LLVM)
```
Ops.td → Dialect.cpp → ConversionPatterns.cpp
         (implement)   (other dialects lower to this)
```

### Key Takeaways

1. **`Ops.td`** declares operations (WHAT)
2. **`Dialect.cpp`** implements operations (HOW they work)
3. **`*Gen.cpp`** uses operations (creating MLIR from source)
4. **This is the standard pattern across all MLIR projects**
5. **Separation enables modularity**: change frontend without touching ops

### Quiz: Where Does This Code Go?

Let's test your understanding:

**Question 1**: Where does this go?
```cpp
def AddOp : Toy_Op<"add"> {
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
}
```
**Answer**: `Ops.td` (TableGen declaration)

**Question 2**: Where does this go?
```cpp
LogicalResult AddOp::verify() {
  if (getLhs().getType() != getRhs().getType())
    return emitOpError("operand types must match");
  return success();
}
```
**Answer**: `Dialect.cpp` (verification implementation)

**Question 3**: Where does this go?
```cpp
Value result = builder.create<AddOp>(loc, lhs, rhs);
```
**Answer**: `MLIRGen.cpp` or any code that generates MLIR (usage)

### Summary

Understanding this file organization is crucial because:
- ✅ You'll see this pattern in **every MLIR dialect**
- ✅ It separates **declaration, implementation, and usage**
- ✅ Makes code easier to navigate and understand
- ✅ Enables **modularity**: swap frontends, add backends independently

Now when you look at any MLIR codebase, you'll immediately understand the structure!

---

## 3.10 From AST to MLIR: MLIRGen

Now that we understand file organization, let's see how to use operations to convert AST to MLIR.

### The MLIRGen Class

`MLIRGen.cpp` contains the `MLIRGenImpl` class that traverses the AST and generates MLIR operations:

```cpp
class MLIRGenImpl {
public:
  MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}

  mlir::ModuleOp mlirGen(ModuleAST &moduleAST) {
    // Create an empty MLIR module
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

    // Generate MLIR for each function
    for (FunctionAST &f : moduleAST)
      mlirGen(f);

    // Verify the module
    if (failed(mlir::verify(theModule))) {
      theModule.emitError("module verification error");
      return nullptr;
    }

    return theModule;
  }

private:
  mlir::ModuleOp theModule;
  mlir::OpBuilder builder;
  llvm::ScopedHashTable<StringRef, mlir::Value> symbolTable;
  // ... helper methods ...
};
```

### The OpBuilder

The `OpBuilder` is MLIR's API for constructing operations:

```cpp
mlir::OpBuilder builder;

// Create operations:
builder.create<ConstantOp>(location, type, value);
builder.create<AddOp>(location, lhs, rhs);
builder.create<ReturnOp>(location, result);
```

The builder tracks:
- **Insertion point**: Where to add new operations
- **Context**: The MLIR context managing memory

### Converting AST Nodes to MLIR

Let's trace through examples:

#### Example 1: Number Literal → ConstantOp

**AST:**
```cpp
NumberExprAST(location, 42.0)
```

**MLIRGen code:**
```cpp
mlir::Value mlirGen(NumberExprAST &num) {
  return builder.create<ConstantOp>(loc(num.loc()), num.getValue());
}
```

**Generated MLIR:**
```mlir
%0 = toy.constant dense<4.200000e+01> : tensor<f64>
```

#### Example 2: Tensor Literal → ConstantOp

**AST:**
```cpp
LiteralExprAST(location, 
  values: [1.0, 2.0, 3.0, 4.0],
  dims: [2, 2]
)
```

**MLIRGen code:**
```cpp
mlir::Value mlirGen(LiteralExprAST &lit) {
  auto type = getType(lit.getDims());
  
  // Flatten the nested literal into a vector
  std::vector<double> data;
  collectData(lit, data);
  
  // Create the MLIR type and attribute
  mlir::Type elementType = builder.getF64Type();
  auto dataType = mlir::RankedTensorType::get(lit.getDims(), elementType);
  auto dataAttribute = mlir::DenseElementsAttr::get(dataType, data);
  
  // Build the operation
  return builder.create<ConstantOp>(loc(lit.loc()), type, dataAttribute);
}
```

**Generated MLIR:**
```mlir
%0 = toy.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf64>
```

#### Example 3: Binary Operation → MulOp/AddOp

**AST:**
```cpp
BinaryExprAST(location, '*',
  lhs: VariableExprAST("a"),
  rhs: VariableExprAST("b")
)
```

**MLIRGen code:**
```cpp
mlir::Value mlirGen(BinaryExprAST &binop) {
  // Recursively generate LHS and RHS
  mlir::Value lhs = mlirGen(*binop.getLHS());
  if (!lhs)
    return nullptr;
  mlir::Value rhs = mlirGen(*binop.getRHS());
  if (!rhs)
    return nullptr;

  auto location = loc(binop.loc());

  // Create the appropriate operation
  switch (binop.getOp()) {
  case '+':
    return builder.create<AddOp>(location, lhs, rhs);
  case '*':
    return builder.create<MulOp>(location, lhs, rhs);
  }
  
  emitError(location, "invalid binary operator");
  return nullptr;
}
```

**Generated MLIR:**
```mlir
%0 = toy.mul %a, %b : tensor<2x3xf64>
```

#### Example 4: Function Call → GenericCallOp

**AST:**
```cpp
CallExprAST(location, "multiply_transpose",
  args: [VariableExprAST("a"), VariableExprAST("b")]
)
```

**MLIRGen code:**
```cpp
mlir::Value mlirGen(CallExprAST &call) {
  llvm::StringRef callee = call.getCallee();
  auto location = loc(call.loc());

  // Handle built-in operations specially
  if (callee == "transpose") {
    if (call.getArgs().size() != 1) {
      emitError(location, "transpose expects 1 argument");
      return nullptr;
    }
    mlir::Value arg = mlirGen(*call.getArgs()[0]);
    return builder.create<TransposeOp>(location, arg);
  }

  // Generic call to user-defined function
  SmallVector<mlir::Value, 4> operands;
  for (auto &expr : call.getArgs()) {
    auto arg = mlirGen(*expr);
    if (!arg)
      return nullptr;
    operands.push_back(arg);
  }

  return builder.create<GenericCallOp>(location, callee, operands);
}
```

**Generated MLIR:**
```mlir
%2 = toy.generic_call @multiply_transpose(%0, %1) 
       : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
```

### Symbol Table Management

Variables need to be tracked so we know what `%a` refers to:

```cpp
// Scoped hash table for variable lookup
llvm::ScopedHashTable<StringRef, mlir::Value> symbolTable;

// Declaring a variable
mlir::LogicalResult declare(StringRef var, mlir::Value value) {
  if (symbolTable.count(var))
    return mlir::failure();  // Already declared!
  symbolTable.insert(var, value);
  return mlir::success();
}

// Looking up a variable
mlir::Value mlirGen(VariableExprAST &expr) {
  if (auto variable = symbolTable.lookup(expr.getName()))
    return variable;
  
  emitError(loc(expr.loc()), "unknown variable '") << expr.getName() << "'";
  return nullptr;
}
```

---

## 3.11 Complete Example: Toy to MLIR

Let's trace through a complete transformation.

### Input: Toy Source Code

```toy
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  var a = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  var c = multiply_transpose(a, b);
  print(c);
}
```

### Step 1: Parse to AST

(We did this in Chapter 2)

### Step 2: MLIRGen Transformation

The `mlirGen` function walks the AST and generates MLIR operations.

### Step 3: Output MLIR

```mlir
module {
  toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
    %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
    %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
    %2 = toy.mul %0, %1 : tensor<*xf64>
    toy.return %2 : tensor<*xf64>
  }
  
  toy.func @main() {
    %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
    %1 = toy.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]> : tensor<6xf64>
    %2 = toy.reshape(%1 : tensor<6xf64>) to tensor<2x3xf64>
    %3 = toy.generic_call @multiply_transpose(%0, %2) 
           : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    toy.print %3 : tensor<*xf64>
    toy.return
  }
}
```

### Key Observations

**1. Functions become `toy.func` operations:**

```mlir
toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>)
```

Note `tensor<*xf64>` - the `*` means "unknown shape". We'll infer shapes in Chapter 5.

**2. Literals become `toy.constant`:**

```mlir
%0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
```

The data is embedded as an attribute (`dense<...>`).

**3. Operations become dialect operations:**

```mlir
%0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
%2 = toy.mul %0, %1 : tensor<*xf64>
```

**4. SSA values are explicit:**

Every value has a name (`%0`, `%1`, `%2`, etc.) and is defined exactly once.

**5. Types are explicit:**

Unlike the AST, MLIR makes all types visible: `tensor<2x3xf64>`.

---

## 3.12 Building and Running Chapter 2

Time to see this in action!

### Building

```powershell
# From the repo root
cmake --build build --target toyc-ch2
```

### Running

```powershell
# Navigate to Ch2
cd toy\Ch2

# Generate MLIR from Toy source
..\..\build\toy\Ch2\toyc-ch2.exe -emit=mlir codegen.toy
```

### Example Session

**Input (`codegen.toy`):**
```toy
def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  var c = a + b;
  print(c);
}
```

**Output (MLIR):**
```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
    %1 = toy.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]> : tensor<6xf64>
    %2 = toy.reshape(%1 : tensor<6xf64>) to tensor<2x3xf64>
    %3 = toy.add %0, %2 : tensor<2x3xf64>
    toy.print %3 : tensor<2x3xf64>
    toy.return
  }
}
```

### Understanding the Output

Compare this to the AST from Chapter 2:

**AST (hierarchical tree):**
```
FunctionAST: main
└── Body
    ├── VarDeclExprAST: a
    ├── VarDeclExprAST: b
    ├── VarDeclExprAST: c
    │   └── BinaryExprAST: +
    └── PrintExprAST
```

**MLIR (flat SSA form):**
```
%0 = constant
%1 = constant
%2 = reshape
%3 = add %0, %2
print %3
return
```

MLIR is:
- ✅ Flat (no deep nesting)
- ✅ Explicit (types visible)
- ✅ SSA form (each value defined once)
- ✅ Verifiable (checked for correctness)

---

## 3.13 Why MLIR Is Better Than AST

Let's revisit the problems from Chapter 2 and see how MLIR solves them.

### Problem 1: Hard to Analyze

**AST:** No semantic information, just syntax

**MLIR Solution:**
- Types are explicit: `tensor<2x3xf64>`
- Operations have defined semantics
- Traits express properties (`Pure`, `Terminator`)
- Built-in verification checks correctness

### Problem 2: Hard to Transform

**AST:** Tree structure tied to surface syntax

**MLIR Solution:**
- Operations are first-class (not nested in trees)
- Pattern matching infrastructure (next chapter!)
- SSA form makes dependencies clear
- Dialects can coexist and interoperate

### Problem 3: Hard to Lower

**AST:** Huge gap between `x + y` and machine code

**MLIR Solution:**
- Progressive lowering through dialects
- `toy.add` → `linalg.generic` → `affine.for` → `llvm.add`
- Each step is a manageable transformation
- Reuse standard dialects

### Problem 4: No Reusability

**AST:** Every language needs its own tools

**MLIR Solution:**
- Common infrastructure for all dialects
- Shared pass system
- Shared type system
- Shared optimization framework

---

## Summary

Let's recap what we've learned:

### Key Concepts

1. **Dialects are Namespaces**
   - Custom vocabularies for specific domains
   - Operations prefixed with dialect name
   - Multiple dialects can coexist

2. **Operations Have Structure**
   - Operands (inputs)
   - Results (outputs)
   - Attributes (compile-time data)
   - Regions (nested code)

3. **TableGen Generates Boilerplate**
   - Define operations declaratively
   - Automatic C++ code generation
   - Reduces errors and repetition

4. **TableGen Syntax**
   - `class` = template
   - `def` = concrete instance
   - `let` = field binding
   - DAG syntax for structured data

5. **MLIRGen Translates AST to MLIR**
   - OpBuilder constructs operations
   - Symbol table tracks variables
   - Recursive traversal of AST
   - Verification ensures correctness

6. **MLIR Is Superior to AST**
   - Explicit semantics
   - Designed for transformation
   - Supports progressive lowering
   - Reusable infrastructure

### What We Built

- Complete Toy dialect definition (8 operations)
- TableGen specifications for each operation
- MLIRGen to convert AST → MLIR
- Working compiler: Toy source → MLIR

---

## What's Next

In **Chapter 4**, we'll learn how to **transform** MLIR code. We'll explore:

- Pattern matching and rewriting
- Declarative Rewrite Rules (DRR)
- Canonicalization (simplification)
- Optimization passes

We'll see transformations like:
- `transpose(transpose(x))` → `x`
- `reshape(reshape(x))` → `reshape(x)`
- `reshape(constant(x))` → `constant(reshape(x))`

This is where MLIR's power really shines!

---

## Exercises

### Exercise 1: Add a Subtraction Operation

Define a `SubOp` in TableGen that subtracts two tensors.

**Steps:**
1. Add to `Ops.td`:
   ```tablegen
   def SubOp : Toy_Op<"sub"> {
     let summary = "element-wise subtraction operation";
     // ... complete the definition
   }
   ```

2. Update `MLIRGen.cpp` to handle subtraction in `mlirGen(BinaryExprAST &binop)`

3. Rebuild and test with: `var c = a - b;`

<details>
<summary>Solution</summary>

```tablegen
def SubOp : Toy_Op<"sub"> {
  let summary = "element-wise subtraction operation";
  let description = [{
    The "sub" operation performs element-wise subtraction between two tensors.
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);
  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}
```

In `MLIRGen.cpp`:
```cpp
case '-':
  return builder.create<SubOp>(location, lhs, rhs);
```
</details>

### Exercise 2: Understand Generated Code

1. Build the project
2. Find the generated file `build/toy/Ch2/Ops.h.inc`
3. Search for `class AddOp`
4. Read through the generated code

**Questions:**
- What methods are generated?
- How are operands accessed?
- Where does verification happen?

### Exercise 3: Trace MLIRGen Execution

Add debug print statements to `MLIRGen.cpp`:

```cpp
mlir::Value mlirGen(BinaryExprAST &binop) {
  llvm::errs() << "Generating binary op: " << binop.getOp() << "\n";
  // ...
}
```

Run the compiler and observe the order of function calls.

### Exercise 4: Custom Assembly Format

The `TransposeOp` uses declarative assembly format:

```tablegen
let assemblyFormat = [{
  `(` $input `:` type($input) `)` attr-dict `to` type(results)
}];
```

Modify it to print as:
```mlir
%1 = toy.transpose %0 : tensor<2x3xf64> => tensor<3x2xf64>
```

**Hint:** Change the format string to use `=>` instead of `to`.

---

## Further Reading

### MLIR Documentation
- **MLIR Language Reference**: [https://mlir.llvm.org/docs/LangRef/](https://mlir.llvm.org/docs/LangRef/)
  - Complete specification of MLIR syntax and semantics
  
- **Operation Definition Specification (ODS)**: [https://mlir.llvm.org/docs/DefiningDialects/Operations/](https://mlir.llvm.org/docs/DefiningDialects/Operations/)
  - Comprehensive ODS/TableGen guide

- **Toy Tutorial**: [https://mlir.llvm.org/docs/Tutorials/Toy/](https://mlir.llvm.org/docs/Tutorials/Toy/)
  - Official MLIR tutorial (which this book expands upon)

### TableGen Resources
- **TableGen Overview**: [https://llvm.org/docs/TableGen/](https://llvm.org/docs/TableGen/)
  - LLVM's TableGen documentation

- **TableGen Programmer's Reference**: [https://llvm.org/docs/TableGen/ProgRef.html](https://llvm.org/docs/TableGen/ProgRef.html)
  - Complete language specification

### Papers
- **MLIR: Scaling Compiler Infrastructure for Domain Specific Computation** (CGO 2021)
  - [https://dl.acm.org/doi/10.1109/CGO51591.2021.9370308](https://dl.acm.org/doi/10.1109/CGO51591.2021.9370308)

---

## Reflection Questions

Before moving to Chapter 4, consider:

1. **Why is SSA form useful?**
   - How does defining each value once help optimization?
   - What problems does it prevent?

2. **What makes a good dialect design?**
   - How granular should operations be?
   - When should you create a new dialect vs. extending existing ones?

3. **TableGen trade-offs**
   - What are the benefits of code generation?
   - What are the costs (complexity, learning curve)?
   - When is manual C++ better?

Keep these questions in mind as we explore transformations in the next chapter!

---

**Next up:** Chapter 4, where we learn to optimize MLIR code through pattern matching and rewriting! 🔧
