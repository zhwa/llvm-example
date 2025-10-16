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

## 3.8 From AST to MLIR: MLIRGen

Now that we've defined operations in TableGen, let's see how to create them from the AST.

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

## 3.9 Complete Example: Toy to MLIR

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

## 3.10 Building and Running Chapter 2

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

## 3.11 Why MLIR Is Better Than AST

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
