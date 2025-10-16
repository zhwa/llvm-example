# Chapter 8: TableGen Mastery

> *"Any sufficiently advanced technology is indistinguishable from magic." - Arthur C. Clarke*

## Introduction

Throughout this book, we've been using **TableGen** - LLVM's domain-specific language for generating C++ code. We've seen it in action:

- **Chapter 3**: Defining operations with `Ops.td`
- **Chapter 4**: Writing transformation patterns with `ToyCombine.td`
- **Chapter 5**: Creating interfaces with `ShapeInferenceInterface.td`

But we've only scratched the surface. TableGen is a powerful metaprogramming tool that can:
- Generate thousands of lines of C++ from concise specifications
- Enforce constraints at code-generation time
- Provide type-safe abstractions
- Enable maintainable, declarative code

In this chapter, you'll master:
- TableGen language fundamentals
- ODS (Operation Definition Specification)
- Advanced pattern matching
- Constraint systems
- Custom backends

By the end, you'll be able to leverage TableGen's full power to build your own dialects and transformations efficiently!

Let's demystify the magic! ✨

---

## 8.1 The History and Philosophy of TableGen

### Why TableGen Was Created

**Historical context:**

In the early days of LLVM (circa 2003), Chris Lattner faced a problem:
- **Target descriptions** required massive amounts of repetitive C++ code
- Each CPU architecture (x86, ARM, PowerPC, etc.) needed:
  - Instruction definitions (1000+ instructions per target)
  - Register definitions (dozens to hundreds)
  - Calling conventions
  - Scheduling information
  - Assembly parsing/printing rules

**The pain points:**
```cpp
// Without TableGen: Manual C++ for EVERY instruction
class ADD_Instruction : public X86Instruction {
  static const char *getAsmString() { return "add"; }
  static unsigned getOpcode() { return 0x01; }
  static unsigned getNumOperands() { return 2; }
  static OperandType getOperandType(unsigned i) {
    return i == 0 ? Register : Register;
  }
  // ... 50+ more lines per instruction
};

class SUB_Instruction : public X86Instruction {
  // ... repeat everything again
};
// Repeat for 1000+ instructions!
```

**The realization**: Most information is **declarative**, not algorithmic!

### The TableGen Solution

**Core insight**: Use a **domain-specific language** for **data specification**:

```tablegen
// TableGen: Concise, declarative
def ADD : Instruction {
  let Mnemonic = "add";
  let Opcode = 0x01;
  let NumOperands = 2;
  let OperandTypes = [Register, Register];
}

def SUB : Instruction {
  let Mnemonic = "sub";
  let Opcode = 0x29;
  let NumOperands = 2;
  let OperandTypes = [Register, Register];
}
```

**Benefits:**
- ✅ **80-90% less code** compared to hand-written C++
- ✅ **Easier to maintain**: Change structure once, regenerate
- ✅ **Fewer bugs**: Generated code is consistent
- ✅ **Domain experts** can contribute without deep C++ knowledge
- ✅ **Multiple backends**: Generate different outputs from same spec

### TableGen's Design Principles

1. **Declarative over Imperative**
   - Describe WHAT, not HOW
   - Let code generator figure out the HOW

2. **Data-Oriented**
   - Focus on relationships between data
   - Not about algorithms or control flow

3. **Compile-Time Code Generation**
   - All processing at build time
   - Zero runtime overhead

4. **Single Source of Truth**
   - One .td file → multiple .inc files
   - No duplication between C++ files

5. **Gradual Typing**
   - Classes provide structure
   - Records are instances
   - Inheritance for reuse

### Evolution Timeline

**2003**: Created for LLVM target descriptions
- Initial use: x86, PowerPC instruction sets
- Focus: Instruction encoding, assembly parsing

**2006-2010**: Expanded to more backends
- ARM, MIPS, SPARC, etc.
- Added scheduling models
- Calling convention descriptions

**2018-2020**: MLIR adoption
- Operation Definition Specification (ODS)
- Pattern rewriting (DRR)
- Interface definitions
- Attribute/type definitions

**Today (2025)**: Industry standard
- Used by: LLVM, MLIR, Flang, CIRCT
- Generates: C++, documentation, tests
- Ecosystem: Multiple backends, tools

### Why TableGen for MLIR?

MLIR inherited TableGen from LLVM for good reasons:

**1. Proven technology**
- Battle-tested in LLVM for 20+ years
- Handles complexity (1000s of operations)
- Known to compiler developers

**2. Perfect fit for operations**
- Operations are data (name, operands, results, traits)
- Need to generate: C++ classes, parsers, printers, verifiers
- Same problem as LLVM instructions!

**3. Extensibility**
- MLIR's dialects = LLVM's targets
- Same need for custom, domain-specific operations
- TableGen's flexibility enables this

**4. Ecosystem benefits**
- Shared tooling with LLVM
- Familiar to LLVM developers
- Reduces learning curve for migration

---

## 8.2 What Is TableGen?

### The Core Idea

**TableGen** is a **code generator**. You write:
- Declarative specifications (`.td` files)
- TableGen processes them
- C++ code is generated

**Analogy:**
- **TableGen** is like a template engine (Jinja, Mustache)
- But for C++ code generation
- With type checking and constraint validation

### Why TableGen?

**Without TableGen:**
```cpp
// Defining operation manually (100+ lines)
class AddOp : public Op<AddOp, OpTrait::NOperands<2>::Impl,
                              OpTrait::OneResult,
                              OpTrait::ZeroSuccessors,
                              OpTrait::ZeroRegions,
                              OpTrait::Pure> {
public:
  using Op::Op;
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
  
  // More boilerplate...
};
```

**With TableGen:**
```tablegen
def AddOp : Toy_Op<"add", [Pure]> {
  let summary = "element-wise addition operation";
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);
  let hasCustomAssemblyFormat = 1;
}
```

**8 lines vs 100+ lines!** TableGen generates all the boilerplate.

### How It Works

```
┌──────────────┐
│  Ops.td      │  ← Your TableGen specification
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ mlir-tblgen  │  ← TableGen compiler
└──────┬───────┘
       │
       ↓
┌──────────────┐
│  Ops.h.inc   │  ← Generated C++ declarations
│  Ops.cpp.inc │  ← Generated C++ definitions
└──────────────┘
```

**The pipeline:**
1. Write `.td` files (TableGen DSL)
2. Run `mlir-tblgen` (code generator)
3. Include generated `.inc` files in your C++
4. Compile final binary

---

## 8.2 TableGen Language Basics

### Records: The Foundation

Everything in TableGen is a **record**. A record is like a struct or class.

**Simple record:**
```tablegen
def MyRecord {
  string name = "example";
  int value = 42;
  bit isActive = 1;
}
```

**What this creates:**
- A record named `MyRecord`
- Three fields: `name`, `value`, `isActive`
- With specified values

### Classes: Templates for Records

**Classes** are like C++ templates - they define structure without creating instances.

```tablegen
class Animal<string species> {
  string type = species;
  int legs;
  bit canFly = 0;
}

def Dog : Animal<"canine"> {
  let legs = 4;
}

def Bird : Animal<"avian"> {
  let legs = 2;
  let canFly = 1;
}
```

**Key syntax:**
- `class Name<params>` - Define a template
- `def Name : Class<args>` - Create an instance
- `let field = value` - Override field values

### Types in TableGen

TableGen has a simple type system:

| Type | Description | Example |
|------|-------------|---------|
| `bit` | Boolean (0 or 1) | `bit isConst = 1;` |
| `int` | Integer | `int size = 10;` |
| `string` | String literal | `string name = "foo";` |
| `bits<n>` | N-bit value | `bits<8> byte = 0xFF;` |
| `list<T>` | List of type T | `list<int> values = [1, 2, 3];` |
| `dag` | Directed acyclic graph | `dag pattern = (OpName $arg);` |
| `code` | C++ code snippet | `code impl = [{...}];` |

### The DAG Type

**DAG** (Directed Acyclic Graph) is TableGen's most powerful type. It represents tree structures.

**Syntax:**
```tablegen
dag example = (Operator arg1, arg2, arg3);
```

**Components:**
- `Operator`: The node type (root of subtree)
- `arg1, arg2, ...`: Children (can be DAGs themselves)
- Can have names: `(Operator $name1:arg1, $name2:arg2)`

**Example:**
```tablegen
// Represents: add(lhs, rhs)
dag addPattern = (AddOp $lhs, $rhs);

// Represents: add(mul(a, b), c)
dag complexPattern = (AddOp (MulOp $a, $b), $c);
```

DAGs are used for:
- Operation arguments: `(ins Type:$name, ...)`
- Operation results: `(outs Type, ...)`
- Pattern matching: `(OpName $arg1, $arg2)`

### Let Statements

**Let** binds values to fields:

```tablegen
class MyClass {
  int field1;
  int field2 = 10;  // Default value
}

def MyInstance : MyClass {
  let field1 = 5;     // Set field1
  let field2 = 20;    // Override field2
}
```

**Rules:**
- Must set fields without defaults
- Can override fields with defaults
- Evaluated at definition time

### Multiclass: Macro Expansion

**Multiclass** generates multiple definitions:

```tablegen
multiclass BinaryOp<string mnemonic> {
  def _Op : Toy_Op<mnemonic, [Pure]> {
    let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
    let results = (outs F64Tensor);
  }
  
  def _Lowering : Pattern<...> {
    // Lowering pattern
  }
}

// Generates: Add_Op, Add_Lowering
defm Add : BinaryOp<"add">;

// Generates: Mul_Op, Mul_Lowering
defm Mul : BinaryOp<"mul">;
```

**`defm`** = "define multiple" - expands the multiclass.

---

## 8.3 Operation Definition Specification (ODS)

ODS is MLIR's TableGen framework for defining operations.

### The Op Class Hierarchy

```tablegen
class Op<Dialect dialect, string mnemonic, list<Trait> traits = []> {
  Dialect opDialect = dialect;
  string opName = mnemonic;
  string summary = "";
  string description = "";
  dag arguments = (ins);
  dag results = (outs);
  // ... many more fields
}
```

**Your operation:**
```tablegen
class Toy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;

def AddOp : Toy_Op<"add", [Pure]> {
  // Inherits all fields from Op
  // Can override or set them
}
```

### Operation Components

Every operation has:

**1. Mnemonic** (operation name)
```tablegen
def AddOp : Toy_Op<"add", ...> {
  // Creates operation: toy.add
}
```

**2. Traits** (compile-time properties)
```tablegen
def AddOp : Toy_Op<"add", [Pure, Commutative]> {
  // Pure: No side effects
  // Commutative: add(a,b) = add(b,a)
}
```

**3. Arguments** (inputs)
```tablegen
let arguments = (ins 
  F64Tensor:$lhs,      // First operand named "lhs"
  F64Tensor:$rhs,      // Second operand named "rhs"
  I64Attr:$value       // Attribute named "value"
);
```

**4. Results** (outputs)
```tablegen
let results = (outs F64Tensor);     // One result
let results = (outs F64Tensor:$output);  // Named result
let results = (outs F64Tensor, I32);     // Multiple results
```

**5. Regions** (nested IR)
```tablegen
let regions = (region AnyRegion:$body);  // For ops with nested code
```

**6. Successors** (control flow targets)
```tablegen
let successors = (successor AnySuccessor:$trueBranch,
                           AnySuccessor:$falseBranch);
```

### Arguments in Detail

**Operands vs. Attributes:**

```tablegen
let arguments = (ins
  // OPERANDS (runtime values, SSA)
  F64Tensor:$input,
  I32:$index,
  
  // ATTRIBUTES (compile-time constants)
  I64Attr:$size,
  StrAttr:$name,
  UnitAttr:$isConst     // Present or absent
);
```

**Key difference:**
- **Operands**: Runtime values (SSA values, can be results of other ops)
- **Attributes**: Compile-time constants (literals, known at IR creation)

**Variadic operands:**
```tablegen
let arguments = (ins Variadic<F64Tensor>:$inputs);
// Accepts 0 or more tensors
```

**Optional operands:**
```tablegen
let arguments = (ins Optional<F64Tensor>:$input);
// Accepts 0 or 1 tensor
```

### Type Constraints

Define what types are valid:

```tablegen
// Predefined constraints
F64Tensor           // tensor<...xf64>
I32                 // i32
AnyType             // Any type
AnyInteger          // Any integer width
AnyFloat            // Any float width

// Custom constraints
def F64Tensor : Type<CPred<"$_self.isa<RankedTensorType>() && "
                           "$_self.cast<RankedTensorType>().getElementType().isF64()">,
                     "f64 tensor">;
```

**`CPred`** = C++ predicate. Generates C++ code to check the type.

### Builders

**Builders** are convenience methods for creating operations:

```tablegen
let builders = [
  // Builder with explicit operands
  OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
    build($_builder, $_state, lhs.getType(), lhs, rhs);
  }]>,
  
  // Builder with type inference
  OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
    build($_builder, $_state, lhs, rhs);
  }]>
];
```

**Special variables:**
- `$_builder`: The `OpBuilder` instance
- `$_state`: The `OperationState` being built
- `$argName`: Access to builder arguments

**Generated C++ usage:**
```cpp
// User code
auto addOp = builder.create<AddOp>(loc, lhs, rhs);

// Calls generated builder
AddOp::build(builder, state, lhs, rhs);
```

### Assembly Format

Control how operations print/parse:

**Declarative format:**
```tablegen
let assemblyFormat = [{
  `(` $input `:` type($input) `)` attr-dict `to` type(results)
}];
```

**This generates:**
```mlir
toy.reshape(%0 : tensor<6xf64>) to tensor<2x3xf64>
```

**Format elements:**
- `` `literal` `` - Literal text
- `$name` - Variable (operand, attribute, result)
- `type($name)` - Type of variable
- `attr-dict` - Attribute dictionary (for extra attributes)
- `operands` - All operands
- `results` - All results

**Custom format:**
```tablegen
let hasCustomAssemblyFormat = 1;
// Must implement parseAddOp() and print() in C++
```

### Verification

Add extra checks beyond type constraints:

```tablegen
let hasVerifier = 1;
```

**Implement in C++:**
```cpp
LogicalResult AddOp::verify() {
  // Custom verification
  if (getLhs().getType() != getRhs().getType())
    return emitError("operand types must match");
  return success();
}
```

### Extra Declarations

Inject C++ code into generated class:

```tablegen
let extraClassDeclaration = [{
  // C++ code added to the class
  bool isCommutative() { return true; }
  
  Value getOtherOperand(Value operand) {
    return operand == getLhs() ? getRhs() : getLhs();
  }
}];
```

---

## 8.4 Complete Operation Example

Let's dissect a full operation definition:

```tablegen
def TransposeOp : Toy_Op<"transpose", [Pure]> {
  // 1. Documentation
  let summary = "transpose operation";
  let description = [{
    The transpose operation reverses the dimensions of its input tensor.
    
    Example:
      %result = toy.transpose(%input : tensor<2x3xf64>) to tensor<3x2xf64>
  }];

  // 2. Operands and attributes
  let arguments = (ins F64Tensor:$input);

  // 3. Results
  let results = (outs F64Tensor:$output);

  // 4. Assembly format
  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type($output)
  }];

  // 5. Builders
  let builders = [
    OpBuilder<(ins "Value":$input)>
  ];

  // 6. Enable features
  let hasCanonicalizer = 1;  // Can register canonicalization patterns
  let hasVerifier = 1;        // Has custom verification
  
  // 7. Extra methods
  let extraClassDeclaration = [{
    // Helper to get transposed shape
    ShapedType getTransposedType();
  }];
}
```

**What gets generated:**

```cpp
class TransposeOp : public Op<TransposeOp, ...traits...> {
public:
  // Constructor
  using Op::Op;
  
  // Name
  static StringRef getOperationName() { return "toy.transpose"; }
  
  // Accessors
  Value getInput() { return getOperand(0); }
  Value getOutput() { return getResult(0); }
  
  // Builders
  static void build(OpBuilder &builder, OperationState &state, Value input);
  
  // Parsing/printing (from assemblyFormat)
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  
  // Verification
  LogicalResult verify();
  
  // Canonicalization
  void getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context);
  
  // User-provided code
  ShapedType getTransposedType();
};
```

**~200 lines of C++ from ~30 lines of TableGen!**

---

## 8.5 Traits: Compile-Time Properties

**Traits** are compile-time properties attached to operations.

### Common Traits

```tablegen
// No side effects (pure computation)
Pure

// Commutative operation: op(a,b) = op(b,a)
Commutative

// Associative: op(op(a,b),c) = op(a,op(b,c))
Associative

// Idempotent: op(op(x)) = op(x)
Idempotent

// Terminator (ends a block)
Terminator

// Must be in specific parent
HasParent<"FuncOp">

// Results and operands have same type
SameOperandsAndResultType

// Results and operands have same shape
SameOperandsAndResultShape

// Results and operands have same element type
SameOperandsAndResultElementType
```

### Using Traits

```tablegen
def AddOp : Toy_Op<"add", [
    Pure,                           // No side effects
    Commutative,                    // a + b = b + a
    SameOperandsAndResultType       // Type consistency
  ]> {
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);
}
```

### Why Traits Matter

**Enable optimizations:**
```cpp
// Because AddOp is Commutative:
if (auto addOp = dyn_cast<AddOp>(op)) {
  if (addOp->hasTrait<OpTrait::Commutative>()) {
    // Can reorder operands for optimization
  }
}
```

**Enforce constraints:**
```cpp
// SameOperandsAndResultType automatically verifies:
if (getLhs().getType() != getResult().getType())
  return error("type mismatch");  // Caught automatically!
```

### Custom Traits

Define your own:

```tablegen
def ElementwiseTrait : NativeOpTrait<"ElementwiseOp"> {
  let cppNamespace = "::mlir::toy";
}

def AddOp : Toy_Op<"add", [ElementwiseTrait]> {
  // ...
}
```

**Implement in C++:**
```cpp
namespace mlir::toy {
template <typename ConcreteType>
class ElementwiseOp : public OpTrait::TraitBase<ConcreteType, ElementwiseOp> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    // Verification logic
    return success();
  }
};
}
```

---

## 8.6 Interfaces in TableGen

We covered interfaces in Chapter 5. Here's the TableGen perspective.

### Defining an Interface

```tablegen
def MyOpInterface : OpInterface<"MyOpInterface"> {
  let description = [{
    Interface for operations that support my custom analysis.
  }];

  let methods = [
    // Method with no arguments
    InterfaceMethod<
      "Get the magic number",
      "int", "getMagicNumber"
    >,
    
    // Method with arguments
    InterfaceMethod<
      "Check compatibility",
      "bool", "isCompatible",
      (ins "Type":$otherType)
    >,
    
    // Method with default implementation
    InterfaceMethod<
      "Get operand count",
      "unsigned", "getNumOperands",
      (ins), [{}],  // Empty args
      [{
        return $_op.getNumOperands();  // Default impl
      }]
    >
  ];
}
```

### Method Signature

```tablegen
InterfaceMethod<
  "description",      // Documentation
  "returnType",       // C++ return type
  "methodName",       // Method name
  (ins "Type":$arg),  // Arguments (optional)
  [{}],               // Argument defaults (optional)
  [{                  // Default implementation (optional)
    // C++ code
  }]
>
```

### Declaring Interface Implementation

```tablegen
def MyOp : Toy_Op<"my_op", [
    DeclareOpInterfaceMethods<MyOpInterface>
  ]> {
  // Declares that MyOp implements all MyOpInterface methods
}

def AnotherOp : Toy_Op<"another", [
    DeclareOpInterfaceMethods<MyOpInterface, ["getMagicNumber"]>
  ]> {
  // Only implements getMagicNumber
  // Uses default implementations for others
}
```

### Complete Example

```tablegen
// 1. Define interface
def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  let methods = [
    InterfaceMethod<
      "Infer and set the output shape for the current operation.",
      "void", "inferShapes"
    >
  ];
}

// 2. Attach to operations
def AddOp : Toy_Op<"add", [
    Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>
  ]> {
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);
}

// 3. Implement in C++
void AddOp::inferShapes() {
  getResult().setType(getLhs().getType());
}
```

---

## 8.7 Declarative Rewrite Rules (DRR)

DRR lets you write transformations in TableGen instead of C++.

### Basic Pattern Structure

```tablegen
Pattern<
  (sourcePattern),        // What to match
  (resultPattern),        // What to replace with
  [(constraints)],        // Optional constraints
  (addBenefit benefit)    // Optional benefit
>
```

### Simple Pattern

```tablegen
// Reshape(Reshape(x)) → Reshape(x)
def ReshapeReshapeOptPattern : Pat<
  (ReshapeOp (ReshapeOp $arg)),  // Match nested reshapes
  (ReshapeOp $arg)                // Replace with single reshape
>;
```

**What this generates:**
```cpp
struct ReshapeReshapeOptPattern : public OpRewritePattern<ReshapeOp> {
  LogicalResult matchAndRewrite(ReshapeOp op, PatternRewriter &rewriter) const {
    // Check if input is also a ReshapeOp
    auto inputReshape = op.getOperand().getDefiningOp<ReshapeOp>();
    if (!inputReshape)
      return failure();
    
    // Replace outer reshape with inner reshape's input
    rewriter.replaceOp(op, inputReshape.getOperand());
    return success();
  }
};
```

### Variables in Patterns

**`$name`** captures values:

```tablegen
Pat<
  (AddOp $lhs, $rhs),     // Capture operands
  (MulOp $rhs, $lhs)      // Reuse in different order
>
```

**Named captures:**
```tablegen
Pat<
  (AddOp:$op $lhs, $rhs),  // Name the operation itself
  ...
>
```

**Use in constraints or native calls:**
```tablegen
Pat<
  (ReshapeOp:$res $arg),
  (replaceWithValue $arg),
  [(TypesMatch $res, $arg)]  // Use captured values
>
```

### Constraints

**Constraints** are boolean conditions:

```tablegen
// Define constraint
def TypesAreIdentical : Constraint<
  CPred<"$0.getType() == $1.getType()">,
  "types must be identical"
>;

// Use in pattern
def RedundantReshapeOptPattern : Pat<
  (ReshapeOp:$res $arg),
  (replaceWithValue $arg),
  [(TypesAreIdentical $res, $arg)]
>;
```

**Built-in constraint helpers:**
```tablegen
// Check attribute value
Constraint<CPred<"$0.getValue() == 0">>

// Check operation property
Constraint<CPred<"$0.hasTrait<OpTrait::Commutative>()">>

// Check type property
Constraint<CPred<"$0.getType().isa<TensorType>()">>
```

### Native Code Calls

**NativeCodeCall** executes C++ to compute results:

```tablegen
// Define native code
def ReshapeConstant : NativeCodeCall<
  "$0.reshape(::llvm::cast<ShapedType>($1.getType()))"
>;

// Use in pattern
def FoldConstantReshapeOptPattern : Pat<
  (ReshapeOp:$res (ConstantOp $arg)),
  (ConstantOp (ReshapeConstant $arg, $res))
>;
```

**What happens:**
```cpp
// Generated code
auto result = arg.reshape(llvm::cast<ShapedType>(res.getType()));
rewriter.replaceOpWithNewOp<ConstantOp>(op, result);
```

**Parameter substitution:**
- `$0`, `$1`, ... = Arguments to NativeCodeCall
- `$_builder` = PatternRewriter
- `$_loc` = Location
- `$_self` = The matched value

### Multiple Results

Some patterns produce multiple values:

```tablegen
def SplitPattern : Pat<
  (SomeOp $input),
  [(FirstOp $input), (SecondOp $input)]
>;
```

Or use auxiliary patterns:

```tablegen
def ComplexPattern : Pattern<
  (dag source),
  [(dag result1), (dag result2)],
  [],  // constraints
  [(dag auxiliary1), (dag auxiliary2)]  // supplemental patterns
>;
```

### Benefits

Assign priority to patterns:

```tablegen
def HighPriorityPattern : Pat<
  ...,
  ...,
  [],
  (addBenefit 10)  // Higher benefit = higher priority
>;

def LowPriorityPattern : Pat<
  ...,
  ...,
  [],
  (addBenefit 1)
>;
```

Higher benefit patterns are tried first.

---

## 8.8 Advanced Pattern Techniques

### Matching Attributes

```tablegen
// Match specific constant value
def ZeroConstant : Pat<
  (AddOp $x, (ConstantOp ConstantAttr<I32, "0">)),
  (replaceWithValue $x)
>;
```

**`ConstantAttr<Type, "value">`** matches constant with specific value.

### Matching Types

```tablegen
// Only match tensors
def TensorOnlyPattern : Pat<
  (AddOp $lhs:$TensorType, $rhs:$TensorType),
  ...
>;
```

**Type constraints:**
- `$name:$TypeConstraint` - Operand must satisfy type constraint

### Nested Patterns

```tablegen
// Match: add(mul(a, b), mul(a, c))
def DistributivePattern : Pat<
  (AddOp (MulOp $a, $b), (MulOp $a, $c)),
  (MulOp $a, (AddOp $b, $c))
>;
```

Matches arbitrarily deep nesting!

### Conditional Rewriting

```tablegen
def ConditionalPattern : Pat<
  (SomeOp $arg),
  (NewOp $arg),
  [(SomeCondition $arg)]
>;
```

Only applies if condition is true.

### Using Operation Results

```tablegen
def UseResultPattern : Pat<
  (SomeOp:$op $input),
  (NewOp $input, $op)  // Can use $op (the operation) as well as its operands
>;
```

### Binding Intermediate Results

```tablegen
def MultiStepPattern : Pattern<
  (SomeOp $input),
  [(NewOp1 (NewOp2 $input))]
>;
```

Generates intermediate operations in order.

---

## 8.9 Real-World Pattern Examples

### Example 1: Constant Folding

```tablegen
// add(const(a), const(b)) → const(a + b)
def FoldConstantAdd : Pat<
  (AddOp (ConstantOp $a), (ConstantOp $b)),
  (ConstantOp (NativeCodeCall<"$0 + $1"> $a, $b))
>;
```

### Example 2: Strength Reduction

```tablegen
// mul(x, 2) → add(x, x)
def MulByTwoToAdd : Pat<
  (MulOp $x, (ConstantOp ConstantAttr<F64Attr, "2.0">)),
  (AddOp $x, $x)
>;
```

### Example 3: Algebraic Simplification

```tablegen
// add(x, 0) → x
def AddZeroPattern : Pat<
  (AddOp $x, (ConstantOp ConstantAttr<F64Attr, "0.0">)),
  (replaceWithValue $x)
>;

// mul(x, 1) → x
def MulOnePattern : Pat<
  (MulOp $x, (ConstantOp ConstantAttr<F64Attr, "1.0">)),
  (replaceWithValue $x)
>;

// mul(x, 0) → 0
def MulZeroPattern : Pat<
  (MulOp $x, (ConstantOp:$zero ConstantAttr<F64Attr, "0.0">)),
  (replaceWithValue $zero)
>;
```

### Example 4: Canonicalization

```tablegen
// Double transpose: transpose(transpose(x)) → x
def RemoveDoubleTranspose : Pat<
  (TransposeOp (TransposeOp $arg)),
  (replaceWithValue $arg)
>;

// Double negation: neg(neg(x)) → x
def RemoveDoubleNegate : Pat<
  (NegOp (NegOp $arg)),
  (replaceWithValue $arg)
>;
```

### Example 5: Fusion

```tablegen
// Fuse consecutive adds: add(add(a, b), c) → add_three(a, b, c)
def FuseAdds : Pat<
  (AddOp (AddOp $a, $b), $c),
  (AddThreeOp $a, $b, $c),
  [(HasOneUse)]  // Only if inner add has no other users
>;
```

---

## 8.10 TableGen Best Practices

### 1. Use Descriptive Names

**Bad:**
```tablegen
def Op1 : Toy_Op<"op1"> {
  let arguments = (ins F64Tensor:$a, F64Tensor:$b);
}
```

**Good:**
```tablegen
def MatrixMultiplyOp : Toy_Op<"matmul"> {
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let summary = "matrix multiplication operation";
}
```

### 2. Document Everything

```tablegen
def MyOp : Toy_Op<"my_op"> {
  let summary = "one-line description";
  let description = [{
    Multi-line detailed description.
    
    Explain:
    - What the operation does
    - Constraints and requirements
    - Examples
    
    Example:
      %result = toy.my_op(%input) : tensor<10xf64>
  }];
}
```

### 3. Organize with Comments

```tablegen
//===----------------------------------------------------------------------===//
// Arithmetic Operations
//===----------------------------------------------------------------------===//

def AddOp : Toy_Op<...> { ... }
def SubOp : Toy_Op<...> { ... }
def MulOp : Toy_Op<...> { ... }

//===----------------------------------------------------------------------===//
// Shape Operations
//===----------------------------------------------------------------------===//

def ReshapeOp : Toy_Op<...> { ... }
def TransposeOp : Toy_Op<...> { ... }
```

### 4. Use Multiclass for Common Patterns

```tablegen
multiclass BinaryArithOp<string mnemonic, string loweredOp> {
  def _Op : Toy_Op<mnemonic, [Pure, Commutative]> {
    let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
    let results = (outs F64Tensor);
    let hasCustomAssemblyFormat = 1;
  }
  
  def _Lowering : Pat<
    (!cast<Op>(NAME # "_Op") $lhs, $rhs),
    (!cast<Op>(loweredOp) $lhs, $rhs)
  >;
}

defm Add : BinaryArithOp<"add", "arith.addf">;
defm Mul : BinaryArithOp<"mul", "arith.mulf">;
```

### 5. Leverage Type Constraints

```tablegen
// Instead of manual verification
def BadOp : Toy_Op<"bad"> {
  let arguments = (ins AnyType:$input);
  let hasVerifier = 1;  // Need to check type manually
}

// Use type constraints
def GoodOp : Toy_Op<"good"> {
  let arguments = (ins F64Tensor:$input);
  // Type checking automatic!
}
```

### 6. Prefer Declarative Assembly Format

```tablegen
// Instead of custom parsing/printing
def CustomOp : Toy_Op<"custom"> {
  let hasCustomAssemblyFormat = 1;
  // Must write parseCustomOp() and print() - 50+ lines
}

// Use declarative format
def DeclarativeOp : Toy_Op<"declarative"> {
  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)
  }];
  // Generated automatically!
}
```

### 7. Test Generated Code

```cpp
// Check what TableGen generates
#include "toy/Ops.h.inc"  // Look at this file!

// Verify:
// - Correct accessors generated
// - Builders work as expected
// - Verification catches errors
```

---

## 8.11 Debugging TableGen - A Complete Guide

Debugging TableGen can be challenging because errors happen at **code generation time**, not runtime. Here's how to debug effectively.

### Understanding the TableGen Workflow

```
Your .td file
      ↓
  mlir-tblgen (parser)
      ↓
  Internal representation (records)
      ↓
  Backend (code generator)
      ↓
  Generated .inc files
      ↓
  C++ compiler
      ↓
  Your executable
```

**Errors can occur at ANY stage!**

### Debug Strategy 1: Inspect Generated Code

**Always look at what TableGen generates:**

```powershell
# On Windows (PowerShell)
# Generate operation declarations
mlir-tblgen --gen-op-decls Ops.td -o Ops.h.inc
type Ops.h.inc

# Generate operation definitions
mlir-tblgen --gen-op-defs Ops.td -o Ops.cpp.inc
type Ops.cpp.inc

# Generate pattern rewrites
mlir-tblgen --gen-rewriters ToyCombine.td -o ToyCombine.inc
type ToyCombine.inc

# Generate dialect declarations
mlir-tblgen --gen-dialect-decls Ops.td -o Dialect.h.inc
type Dialect.h.inc
```

**What to look for:**
- Are the expected classes generated?
- Do method signatures match your expectations?
- Are includes correct?

**Example: Checking generated operation**

Your `.td`:
```tablegen
def AddOp : Toy_Op<"add", [Pure]> {
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);
}
```

Generated in `Ops.h.inc`:
```cpp
class AddOp : public ::mlir::Op<AddOp, /* traits */> {
public:
  // Check these are generated:
  Value getLhs();  // Accessor for $lhs
  Value getRhs();  // Accessor for $rhs
  static StringRef getOperationName() { return "toy.add"; }
};
```

### Debug Strategy 2: Use TableGen Debug Flags

**Print all records:**
```powershell
mlir-tblgen --print-records Ops.td
```

This shows EVERYTHING TableGen knows about:
- All defined records
- All classes
- All field values
- Inheritance relationships

**Example output:**
```
def AddOp {
  string Mnemonic = "add";
  dag arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  dag results = (outs F64Tensor);
  list<Trait> traits = [Pure];
  Dialect dialect = Toy_Dialect;
}
```

**Print specific records:**
```powershell
# Print only operation records
mlir-tblgen --print-records Ops.td | Select-String -Pattern "def.*Op"
```

**Dump with dependencies:**
```powershell
mlir-tblgen -d Ops.td.d --gen-op-decls Ops.td -o Ops.h.inc
type Ops.td.d
```

Shows all files that Ops.td depends on (useful for tracking include issues).

### Debug Strategy 3: Incremental Building

**Start simple, add complexity:**

```tablegen
// Step 1: Minimal operation (verify this works first)
def TestOp : Toy_Op<"test"> {
  let arguments = (ins);
  let results = (outs);
}

// Step 2: Add operands
def TestOp : Toy_Op<"test"> {
  let arguments = (ins I32:$input);
  let results = (outs);
}

// Step 3: Add results
def TestOp : Toy_Op<"test"> {
  let arguments = (ins I32:$input);
  let results = (outs I32);
}

// Step 4: Add traits, verifier, etc.
```

After each step, run `mlir-tblgen` and verify it compiles.

### Debug Strategy 4: Common Errors and Solutions

#### Error 1: Undefined Reference to Record

```
error: Undefined reference to record: 'F64Tensor'
```

**Cause:** Missing include or typo in type name.

**Debug steps:**
1. Check includes at top of file:
   ```tablegen
   include "mlir/IR/OpBase.td"
   include "mlir/Interfaces/SideEffectInterfaces.td"
   ```

2. Verify type is defined:
   ```tablegen
   // Should be defined somewhere
   def F64Tensor : Type<...>;
   ```

3. Check for typos: `F64Tensor` vs `F64Tensro`

**Solution:**
```tablegen
// Add missing include
include "toy/ToyTypes.td"

// Or define inline
def F64Tensor : Type<CPred<"$_self.isa<RankedTensorType>()">, "f64 tensor">;
```

#### Error 2: Type Mismatch in Pattern

```
error: Pattern type mismatch between source and result
```

**Cause:** Pattern's result type doesn't match operation's result type.

**Debug steps:**
1. Print the pattern:
   ```powershell
   mlir-tblgen --print-records ToyCombine.td | Select-String -Pattern "Pattern" -Context 5,5
   ```

2. Check types explicitly:
   ```tablegen
   // BAD: Types don't match
   def BadPattern : Pat<
     (AddOp F64Tensor:$lhs, I32:$rhs),  // F64 + I32
     (MulOp $lhs, $rhs)                  // Result is F64? I32? Unclear!
   >;
   
   // GOOD: Types are clear
   def GoodPattern : Pat<
     (AddOp F64Tensor:$lhs, F64Tensor:$rhs),
     (MulOp $lhs, $rhs)  // Both F64, result is F64
   >;
   ```

#### Error 3: Missing Required Field

```
error: Field 'arguments' is not initialized
```

**Cause:** Operation definition missing required fields.

**Debug steps:**
1. Check base class requirements:
   ```tablegen
   // What does Toy_Op require?
   class Toy_Op<string mnemonic, list<Trait> traits = []> : Op<...> {
     // Look at Op class to see required fields
   }
   ```

2. Check all required fields are set

**Solution:**
```tablegen
// BAD: Missing fields
def BadOp : Toy_Op<"bad"> {
  // arguments not set!
  // results not set!
}

// GOOD: All required fields set
def GoodOp : Toy_Op<"good"> {
  let arguments = (ins);  // Even if empty, must be present
  let results = (outs);
}
```

#### Error 4: DAG Syntax Errors

```
error: Expected ',' or ')' in dag init
```

**Cause:** Malformed DAG expression.

**Debug steps:**
1. Check parentheses balance
2. Check commas
3. Check for missing types

**Common mistakes:**
```tablegen
// BAD: Missing type
let arguments = (ins $input);  // ERROR: What type?

// GOOD: Type specified
let arguments = (ins I32:$input);

// BAD: Missing comma
let arguments = (ins I32:$a I32:$b);  // ERROR

// GOOD: Comma present
let arguments = (ins I32:$a, I32:$b);

// BAD: Missing parentheses
let arguments = ins I32:$a;  // ERROR

// GOOD: Parentheses
let arguments = (ins I32:$a);
```

#### Error 5: Circular Dependencies

```
error: Circular dependency detected in 'MyOp'
```

**Cause:** Class or record depends on itself.

**Debug steps:**
1. Draw dependency graph:
   ```
   MyOp → MyTrait → MyInterface → MyOp  (CIRCULAR!)
   ```

2. Break the cycle:
   ```tablegen
   // BAD: Circular
   class A : B { }
   class B : C { }
   class C : A { }  // Back to A!
   
   // GOOD: Linear
   class A : Base { }
   class B : Base { }
   class C : Base { }
   ```

### Debug Strategy 5: Use Verbose Mode

```powershell
# See what TableGen is doing
mlir-tblgen -v --gen-op-decls Ops.td 2>&1 | Tee-Object debug.log
```

Output shows:
- Files being parsed
- Records being created
- Includes being processed
- Backend invocations

### Debug Strategy 6: Diff Generated Files

When something breaks after a change:

```powershell
# Generate before change
mlir-tblgen --gen-op-defs Ops.td -o Ops.before.inc

# Make your change

# Generate after change
mlir-tblgen --gen-op-defs Ops.td -o Ops.after.inc

# Compare
diff Ops.before.inc Ops.after.inc
```

**What changed?** That's likely your bug!

### Debug Strategy 7: Minimal Reproduction

Create a minimal `.td` file that reproduces the error:

```tablegen
// minimal_repro.td
include "mlir/IR/OpBase.td"

def TestDialect : Dialect {
  let name = "test";
}

def TestOp : Op<TestDialect, "test"> {
  // Minimal operation that triggers error
  let arguments = (ins /* your problematic definition */);
}
```

Share this on LLVM Discourse for help!

### Debug Strategy 8: Check CMake Configuration

Sometimes errors are in the build system, not TableGen:

```cmake
# Verify TableGen is being called correctly
set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)  # Check this line
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_public_tablegen_target(MyDialectIncGen)

# Verify dependencies
add_dependencies(MyDialect MyDialectIncGen)
```

**Common CMake mistakes:**
- Wrong target name in `set(LLVM_TARGET_DEFINITIONS ...)`
- Missing `add_public_tablegen_target(...)`
- Missing dependency: `add_dependencies(...)`
- Wrong backend flag: `-gen-op-decls` vs `-gen-op-defs`

### Debugging Workflow Summary

```
1. Error appears
        ↓
2. Read error message (note file & line)
        ↓
3. Generate .inc files manually
        ↓
4. Inspect generated code
        ↓
5. Use --print-records to see data
        ↓
6. Create minimal reproduction
        ↓
7. Fix issue
        ↓
8. Verify with incremental changes
```

### Useful TableGen Commands Reference

```powershell
# View all records
mlir-tblgen --print-records Ops.td

# View specific backend output
mlir-tblgen --gen-op-decls Ops.td
mlir-tblgen --gen-op-defs Ops.td
mlir-tblgen --gen-rewriters Patterns.td
mlir-tblgen --gen-dialect-decls Ops.td
mlir-tblgen --gen-dialect-defs Ops.td
mlir-tblgen --gen-typedef-decls Types.td
mlir-tblgen --gen-attrdef-decls Attributes.td

# Debug mode
mlir-tblgen -v Ops.td
mlir-tblgen -d Ops.td.d --gen-op-decls Ops.td

# Check syntax only (no generation)
mlir-tblgen --print-records Ops.td > nul

# View help
mlir-tblgen --help
```

### When to Ask for Help

After trying the above, if still stuck:

1. **Create minimal reproduction** (< 50 lines)
2. **Show error message** (exact text)
3. **Show what you tried** (debugging steps)
4. **Post on LLVM Discourse**: https://discourse.llvm.org/c/mlir/

The community is helpful, but help them help you with clear information!

---

## 8.12 Advanced Topics

### Custom Backends

TableGen can generate code for any purpose:

```cpp
// Custom backend (in C++)
class MyBackend : public llvm::TableGenBackend {
  void run(raw_ostream &OS) override {
    // Read TableGen records
    // Generate custom code
    OS << "// Generated code\n";
  }
};
```

Register and use:
```powershell
mlir-tblgen --gen-my-backend Ops.td -o output.txt
```

### Record Filtering

```tablegen
// Define a marker
def NeedsSpecialHandling : OpInterface<"..."> { ... };

// Mark operations
def SpecialOp : Toy_Op<"special", [NeedsSpecialHandling]> { ... };
def NormalOp : Toy_Op<"normal"> { ... };
```

**In backend:**
```cpp
for (Record *def : Records.getAllDerivedDefinitions("Op")) {
  if (def->isSubClassOf("NeedsSpecialHandling")) {
    // Generate special code
  }
}
```

### Template Metaprogramming

```tablegen
class GenericOp<string mnemonic, Type inputType, Type outputType>
    : Toy_Op<mnemonic> {
  let arguments = (ins inputType:$input);
  let results = (outs outputType:$output);
}

def F64ToI32CastOp : GenericOp<"cast_f64_i32", F64, I32>;
def I32ToF64CastOp : GenericOp<"cast_i32_f64", I32, F64>;
```

Generate families of similar operations!

### Conditional Compilation

```tablegen
// Define feature flags
def EnableExperimentalOps : SubtargetFeature<...>;

// Conditionally include operations
#ifdef ENABLE_EXPERIMENTAL
def ExperimentalOp : Toy_Op<...> { ... }
#endif
```

Control what gets compiled based on configuration.

---

## 8.13 C++ vs TableGen: Lowering Comparison

One of the most common questions: **"Should I write my lowering in C++ or TableGen?"**

The answer: **It depends!** Let's compare with real examples.

### The Same Pattern, Two Ways

Let's implement the same optimization: **Reshape(Reshape(x)) → Reshape(x)**

This eliminates redundant reshape operations.

#### Approach 1: C++ Implementation

**File: `ToyCombine.cpp`**

```cpp
#include "toy/Dialect.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::toy;

namespace {

/// Fold reshape(reshape(x)) -> reshape(x)
struct SimplifyRedundantReshape : public OpRewritePattern<ReshapeOp> {
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp op,
                                 PatternRewriter &rewriter) const override {
    // Step 1: Get the input to this reshape
    Value input = op.getOperand();
    
    // Step 2: Check if input comes from another reshape
    auto producerOp = input.getDefiningOp<ReshapeOp>();
    if (!producerOp)
      return failure();  // Input is not a reshape
    
    // Step 3: Get the original input (before first reshape)
    Value originalInput = producerOp.getOperand();
    
    // Step 4: Create new reshape: reshape(originalInput) to final shape
    rewriter.replaceOpWithNewOp<ReshapeOp>(
        op,                    // Operation to replace
        op.getType(),          // Result type (final shape)
        originalInput          // Input (skip intermediate reshape)
    );
    
    return success();
  }
};

} // namespace

/// Register the pattern
void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.add<SimplifyRedundantReshape>(context);
}
```

**Characteristics of C++ approach:**

✅ **Pros:**
- **Full control**: Can implement ANY logic
- **Debugging**: Use debugger, print statements
- **Complex conditions**: Easy to write elaborate checks
- **Dynamic behavior**: Runtime decisions
- **IDE support**: Auto-complete, refactoring

❌ **Cons:**
- **Verbose**: ~30 lines for simple pattern
- **Boilerplate**: OpRewritePattern, matchAndRewrite, etc.
- **Manual registration**: Must call getCanonicalizationPatterns
- **Compile time**: Changes require C++ recompilation
- **Error-prone**: Easy to make mistakes in rewrite logic

#### Approach 2: TableGen Implementation (DRR)

**File: `ToyCombine.td`**

```tablegen
// Fold reshape(reshape(x)) -> reshape(x)
def ReshapeReshapeOptPattern : Pat<
  (ReshapeOp (ReshapeOp $arg)),  // Match: nested reshapes
  (ReshapeOp $arg)                // Replace: single reshape
>;
```

**That's it! 3 lines vs 30+ lines!**

**Characteristics of TableGen approach:**

✅ **Pros:**
- **Concise**: 90% less code
- **Declarative**: What, not how
- **Type-safe**: Verified at generation time
- **Maintainable**: Easy to read and modify
- **No boilerplate**: Pattern matching auto-generated
- **Auto-registration**: Automatically added to pattern set

❌ **Cons:**
- **Limited expressiveness**: Can't do complex logic
- **Harder to debug**: No debugger, must inspect generated code
- **Learning curve**: New syntax to learn
- **Static only**: Can't make runtime decisions

### Side-by-Side: More Complex Pattern

Let's do a more complex pattern: **Constant folding for reshape**

**Pattern**: `reshape(const(x)) → const(reshape(x))`

#### C++ Implementation

```cpp
/// Fold reshape(constant) -> constant
struct FoldConstantReshape : public OpRewritePattern<ReshapeOp> {
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp op,
                                 PatternRewriter &rewriter) const override {
    // Match: Check if input is a constant
    auto constantOp = op.getOperand().getDefiningOp<ConstantOp>();
    if (!constantOp)
      return failure();
    
    // Get constant value
    DenseElementsAttr value = constantOp.getValue();
    
    // Get reshape's result type (new shape)
    auto resultType = op.getType().cast<RankedTensorType>();
    
    // Reshape the constant attribute
    auto reshapedAttr = value.reshape(resultType);
    
    // Replace with new constant
    rewriter.replaceOpWithNewOp<ConstantOp>(op, reshapedAttr);
    
    return success();
  }
};
```

**~20 lines of C++**

#### TableGen Implementation (DRR)

```tablegen
// Define a native code call to reshape the constant
def ReshapeConstant : NativeCodeCall<
  "$0.reshape(::llvm::cast<ShapedType>($1.getType()))"
>;

// Pattern: reshape(const) -> const(reshaped_value)
def FoldConstantReshapeOptPattern : Pat<
  (ReshapeOp:$res (ConstantOp $arg)),
  (ConstantOp (ReshapeConstant $arg, $res))
>;
```

**~7 lines of TableGen** (including native code helper)

### When to Use Each Approach

#### Use **TableGen (DRR)** when:

1. **Pattern is structural**
   - Matching operation shapes: `op1(op2(x))`, `op1(x, const(y))`
   - Simple rewrites: replace with different operation
   - Algebraic simplifications: `x + 0 → x`, `x * 1 → x`

2. **Logic is declarative**
   - "If I see THIS, replace with THAT"
   - Type-based matching
   - Constant folding with native code calls

3. **Want conciseness**
   - Faster to write and maintain
   - Less code to review

4. **Pattern is common**
   - Canonicalization patterns
   - Peephole optimizations
   - Algebraic identities

**Examples from Toy dialect:**
```tablegen
// All these are perfect for TableGen:

// Double transpose: transpose(transpose(x)) → x
def RemoveDoubleTranspose : Pat<
  (TransposeOp (TransposeOp $arg)),
  (replaceWithValue $arg)
>;

// Transpose of constant: transpose(const(x)) → const(transpose(x))
def TransposeConstant : Pat<...>;

// Reshape to same shape: reshape(x) to same shape → x
def RedundantReshape : Pat<...>;
```

#### Use **C++** when:

1. **Logic is algorithmic**
   - Need loops, recursion
   - Complex data structure traversal
   - Multi-step transformations

2. **Need runtime information**
   - Check values (not just types)
   - Analyze def-use chains
   - Need to query analyses

3. **Complex conditions**
   - Multiple checks with &&, ||
   - Nested conditionals
   - Need helper functions

4. **Dialect-wide transformations**
   - Whole-function analysis
   - Inter-procedural optimization
   - Graph rewrites

5. **Need debugging**
   - Step through with debugger
   - Print intermediate states
   - Complex error handling

**Examples requiring C++:**

```cpp
// Shape inference: Need to analyze types and propagate shapes
class ShapeInferencePass : public PassWrapper<...> {
  void runOnOperation() override {
    getOperation().walk([](toy::FuncOp func) {
      // Analyze function, propagate shapes through calls
      // Too complex for TableGen!
    });
  }
};

// Dead code elimination: Need def-use analysis
struct DCE : public OpRewritePattern<Operation> {
  LogicalResult matchAndRewrite(Operation *op, ...) const override {
    // Check if operation has no users
    if (!op->use_empty())
      return failure();
    
    // Check if operation has side effects
    if (!isPure(op))
      return failure();
    
    rewriter.eraseOp(op);
    return success();
  }
};

// Loop fusion: Complex analysis
struct FuseAffineLoops : public OpRewritePattern<AffineForOp> {
  LogicalResult matchAndRewrite(...) {
    // Check dependency, check bounds, check profitability
    // Way too complex for TableGen!
  }
};
```

### Hybrid Approach: Best of Both Worlds

**You can combine both!**

```cpp
// In ToyCombine.cpp
void MyDialect::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  // Add TableGen-generated patterns
  populateWithGenerated(results);
  
  // Add C++ patterns for complex cases
  results.add<ComplexPatternRequiringCpp,
              AnotherComplexPattern>(context);
}
```

### Performance Comparison

**Compile time:**
- **TableGen**: Slightly slower build (code generation step)
- **C++**: Faster build (no generation)
- **Winner**: Depends on pattern complexity

**Runtime:**
- **TableGen**: Generated C++ code is just as fast
- **C++**: Hand-written code
- **Winner**: Tie (same performance)

**Development time:**
- **TableGen**: Much faster to write
- **C++**: More time needed
- **Winner**: TableGen for simple patterns

### Maintenance Comparison

**Updating patterns:**

**TableGen:**
```tablegen
// Before
def OldPattern : Pat<(OpA $x), (OpB $x)>;

// After: Just change one line
def NewPattern : Pat<(OpA $x), (OpC $x)>;
```

**C++:**
```cpp
// Before
struct OldPattern : public OpRewritePattern<OpA> {
  LogicalResult matchAndRewrite(OpA op, ...) {
    rewriter.replaceOpWithNewOp<OpB>(op, op.getOperand());
    return success();
  }
};

// After: Change multiple lines
struct NewPattern : public OpRewritePattern<OpA> {
  LogicalResult matchAndRewrite(OpA op, ...) {
    rewriter.replaceOpWithNewOp<OpC>(op, op.getOperand());
    return success();
  }
};
```

**Winner**: TableGen (much easier to maintain)

### Real-World Examples from MLIR

**TableGen dominates in:**
- **Arithmetic dialect**: Constant folding, algebraic simplifications
- **Vector dialect**: Vector-vector operations
- **Affine dialect**: Affine expression simplification

**C++ dominates in:**
- **Linalg transformations**: Tiling, fusion, vectorization
- **Shape inference**: Type propagation
- **Buffer optimization**: Buffer allocation, deallocation

### Decision Framework

```
Is it a simple structural pattern?
    ├─ Yes → Use TableGen
    │
    └─ No → Does it need runtime information?
            ├─ Yes → Use C++
            │
            └─ No → Does it need complex logic?
                    ├─ Yes → Use C++
                    │
                    └─ No → Use TableGen
```

### Recommendations

**Start with TableGen:**
- Try to express your pattern in DRR first
- If it's too complex, fall back to C++
- Most patterns (70-80%) can be done in TableGen!

**Use C++ when necessary:**
- Don't force complex logic into TableGen
- C++ is there for hard cases
- Hybrid approach is perfectly fine

**Mix and match:**
- Use TableGen for simple, common patterns
- Use C++ for complex, rare patterns
- Register both in your dialect

### Summary of Comparison

| Aspect | TableGen (DRR) | C++ |
|--------|----------------|-----|
| **Lines of code** | 3-10 | 20-50 |
| **Ease of writing** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Expressiveness** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Debugging** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Maintainability** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Type safety** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Runtime performance** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Compile time** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Learning curve** | ⭐⭐⭐ | ⭐⭐⭐⭐ |

**Conclusion**: Use TableGen when possible, C++ when necessary!

---

## Summary

Let's recap what we've mastered:

### Key Concepts

1. **TableGen Basics**
   - Records and classes
   - Types: bit, int, string, dag, list, code
   - Let statements and field bindings
   - Multiclass for macro expansion

2. **ODS (Operation Definition Specification)**
   - Op class hierarchy
   - Arguments (operands + attributes)
   - Results, regions, successors
   - Traits and interfaces
   - Builders and assembly format

3. **Interfaces**
   - OpInterface for polymorphism
   - InterfaceMethod definitions
   - Default implementations
   - Declaration with DeclareOpInterfaceMethods

4. **DRR (Declarative Rewrite Rules)**
   - Pattern structure: source → result
   - Variables and captures
   - Constraints for conditional rewriting
   - NativeCodeCall for C++ computations
   - Benefits for pattern prioritization

5. **Best Practices**
   - Descriptive names and documentation
   - Organization and modularity
   - Leverage type constraints
   - Prefer declarative over imperative
   - Test generated code

### What We Accomplished

- ✅ Understood TableGen language fundamentals
- ✅ Mastered ODS for operation definitions
- ✅ Wrote declarative transformation patterns
- ✅ Debugged TableGen specifications
- ✅ Learned advanced techniques

### The Power of TableGen

**Before TableGen:**
- 100+ lines of C++ per operation
- Error-prone boilerplate
- Hard to maintain
- Difficult to ensure consistency

**With TableGen:**
- 10-20 lines of declarative spec
- Generated code is correct by construction
- Easy to maintain and modify
- Consistency enforced automatically

**TableGen is the secret sauce that makes MLIR's extensibility possible!**

---

## What's Next

In **Chapter 9**, we'll apply everything we've learned to design **custom dialects** from scratch:

- Design principles for dialects
- When to create a new dialect vs. reusing existing ones
- Layering and composition
- Real-world dialect examples (ML, DSP, Graphics)

And in **Chapter 10**, we'll explore the **MLIR ecosystem**:
- Integration with existing tools
- Compilation pipelines
- Multi-target code generation
- Production deployment

---

## Exercises

### Exercise 1: Define a New Operation

Create a `toy.select` operation (ternary conditional):

```tablegen
def SelectOp : Toy_Op<"select"> {
  // TODO: Define arguments (condition, true_val, false_val)
  // TODO: Define result
  // TODO: Add appropriate traits
  // TODO: Add assembly format
}
```

**Should work:**
```mlir
%result = toy.select %cond, %true_val, %false_val : tensor<2x3xf64>
```

### Exercise 2: Write Optimization Patterns

Define DRR patterns for:

1. **Constant condition:** `select(const(1), a, b)` → `a`
2. **Same values:** `select(c, a, a)` → `a`
3. **Nested select:** `select(c, select(c, a, b), d)` → `select(c, a, d)`

### Exercise 3: Create an Interface

Define a `BroadcastableOp` interface:

```tablegen
def BroadcastableOpInterface : OpInterface<"BroadcastableOp"> {
  let methods = [
    // TODO: Method to check if operation supports broadcasting
    // TODO: Method to get broadcast dimensions
  ];
}
```

Apply to operations that support broadcasting.

### Exercise 4: Build a Multiclass

Create a multiclass for comparison operations (eq, ne, lt, gt, le, ge):

```tablegen
multiclass CompareOp<string mnemonic, string predicate> {
  // TODO: Define operation
  // TODO: Define lowering pattern
}

defm Eq : CompareOp<"eq", "==">;
defm Ne : CompareOp<"ne", "!=">;
// ... etc
```

---

## Further Reading

### TableGen Documentation

- **TableGen Language Reference**: [https://llvm.org/docs/TableGen/ProgRef.html](https://llvm.org/docs/TableGen/ProgRef.html)
- **MLIR ODS Documentation**: [https://mlir.llvm.org/docs/DefiningDialects/Operations/](https://mlir.llvm.org/docs/DefiningDialects/Operations/)
- **DRR Documentation**: [https://mlir.llvm.org/docs/DeclarativeRewrites/](https://mlir.llvm.org/docs/DeclarativeRewrites/)

### LLVM TableGen Uses

- **Target descriptions**: CPU instruction sets
- **Scheduling models**: Instruction latency/throughput
- **Calling conventions**: ABI specifications
- **Register allocation**: Register classes and constraints

### Domain-Specific Languages

- **Why DSLs matter**: Appropriate abstraction for domain
- **Internal vs. External DSLs**: Embedded vs. standalone
- **TableGen as a DSL**: For code generation domain

---

## Reflection Questions

Before moving to Chapter 9, consider:

1. **When to use TableGen?**
   - What problems is it good for?
   - When is manual C++ better?
   - How do you decide?

2. **Abstraction vs. Control**
   - TableGen abstracts C++ generation
   - But sometimes you need control
   - How to balance?

3. **Evolution and Maintenance**
   - How do TableGen specs evolve?
   - How to maintain backward compatibility?
   - Versioning strategies?

4. **Learning Curve**
   - TableGen has steep initial learning curve
   - But huge productivity payoff
   - How to introduce to a team?

5. **Limitations**
   - What can't TableGen express?
   - When do you hit the limits?
   - How to work around them?

These questions touch on fundamental tooling and productivity trade-offs!

---

**You've now mastered TableGen - the foundation of MLIR's extensibility!** 🎯

**Next up:** Chapter 9, where we'll design custom dialects and build domain-specific optimizations! 🏗️
