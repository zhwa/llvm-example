# Chapter 2: Building Intuition - The AST Phase

> *"Every journey begins with a single step." - Lao Tzu*

## Introduction

In Chapter 1, we explored the conceptual landscape of compilers and understood *why* MLIR exists. Now it's time to write some code!

But here's a strategic decision: **we're not starting with MLIR yet**. Instead, we're going to build a traditional compiler frontend using familiar techniques. We'll implement a lexer, a parser, and construct an Abstract Syntax Tree (AST) for the Toy language.

**Why start here?**

1. **Familiarity breeds confidence**: If you've ever used regular expressions or written a JSON parser, you'll recognize these concepts
2. **Appreciate the evolution**: By building a traditional AST first, you'll better understand what MLIR improves upon
3. **Gradual learning curve**: Master one thing at a time, not everything at once
4. **Real foundation**: The AST we build here feeds directly into MLIR in Chapter 3

Think of this chapter as your training ground. We'll work with concepts you can reason about before we introduce MLIR's more sophisticated machinery.

---

## 2.1 The Toy Language Specification

Before we can compile Toy programs, we need to understand what Toy *is*. Let's define our language.

### Core Features

**1. Variables and Type Annotations**
```toy
var a = [[1, 2, 3], [4, 5, 6]];     # Inferred shape: <2, 3>
var b<2, 3> = [1, 2, 3, 4, 5, 6];   # Explicit shape annotation
```

**2. Functions**
```toy
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}
```

**3. Built-in Operations**
- `transpose(tensor)` - Matrix transpose
- `a * b` - Element-wise multiplication
- `print(tensor)` - Display values

**4. Tensor Literals**
```toy
[[1, 2, 3], [4, 5, 6]]              # 2D tensor
[1, 2, 3, 4, 5, 6]                  # 1D tensor  
```

### What Toy Doesn't Have

To keep things simple, Toy intentionally lacks:
- ❌ Control flow (`if`, `while`, `for`)
- ❌ Multiple data types (only 64-bit floats)
- ❌ Classes or structs
- ❌ Pointers or references
- ❌ Standard library

This isn't a limitation—it's intentional minimalism. We're learning compiler techniques, not language design.

### Example Toy Program

Let's look at a complete program (from `toy/Ch1/ast.toy`):

```toy
# User defined generic function that operates on unknown shaped arguments.
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  # Define a variable `a` with shape <2, 3>
  var a = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];

  # This call will specialize `multiply_transpose` with <2, 3> for both arguments
  var c = multiply_transpose(a, b);
  
  # Another call with the same shapes reuses the specialized version
  var d = multiply_transpose(b, a);
  
  # Different shapes trigger another specialization
  var e = multiply_transpose(b, c);
  
  # This would trigger a shape inference error (we'll see this in Chapter 5)
  # var f = multiply_transpose(transpose(a), c);
}
```

**Key observations:**
- Functions are generic over shapes
- Variables store tensors (multi-dimensional arrays)
- Operations work element-wise or structurally
- Comments use `#` (like Python)

---

## 2.2 Traditional Compilation: The Three-Phase Approach

Now let's build a compiler frontend for Toy. The process has three phases:

```
Toy Source Text  →  [Lexer]  →  Tokens  →  [Parser]  →  AST  →  [Semantic Analysis]
```

We'll implement each phase step by step.

---

## 2.3 Phase 1: Lexical Analysis (The Lexer)

### What Does a Lexer Do?

A lexer (also called a scanner or tokenizer) converts source text into a stream of **tokens**. Think of it as breaking a sentence into words.

**Input (source text):**
```toy
def square(x) {
  return x * x;
}
```

**Output (token stream):**
```
[tok_def] [tok_identifier:"square"] [tok_parenthese_open] 
[tok_identifier:"x"] [tok_parenthese_close] [tok_bracket_open]
[tok_return] [tok_identifier:"x"] ['*'] [tok_identifier:"x"] [';']
[tok_bracket_close] [tok_eof]
```

Each token is a meaningful unit: a keyword, an identifier, an operator, or punctuation.

### The Token Type

Let's examine how tokens are defined (from `toy/Ch1/include/toy/Lexer.h`):

```cpp
// List of Token returned by the lexer.
enum Token : int {
  // Single-character tokens use their ASCII value
  tok_semicolon = ';',
  tok_parenthese_open = '(',
  tok_parenthese_close = ')',
  tok_bracket_open = '{',
  tok_bracket_close = '}',
  tok_sbracket_open = '[',
  tok_sbracket_close = ']',

  tok_eof = -1,

  // Keywords
  tok_return = -2,
  tok_var = -3,
  tok_def = -4,

  // Identifiers and literals
  tok_identifier = -5,
  tok_number = -6,
};
```

**Design choice**: Single-character tokens (like `'('` or `'*'`) use their ASCII value directly. This is a clever trick—you can treat them as both characters and tokens! Multi-character tokens (like `return` or identifiers) use negative values.

### Location Tracking

Good compilers provide helpful error messages. To do that, we need to track where each token came from:

```cpp
/// Structure definition a location in a file.
struct Location {
  std::shared_ptr<std::string> file; ///< filename
  int line;                          ///< line number
  int col;                           ///< column number
};
```

When we report an error like "undefined variable `x`", we can say exactly where: `@example.toy:12:5`.

### The Lexer Class

The lexer is designed as an abstract base class:

```cpp
class Lexer {
public:
  /// Create a lexer for the given filename
  Lexer(std::string filename)
      : lastLocation(
            {std::make_shared<std::string>(std::move(filename)), 0, 0}) {}
  virtual ~Lexer() = default;

  /// Look at the current token in the stream.
  Token getCurToken() { return curTok; }

  /// Move to the next token in the stream and return it.
  Token getNextToken() { return curTok = getTok(); }

  /// Return the current identifier (prereq: getCurToken() == tok_identifier)
  llvm::StringRef getId() { return identifierStr; }

  /// Return the current number (prereq: getCurToken() == tok_number)
  double getValue() { return numVal; }

  /// Return the location for the beginning of the current token.
  Location getLastLocation() { return lastLocation; }

private:
  virtual llvm::StringRef readNextLine() = 0;  // Implemented by subclass
  // ... implementation details ...
};
```

**Key methods:**
- `getCurToken()`: Peek at current token without consuming it
- `getNextToken()`: Advance to next token
- `getId()` / `getValue()`: Get associated data for identifier/number tokens
- `getLastLocation()`: Where did this token come from?

### How Tokenization Works

The heart of the lexer is the `getTok()` method. Let's walk through it:

```cpp
Token getTok() {
  // Skip any whitespace.
  while (isspace(lastChar))
    lastChar = Token(getNextChar());

  // Save the current location before reading the token characters.
  lastLocation.line = curLineNum;
  lastLocation.col = curCol;

  // Identifier: [a-zA-Z][a-zA-Z0-9_]*
  if (isalpha(lastChar)) {
    identifierStr = (char)lastChar;
    while (isalnum((lastChar = Token(getNextChar()))) || lastChar == '_')
      identifierStr += (char)lastChar;

    // Check if it's a keyword
    if (identifierStr == "return")
      return tok_return;
    if (identifierStr == "def")
      return tok_def;
    if (identifierStr == "var")
      return tok_var;
    return tok_identifier;
  }

  // Number: [0-9.]+
  if (isdigit(lastChar) || lastChar == '.') {
    std::string numStr;
    do {
      numStr += lastChar;
      lastChar = Token(getNextChar());
    } while (isdigit(lastChar) || lastChar == '.');

    numVal = strtod(numStr.c_str(), nullptr);
    return tok_number;
  }

  // Comment until end of line
  if (lastChar == '#') {
    do {
      lastChar = Token(getNextChar());
    } while (lastChar != EOF && lastChar != '\n' && lastChar != '\r');

    if (lastChar != EOF)
      return getTok();  // Skip the comment, get next token
  }

  // Check for end of file
  if (lastChar == EOF)
    return tok_eof;

  // Otherwise, just return the character as its ASCII value
  Token thisChar = Token(lastChar);
  lastChar = Token(getNextChar());
  return thisChar;
}
```

**Step-by-step:**

1. **Skip whitespace**: Spaces, tabs, newlines don't matter in Toy
2. **Record location**: Before we read anything, remember where we are
3. **Identifier or keyword?**
   - Starts with letter → read alphanumeric characters
   - Check if it's a keyword (`return`, `def`, `var`)
   - Otherwise, it's an identifier (variable or function name)
4. **Number?**
   - Starts with digit or `.` → read numeric characters
   - Convert to `double` with `strtod()`
5. **Comment?**
   - Starts with `#` → skip to end of line
   - Recursively call `getTok()` to get next real token
6. **Single character**: Return as token (like `'('`, `'*'`, `';'`)

### The LexerBuffer Implementation

The abstract `Lexer` class needs a concrete implementation. `LexerBuffer` reads from memory:

```cpp
class LexerBuffer final : public Lexer {
public:
  LexerBuffer(const char *begin, const char *end, std::string filename)
      : Lexer(std::move(filename)), current(begin), end(end) {}

private:
  /// Provide one line at a time to the Lexer
  llvm::StringRef readNextLine() override {
    auto *begin = current;
    while (current <= end && *current && *current != '\n')
      ++current;
    if (current <= end && *current)
      ++current;
    llvm::StringRef result{begin, static_cast<size_t>(current - begin)};
    return result;
  }
  
  const char *current, *end;
};
```

This design separates concerns:
- Base `Lexer`: Knows how to tokenize
- `LexerBuffer`: Knows how to read from memory
- Could also implement `LexerFile`, `LexerStream`, etc.

### Try It Yourself

**Exercise**: What tokens would the lexer produce for this line?

```toy
var x = [1, 2, 3];
```

<details>
<summary>Click to see answer</summary>

```
[tok_var] [tok_identifier:"x"] ['='] [tok_sbracket_open] 
[tok_number:1] [','] [tok_number:2] [','] [tok_number:3] 
[tok_sbracket_close] [';'] [tok_eof]
```

</details>

---

## 2.4 Phase 2: Syntax Analysis (The Parser)

### What Does a Parser Do?

The parser takes the flat stream of tokens and builds a **tree structure** that represents the program's syntax. This tree is called an Abstract Syntax Tree (AST).

**Why a tree?** Because programming languages have nested structure:
- Functions contain statements
- Statements contain expressions
- Expressions contain sub-expressions

A tree naturally represents this hierarchy.

### Parsing Strategy: Recursive Descent

The Toy parser uses **recursive descent** - a straightforward parsing technique where each grammar rule becomes a function.

**Grammar rule (informal):**
```
expression = primary (operator primary)*
primary = identifier | number | '(' expression ')' | '[' tensorLiteral ']'
```

**Parser functions:**
```cpp
std::unique_ptr<ExprAST> parseExpression();
std::unique_ptr<ExprAST> parsePrimary();
std::unique_ptr<ExprAST> parseTensorLiteralExpr();
```

The parser calls these functions recursively to handle nested structures.

### The Parser Class Structure

```cpp
class Parser {
public:
  /// Create a Parser for the supplied lexer.
  Parser(Lexer &lexer) : lexer(lexer) {}

  /// Parse a full Module. A module is a list of function definitions.
  std::unique_ptr<ModuleAST> parseModule() {
    lexer.getNextToken(); // prime the lexer

    // Parse functions one at a time
    std::vector<FunctionAST> functions;
    while (auto f = parseDefinition()) {
      functions.push_back(std::move(*f));
      if (lexer.getCurToken() == tok_eof)
        break;
    }
    
    // Verify we reached EOF
    if (lexer.getCurToken() != tok_eof)
      return parseError<ModuleAST>("nothing", "at end of module");

    return std::make_unique<ModuleAST>(std::move(functions));
  }

private:
  Lexer &lexer;
  // ... parsing methods ...
};
```

The parser holds a reference to the lexer and implements various `parse*()` methods.

### Parsing Building Blocks

Let's examine key parsing methods:

#### 1. Parsing Numbers

```cpp
/// numberexpr ::= number
std::unique_ptr<ExprAST> parseNumberExpr() {
  auto loc = lexer.getLastLocation();  // Record where this came from
  auto result = std::make_unique<NumberExprAST>(std::move(loc), lexer.getValue());
  lexer.consume(tok_number);  // Advance past the number
  return std::move(result);
}
```

**Simple rule**: Current token is a number → create `NumberExprAST` node → advance lexer.

#### 2. Parsing Identifiers (Variables or Function Calls)

```cpp
/// identifierexpr
///   ::= identifier           (variable reference)
///   ::= identifier '(' args ')' (function call)
std::unique_ptr<ExprAST> parseIdentifierExpr() {
  std::string name(lexer.getId());
  auto loc = lexer.getLastLocation();
  lexer.getNextToken(); // eat identifier

  // Is this a function call?
  if (lexer.getCurToken() != '(')
    return std::make_unique<VariableExprAST>(std::move(loc), name);

  // Yes, it's a function call - parse arguments
  lexer.consume(Token('('));
  std::vector<std::unique_ptr<ExprAST>> args;
  if (lexer.getCurToken() != ')') {
    while (true) {
      if (auto arg = parseExpression())
        args.push_back(std::move(arg));
      else
        return nullptr;

      if (lexer.getCurToken() == ')')
        break;

      if (lexer.getCurToken() != ',')
        return parseError<ExprAST>(", or )", "in argument list");
      lexer.getNextToken();
    }
  }
  lexer.consume(Token(')'));

  // Special handling for built-in print
  if (name == "print") {
    if (args.size() != 1)
      return parseError<ExprAST>("<single arg>", "as argument to print()");
    return std::make_unique<PrintExprAST>(std::move(loc), std::move(args[0]));
  }

  // Regular user-defined function call
  return std::make_unique<CallExprAST>(std::move(loc), name, std::move(args));
}
```

**Logic flow:**
1. Read identifier name
2. Peek ahead: is next token `'('`?
   - **No**: It's a variable reference
   - **Yes**: It's a function call - parse argument list
3. Special case: `print` is a built-in operation

#### 3. Parsing Tensor Literals

Tensor literals can be nested (for multi-dimensional arrays):

```toy
[1, 2, 3]              # 1D: shape <3>
[[1, 2], [3, 4]]       # 2D: shape <2, 2>
```

```cpp
/// tensorLiteral ::= [ literalList ]
/// literalList ::= tensorLiteral | tensorLiteral, literalList
std::unique_ptr<ExprAST> parseTensorLiteralExpr() {
  auto loc = lexer.getLastLocation();
  lexer.consume(Token('['));

  // Hold the list of values at this nesting level
  std::vector<std::unique_ptr<ExprAST>> values;
  std::vector<int64_t> dims;
  
  do {
    // Can be another nested array or a number
    if (lexer.getCurToken() == '[') {
      values.push_back(parseTensorLiteralExpr());  // Recursive!
      if (!values.back())
        return nullptr;
    } else {
      if (lexer.getCurToken() != tok_number)
        return parseError<ExprAST>("<num> or [", "in literal expression");
      values.push_back(parseNumberExpr());
    }

    if (lexer.getCurToken() == ']')
      break;

    if (lexer.getCurToken() != ',')
      return parseError<ExprAST>("] or ,", "in literal expression");

    lexer.getNextToken(); // eat ','
  } while (true);
  
  lexer.getNextToken(); // eat ']'

  // Fill in the dimensions
  dims.push_back(values.size());

  // If there are nested arrays, verify uniform dimensions
  if (llvm::any_of(values, [](std::unique_ptr<ExprAST> &expr) {
        return llvm::isa<LiteralExprAST>(expr.get());
      })) {
    auto *firstLiteral = llvm::dyn_cast<LiteralExprAST>(values.front().get());
    if (!firstLiteral)
      return parseError<ExprAST>("uniform well-nested dimensions",
                                 "inside literal expression");

    auto firstDims = firstLiteral->getDims();
    dims.insert(dims.end(), firstDims.begin(), firstDims.end());

    // Verify all elements have the same dimensions
    for (auto &expr : values) {
      auto *exprLiteral = llvm::cast<LiteralExprAST>(expr.get());
      if (exprLiteral->getDims() != firstDims)
        return parseError<ExprAST>("uniform well-nested dimensions",
                                   "inside literal expression");
    }
  }

  return std::make_unique<LiteralExprAST>(std::move(loc), std::move(values),
                                          std::move(dims));
}
```

**Key insight**: This is recursive. A tensor literal can contain other tensor literals. The parser handles this naturally through recursion.

### Handling Operator Precedence

One tricky aspect of parsing is **operator precedence**. In `a + b * c`, we want multiplication to happen first, even though `+` appears earlier.

The Toy parser uses **precedence climbing**:

```cpp
/// Get the precedence of the current token, or -1 if not a binary operator.
int getTokPrecedence() {
  if (!isascii(lexer.getCurToken()))
    return -1;

  // Make sure it's a declared binop.
  int tokPrec = binopPrecedence[lexer.getCurToken()];
  if (tokPrec <= 0)
    return -1;
  return tokPrec;
}

/// binopPrecedence is defined elsewhere as:
/// std::map<char, int> binopPrecedence = { {'*', 40} };
```

Then, when parsing binary expressions:

```cpp
/// binoprhs ::= (operator primary)*
std::unique_ptr<ExprAST> parseBinOpRHS(int exprPrec,
                                       std::unique_ptr<ExprAST> lhs) {
  while (true) {
    int tokPrec = getTokPrecedence();

    // If this binop binds less tightly than the current precedence, we're done
    if (tokPrec < exprPrec)
      return lhs;

    int binOp = lexer.getCurToken();
    lexer.consume(Token(binOp));
    auto loc = lexer.getLastLocation();

    // Parse the right-hand side
    auto rhs = parsePrimary();
    if (!rhs)
      return parseError<ExprAST>("expression", "to complete binary operator");

    // If the next operator binds tighter, let it take rhs first
    int nextPrec = getTokPrecedence();
    if (tokPrec < nextPrec) {
      rhs = parseBinOpRHS(tokPrec + 1, std::move(rhs));
      if (!rhs)
        return nullptr;
    }

    // Merge lhs and rhs
    lhs = std::make_unique<BinaryExprAST>(std::move(loc), binOp,
                                          std::move(lhs), std::move(rhs));
  }
}
```

**How it works:**

For `a + b * c`:
1. Parse `a` as LHS
2. See `+` (precedence 20), parse `b` as RHS
3. Peek ahead: see `*` (precedence 40)
4. Since 40 > 20, `*` "steals" `b` as its LHS
5. Recursively parse `b * c` first
6. Then combine `a + (b * c)`

This ensures correct evaluation order without building an explicit precedence table.

---

## 2.5 Phase 3: The Abstract Syntax Tree

### What Is an AST?

An Abstract Syntax Tree represents the syntactic structure of your program. Each node in the tree represents a construct in your source code.

**Example:**

```toy
def add(x, y) {
  return x + y;
}
```

**AST representation:**
```
FunctionAST
├── PrototypeAST: "add"
│   ├── Parameter: "x"
│   └── Parameter: "y"
└── Body (ExprASTList)
    └── ReturnExprAST
        └── BinaryExprAST: '+'
            ├── VariableExprAST: "x"
            └── VariableExprAST: "y"
```

### AST Node Hierarchy

All AST nodes inherit from a base class (from `toy/Ch1/include/toy/AST.h`):

```cpp
/// Base class for all expression nodes.
class ExprAST {
public:
  enum ExprASTKind {
    Expr_VarDecl,
    Expr_Return,
    Expr_Num,
    Expr_Literal,
    Expr_Var,
    Expr_BinOp,
    Expr_Call,
    Expr_Print,
  };

  ExprAST(ExprASTKind kind, Location location)
      : kind(kind), location(std::move(location)) {}
  virtual ~ExprAST() = default;

  ExprASTKind getKind() const { return kind; }
  const Location &loc() { return location; }

private:
  const ExprASTKind kind;
  Location location;
};
```

**Design pattern**: This uses **LLVM-style RTTI** (Run-Time Type Information). Instead of C++'s `dynamic_cast`, we store an explicit `kind` tag. This is faster and more predictable.

### Key AST Node Types

#### 1. NumberExprAST - Numeric Literals

```cpp
class NumberExprAST : public ExprAST {
  double val;

public:
  NumberExprAST(Location loc, double val)
      : ExprAST(Expr_Num, std::move(loc)), val(val) {}

  double getValue() { return val; }

  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Num; }
};
```

Represents literal numbers like `42` or `3.14`.

#### 2. VariableExprAST - Variable References

```cpp
class VariableExprAST : public ExprAST {
  std::string name;

public:
  VariableExprAST(Location loc, llvm::StringRef name)
      : ExprAST(Expr_Var, std::move(loc)), name(name) {}

  llvm::StringRef getName() { return name; }

  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Var; }
};
```

Represents references to variables like `x` or `myTensor`.

#### 3. BinaryExprAST - Binary Operations

```cpp
class BinaryExprAST : public ExprAST {
  char op;
  std::unique_ptr<ExprAST> lhs, rhs;

public:
  BinaryExprAST(Location loc, char op, std::unique_ptr<ExprAST> lhs,
                std::unique_ptr<ExprAST> rhs)
      : ExprAST(Expr_BinOp, std::move(loc)), op(op), 
        lhs(std::move(lhs)), rhs(std::move(rhs)) {}

  char getOp() { return op; }
  ExprAST *getLHS() { return lhs.get(); }
  ExprAST *getRHS() { return rhs.get(); }

  static bool classof(const ExprAST *c) { return c->getKind() == Expr_BinOp; }
};
```

Represents operations like `a * b` or `x + y`. Note how it holds pointers to sub-expressions (the operands).

#### 4. CallExprAST - Function Calls

```cpp
class CallExprAST : public ExprAST {
  std::string callee;
  std::vector<std::unique_ptr<ExprAST>> args;

public:
  CallExprAST(Location loc, const std::string &callee,
              std::vector<std::unique_ptr<ExprAST>> args)
      : ExprAST(Expr_Call, std::move(loc)), callee(callee),
        args(std::move(args)) {}

  llvm::StringRef getCallee() { return callee; }
  llvm::ArrayRef<std::unique_ptr<ExprAST>> getArgs() { return args; }

  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Call; }
};
```

Represents function calls like `transpose(a)` or `multiply_transpose(a, b)`.

#### 5. VarDeclExprAST - Variable Declarations

```cpp
class VarDeclExprAST : public ExprAST {
  std::string name;
  VarType type;
  std::unique_ptr<ExprAST> initVal;

public:
  VarDeclExprAST(Location loc, llvm::StringRef name, VarType type,
                 std::unique_ptr<ExprAST> initVal)
      : ExprAST(Expr_VarDecl, std::move(loc)), name(name),
        type(std::move(type)), initVal(std::move(initVal)) {}

  llvm::StringRef getName() { return name; }
  ExprAST *getInitVal() { return initVal.get(); }
  const VarType &getType() { return type; }

  static bool classof(const ExprAST *c) { return c->getKind() == Expr_VarDecl; }
};
```

Represents declarations like `var a = [[1, 2], [3, 4]];`.

### Function and Module Nodes

Functions have two parts: a prototype (signature) and a body:

```cpp
/// Function prototype (name and parameters)
class PrototypeAST {
  Location location;
  std::string name;
  std::vector<std::unique_ptr<VariableExprAST>> args;

public:
  PrototypeAST(Location location, const std::string &name,
               std::vector<std::unique_ptr<VariableExprAST>> args)
      : location(std::move(location)), name(name), args(std::move(args)) {}

  llvm::StringRef getName() const { return name; }
  llvm::ArrayRef<std::unique_ptr<VariableExprAST>> getArgs() { return args; }
};

/// Complete function definition
class FunctionAST {
  std::unique_ptr<PrototypeAST> proto;
  std::unique_ptr<ExprASTList> body;

public:
  FunctionAST(std::unique_ptr<PrototypeAST> proto,
              std::unique_ptr<ExprASTList> body)
      : proto(std::move(proto)), body(std::move(body)) {}
      
  PrototypeAST *getProto() { return proto.get(); }
  ExprASTList *getBody() { return body.get(); }
};

/// A module is a list of functions
class ModuleAST {
  std::vector<FunctionAST> functions;

public:
  ModuleAST(std::vector<FunctionAST> functions)
      : functions(std::move(functions)) {}

  auto begin() { return functions.begin(); }
  auto end() { return functions.end(); }
};
```

**Structure:**
- `ModuleAST` contains multiple functions
- Each `FunctionAST` has a prototype and body
- The body is a list of expressions

---

## 2.6 Putting It All Together

Now let's see the complete pipeline in action.

### The Main Compiler Entry Point

From `toy/Ch1/toyc.cpp`:

```cpp
int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");

  // Read the input file
  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST)
    return 1;

  // Dump the AST
  switch (emitAction) {
  case Action::DumpAST:
    dump(*moduleAST);
    return 0;
  default:
    llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  }

  return 0;
}

std::unique_ptr<toy::ModuleAST> parseInputFile(llvm::StringRef filename) {
  // Load the file into memory
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }
  
  auto buffer = fileOrErr.get()->getBuffer();
  
  // Create lexer
  LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
  
  // Create parser and parse
  Parser parser(lexer);
  return parser.parseModule();
}
```

**The flow:**
1. Load source file into memory
2. Create a `LexerBuffer` with the file contents
3. Create a `Parser` with the lexer
4. Call `parseModule()` to build the AST
5. Dump the AST for inspection

### Visualizing the AST

The `dump()` function (from `toy/Ch1/parser/AST.cpp`) prints a readable representation of the AST:

```cpp
void ASTDumper::dump(ModuleAST *node) {
  for (auto &f : *node)
    dump(&f);
}

void ASTDumper::dump(FunctionAST *node) {
  indent();
  llvm::errs() << "Function \n";
  dump(node->getProto());
  dump(node->getBody());
}

void ASTDumper::dump(BinaryExprAST *node) {
  INDENT();
  llvm::errs() << "BinOp: " << node->getOp() << " " << loc(node) << "\n";
  dump(node->getLHS());
  dump(node->getRHS());
}
```

The dumper traverses the tree recursively, printing each node with indentation to show nesting.

---

## 2.7 Building and Running Chapter 1

Let's compile and test the Toy compiler!

### Building with CMake

The project uses CMake. From the repo root:

```powershell
# Configure (first time only)
cmake --preset default

# Build Chapter 1
cmake --build build --target toyc-ch1
```

This produces an executable `toyc-ch1` (or `toyc-ch1.exe` on Windows).

### Running the Compiler

```powershell
# Navigate to the Chapter 1 directory
cd toy\Ch1

# Compile and dump the AST
..\..\build\toy\Ch1\toyc-ch1.exe -emit=ast ast.toy
```

### Understanding the Output

Let's trace through a simple example:

**Input (`simple.toy`):**
```toy
def add(x, y) {
  return x + y;
}
```

**Output (AST dump):**
```
Module:
  Function 
    Proto 'add' @simple.toy:1:1
      Param: x @simple.toy:1:9
      Param: y @simple.toy:1:12
    Block {
      Return @simple.toy:2:3
        BinOp: + @simple.toy:2:10
          var: x @simple.toy:2:10
          var: y @simple.toy:2:14
    } // Block
```

**Reading the output:**
- Function named "add" defined at line 1, column 1
- Two parameters: `x` and `y`
- Body contains a return statement
- Return expression is a binary operation `+`
- Left operand is variable `x`, right operand is variable `y`

### Experimenting

Try modifying `ast.toy` and re-running the compiler:

**1. Add a new function:**
```toy
def square(x) {
  return x * x;
}
```

**2. Try invalid syntax:**
```toy
def broken( {
  return 42;
}
```

What error do you get? Where does it point you?

**3. Add nested tensor literals:**
```toy
def main() {
  var matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
}
```

How does the AST represent the nested structure?

---

## 2.8 Why AST Isn't Enough

We now have a working parser that builds an AST. But there are significant limitations:

### Problem 1: Hard to Analyze

Want to check if a variable is defined before use? You'd have to:
- Traverse the entire tree
- Build symbol tables manually
- Track scopes yourself
- Handle complex control flow

**AST doesn't help** - it just reflects syntax, not semantics.

### Problem 2: Hard to Transform

Want to optimize `x * 1` to `x`? You'd need to:
- Pattern match across tree structures
- Preserve source locations
- Handle nested cases
- Ensure correctness

**AST structure is awkward** for transformations because it's tied to surface syntax.

### Problem 3: Hard to Lower

Want to generate machine code? You need to:
- Decide on calling conventions
- Manage registers or memory
- Handle types and sizes
- Deal with platform differences

**AST is too high-level** - there's a huge gap between `x + y` and assembly instructions.

### Problem 4: No Reusability

Every language that parses to an AST needs its own:
- Optimization passes
- Analysis tools
- Code generators
- Debuggers

**AST is language-specific** - you can't share infrastructure.

### Enter MLIR

This is where MLIR comes in. In the next chapter, we'll transform our AST into MLIR, which:

✅ **Has explicit semantics**: Types, operations, and side effects are clear
✅ **Is designed for transformation**: Rewrite patterns, passes, optimizations
✅ **Supports lowering**: Progressive refinement through dialects
✅ **Is reusable**: Share infrastructure across languages and domains

The AST was necessary to understand the structure of our source code. But to do anything useful (optimize, analyze, compile), we need something better.

---

## Summary

Let's recap what we've built and learned:

### Key Concepts

1. **Lexer (Tokenizer)**
   - Converts source text into tokens
   - Tracks source locations for error reporting
   - Handles keywords, identifiers, numbers, and punctuation

2. **Parser**
   - Uses recursive descent parsing
   - Builds an Abstract Syntax Tree from tokens
   - Handles operator precedence correctly
   - Reports syntax errors with locations

3. **Abstract Syntax Tree (AST)**
   - Tree structure representing program syntax
   - Each node is a language construct
   - Uses LLVM-style RTTI for type checking
   - Preserves source locations

4. **Complete Pipeline**
   - Source file → Lexer → Tokens → Parser → AST
   - Can visualize the AST structure
   - Foundation for next steps

### Why This Matters

- **Familiar foundation**: Traditional compiler techniques everyone can understand
- **Appreciate evolution**: See what MLIR improves upon
- **Working code**: We built a real, functioning parser
- **Necessary step**: The AST feeds into MLIR generation (Chapter 3)

### Limitations We Discovered

- ASTs are hard to analyze
- ASTs are awkward for transformations
- ASTs don't help with lowering
- AST-based tools aren't reusable

These limitations motivate MLIR's design!

---

## What's Next

In **Chapter 3**, we'll take our AST and convert it to MLIR. We'll learn:

- What a **dialect** is and how to define one
- How to use **TableGen** to define operations
- How **MLIRGen** translates AST to MLIR
- Why MLIR's representation is superior for optimization and transformation

We'll see the same Toy programs represented in a completely different way—one that unlocks powerful capabilities.

---

## Exercises

### Exercise 1: Add a New Built-in Function

Extend the parser to recognize `reshape` as a built-in function (like `print`). It should accept two arguments: a tensor and a shape.

**Hint**: Modify `parseIdentifierExpr()` to check for `"reshape"`.

<details>
<summary>Solution sketch</summary>

```cpp
if (name == "reshape") {
  if (args.size() != 2)
    return parseError<ExprAST>("<tensor> and <shape>", "as arguments to reshape()");
  return std::make_unique<ReshapeExprAST>(std::move(loc), 
                                          std::move(args[0]), 
                                          std::move(args[1]));
}
```

You'd also need to define `ReshapeExprAST` in `AST.h`.
</details>

### Exercise 2: Trace the Parser

Given this input:
```toy
var x = 1 + 2 * 3;
```

Trace through the parser's execution:
1. What tokens does the lexer produce?
2. What order are the parsing methods called?
3. How does precedence affect the resulting AST?

### Exercise 3: Add Floating-Point Error Checking

The current lexer accepts malformed numbers like `1.2.3`. Modify `getTok()` to only allow one decimal point per number.

### Exercise 4: Extend the AST Dumper

Modify the AST dumper to show the shape annotations for variable declarations. Currently it shows:
```
VarDecl x @file:1:5
```

Make it show:
```
VarDecl x<2,3> @file:1:5
```

**Hint**: Modify `dump(VarDeclExprAST*)` in `AST.cpp`.

---

## Further Reading

### Parsing Techniques
- **"Crafting Interpreters"** by Robert Nystrom (free online)
  - Excellent explanation of recursive descent parsing
  - [https://craftinginterpreters.com/](https://craftinginterpreters.com/)

- **"Engineering a Compiler"** by Cooper & Torczon
  - Chapter 3: Parsing
  - More formal treatment of grammars and parsing

### LLVM Coding Standards
- **LLVM Programmer's Manual**: [https://llvm.org/docs/ProgrammersManual.html](https://llvm.org/docs/ProgrammersManual.html)
- **LLVM Coding Standards**: [https://llvm.org/docs/CodingStandards.html](https://llvm.org/docs/CodingStandards.html)
- Learn about `StringRef`, `ArrayRef`, smart pointers, and RTTI

### Abstract Syntax Trees
- **"Types and Programming Languages"** by Benjamin Pierce
  - Chapter 3: Untyped Arithmetic Expressions
  - Formal treatment of AST structure

---

## Reflection Questions

Before moving to Chapter 3, consider:

1. **What syntax features make parsing easier or harder?**
   - How does Toy's simple syntax help?
   - What would make parsing more complex?

2. **Where does semantic information belong?**
   - The AST captures syntax, but what about types?
   - When should we check if variables are defined?

3. **How would you extend Toy?**
   - What language features would you add?
   - How would they affect the lexer, parser, and AST?

Keep these questions in mind as we introduce MLIR. You'll see how MLIR's design answers many of these questions elegantly.

---

**Next up:** Chapter 3, where we define our first MLIR dialect and learn the mysterious art of TableGen! 🎯
