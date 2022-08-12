# SQLGen

This project was taken from https://github.com/mshockwave/SQLGen, which is part of the tutorial _"How to write a TableGen backend"_ in 2021 LLVM Developers' Meeting.
Only a few necessary changes are made in order to build it.
For example, the orginal code disabled Runtime Type Information (RTTI) by ```/Gy-```. This restriction is removed in this project.

## Example usage
Given the following TableGen file `SampleQuery.td`:
```tblgen
class Query <string table, dag query_fields = (all), dag condition = (none)> {
  string TableName = table;
  dag Fields = query_fields;
  dag WhereClause = condition;
  list<string> OrderedBy = [];
}

def : Query<"Orders", (fields "Person", "Amount")>;
```
We can use the following command to generate the corresponding SQL query:
```bash
$ .build/sqlgen SampleQuery.td -o SampleQuery.sql
$ cat SampleQuery.sql
SELECT Person, Amount FROM Orders;
$
```

## Testing
SQLGen is using [LLVM LIT](https://pypi.org/project/lit) as the testing harness. Please install it first:
```bash
pip3 install lit
```
SQLGen is also using the functionality of [FileCheck](https://llvm.org/docs/CommandGuide/FileCheck.html).
However, we support two variants of FileCheck:
 1. Using the `FileCheck` command line tool from a LLVM build and put in your PATH. Note that unfortunately, LLVM doesn't ship `FileCheck` in their release tarball.
 2. The recommended way: install the `filecheck` [python package / command line tool](https://github.com/mull-project/FileCheck.py) via `pip3 install filecheck`.

After the `lit` is installed and FileCheck is setup, launch this command to run the tests:
```bash
cd .build
ninja check
```