#pragma once

#include <vector>

const int STACK_SIZE = 256;
const int ACCOUNT_SIZE = 10;
const int MAX_CONSTANTS = 32;
const int MAX_ARUGMENTS = 8;

enum class OpCode {
    // ( a b -- a+b )
    Add,
    // ( a b -- a-b )
    Sub,
    // ( a b -- a*b )
    Mul,
    // ( a b -- a/b )
    Div,
    // ( a b -- a^b )
    Pow,
    // ( a -- a a )
    Dup,
    // ( a b -- b a )
    Rot,
    // ( n m -- a )
    // Load m-th value from n-th account
    Load,
    // ( a n m -- )
    // Store a into the m-th slot of the n-th account
    Store,
    // ( -- a )
    // The operand is the immediate next pseudo opcode.
    // It should be an index into the constant pool.
    Const,
    // ( n -- a )
    // Load the n-th argument
    Arg,
};

// Account structures:
// Wallet accounts: balance at slot 0
class Account {
public:
    long long int state[ACCOUNT_SIZE]{0};
    void display();
};

class Program {
public:
    Program() {}
    void Add();
    void Sub();
    void Mul();
    void Div();
    void Pow();
    void Dup();
    void Rot();
    void Load();
    void Store();
    void Const(long long int value);
    void Arg();

    void eval(
        std::vector<long long int> params,
        std::vector<int>& account_indices,
        std::vector<Account>& accounts
    );

    std::vector<OpCode> instructions;
    std::vector<long long int> constant_pool;
};
