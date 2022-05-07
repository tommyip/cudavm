#pragma once

#include <vector>
#include <array>
#include <iostream>

#include "utils.h"

const size_t STACK_SIZE = 16;
const size_t ACCOUNT_SIZE = 8;
const size_t MAX_INSTRUCTIONS = 1024;
const size_t MAX_ACCOUNTS = 8;
const size_t MAX_CONSTANTS = 8;
const size_t MAX_ARGUMENTS = 8;

enum class OpCode : unsigned char {
    // ( a b -- a+b )
    Add = 0,
    // ( a b -- a-b )
    Sub = 1,
    // ( a b -- a*b )
    Mul = 2,
    // ( a b -- a/b )
    Div = 3,
    // ( a b -- a^b )
    Pow = 4,
    // ( a b -- a<b )
    // 1 if a < b, 0 otherwise
    Lt = 5,
    // ( 0 -- 1 )
    // ( a -- 0 )
    Not = 6,
    // ( a -- a a )
    Dup = 7,
    // ( a b -- b a )
    Rot = 8,
    // ( n m -- a )
    // Load m-th value from n-th account
    Load = 9,
    // ( a n m -- )
    // Store a into the m-th slot of the n-th account
    Store = 10,
    // ( -- a )
    // The operand is the immediate next pseudo opcode.
    // It should be an index into the constant pool.
    Const = 11,
    // ( n -- a )
    // Load the n-th argument
    Arg = 12,
    // ( a -- )
    // Return from the program if `a` is 0
    // All the account changes will be discarded
    Assert = 13,

    // Padding
    NoOp = 14,
};

std::ostream& operator<<(std::ostream& os, OpCode const& opcode);

// Account structures:
// Wallet accounts: balance at slot 0
class Account {
public:
    std::array<int, ACCOUNT_SIZE> state;
    void display();
};

bool operator==(const Account& lhs, const Account& rhs);

class Program {
public:
    Program() {}
    void Add();
    void Sub();
    void Mul();
    void Div();
    void Pow();
    void Lt();
    void Not();
    void Dup();
    void Rot();
    void Load();
    void Store();
    void Const(int value);
    void Arg();
    void Assert();

    void eval(
        std::vector<int>& args,
        std::vector<int>& account_indices,
        std::vector<Account>& accounts
    );

    std::vector<OpCode> instructions;
    std::vector<int> constant_pool;

private:
    void check_contraints();
};
