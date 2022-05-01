#pragma once

#include <vector>
#include <array>
#include <iostream>

#include "utils.h"

const size_t STACK_SIZE = 64;
const size_t ACCOUNT_SIZE = 10;
const size_t MAX_INSTRUCTIONS = 256;
const size_t MAX_ACCOUNTS = 10;
const size_t MAX_CONSTANTS = 32;
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
    // ( a -- a a )
    Dup = 5,
    // ( a b -- b a )
    Rot = 6,
    // ( n m -- a )
    // Load m-th value from n-th account
    Load = 7,
    // ( a n m -- )
    // Store a into the m-th slot of the n-th account
    Store = 8,
    // ( -- a )
    // The operand is the immediate next pseudo opcode.
    // It should be an index into the constant pool.
    Const = 9,
    // ( n -- a )
    // Load the n-th argument
    Arg = 10,

    // Padding
    NoOp = 11,
};

std::ostream& operator<<(std::ostream& os, OpCode const& opcode);

// Account structures:
// Wallet accounts: balance at slot 0
class Account {
public:
    std::array<long long int, ACCOUNT_SIZE> state;
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
    void Dup();
    void Rot();
    void Load();
    void Store();
    void Const(long long int value);
    void Arg();

    void eval(
        std::vector<long long int>& args,
        std::vector<unsigned int>& account_indices,
        std::vector<Account>& accounts
    );

    std::vector<OpCode> instructions;
    std::vector<long long int> constant_pool;

private:
    void check_contraints();
};
