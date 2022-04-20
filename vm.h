#include <vector>

const int STACK_SIZE = 256;
const int ACCOUNT_SIZE = 10;
const int MAX_CONSTANTS = 32;

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
};

OpCode Const(int idx);

class Program {
public:
    Program(
        std::vector<OpCode> instructions,
        long long int *constant_pool
    ) : instructions(instructions), constant_pool(constant_pool) {}

    std::vector<OpCode> instructions;
    long long int *constant_pool;
};

class Account {
public:
    Account() {}

    long long int state[ACCOUNT_SIZE];
    void display();
};

void eval(
    Program& program,
    std::vector<int>& account_indices,
    std::vector<Account>& accounts
);
