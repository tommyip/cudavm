#include <iostream>
#include <vector>

#include "vm.h"

int main() {
    std::vector<OpCode> instructions{
        OpCode::Const, Const(0),
        OpCode::Const, Const(0),
        OpCode::Load,
        OpCode::Const, Const(0),
        OpCode::Const, Const(1),
        OpCode::Load,
        OpCode::Mul,

        OpCode::Const, Const(1),
        OpCode::Const, Const(0),
        OpCode::Store
    };
    long long int constant_pool[] = { 0, 1, 2 };
    Program program(instructions, constant_pool);

    Account account1;
    account1.state[0] = 42;
    account1.state[1] = 1337;
    Account account2;
    std::vector<Account> accounts{account1, account2};

    std::vector<int> account_indices{0, 1};

    eval(program, account_indices, accounts);
    for (unsigned int i = 0; i < account_indices.size(); ++i) {
        std::cout << "Account " << i << ": ";
        accounts[account_indices[i]].display();
        std::cout << std::endl;
    }
}
