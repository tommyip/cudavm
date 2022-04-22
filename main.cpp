#include <iostream>
#include <vector>

#include "vm.h"

Program constant_swap() {
    Program prog;

    prog.Const(0);
    prog.Const(0);
    prog.Load();
    prog.Const(0);
    prog.Const(1);
    prog.Load();
    prog.Mul();

    prog.Const(1);
    prog.Const(0);
    prog.Store();

    return prog;
}

int main() {
    Program constant_swap_program = constant_swap();

    Account account1;
    account1.state[0] = 42;
    account1.state[1] = 1337;
    Account account2;
    std::vector<Account> accounts{account1, account2};

    std::vector<int> account_indices{0, 1};

    constant_swap_program.eval(account_indices, accounts);
    for (unsigned int i = 0; i < account_indices.size(); ++i) {
        std::cout << "Account " << i << ": ";
        accounts[account_indices[i]].display();
        std::cout << std::endl;
    }
}
