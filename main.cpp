#include <iostream>
#include <vector>

#include "vm.h"

// Constant Product AMM
//
// Accounts:
// 0. Pool source reserve
// 1. Pool destination reserve
// 2. Source wallet
// 3. Destination wallet
//
// Parameters:
// 0. Amount of in token to swap
//
// Equations:
//
// Constant Product:
// x * y = k  [x => pool source reserve, y => pool destination reserve]
// 
// Constant Product Swap:
// (x + u) * (y - v) = k  [u => deposit, v => withdrawal]
//
// Withdrawal amount:
// v = y * u / (x + u)
Program constant_swap() {
    Program prog;

    // Calculate v
    // Load y: pool destination reserve
    prog.Const(1); prog.Const(0); prog.Load();
    // Load u: argument0 / deposit amount
    prog.Const(0); prog.Arg();
    // y * u
    prog.Mul();
    // Load x: pool source reserve
    prog.Const(0); prog.Const(0); prog.Load();
    // Load u
    prog.Const(0); prog.Arg();
    // x + u
    prog.Add();
    // v = y * u / (x + u)
    prog.Div();
    
    // Debit pool destination reserve
    prog.Dup();
    prog.Const(1); prog.Const(0); prog.Load();
    prog.Rot();
    prog.Sub();
    prog.Const(1); prog.Const(0); prog.Store();

    // Credit destination wallet
    prog.Const(3); prog.Const(0); prog.Load();
    prog.Add();
    prog.Const(3); prog.Const(0); prog.Store();

    // Credit pool source reserve
    prog.Const(0); prog.Const(0); prog.Load();
    prog.Const(0); prog.Arg();
    prog.Add();
    prog.Const(0); prog.Const(0); prog.Store();

    // Debit source wallet
    prog.Const(2); prog.Const(0); prog.Load();
    prog.Const(0); prog.Arg();
    prog.Sub();
    prog.Const(2); prog.Const(0); prog.Store();

    return prog;
}

int main() {
    Program constant_swap_program = constant_swap();

    Account swapper_tok1;
    swapper_tok1.state[0] = 4200;
    Account swapper_tok2;
    swapper_tok2.state[0] = 133700;

    Account pool_tok1;
    pool_tok1.state[0] = 1000000;
    Account pool_tok2;
    pool_tok2.state[0] = 1000000;

    std::vector<Account> accounts{swapper_tok1, swapper_tok2, pool_tok1, pool_tok2};

    std::vector<int> account_indices{2, 3, 0, 1};
    std::vector<long long int> arguments{1000};
    constant_swap_program.eval(arguments, account_indices, accounts);

    std::vector<int> account_indices1{3, 2, 1, 0};
    std::vector<long long int> arguments1{1000};
    constant_swap_program.eval(arguments1, account_indices1, accounts);

    for (unsigned int i = 0; i < account_indices.size(); ++i) {
        std::cout << "Account " << i << ": ";
        accounts[account_indices[i]].display();
        std::cout << std::endl;
    }
}
