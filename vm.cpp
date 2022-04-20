#include <cmath>
#include <stdio.h>
#include <iostream>

#include "vm.h"

OpCode Const(int idx) {
    return static_cast<OpCode>(idx);
}

void Account::display() {
    for (int i = 0; i < ACCOUNT_SIZE; ++i) {
        if (i > 0) {
            std::cout << " ";
        }
        std::cout << this->state[i];
    }
}

void eval(
    Program& program,
    std::vector<int>& account_indices,
    std::vector<Account>& accounts
) {
    std::vector<long long int> stack;

    std::vector<OpCode>::iterator pc = program.instructions.begin();
    while (pc != program.instructions.end()) {
        switch (*pc) {
            case OpCode::Add:
            {
                auto b = stack.back(); stack.pop_back();
                auto a = stack.back(); stack.pop_back();
                auto res = a + b;
                stack.push_back(res);
                printf("Add   ( %lld %lld -- %lld )\n", a, b, res);
                break;
            }
            case OpCode::Sub:
            {
                auto b = stack.back(); stack.pop_back();
                auto a = stack.back(); stack.pop_back();
                auto res = a - b;
                stack.push_back(res);
                printf("Sub   ( %lld %lld -- %lld )\n", a, b, res);
                break;
            }
            case OpCode::Mul:
            {
                auto b = stack.back(); stack.pop_back();
                auto a = stack.back(); stack.pop_back();
                auto res = a * b;
                stack.push_back(res);
                printf("Mul   ( %lld %lld -- %lld )\n", a, b, res);
                break;
            }
            case OpCode::Div:
            {
                auto b = stack.back(); stack.pop_back();
                auto a = stack.back(); stack.pop_back();
                auto res = a / b;
                stack.push_back(res);
                printf("Div   ( %lld %lld -- %lld )\n", a, b, res);
                break;
            }
            case OpCode::Pow:
            {
                auto b = stack.back(); stack.pop_back();
                auto a = stack.back(); stack.pop_back();
                auto res = (long long int)std::pow((double)a, (double)b);
                printf("Pow   ( %lld %lld -- %lld )\n", a, b, res);
                stack.push_back(res);
                break;
            }
            case OpCode::Load:
            {
                auto m = stack.back(); stack.pop_back();
                auto n = stack.back(); stack.pop_back();
                auto res = accounts[account_indices[n]].state[m];
                stack.push_back(res);
                printf("Load  ( %lld %lld -- %lld )\n", n, m, res);
                break;
            }
            case OpCode::Store:
            {
                auto m = stack.back(); stack.pop_back();
                auto n = stack.back(); stack.pop_back();
                auto a = stack.back(); stack.pop_back();
                accounts[account_indices[n]].state[m] = a;
                printf("Store ( %lld %lld %lld -- )\n", a, n, m);
                break;
            }
            case OpCode::Const:
            {
                int const_idx = static_cast<int>(*(++pc));
                long long int value = program.constant_pool[const_idx];
                stack.push_back(value);
                printf("Const ( -- %lld )\n", value);
                break;
            }
        }
        ++pc;
    }
}
