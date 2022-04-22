#include <cmath>
#include <stdio.h>
#include <iostream>
#include <stdexcept>
#include <algorithm>

#include "vm.h"

void Program::Add() {
    this->instructions.push_back(OpCode::Add);
}

void Program::Sub() {
    this->instructions.push_back(OpCode::Sub);
}

void Program::Mul() {
    this->instructions.push_back(OpCode::Mul);
}

void Program::Div() {
    this->instructions.push_back(OpCode::Div);
}

void Program::Pow() {
    this->instructions.push_back(OpCode::Pow);
}

void Program::Load() {
    this->instructions.push_back(OpCode::Load);
}

void Program::Store() {
    this->instructions.push_back(OpCode::Store);
}

void Program::Const(long long int value) {
    auto it = std::find(this->constant_pool.begin(), this->constant_pool.end(), value);
    int index;
    if (it != this->constant_pool.end()) {
        index = it - this->constant_pool.begin();
    } else {
        index = this->constant_pool.size();
        if (index >= MAX_CONSTANTS) {
            throw std::length_error("Program has too many constants");
        }
        this->constant_pool.push_back(value);
    }
    this->instructions.push_back(OpCode::Const);
    this->instructions.push_back(static_cast<OpCode>(index));
}

void Program::eval(
    std::vector<int>& account_indices,
    std::vector<Account>& accounts
) {
    std::vector<long long int> stack;

    std::vector<OpCode>::iterator pc = this->instructions.begin();
    while (pc != this->instructions.end()) {
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
                long long int value = this->constant_pool[const_idx];
                stack.push_back(value);
                printf("Const ( -- %lld )\n", value);
                break;
            }
        }
        ++pc;
    }
}

void Account::display() {
    for (int i = 0; i < ACCOUNT_SIZE; ++i) {
        if (i > 0) {
            std::cout << " ";
        }
        std::cout << this->state[i];
    }
}
