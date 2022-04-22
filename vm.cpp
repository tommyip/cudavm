#include <cmath>
#include <stdio.h>
#include <iostream>
#include <stdexcept>
#include <algorithm>

#include "vm.h"

void Program::Add() {
    this->instructions.push_back(OpCode::Add);
    this->check_contraints();
}

void Program::Sub() {
    this->instructions.push_back(OpCode::Sub);
    this->check_contraints();
}

void Program::Mul() {
    this->instructions.push_back(OpCode::Mul);
    this->check_contraints();
}

void Program::Div() {
    this->instructions.push_back(OpCode::Div);
    this->check_contraints();
}

void Program::Pow() {
    this->instructions.push_back(OpCode::Pow);
    this->check_contraints();
}

void Program::Dup() {
    this->instructions.push_back(OpCode::Dup);
    this->check_contraints();
}

void Program::Rot() {
    this->instructions.push_back(OpCode::Rot);
    this->check_contraints();
}

void Program::Load() {
    this->instructions.push_back(OpCode::Load);
    this->check_contraints();
}

void Program::Store() {
    this->instructions.push_back(OpCode::Store);
    this->check_contraints();
}

void Program::Const(long long int value) {
    auto it = std::find(this->constant_pool.begin(), this->constant_pool.end(), value);
    int index;
    if (it != this->constant_pool.end()) {
        index = it - this->constant_pool.begin();
    } else {
        index = this->constant_pool.size();
        this->constant_pool.push_back(value);
    }
    this->instructions.push_back(OpCode::Const);
    this->instructions.push_back(static_cast<OpCode>(index));
    this->check_contraints();
}

void Program::Arg() {
    this->instructions.push_back(OpCode::Arg);
    this->check_contraints();
}

void Program::eval(
    std::vector<long long int>& args,
    std::vector<int>& account_indices,
    std::vector<Account>& accounts
) {
    if (args.size() > MAX_ARUGMENTS) {
        throw std::length_error("Too many arguments");
    }
    if (account_indices.size() > MAX_ACCOUNTS) {
        throw std::length_error("Too many account indices");
    }

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
                debug_printf("Add   ( %lld %lld -- %lld )\n", a, b, res);
                break;
            }
            case OpCode::Sub:
            {
                auto b = stack.back(); stack.pop_back();
                auto a = stack.back(); stack.pop_back();
                auto res = a - b;
                stack.push_back(res);
                debug_printf("Sub   ( %lld %lld -- %lld )\n", a, b, res);
                break;
            }
            case OpCode::Mul:
            {
                auto b = stack.back(); stack.pop_back();
                auto a = stack.back(); stack.pop_back();
                auto res = a * b;
                stack.push_back(res);
                debug_printf("Mul   ( %lld %lld -- %lld )\n", a, b, res);
                break;
            }
            case OpCode::Div:
            {
                auto b = stack.back(); stack.pop_back();
                auto a = stack.back(); stack.pop_back();
                auto res = a / b;
                stack.push_back(res);
                debug_printf("Div   ( %lld %lld -- %lld )\n", a, b, res);
                break;
            }
            case OpCode::Pow:
            {
                auto b = stack.back(); stack.pop_back();
                auto a = stack.back(); stack.pop_back();
                auto res = (long long int)std::pow((double)a, (double)b);
                debug_printf("Pow   ( %lld %lld -- %lld )\n", a, b, res);
                stack.push_back(res);
                break;
            }
            case OpCode::Dup:
            {
                auto a = stack.back();
                debug_printf("Dup   ( %lld -- %lld %lld )\n", a, a, a);
                stack.push_back(a);
                break;
            }
            case OpCode::Rot:
            {
                auto b = stack.back(); stack.pop_back();
                auto a = stack.back(); stack.pop_back();
                debug_printf("Dup   ( %lld %lld -- %lld %lld )\n", a, b, b, a);
                stack.push_back(b);
                stack.push_back(a);
                break;
            }
            case OpCode::Load:
            {
                auto m = stack.back(); stack.pop_back();
                auto n = stack.back(); stack.pop_back();
                auto res = accounts[account_indices[n]].state[m];
                stack.push_back(res);
                debug_printf("Load  ( %lld %lld -- %lld )\n", n, m, res);
                break;
            }
            case OpCode::Store:
            {
                auto m = stack.back(); stack.pop_back();
                auto n = stack.back(); stack.pop_back();
                auto a = stack.back(); stack.pop_back();
                accounts[account_indices[n]].state[m] = a;
                debug_printf("Store ( %lld %lld %lld -- )\n", a, n, m);
                break;
            }
            case OpCode::Const:
            {
                int const_idx = static_cast<int>(*(++pc));
                long long int value = this->constant_pool[const_idx];
                stack.push_back(value);
                debug_printf("Const ( -- %lld )\n", value);
                break;
            }
            case OpCode::Arg:
            {
                auto n = stack.back(); stack.pop_back();
                auto res = args[n];
                stack.push_back(res);
                debug_printf("Arg   ( %lld -- %lld )\n", n, res);
                break;
            }
        }
        ++pc;
    }
}

void Program::check_contraints() {
    if (this->instructions.size() > MAX_INSTRUCTIONS) {
        throw std::length_error("Too many instructions");
    }
    if (this->constant_pool.size() > MAX_CONSTANTS) {
        throw std::length_error("Too many constants");
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
