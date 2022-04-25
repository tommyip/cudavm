#pragma once

#include <vector>

#include "vm.h"

struct ScheduledInvocation {
    int program_id;
    std::vector<long long int> *args;
    std::vector<int> *account_indices;
};

class CudaVM {
public:
    int register_program(Program program);
    int register_account(Account account);
    void schedule_invocation(
        int program_id,
        std::vector<long long int> *args,
        std::vector<int> *account_indices
    );
    void execute_serial();
    void execute_parallel();

    std::vector<Account> accounts;

private:
    std::vector<Program> programs;
    std::vector<ScheduledInvocation> invocations;
};
