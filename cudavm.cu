#include "cudavm.h"

int CudaVM::register_program(Program program) {
    int program_id = this->programs.size();
    this->programs.push_back(program);
    return program_id;
}

int CudaVM::register_account(Account account) {
    int account_index = this->accounts.size();
    this->accounts.push_back(account);
    return account_index;
}

void CudaVM::schedule_invocation(
    int program_id,
    std::vector<long long int> *args,
    std::vector<int> *account_indices
) {
    this->invocations.push_back(ScheduledInvocation { program_id, args, account_indices });
}

void CudaVM::execute_serial() {
    for (auto& invocation : this->invocations) {
        Program& program = this->programs[invocation.program_id];
        program.eval(*(invocation.args), *(invocation.account_indices), this->accounts);
    }
}
