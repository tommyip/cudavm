#include <algorithm>

#include "cudavm.h"

// Credit: https://stackoverflow.com/a/1505791
__host__ __device__
long long int int_pow(long long int base, long long int exp) {
    if (exp == 0) return 1;
    if (exp == 1) return base;

    long long int tmp = pow(base, exp / 2);
    if (exp % 2 == 0) return tmp * tmp;
    else return base * tmp * tmp;
}

__global__
void cuda_eval(
    int *invocations,
    int n_invocations,
    OpCode *global_instructions,
    long long int *global_constant_pool,
    unsigned int *global_program_id,
    long long int *global_args,
    unsigned int *global_account_indices,
    long long int *global_account
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= n_invocations) return;
    int invocation_id = invocations[tid];

    unsigned int program_id = global_program_id[invocation_id];
    OpCode *instructions = &global_instructions[program_id * MAX_INSTRUCTIONS];
    long long int *constant_pool = &global_constant_pool[program_id * MAX_CONSTANTS];
    long long int *args = &global_args[invocation_id * MAX_ARGUMENTS];
    unsigned int *account_indices = &global_account_indices[invocation_id * MAX_ACCOUNTS];

    long long int stack[STACK_SIZE];
    size_t sp = 0;
    size_t pc = 0;
    bool running = true;
    while (pc < MAX_INSTRUCTIONS && running) {
        switch (instructions[pc]) {
            case OpCode::Add:
            {
                auto b = stack[--sp];
                auto a = stack[--sp];
                auto res = a + b;
                stack[sp++] = res;
                // debug_printf("Add   ( %lld %lld -- %lld )\n", a, b, res);
                break;
            }
            case OpCode::Sub:
            {
                auto b = stack[--sp];
                auto a = stack[--sp];
                auto res = a - b;
                stack[sp++] = res;
                // debug_printf("Sub   ( %lld %lld -- %lld )\n", a, b, res);
                break;
            }
            case OpCode::Mul:
            {
                auto b = stack[--sp];
                auto a = stack[--sp];
                auto res = a * b;
                stack[sp++] = res;
                // debug_printf("Mul   ( %lld %lld -- %lld )\n", a, b, res);
                break;
            }
            case OpCode::Div:
            {
                auto b = stack[--sp];
                auto a = stack[--sp];
                auto res = a / b;
                stack[sp++] = res;
                // debug_printf("Div   ( %lld %lld -- %lld )\n", a, b, res);
                break;
            }
            case OpCode::Pow:
            {
                auto b = stack[--sp];
                auto a = stack[--sp];
                auto res = int_pow(a, b);
                stack[sp++] = res;
                // debug_printf("Pow   ( %lld %lld -- %lld )\n", a, b, res);
                break;
            }
            case OpCode::Dup:
            {
                auto a = stack[sp - 1];
                stack[sp++] = a;
                // debug_printf("Dup   ( %lld -- %lld %lld )\n", a, a, a);
                break;
            }
            case OpCode::Rot:
            {
                auto tmp = stack[sp - 1];
                stack[sp - 1] = stack[sp - 2];
                stack[sp - 2] = tmp;
                // debug_printf("Rot   ( %lld %lld -- %lld %lld )\n", stack[sp - 1], stack[sp - 2], stack[sp - 2], stack[sp - 1]);
                break;
            }
            case OpCode::Load:
            {
                auto m = stack[--sp];
                auto n = stack[--sp];
                auto account_idx = account_indices[n];
                auto *account = &global_account[account_idx * ACCOUNT_SIZE];
                auto res = account[m];
                stack[sp++] = res;
                // debug_printf("Load  ( %lld %lld -- %lld )\n", n, m, res);
                break;
            }
            case OpCode::Store:
            {
                auto m = stack[--sp];
                auto n = stack[--sp];
                auto a = stack[--sp];
                auto account_idx = account_indices[n];
                auto *account = &global_account[account_idx * ACCOUNT_SIZE];
                account[m] = a;
                // debug_printf("Store ( %lld %lld %lld -- )\n", a, n, m);
                break;
            }
            case OpCode::Const:
            {
                size_t const_idx = static_cast<size_t>(instructions[++pc]);
                auto res = constant_pool[const_idx];
                stack[sp++] = res;
                // debug_printf("Const ( -- %lld )\n", res);
                break;
            }
            case OpCode::Arg:
            {
                size_t n = static_cast<size_t>(stack[sp - 1]);
                auto res = args[n];
                stack[sp - 1] = res;
                // debug_printf("Arg   ( %lld -- %lld )\n", n, res);
                break;
            }
            case OpCode::NoOp:
            {
                running = false;
                break;
            }
        }
        ++pc;
    }
}

unsigned int CudaVM::register_program(Program program) {
    unsigned int program_id = this->programs.size();
    this->programs.push_back(program);
    return program_id;
}

unsigned int CudaVM::register_account(Account account) {
    unsigned int account_index = this->accounts.size();
    this->accounts.push_back(account);
    return account_index;
}

void CudaVM::schedule_invocation(
    unsigned int program_id,
    std::vector<long long int>& args,
    std::vector<unsigned int>& account_indices
) {
    this->invocations.push_back(ScheduledInvocation { program_id, args, account_indices });
}

void CudaVM::execute_serial() {
    for (auto& invocation : this->invocations) {
        Program& program = this->programs[invocation.program_id];
        program.eval(invocation.args, invocation.account_indices, this->accounts);
    }
}

void CudaVM::execute_serial(std::vector<int>& invocation_indices) {
    for (auto& invocation_idx : invocation_indices) {
        auto& invocation = this->invocations[invocation_idx];
        auto& program = this->programs[invocation.program_id];
        program.eval(invocation.args, invocation.account_indices, this->accounts);
    }
}

void CudaVM::execute_parallel() {
    size_t n_programs = this->programs.size();
    size_t n_invocations = this->invocations.size();
    debug_printf("execute_parallel n_programs=%d n_invocations=%d\n", n_programs, n_invocations);

    auto opt_start = clock_start();
    Chunks *chunks = this->optimize_invocation_order();
    auto opt_secs = clock_end(opt_start);
    debug_printf("chunks=%d max_chunk=%d cpu_chunk_size=%d\n",
        chunks->gpu_indices.size() - 1,
        chunks->gpu_indices[0].size(),
        chunks->cpu_indices.size());
    debug_printf("txns optimization duration: %fs\n", opt_secs);
    int max_invocation_size = sizeof(int) * chunks->gpu_indices[0].size();
    int *d_invocations;
    gpuErrchk(cudaMalloc(&d_invocations, max_invocation_size));

    std::vector<OpCode> h_global_instructions;
    h_global_instructions.reserve(n_programs * MAX_INSTRUCTIONS);

    std::vector<long long int> h_global_constant_pool;
    h_global_constant_pool.reserve(n_programs * MAX_CONSTANTS);

    for (auto& program : this->programs) {
        push_with_padding(program.instructions, h_global_instructions, MAX_INSTRUCTIONS, OpCode::NoOp);
        push_with_padding(program.constant_pool, h_global_constant_pool, MAX_CONSTANTS, 0ll);
    }

    std::vector<unsigned int> h_global_program_id;
    h_global_program_id.reserve(n_invocations);

    std::vector<long long int> h_global_args;
    h_global_args.reserve(n_invocations * MAX_ARGUMENTS);

    std::vector<unsigned int> h_global_account_indices;
    h_global_account_indices.reserve(n_invocations * MAX_ACCOUNTS);

    for (auto& invocation : this->invocations) {
        h_global_program_id.push_back(invocation.program_id);

        push_with_padding(invocation.args, h_global_args, MAX_ARGUMENTS, 0ll);
        push_with_padding(invocation.account_indices, h_global_account_indices, MAX_ACCOUNTS, 0u);
    }

    OpCode *d_global_instructions;
    long long int *d_global_constant_pool;
    unsigned int *d_global_program_id;
    long long int *d_global_args;
    unsigned int *d_global_account_indices;
    long long int *d_global_account;

    size_t global_instructions_size = sizeof(OpCode) * h_global_instructions.size();
    size_t global_constant_pool_size = sizeof(long long int) * h_global_constant_pool.size();
    size_t global_program_id_size = sizeof(unsigned int) * h_global_program_id.size();
    size_t global_args_size = sizeof(long long int) * h_global_args.size();
    size_t global_account_indices_size = sizeof(unsigned int) * h_global_account_indices.size();
    size_t global_account_size = sizeof(long long int) * this->accounts.size() * ACCOUNT_SIZE;

    // #if DEBUG
    // std::cout << "h_global_instructions: ";
    // print_vector(h_global_instructions);
    // std::cout << "h_global_constant_pool: ";
    // print_vector(h_global_constant_pool);
    // std::cout << "h_global_program_id: ";
    // print_vector(h_global_program_id);
    // std::cout << "h_global_args: ";
    // print_vector(h_global_args);
    // std::cout << "h_global_account_indices: ";
    // print_vector(h_global_account_indices);
    // std::cout << "accounts: ";
    // std::vector<long long int> global_accounts(this->accounts.size()*ACCOUNT_SIZE);
    // memcpy(global_accounts.data(), this->accounts.data(), global_account_size);
    // print_vector(global_accounts);
    // #endif

    gpuErrchk(cudaMalloc(&d_global_instructions, global_instructions_size));
    gpuErrchk(cudaMalloc(&d_global_constant_pool, global_constant_pool_size));
    gpuErrchk(cudaMalloc(&d_global_program_id, global_program_id_size));
    gpuErrchk(cudaMalloc(&d_global_args, global_args_size));
    gpuErrchk(cudaMalloc(&d_global_account_indices, global_account_indices_size));
    gpuErrchk(cudaMalloc(&d_global_account, global_account_size));

    gpuErrchk(cudaMemcpy(d_global_instructions, h_global_instructions.data(), global_instructions_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_global_constant_pool, h_global_constant_pool.data(), global_constant_pool_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_global_program_id, h_global_program_id.data(), global_program_id_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_global_args, h_global_args.data(), global_args_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_global_account_indices, h_global_account_indices.data(), global_account_indices_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_global_account, this->accounts.data(), global_account_size, cudaMemcpyHostToDevice));

    dim3 block_dim(BLOCK_SIZE, 1, 1);

    for (int i = 0; i < chunks->gpu_indices.size(); ++i) {
        auto& invocations = chunks->gpu_indices[i];
        gpuErrchk(cudaMemcpy(d_invocations, invocations.data(), sizeof(int) * invocations.size(), cudaMemcpyHostToDevice));

        dim3 grid_dim(ceil_div(invocations.size(), BLOCK_SIZE), 1, 1);
        cuda_eval<<<grid_dim, block_dim>>>(
            d_invocations,
            invocations.size(),
            d_global_instructions,
            d_global_constant_pool,
            d_global_program_id,
            d_global_args,
            d_global_account_indices,
            d_global_account
        );
        gpuErrchk(cudaPeekAtLastError());
    }

    cudaMemcpy(this->accounts.data(), d_global_account, global_account_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    if (chunks->cpu_indices.size() > 0) {
        this->execute_serial(chunks->cpu_indices);
    }
}

Chunks::Chunks(int max_chunks, int max_chunk_size)
    : max_chunks(max_chunks), max_chunk_size(max_chunk_size) {}

void Chunks::add_invocation(int i, ScheduledInvocation& invocation) {
    int j;
    for (j = 0; j < this->gpu_indices.size(); ++j) {
        bool is_disjoint = true;

        for (int k = 0; k < invocation.account_indices.size(); ++k) {
            if (this->allocated_accounts[j * this->max_chunk_size + invocation.account_indices[k]]) {
                is_disjoint = false;
                break;
            }
        }
        if (is_disjoint) {
            for (int k = 0; k < invocation.account_indices.size(); ++k) {
                this->allocated_accounts[j * this->max_chunk_size + invocation.account_indices[k]] = true;
            }
            this->gpu_indices[j].push_back(i);
            return;
        }
    }
    if (j < this->max_chunks) {
        this->allocated_accounts.resize(this->allocated_accounts.size() + this->max_chunk_size, false);
        for (int k = 0; k < invocation.account_indices.size(); ++k) {
            this->allocated_accounts[j * this->max_chunk_size + invocation.account_indices[k]] = true;
        }
        this->gpu_indices.push_back(std::vector<int>{i});
    } else {
        this->cpu_indices.push_back(i);
    }
}

// 1. Group transactions s.t. all accounts in each chunk are disjoint
//    This prevents race condition
// 2. Group transactions within each chunk by program into thread blocks
// 3. Chunks smaller then BLOCK_SIZE are merged together to be executed on CPU
Chunks* CudaVM::optimize_invocation_order() {
    Chunks *chunks = new Chunks(256, this->accounts.size());
    for (int i = 0; i < this->invocations.size(); ++i) {
        chunks->add_invocation(i, this->invocations[i]);
    }
    // for (int i = 0; i < chunks->gpu_indices.size(); ++i) {
    //     printf("Chunk %d has %d invocations\n", i, chunks->gpu_indices[i].size());
    // }
    // printf("CPU chunk has %d invocations\n", chunks->cpu_indices.size());
    return chunks;
}
