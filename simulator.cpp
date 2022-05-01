#include <random>
#include <vector>

#include "contracts.h"
#include "simulator.h"

struct Pool {
    unsigned int tok1_reserve_idx;
    unsigned int tok2_reserve_idx;
};

struct Swapper {
    unsigned int tok1_wallet_idx;
    unsigned int tok2_wallet_idx;
    size_t pool_idx;
};

void generate_transactions(
    CudaVM& vm,
    size_t n_pools,
    size_t n_swappers,
    size_t n_transactions
) {
    std::random_device rd;
    std::mt19937 mt(rd());

    unsigned int constant_swap_id = vm.register_program(constant_swap());

    std::normal_distribution<double> pool_dist(10000000000.0, 100000.0);
    std::vector<Pool> pools;
    for (size_t i = 0; i < n_pools; ++i) {
        Account pool_tok1;
        pool_tok1.state[0] = (unsigned int)pool_dist(mt);
        Account pool_tok2;
        pool_tok2.state[0] = (unsigned int)pool_dist(mt);
        unsigned int tok1_reserve_idx = vm.register_account(pool_tok1);
        unsigned int tok2_reserve_idx = vm.register_account(pool_tok2);
        pools.push_back(Pool { tok1_reserve_idx, tok2_reserve_idx });
    }

    std::normal_distribution<double> swapper_dist(1000000.0, 100000.0);
    std::uniform_int_distribution<size_t> pool_idx_dist(0, n_pools - 1);
    std::vector<Swapper> swappers;
    for (size_t i = 0; i < n_swappers; ++i) {
        Account swapper_tok1;
        swapper_tok1.state[0] = (unsigned int)swapper_dist(mt);
        Account swapper_tok2;
        swapper_tok2.state[0] = (unsigned int)swapper_dist(mt);
        unsigned int tok1_wallet_idx = vm.register_account(swapper_tok1);
        unsigned int tok2_wallet_idx = vm.register_account(swapper_tok2);
        size_t pool_idx = pool_idx_dist(mt);
        swappers.push_back(Swapper { tok1_wallet_idx, tok2_wallet_idx, pool_idx });
    }

    std::uniform_int_distribution<size_t> swapper_idx_dist(0, n_swappers - 1);
    std::normal_distribution<double> swap_amount_dist(100000.0, 10000.0);
    std::uniform_int_distribution<char> direction_dist(0, 1);
    for (size_t i = 0; i < n_transactions; ++i) {
        Swapper& swapper = swappers[swapper_idx_dist(mt)];
        Pool& pool = pools[swapper.pool_idx];
        unsigned int swap_amount = swap_amount_dist(mt);
        std::vector<long long int> *args = new std::vector<long long int>{swap_amount};
        std::vector<unsigned int> *account_indices = new std::vector<unsigned int>();
        if (direction_dist(mt) == 0) {
            // Swap A to B
            account_indices->push_back(pool.tok1_reserve_idx);
            account_indices->push_back(pool.tok2_reserve_idx);
            account_indices->push_back(swapper.tok1_wallet_idx);
            account_indices->push_back(swapper.tok2_wallet_idx);
        } else {
            // Swap B to A
            account_indices->push_back(pool.tok2_reserve_idx);
            account_indices->push_back(pool.tok1_reserve_idx);
            account_indices->push_back(swapper.tok2_wallet_idx);
            account_indices->push_back(swapper.tok1_wallet_idx);
        }
        vm.schedule_invocation(constant_swap_id, args, account_indices);
    }
}
