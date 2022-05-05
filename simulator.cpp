#include <random>
#include <vector>
#include <algorithm>

#include "contracts.h"
#include "simulator.h"

struct Pool {
    int tok1_reserve_idx;
    int tok2_reserve_idx;
};

struct Swapper {
    int tok1_wallet_idx;
    int tok2_wallet_idx;
    size_t pool_idx;
};

void generate_transactions(
    CudaVM& vm,
    size_t n_pools,
    size_t n_swappers,
    size_t n_swap_txns,
    size_t n_payment_txns
) {
    std::random_device rd;
    std::mt19937 mt(rd());

    int constant_swap_id = vm.register_program(constant_swap());
    int payment_id = vm.register_program(payment());

    std::normal_distribution<double> pool_dist(10000000.0, 10000.0);
    std::vector<Pool> pools;
    for (size_t i = 0; i < n_pools; ++i) {
        Account pool_tok1;
        pool_tok1.state[0] = (int)pool_dist(mt);
        Account pool_tok2;
        pool_tok2.state[0] = (int)pool_dist(mt);
        int tok1_reserve_idx = vm.register_account(pool_tok1);
        int tok2_reserve_idx = vm.register_account(pool_tok2);
        pools.push_back(Pool { tok1_reserve_idx, tok2_reserve_idx });
    }

    std::normal_distribution<double> swapper_dist(100000.0, 10000.0);
    std::uniform_int_distribution<size_t> pool_idx_dist(0, n_pools - 1);
    std::vector<Swapper> swappers;
    for (size_t i = 0; i < n_swappers; ++i) {
        Account swapper_tok1;
        swapper_tok1.state[0] = (int)swapper_dist(mt);
        Account swapper_tok2;
        swapper_tok2.state[0] = (int)swapper_dist(mt);
        int tok1_wallet_idx = vm.register_account(swapper_tok1);
        int tok2_wallet_idx = vm.register_account(swapper_tok2);
        size_t pool_idx = pool_idx_dist(mt);
        swappers.push_back(Swapper { tok1_wallet_idx, tok2_wallet_idx, pool_idx });
    }

    std::uniform_int_distribution<size_t> swapper_idx_dist(0, n_swappers - 1);
    std::normal_distribution<double> swap_amount_dist(100000.0, 10000.0);
    std::uniform_int_distribution<char> direction_dist(0, 1);
    for (size_t i = 0; i < n_swap_txns; ++i) {
        Swapper& swapper = swappers[swapper_idx_dist(mt)];
        Pool& pool = pools[swapper.pool_idx];
        std::vector<int> args{(int)swap_amount_dist(mt)};
        std::vector<int> account_indices;
        account_indices.reserve(4);
        if (direction_dist(mt) == 0) {
            // Swap A to B
            account_indices.push_back(pool.tok1_reserve_idx);
            account_indices.push_back(pool.tok2_reserve_idx);
            account_indices.push_back(swapper.tok1_wallet_idx);
            account_indices.push_back(swapper.tok2_wallet_idx);
        } else {
            // Swap B to A
            account_indices.push_back(pool.tok2_reserve_idx);
            account_indices.push_back(pool.tok1_reserve_idx);
            account_indices.push_back(swapper.tok2_wallet_idx);
            account_indices.push_back(swapper.tok1_wallet_idx);
        }
        vm.schedule_invocation(constant_swap_id, args, account_indices);
    }

    std::normal_distribution<double> payment_amount_dist(1000.0, 100.0);
    for (size_t i = 0; i < n_payment_txns; ++i) {
        int payer_idx = swapper_idx_dist(mt);
        Swapper& payer = swappers[payer_idx];
        int payer_wallet_idx = i % 2 == 0 ? payer.tok1_wallet_idx : payer.tok2_wallet_idx;

        int payee_idx = swapper_idx_dist(mt);
        if (payee_idx == payer_idx) {
            payee_idx = (payee_idx + 1) % n_swappers;
        }
        Swapper& payee = swappers[payee_idx];
        int payee_wallet_idx = i % 2 == 0 ? payee.tok1_wallet_idx : payee.tok2_wallet_idx;

        std::vector<int> args{(int)payment_amount_dist(mt)};
        std::vector<int> account_indices{payer_wallet_idx, payee_wallet_idx};
        vm.schedule_invocation(payment_id, args, account_indices);
    }

    std::shuffle(vm.invocations.begin(), vm.invocations.end(), mt);
}
