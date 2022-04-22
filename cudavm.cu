#include "cudavm.h"

void CudaVM::schedule_invocation(
    std::vector<long long int>& params,
    std::vector<int>& account_indices
) {
    this->params.reserve(this->params.size() + MAX_ARUGMENTS);
    this->params.insert(this->params.end(), params.begin(), params.end());
    for (int i = 0; i < MAX_ARUGMENTS - params.size(); ++i) {
        this->params.push_back(0);
    }
    this->account_indices.reserve(this->account_indices.size() + MAX_ACCOUNTS);
    this->account_indices.insert(this->account_indices.end(), account_indices.begin(), account_indices.end());
    for (int i = 0; i < MAX_ACCOUNTS - account_indices.size(); ++i) {
        this->account_indices.push_back(0);
    }

    ++this->n_calls;
}
