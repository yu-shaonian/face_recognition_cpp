#include "torch/torch.h"
#include "torch/script.h"

int main()
{
    torch::Tensor output = torch::randn({ 3,2 });
    std::cout << output;

    return 0;
}
