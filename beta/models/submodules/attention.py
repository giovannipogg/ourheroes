import torch


class Attention(torch.nn.Module):
    linear: torch.nn.Linear
    u_att: torch.nn.Parameter
    softmax: torch.nn.Softmax

    def __init__(self, input_size: int):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=input_size, out_features=input_size)
        self.u_att = torch.nn.Parameter(torch.zeros(input_size))
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attention as described in the reference paper.

        :param x: an encoded sentence or section [shape=(N, input_size)].
        :return: the result of attention [shape=(input_size)]
        """
        u = self.linear(x)  # (N, input_size) -> (N, input_size)
        u = torch.tanh(u)
        u = torch.matmul(u, self.u_att)  # (N, input_size)  @ (input_size) -> (N,)
        alpha = self.softmax(u)
        output = alpha.unsqueeze(-1) * x  # (N, 1) * (N, input_size) -> (N, input_size)
        output = torch.sum(output, dim=0)  # (N, input_size) -> (input_size,)
        return output

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
