import jittor as jt
from jittor import nn


class MLP(jt.Module):

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def execute(self, x):
        return self.model(x)


def main():
    x = jt.randn((2, 3, 4))
    mlp = MLP(in_dim=4, hidden_dim=5, out_dim=4)
    y = mlp(x)
    print(y.shape)


if __name__ == '__main__':
    main()

