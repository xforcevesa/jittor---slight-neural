import jittor as jt
jt.flags.use_cuda = 1


class EncoderBlock(jt.nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.wk = jt.nn.Linear(512, 512)
        self.wq = jt.nn.Linear(512, 512)
        self.wv = jt.nn.Linear(512, 512)
        self.wo = jt.nn.Linear(512, 512)

        self.attn_scale = 512 ** (-0.5)

        self.mlp = jt.nn.Sequential(
            jt.nn.Linear(512, 512),
            jt.nn.GELU(),
            jt.nn.Linear(512, 512),
        )

    def execute(self, x):
        k, q, v = self.wk(x), self.wq(x), self.wv(x)
        attn = q @ k.transpose(1, 2) * self.attn_scale
        attn = jt.nn.softmax(attn, dim=-1)
        attn = attn @ v
        x = self.wo(attn) + x
        x = x + self.mlp(x)
        return x


class Classifier(jt.nn.Module):

    def __init__(self, num_classes: int):
        super().__init__()
        self.embed = jt.nn.Sequential(
            jt.nn.Linear(1, 512),
            jt.nn.ELU(),
            jt.nn.Linear(512, 512),
        )

        self.encoders = jt.nn.ModuleList([EncoderBlock() for _ in range(3)])

        self.de_embed = jt.nn.Sequential(
            jt.nn.Linear(512, 512),
            jt.nn.ELU(),
            jt.nn.Linear(512, 1),
            jt.nn.Flatten()
        )

        self.mlp = jt.nn.Sequential(
            jt.nn.Linear(512, 512),
            jt.nn.ELU(),
            jt.nn.Linear(512, num_classes),
        )

    def execute(self, x):
        x = x.unsqueeze(-1)
        x = self.embed(x)
        for encoder in self.encoders:
            x = encoder(x)
        x = self.de_embed(x)
        x = self.mlp(x)
        return x
    
def test():
    mm = Classifier(374)
    nn = jt.randn(123, 512)
    oo = mm(nn)
    num_parameters = sum(p.numel() for p in mm.parameters())
    print(f'{oo.shape = }')
    print(f'{num_parameters = }')


if __name__ == '__main__':
    test()

