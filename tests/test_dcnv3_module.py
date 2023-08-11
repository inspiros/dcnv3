import torch

from dcnv3 import DCNv3


def test_dcnv3_module():
    m = DCNv3(4, group=2).cuda()
    print(m)

    x = torch.randn(1, 10, 10, 4).cuda()

    output = m(x)
    print(output.shape)


if __name__ == '__main__':
    test_dcnv3_module()
