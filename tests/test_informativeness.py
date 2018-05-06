"""Behavior tests for models.informativeness."""

import nonce2vec.models.informativeness as info
import torch.nn.functional as F
import torch
import numpy
from torch.autograd import Variable
from torch import FloatTensor


def test_entropy():
    x = Variable(FloatTensor([[0, 10, 0], [4, 4, 4]]))
    soft = F.softmax(x, dim=1)
    log = F.log_softmax(x, dim=1)
    print(soft)
    print(log)
    plogp = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
    print(plogp)
    print(-1.0 * plogp.sum(dim=1))
    print(x)
    print(info.entropy(x))
    print('test.;')
    print(numpy.log(3))
    print(-info.entropy(x)[0] / numpy.log(3) + 1)
    print(-info.entropy(x)[1] / numpy.log(3) + 1)
    assert 1 == 2
