# coding=utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
#from sru import SRU, SRUCell

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut, nlayer, dropout):
        # nIn=512, nHidden=256, nOut = 2441, nlayder=2, dropout=0.5
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, num_layers = nlayer, dropout = dropout, bidirectional = True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        # input: 64x256x512
        recurrent, _ = self.rnn(input)
        # recurrent: 64x256x512
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn, dropout, leakyRelu=False, RRelu=False):
        super(CRNN, self).__init__()
        assert imgH == 48, 'imgH must be 48'

        ks = [3, 3, 3, 3, 3, 3, 3]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 128, 256, 256, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.1, inplace=True))
            elif RRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.RReLU(inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))
        # start from 64x48x256
        convRelu(0, True)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x24x128...

        convRelu(1, True)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x12x64...
        convRelu(2, True)
        convRelu(3, True)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x6x64...
        convRelu(4, True)
        convRelu(5, True)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x3x64...
        # cnn.add_module('dilation0', nn.Conv2d(512, 512, (3, 3), dilation=2, padding=(1, 0), stride=(1, 1)))
        convRelu(6, True)  # 512x1x64...
        # cnn.add_module('last_max', nn.MaxPool2d((2,1), stride=1))
        #
        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nclass, n_rnn, dropout))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)

        # print('conv.size(): {}'.format(conv.size()))
        # output = conv

        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return output


class op_cl(nn.Module):
    def __init__(self):
        super(op_cl, self).__init__()
        cnn = nn.Sequential()
        cnn.add_module('conv1', nn.Conv2d(1, 6, (3, 3), padding=1, dilation=2))
        cnn.add_module('relu1',
                       nn.LeakyReLU(0.1, inplace=True))
        # cnn.add_module('max', nn.MaxPool2d(2, 2))
        self.cnn = cnn
        # self.conv1 = nn.Conv2d(1, 6, (3, 3), dilation=1, padding=1)

    def forward(self, x):
        out = self.cnn(x)

        return out



if __name__ == '__main__':
    # input_img = torch.Tensor(5, 1, 48, 256)
    # input_img = Variable(input_img)
    #
    # net  = CRNN(48, 1, 2441, 256, 2, 0.5, 0.0, leakyRelu=True)
    # output = net(input_img)
    # print(output.size())


    rnn = nn.LSTM(10,20,2,bidirectional=True,num_layers=2, dropout=0.5)
    input = torch.randn(5,3,10)
    output = rnn(input)
    print(output[0].shape)