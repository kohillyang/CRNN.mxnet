import mxnet as mx
import mxnet.gluon.nn as nn
import pandas as pd
import numpy as np
import os
import cv2
import tqdm
import logging
import mxnet.autograd as ag
import time
import gluoncv


class CRNN(nn.HybridBlock):
    def __init__(self, n_out):
        super(CRNN, self).__init__()
        self.cnn = gluoncv.model_zoo.resnet18_v1b(pretrained=True, dilated=True)
        self.lstm = mx.gluon.rnn.LSTM(hidden_size=1024, num_layers=2, layout='NTC',
                                      dropout=0.5, bidirectional=True)
        self.fc = nn.Dense(units=n_out, flatten=False)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.cnn(x)
        x = x.max(axis=2)  # n, c, t
        x = x.squeeze()
        x = x.transpose(axes=(0, 2, 1))
        x = self.lstm(x)
        x = self.fc(x)
        return x


class CRNNDataset(object):
    def __init__(self, csv_path="/data2/zyx/yks/dianke_ocr/dataset/train_set.csv",
                 image_path="/data2/zyx/yks/dianke_ocr/dataset/train_val_dataset",
                 max_sentence_length_pad=64,
                 words=None,
                 ):
        self.anno = pd.read_csv(csv_path)
        self.anno = np.array(self.anno)
        if words is None:
            self.sentences = self.anno[:, 1]
            self.one_sentence = "".join(self.sentences)
            self.words = list(set(self.one_sentence))
            self.words.sort()
        else:
            self.words = words
        print(len(self.words))
        # index of zero is reserved for unknown characters, and the last word is for blank.
        self.words_dict = {w: (i + 1) for i, w in enumerate(self.words)}
        self.image_path = image_path
        self.max_sentence_length_pad = max_sentence_length_pad

    def __getitem__(self, idx):
        path, sentence = self.anno[idx]
        path = os.path.join(self.image_path, path)
        assert os.path.exists(path)
        image = cv2.imread(path)
        if image is None:
            logging.warning("{} is None".format(path))
            return self[(idx+1) % len(self)]
        image = image[:, :, ::-1]
        def _get_w_idx(w):
            try:
                return self.words_dict[w]
            except KeyError as e:
                return 0

        def image_pad(img_ori, dshape=(96, 512)):
            fscale = min(dshape[0] / img_ori.shape[0], dshape[1] / img_ori.shape[1])
            img_resized = cv2.resize(img_ori, dsize=(0, 0), fx=fscale, fy=fscale)  # type: np.ndarray
            img_padded = np.zeros(shape=(int(dshape[0]), int(dshape[1]), 3), dtype=np.float32)
            img_padded[:img_resized.shape[0], :img_resized.shape[1], :img_resized.shape[2]] = img_resized
            return img_padded

        # convert word to index
        sentence_integer = [_get_w_idx(w) for w in sentence]
        # pad to fix length, please note that these words for padding should not have any contribution to loss.
        sentence_integer = sentence_integer[:self.max_sentence_length_pad]
        word_len = len(sentence_integer)
        sentence_integer_padded = sentence_integer + [0] * (self.max_sentence_length_pad - len(sentence_integer))
        image_padded = image_pad(image)
        return image_padded.transpose(2, 0, 1) / 255.0 - 0.5, sentence_integer_padded, word_len

    def __len__(self):
        return len(self.anno)

    def viz(self):
        import matplotlib.pyplot as plt
        for x, sen, sl in self:
            plt.imshow(((x.transpose(1, 2, 0) + .5) * 255).astype(np.uint8))
            print(sen)
            plt.show()

    def max_sentences_len(self):
        return max(map(len, self.anno[:, 1]))


class SentenceAccuMetric(mx.metric.EvalMetric):
    def update(self, labels, preds):
        # type:(mx.nd.NDArray, mx.nd.NDArray)-> None
        # labels: N x T, preds: N x T X C
        blank_word = preds.shape[2] - 1
        labels = labels.asnumpy().astype('i')
        preds = preds.argmax(axis=2).asnumpy().astype('i')
        for p, l in zip(preds, labels):
            lw = -1
            sen = []
            for w in p:
                if lw != w:
                    sen.append(w)
                    lw = w
            p = np.array(sen)
            p = p[np.where(p != blank_word)]
            p = list(p)
            l = l[np.where(l != 0)]
            p = list(p)
            l = list(l)
            if p == l:
                self.sum_metric += 1.0
        self.num_inst += preds.shape[0]


def train_crnn(net, train_dataset, val_dataset=None, gpus=[7], base_lr=1e-3, momentum=.9, wd=1e-4, log_interval=50):
    criterion = mx.gluon.loss.CTCLoss(layout='NTC', label_layout='NT')
    train_loader = mx.gluon.data.DataLoader(train_dataset, shuffle=True, batch_size=16, num_workers=16)
    if val_dataset is not None:
        val_loader = mx.gluon.data.DataLoader(val_dataset, shuffle=True, batch_size=32)
    ctx_list = [mx.gpu(x) for x in gpus]
    net.collect_params().reset_ctx(ctx_list)
    net.hybridize(static_alloc=True, static_shape=True)
    trainer = mx.gluon.Trainer(
        net.collect_params(),
        'adam',
        {'learning_rate': base_lr,
         # 'wd': wd,
         # 'momentum': momentum,
         'clip_gradient': 5})
    metric = mx.metric.Loss(name="ctx_loss")
    acc_metric = SentenceAccuMetric(name="accu")
    eval_metrics = mx.metric.CompositeEvalMetric()
    eval_metrics.add(metric)
    eval_metrics.add(acc_metric)
    btic = time.time()
    step = 0
    for n_epoch in range(100):
        if n_epoch == 4:
            trainer.set_learning_rate(base_lr * 0.1)
        for n_batch, data_batch in enumerate(train_loader):
            data, label, label_lengths = [x.as_in_context(ctx_list[0]).astype('f') for x in data_batch]
            # label_cat = [l[:l_l.asscalar()] for l,l_l in zip(label, label_lengths)]
            # label_cat = mx.nd.concat(*label_cat, dim=0)
            # label_cat = label_cat.asnumpy()
            with ag.record():
                y = net(data)
                # loss = criterion(y.reshape(1, -1, y.shape[2]), label_cat.reshape(1, -1))  # type: mx.nd.NDArray
                loss = criterion(y, label, mx.nd.array([y.shape[1]] * y.shape[0], ctx=y.context), label_lengths)
                loss = loss / data.shape[0]
                loss = loss.sum()
            ag.backward(loss)
            trainer.step(batch_size=1)
            metric.update(None, preds=loss)
            acc_metric.update(labels=label, preds=y)
            step += 1
            if n_batch % 4000 == 0:
                save_path = "output/weight-{}-{}-{:.3f}.params".format(n_epoch, n_batch, acc_metric.get()[1])
                net.collect_params().save(save_path)
                trainer.save_states(save_path + ".trainer")
            if n_batch % log_interval == 0:
                msg = ','.join(['{}={:.5f}'.format(w, v) for w, v in zip(*eval_metrics.get())])
                msg += ",lr={}".format(trainer.learning_rate)
                msg += ",Speed: {:.3f} samples/sec".format((log_interval * data.shape[0]) / (time.time() - btic), )
                logging.info("Epoch={},Step={},N_Batch={},".format(n_epoch, step, n_batch) + msg)
                btic = time.time()
                eval_metrics.reset()
                acc_metric.reset()
def validate(net, val_dataset, gpus = [6]):
    val_loader = mx.gluon.data.DataLoader(val_dataset, shuffle=True, batch_size=32)
    ctx_list = [mx.gpu(x) for x in gpus]
    net.collect_params().reset_ctx(ctx_list)
    net.hybridize(static_alloc=True, static_shape=True)
    acc_metric = SentenceAccuMetric(name="accu")
    for n_batch, data_batch in enumerate(val_loader):
        data, label, label_lengths = [x.as_in_context(ctx_list[0]).astype('f') for x in data_batch]
        y = net(data)
        acc_metric.update(labels=label, preds=y)
    print(acc_metric.get())

if __name__ == '__main__':
    da = CRNNDataset()
    # print(len(da.words))
    # print(da.max_sentences_len())
    # da.viz()
    # import tqdm

    logging.basicConfig(level=logging.INFO)
    net = CRNN(n_out=6000)
    # net.collect_params().load("output/weight-37-16000-0.657.params")
    # loading parameters from torch
    net.initialize(init=mx.init.Normal())
    train_crnn(net, CRNNDataset(max_sentence_length_pad=62), gpus=[5])
    # train_dataset = CRNNDataset(max_sentence_length_pad=62)
    # val_dataset = CRNNDataset(csv_path="/data2/zyx/yks/dianke_ocr/dataset/val_set.csv",
    #                           max_sentence_length_pad=62, words=train_dataset.words)
    # validate(net, val_dataset=val_dataset)