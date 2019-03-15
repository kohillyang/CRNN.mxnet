import os
# os.environ["MXNET_ENGINE_TYPE"]="NaiveEngine"
import mxnet as mx
import time
import threading
import numpy as np
import cv2
import os


cv2.setNumThreads(1)  # Sometimes we need this to avoid deadlock, especially in multi-processing environments.


class TestOP(mx.operator.CustomOp):
    def __init__(self, *args, **kwargs):
        super(TestOP, self).__init__(*args, **kwargs)
        print("init")

    def forward(self, is_train, req, in_data, out_data, aux):
        try:
            x = in_data[0].asnumpy()
            print("ss")
            x = np.ones(shape=(1024, 1024, 300))
            x_resized = cv2.resize(x, (0, 0), fx=0.5, fy=0.5)
            x_resized_sum = x_resized.sum()
            print('ee', x_resized_sum)
        except Exception as e:
            print(e)

@mx.operator.register("test_op")
class TestOPProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(TestOPProp, self).__init__()

    def list_arguments(self):
        return ['x']

    def list_outputs(self):
        return ['y']

    def infer_shape(self, in_shape):
        return in_shape, in_shape

    def create_operator(self, ctx, shapes, dtypes):
        return TestOP()


ctx_list = [mx.gpu(x) for x in [0, 1, 2, 3]]
x_list = [mx.nd.ones(shape=(1, 2), ctx=c) for c in ctx_list]

data = mx.sym.var(name="data")
y = mx.sym.Custom(data, op_type="test_op")
y = mx.sym.identity(y, name="identity")
sym_block = mx.gluon.SymbolBlock(outputs=y, inputs=data)
sym_block.collect_params().reset_ctx(ctx_list)


def forward(x, ctx):
    # print("enter", x)
    re = sym_block(x)
    re.wait_to_read()
    # print("exit")
    return re


# for x, c in zip(x_list, ctx_list):
#     forward(x, c)
# mx.nd.waitall()
threads = []
for x, c in zip(x_list, ctx_list):
    t = threading.Thread(target=forward, args=(x, c))
    t.daemon = True
    t.start()
#
for t in threads:
    t.join()
mx.nd.waitall()