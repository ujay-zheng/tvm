from tvm import auto_scheduler
from tvm import te
from tvm import topi
import tvm

# ret [out, bais, data, weight]

@auto_scheduler.register_workload
def linear(bs, indim, outdim):
    data = te.placeholder((bs, indim), name='data', dtype="float32")
    weight = te.placeholder((outdim, indim), name='weight', dtype="float32")
    bias = te.placeholder((outdim, ), name="bias", dtype="float32")
    out = topi.nn.dense(data, weight, bias=bias, out_dtype="float32")
    return [out, bias, data, weight ]

@auto_scheduler.register_workload
def linear_bnorm_relu(bs, indim, outdim):
    data = te.placeholder((bs, indim), name='data', dtype="float32")
    weight = te.placeholder((outdim, indim), name='weight', dtype="float32")
    bias = te.placeholder((outdim, ), name="bias", dtype="float32")
    out = topi.nn.relu(out)
    return [out, bias, data, weight ]