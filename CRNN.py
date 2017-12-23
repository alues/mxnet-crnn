#! /usr/bin/python
# -*- coding:utf-8 -*-

# Init Mxnet Module
import mxnet as mx

from collections import namedtuple

LSTM_State = namedtuple("LSTMState", ["c", "h"])
LSTM_Param = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])
LSTM_Model = namedtuple("LSTMModel", ["rnn_exec", "symbol",
                                     "init_states", "last_states", "forward_state", "backward_state",
                                     "seq_data", "seq_labels", "seq_outputs",
                                     "param_blocks"])


"""LSTM Cell symbol"""
def LSTM_Cell(num_hidden, t_indata, last_state, param, seq_idx, layer_idx):
    i2h = mx.sym.FullyConnected(data=t_indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 4,
                                name="LSTM_t%d_l%d_i2h" % (seq_idx, layer_idx))
    h2h = mx.sym.FullyConnected(data=last_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden * 4,
                                name="LSTM_t%d_l%d_h2h" % (seq_idx, layer_idx))
    gates = i2h + h2h
    slice_gates = mx.sym.split(gates, num_outputs=4,
                                      name="t%d_l%d_slice" % (seq_idx, layer_idx))
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * last_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    return LSTM_State(c=next_c, h=next_h)


def crnn(num_lstm_layer, seq_len, num_hidden, label_length, label_size, dropout=0.):
    last_states = []
    forward_param = []
    backward_param = []
    for i in range(num_lstm_layer * 2):
        last_states.append(LSTM_State(c=mx.sym.Variable("l%d_init_c" % i), h=mx.sym.Variable("l%d_init_h" % i)))
        if i % 2 == 0:
            forward_param.append(LSTM_Param(i2h_weight=mx.sym.Variable("l%d_i2h_weight" % i),
                                           i2h_bias=mx.sym.Variable("l%d_i2h_bias" % i),
                                           h2h_weight=mx.sym.Variable("l%d_h2h_weight" % i),
                                           h2h_bias=mx.sym.Variable("l%d_h2h_bias" % i)))
        else:
            backward_param.append(LSTM_Param(i2h_weight=mx.sym.Variable("l%d_i2h_weight" % i),
                                            i2h_bias=mx.sym.Variable("l%d_i2h_bias" % i),
                                            h2h_weight=mx.sym.Variable("l%d_h2h_weight" % i),
                                            h2h_bias=mx.sym.Variable("l%d_h2h_bias" % i)))

    # Input
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')

    kernel_size = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (2, 2)]
    padding_size = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (0, 0)]
    stride_size = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]
    layer_size = [64, 128, 256, 256, 512, 512, 512]
    def Conv_Relu(conv_idx, indata, bn=False):
        layer = mx.sym.Convolution(
            name="Conv_Relu_l%d" % (conv_idx),
            data=indata,
            kernel=kernel_size[conv_idx],
            pad=padding_size[conv_idx],
            stride=stride_size[conv_idx],
            num_filter=layer_size[conv_idx],
        )
        if bn:
            layer = mx.sym.BatchNorm(data=layer, name='Batch_Norm_l%d' % (conv_idx))
            
        layer = mx.sym.LeakyReLU(data=layer,name='Relu_Leaky_l%d' % (conv_idx))
        return layer

    # Test Input: 1x32x128
    net = Conv_Relu(conv_idx=0, indata=data) 
    net = mx.sym.Pooling(data=net, name='Pool_0', pool_type='max', kernel=(2, 2), stride=(2, 2)) # Output: 64x16x64
    net = Conv_Relu(conv_idx=1, indata=net)
    net = mx.sym.Pooling(data=net, name='Pool_1', pool_type='max', kernel=(2, 2), stride=(2, 2)) # Output: 128x8x32
    net = Conv_Relu(conv_idx=2, indata=net, bn=True)
    net = Conv_Relu(conv_idx=3, indata=net)
    net = mx.sym.Pooling(data=net, name='Pool_2', pool_type='max', kernel=(2, 2), stride=(2, 1), pad=(0, 1)) # Output: 256x4x33
    net = Conv_Relu(conv_idx=4, indata=net, bn=True)
    net = Conv_Relu(conv_idx=5, indata=net)
    net = mx.sym.Pooling(data=net, name='Pool_3', pool_type='max', kernel=(2, 2), stride=(2, 1), pad=(0, 1)) # Output: 512x2x34
    net = Conv_Relu(conv_idx=6, indata=net, bn=True) # Output: 512x1x33

    slices_net = mx.sym.split(data=net, axis=3, num_outputs=seq_len, squeeze_axis=True)
    
    # arg_shape, output_shape, aux_shape = net.infer_shape( **{"data": (1, 1, 32, 128)})
    # print(output_shape)

    # this block only use for parameter display
    # ############################
    # init_c = [('l%d_init_c' % l, (batch_size, num_hidden)) for l in range(num_lstm_layer * 2)]
    # init_h = [('l%d_init_h' % l, (batch_size, num_hidden)) for l in range(num_lstm_layer * 2)]
    # init_states = init_c + init_h
    # init_values = {x[0]: x[1] for x in init_states}
    # ############################
    forward_hidden = []
    for seqidx in range(seq_len):
        hidden = mx.sym.flatten(data=slices_net[seqidx])
        for i in range(num_lstm_layer):
            next_state = LSTM_Cell(
                num_hidden=num_hidden, 
                t_indata=hidden,
                last_state=last_states[2 * i],
                param=forward_param[i],
                seq_idx=seqidx, 
                layer_idx=0,
            )
            hidden = next_state.h
            last_states[2 * i] = next_state
        forward_hidden.append(hidden)

    backward_hidden = []
    for seqidx in range(seq_len):
        k = seq_len - seqidx - 1
        hidden = mx.sym.flatten(data=slices_net[k])
        for i in range(num_lstm_layer):
            next_state = LSTM_Cell(
                num_hidden=num_hidden, 
                t_indata=hidden,
                last_state=last_states[2 * i + 1],
                param=backward_param[i],
                seq_idx=k, 
                layer_idx=1,
            )
            hidden = next_state.h
            last_states[2 * i + 1] = next_state
        backward_hidden.insert(0, hidden)

    hidden_all = []
    for i in range(seq_len):
        hidden_all.append(mx.sym.concat(*[forward_hidden[i], backward_hidden[i]], dim=1))

    hidden_concat = mx.sym.concat(*hidden_all, dim=0)

    pred = mx.sym.FullyConnected(data=hidden_concat, num_hidden=label_size)
    # arg_shape, output_shape, aux_shape = pred.infer_shape(
    #     **dict(init_values, **{"data": (batch_size, 1, 32, 256)}))
    # print(output_shape)
    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data=label, dtype='int32')
    sm = mx.sym.WarpCTC(data=pred, label=label, label_length=label_length, input_length=seq_len)
    # you can observer the parameter of network in this way.
    # mx.viz is not recommend for network which contains complex lstm
    # arg_shape, output_shape, aux_shape = sm.infer_shape(**dict(init_values, **{"data": (batch_size, 8, 32, 256),"label":(batch_size,num_label)}))
    # print(output_shape)
    # mx.viz.print_summary(sm,shape=dict(init_values, **{"data": (batch_size, 8, 32, 256),"label":(batch_size,num_label)}))
    return sm


# if __name__ == '__main__':
#     model = crnn(2,32,200,3820,24,0.3)
#     print model