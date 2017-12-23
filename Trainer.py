#! /usr/bin/python
# -*- coding:utf-8 -*-

# Init Mxnet Module
import mxnet as mx
from mxnet.model import BatchEndParam

# Init Basic Module
import cv2
import numpy as np
import random
import os, logging, platform, time

# Init Private Module
import Config
from CRNN import crnn

class Data_Iter(mx.io.DataIter):
    def __init__(self, batch_size, data_path, data_shape, init_states, data_dimensions=1, label_length=6):
        super(Data_Iter, self).__init__()
        self.batch_size = batch_size
        self.data_dimensions = data_dimensions
        self.data_path = data_path
        self.data_shape = data_shape
        self.label_length = label_length

        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

        self.provide_data = [('data', (batch_size, self.data_dimensions, self.data_shape[0], self.data_shape[1]))] + init_states

        self.provide_label = [('label', (self.batch_size, self.label_length))]

        self.reset()

    def reset(self):
        self.data_list = os.listdir(self.data_path)
        random.shuffle(self.data_list)
        self.data_count = len(self.data_list)
        self.data_index = 0

    def iter_next(self):
        return self.data_count > self.data_index

    def next(self):
        if self.iter_next():
            data_padding = (self.data_index + self.batch_size) - self.data_count
            data_padding = 0 if data_padding <= 0 else data_padding

            data = []
            label = []

            for i in range(self.batch_size - data_padding):
                # Load Data
                img = cv2.imread(self.data_path + self.data_list[self.data_index], cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, self.data_shape)
                img = img.transpose(1, 0)
                img = img.reshape(self.data_shape)
                img = np.multiply(img, 1 / 255.0)

                data.append([img,])
                label.append(self.Get_Labels(self.data_list[self.data_index]))

                self.data_index += 1

            data = mx.nd.array(data)
            label = mx.nd.array(label)

            if data_padding > 0:
                # Data Pading
                tmp_data_padding = mx.nd.zeros(shape=(data_padding,) + data[0].shape)
                data = mx.nd.concatenate([data, tmp_data_padding], axis=0)

                # Label Pading
                tmp_label_pading = mx.nd.zeros(shape=(data_padding,) + label[0].shape)
                label = mx.nd.concatenate([label, tmp_label_pading], axis=0)

            data_all = [data] + self.init_state_arrays
            label_all = [label]

            return mx.io.DataBatch(data=data_all, label=label_all)

        else:
            raise StopIteration

    def Get_Labels(self, label_text):
        tmp_label = np.zeros(self.label_length)
        label_raw = label_text.split("_")[0]
        for i in range(len(label_raw)):
            tmp_label[i] = Config.chars_index[label_raw[i].lower()]
        return tmp_label

def ctc_label(p):
    ret = []
    p1 = [0] + p
    for i in range(len(p)):
        c1 = p1[i]
        c2 = p1[i+1]
        if c2 == 0 or c2 == c1:
            continue
        ret.append(c2)
    return ret

def remove_blank(l):
    ret = []
    for i in range(len(l)):
        if l[i] == 0:
            break
        ret.append(l[i])
    return ret

def Accuracy(label, pred):
    global Model_Batch_Size
    global Model_LSTM_SEQ_Length

    hit = 0.
    total = 0.
    for i in range(Model_Batch_Size):
        l = remove_blank(label[i])
        p = []
        for k in range(Model_LSTM_SEQ_Length):
            p.append(np.argmax(pred[k * Model_Batch_Size + i]))
        p = ctc_label(p)
        if len(p) == len(l):
            match = True
            for k in range(len(p)):
                if p[k] != int(l[k]):
                    match = False
                    break
            if match:
                hit += 1.0
        total += 1.0
    return hit / total

Model_Batch_Size = 500
Model_LSTM_SEQ_Length = 33

class Trainer_Manager():
    def __init__(self):
        # ===== Config =====
        self.Data_Shape = (32, 128)
        self.Data_Train_Path = "/home/alues/Downloads/Ok/"
        self.Data_Val_Path = "/home/alues/Downloads/Ok/"

        # ===== Module Config =====
        self.Model_Batch_Size = Model_Batch_Size
        self.Model_Optimizer = 'sgd'
        self.Model_Learn_Rate = 0.001

        # ===== Model Pre Training =====
        self.Model_Pre_Training = True
        self.Model_Pre_Training_Prefix = "Epoch_Saved/ocr_checkpoint"
        self.Model_Pre_Training_Epoch = 47

        # ===== Runing Config =====
        GPU_Num = 1
        self.Model_Device = [mx.gpu(i) for i in range(GPU_Num)]

        # ==== Module Init =====
        self.__Init_LSTM()
        self.__Init_Data_Iter()
        self.__Init_Model_Net()
        self.__Init_Model_Params()
        self.__Init_Model()
        self.__Train()
        print "Done"

    def __Init_LSTM(self):
        self.Model_LSTM_Layer_Num = 2
        self.Model_LSTM_Hidden_Num = 256
        self.Model_LSTM_SEQ_Length = Model_LSTM_SEQ_Length
        self.Model_LSTM_Label_Length = 6

        init_c = [('l%d_init_c' % l, (self.Model_Batch_Size, self.Model_LSTM_Hidden_Num)) for l in
                  range(self.Model_LSTM_Layer_Num * 2)]
        init_h = [('l%d_init_h' % l, (self.Model_Batch_Size, self.Model_LSTM_Hidden_Num)) for l in
                  range(self.Model_LSTM_Layer_Num * 2)]
        self.LSTM_init_states = init_c + init_h

    def __Init_Data_Iter(self):
        self.Data_Train_Iter = Data_Iter(
            batch_size=self.Model_Batch_Size,
            data_path=self.Data_Train_Path,
            data_shape=self.Data_Shape,
            label_length=self.Model_LSTM_Label_Length,
            init_states=self.LSTM_init_states,
        )

        # self.Data_Val_Iter = Data_Iter(
        #     batch_size=self.Model_Batch_Size,
        #     data_path=self.Data_Val_Path,
        #     data_shape=self.Data_Shape,
        #     label_length=self.Model_LSTM_Label_Length,
        #     init_states=self.LSTM_init_states
        # )

        self.Data_Val_Iter =None

    def __Init_Model_Net(self):
        if self.Model_Pre_Training:
            self.Model_Net = mx.sym.load('%s-symbol.json' % self.Model_Pre_Training_Prefix)
        else:
            self.Model_Net = crnn(
                num_lstm_layer=self.Model_LSTM_Layer_Num,
                seq_len=self.Model_LSTM_SEQ_Length,
                num_hidden=self.Model_LSTM_Hidden_Num,
                label_length=self.Model_LSTM_Label_Length,
                label_size=len(Config.chars_index),
                dropout=0.,
            )

    def __Init_Model_Params(self):
        if self.Model_Pre_Training:
            save_dict = mx.nd.load('%s-%04d.params' % (self.Model_Pre_Training_Prefix, self.Model_Pre_Training_Epoch))
            self.arg_params = {}
            self.aux_params = {}
            for k, v in save_dict.items():
                tp, name = k.split(':', 1)
                if tp == 'arg':
                    self.arg_params[name] = v
                if tp == 'aux':
                    self.aux_params[name] = v
            self.Model_Initializer = mx.init.Uniform(0.01)
        else:
            self.arg_params = None
            self.aux_params = None
            self.Model_Initializer = mx.init.Xavier(factor_type="in", magnitude=2.34)

    def __Init_Model(self):
        self.Model = mx.mod.Module(
            symbol=self.Model_Net,
            context=self.Model_Device,
            data_names=[items[0] for items in self.Data_Train_Iter.provide_data],
            label_names=[items[0] for items in self.Data_Train_Iter.provide_label],
            logger=logging,
        )

        # ===== Module Status Init =====
        self.Model.bind(
            data_shapes=self.Data_Train_Iter.provide_data,
            label_shapes=self.Data_Train_Iter.provide_label,
            for_training=True,
            force_rebind=True,
        )

        self.Model.init_params(
            initializer=self.Model_Initializer,
            arg_params=self.arg_params,
            aux_params=self.aux_params,
            allow_missing=True,
            force_init=True,
        )

        self.Model.init_optimizer(
            optimizer=self.Model_Optimizer,
            optimizer_params={
                'learning_rate': self.Model_Learn_Rate,
                'wd': 0.000001,
                # 'beta1': 0.5,
            }
        )

    def __Train(self):
        begin_epoch = 0
        num_epoch = 1000

        eval_metric = mx.metric.np(Accuracy, allow_extra_outputs=True)
        validation_metric = None

        batch_end_callback = mx.callback.Speedometer(self.Model_Batch_Size, frequent=10),
        epoch_end_callback = mx.callback.do_checkpoint("Epoch_Saved/ocr_checkpoint", period=1)

        eval_end_callback = None
        eval_batch_end_callback = None

        if validation_metric is None:
            validation_metric = eval_metric
        if not isinstance(eval_metric, mx.metric.EvalMetric):
            eval_metric = mx.metric.create(eval_metric)

        ################################################################################
        # Training Loop                                                                #
        ################################################################################

        for epoch in range(begin_epoch, num_epoch):
            tic = time.time()

            eval_metric.reset()
            nbatch = 0
            data_iter = iter(self.Data_Train_Iter)
            end_of_batch = False
            next_data_batch = next(data_iter)

            while not end_of_batch:
                data_batch = next_data_batch

                self.Model.forward_backward(data_batch)
                self.Model.update()

                try:
                    # pre fetch next batch
                    next_data_batch = next(data_iter)
                    self.Model.prepare(next_data_batch)
                except StopIteration:
                    end_of_batch = True

                self.Model.update_metric(eval_metric, data_batch.label)

                if batch_end_callback is not None:
                    batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                     eval_metric=eval_metric,
                                                     locals=locals())
                    for callback in mx.module.base_module._as_list(batch_end_callback):
                        callback(batch_end_params)

                nbatch += 1

            # one epoch of training is finished
            for name, val in eval_metric.get_name_value():
                self.Model.logger.info('Epoch[%d] Train-%s=%f', epoch, name, val)

            toc = time.time()
            self.Model.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc - tic))

            # sync aux params across devices
            arg_params, aux_params = self.Model.get_params()
            self.Model.set_params(arg_params, aux_params)

            if epoch_end_callback is not None:
                for callback in mx.module.base_module._as_list(epoch_end_callback):
                    callback(epoch, self.Model.symbol, arg_params, aux_params)

            # ----------------------------------------
            # evaluation on validation set
            if self.Data_Val_Iter:
                res = self.Model.score(self.Data_Val_Iter, validation_metric,
                                       score_end_callback=eval_end_callback,
                                       batch_end_callback=eval_batch_end_callback, epoch=epoch)
                # TODO: pull this into default
                for name, val in res:
                    self.Model.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)

            # end of 1 epoch, reset the data-iter for another epoch
            self.Data_Train_Iter.reset()


if __name__ == '__main__':
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('Running on %s', platform.node())

    Trainer_Manager()
