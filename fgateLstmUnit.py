#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-5-9 上午10:16
# @Author  : Tianyu Liu

import jittor as jt
from jittor import Module
import pickle


class fgateLstmUnit(Module):
    def __init__(self, hidden_size, input_size, field_size, scope_name):
        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.field_size = field_size
        self.scope_name = scope_name

        self.W = jt.init.xavier_uniform((self.input_size + self.hidden_size, 4 * self.hidden_size))
        self.b = jt.zeros([4 * self.hidden_size])
        self.W1 = jt.init.xavier_uniform((self.field_size, 2 * self.hidden_size))
        self.b1 = jt.zeros([2 * hidden_size])

    def execute(self, x, fd, s, finished = None):
        """
        :param x: batch * input
        :param s: (h,s,d)
        :param finished:
        :return:
        """
        h_prev, c_prev = s  # batch * hidden_size

        x = jt.concat([x, h_prev], 1)

        i, j, f, o = jt.split(jt.matmul(x, self.W) + self.b, self.hidden_size, 1)
        r, d = jt.split(jt.matmul(fd, self.W1) + self.b1, self.hidden_size, 1)

        # Final Memory cell
        c = jt.sigmoid(f+1.0) * c_prev + jt.sigmoid(i) * jt.tanh(j) + jt.sigmoid(r) * jt.tanh(d)  # batch * hidden_size
        h = jt.sigmoid(o) * jt.tanh(c)

        out, state = h, (h, c)
        if finished is not None:
            finished = jt.view(finished, (-1, 1))
            out = (1 - jt.int(finished)) * h
            state = ((1 - jt.int(finished)) * h + jt.int(finished) * h_prev, (1 - jt.int(finished)) * c + jt.int(finished) * c_prev)
            # out = tf.multiply(1 - finished, h)
            # state = (tf.multiply(1 - finished, h) + tf.multiply(finished, h_prev),
            #          tf.multiply(1 - finished, c) + tf.multiply(finished, c_prev))

        return out, state