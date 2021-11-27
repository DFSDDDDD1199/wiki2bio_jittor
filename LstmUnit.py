#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-27 下午8:34
# @Author  : Tianyu Liu

import jittor as jt
from jittor import Module
import pickle


class LstmUnit(Module):
    def __init__(self, hidden_size, input_size, scope_name):
        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.scope_name = scope_name

        self.W = jt.init.xavier_uniform((self.input_size + self.hidden_size, 4 * self.hidden_size))
        self.b = jt.zeros([4 * self.hidden_size])

    def execute(self, x, s, finished = None):
        h_prev, c_prev = s

        x = jt.concat([x, h_prev], 1)
        i, j, f, o = jt.split(jt.matmul(x, self.W) + self.b, self.hidden_size, 1)

        # Final Memory cell
        c = jt.sigmoid(f+1.0) * c_prev + jt.sigmoid(i) * jt.tanh(j)
        h = jt.sigmoid(o) * jt.tanh(c)

        out, state = h, (h, c)
        if finished is not None:
            finished = jt.view(finished, [-1, 1])
            out = (1 - jt.int(finished)) * h
            state = ((1 - jt.int(finished)) * h + jt.int(finished) * h_prev,
                     (1 - jt.int(finished)) * c + jt.int(finished) * c_prev)
            # out = tf.multiply(1 - finished, h)
            # state = (tf.multiply(1 - finished, h) + tf.multiply(finished, h_prev),
            #          tf.multiply(1 - finished, c) + tf.multiply(finished, c_prev))

        return out, state
