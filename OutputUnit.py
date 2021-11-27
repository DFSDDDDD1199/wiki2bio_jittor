#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-27 下午8:36
# @Author  : Tianyu Liu

import jittor as jt
from jittor import Module
import pickle


class OutputUnit(Module):
    def __init__(self, input_size, output_size, scope_name):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.scope_name = scope_name

        self.W = jt.init.xavier_uniform((input_size, output_size))
        self.b = jt.zeros([output_size])

    def execute(self, x, finished = None):
        out = jt.matmul(x, self.W) + self.b

        if finished is not None:
            # out = jt.where(finished, jt.zeros_like(out), out)
            finished = jt.view(finished, [-1, 1])
            out = jt.int(1 - finished) * out
        return out

