#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-5-12 下午10:47
# @Author  : Tianyu Liu

import jittor as jt
from jittor import Module
import pickle


class dualAttentionWrapper(Module):
    def __init__(self, hidden_size, input_size, field_size, scope_name):
        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.field_size = field_size
        self.scope_name = scope_name

        self.Wh = jt.init.xavier_uniform((input_size, hidden_size))
        self.bh = jt.init.uniform((hidden_size))
        self.Ws = jt.init.xavier_uniform((input_size, hidden_size))
        self.bs = jt.init.uniform((hidden_size))
        self.Wo = jt.init.xavier_uniform((2 * input_size, hidden_size))
        self.bo = jt.init.uniform((hidden_size))
        self.Wf = jt.init.xavier_uniform((field_size, hidden_size))
        self.bf = jt.init.uniform((hidden_size))
        self.Wr = jt.init.xavier_uniform((input_size, hidden_size))
        self.br = jt.init.uniform((hidden_size))

    def execute(self, x, hs, fds, coverage = None, finished = None):
        hs = jt.permute(hs, [1, 0, 2])  # input_len * batch * input_size
        fds = jt.permute(fds, [1, 0, 2])

        hs2d = jt.reshape(hs, [-1, self.input_size])
        phi_hs2d = jt.tanh(jt.matmul(hs2d, self.Wh) + self.bh)
        phi_hs = jt.reshape(phi_hs2d, hs.shape)
        fds2d = jt.reshape(fds, [-1, self.field_size])
        phi_fds2d = jt.tanh(jt.matmul(fds2d, self.Wf) + self.bf)
        phi_fds = jt.reshape(phi_fds2d, hs.shape)

        gamma_h = jt.tanh(jt.matmul(x, self.Ws) + self.bs)  # batch * hidden_size
        alpha_h = jt.tanh(jt.matmul(x, self.Wr) + self.br)
        fd_weights = jt.sum(phi_fds * alpha_h, dim=2, keepdims=True)
        fd_weights = jt.exp(fd_weights - jt.max(fd_weights, dim=0, keepdims=True))
        fd_weights = jt.divide(fd_weights, (1e-6 + jt.sum(fd_weights, dim=0, keepdims=True)))

        weights = jt.sum(phi_hs * gamma_h, dim=2, keepdims=True)  # input_len * batch
        weights = jt.exp(weights - jt.max(weights, dim=0, keepdims=True))
        weights = jt.divide(weights, (1e-6 + jt.sum(weights, dim=0, keepdims=True)))
        weights = jt.divide(weights * fd_weights, (1e-6 + jt.sum(weights * fd_weights, dim=0, keepdims=True)))
        
        context = jt.sum(hs * weights, dim=0)  # batch * input_size

        out = jt.tanh(jt.matmul(jt.concat([context, x], -1), self.Wo) + self.bo)

        if finished is not None:
            finished = jt.view(finished, [-1, 1])
            out = (1 - jt.int(finished)) * out
        return out, weights

