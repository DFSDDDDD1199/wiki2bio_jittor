import pickle
import logging
import numpy as np

from AttentionUnit import AttentionWrapper
from dualAttentionUnit import dualAttentionWrapper
from LstmUnit import LstmUnit
from fgateLstmUnit import fgateLstmUnit
from OutputUnit import OutputUnit

from jittor import Module
from jittor import nn
import jittor as jt

def clip_by_global_norm(x, clip_norm):
    global_norm = jt.sqrt(jt.sum(x*x))
    x * clip_norm / jt.maximum(global_norm, clip_norm)
    pass

class SeqUnit(Module):
    def __init__(self, hidden_size, emb_size, field_size, pos_size,
                 source_vocab, field_vocab, position_vocab, target_vocab,
                 field_concat, position_concat,
                 fgate_enc, dual_att,
                 encoder_add_pos, decoder_add_pos,
                 name, lr,
                 start_token=2, stop_token=2, max_length=150,
                 ):
        '''
            hidden_size: the size of the hidden state, used in fgateLstmUnit or LstmUnit
            emb_size: the size of the word embedding, used in the decoder LstmUnit or to calcuate the input size of the encoder fgateLstmUnit or LstmUnit
            field_size: the size of the table field embedding
            pos_size: the size of the word position embedding

            field_concat: default False, whether to concatenate the field embedding to the word embedding in the encoder input
            position_concat: default False, whether to concatenate the position embedding to the word embedding in the encoder input
            encoder_add_pos: default True, whether to concatenate the position embedding to the field embedding in the fgateLstmUnit input
            decoder_add_pos: default True, whether to concatenate the position embedding to the field embedding in the dualAttentionWrapper input

            uni_size: the size of the input vector to the encoder
            field_encoder_size: the size of the input vector to the fgateLstmUnit
            field_attention_size: the size of the input vector to the dualAttentionWrapper

            source_vocab: default 20003, the size of the input word vocabulary
            field_vocab: default 1480, the size of the field vocabulary
            position_vocab: default 31, the size of the position vocabulary
            target_vocab: default 20003, the size of the output word vocabulary

            fgate_enc: default True, whether to use the fgateLstmUnit as the encoder LSTM
            dual_att: default True, whether to use the dual attention layer in the decoder LSTM

            name: used when saving the model parameters

            start_token: default 2, used in the decoder stage
            stop_token: default 2, used in the decoder stage
            max_length: default 150, used in the decoder stage


            Compared to the Tensorflow version, the following parameters are excluded:
                batch_size,
                learning_rate,
                scope_name

            word index 0 -> Padding
            word index 1 -> START_TOKEN
            word index 2 -> END_TOKEN
            word index 3 -> UNK_TOKEN
        '''

        super().__init__()
        # assignments of parameters related to the size of vectors
        self.hidden_size = hidden_size
        self.emb_size = emb_size  # embedding of the word
        self.field_size = field_size
        self.pos_size = pos_size
        # assignments of parameters related to the concatenation of vectors
        self.field_concat = field_concat
        self.position_concat = position_concat
        self.encoder_add_pos = encoder_add_pos
        self.decoder_add_pos = decoder_add_pos
        # calculate the size of vectors that are used in the encoder or the decoder
        self.uni_size = self.emb_size if (not self.field_concat) else (self.emb_size + self.field_size)
        self.uni_size = self.uni_size if (not self.position_concat) else (self.uni_size + 2 * self.pos_size)
        self.field_encoder_size = field_size if (not self.encoder_add_pos) else (self.field_size + 2 * self.pos_size)
        self.field_attention_size = field_size if (not self.decoder_add_pos) else (self.field_size + 2 * self.pos_size)
        # assignments of parameters related to the size of vocabularies
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.field_vocab = field_vocab
        self.position_vocab = position_vocab
        # assignments of parameters related to the encoder/decoder structure
        self.fgate_enc = fgate_enc
        self.dual_att = dual_att
        # assignments of parameters used in the decoder stage
        self.start_token = start_token
        self.stop_token = stop_token
        self.max_length = max_length
        # other parameters
        self.name = name
        self.grad_clip = 5.0  # ? might be of no use

        self._lr = lr

        # network components related to encoder and decoder
        if self.fgate_enc:
            logging.info("field-gated encoder LSTM")
            self.enc_lstm = fgateLstmUnit(self.hidden_size, self.uni_size, self.field_encoder_size, 'encoder_select')
            # even though the scope_name might have no meaning in jittor, still have it to make the function interface consistent
        else:
            logging.info("normal encoder LSTM")
            self.enc_lstm = LstmUnit(self.hidden_size, self.uni_size, 'encoder_lstm')

        self.dec_lstm = LstmUnit(self.hidden_size, self.emb_size, "decoder_lstm")
        self.dec_out = OutputUnit(self.hidden_size, self.target_vocab, "decoder_output")

        self.units.update({
            'encoder_lstm': self.enc_lstm,
            'decoder_lstm': self.dec_lstm,
            'decoder_output': self.dec_out
        })

        # network components related to attention units
        if self.dual_att:
            logging.info('dual attention mechanism used')
            self.att_layer = dualAttentionWrapper(self.hidden_size, self.hidden_size, self.field_attention_size,
                                                  "attention")
            self.units.update({'attention': self.att_layer})
            # remove the en_outputs parameter
        else:
            logging.info("normal attention used")
            self.att_layer = AttentionWrapper(self.hidden_size, self.hidden_size, "attention")
            self.units.update({'attention': self.att_layer})
            # remove the en_outputs parameter

        # parameters of embedding matrices
        self.word_embedding = nn.Embedding(self.source_vocab, self.emb_size)
        # ??? alarming...there is a stop_grad() of the weight inside the implementation of embedding,
        # therefore I am not sure whether this embedding will not be trained

        if self.field_concat or self.fgate_enc or self.encoder_add_pos or self.decoder_add_pos:
            self.field_embedding = nn.Embedding(self.field_vocab, self.field_size)
        if self.position_concat or self.encoder_add_pos or self.decoder_add_pos:
            self.position_embedding = nn.Embedding(self.position_vocab, self.pos_size)
            self.right_position_embedding = nn.Embedding(self.position_vocab, self.pos_size)

        if self.field_concat or self.fgate_enc:
            self.params.update({'fembedding': self.field_embedding})
        if self.position_concat or self.encoder_add_pos or self.decoder_add_pos:
            self.params.update({'pembedding': self.position_embedding})
            self.params.update({'rembedding': self.right_position_embedding})
        self.params.update({'embedding': self.word_embedding})

    def execute(self, batched_data):
        '''
            encoder_input <--> batched_data['enc_in']
                the zero-padded words in the table, shape (batch_size, max_text_len)
            encoder_field <--> batched_data['enc_fd']
                the zero-padded field in the table, shape (batch_size, max_text_len)
            encoder_pos <--> batched_data['enc_pos']
                the zero-padded position in the table, shape (batch_size, max_text_len)
            encoder_rpos <--> batched_data['enc_rpos']
                the zero-padded right position in the table, shape (batch_size, max_text_len)
            encoder_len <--> batched_data['enc_len']
                the actual length of the words/fields/pos/rpos (without padding), shape (batch_size, 1)
            decoder_input <--> batched_data['dec_in']
                the zero-padded summary, shape (batch_size, max_summary_len)
            decoder_output <--> batched_data['dec_out']
                the summary that is appended first an END_TOKEN and then zero-padded
            decoder_len <--> batch_data['dec_len']
                the actual length of the summary (without any END_TOKEN or padding)
        '''
        # encoder related data, the original type is numpy
        encoder_input = jt.array(batched_data['enc_in'])
        encoder_field = jt.array(batched_data['enc_fd'])
        encoder_pos = jt.array(batched_data['enc_pos'])
        encoder_rpos = jt.array(batched_data['enc_rpos'])
        encoder_len = jt.array(batched_data['enc_len'])

        # decoder related data
        decoder_input = jt.array(batched_data['dec_in'])
        decoder_output = jt.array(batched_data['dec_out'])
        decoder_len = jt.array(batched_data['dec_len'])

        # ======================================== embedding lookup ======================================== #
        encoder_embed = self.word_embedding(encoder_input)  # (batchsize, max_text_len, emb_size)
        decoder_embed = self.word_embedding(decoder_input)  # (batchsize, max_summary_len, emb_size)

        if self.field_concat or self.fgate_enc or self.encoder_add_pos or self.decoder_add_pos:
            # if the field embedding is needed
            field_embed = self.field_embedding(encoder_field)  # (batchsize, max_text_len, field_size)
            field_pos_embed = field_embed
            if self.field_concat:
                # if we requires the field embedding is concatenated to the word embedding
                encoder_embed = jt.concat([encoder_embed, field_embed],
                                          dim=2)  # (batchsize, max_text_len, emb_size + field_size)
        
        field_pos_embed = None
        if self.position_concat or self.encoder_add_pos or self.decoder_add_pos:
            pos_embed = self.position_embedding(encoder_pos)  # (batchsize, max_text_len, pos_size)
            rpos_embed = self.right_position_embedding(encoder_rpos)  # (batchsize, max_text_len, pos_size)
            if self.position_concat:
                encoder_embed = jt.concat([encoder_embed, pos_embed, rpos_embed],
                                          dim=2)  # (batchsize, max_text_len, emb_size + field_size + 2 * pos_size)
                field_pos_embed = jt.concat([field_embed, pos_embed, rpos_embed], dim=2)
            elif self.encoder_add_pos or self.decoder_add_pos:
                field_pos_embed = jt.concat([field_embed, pos_embed, rpos_embed], dim=2)

        # ======================================== encoder ======================================== #
        if self.fgate_enc:
            logging.info('field gated encoder used')
            en_outputs, en_state = self.fgate_encoder(encoder_embed, field_pos_embed, encoder_len)
        else:
            logging.info('normal encoder used')
            en_outputs, en_state = self.encoder(encoder_embed, encoder_len)
            # en_outputs shape (batch_size, max_text_len, hidden_size)
            # en_state (shape (batch_size, hidden_size), shape (batch_size, hidden_size))
        # ======================================== decoder ======================================== #
        if self.is_training:
            # decoder for training
            de_outputs, de_state = self.decoder_t(en_outputs, field_pos_embed, en_state, decoder_embed, decoder_len)
            de_outputs = jt.nn.softmax(de_outputs, dim=-1)
            loss = jt.nn.cross_entropy_loss
            ori_shape = decoder_output.shape
            de_outputs.view(-1, de_outputs.shape[-1])
            decoder_output.view(-1)
            losses = loss(de_outputs.view(-1, de_outputs.shape[-1]), decoder_output.view(-1), reduction=None).view(ori_shape)
            mask = jt.nn.sign(jt.float(decoder_output))
            losses = mask * losses
            mean_loss = jt.mean(losses)
            optimizer = jt.nn.Adam(self.parameters(), lr=self._lr)
            optimizer.step(mean_loss)
            return mean_loss
        else:
            # decoder for testing
            g_tokens, atts = self.decoder_g(en_outputs, field_pos_embed, en_state)
            return g_tokens, atts

    def encoder(self, inputs, inputs_len):
        # different from the original code which uses the encoder_input, but should be equivalent
        batch_size = inputs.shape[0]
        max_time = inputs.shape[1]
        hidden_size = self.hidden_size

        time = 0
        h0 = (jt.zeros([batch_size, hidden_size], dtype=jt.float32),
              jt.zeros([batch_size, hidden_size], dtype=jt.float32))
        f0 = jt.zeros([batch_size], dtype=jt.bool)
        # indicates whether we have read all of the words in a sample of a batch
        inputs_ta = jt.permute(inputs, [1, 0, 2])
        # exchange the dimension of batch_size and words
        # so that we feed into the lstm a batch of words
        emit_ta = []
        # store the output state of lstm

        x_t = inputs_ta[0, :, :]
        s_t = h0
        finished = f0
        while not finished.all():
            o_t, s_t = self.enc_lstm(x_t, s_t, finished)
            emit_ta.append(o_t)
            finished = (time + 1) >= inputs_len
            x_t = jt.zeros((batch_size, self.uni_size), dtype=jt.float32) if finished.all() else inputs_ta[time + 1, :,
                                                                                                 :]
            time = time + 1
        emit_ta = jt.stack(emit_ta, dim=0)
        outputs = jt.permute(emit_ta, [1, 0, 2])

        state = s_t
        return outputs, state

    def fgate_encoder(self, inputs, fields, inputs_len):
        # different from the original code which uses the encoder_input, but should be equivalent
        batch_size = inputs.shape[0]
        max_time = inputs.shape[1]
        hidden_size = self.hidden_size

        time = 0
        h0 = (jt.zeros([batch_size, hidden_size], dtype=jt.float32),
              jt.zeros([batch_size, hidden_size], dtype=jt.float32))
        finished = jt.zeros([batch_size], dtype=jt.bool)
        inputs_ta = jt.permute(inputs, [1, 0, 2])
        fields_ta = jt.permute(fields, [1, 0, 2])
        emit_ta = []

        x_t = inputs_ta[0, :, :]
        d_t = fields_ta[0, :, :]
        s_t = h0
        # finished = f0
        while not finished.all():
            o_t, s_t = self.enc_lstm(x_t, d_t, s_t, finished)
            emit_ta.append(o_t)
            finished = (time + 1) >= inputs_len
            x_t = jt.zeros((batch_size, self.uni_size), dtype=jt.float32) if finished.all() else inputs_ta[time + 1, :,
                                                                                                 :]
            d_t = jt.zeros((batch_size, self.field_attention_size), dtype=jt.float32) if finished.all() else fields_ta[
                                                                                                             time + 1,
                                                                                                             :, :]
            # although self.field_attention_size = self.field_encoder_size, the input parameter of fgateLstmUnit self.field_encoder_size and
            # therefore this should be used.
            time = time + 1
        emit_ta = jt.stack(emit_ta, dim=0)
        outputs = jt.permute(emit_ta, [1, 0, 2])  # shape (batch_size, max_text_len, hidden_size)

        state = s_t  # (shape (batch_size, hidden_size), shape (batch_size, hidden_size))
        return outputs, state

    def decoder_t(self, en_outputs, field_inputs, initial_state, inputs, inputs_len):
        # en_outputs, field_inputs, initial_state, inputs, inputs_len: en_outputs, field_pos_embed, en_state, self.decoder_embed, self.decoder_len

        batch_size = inputs.shape[0]
        max_time = inputs.shape[1]
        # encoder_len = tf.shape(self.encoder_input)[1]

        time = 0
        h0 = initial_state
        f0 = jt.zeros([batch_size], dtype=jt.bool)
        x0 = self.word_embedding(jt.full([batch_size], self.start_token))
        inputs_ta = jt.permute(inputs, [1, 0, 2])
        emit_ta = []

        x_t = x0
        s_t = h0
        finished = f0
        while not finished.all():
            o_t, s_t = self.dec_lstm(x_t, s_t, finished)
            if self.dual_att:
                o_t, _ = self.att_layer(o_t, en_outputs, field_inputs)  # add the en_outputs and field_inputs here
            else:
                o_t, _ = self.att_layer(o_t, en_outputs) # for ordinary attention units
            o_t = self.dec_out(o_t, finished)
            emit_ta.append(o_t)
            finished = time >= inputs_len
            x_t = jt.zeros((batch_size, self.emb_size), dtype=jt.float32) if finished.all() else inputs_ta[time, :, :]
            time = time + 1

        emit_ta = jt.stack(emit_ta, dim=0)
        outputs = jt.permute(emit_ta, [1, 0, 2])

        state = s_t
        return outputs, state

    def decoder_g(self, en_outputs, field_inputs, initial_state):
        # initial_state (shape (batch_size, hidden_size), shape (batch_size, hidden_size))
        # different from the original code which uses the encoder_input, but should be equivalent
        batch_size = jt.shape(initial_state[0].shape)[0]
        # encoder_len = tf.shape(self.encoder_input)[1]

        time = 0
        h0 = initial_state
        f0 = jt.zeros([batch_size], dtype=jt.bool)
        x0 = self.word_embedding(jt.full([batch_size], self.start_token))
        emit_ta = []
        att_ta = []

        x_t = x0
        s_t = h0
        finished = f0
        while not finished.all():
            o_t, s_t = self.dec_lstm(x_t, s_t, finished)
            if self.dual_att:
                o_t, w_t = self.att_layer(o_t, en_outputs, field_inputs)  # add the en_outputs and field_inputs here
            else:
                o_t, w_t = self.att_layer(o_t, en_outputs) # for ordinary attention units
            o_t = self.dec_out(o_t, finished)
            emit_ta.append(o_t)
            att_ta.append(w_t)
            next_token, _ = jt.argmax(o_t, 1)  # the jt.argmax will return a tuple indicating the index and the value
            x_t = self.word_embedding(next_token)
            finished = jt.bitwise_or(finished, (next_token == self.stop_token))
            finished = jt.bitwise_or(finished, (time >= self.max_length))
            time = time + 1

        emit_ta = jt.stack(emit_ta, dim=0)
        outputs = jt.permute(emit_ta, [1, 0, 2])
        pred_tokens = jt.argmax(outputs, 2)
        atts = jt.stack(att_ta, dim=0)
        return pred_tokens, atts

    def save(self, path):
        for u in self.units:
            # call the save function of each unit
            self.units[u].save(path + u + ".pkl")

        jt.save(self.params, path + self.name + ".pkl")

    def load(self, path):
        for u in self.units:
            self.units[u].load(path + u + ".pkl")
        self.params = jt.load(path + self.name + ".pkl")