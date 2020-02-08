import numpy as np
import os
import tensorflow as tf

class VanillaRNNBase(object):
    def __init__(self, batch_size=24, hidden_units=64, timesteps=250,
                 num_classes=2, embed=None):
        self.batch_size   = batch_size
        self.hidden_units = hidden_units
        self.timesteps    = timesteps
        self.num_classes  = num_classes
        self.embed        = embed
        self.embed_shape  = embed.shape

        self.x_ph   = tf.placeholder(tf.int32,   [self.batch_size, self.timesteps])
        self.y_ph   = tf.placeholder(tf.float32, [self.batch_size, self.num_classes])
        self.len_ph = tf.placeholder(tf.int32)

        self.embed = self.get_embed(self.embed)
        self.cell_weight, self.cell_bias, self.fc_weight, self.fc_bias = self.get_weights()

        self.data = tf.nn.embedding_lookup(self.embed, self.x_ph)
        self.data = tf.transpose(self.data, [1, 0, 2])

        self.h_t     = tf.zeros([self.batch_size, self.hidden_units], dtype=tf.float32)
        self.history = [self.h_t]

        for step in range(self.timesteps):
            self.input   = tf.concat([self.data[step], self.h_t], 1)
            self.h_t     = tf.tanh(tf.add(tf.matmul(self.input, self.cell_weight), self.cell_bias))
            self.history = tf.concat([self.history, [self.h_t]], 0)

        self.indices = tf.stack([tf.range(batch_size), self.len_ph], 1)
        self.history = tf.transpose(self.history, [1, 0, 2])
        self.outputs = tf.gather_nd(self.history, self.indices)
        self.predict = tf.add(tf.matmul(self.outputs, self.fc_weight), self.fc_bias)
        self.correct = tf.equal(tf.argmax(self.predict, 1), tf.argmax(self.y_ph, 1))

        self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))


    def get_embed(self, embed):
        raise NotImplementedError


    def get_weights(self):
        raise NotImplementedError



class VanillaRNN(VanillaRNNBase):
    def __init__(self, batch_size=24, hidden_units=64, timesteps=250, num_classes=2,
                 embed=None, reg=False):
        super().__init__(batch_size, hidden_units, timesteps, num_classes,
                         embed)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                   logits=self.predict, labels=self.y_ph))
        if reg:
            reg_weight = tf.nn.l2_loss(self.cell_weight) + tf.nn.l2_loss(self.fc_weight)
            reg_embed = tf.nn.l2_loss(self.embed)
            self.loss += 1e-4 * (reg_weight + reg_embed)


    def get_embed(self, embed):
        return embed


    def get_weights(self):
        cell_weight = tf.get_variable('cell_weight', [self.embed_shape[1] + self.hidden_units,
                                                      self.hidden_units])
        cell_bias   = tf.get_variable('cell_bias',   [self.hidden_units])
        fc_weight   = tf.get_variable('fc_weight',   [self.hidden_units, self.num_classes])
        fc_bias     = tf.get_variable('fc_bias',     [self.num_classes])

        return cell_weight, cell_bias, fc_weight, fc_bias



class VanillaRNNDual(VanillaRNNBase):
    def __init__(self, weight_path, batch_size=24, hidden_units=64, timesteps=250,
                 num_classes=2, embed=None, diff_path=None, params=None,
                 seed=1024):
        self.weight_path = weight_path

        super().__init__(batch_size, hidden_units, timesteps, num_classes,
                         embed)

        self.diverge  = np.load(diff_path)
        self._theta   = params[0]
        self._epsilon = params[1]
        self._lambda  = params[2]

        faulty_dim = self.get_faulty()
        perturb_i = self.get_perturb((0, self.embed_shape[1]), faulty_dim,
                                     self._epsilon, seed)
        perturb_h = self.get_perturb((self.embed_shape[1], len(self.diverge)),
                                     faulty_dim, self._lambda, seed)

        self.data_ptb = tf.nn.embedding_lookup(self.embed, self.x_ph)
        self.data_ptb = tf.add(self.data_ptb, perturb_i)
        self.data_ptb = tf.transpose(self.data_ptb, [1, 0, 2])


        self.h_t_ptb  = tf.zeros([self.batch_size, self.hidden_units], dtype=tf.float32)
        self.history_ptb = [self.h_t_ptb]

        for step in range(self.timesteps):
            self.h_t_ptb     = tf.add(self.h_t_ptb, perturb_h)
            self.input_ptb   = tf.concat([self.data_ptb[step], self.h_t_ptb], 1)
            self.h_t_ptb     = tf.tanh(tf.add(tf.matmul(self.input_ptb, self.cell_weight),
                                                        self.cell_bias))
            self.history_ptb = tf.concat([self.history_ptb, [self.h_t_ptb]], 0)

        self.history_ptb = tf.transpose(self.history_ptb, [1, 0, 2])
        self.outputs_ptb = tf.gather_nd(self.history_ptb, self.indices)
        self.predict_ptb = tf.add(tf.matmul(self.outputs_ptb, self.fc_weight), self.fc_bias)

        loss_ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                 logits=self.predict, labels=self.y_ph))
        loss_l2 = tf.nn.l2_loss(self.outputs_ptb - self.outputs)

        self.loss = loss_ce + loss_l2



    def get_faulty(self):
        diverge = sorted(self.diverge)[::-1]
        dim = int(round(self._theta * len(diverge))) - 1

        return diverge[dim]


    def get_perturb(self, ptb_range, faulty_dim, stddev, seed):
        if ptb_range[0] == 0:
            shape = np.array([self.batch_size, self.timesteps])
            axis  = 2
        else:
            shape = np.array([self.batch_size])
            axis  = 1

        perturb = tf.zeros(np.append(shape, 0), dtype=tf.float32)
        ptb_vec = tf.random_normal(np.append(shape, 1),
                                   stddev=stddev, seed=seed)
        hold_dims  = 0
        buggy_dims = 0
        for i in range(ptb_range[0], ptb_range[1]):
            if self.diverge[i] > faulty_dim:
                buggy_dims += 1
                if hold_dims > 0:
                    hold_vec = tf.zeros(np.append(shape, hold_dims),
                                        dtype=tf.float32)
                    perturb  = tf.concat([perturb, hold_vec, ptb_vec], axis)
                    hold_dims = 0
                else:
                    perturb = tf.concat([perturb, ptb_vec], axis)
            else:
                hold_dims += 1
        hold_vec = tf.zeros(np.append(shape, hold_dims),
                            dtype=tf.float32)
        perturb  = tf.concat([perturb, hold_vec], axis)
        print(perturb.shape, buggy_dims, '/', ptb_range[1] - ptb_range[0])

        return perturb


    def get_embed(self, embed):
        assert self.embed_shape is not None

        embed = tf.Variable(embed, name='embed')

        return embed


    def get_weights(self):
        cell_weight = self.load_tensor(self.weight_path, 'cell_weight.npy')
        cell_bias   = self.load_tensor(self.weight_path, 'cell_bias.npy')
        fc_weight   = self.load_tensor(self.weight_path, 'fc_weight.npy')
        fc_bias     = self.load_tensor(self.weight_path, 'fc_bias.npy')

        return cell_weight, cell_bias, fc_weight, fc_bias


    def load_tensor(self, path, name):
        return tf.convert_to_tensor(np.load(os.path.join(path, name)))
