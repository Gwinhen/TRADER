import argparse
import numpy as np
import os
import sys
import tensorflow as tf
import time

from data  import DataGenerator
from vanilla_rnn import VanillaRNN, VanillaRNNDual
from lstm import LSTM, LSTMDual
from gru import GRU, GRUDual
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train(model, iters, batch_size, data_dir, dataset,
          ckpt_path, save_iter, saver, sess):
    data = DataGenerator(data_dir, dataset=dataset, portion='tra')

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    train_op  = optimizer.minimize(model.loss)
    sess.run(tf.global_variables_initializer())

    for i in range(iters):
        if i % (data.size / batch_size) == 0:
            data.shuffle()

        seqs, lens, gold = data.get_batch(i)
        loss, acc, _ = sess.run([model.loss, model.accuracy, train_op],
                                {model.x_ph: seqs,
                                 model.y_ph: gold,
                                 model.len_ph: lens})

        if ((i + 1) % 100 == 0):
            print('| Iter {:3}: loss {:.4f}, acc {:.4f} |'.format(i+1, loss, acc))

        if ((i + 1) == save_iter):
            saver.save(sess, ckpt_path, write_meta_graph=False)
            print('| Saved to {}'.format(ckpt_path))
            break


def test(model, batch_size, data_dir, dataset, sess):
    data = DataGenerator(data_dir, dataset=dataset, portion='test')

    iters = int(np.ceil(1.0 * data.size / batch_size))
    preds = 0

    for i in range(iters):
        seqs, lens, gold = data.get_batch(i)
        pred = sess.run(model.correct, {model.x_ph: seqs,
                                        model.y_ph: gold,
                                        model.len_ph: lens})

        if (i+1) * batch_size < data.size:
            preds += len(pred[pred == True])
        else:
            remain = data.size - i * batch_size
            pred = pred[:remain]
            preds += len(pred[pred == True])

        if i % 100 == 0:
            sys.stdout.write('\r{}/{}'.format(i, iters))
            sys.stdout.flush()

    accuracy = 100. * preds / data.size
    print('\rTest accuracy: {:.2f}'.format(accuracy))


def trace(model, model_name, model_size, batch_size, data_dir, dataset,
          embed_name, embed, sess):
    data = DataGenerator(data_dir, dataset=dataset, portion='val')

    iters = int(np.ceil(1.0 * data.size / batch_size))
    succ = []
    fail = []
    ins  = []
    outs = []

    for i in range(iters):
        seqs, lens, gold = data.get_batch(i, batch_size=batch_size)
        if model_name == 'lstm':
            pred, cell_state, history, fc_w, fc_b\
                = sess.run([model.correct, model.cell_state, model.history,
                            model.fc_weight, model.fc_bias],
                            {model.x_ph: seqs, model.y_ph: gold,
                             model.len_ph: lens})
        else:
            pred, history, fc_w, fc_b\
                = sess.run([model.correct, model.history,
                            model.fc_weight, model.fc_bias],
                            {model.x_ph: seqs, model.y_ph: gold,
                             model.len_ph: lens})

        for j in range(batch_size):
            idx = j + i * batch_size
            if idx >= data.size:
                break

            if pred[j]:
                succ.append(idx)
            else:
                fail.append(idx)

            step_outs = np.add(np.matmul(history[j], fc_w), fc_b)

            _ins  = []
            _outs = []
            for t in range(lens[j] + 1):
                if model_name == 'lstm':
                    c_t = cell_state[j, t]
                h_t = history[j, t]

                if t < lens[j] or lens[j] == 0:
                    x_t = embed[seqs[j, t]]
                    if model_name == 'lstm':
                        step_in = np.concatenate([x_t, c_t, h_t])
                    else:
                        step_in = np.concatenate([x_t, h_t])
                    _ins.append(step_in)
                if t > 0 or lens[j] == 0:
                    _outs.append(step_outs[t])

            ins.append(np.array(_ins))
            outs.append(np.array(_outs))

        if dataset == 'Yelp' and i >= 99:
            break

    succ_ins  = np.concatenate(np.take(np.array(ins),  succ, axis=0))
    succ_outs = np.concatenate(np.take(np.array(outs), succ, axis=0))
    fail_ins  = np.concatenate(np.take(np.array(ins),  fail, axis=0))
    fail_outs = np.concatenate(np.take(np.array(outs), fail, axis=0))

    prefix = '{}/{}/traces/val_{}_{}_{}'.format(
              data_dir, dataset, model_name, model_size, embed_name)
    np.save('{}_succ_ins'.format(prefix),  succ_ins)
    np.save('{}_succ_outs'.format(prefix), succ_outs)
    np.save('{}_fail_ins'.format(prefix),  fail_ins)
    np.save('{}_fail_outs'.format(prefix), fail_outs)

    size = 0
    for pred in ['succ', 'fail']:
        for step in ['ins', 'outs']:
            path = '{}_{}_{}.npy'.format(prefix, pred, step)
            size += os.path.getsize(path)

    return (size / 1024. / 1024.)


def divergence(model_name, model_size, data_dir, dataset, embed_name):
    prefix = '{}/{}/traces/val_{}_{}_{}'.format(
              data_dir, dataset, model_name, model_size, embed_name)

    time_usage = 0
    scores = []
    machines = []
    for pred in ['succ', 'fail']:
        ins  = np.load('{}_{}_ins.npy'.format(prefix,  pred))
        outs = np.load('{}_{}_outs.npy'.format(prefix, pred))

        time_start = time.time()

        regressor = LinearRegression()

        regressor.fit(ins, outs)

        joblib.dump(regressor, '{}_{}_regressor.pkl'.format(prefix, pred))
        machines.append(regressor)

        score = regressor.score(ins, outs)
        scores.append(score)

        if pred == 'fail':
            out_oracle = np.argmax(machines[0].predict(ins), axis=1)
            out_buggy  = np.argmax(machines[1].predict(ins), axis=1)

            diff = out_oracle - out_buggy
            diff = np.where(diff != 0)[0]

            weighted_div = np.zeros(ins.shape[1])
            for i in diff:
                weighted_oracle = machines[0].coef_[out_oracle[i]] * ins[i]
                weighted_buggy  = machines[1].coef_[out_buggy[i]]  * ins[i]
                weighted_div += abs(weighted_oracle - weighted_buggy)

            np.save('{}_diff'.format(prefix), weighted_div)

        time_end = time.time()
        time_usage += time_end - time_start

    print('='*80)
    print('Fitting score for oracle machine:\t{:.3f}'.format(scores[0]))
    print('Fitting score for buggy machine:\t{:.3f}'.format(scores[1]))

    return time_usage


def regulate(model, iters, batch_size, data_dir, dataset,
             embed_path, save_iter, saver, sess):
    data = DataGenerator(data_dir, dataset=dataset, portion='tra')

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    train_op  = optimizer.minimize(model.loss, var_list=[model.embed])
    var_list  = [v.name for v in tf.trainable_variables()]
    sess.run(tf.global_variables_initializer())

    for i in range(iters):
        if i % (data.size / batch_size) == 0:
            data.shuffle()

        seqs, lens, gold = data.get_batch(i)
        loss, acc, w, _ = sess.run([model.loss, model.accuracy, var_list, train_op],
                                {model.x_ph: seqs,
                                 model.y_ph: gold,
                                 model.len_ph: lens})

        if ((i + 1) % 100 == 0):
            print('| Iter {:3}: loss {:.4f}, acc {:.4f} |'.format(i+1, loss, acc))

        if ((i + 1) == save_iter):
            weight_dict = {}
            for key, value in zip(var_list, w):
                weight_dict[key] = value

            embed = weight_dict['embed:0']
            np.save(embed_path, embed)


def main(args):
    if args.phase == 'diverge':
        divergence(args.model_name, args.model_size, args.data_dir,
                   args.dataset, args.embed_name)
    else:
        args.batch_size = 512 if args.dataset == 'Yelp' else args.batch_size
        classes = {'AppReviews': 3, 'IMDB': 2, 'JIRA': 2,
                   'StackOverflow': 3, 'Yelp': 2}
        max_len = {'AppReviews': 231, 'IMDB': 250, 'JIRA': 49,
                   'StackOverflow': 52, 'Yelp': 250}
        num_classes = classes[args.dataset]
        timesteps = max_len[args.dataset]

        sess = tf.Session()

        if args.method == 'trader':
            embed_subfix = '{}_{}_{}'.format(args.model_name, args.model_size,
                                             args.embed_name)
        else:
            embed_subfix = args.embed_name

        embed_path = os.path.join(args.data_dir, args.dataset,
                                  'embed_{}.npy'.format(embed_subfix))
        embed = np.load(embed_path)

        if args.phase in ['train', 'test', 'trace']:
            reg = True if args.method == 'rs' else False

            if args.model_name == 'rnn':
                model = VanillaRNN(batch_size=args.batch_size,
                                   hidden_units=args.model_size, timesteps=timesteps,
                                   num_classes=num_classes, embed=embed, reg=reg)
            elif args.model_name == 'lstm':
                model = LSTM(batch_size=args.batch_size,
                             hidden_units=args.model_size, timesteps=timesteps,
                             num_classes=num_classes, embed=embed, reg=reg)
            elif args.model_name == 'gru':
                model = GRU(batch_size=args.batch_size,
                            hidden_units=args.model_size, timesteps=timesteps,
                            num_classes=num_classes, embed=embed)
            else:
                print('ERROR: Unsupported model!!!')

            if args.method == 'ori':
                subfix = '_original'
            elif args.method == 'rs':
                subfix = '_rs'
            elif args.method == 'trader':
                subfix = '_trader'
            else:
                print('ERROR: Unsupported method!!!')

            if args.phase == 'trace':
                subfix = '_tra'

            ckpt_path = os.path.join(args.ckpt_dir, '{}_{}_{}_{}{}.ckpt'.format(
                                     args.dataset, args.model_name, args.model_size,
                                     args.embed_name, subfix))

            saver = tf.train.Saver(max_to_keep=1)

            if args.phase == 'train':
                train(model, args.iters, args.batch_size, args.data_dir,
                      args.dataset, ckpt_path, 500, saver, sess)
            elif args.phase == 'test':
                print('='*80)
                print('Testing {}'.format(ckpt_path))
                saver.restore(sess, ckpt_path)
                test(model, args.batch_size, args.data_dir, args.dataset, sess)
                print('='*80)
            elif args.phase == 'trace':
                saver.restore(sess, ckpt_path)
                space = trace(model, args.model_name, args.model_size,
                              args.batch_size, args.data_dir, args.dataset,
                              args.embed_name, embed, sess)
                time_usage = divergence(args.model_name, args.model_size,
                                        args.data_dir, args.dataset, args.embed_name)
                print('Time overhead:\t{:.3f} (s)'.format(time_usage))
                print('Space overhead:\t{:.3f} (M)'.format(space))
                print('='*80)

        if args.phase == 'regulate':
            weight_path = os.path.join(args.ckpt_dir, 'weights/{}/{}/{}/{}'.format(
                                       args.dataset, args.model_name,
                                       args.model_size, args.embed_name))
            diff_path   = os.path.join(args.data_dir, args.dataset,
                                       'traces/val_{}_{}_{}_diff.npy'.format(
                                       args.model_name, args.model_size,
                                       args.embed_name))
            save_path   = os.path.join(args.data_dir, args.dataset,
                                       'embed_{}_{}_{}.npy'.format(args.model_name,
                                       args.model_size, args.embed_name))
            params = [0.1, 0.1, 1e-4]
            if args.model_name == 'rnn':
                model = VanillaRNNDual(weight_path, hidden_units=args.model_size,
                                       timesteps=timesteps, num_classes=num_classes,
                                       embed=embed, diff_path=diff_path, params=params)
            elif args.model_name == 'lstm':
                model = LSTMDual(weight_path, hidden_units=args.model_size,
                                 timesteps=timesteps, num_classes=num_classes,
                                 embed=embed, diff_path=diff_path, params=params)
            elif args.model_name == 'gru':
                model = GRUDual(weight_path, hidden_units=args.model_size,
                                timesteps=timesteps, num_classes=num_classes,
                                embed=embed, diff_path=diff_path, params=params)
            else:
                print('ERROR: Unsupported model!!!')

            saver = tf.train.Saver(max_to_keep=1)
            regulate(model, args.iters, args.batch_size, args.data_dir,
                     args.dataset, save_path, 500, saver, sess)

        sess.close()


if __name__ == '__main__':
    seed = 1024
    np.random.seed(seed)
    tf.set_random_seed(seed)

    parser = argparse.ArgumentParser(description='Process input arguments.')

    parser.add_argument('--phase',      default='test', help='phase of framework')
    parser.add_argument('--method',     default='ori', help='training method')
    parser.add_argument('--iters',  default=500, type=int, help='training iterations')
    parser.add_argument('--batch_size', default=24, type=int, help='batch size')
    parser.add_argument('--data_dir',   default='../data', help='data directory')
    parser.add_argument('--ckpt_dir',   default='../models', help='model directory')
    parser.add_argument('--dataset',    default='AppReviews', help='dataset name')
    parser.add_argument('--model_name', default='rnn', help='type of RNN model')
    parser.add_argument('--model_size', default=64, type=int,
                        help='number of hidden units')
    parser.add_argument('--embed_name', default='glove', help='type of embedding')

    args = parser.parse_args()

    main(args)
