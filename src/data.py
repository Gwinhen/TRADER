import numpy as np
import os

class DataGenerator(object):
    def __init__(self, path, dataset='AppReviews', portion='train'):
        self.path    = path
        self.dataset = dataset
        self.portion = portion

        joined_path = os.path.join(self.path, self.dataset, self.portion)

        if self.dataset in ['AppReviews', 'JIRA', 'StackOverflow', 'Yelp']:
            self.seqs = np.load(joined_path + '_seqs.npy').astype(np.int32)
            self.lens = np.load(joined_path + '_lens.npy').astype(np.int32)
            self.gold = np.load(joined_path + '_gold.npy').astype(np.int32)
            if self.dataset == 'Yelp' and self.portion == 'tra':
                idx = np.load(joined_path + '_idx.npy')
                self.seqs = self.seqs[idx]
                self.lens = self.lens[idx]
                self.gold = self.gold[idx]
            self.size = len(self.seqs)
        elif self.dataset == 'IMDB':
            self.seqs_pos = np.load(joined_path + '_seqs_pos.npy').astype(np.int32)
            self.lens_pos = np.load(joined_path + '_lens_pos.npy').astype(np.int32)
            self.seqs_neg = np.load(joined_path + '_seqs_neg.npy').astype(np.int32)
            self.lens_neg = np.load(joined_path + '_lens_neg.npy').astype(np.int32)
            self.size = len(self.seqs_pos) + len(self.seqs_neg)


    def get_batch(self, batch_idx, batch_size=24):
        seqs = []
        lens = []
        gold = []
        for i in range(batch_size):
            if self.dataset == 'IMDB':
                idx = (batch_size * batch_idx + i) // 2
                idx = idx % (self.size // 2)

                if i % 2 == 0:
                    seqs.append(self.seqs_pos[idx])
                    lens.append(self.lens_pos[idx])
                    gold.append([1, 0])
                else:
                    seqs.append(self.seqs_neg[idx])
                    lens.append(self.lens_neg[idx])
                    gold.append([0, 1])
            else:
                idx = batch_size * batch_idx + i
                idx = idx % self.size

                seqs.append(self.seqs[idx])
                lens.append(self.lens[idx])

                if self.dataset == 'JIRA':
                    label = np.zeros([2])
                    label[(self.gold[idx] + 1) // 2] = 1
                    gold.append(label)
                elif self.dataset == 'Yelp':
                    label = np.zeros([2])
                    gidx = 0 if self.gold[idx] == -1 else 1
                    label[gidx] = 1
                    gold.append(label)
                else:
                    label = np.zeros([3])
                    label[self.gold[idx]+1] = 1
                    gold.append(label)

        seqs = np.array(seqs)
        lens = np.array(lens)
        gold = np.array(gold)

        return seqs, lens, gold


    def shuffle(self):
        indices = np.random.permutation(self.size)
        if self.dataset in ['AppReviews', 'JIRA', 'StackOverflow', 'Yelp']:
            self.seqs = self.seqs[indices]
            self.lens = self.lens[indices]
            self.gold = self.gold[indices]
        elif self.dataset == 'IMDB':
            self.seqs_pos = self.seqs_pos[indices]
            self.lens_pos = self.lens_pos[indices]
            self.seqs_neg = self.seqs_neg[indices]
            self.lens_neg = self.lens_neg[indices]
