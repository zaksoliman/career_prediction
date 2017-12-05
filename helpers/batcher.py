import numpy as np
import math


class Batcher:
   
    def __init__(self, batch_size, step_num, data, n_classes, shuffle=False, shuffle_seed=123):

        if shuffle:
            np.random.seed(shuffle_seed)
            self.data = np.random.shuffle(self.data)
        else:
            self.data = data #sorted(self.data, key=lambda x: len(x), reverse=True)

        self.num_of_samples = len(self.data)
        self.batch_size = batch_size
        self.batch_num = 0
        self.max_batch_num = int(math.ceil(self.num_of_samples / self.batch_size))
        self.step_num = step_num
        self.n_classes = n_classes
        self.one_hot_lookup = np.eye(n_classes, dtype=np.int8)

    def next(self):
        batch_size = self.batch_size
        if self.batch_num == self.max_batch_num - 1: # i.e. at the last batch
            # We put the rest of the data in the last batch
            batch_size = self.num_of_samples - (self.batch_size * (self.max_batch_num - 1))
            
        seqs = np.zeros((batch_size, self.step_num), dtype=np.int32)
        targets = np.zeros((batch_size, self.step_num, self.n_classes), dtype=np.int32)
        seqs_length = np.zeros(batch_size, dtype=np.int32)

        for i in range(batch_size):
            example = self.data[self.batch_num * self.batch_size + i]
            seq = example[:-1]
            target = example[1:]
            seqs[i, :len(seq)] = seq
            seqs_length[i] = len(seq)
            targets[i, :len(target), :] = self.one_hot_lookup[target]

        if self.batch_num == self.max_batch_num - 1 or self.max_batch_num == 0:
            self.batch_num = 0
            if self.max_batch_num != 0:
                np.random.shuffle(self.data)
        else:
            self.batch_num += 1
            
        return seqs, seqs_length, targets

class BOW_Batcher:

    def __init__(self, batch_size, step_num, input_data, target_data, n_classes, vocab_size, concat=False, shuffle=False, shuffle_seed=123):
        print("Building batcher")
        self.input_data = input_data  # sorted(self.data, key=lambda x: len(x), reverse=True)
        self.target_data = target_data
        if len(self.input_data) != len(self.target_data):
            print("Data not same size!!!")
        self.vocab_size = vocab_size
        self.num_of_samples = len(self.input_data)
        self.batch_size = batch_size
        self.batch_num = 0
        self.max_batch_num = int(math.ceil(self.num_of_samples / self.batch_size))
        self.step_num = step_num
        self.n_classes = n_classes
        self.one_hot_lookup = np.eye(n_classes, dtype=np.int8)
        self.concat = concat

    def next(self):
        batch_size = self.batch_size
        if self.batch_num == self.max_batch_num - 1:  # i.e. at the last batch
            # We put the rest of the data in the last batch
            batch_size = self.num_of_samples - (self.batch_size * (self.max_batch_num - 1))

        input_seqs = np.zeros((batch_size, self.step_num, self.vocab_size), dtype=np.int32)
        targets = np.zeros((batch_size, self.step_num), dtype=np.int32)
        seqs_length = np.zeros(batch_size, dtype=np.int32)

        for i in range(batch_size):
            input_seq = self.input_data[self.batch_num * self.batch_size + i]
            target_seq = self.target_data[self.batch_num * self.batch_size + i]

            for j, bow in enumerate(input_seq):
                input_seqs[i][j][bow] = 1

            seqs_length[i] = len(input_seq)
            targets[i, :len(target_seq)] = target_seq

        if self.batch_num == self.max_batch_num - 1 or self.max_batch_num == 0:
            self.batch_num = 0
            if self.max_batch_num != 0:
                p = np.random.permutation(len(self.input_data))
                self.input_data = self.input_data[p]
                self.target_data = self.target_data[p]
        else:
            self.batch_num += 1

        return input_seqs, seqs_length, targets
