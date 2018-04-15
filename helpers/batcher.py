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
            example = self.data[self.batch_num * self.batch_size + i][1]
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

    def __init__(self, batch_size, step_num, input_data, target_data, n_classes, vocab_size, shuffle=False, shuffle_seed=123):
        print("Building batcher")
        self.input_data = input_data  # sorted(self.data, key=lambda x: len(x), reverse=True)
        self.target_data = target_data
        if len(self.input_data) != len(self.target_data):
            print("Data not same size!!!")
            print(f"len(input_data)={len(self.input_data)}\nlen(target_data)={len(self.target_data)}")
        self.vocab_size = vocab_size
        self.num_of_samples = len(self.input_data)
        self.batch_size = batch_size
        self.batch_num = 0
        self.max_batch_num = int(math.ceil(self.num_of_samples / self.batch_size))
        self.step_num = step_num
        self.n_classes = n_classes
        self.one_hot_lookup = np.eye(n_classes, dtype=np.int8)

    def next(self):
        batch_size = self.batch_size
        if self.batch_num == self.max_batch_num - 1:  # i.e. at the last batch
            # We put the rest of the data in the last batch
            batch_size = self.num_of_samples - (self.batch_size * (self.max_batch_num - 1))

        input_seqs = np.zeros((batch_size, self.step_num, self.vocab_size), dtype=np.int32)
        targets = np.zeros((batch_size, self.step_num, self.n_classes), dtype=np.int32)
        seqs_length = np.zeros(batch_size, dtype=np.int32)

        for i in range(batch_size):
            input_seq = self.input_data[self.batch_num * self.batch_size + i][1]
            target_seq = self.target_data[self.batch_num * self.batch_size + i][1]

            for j, bow in enumerate(input_seq):
                for word in bow:
                    input_seqs[i][j][word] += 1

            seqs_length[i] = len(input_seq)
            targets[i, :len(target_seq), :] = self.one_hot_lookup[target_seq]

        if self.batch_num == self.max_batch_num - 1 or self.max_batch_num == 0:
            self.batch_num = 0
            if self.max_batch_num != 0:
                zipped = list(zip(self.input_data, self.target_data))
                np.random.shuffle(zipped)
                self.input_data, self.target_data = zip(*zipped)
        else:
            self.batch_num += 1

        return input_seqs, seqs_length, targets

class MultiLabelBatcher:

    def __init__(self, batch_size, step_num, input_data, target_data, n_classes, shuffle_seed=123):
        print("Building batcher")
        self.input_data = input_data
        self.target_data = target_data
        if len(self.input_data) != len(self.target_data):
            print("Data not same size!!!")
        self.num_of_samples = len(self.input_data)
        self.batch_size = batch_size
        self.batch_num = 0
        self.max_batch_num = int(math.ceil(self.num_of_samples / self.batch_size))
        self.step_num = step_num
        self.n_classes = n_classes
        np.random.seed(shuffle_seed)

    def next(self):
        batch_size = self.batch_size
        if self.batch_num == self.max_batch_num - 1:  # i.e. at the last batch
            # We put the rest of the data in the last batch
            batch_size = self.num_of_samples - (self.batch_size * (self.max_batch_num - 1))

        input_seqs = np.zeros((batch_size, self.step_num), dtype=np.int32)
        targets = np.zeros((batch_size, self.n_classes), dtype=np.int32)
        seqs_length = np.zeros(batch_size, dtype=np.int32)

        for i in range(batch_size):
            input_seq = self.input_data[self.batch_num * self.batch_size + i]
            target_labels = self.target_data[self.batch_num * self.batch_size + i]

            input_seqs[i, :len(input_seq)] = input_seq
            targets[i, target_labels] = 1
            seqs_length[i] = len(input_seq)

        if self.batch_num == self.max_batch_num - 1 or self.max_batch_num == 0:
            self.batch_num = 0
            if self.max_batch_num != 0:
                zipped = list(zip(self.input_data, self.target_data))
                np.random.shuffle(zipped)
                self.input_data, self.target_data = zip(*zipped)
        else:
            self.batch_num += 1

        return input_seqs, seqs_length, targets

class SequenceBatcher:
    def __init__(self, batch_size, step_num, data, n_classes, shuffle_seed=123):

        np.random.seed(shuffle_seed)
        self.data = data
        self.num_of_samples = len(self.data)
        self.batch_size = batch_size
        self.batch_num = 0
        self.max_batch_num = int(math.ceil(self.num_of_samples / self.batch_size))
        self.step_num = step_num
        self.n_classes = n_classes
        self.one_hot_lookup = np.eye(n_classes, dtype=np.int16)

    def next(self):
        batch_size = self.batch_size
        if self.batch_num == self.max_batch_num - 1:  # i.e. at the last batch
            # We put the rest of the data in the last batch
            batch_size = self.num_of_samples - (self.batch_size * (self.max_batch_num - 1))

        input_seqs = np.zeros((batch_size, self.step_num), dtype=np.int32)
        targets = np.zeros((batch_size, self.n_classes), dtype=np.int32)
        seqs_length = np.zeros(batch_size, dtype=np.int32)

        for i in range(batch_size):
            example = self.data[self.batch_num * self.batch_size + i]
            sequence = example[:-1]
            target_label = example[-1]
            input_seqs[i, :len(sequence)] = sequence
            seqs_length[i] = len(sequence)
            targets[i,:] = self.one_hot_lookup[target_label]

        if self.batch_num == self.max_batch_num - 1 or self.max_batch_num == 0:
            self.batch_num = 0
            if self.max_batch_num != 0:
                np.random.shuffle(self.data)
        else:
            self.batch_num += 1

        return input_seqs, seqs_length, targets

class FeedForwardBatcher:

    def __init__(self):
        pass

    def next(self):
        pass