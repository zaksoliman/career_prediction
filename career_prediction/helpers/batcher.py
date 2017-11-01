import numpy as np
import math


class Batcher:
   
    def __init__(self, batch_size, step_num, data, shuffle=False, shuffle_seed=123):

        if shuffle:
            np.random.seed(shuffle_seed)
            self.data = np.random.shuffle(self.data)
        else:
            self.data = data #sorted(self.data, key=lambda x: len(x), reverse=True)

        self.num_of_samples = len(self.data)
        self.batch_size = batch_size
        self.batch_num = 0
        self.max_batch_num = int(math.floor(self.num_of_samples / self.batch_size))
        self.step_num = step_num
        
    def next(self):

        batch_size = self.batch_size
        if self.batch_num == self.max_batch_num:
            # We put the rest of the data in the last batch
            batch_size = self.num_of_samples - (self.batch_size * (self.max_batch_num))
            
        seqs = np.zeros((batch_size, self.step_num), dtype=np.int32)
        targets = np.zeros((batch_size, self.step_num), dtype=np.int32)
        seqs_length = np.zeros(batch_size, dtype=np.int32)

        for i in range(batch_size):
            example = self.data[self.batch_num * self.batch_size + i]
            seq = example[:-1]
            target = example[1:]
            seqs[i, :len(seq)] = seq
            seqs_length[i] = len(seq)
            targets[i, :len(target)] = target

        if self.batch_num == self.max_batch_num or self.max_batch_num == 0:
            self.batch_num = 0
            if self.max_batch_num != 0:
                np.random.shuffle(self.data)
        else:
            self.batch_num += 1
            
        return seqs, seqs_length, targets

