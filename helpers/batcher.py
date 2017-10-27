import numpy as np
import math
from random import shuffle


class Batcher:
   
    def __init__(self, config, data):
        self.data = data

        # shuffle(self.data)
        # self.data = sorted(self.data, key=lambda x: len(x), reverse=True)
        self.num_of_samples = len(self.data)
        self.batch_size = config.batch_size
        self.batch_num = 0
        self.max_batch_num = int( math.ceil(self.num_of_samples / self.batch_size))
        self.step_num = config.step_num
        
    def next(self):
        batch_size = self.batch_size
        if self.batch_num == self.max_batch_num-1:
            batch_size = self.num_of_samples - (self.batch_size * (self.max_batch_num-1)) 
            
        job_input = np.zeros((batch_size, self.step_num), dtype=np.int32)
        job_length = np.zeros(batch_size, dtype=np.int32)
        target = np.zeros(batch_size, dtype=np.int32)

        for i in range(batch_size):
            # TODO: Clean up (max length)
            example = self.data[self.batch_num * self.batch_size + i]
            inp = example[:-1]
            job_input[i, :len(inp)] = inp
            job_length[i] = len(inp)
            target[i] = example[-1]

        if self.batch_num == self.max_batch_num-1 or self.max_batch_num == 0:
            self.batch_num = 0
            if self.max_batch_num != 0:
                shuffle(self.data)
        else:
            self.batch_num += 1
            
        return job_input, job_length, target

