import tensorflow as tf
import os, random, string
from helpers.batcher import Batcher
import helpers.loader as loader
from time import time


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"


class Model:

    def __init__(self, data, target, class_mapping, use_dropout=False, n_titles=550,
                 keep_prob=0.5, hidden_dim=250, use_attention=False, attention_dim=100, use_embedding=False,
                 embedding_dim=50, rnn_cell_type='GRU', max_timesteps=32, learning_rate=0.001, batch_size=50,
                 n_epochs=100, log_interval=200, restore=False, store_dir="/data/rali7/Tmp/solimanz/data/models/",
                 log_dir="../.log/",):

        self.log_interval = log_interval
        self.titles_to_id = class_mapping
        self.restore = restore
        self.keep_prob = keep_prob
        self.use_dropout = use_dropout
        self.n_titles = n_titles
        self.n_epochs = n_epochs
        self.use_embedding= use_embedding
        self.rnn_cell_type = rnn_cell_type
        self.emb_dim = embedding_dim
        self.max_timesteps = max_timesteps
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.use_att = use_attention
        self.att_dim = attention_dim
        self.data = data
        self.target = target
        self.store_dir = store_dir
        self.log_dir = log_dir

    def predict(self):

        # Keep probability for the dropout
        self.dropout = tf.placeholder(tf.float32)
        # Our list of job title sequences (padded to max_timesteps)
        self.titles_input_data = tf.placeholder(tf.int32, [None, self.max_timesteps], name="titles_input_data")
        # matrix that will hold the length of out sequences
        self.seq_lengths = tf.placeholder(tf.int32, [None], name="seq_lengths")
        self.target = tf.placeholder(tf.int32, [None, self.max_timesteps], name="labels")

        # Do embedding
        with tf.device("/cpu:0"):
            if self.use_embedding:
                title_embedding = tf.get_variable(name="job_embedding",
                                                  shape=[self.n_titles, self.emb_dim],
                                                  dtype=tf.float32,
                                                  initializer=tf.contrib.layers.xavier_initializer(),
                                                  trainable=True)
            else:
                title_embedding = tf.Variable(tf.eye(self.n_titles), trainable=False)


            self.title_emb_input = tf.nn.embedding_lookup(title_embedding, self.titles_input_data)

        # Decide on out RNN cell type
        if self.rnn_cell_type == 'GRU':
            cell = tf.nn.rnn_cell.GRUCell(self.hidden_dim)
        elif self.rnn_cell_type == 'RNN':
            cell = tf.nn.rnn_cell.BasicRNNCell(self.hidden_dim)
        elif self.rnn_cell_type == 'LSTM':
            cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim)

        # Adding dropout
        if self.use_dropout:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout)

        self.output, self.prev_states = tf.nn.dynamic_rnn(cell,
                                                    self.title_emb_input,
                                                    sequence_length=self.seq_lengths,
                                                    dtype=tf.float32,
                                                    parallel_iterations=1024)
        self.logit = tf.layers.dense(self.output,
                                     self.n_titles,
                                     activation=None,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.contrib.layers.xavier_initializer())
        self.prediction = tf.nn.softmax(self.logit)

        return self.prediction

    def cost(self, output, target):
        # Compute cross entropy for each frame.
        cross_entropy = target * tf.log(output)
        cross_entropy = -tf.reduce_sum(cross_entropy, 2)
        mask = tf.sign(tf.reduce_max(tf.abs(target), 2))
        cross_entropy *= mask
        # Average over actual sequence lengths.
        cross_entropy = tf.reduce_sum(cross_entropy, 1)
        cross_entropy /= tf.reduce_sum(mask, 1)
        return tf.reduce_mean(cross_entropy)

    def loss(self):
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logit, labels=self.target)
        self.loss = tf.reduce_mean(self.loss)

        optimizer = tf.train.AdamOptimizer(self.lr)  # GradientDescentOptimizer RMSPropOptimizer
        self.train_op = optimizer.minimize(self.loss)
        self.distribution = tf.nn.softmax(self.logit)
        self.correct_pred = tf.equal(tf.cast(tf.argmax(self.distribution, 1), tf.int32), self.target)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        return self.loss

    def train(self):

        train_data, test_data = self.data["train_data"], self.data["test_data"]

        print("Creating batchers")
        train_batcher = Batcher(self, train_data)
        test_batcher = Batcher(self, test_data)

        # Assume that you have 12GB of GPU memory and want to allocate ~4GB:
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True

        with tf.Session(config=gpu_config) as sess:

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            path = os.path.join(self.store_dir,
                                f"title_seq_rnn_{self.hidden_dim}_{self.keep_prob}_{self.use_embedding}_{self.emb_dim}.ckpt")
            if self.restore is not None and os.path.isfile(path):
                saver.restore(sess, path)
                print("model restored")

            for e in range(self.n_epochs):
                start_time = time()
                batch = 0
                avg_loss = 0
                avg_acc = 0
                for _ in range(train_batcher.max_batch_num):
                    with tf.device("/cpu:0"):
                        title_seq, seq_lengths, target = train_batcher.next()

                    loss, _, acc = sess.run([self.loss, self.train_op, self.accuracy],
                                            {
                                                self.titles_input_data: title_seq,
                                                self.seq_lengths: seq_lengths,
                                                self.target: target,
                                                self.dropout: self.keep_prob
                                            })

                    if batch % self.log_interval == 0 and batch > 0:
                        elapsed = time() - start_time
                        print(
                            f'| epoch {e} | {batch}/{self.batch_size} batches | lr {self.lr} | '
                            f'ms/batch {elapsed * 1000 / self.log_interval} | loss {loss} | acc {acc}')

                        start_time = time()

                    batch += 1

                print(f"Epoch:, {(e + 1)}")

                for _ in range(test_batcher.max_batch_num):
                    with tf.device("/cpu:0"):
                        job_input_data, job_length, target = test_batcher.next()

                    cost, correct_pred, acc = sess.run([self.loss, self.correct_pred, self.accuracy],
                                                       {self.titles_input_data: job_input_data,
                                                        self.seq_lengths: job_length,
                                                        self.target: target,
                                                        self.dropout: 1.0})

                if e % 10 == 0:
                    save_path = saver.save(sess, path)
                    print("model saved in file: %s" % save_path)

                print('Cost on test: {cost} Accuracy on test: {acc}')

def main():
    seq_model = Model()


if __name__ == "__main__":
    main()
