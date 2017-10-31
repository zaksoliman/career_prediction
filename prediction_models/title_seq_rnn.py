import tensorflow as tf
import os, random, string
from helpers.batcher import Batcher
import helpers.loader as loader
from time import time


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"


class Model:

    def __init__(self, data_path="/data/rali7/Tmp/solimanz/data/datasets/title_seq.json", use_dropout=False, n_titles=550,
                 keep_prob=0.5, hidden_dim=250, use_attention=False, attention_dim=100, use_embedding=False,
                 embedding_dim=50, rnn_cell_type='LSTM', max_timesteps=31, learning_rate=0.001, batch_size=50,
                 n_epochs=100, store_interval=200, restore=False, store_dir="/data/rali7/Tmp/solimanz/data/models/",
                 log_dir="/data/rali7/Tmp/solimanz/tf_logs/",):

        self.sotre_interval = store_interval
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
        self.data_path = data_path
        self.store_dir = store_dir
        self.log_dir = log_dir

    def predict(self):

        # Keep probability for the dropout
        self.keep_prob = tf.placeholder(tf.float32)
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
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

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

    def loss(self):
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logit, labels=self.target)
        self.loss = tf.reduce_mean(self.loss)

        optimizer = tf.train.AdamOptimizer(self.lr)  # GradientDescentOptimizer RMSPropOptimizer
        self.train_op = optimizer.minimize(self.loss)
        self.distribution = tf.nn.softmax(self.logit)
        self.correct_pred = tf.equal(tf.cast(tf.argmax(self.distribution, 1), tf.int32), self.target)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def train(self):

        title_to_id, train_data, test_data = loader.load_data(self.data_path)
        self.title_count = len(title_to_id) + 1

        print("Creating batchers")
        train_batcher = Batcher(self, train_data)
        test_batcher = Batcher(self, test_data)

        # Assume that you have 12GB of GPU memory and want to allocate ~4GB:
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True

        with tf.Session(config=gpu_config) as sess:

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            # TODO: give a unique path name to save models to
            path = os.path.join(".ckpt")
            if self.restore is not None and os.path.isfile(path):
                saver.restore(sess, path)
                print("model restored")

            for e in range(self.n_epochs):
                avg_cost = 0
                avg_acc = 0
                start_time = time()
                batch = 0
                for _ in range(train_batcher.max_batch_num):
                    with tf.device("/cpu:0"):
                        job_input_data, job_length, targets = train_batcher.next()

                    cost, _, acc = sess.run([self.loss, self.train_op, self.accuracy],
                                            {
                                                self.titles_input_data: job_input_data,
                                                self.seq_lengths: job_length,
                                                self.target: target,
                                                self.keep_prob: self.keep_prob
                                            })

                    # Compute average loss
                    avg_cost += cost
                    avg_acc += acc

                    if batch % self.sotre_interval == 0 and batch > 0:
                        cur_loss = avg_cost / batch  # config.log_interval
                        cur_acc = avg_acc / batch
                        elapsed = time() - start_time
                        print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:1.5f} | ms/batch {:5.2f} | '
                              'loss {:5.6f} | acc {:8.2f}'.format(
                            e, batch, self.max_train_batches / self.batch_size, self.lr,
                                      elapsed * 1000 / self.sotre_interval, cur_loss, cur_acc))

                        start_time = time()

                    batch += 1
                    if self.max_train_batches != -1 and batch == self.max_train_batches:
                        break
                print("Epoch:", '%04d' % (e + 1), "cost=", "{:.5f}".format(avg_cost/batch), "Accuracy=", "{:.2f}".format(avg_acc/batch))

                for _ in range(test_batcher.max_batch_num):
                    with tf.device("/cpu:0"):
                        job_input_data, job_length, target = test_batcher.next()

                    cost, correct_pred, acc = sess.run([self.loss, self.correct_pred, self.accuracy],
                                         {self.titles_input_data: job_input_data,
                                          self.seq_lengths: job_length,
                                          self.target: target,
                                          self.keep_prob: 1.0})

                if e % 10 == 0:
                    save_path = saver.save(sess, os.path.join(self.restore, self.encoder +
                                                              self.rnn_type +
                                                              str(self.hidden_dim) +
                                                              str(self.use_emb) +
                                                              str(self.use_attention) + ".ckpt"))
                    print("model saved in file: %s" % save_path)

                print('Cost on test:', cost, 'Accuracy on test:', acc)

def main():
    seq_model = Model()


if __name__ == "__main__":
    main()
