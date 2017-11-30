import tensorflow as tf
import os, random, string
from helpers.loader import load_data
from helpers.batcher import Batcher
from time import time
#from helpers.data_viz import print_dists
import numpy as np
import random
from pprint import pprint


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"


class Model:

    def __init__(self, train_data, test_data=None, class_mapping=None, use_dropout=True, n_titles=550, num_layers=1,
                 keep_prob=0.5, hidden_dim=250, use_attention=False, attention_dim=100, use_embedding=True,
                 embedding_dim=100, rnn_cell_type='LSTM', max_timesteps=31, learning_rate=0.001, batch_size=100,
                 n_epochs=800, log_interval=200, store_model=True, restore=True, store_dir="/data/rali7/Tmp/solimanz/data/models/",
                 log_dir=".log/",):

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
        self.train_data = train_data
        self.test_data = test_data
        self.store_dir = store_dir
        self.log_dir = log_dir
        self.store = store_model
        self.num_layers = num_layers
        self.hparams = f"title_seq_{rnn_cell_type}_{num_layers}_layers_cell_lr_{learning_rate}_use_emb={use_embedding}_emb_dim={embedding_dim}" \
                       f"hdim={hidden_dim}_dropout={keep_prob}_data_size={len(self.train_data)}"

        self.build_model()

    def build_model(self):
        self._predict()
        self._loss()
        self._accuracy()
        self._optimize()
        self.summ = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(os.path.join(self.log_dir, self.hparams))

    def _predict(self):
        """
        Build the inference graph

        :return:
        """
        # Keep probability for the dropout
        self.dropout = tf.placeholder(tf.float32, name="dropout_prob")
        # Our list of job title sequences (padded to max_timesteps)
        self.titles_input_data = tf.placeholder(tf.int32, [None, self.max_timesteps], name="titles_input_data")
        # matrix that will hold the length of out sequences
        self.seq_lengths = tf.placeholder(tf.int32, [None], name="seq_lengths")
        self.target = tf.placeholder(tf.int32, [None, self.max_timesteps, self.n_titles], name="labels")
        tf.summary.histogram("targets", self.target)
        # Do embedding
        with tf.device("/cpu:0"):
            if self.use_embedding:
                title_embedding = tf.get_variable(name="title_embeddings",
                                                  shape=[self.n_titles, self.emb_dim],
                                                  dtype=tf.float32,
                                                  initializer=tf.contrib.layers.xavier_initializer(),
                                                  trainable=True)
            else:
                title_embedding = tf.Variable(tf.eye(self.n_titles), trainable=False, name="title_one_hot_encoding")

            # tile_emb_input has shape batch_size x times steps x emb_dim
            self.title_emb_input = tf.nn.embedding_lookup(title_embedding, self.titles_input_data, name="encoded_in_seq")

        # Decide on out RNN cell type
        if self.rnn_cell_type == 'RNN':
            get_cell = tf.nn.rnn_cell.BasicRNNCell
        elif self.rnn_cell_type == 'LSTM':
            get_cell = tf.nn.rnn_cell.LSTMCell
        else: # Default to GRU
            get_cell = tf.nn.rnn_cell.GRUCell

        cells = [get_cell(self.hidden_dim) for _ in range(self.num_layers)]
        if self.use_dropout:
            cells = [tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout) for cell in cells]

        cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

        self.output, self.prev_states = tf.nn.dynamic_rnn(cell,
                                                    self.title_emb_input,
                                                    sequence_length=self.seq_lengths,
                                                    dtype=tf.float32,
                                                    parallel_iterations=1024)

        tf.summary.histogram("RNN_out", self.output)
        output = tf.reshape(self.output, [-1, self.hidden_dim])
        tf.summary.histogram("reshaped_output", output)


        self.logit = tf.layers.dense(output,
                                     self.n_titles,
                                     activation=None,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.contrib.layers.xavier_initializer(), name="fc_logit")
        tf.summary.histogram("Logits", self.logit)

        prediction_softmax = tf.nn.softmax(self.logit, name="prediction")
        tf.summary.histogram("logit_softmax", prediction_softmax)
        self.prediction = tf.reshape(prediction_softmax, [-1, self.max_timesteps, self.n_titles])
        tf.summary.histogram("prediction", self.prediction)

        return self.prediction

    def _loss(self):
        with tf.name_scope("xent"):
            #cross_entropy = -tf.reduce_sum(self.target_one_hot * tf.log(self.prediction), [1, 2])
            #self.cross_entropy = tf.reduce_mean(cross_entropy)

            # Compute cross entropy for each frame.
            cross_entropy = tf.cast(self.target, tf.float32) * tf.log(self.prediction)
            cross_entropy = -tf.reduce_sum(cross_entropy, 2)
            mask = tf.sign(tf.reduce_max(tf.abs(self.target), 2))
            cross_entropy *= tf.cast(mask, tf.float32)
            # Average over actual sequence lengths.
            cross_entropy = tf.reduce_sum(cross_entropy, 1)
            cross_entropy /= tf.reduce_sum(tf.cast(mask, tf.float32), 1)
            self.cross_entropy = tf.reduce_mean(cross_entropy)
            tf.summary.scalar("xent", self.cross_entropy)
            return self.cross_entropy

    def _optimize(self):
        with tf.name_scope("train"):
            self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.cross_entropy)
            return self.optimize

    def _accuracy(self):
        with tf.name_scope("accuracy"):
            correct = tf.equal(
                tf.argmax(self.target, 2, output_type=tf.int32), tf.argmax(self.prediction, 2, output_type=tf.int32))
            correct = tf.cast(correct, tf.float32)
            mask = tf.sign(tf.reduce_max(tf.abs(self.target), reduction_indices=2))
            correct *= tf.cast(mask, tf.float32)
            # Average over actual sequence lengths.
            correct = tf.reduce_sum(correct, reduction_indices=1)
            correct /= tf.cast(self.seq_lengths, tf.float32)
            self.accuracy =  tf.reduce_mean(correct)
            tf.summary.scalar("training_accuracy", self.accuracy)
            self.test_summary = tf.summary.scalar("test_accuracy", self.accuracy)
            return self.accuracy

    def train(self):

        print("Creating batchers")
        train_batcher = Batcher(batch_size=self.batch_size, step_num=self.max_timesteps, data=self.train_data)
        test_batcher = Batcher(batch_size=self.batch_size, step_num=self.max_timesteps,  data=self.test_data)

        # Assume that you have 12GB of GPU memory and want to allocate ~4GB:
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True

        with tf.Session(config=gpu_config) as sess:

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            path = os.path.join(self.store_dir, f"{self.hparams}.ckpt")
            if self.restore and os.path.isfile(path):
                saver.restore(sess, path)
                print("**model restored")

            self.writer.add_graph(sess.graph)

            for e in range(self.n_epochs):
                start_time = time()
                batch = 0
                for b in range(train_batcher.max_batch_num):

                    with tf.device("/cpu:0"):
                        title_seq, seq_lengths, target = train_batcher.next()

                    loss, _, acc, summary = sess.run([self.cross_entropy, self.optimize, self.accuracy, self.summ],
                                            {
                                                self.titles_input_data: title_seq,
                                                self.seq_lengths: seq_lengths,
                                                self.target: target,
                                                self.dropout: self.keep_prob
                                            })

                    tf.summary.scalar("accs", acc)
                    self.writer.add_summary(summary, b)

                    if batch % self.log_interval == 0 and batch > 0:
                        elapsed = time() - start_time
                        print(
                            f'| epoch {e} | {train_batcher.batch_num}/{train_batcher.max_batch_num} batches | lr {self.lr} | '
                            f'ms/batch {elapsed * 1000 / self.log_interval} | loss {loss} | acc {acc}')

                        start_time = time()

                    batch += 1

                print(f"Epoch:, {(e + 1)}")

                avg_acc = []
                for tb in range(test_batcher.max_batch_num):
                    with tf.device("/cpu:0"):
                        test_title_seq, test_seq_lengths, test_target = train_batcher.next()

                    test_acc, test_summ, pred = sess.run([self.accuracy, self.test_summary, self.prediction],
                                                   {
                                                       self.titles_input_data: test_title_seq,
                                                       self.seq_lengths: test_seq_lengths,
                                                       self.target: test_target,
                                                       self.dropout: 1.0
                                                   })
                    if test_acc > 0:
                        avg_acc.append(test_acc)

                    #print_dists(self.titles_to_id, test_seq_lengths, test_title_seq, pred, test_target, f_name=self.hparams)
                    self.writer.add_summary(test_summ, tb)

                print(f"Accuracy on test: {sum(avg_acc) / len(avg_acc)}")
                if self.store and e % 10 == 0:
                    save_path = saver.save(sess, path)
                    print("model saved in file: %s" % save_path)


def main():
    path = "/data/rali7/Tmp/solimanz/data/datasets/1/title_sequences.json"
    mapping, train_data, test_data = load_data(path)
    seq_model = Model(train_data=train_data, num_layers=2)
    seq_model.train()


if __name__ == "__main__":
    main()
