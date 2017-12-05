import tensorflow as tf
import os, random, string
from helpers.loader import load_data
from helpers.batcher import BOW_Batcher
from time import time
#from helpers.data_viz import print_dists
import numpy as np
import random
from pprint import pprint


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"


class Model:

    def __init__(self, train_inputs, train_targets, test_inputs, test_targets,  n_titles, vocab_size, class_mapping=None,
                 use_dropout=True, num_layers=1, keep_prob=0.5, hidden_dim=250, use_attention=False, attention_dim=100,
                 use_embedding=True, max_grad_norm=5, embedding_dim=100, rnn_cell_type='LSTM', max_timesteps=33,
                 learning_rate=0.001, batch_size=100, n_epochs=800, log_interval=200, store_model=True, restore=True,
                 store_dir="/data/rali7/Tmp/solimanz/data/models/", log_dir=".log/", name=''):

        self.log_interval = log_interval
        self.titles_to_id = class_mapping
        self.restore = restore
        self.max_grad_norm = max_grad_norm
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
        self.train_inputs = train_inputs
        self.train_targets = train_targets
        self.test_inputs = test_inputs
        self.test_targets = test_targets
        self.vocab_size = vocab_size
        self.store_dir = store_dir
        self.log_dir = log_dir
        self.store = store_model
        self.num_layers = num_layers
        self.hparams = f"{name}_bow_title_seq_{rnn_cell_type}_{num_layers}_layers_cell_lr_{learning_rate}_use_emb={use_embedding}_emb_dim={embedding_dim}_" \
                       f"hdim={hidden_dim}_dropout={keep_prob}_data_size={len(self.train_inputs)}_vocab_size={self.vocab_size}"
        self.checkpoint_dir = os.path.join(self.store_dir, f"{self.hparams}")

        self.build_model()

    def build_model(self):
        self._predict()
        self._loss()
        self._accuracy()
        self._optimize()
        self._top_2_metric()
        self._top_3_metric()
        self._top_4_metric()
        self._top_5_metric()
        self.train_summ_op = tf.summary.merge([
            self.train_acc_summ,
            self.train_loss_summ,
            self.train_top_2_summ,
            self.train_top_3_summ,
            self.train_top_4_summ,
            self.train_top_5_summ])
        self.test_summ_op = tf.summary.merge([
            self.test_acc_summ,
            self.test_loss_summ,
            self.test_top_2_summ,
            self.test_top_3_summ,
            self.test_top_4_summ,
            self.test_top_5_summ])
        self.writer = tf.summary.FileWriter(os.path.join(self.log_dir, self.hparams))
        self.saver = tf.train.Saver()

    def _predict(self):
        """
        Build the inference graph

        """
        # Keep probability for the dropout
        self.dropout = tf.placeholder(tf.float32, name="dropout_prob")
        # Our list of job title sequences (padded to max_timesteps)
        self.titles_input_data = tf.placeholder(tf.float32, [None, self.max_timesteps, self.vocab_size], name="titles_input_data")
        # matrix that will hold the length of out sequences
        self.seq_lengths = tf.placeholder(tf.int32, [None], name="seq_lengths")
        self.target_inputs = tf.placeholder(tf.int32, [None, self.max_timesteps], name="labels")
        # Do embedding
        with tf.device("/cpu:0"):
            onehot = tf.Variable(tf.eye(self.n_titles, dtype=tf.int16), trainable=False, name="title_one_hot_encoding")
            # tile_emb_input has shape batch_size x times steps x emb_dim
            self.targets = tf.nn.embedding_lookup(onehot, self.target_inputs, name="onehot_encoded_seq")

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
                                                    self.titles_input_data,
                                                    sequence_length=self.seq_lengths,
                                                    dtype=tf.float32,
                                                    parallel_iterations=1024)
        output = tf.reshape(self.output, [-1, self.hidden_dim])


        self.logit = tf.layers.dense(output,
                                     self.n_titles,
                                     activation=None,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.contrib.layers.xavier_initializer(), name="fc_logit")

        prediction_softmax = tf.nn.softmax(self.logit, name="prediction")
        self.prediction = tf.reshape(prediction_softmax, [-1, self.max_timesteps, self.n_titles])

        return self.prediction

    def _loss(self):
        with tf.name_scope("xent"):
            #cross_entropy = -tf.reduce_sum(self.target_one_hot * tf.log(self.prediction), [1, 2])
            #self.cross_entropy = tf.reduce_mean(cross_entropy)

            # Compute cross entropy for each frame.
            cross_entropy = tf.cast(self.targets, tf.float32) * tf.log(self.prediction)
            cross_entropy = -tf.reduce_sum(cross_entropy, 2)
            mask = tf.sign(tf.reduce_max(tf.abs(self.targets), 2))
            cross_entropy *= tf.cast(mask, tf.float32)
            # Average over actual sequence lengths.
            cross_entropy = tf.reduce_sum(cross_entropy, 1)
            cross_entropy /= tf.reduce_sum(tf.cast(mask, tf.float32), 1)
            self.cross_entropy = tf.reduce_mean(cross_entropy)
            self.train_loss_summ = tf.summary.scalar("train_xent", self.cross_entropy)
            self.test_loss_summ = tf.summary.scalar("test_xent", self.cross_entropy)
            return self.cross_entropy

    def _optimize(self):
        with tf.name_scope("train"):
            #self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.cross_entropy)
            tvars = tf.trainable_variables()
            optimizer = tf.train.AdamOptimizer(self.lr)
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cross_entropy, tvars), self.max_grad_norm)
            self.optimize = optimizer.apply_gradients(zip(grads, tvars))

            return self.optimize

    def _accuracy(self):
        with tf.name_scope("accuracy"):
            correct = tf.equal(
                tf.argmax(self.targets, axis=2, output_type=tf.int32),
                tf.argmax(self.prediction, axis=2, output_type=tf.int32))
            correct = tf.cast(correct, tf.int16)
            mask = tf.sign(tf.reduce_max(tf.abs(self.targets), reduction_indices=2))
            correct *= mask
            # Average over actual sequence lengths.
            correct = tf.reduce_sum(correct, reduction_indices=1)
            correct /= tf.cast(self.seq_lengths, tf.float32)
            self.accuracy =  tf.reduce_mean(correct)
            self.train_acc_summ = tf.summary.scalar("training_accuracy", self.accuracy)
            self.test_acc_summ = tf.summary.scalar("test_accuracy", self.accuracy)
            return self.accuracy

    def _top_2_metric(self):
        with tf.name_scope("top_k_accs"):
            with tf.name_scope("top_2"):
                values, indices = tf.nn.top_k(self.prediction, k=2, name='top_2_op')
                labels = tf.argmax(self.targets, axis=2, output_type=tf.int32)
                labels = tf.reshape(labels, [-1, self.max_timesteps, 1])
                correct = tf.reduce_max(tf.cast(tf.equal(indices, labels), dtype=tf.int32), reduction_indices=2)
                mask = tf.sign(tf.reduce_max(tf.abs(self.targets), reduction_indices=2))
                correct *= mask
                # Average over actual sequence lengths.
                correct = tf.reduce_sum(correct, reduction_indices=1)
                correct /= self.seq_lengths

                self.top_2_acc = tf.reduce_mean(correct)
                self.train_top_2_summ = tf.summary.scalar("training_top_2_accuracy", self.top_2_acc)
                self.test_top_2_summ = tf.summary.scalar("test_top_2_accuracy", self.top_2_acc)

                return self.top_2_acc

    def _top_3_metric(self):

        with tf.name_scope("top_k_accs"):
            with tf.name_scope("top_3"):
                values, indices = tf.nn.top_k(self.prediction, k=3, name='top_3_op')
                labels = tf.argmax(self.targets, axis=2, output_type=tf.int32)
                labels = tf.reshape(labels, [-1, self.max_timesteps, 1])
                correct = tf.reduce_max(tf.cast(tf.equal(indices, labels), dtype=tf.int32), reduction_indices=2)
                mask = tf.sign(tf.reduce_max(tf.abs(self.targets), reduction_indices=2))
                correct *= mask
                # Average over actual sequence lengths.
                correct = tf.reduce_sum(correct, reduction_indices=1)
                correct /= self.seq_lengths

                self.top_3_acc = tf.reduce_mean(correct)
                self.train_top_3_summ = tf.summary.scalar("training_top_3_accuracy", self.top_3_acc)
                self.test_top_3_summ = tf.summary.scalar("test_top_3_accuracy", self.top_3_acc)

                return self.top_3_acc

    def _top_4_metric(self):

        with tf.name_scope("top_k_accs"):
            with tf.name_scope("top_4"):
                values, indices = tf.nn.top_k(self.prediction, k=4, name='top_4_op')
                labels = tf.argmax(self.targets, axis=2, output_type=tf.int32)
                labels = tf.reshape(labels, [-1, self.max_timesteps, 1])
                correct = tf.reduce_max(tf.cast(tf.equal(indices, labels), dtype=tf.int32), reduction_indices=2)
                mask = tf.sign(tf.reduce_max(tf.abs(self.targets), reduction_indices=2))
                correct *= mask
                # Average over actual sequence lengths.
                correct = tf.reduce_sum(correct, reduction_indices=1)
                correct /= self.seq_lengths

                self.top_4_acc = tf.reduce_mean(correct)
                self.train_top_4_summ = tf.summary.scalar("training_top_4_accuracy", self.top_4_acc)
                self.test_top_4_summ = tf.summary.scalar("test_top_4_accuracy", self.top_4_acc)

                return self.top_4_acc

    def _top_5_metric(self):

        with tf.name_scope("top_k_accs"):
            with tf.name_scope("top_5"):
                values, indices = tf.nn.top_k(self.prediction, k=5, name='top_5_op')
                labels = tf.argmax(self.targets, axis=2, output_type=tf.int32)
                labels = tf.reshape(labels, [-1, self.max_timesteps, 1])
                correct = tf.reduce_max(tf.cast(tf.equal(indices, labels), dtype=tf.int32), reduction_indices=2)
                mask = tf.sign(tf.reduce_max(tf.abs(self.targets), reduction_indices=2))
                correct *= mask
                # Average over actual sequence lengths.
                correct = tf.reduce_sum(correct, reduction_indices=1)
                correct /= self.seq_lengths

                self.top_5_acc = tf.reduce_mean(correct)
                self.train_top_5_summ =tf.summary.scalar("training_top_5_accuracy", self.top_5_acc)
                self.test_top_5_summ = tf.summary.scalar("test_top_5_accuracy", self.top_5_acc)

                return self.top_5_acc

    def train(self):

        print("Creating batchers")
        train_batcher = BOW_Batcher(batch_size=self.batch_size, step_num=self.max_timesteps, input_data=self.train_inputs,
                                    target_data=self.train_targets, n_classes=self.n_titles, vocab_size=self.vocab_size)
        test_batcher = BOW_Batcher(batch_size=self.batch_size, step_num=self.max_timesteps, input_data=self.test_inputs,
                                    target_data=self.test_targets, n_classes=self.n_titles, vocab_size=self.vocab_size)

        # Assume that you have 12GB of GPU memory and want to allocate ~4GB:
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True

        with tf.Session(config=gpu_config) as sess:

            sess.run(tf.global_variables_initializer())
            self.writer.add_graph(sess.graph)

            if self.load(sess, self.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

            for e in range(self.n_epochs):
                start_time = time()
                batch = 0
                for b in range(train_batcher.max_batch_num):

                    with tf.device("/cpu:0"):
                        title_seq, seq_lengths, targets = train_batcher.next()

                    loss, _, acc, top_2_acc, top_3_acc, top_4_acc, top_5_acc, summary = sess.run([self.cross_entropy, self.optimize, self.accuracy,
                                                                 self.top_2_acc,
                                                                 self.top_3_acc,
                                                                 self.top_4_acc,
                                                                 self.top_5_acc,
                                                                 self.train_summ_op],
                                            {
                                                self.titles_input_data: title_seq,
                                                self.seq_lengths: seq_lengths,
                                                self.target_inputs: targets,
                                                self.dropout: self.keep_prob
                                            })

                    self.writer.add_summary(summary, b)

                    if batch % self.log_interval == 0 and batch > 0:
                        elapsed = time() - start_time
                        print(
                            f'| epoch {e} | {train_batcher.batch_num}/{train_batcher.max_batch_num} batches | lr {self.lr} | '
                            f'ms/batch {elapsed * 1000 / self.log_interval} | loss {loss:.4f} | acc: {acc*100:.2f}% | top 2 acc: {top_2_acc*100:.2f}%'
                            f' | top 3 acc: {top_3_acc*100:.2f}% | top 4 acc: {top_4_acc*100:.2f}% | top 5 acc: {top_5_acc*100:.2f}%')

                        start_time = time()

                    batch += 1

                print(f"Epoch:, {(e + 1)}")

                avg_acc = []
                avg_top_2 = []
                avg_top_3 = []
                avg_top_4 = []
                avg_top_5 = []

                for tb in range(test_batcher.max_batch_num):
                    with tf.device("/cpu:0"):
                        test_title_seq, test_seq_lengths, test_target = test_batcher.next()

                    test_acc, test_top_2, test_top_3, test_top_4, test_top_5, test_summ, pred = sess.run([self.accuracy,
                                                                                                          self.top_2_acc,
                                                                                                          self.top_3_acc,
                                                                                                          self.top_4_acc,
                                                                                                          self.top_5_acc,
                                                                                                          self.test_summ_op,
                                                                                                          self.prediction],
                                                   {
                                                       self.titles_input_data: test_title_seq,
                                                       self.seq_lengths: test_seq_lengths,
                                                       self.target_inputs: test_target,
                                                       self.dropout: 1.0
                                                   })
                    if test_acc > 0:
                        avg_acc.append(test_acc)
                    if test_top_2 > 0:
                        avg_top_2.append(test_top_2)
                    if test_top_3 > 0:
                        avg_top_3.append(test_top_3)
                    if test_top_4 > 0:
                        avg_top_4.append(test_top_4)
                    if test_top_5 > 0:
                        avg_top_5.append(test_top_5)

                    #print_dists(self.titles_to_id, test_seq_lengths, test_title_seq, pred, test_target, f_name=self.hparams)
                    self.writer.add_summary(test_summ, tb)

                print(f"Accuracy on test: {sum(avg_acc)/len(avg_acc)*100:.2f}%")
                print(f"Top 2 accuracy on test: {sum(avg_top_2)/len(avg_top_2)*100:.2f}%")
                print(f"Top 3 accuracy on test: {sum(avg_top_3)/len(avg_top_3)*100:.2f}%")
                print(f"Top 4 accuracy on test: {sum(avg_top_4)/len(avg_top_4)*100:.2f}%")
                print(f"Top 5 accuracy on test: {sum(avg_top_5)/len(avg_top_5)*100:.2f}%")

                if self.store and e % 10 == 0:
                    save_path = self.save(sess, self.checkpoint_dir, e)
                    print("model saved in file: %s" % save_path)

    def save(self, sess, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        return self.saver.save(sess,
                        os.path.join(checkpoint_dir, self.hparams),
                        global_step=step)

    def load(self, sess, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False


def main():
    path = "/data/rali7/Tmp/solimanz/data/datasets/1/title_sequences.json"
    mapping, train_data, test_data = load_data(path)
    seq_model = Model(train_data=train_data, num_layers=2)
    seq_model.train()


if __name__ == "__main__":
    main()
