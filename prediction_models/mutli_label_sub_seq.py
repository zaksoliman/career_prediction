import tensorflow as tf
import os, random, string
from helpers.loader import load_data
from helpers.batcher import MultiLabelBatcher as TrainBatcher
from helpers.batcher import SequenceBatcher as TestBatcher
from time import time
from sklearn.metrics import roc_auc_score
# from helpers.data_viz import print_dists
import numpy as np
import random
from pprint import pprint


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


class Model:

    def __init__(self, train_inputs, train_targets, test_inputs, n_titles, class_mapping=None,
                 use_dropout=True, keep_prob=0.5, use_residual=False, num_layers=1, hidden_dim=512, freeze_emb=False,
                 max_grad_norm=5, embedding_dim=300, rnn_cell_type='LSTM', max_timesteps=33, learning_rate=0.001,
                 batch_size=100, n_epochs=800, log_interval=100, store_model=True, restore=True, log_dir=".log", name='',
                 store_dir="/data/rali7/Tmp/solimanz/data/models/"):

        self.log_interval = log_interval
        self.titles_to_id = class_mapping
        self.restore = restore
        self.max_grad_norm = max_grad_norm
        self.keep_prob = keep_prob
        self.use_dropout = use_dropout
        self.n_titles = n_titles
        self.n_epochs = n_epochs
        self.freeze_embedding = freeze_emb
        self.rnn_cell_type = rnn_cell_type
        self.emb_dim = embedding_dim
        self.max_timesteps = max_timesteps
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.train_inputs = train_inputs
        self.train_targets = train_targets
        self.test_inputs = test_inputs
        self.store_dir = store_dir
        self.log_dir = log_dir
        self.store = store_model
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.hparams = f"{name}_fasttext_mutlilabel_title_seq_{rnn_cell_type}_{num_layers}_" \
                       f"layers_cell_lr_{learning_rate}_freeze_emb={freeze_emb}_emb_dim={embedding_dim}_" \
                       f"hdim={hidden_dim}_dropout={keep_prob}_data_size={len(self.train_inputs)}_nlabels={self.n_titles}"
        self.checkpoint_dir = os.path.join(self.store_dir, f"{self.hparams}")

        self.build_model()

    def build_model(self):
        self._predict()
        self._loss()
        self._train_accuracy()
        self._test_accuracy()
        self._hamming_loss()
        self._hinge_loss()
        self._optimize()
        self.train_summ_op = tf.summary.merge([
            self.train_acc_summ,
            self.train_loss_summ,
            self.hamming_summ,
            self.hinge_loss_summ
        ])
        self.test_summ_op = tf.summary.merge([
            self.test_acc_summ
        ])
        self.writer = tf.summary.FileWriter(os.path.join(self.log_dir, self.hparams))
        self.saver = tf.train.Saver()

    def _predict(self):
        """
        Build the inference graph

        """
        # Keep probability for the dropout
        self.dropout = tf.placeholder(tf.float32, name="dropout_prob")
        # Our list of job title sequences (padded to max_time_steps)
        self.titles_input_data = tf.placeholder(tf.int32, [None, self.max_timesteps], name="titles_input_data")
        # matrix that will hold the length of out sequences
        self.seq_lengths = tf.placeholder(tf.int32, [None], name="seq_lengths")
        self.targets = tf.placeholder(tf.int32, [None, self.n_titles], name="targets")

        with tf.device("/cpu:0"):
            title_embedding = tf.get_variable(name="title_embedding",
                                             shape=[self.n_titles, self.emb_dim],
                                             dtype=tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer(),
                                             trainable=not self.freeze_embedding)

            input_seqs = tf.nn.embedding_lookup(title_embedding, self.titles_input_data, name="embedding_lookup")

        # Decide on out RNN cell type
        if self.rnn_cell_type == 'RNN':
            get_cell = tf.nn.rnn_cell.BasicRNNCell
        elif self.rnn_cell_type == 'LSTM':
            get_cell = tf.nn.rnn_cell.LSTMCell
        else:  # Default to GRU
            get_cell = tf.nn.rnn_cell.GRUCell

        cells = [get_cell(self.hidden_dim) for _ in range(self.num_layers)]

        if self.use_dropout:
            cells = [tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout) for cell in cells]
        if self.use_residual:
            cells = [tf.contrib.rnn.ResidualWrapper(cell) for i, cell in enumerate(cells) if i > 0]

        cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

        self.output, self.prev_states = tf.nn.dynamic_rnn(cell,
                                                          input_seqs,
                                                          sequence_length=self.seq_lengths,
                                                          dtype=tf.float32,
                                                          parallel_iterations=1024)

        batch_range = tf.range(tf.shape(self.output)[0])
        indices = tf.stack([batch_range, self.seq_lengths - 1], axis=1)
        out = tf.gather_nd(self.output, indices)

        self.logits = tf.layers.dense(out,
                                     self.n_titles,
                                     activation=None,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.contrib.layers.xavier_initializer(), name="fc_logit")

        self.prediction = tf.nn.sigmoid(self.logits, name="predictions")


        return self.prediction

    def _loss(self):
        with tf.name_scope("xent"):
            self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.cast(self.targets, dtype=tf.float32),
                logits=self.logits))
            self.train_loss_summ = tf.summary.scalar("train_xent", self.cross_entropy)

    def _hinge_loss(self):
        with tf.name_scope("hinge_loss"):
            self.hinge = tf.losses.hinge_loss(self.targets, self.logits)
            self.hinge_loss_summ = tf.summary.scalar("hinge_loss", self.hinge)

    def _optimize(self):
        with tf.name_scope("train"):
            #tvars = tf.trainable_variables()
            #optimizer = tf.train.AdamOptimizer(self.lr)
            #grads, _ = tf.clip_by_global_norm(tf.gradients(self.cross_entropy, tvars), self.max_grad_norm)
            #self.optimize = optimizer.apply_gradients(zip(grads, tvars))
            self.optimize = tf.train.AdamOptimizer(learning_rate = self.lr).minimize(self.hinge)
            return self.optimize

    def _train_accuracy(self):
        with tf.name_scope("train_accuracy"):
            predicted_labels = tf.cast(tf.round(self.prediction), dtype=tf.int32)
            correct_prediction = tf.equal(predicted_labels, self.targets)
            self.train_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.train_acc_summ = tf.summary.scalar("training_accuracy", self.train_accuracy)
            return self.train_accuracy

    def _test_accuracy(self):
        # Check if at least one of the predicted labels is correct
        with tf.name_scope("test_accuracy"):
            predicted_labels = tf.cast(tf.round(self.prediction), dtype=tf.int32)

            ground_truth = tf.argmax(self.targets, axis=1, output_type=tf.int32)
            batch_range = tf.range(tf.shape(self.targets)[0], dtype=tf.int32)
            indices = tf.stack([batch_range, ground_truth], axis=1)

            label_predictions = tf.gather_nd(predicted_labels, indices)
            correct_prediction = tf.equal(label_predictions, ground_truth)
            self.test_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.test_acc_summ = tf.summary.scalar("test_accuracy", self.test_accuracy)
            return self.test_accuracy

    def _hamming_loss(self):
        with tf.name_scope("train_hamming_loss"):
            predicted_labels = tf.cast(tf.round(self.prediction), dtype=tf.int32)
            self.hamming_loss = tf.reduce_mean(tf.cast(
                tf.logical_xor(
                    tf.cast(self.targets, dtype=tf.bool),
                    tf.cast(predicted_labels, dtype=tf.bool)),
                dtype=tf.float32))
            self.hamming_summ = tf.summary.scalar("hamming_loss", self.hamming_loss)
            return self.hamming_loss

    def train(self):

        print("Creating batchers")
        train_batcher = TrainBatcher(
            batch_size=self.batch_size,
            step_num=self.max_timesteps,
            input_data=self.train_inputs,
            target_data=self.train_targets,
            n_classes=self.n_titles)
        test_batcher = TestBatcher(
            batch_size=self.batch_size,
            step_num=self.max_timesteps,
            data=self.test_inputs,
            n_classes=self.n_titles)

        # Assume that you have 12GB of GPU memory and want to allocate ~4GB:
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True

        with tf.Session(config=gpu_config) as sess:

            #sess.run(tf.global_variables_initializer())
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)
            self.writer.add_graph(sess.graph)

            print("Attempting to lead checkpoint...")
            if self.load(sess, self.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

            print("Started Training...")
            for e in range(self.n_epochs):
                start_time = time()
                batch = 0
                for b in range(train_batcher.max_batch_num):
                    with tf.device("/cpu:0"):
                        title_seq, seq_lengths, targets = train_batcher.next()

                    loss, _, acc, hamming_loss, summary = sess.run([
                        self.hinge,
                        self.optimize,
                        self.train_accuracy,
                        self.hamming_loss,
                        self.train_summ_op
                    ],
                    feed_dict= {
                        self.titles_input_data: title_seq,
                        self.seq_lengths: seq_lengths,
                        self.targets: targets,
                        self.dropout: self.keep_prob
                    })

                    if batch % self.log_interval == 0 and batch > 0:
                        elapsed = time() - start_time
                        print(
                            f"| epoch {e} | {train_batcher.batch_num}/{train_batcher.max_batch_num} batches | lr {self.lr} | "
                            f"ms/batch {elapsed * 1000 / self.log_interval:.4f} | loss {loss:.10f} | acc: {acc*100:.2f}% | "
                            f"hamming loss: {hamming_loss:.4f}")

                        start_time = time()

                    batch += 1

                avg_acc = []
                for tb in range(test_batcher.max_batch_num):
                    with tf.device("/cpu:0"):
                        test_title_seq, test_seq_lengths, test_target = test_batcher.next()

                    test_acc, test_summ = sess.run([
                        self.test_accuracy,
                        self.test_summ_op
                    ],
                    feed_dict={
                        self.titles_input_data: test_title_seq,
                        self.seq_lengths: test_seq_lengths,
                        self.targets: test_target,
                        self.dropout: 1.0
                    })

                    avg_acc.append(test_acc)

                print(f"Accuracy on test: {sum(avg_acc)/len(avg_acc)*100:.2f}%")
                print(f"Starting Epoch:, {(e + 1)}")

                if self.store and e % 5 == 0:
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
