import tensorflow as tf
import os, random, string
from helpers.loader import load_data
from helpers.batcher import MultiLabelBatcher as Batcher
from time import time
# from helpers.data_viz import print_dists
import numpy as np
import random
from pprint import pprint


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"


class Model:

    def __init__(self, train_inputs, train_targets, test_inputs, test_targets, n_titles, n_tokens, class_mapping=None,
                 use_dropout=True, num_layers=1, keep_prob=0.5, hidden_dim=512, freeze_emb=True, max_grad_norm=5,
                 embedding_dim=300, rnn_cell_type='LSTM', max_timesteps=33, learning_rate=0.001, batch_size=100,
                 n_epochs=800, log_interval=2000, store_model=True, restore=True,
                 emb_path="/data/rali7/Tmp/solimanz/data/datasets/5/embeddings_small.npy",
                 store_dir="/data/rali7/Tmp/solimanz/data/models/", log_dir=".log", name=''):

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
        self.test_targets = test_targets
        self.n_tokens = n_tokens
        self.store_dir = store_dir
        self.log_dir = log_dir
        self.store = store_model
        self.num_layers = num_layers
        self.emb_path = emb_path
        self.hparams = f"{name}_fasttext_mutlilabel_title_seq_{rnn_cell_type}_{num_layers}_layers_cell_lr_{learning_rate}_freeze_emb={freeze_emb}_emb_dim={embedding_dim}_" \
                       f"hdim={hidden_dim}_dropout={keep_prob}_data_size={len(self.train_inputs)}_vocab_size={self.n_tokens}"
        self.checkpoint_dir = os.path.join(self.store_dir, f"{self.hparams}")

        self.build_model()

    def build_model(self):
        self._predict()
        self._loss()
        self._total_accuracy()
        self._optimize()
        self._exact_accuracy()
        self.train_summ_op = tf.summary.merge([
            self.train_acc_summ,
            self.train_acc_all_summ,
            self.train_loss_summ])
        self.test_summ_op = tf.summary.merge([
            self.test_acc_summ,
            self.test_acc_all_summ,
            self.test_loss_summ])
        self.writer = tf.summary.FileWriter(os.path.join(self.log_dir, self.hparams))
        self.saver = tf.train.Saver()

    def _predict(self):
        """
        Build the inference graph

        """

        # Keep probability for the dropout
        self.dropout = tf.placeholder(tf.float32, name="dropout_prob")
        # Our list of job title sequences (padded to max_time_steps)
        self.titles_input_data = tf.placeholder(tf.int32, [None, self.max_timesteps],
                                                name="titles_input_data")
        # matrix that will hold the length of out sequences
        self.seq_lengths = tf.placeholder(tf.int32, [None], name="seq_lengths")
        self.targets = tf.placeholder(tf.int32, [None, self.max_timesteps, self.n_tokens], name="targets")

        with tf.device("/cpu:0"):
            embeddings_matrix = np.load(self.emb_path)
            title_embedding = tf.get_variable(name="title_embedding",
                                             shape=[self.n_titles, self.emb_dim],
                                             dtype=tf.float32,
                                             initializer=tf.constant_initializer(embeddings_matrix),
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

        cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

        self.output, self.prev_states = tf.nn.dynamic_rnn(cell,
                                                          input_seqs,
                                                          sequence_length=self.seq_lengths,
                                                          dtype=tf.float32,
                                                          parallel_iterations=1024)

        output = tf.reshape(self.output, [-1, self.hidden_dim])
        self.logit = tf.layers.dense(output,
                                     self.n_tokens,
                                     activation=None,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.contrib.layers.xavier_initializer(), name="fc_logit")

        logits_sigmoid = tf.nn.sigmoid(self.logit, name="predictions")
        self.prediction = tf.reshape(logits_sigmoid, [-1, self.max_timesteps, self.n_tokens], name="prediction_sigmoid")
        self.logit = tf.reshape(self.logit, [-1, self.max_timesteps, self.n_tokens])
        self.preds = tf.round(self.prediction)
        return self.prediction

    def _loss(self):
        with tf.name_scope("xent"):
            # cross_entropy = -tf.reduce_sum(self.target_one_hot * tf.log(self.prediction), [1, 2])
            # self.cross_entropy = tf.reduce_mean(cross_entropy)

            # Compute cross entropy for each frame.
            # cross_entropy = tf.cast(self.targets, tf.float32) * tf.log(self.prediction)
            # cross_entropy = -tf.reduce_sum(cross_entropy, 2)
            # mask = tf.sequence_mask(self.seq_lengths, maxlen=self.max_timesteps, dtype=tf.int32)
            # cross_entropy *= tf.cast(mask, tf.float32)
            # # Average over actual sequence lengths.
            # cross_entropy = tf.reduce_sum(cross_entropy, 1)
            # cross_entropy /= tf.reduce_sum(tf.cast(mask, tf.float32), 1)
            # self.cross_entropy = tf.reduce_mean(cross_entropy)

            self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.cast(self.targets, dtype=tf.float32),
                logits=self.logit))
            self.train_loss_summ = tf.summary.scalar("train_xent", self.cross_entropy)
            self.test_loss_summ = tf.summary.scalar("test_xent", self.cross_entropy)
            return self.cross_entropy

    def _optimize(self):
        with tf.name_scope("train"):
            # self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.cross_entropy)
            tvars = tf.trainable_variables()
            optimizer = tf.train.AdamOptimizer(self.lr)
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cross_entropy, tvars), self.max_grad_norm)
            self.optimize = optimizer.apply_gradients(zip(grads, tvars))

            return self.optimize

    def _total_accuracy(self):
        with tf.name_scope("accuracy"):
            predicted_labels = tf.cast(tf.round(self.prediction), dtype=tf.bool)
            same = tf.logical_and(predicted_labels, tf.cast(self.targets, dtype=tf.bool))
            same = tf.reduce_sum(tf.cast(same, dtype=tf.int32), reduction_indices=2)
            mask = tf.sequence_mask(self.seq_lengths, maxlen=self.max_timesteps, dtype=tf.int32)
            same *= mask
            same = tf.cast(same, dtype=tf.float32)

            same = tf.reduce_sum(same, reduction_indices=[1,0])
            total = tf.reduce_sum(self.targets, reduction_indices=[2,1,0])

            self.accuracy = same/tf.cast(total, dtype=tf.float32)
            self.train_acc_summ = tf.summary.scalar("training_accuracy", self.accuracy)
            self.test_acc_summ = tf.summary.scalar("test_accuracy", self.accuracy)
            return self.accuracy

    def _exact_accuracy(self):
        with tf.name_scope("accuracy"):

            predicted_labels = tf.cast(tf.round(self.prediction), dtype=tf.bool)
            same = tf.logical_and(predicted_labels, tf.cast(self.targets, dtype=tf.bool))
            same = tf.sign(tf.reduce_sum(tf.cast(same, dtype=tf.int32), reduction_indices=2))
            mask = tf.sequence_mask(self.seq_lengths, maxlen=self.max_timesteps, dtype=tf.int32)
            same *= mask
            same = tf.cast(same, dtype=tf.float32)
            # Average over actual sequence lengths.
            same = tf.reduce_sum(same, reduction_indices=1)
            same /= tf.cast(self.seq_lengths, tf.float32)

            #correct = tf.equal(predicted_labels, self.targets)
            #all_labels_true = tf.reduce_min(tf.cast(correct, dtype=tf.float32), 1)
            #self.accuracy2 = tf.reduce_mean(all_labels_true)
            self.accuracy2 = tf.reduce_mean(same)
            self.train_acc_all_summ = tf.summary.scalar("training_accuracy_all", self.accuracy2)
            self.test_acc_all_summ = tf.summary.scalar("test_accuracy_all", self.accuracy2)
            return self.accuracy2

    def train(self):

        print("Creating batchers")
        train_batcher = Batcher(batch_size=self.batch_size, step_num=self.max_timesteps,
                                    input_data=self.train_inputs,
                                    target_data=self.train_targets, n_classes=self.n_titles, n_tokens=self.n_tokens)
        test_batcher = Batcher(batch_size=self.batch_size, step_num=self.max_timesteps, input_data=self.test_inputs,
                                   target_data=self.test_targets, n_classes=self.n_titles, n_tokens=self.n_tokens)

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

                    loss, _, acc, acc2, summary = sess.run([
                        self.cross_entropy,
                        self.optimize,
                        self.accuracy,
                        self.accuracy2,
                        self.train_summ_op
                    ],
                        {
                            self.titles_input_data: title_seq,
                            self.seq_lengths: seq_lengths,
                            self.targets: targets,
                            self.dropout: self.keep_prob
                        })


                    if batch % self.log_interval == 0 and batch > 0:
                        elapsed = time() - start_time
                        print(
                            f'| epoch {e} | {train_batcher.batch_num}/{train_batcher.max_batch_num} batches | lr {self.lr} | '
                            f'ms/batch {elapsed * 1000 / self.log_interval} | loss {loss:.10f} | acc: {acc*100:.2f}% | '
                            f'acc2: {acc2*100:.2f}%')

                        start_time = time()

                    batch += 1

                print(f"Epoch:, {(e + 1)}")

                avg_acc = []
                avg_acc2 = []
                for tb in range(test_batcher.max_batch_num):
                    with tf.device("/cpu:0"):
                        test_title_seq, test_seq_lengths, test_target = test_batcher.next()

                    test_acc, test_acc2, test_summ = sess.run([
                        self.accuracy,
                        self.accuracy2,
                        self.test_summ_op
                    ],
                        {
                            self.titles_input_data: test_title_seq,
                            self.seq_lengths: test_seq_lengths,
                            self.targets: test_target,
                            self.dropout: 1.0
                        })


                    avg_acc.append(test_acc)
                    avg_acc2.append(test_acc2)
                    # print_dists(self.titles_to_id, test_seq_lengths, test_title_seq, pred, test_target, f_name=self.hparams)
                print(f"Accuracy on test: {sum(avg_acc)/len(avg_acc)*100:.2f}%")
                print(f"Accuracy (all labels) on test: {sum(avg_acc2)/len(avg_acc2)*100:.2f}%")

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
