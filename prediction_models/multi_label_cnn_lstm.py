import tensorflow as tf
import os, random, string
from helpers.loader import load_data
from helpers.batcher import MultiLabelSkillBatcher as Batcher
from time import time
import numpy as np
import random
from pprint import pprint


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def prod(a):
    p = 1
    for e in a:
        p *= e
    return p

class Model:

    def __init__(self, train_inputs, train_targets, train_skills, test_inputs, test_targets, test_skills, n_titles,
                 n_labels, n_skills, max_skills, class_mapping=None, use_dropout=True, num_layers=1, keep_prob=0.5,
                 hidden_dim=512, n_filters=2, kernel_sizes=(2,3,4,5), freeze_emb=False, max_grad_norm=5, embedding_dim=300,
                 rnn_cell_type='LSTM', max_timesteps=33, learning_rate=0.001, batch_size=100, n_epochs=800,
                 log_interval=20, store_model=True, restore=True, emb_path="", skill_emb_path='',
                 store_dir="/data/rali7/Tmp/solimanz/data/models/", log_dir=".log", name=''):

        self.log_interval = log_interval
        self.titles_to_id = class_mapping
        self.restore = restore
        self.max_skills = max_skills
        self.n_skills = n_skills
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
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
        self.train_skills = train_skills
        self.test_inputs = test_inputs
        self.test_targets = test_targets
        self.test_skills = test_skills
        self.n_labels = n_labels
        self.store_dir = store_dir
        self.log_dir = log_dir
        self.store = store_model
        self.num_layers = num_layers
        self.emb_path = emb_path
        self.skill_emb_path = skill_emb_path
        self.hparams = f"{name}_cnn_v2_fasttext_{rnn_cell_type}_{num_layers}_layers_cell_lr_{learning_rate}_n_filters={n_filters}_{len(kernel_sizes)}_kernels_freeze_emb={freeze_emb}_emb_dim={embedding_dim}_" \
                       f"hdim={hidden_dim}_dropout={keep_prob}_data_size={len(self.train_inputs)}_n_labels={self.n_labels}"
        self.checkpoint_dir = os.path.join(self.store_dir, f"{self.hparams}")

        self.build_model()


    def build_model(self):
        self._predict()
        self._loss()
        self._accuracy()
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
        self.titles_input_data = tf.placeholder(tf.int32, [None, self.max_timesteps], name="titles_input_data")
        self.skills_input = tf.placeholder(tf.int32, [None, self.max_skills], name="skills_input_data")
        # matrix that will hold the length of out sequences
        self.seq_lengths = tf.placeholder(tf.int32, [None], name="seq_lengths")
        self.targets = tf.placeholder(tf.int32, [None, self.n_labels], name="targets")

        with tf.device("/cpu:0"):
            skill_embs = np.load(self.skill_emb_path)
            skill_embedding = tf.get_variable(name="skill_embedding",
                                              shape=[skill_embs.shape[0], skill_embs.shape[1]],
                                              dtype=tf.float32,
                                              initializer=tf.constant_initializer(skill_embs),
                                              trainable=not self.freeze_embedding)
            embeddings_matrix = np.load(self.emb_path)
            self.emb_dim = embeddings_matrix.shape[1]
            title_embedding = tf.get_variable(name="title_embedding",
                                             shape=[self.n_titles, self.emb_dim],
                                             dtype=tf.float32,
                                             initializer=tf.constant_initializer(embeddings_matrix),
                                             trainable=not self.freeze_embedding)

            self.title_emb_input = tf.nn.embedding_lookup(title_embedding, self.titles_input_data, name="embedding_lookup")
            self.skill_emb_input = tf.nn.embedding_lookup(skill_embedding, self.skills_input, name="skills_lookup")

        ##################
        #   CNN Encoder  #
        ##################
        skills_shape = self.skill_emb_input.get_shape()
        print(f"Shape of skill embs: {self.skill_emb_input.get_shape()}")
        skill_embs = tf.reshape(self.skill_emb_input, [-1, skills_shape[1], skills_shape[2], 1])
        print(f"Shape of skill embs after reshape: {skill_embs.get_shape()}")

        kernel_sizes = self.kernel_sizes
        conv_layers = []
        pool_layers = []

        for i, size in enumerate(kernel_sizes):
            with tf.name_scope(f"conv-maxpool-{size}"):
                conv = tf.layers.conv2d(skill_embs,
                                        filters=self.n_filters,
                                        kernel_size=[size, skills_shape[2]],
                                        padding="valid",
                                        activation=tf.nn.relu)
                conv_shape = conv.get_shape().as_list()
                print(f"Shape of Conv Layer: {conv_shape}")
                pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[conv_shape[1], 1], strides=1, padding='same')
                print(f"Shape of pool Layer: {pool.get_shape()}")
                pool_layers.append(pool)
                conv_layers.append(conv)

        pool_concat = tf.concat(pool_layers, axis=1)
        concat_shape = pool_concat.get_shape().as_list()
        print(f"Shape after concatination: {concat_shape}")

        pool_flat = tf.reshape(pool_concat, [-1, prod(concat_shape[1:])])
        print(f"Shape of flat pool: {pool_flat.get_shape()}")

        with tf.name_scope("pooling-dropout"):
            skill_context = tf.nn.dropout(pool_flat, self.keep_prob)

        c_state = tf.layers.dense(inputs=skill_context, units=self.hidden_dim, activation=tf.nn.relu)
        m_state = tf.layers.dense(inputs=skill_context, units=self.hidden_dim, activation=tf.nn.relu)
        init_states = tf.contrib.rnn.LSTMStateTuple(c_state, m_state)


        #########
        #  RNN  #
        #########
        # Decide on out RNN cell type
        if self.rnn_cell_type == 'RNN':
            get_cell = tf.nn.rnn_cell.BasicRNNCell
        elif self.rnn_cell_type == 'LSTM':
            get_cell = tf.nn.rnn_cell.LSTMCell
        else:  # Default to GRU
            get_cell = tf.nn.rnn_cell.GRUCell

        cell = get_cell(self.hidden_dim)

        if self.use_dropout:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout)

        self.output, self.prev_states = tf.nn.dynamic_rnn(cell,
                                                          self.title_emb_input,
                                                          initial_state=init_states,
                                                          sequence_length=self.seq_lengths,
                                                          dtype=tf.float32,
                                                          parallel_iterations=1024)


        batch_range = tf.range(tf.shape(self.output)[0])
        indices = tf.stack([batch_range, self.seq_lengths - 1], axis=1)
        # Last output is shape (batch size, hidden_dim)
        self.last_output = tf.gather_nd(self.output, indices)

        # Logits of shape (batch size, n_labels)
        self.logit = tf.layers.dense(self.last_output,
                                     self.n_labels,
                                     activation=None,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.contrib.layers.xavier_initializer(),
                                     name="fc_logit")

        self.predictions = tf.nn.sigmoid(self.logit, name="predictions")
        return self.predictions

    def _loss(self):
        with tf.name_scope("xent"):
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

    def _accuracy(self):
        with tf.name_scope("accuracy1"):
            correct_prediction = tf.equal(tf.cast(tf.round(self.predictions), dtype=tf.int32), self.targets)
            self.accuracy1 = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.train_acc_summ = tf.summary.scalar("training_accuracy", self.accuracy1)
            self.test_acc_summ = tf.summary.scalar("test_accuracy", self.accuracy1)
            return self.accuracy1

    def _exact_accuracy(self):
        with tf.name_scope("accuracy2"):
            correct_prediction = tf.equal(tf.cast(tf.round(self.predictions), dtype=tf.int32), self.targets)
            all_labels_true = tf.reduce_min(tf.cast(correct_prediction, tf.float32), 1)
            self.accuracy2 = tf.reduce_mean(all_labels_true)
            self.train_acc_all_summ = tf.summary.scalar("training_accuracy_all", self.accuracy2)
            self.test_acc_all_summ = tf.summary.scalar("test_accuracy_all", self.accuracy2)
            return self.accuracy2

    def train(self):

        print("Creating batchers")
        train_batcher = Batcher(batch_size=self.batch_size, step_num=self.max_timesteps, input_data=self.train_inputs,
                                target_data=self.train_targets, n_classes=self.n_labels, n_skills=self.n_skills,
                                max_skills=self.max_skills, skill_data=self.train_skills)
        test_batcher = Batcher(batch_size=self.batch_size, step_num=self.max_timesteps, input_data=self.test_inputs,
                                target_data=self.test_targets, n_classes=self.n_labels, n_skills=self.n_skills,
                                max_skills=self.max_skills, skill_data=self.test_skills)

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
                        title_seq, seq_lengths, targets, skills = train_batcher.next()

                    loss, _, acc, acc2, summary = sess.run([
                        self.cross_entropy,
                        self.optimize,
                        self.accuracy1,
                        self.accuracy2,
                        self.train_summ_op
                    ],
                        {
                            self.titles_input_data: title_seq,
                            self.seq_lengths: seq_lengths,
                            self.targets: targets,
                            self.skills_input: skills,
                            self.dropout: self.keep_prob
                        })


                    if batch % self.log_interval == 0 and batch > 0:
                        elapsed = time() - start_time
                        print(
                            f'| epoch {e} | {train_batcher.batch_num}/{train_batcher.max_batch_num} batches | lr {self.lr} | '
                            f'ms/batch {elapsed * 1000 / self.log_interval:.3f} | loss {loss:.5f} | acc: {acc*100:.2f}% | '
                            f'acc2: {acc2*100:.2f}%')

                        start_time = time()

                    batch += 1

                print(f"Epoch:, {(e + 1)}")

                avg_acc = []
                avg_acc2 = []
                for tb in range(test_batcher.max_batch_num):
                    with tf.device("/cpu:0"):
                        test_title_seq, test_seq_lengths, test_target, skills = test_batcher.next()

                    test_acc, test_acc2, test_summ, loss = sess.run([
                        self.accuracy1,
                        self.accuracy2,
                        self.test_summ_op,
                        self.cross_entropy
                    ],
                        {
                            self.titles_input_data: test_title_seq,
                            self.seq_lengths: test_seq_lengths,
                            self.targets: test_target,
                            self.skills_input: skills,
                            self.dropout: 1.0
                        })


                    avg_acc.append(test_acc)
                    avg_acc2.append(test_acc2)
                    # print_dists(self.titles_to_id, test_seq_lengths, test_title_seq, pred, test_target, f_name=self.hparams)
                print(f"Accuracy on test: {sum(avg_acc)/len(avg_acc)*100:.2f}%")
                print(f"Accuracy (all labels) on test: {sum(avg_acc2)/len(avg_acc2)*100:.2f}%")
                print(f"Loss on test: {loss:.5f}")

                if self.store and e % 10 == 0:
                    save_path = self.save(sess, self.checkpoint_dir, e)
                    print("model saved in file: %s" % save_path)

    def test(self):

        base_path = "/data/rali7/Tmp/solimanz/data/multilabel_lstm_cnn_predictions"

        if self.n_titles == 551:
            path = os.path.join(base_path, 'top550')
        elif self.n_titles == 7003:
            path = os.path.join(base_path, 'reduced7k')
        else:
            print("Number of job title labels doesn't match 550 or 7000")
            return

        test_batcher = Batcher(batch_size=self.batch_size, step_num=self.max_timesteps, input_data=self.test_inputs,
                                target_data=self.test_targets, n_classes=self.n_labels, n_skills=self.n_skills,
                                max_skills=self.max_skills, skill_data=self.test_skills)

        # Assume that you have 12GB of GPU memory and want to allocate ~4GB:
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True

        with tf.Session(config=gpu_config) as sess:

            sess.run(tf.global_variables_initializer())
            if self.load(sess, self.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
                return

            print(f"Number of batches: {test_batcher.max_batch_num}")
            print(f"Size of dataset: {len(self.test_inputs)}")
            print(f"Batch Size: {self.batch_size}")

            for tb in range(test_batcher.max_batch_num):
                print(f"Batch #{tb}")
                with tf.device("/cpu:0"):
                    test_title_seq, test_seq_lengths, test_target, skills = test_batcher.next()

                pred = sess.run([self.predictions],
                                {
                                    self.titles_input_data: test_title_seq,
                                    self.seq_lengths: test_seq_lengths,
                                    self.targets: test_target,
                                    self.skills_input: skills,
                                    self.dropout: 1.0
                                })

                np.save(os.path.join(path, 'predictions', f'predictions_batch_{tb}.npy'), pred[0])
                np.save(os.path.join(path, 'seq_lengths', f'seq_lengths_batch_{tb}.npy'), test_seq_lengths)

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
