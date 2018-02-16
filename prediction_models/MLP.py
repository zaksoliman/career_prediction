import tensorflow as tf
from time import time
import math
import numpy as np
import os
from statistics import mean


class FeedFowardModel():

    def __init__(self, train_data, test_data, train_targets, test_targets, input_dim, n_labels, learning_rate=0.001,
                 n_epochs=100, batch_size=200, n_layers=2, hiddden_dim=250, use_emb=True, emb_dim=300, use_dropout=True,
                 keep_prob=0.5, store_model=False, ds_name='', store_dir="/data/rali7/Tmp/solimanz/data/models/"):

        self.train_data = train_data
        self.test_data = test_data
        self.train_targets = train_targets
        self.test_targets = test_targets
        self.input_dim = input_dim
        self.n_labels = n_labels
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.use_emb = use_emb
        self.emb_dim = emb_dim
        self.hidden_dim = hiddden_dim
        self.use_dropout = use_dropout
        self.keep_prob = keep_prob
        self.store = store_model
        self.log_interval = 20
        self.hparams = f"=ff_{ds_name}_lr={learning_rate}_batch_size={batch_size}_n_layers={n_layers}_hidden_dim={hiddden_dim}" \
                       f"_use_emb={use_emb}_emb_dim={emb_dim}_dropout={keep_prob}"
        self.checkpoint_dir = os.path.join(store_dir, self.hparams)
        self.build_model()

    def build_model(self):
        self.mle()
        self._loss()
        self._optimize()
        self._acc()
        self.saver = tf.train.Saver()

    def mle(self):

        self.dropout = tf.placeholder(tf.float32, name="dropout_prob")

        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim])

        self.targets = tf.placeholder(dtype=tf.int32, shape=[None, self.n_labels])

        h1 = tf.layers.dropout(tf.layers.dense(self.inputs, self.hidden_dim, tf.nn.relu), self.dropout)
        h2 = tf.layers.dropout(tf.layers.dense(h1, self.hidden_dim, tf.nn.relu), self.dropout)

        # Output
        self.logits = tf.layers.dense(h2, self.n_labels)
        self.pred_softmax = tf.nn.softmax(self.logits)

    def _loss(self):
        with tf.name_scope("xent"):
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=tf.argmax(self.targets, 1)))
            return self.loss

    def _acc(self):
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.targets, 1), tf.argmax(self.pred_softmax, 1)), tf.float32))
        return self.acc

    def _optimize(self):
        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        return self.optimize

    def train(self):
        # Assume that you have 12GB of GPU memory and want to allocate ~4GB:
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        np.random.seed(1234)

        with tf.Session(config=gpu_config) as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            if self.load(sess, self.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

            n_batches = int(math.ceil(len(self.train_data)/self.batch_size))
            print(f"{n_batches} batches to be done")

            for e in range(self.n_epochs):
                print(f"Epoch {e}:")

                start_time = time()
                batch_start = 0
                batch_end = min(self.batch_size, len(self.train_data))

                losses = []
                accs = []

                for b in range(n_batches):
                    with tf.device("/cpu:0"):
                        # get next batch
                        train_batch = self.train_data[batch_start:batch_end]
                        targets_batch = self.train_targets[batch_start:batch_end]

                        batch_start, batch_end = batch_end, min(batch_end + self.batch_size, len(self.train_data)-1)

                    loss, _, acc = sess.run([
                        self.loss,
                        self.optimize,
                        self.acc
                    ],
                        {
                            self.inputs: train_batch,
                            self.targets: targets_batch,
                            self.dropout: self.keep_prob
                        })

                    losses.append(loss)
                    accs.append(acc)

                    # if b % self.log_interval == 0 and b > 0:
                    #     elapsed = time() - start_time
                    #     print(
                    #         f'| epoch {e} | {b}/{n_batches} batches | lr {self.lr} | '
                    #         f'ms/batch {elapsed * 1000 / self.log_interval:.2f} | loss {loss:.3f} | acc: {acc*100:.2f} |')

                if self.store and e % 10 == 0:
                    save_path = self.save(sess, self.checkpoint_dir, e)
                    print("model saved in file: %s" % save_path)


                tst_loss, tst_acc = sess.run([
                    self.loss,
                    self.acc,
                ],
                    {
                        self.inputs: self.test_data,
                        self.targets: self.test_targets,
                        self.dropout: 1.0
                    })

                print(f"Acc on test: {tst_acc*100:.2f} | Acc on train: {sum(accs)/len(accs)*100:.2f}")
                print(f"Loss on test: {tst_loss:.4f} | Loss on train: {sum(losses)/len(losses):.4f}")

                print("Shuffling data for next epoch...")
                zipped = list(zip(self.train_data, self.train_targets))
                np.random.shuffle(zipped)
                self.train_data, self.train_targets = zip(*zipped)

    def test(self):
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        np.random.seed(1234)

        base_path = "/data/rali7/Tmp/solimanz/data/feed_forward/model_predictions"

        if self.n_labels == 550:
            path = os.path.join(base_path, 'top550', self.name)
        elif self.n_labels == 7000:
            path = os.path.join(base_path, 'reduced7000', self.name)
        else:
            print("Number of job title labels doesn't match 550 or 7000")
            return

        with tf.Session(config=gpu_config) as sess:

            sess.run(tf.global_variables_initializer())
            if self.load(sess, self.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

            pred = sess.run([self.pred_softmax],
                {
                    self.inputs: self.test_data,
                    self.targets: self.test_targets,
                    self.dropout: 1.0
                })

            np.save(os.path.join(path, f'predictions.npy'), pred)
            np.save(os.path.join(path, f'targets.npy'), self.test_targets)
            np.save(os.path.join(path, f'inputs.npy'), self.test_data)

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