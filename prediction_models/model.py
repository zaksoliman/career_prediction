import tensorflow as tf
import os, random, string
from batcher import Batcher
import loader
from time import time


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"


class Model:
    def __init__(self, config):

        # Place Holders
        #self.keep_prob = tf.placeholder(tf.float32)
        self.job_input_data = tf.placeholder(tf.int32, [None, config.job_length], name="job_input_data")

        batch_size = self.job_input_data.shape[0].value
        if batch_size is None:
            batch_size = config.batch_size

        self.job_length = tf.placeholder(tf.int32, [None], name="job_length")
        self.target = tf.placeholder(tf.int32, [batch_size])

        with tf.device("/cpu:0"):
            job_embedding = tf.Variable(tf.eye(config.job_num), trainable=False)

            #             job_embedding = tf.get_variable(name="job_embedding", shape=[config.job_num, config.emb_dim]
            #                                 ,dtype=data_type(), initializer=tf.constant_initializer(embeddings_matrix)
            #                                  , trainable=True)

            self.job_input = tf.nn.embedding_lookup(job_embedding, self.job_input_data)

        if config.encoder == "averaging":
            self.representation = tf.add_n(self.job_input)

        elif config.encoder == "rnn":
            if config.rnn_type == 'LSTM':
                cell = tf.nn.rnn_cell.LSTMCell(config.hidden_dim)
            elif config.rnn_type == 'GRU':
                cell = tf.nn.rnn_cell.GRUCell(config.hidden_dim)
            elif config.rnn_type == 'RNN':
                cell = tf.nn.rnn_cell.BasicRNNCell(config.hidden_dim)

            #cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

            if config.use_attention:
                outputs, _ = tf.nn.dynamic_rnn(cell, self.job_input, sequence_length=self.job_length, dtype=tf.float32,
                                               parallel_iterations=1024)
                birnn = tf.transpose(outputs, perm=[1, 0, 2])
                self.representation, _ = self.attentive_sum(birnn, input_dim=config.hidden_dim,
                                                            hidden_dim=config.att_dim)

            else:
                _, self.representation = tf.nn.dynamic_rnn(cell, self.job_input, sequence_length=self.job_length,
                                                           dtype=tf.float32, parallel_iterations=1024)

            if config.rnn_type == 'LSTM':
                self.representation = self.representation.h

        elif config.encoder == "biRnn":

            if config.rnn_type == 'LSTM':
                cell_fw = tf.nn.rnn_cell.LSTMCell(config.hidden_dim)
                cell_bw = tf.nn.rnn_cell.LSTMCell(config.hidden_dim)
            elif config.rnn_type == 'GRU':
                cell_fw = tf.nn.rnn_cell.GRUCell(config.hidden_dim)
                cell_bw = tf.nn.rnn_cell.GRUCell(config.hidden_dim)
            elif config.rnn_type == 'RNN':
                cell_fw = tf.nn.rnn_cell.BasicRNNCell(config.hidden_dim)
                cell_bw = tf.nn.rnn_cell.BasicRNNCell(config.hidden_dim)

            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw,
                                                    output_keep_prob=self.keep_prob)  # input_keep_prob=self.keep_prob,
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=self.keep_prob)

            if config.use_attention:
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.job_input,
                                                             sequence_length=self.job_length, dtype=tf.float32,
                                                             parallel_iterations=1024)

                birnn = tf.transpose(tf.concat([outputs[0], outputs[1]], 2), perm=[1, 0, 2])
                self.representation, _ = self.attentive_sum(birnn, input_dim=config.hidden_dim * 2,
                                                            hidden_dim=config.att_dim)

            else:
                _, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.job_input,
                                                                   sequence_length=self.job_length, dtype=tf.float32,
                                                                   parallel_iterations=1024)
                fwd_out_all, bwd_out_all = output_states

                if config.rnn_type == 'LSTM':
                    fwd_out_all = fwd_out_all.h
                    bwd_out_all = bwd_out_all.h

                self.representation = tf.concat([fwd_out_all, bwd_out_all], 1)

        self.logit = tf.layers.dense(self.representation, config.job_num, activation=None,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.contrib.layers.xavier_initializer())

        # Loss Function
        tvars = tf.trainable_variables()
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logit, labels=self.target))

        # regularization
        self.loss += tf.add_n([tf.nn.l2_loss(v) for v in tvars if "Bias" not in v.name]) * config.l2_norm

        optimizer = tf.train.AdamOptimizer(config.lr)  # GradientDescentOptimizer RMSPropOptimizer
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), config.max_grad_norm)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        #         self.train_op = optimizer.minimize(self.loss)

        self.distribution = tf.nn.softmax(self.logit)
        self.correct_pred = tf.equal(tf.cast(tf.argmax(self.distribution, 1), tf.int32), self.target)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


def weight_variable(shape):
    return tf.get_variable(name=id_generator(), shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def id_generator(size=3, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def attentive_sum(inputs, input_dim, hidden_dim):
    with tf.variable_scope("attention"):
        seq_length = inputs.shape[0].value
        W = weight_variable((input_dim, hidden_dim))
        U = weight_variable((hidden_dim, 1))
        tf.get_variable_scope().reuse_variables()
        temp1 = [tf.nn.tanh(tf.matmul(inputs[i], W)) for i in range(seq_length)]
        temp2 = [tf.matmul(temp1[i], U) for i in range(seq_length)]

        pre_activations = tf.concat(temp2, 1)

        attentions = tf.split(tf.nn.softmax(pre_activations), seq_length, 1)
        weighted_inputs = [tf.multiply(inputs[i], attentions[i]) for i in range(seq_length)]
        output = tf.add_n(weighted_inputs)

    return output, attentions


class TaggerConfigEnglish(object):
    data_path = "/data/rali7/Tmp/solimanz/data/datasets/title_seq.json"

    # Hyper params

    keep_prob = 0.5
    job_length = 32
    batch_size = 100
    job_num = 100000
    hidden_dim = 128
    emb_dim = 50
    encoder = "rnn"  # averaging rnn biRnn
    rnn_type = 'LSTM'
    l2_norm = 0.001
    step_num = 32
    use_attention = False
    att_dim = 100
    lr = 0.1
    restore = None  # "/u/ghaddara/workspace/TensorFlow/model"
    max_grad_norm = 5
    n_epochs = 50
    log_interval = 2000
    max_train_batches = 10


def train(config):
    id_to_job, data_train, data_test = loader.load_data(config)

    config.target_num = 4
    config.job_num = len(id_to_job)

    print("Creating batchers")
    train_batcher = Batcher(config, data_train)
    test_batcher = Batcher(config, data_test)

    model = Model(config)

    # Assume that you have 12GB of GPU memory and want to allocate ~4GB:
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    with tf.Session(config=gpu_config) as sess:

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        if config.restore is not None:
            saver.restore(sess, 'model.ckpt')
            print("model restored")

        for e in range(config.n_epochs):
            avg_cost = 0.
            avg_acc = 0.
            epoch_start_time = time()
            start_time = time()
            batch = 0
            for _ in range(train_batcher.max_batch_num):
                with tf.device("/cpu:0"):
                    job_input_data, job_length, target = train_batcher.next()

                cost, _, acc = sess.run([model.loss, model.train_op, model.accuracy],
                                        {
                                            model.job_input_data: job_input_data,
                                            model.job_length: job_length,
                                            model.target: target
                                            #model.keep_prob: config.keep_prob
                                        })

                # Compute average loss
                avg_cost += cost
                avg_acc += acc

                if batch % config.log_interval == 0 and batch > 0:
                    cur_loss = avg_cost / batch  # config.log_interval
                    cur_acc = avg_acc / batch
                    elapsed = time.time() - start_time
                    print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:1.5f} | ms/batch {:5.2f} | '
                          'loss {:5.6f} | ppl {:8.2f}'.format(
                        e, batch, config.max_train_batches // config.batch_size, config.lr,
                                  elapsed * 1000 / config.log_interval, cur_loss, cur_acc))

                    start_time = time.time()

                batch += 1
                if config.max_train_batches != -1 and batch == config.max_train_batches:
                    break
            print("Epoch:", '%04d' % (e + 1), "cost=", "{:.9f}".format(avg_cost), "Accuracy=", "{:.9f}".format(avg_acc))

            for _ in range(test_batcher.max_batch_num):
                with tf.device("/cpu:0"):
                    job_input_data, job_length, target = test_batcher.next()

                cost, acc = sess.run([model.loss, model.accuracy],
                                     {model.job_input_data: job_input_data,
                                      model.job_length: job_length,
                                      model.target: target
                                      # model.keep_prob: 1.0
                                     })

            #                     if e % 10 == 0:
            #                         save_path = saver.save(sess, "model/model.ckpt")
            #                 print("model saved in file: %s" % save_path)

            print('Cost on test:', cost, 'Accuracy on dev:', acc)


def main():
    config = TaggerConfigEnglish()
    train(config)


if __name__ == "__main__":
    main()
