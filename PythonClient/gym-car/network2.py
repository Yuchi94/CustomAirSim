import tensorflow as tf
import numpy as np

class DDPG():
    def __init__(self,
                 actor_layers,
                 critic_a_layers,
                 critic_s_layers,
                 lstm_layers,
                 trace_length,
                 actor_lr,
                 critic_lr,
                 obs_space,
                 action_space,
                 tau,
                 batch_size):

        self.actor_layers = actor_layers
        self.critic_a_layers = critic_a_layers
        self.critic_s_layers = critic_s_layers
        self.lstm_layers = lstm_layers
        self.trace_length = trace_length
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.obs_space = obs_space
        self.action_space = action_space
        self.tau = tau
        self.batch_size = batch_size

    def buildNetwork(self, path = None):
        if path: # load network
            pass
        else:
            self.actor_input = {}
            self.action_prob = {}
            self.action_summary = {}
            self.critic_state_input = {}
            self.critic_action_input = {}
            self.state_values = {}
            self.value_summary = {}

            self.buildOnlineNetwork()
            self.buildTargetNetwork()

            online = tf.trainable_variables('Online')
            target = tf.trainable_variables('Target')

            print(online)
            print(target)

            self.updateOp = [target[i].assign(self.tau * online[i] + (1 - self.tau) * target[i]) for i in range(len(target))]
            self.fullUpdateOp = [target[i].assign(online[i]) for i in range(len(target))]


            init = tf.initialize_all_variables()
            self.sess = tf.InteractiveSession()
            # self.saver = tf.train.Saver()
            self.sess.run(init)

            self.writer = tf.summary.FileWriter("/media/yoshi/Seagate Expansion Drive/tensorboard_log", graph=tf.get_default_graph())

    def buildOnlineNetwork(self):

        with tf.variable_scope("Online"):
            with tf.variable_scope("Actor"):
                self.actor_input["Online"] = tf.placeholder(tf.float32, [None] + [self.trace_length] + self.obs_space,
                                                  name='input_state')

                # lstm_cells = tf.contrib.rnn.MultiRNNCell(
                #     ([tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(i) for i in self.lstm_layers]))
                #
                # encoder_output, encoder_state = tf.nn.dynamic_rnn(lstm_cells, self.actor_input["Online"], dtype=tf.float32)
                #
                # layer = tf.unstack(encoder_output, axis = 1)[-1]

                input_unstacked = tf.unstack(self.actor_input["Online"], axis=1)

                conv_flat = []
                for i in range(self.trace_length):
                    conv1 = tf.layers.conv2d(input_unstacked[i], 16, 8, [4, 4], activation=tf.nn.relu,
                                             name="online_critic_conv_layer_1", reuse=tf.AUTO_REUSE)
                    conv2 = tf.layers.conv2d(conv1, 32, 4, [4, 4], activation=tf.nn.relu,
                                             name="online_critic_conv_layer_2",
                                             reuse=tf.AUTO_REUSE)
                    conv_flat.append(tf.layers.flatten(conv2))

                input_restacked = tf.stack(conv_flat, axis = 1)

                lstm_cells = tf.contrib.rnn.MultiRNNCell(
                    ([tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(i) for i in self.lstm_layers]))

                encoder_output, encoder_state = tf.nn.dynamic_rnn(lstm_cells, input_restacked, dtype=tf.float32)
                layer = tf.unstack(encoder_output, axis = 1)[-1]

                for i in range(len(self.actor_layers)):
                    layer = tf.layers.dense(layer, self.actor_layers[i])
                    layer = tf.layers.batch_normalization(layer)
                    layer = tf.nn.relu(layer)

                self.action_prob["Online"] = tf.concat([tf.layers.dense(layer, 1, activation=tf.nn.sigmoid), tf.layers.dense(layer, 1, activation=tf.nn.tanh)], axis = 1)

                self.action_gradient = tf.placeholder(tf.float32, [None] + self.action_space)
                self.unnormalized_actor_gradients = tf.gradients(self.action_prob["Online"],
                                                                 tf.trainable_variables('Online/Actor'),
                                                                 -self.action_gradient)
                self.actor_gradients = list(
                    map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))
                self.actor_optimize = tf.train.AdamOptimizer(self.actor_lr). \
                    apply_gradients(zip(self.actor_gradients, tf.trainable_variables('Online/Actor')))

            with tf.variable_scope("Critic"):
                self.critic_state_input["Online"] = tf.placeholder(tf.float32,
                                                                   [None] + [self.trace_length] + self.obs_space,
                                                                   name='input_state')
                self.critic_action_input["Online"] = tf.placeholder(tf.float32, [None] + self.action_space,
                                                                    name='input_action')

                # lstm_cells = tf.contrib.rnn.MultiRNNCell(
                #     ([tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(i) for i in self.lstm_layers]))
                #
                # encoder_output, encoder_state = tf.nn.dynamic_rnn(lstm_cells, self.critic_state_input["Online"], dtype=tf.float32)
                #
                # s_layer = tf.unstack(encoder_output, axis=1)[-1]

                input_unstacked = tf.unstack(self.critic_state_input["Online"], axis=1)

                conv_flat = []
                for i in range(self.trace_length):
                    conv1 = tf.layers.conv2d(input_unstacked[i], 16, 8, [4, 4], activation=tf.nn.relu,
                                             name="online_critic_conv_layer_1", reuse=tf.AUTO_REUSE)
                    conv2 = tf.layers.conv2d(conv1, 32, 4, [4, 4], activation=tf.nn.relu,
                                             name="online_critic_conv_layer_2",
                                             reuse=tf.AUTO_REUSE)
                    conv_flat.append(tf.layers.flatten(conv2))

                input_restacked = tf.stack(conv_flat, axis = 1)

                lstm_cells = tf.contrib.rnn.MultiRNNCell(
                    ([tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(i) for i in self.lstm_layers]))

                encoder_output, encoder_state = tf.nn.dynamic_rnn(lstm_cells, input_restacked, dtype=tf.float32)
                s_layer = tf.unstack(encoder_output, axis = 1)[-1]

                for i in range(len(self.critic_s_layers) - 1):
                    s_layer = tf.layers.dense(s_layer, self.critic_s_layers[i])
                    s_layer = tf.layers.batch_normalization(s_layer)
                    s_layer = tf.nn.relu(s_layer)
                s_layer = tf.layers.dense(s_layer, self.critic_s_layers[-1])

                a_layer = self.critic_action_input["Online"]
                for i in range(len(self.critic_a_layers) - 1):
                    a_layer = tf.layers.dense(a_layer, self.critic_a_layers[i])
                    a_layer = tf.layers.batch_normalization(a_layer)
                    a_layer = tf.nn.relu(a_layer)
                a_layer = tf.layers.dense(a_layer, self.critic_a_layers[-1])

                layer = tf.nn.relu(s_layer + a_layer)

                self.state_values["Online"] = tf.layers.dense(layer, self.action_space[0])

                self.predicted_q_value = tf.placeholder(tf.float32, [None] + self.action_space)
                self.critic_loss = tf.losses.mean_squared_error(self.predicted_q_value, self.state_values["Online"])
                self.critic_optimize = tf.train.AdamOptimizer(self.critic_lr).minimize(self.critic_loss)
                self.action_grads = tf.gradients(self.state_values["Online"], self.critic_action_input["Online"])


    def buildTargetNetwork(self):

        with tf.variable_scope("Target"):
            with tf.variable_scope("Actor"):
                self.actor_input["Target"] = tf.placeholder(tf.float32, [None] +  [self.trace_length] + self.obs_space,
                                                  name='input_state')
                # lstm_cells = tf.contrib.rnn.MultiRNNCell(
                #     ([tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(i) for i in self.lstm_layers]))
                #
                # encoder_output, encoder_state = tf.nn.dynamic_rnn(lstm_cells, self.actor_input["Target"], dtype=tf.float32)
                #
                # layer = tf.unstack(encoder_output, axis = 1)[-1]

                input_unstacked = tf.unstack(self.actor_input["Target"], axis=1)

                conv_flat = []
                for i in range(self.trace_length):
                    conv1 = tf.layers.conv2d(input_unstacked[i], 16, 8, [4, 4], activation=tf.nn.relu,
                                             name="online_critic_conv_layer_1", reuse=tf.AUTO_REUSE)
                    conv2 = tf.layers.conv2d(conv1, 32, 4, [4, 4], activation=tf.nn.relu,
                                             name="online_critic_conv_layer_2",
                                             reuse=tf.AUTO_REUSE)
                    conv_flat.append(tf.layers.flatten(conv2))

                input_restacked = tf.stack(conv_flat, axis = 1)

                lstm_cells = tf.contrib.rnn.MultiRNNCell(
                    ([tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(i) for i in self.lstm_layers]))

                encoder_output, encoder_state = tf.nn.dynamic_rnn(lstm_cells, input_restacked, dtype=tf.float32)
                layer = tf.unstack(encoder_output, axis = 1)[-1]

                for i in range(len(self.actor_layers)):
                    layer = tf.layers.dense(layer, self.actor_layers[i])
                    layer = tf.layers.batch_normalization(layer)
                    layer = tf.nn.relu(layer)

                self.action_prob["Target"] = tf.concat([tf.layers.dense(layer, 1, activation=tf.nn.sigmoid), tf.layers.dense(layer, 1, activation=tf.nn.tanh)], axis = 1)


            with tf.variable_scope("Critic"):
                self.critic_state_input["Target"] = tf.placeholder(tf.float32, [None] + [self.trace_length] + self.obs_space)
                self.critic_action_input["Target"] = tf.placeholder(tf.float32, [None] + self.action_space)

                # lstm_cells = tf.contrib.rnn.MultiRNNCell(
                #     ([tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(i) for i in self.lstm_layers]))
                #
                # encoder_output, encoder_state = tf.nn.dynamic_rnn(lstm_cells, self.critic_state_input["Target"], dtype=tf.float32)
                #
                # s_layer = tf.unstack(encoder_output, axis=1)[-1]

                input_unstacked = tf.unstack(self.critic_state_input["Target"], axis=1)

                conv_flat = []
                for i in range(self.trace_length):
                    conv1 = tf.layers.conv2d(input_unstacked[i], 16, 8, [4, 4], activation=tf.nn.relu,
                                             name="online_critic_conv_layer_1", reuse=tf.AUTO_REUSE)
                    conv2 = tf.layers.conv2d(conv1, 32, 4, [4, 4], activation=tf.nn.relu,
                                             name="online_critic_conv_layer_2",
                                             reuse=tf.AUTO_REUSE)
                    conv_flat.append(tf.layers.flatten(conv2))

                input_restacked = tf.stack(conv_flat, axis = 1)

                lstm_cells = tf.contrib.rnn.MultiRNNCell(
                    ([tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(i) for i in self.lstm_layers]))

                encoder_output, encoder_state = tf.nn.dynamic_rnn(lstm_cells, input_restacked, dtype=tf.float32)
                s_layer = tf.unstack(encoder_output, axis = 1)[-1]


                for i in range(len(self.critic_s_layers) - 1):
                    s_layer = tf.layers.dense(s_layer, self.critic_s_layers[i])
                    s_layer = tf.layers.batch_normalization(s_layer)
                    s_layer = tf.nn.relu(s_layer)
                s_layer = tf.layers.dense(s_layer, self.critic_s_layers[-1])

                a_layer = self.critic_action_input["Target"]
                for i in range(len(self.critic_a_layers) - 1):
                    a_layer = tf.layers.dense(a_layer, self.critic_a_layers[i])
                    a_layer = tf.layers.batch_normalization(a_layer)
                    a_layer = tf.nn.relu(a_layer)
                a_layer = tf.layers.dense(a_layer, self.critic_a_layers[-1])

                layer = tf.nn.relu(s_layer + a_layer)

                self.state_values["Target"] = tf.layers.dense(layer, self.action_space[0])

    def updateTargetNetwork(self):
        self.sess.run(self.updateOp)

    def initTargetNetwork(self):
        self.sess.run(self.fullUpdateOp)


    ###ACTOR###
    def getOnlineActionProb(self, input_state, i = None):
        feed_dict = {self.actor_input["Online"]: input_state}

        action_prob = self.sess.run(self.action_prob["Online"], feed_dict=feed_dict)

        return action_prob
        
    def getTargetActionProb(self, input_state, i = None):
        feed_dict = {self.actor_input["Target"]: input_state}

        action_prob  = self.sess.run(self.action_prob["Target"], feed_dict=feed_dict)


        return action_prob

    def trainActor(self, inputs, gradients):
        feed_dict = {self.actor_input["Online"]: inputs,
                    self.action_gradient: gradients}

        self.sess.run(self.actor_optimize, feed_dict = feed_dict)


    ###CRITIC###
    def getOnlineStateValues(self, input_state, input_action, i = None):
        feed_dict = {self.critic_state_input["Online"]: input_state,
                     self.critic_action_input["Online"]: input_action}

        state_values = self.sess.run(self.state_values["Online"], feed_dict=feed_dict)
    
        return state_values

    def getTargetStateValues(self, input_state, input_action, i = None):
        feed_dict = {self.critic_state_input["Target"]: input_state,
                     self.critic_action_input["Target"]: input_action}

        state_values = self.sess.run(self.state_values["Target"], feed_dict=feed_dict)

        return state_values

    def trainCritic(self, input_state, input_action, predicted_q_value):
        feed_dict = {self.critic_state_input["Online"]: input_state,
                     self.critic_action_input["Online"]: input_action,
                     self.predicted_q_value: predicted_q_value}

        self.sess.run(self.critic_optimize, feed_dict=feed_dict)

    def getActionGradients(self, input_state, input_action): #TODO merge into getOnlineStateValues
        feed_dict = {self.critic_state_input["Online"]: input_state,
                     self.critic_action_input["Online"]: input_action}

        return self.sess.run(self.action_grads, feed_dict=feed_dict)
