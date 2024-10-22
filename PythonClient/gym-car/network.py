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

        self.updateOp = [target[i].assign(self.tau * online[i] + (1 - self.tau) * target[i]) for i in range(len(target))]
        self.fullUpdateOp = [target[i].assign(online[i]) for i in range(len(target))]


        init = tf.initialize_all_variables()
        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver(max_to_keep=None)
        if path: # load network
            self.saver.restore(self.sess, path)
        else:
            self.sess.run(init)

        self.writer = tf.summary.FileWriter("/media/yoshi/Seagate Expansion Drive/tensorboard_log6", graph=tf.get_default_graph())

    def saveNetwork(self, path):
        self.saver.save(self.sess, path)

    def buildOnlineNetwork(self):

        with tf.variable_scope("Online"):
            with tf.variable_scope("Actor"):
                self.actor_input["Online"] = tf.placeholder(tf.float32, [None] + [self.trace_length] + self.obs_space,
                                                  name='input_state')

                input_unstacked = tf.unstack(self.actor_input["Online"], axis = 1)
                conv_flat = []

                for i in range(self.trace_length):
                    conv1 = tf.layers.conv2d(input_unstacked[i], 16, 8, [4, 4], activation=tf.nn.relu, name="online_actor_conv_layer_1", reuse=tf.AUTO_REUSE)
                    conv2 = tf.layers.conv2d(conv1, 32, 4, [4, 4], activation=tf.nn.relu, name="online_actor_conv_layer_2", reuse=tf.AUTO_REUSE)
                    conv3 = tf.layers.conv2d(conv2, 64, 3, [1, 1], activation=tf.nn.relu, name="online_actor_conv_layer_3", reuse=tf.AUTO_REUSE)
                    conv_flat.append(tf.layers.flatten(conv3))

                input_restacked = tf.stack(conv_flat, axis = 1)
                post_conv_summary = tf.summary.histogram('Post Conv', input_restacked)

                lstm_cells = tf.contrib.rnn.MultiRNNCell(
                    ([tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(i) for i in self.lstm_layers]))

                encoder_output, encoder_state = tf.nn.dynamic_rnn(lstm_cells, input_restacked, dtype=tf.float32)

                layer = tf.unstack(encoder_output, axis = 1)[-1]
                post_lstm_summary = tf.summary.histogram('Post LSTM', layer)

                for i in range(len(self.actor_layers)):
                    layer = tf.layers.dense(layer, self.actor_layers[i], name = 'FC_layer_' + str(i))
                    layer = tf.layers.batch_normalization(layer, name = 'BN_layer_' + str(i))
                    layer = tf.nn.relu(layer)

                #(throttle1, steering1)
                throttle = tf.layers.dense(layer, 1, name='output_throttle')
                steering = tf.layers.dense(layer, 1, name='output_steering')

                self.action_prob["Online"] = tf.concat([tf.nn.tanh(throttle),tf.nn.tanh(steering)],1)

                throttle_summary = tf.summary.histogram('Target Throttle', throttle)
                steering_summary = tf.summary.histogram('Target Steering', steering)

                self.action_summary["Online"] = tf.summary.merge([throttle_summary, steering_summary, post_conv_summary, post_lstm_summary])

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

                input_unstacked = tf.unstack(self.critic_state_input["Online"], axis=1)
                conv_flat = []

                for i in range(self.trace_length):
                    conv1 = tf.layers.conv2d(input_unstacked[i], 16, 8, [4, 4], activation=tf.nn.relu,
                                             name="online_critic_conv_layer_1", reuse=tf.AUTO_REUSE)
                    conv2 = tf.layers.conv2d(conv1, 32, 4, [4, 4], activation=tf.nn.relu, name="online_critic_conv_layer_2",
                                             reuse=tf.AUTO_REUSE)
                    conv3 = tf.layers.conv2d(conv2, 32, 4, [1, 1], activation=tf.nn.relu, name="online_critic_conv_layer_3",
                                             reuse=tf.AUTO_REUSE)
                    conv_flat.append(tf.layers.flatten(conv3))

                input_restacked = tf.stack(conv_flat, axis=1)
                lstm_cells = tf.contrib.rnn.MultiRNNCell(
                    ([tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(i) for i in self.lstm_layers]))

                encoder_output, encoder_state = tf.nn.dynamic_rnn(lstm_cells, input_restacked, dtype=tf.float32)

                s_layer = tf.unstack(encoder_output, axis=1)[-1]
                for i in range(len(self.critic_s_layers) - 1):
                    s_layer = tf.layers.dense(s_layer, self.critic_s_layers[i], name='online_critic_FC_slayer_' + str(i))
                    s_layer = tf.layers.batch_normalization(s_layer, name = 'BN_slayer_' + str(i))
                    s_layer = tf.nn.relu(s_layer)
                s_layer = tf.layers.dense(s_layer, self.critic_s_layers[-1], name='online_critic_FC_slayer_-1' + str(i))

                a_layer = self.critic_action_input["Online"]
                for i in range(len(self.critic_a_layers) - 1):
                    a_layer = tf.layers.dense(a_layer, self.critic_a_layers[i], name='online_critic_FC_alayer_' + str(i))
                    a_layer = tf.layers.batch_normalization(a_layer, name = 'BN_alayer_' + str(i))
                    a_layer = tf.nn.relu(a_layer)
                a_layer = tf.layers.dense(a_layer, self.critic_a_layers[-1], name='online_critic_FC_alayer_-1')

                layer = tf.nn.relu(s_layer + a_layer)

                self.state_values["Online"] = tf.layers.dense(layer, self.action_space[0], name='output_layer')

                self.value_summary["Online"] = tf.summary.histogram('Online State Values', self.state_values["Online"])

                self.predicted_q_value = tf.placeholder(tf.float32, [None] + self.action_space)
                self.critic_loss = tf.losses.mean_squared_error(self.predicted_q_value, self.state_values["Online"])
                self.critic_optimize = tf.train.AdamOptimizer(self.critic_lr).minimize(self.critic_loss)
                self.action_grads = tf.gradients(self.state_values["Online"], self.critic_action_input["Online"])


    def buildTargetNetwork(self):

        with tf.variable_scope("Target"):
            with tf.variable_scope("Actor"):
                self.actor_input["Target"] = tf.placeholder(tf.float32, [None] + [self.trace_length] + self.obs_space,
                                                       name='input_state')
                im_sum_1 = tf.summary.image('input_image_1', self.actor_input["Target"][:, 0, :, :], 1)
                im_sum_2 = tf.summary.image('input_image_2', self.actor_input["Target"][:, 1, :, :], 1)
                im_sum_3 = tf.summary.image('input_image_3', self.actor_input["Target"][:, 2, :, :], 1)
                im_sum_4 = tf.summary.image('input_image_4', self.actor_input["Target"][:, 3, :, :], 1)
                im_sum_5 = tf.summary.image('input_image_5', self.actor_input["Target"][:, 4, :, :], 1)

                input_unstacked = tf.unstack(self.actor_input["Target"], axis = 1)
                conv_flat = []

                for i in range(self.trace_length):
                    conv1 = tf.layers.conv2d(input_unstacked[i], 16, 8, [4, 4], activation=tf.nn.relu, name="tar_actor_conv_layer_1", reuse=tf.AUTO_REUSE)
                    post_conv_1 = tf.summary.image('post_conv_1', conv1[:,:,:,0,None], 1)

                    conv2 = tf.layers.conv2d(conv1, 32, 4, [4, 4], activation=tf.nn.relu, name="tar_actor_conv_layer_2", reuse=tf.AUTO_REUSE)
                    post_conv_2 = tf.summary.image('post_conv_2', conv2[:,:,:,0,None], 1)

                    conv3 = tf.layers.conv2d(conv2, 64, 3, [1, 1], activation=tf.nn.relu, name="tar_actor_conv_layer_3", reuse=tf.AUTO_REUSE)
                    post_conv_3 = tf.summary.image('post_conv_3', conv3[:,:,:,0,None], 1)
                    conv_flat.append(tf.layers.flatten(conv3))

                self.image_summary = tf.summary.merge([im_sum_1, im_sum_2, im_sum_3, im_sum_4, im_sum_5, post_conv_1, post_conv_2, post_conv_3])

                input_restacked = tf.stack(conv_flat, axis = 1)
                lstm_cells = tf.contrib.rnn.MultiRNNCell(
                    ([tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(i) for i in self.lstm_layers]))

                encoder_output, encoder_state = tf.nn.dynamic_rnn(lstm_cells, input_restacked, dtype=tf.float32)

                layer = tf.unstack(encoder_output, axis = 1)[-1]

                for i in range(len(self.actor_layers)):
                    layer = tf.layers.dense(layer, self.actor_layers[i], name='tar_actor_FC_layer_' + str(i))
                    layer = tf.layers.batch_normalization(layer, name = 'BN_layer_' + str(i))
                    layer = tf.nn.relu(layer)

                # (throttle1, steering1, brake1)
                throttle = tf.layers.dense(layer, 1, name='output_throttle')
                steering = tf.layers.dense(layer, 1, name='output_steering')

                self.action_prob["Target"] = tf.concat([tf.nn.tanh(throttle),tf.nn.tanh(steering)],1)

                throttle_summary = tf.summary.histogram('Target Throttle', self.action_prob["Target"][:,0])
                steering_summary = tf.summary.histogram('Target Steering', self.action_prob["Target"][:,1])

                self.action_summary["Target"] = tf.summary.merge([throttle_summary, steering_summary])

            with tf.variable_scope("Critic"):
                self.critic_state_input["Target"] = tf.placeholder(tf.float32, [None] + [self.trace_length] + self.obs_space,
                                                                   name='input_state')
                self.critic_action_input["Target"] = tf.placeholder(tf.float32, [None] + self.action_space,
                                                                    name='input_action')

                input_unstacked = tf.unstack(self.critic_state_input["Target"], axis = 1)
                conv_flat = []

                for i in range(self.trace_length):
                    conv1 = tf.layers.conv2d(input_unstacked[i], 16, 8, [4, 4], activation=tf.nn.relu, name="tar_critic_conv_layer_1", reuse=tf.AUTO_REUSE)
                    conv2 = tf.layers.conv2d(conv1, 32, 4, [4, 4], activation=tf.nn.relu, name="tar_critic_conv_layer_2", reuse=tf.AUTO_REUSE)
                    conv3 = tf.layers.conv2d(conv2, 32, 4, [1, 1], activation=tf.nn.relu, name="tar_critic_conv_layer_3",
                                             reuse=tf.AUTO_REUSE)
                    conv_flat.append(tf.layers.flatten(conv3))

                input_restacked = tf.stack(conv_flat, axis = 1)
                lstm_cells = tf.contrib.rnn.MultiRNNCell(
                    ([tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(i) for i in self.lstm_layers]))

                encoder_output, encoder_state = tf.nn.dynamic_rnn(lstm_cells, input_restacked, dtype=tf.float32)

                s_layer = tf.unstack(encoder_output, axis=1)[-1]
                for i in range(len(self.critic_s_layers) - 1):
                    s_layer = tf.layers.dense(s_layer, self.critic_s_layers[i], name='tar_critic_FC_slayer_' + str(i))
                    s_layer = tf.layers.batch_normalization(s_layer, name = 'BN_slayer_' + str(i))
                    s_layer = tf.nn.relu(s_layer)
                s_layer = tf.layers.dense(s_layer, self.critic_s_layers[-1], name='tar_critic_FC_slayer_-1')

                a_layer = self.critic_action_input["Target"]
                for i in range(len(self.critic_a_layers) - 1):
                    a_layer = tf.layers.dense(a_layer, self.critic_a_layers[i], name='tar_critic_FC_alayer_' + str(i))
                    a_layer = tf.layers.batch_normalization(a_layer, name = 'BN_alayer_' + str(i))
                    a_layer = tf.nn.relu(a_layer)
                a_layer = tf.layers.dense(a_layer, self.critic_a_layers[-1], name='tar_critic_FC_alayer_-1')

                layer = tf.nn.relu(s_layer + a_layer)

                self.state_values["Target"] = tf.layers.dense(layer, self.action_space[0], name='output_layer')
                self.value_summary["Target"] = tf.summary.histogram('Target State Values', self.state_values["Target"])

    def updateTargetNetwork(self):
        self.sess.run(self.updateOp)

    def initTargetNetwork(self):
        self.sess.run(self.fullUpdateOp)


    ###ACTOR###
    def getOnlineActionProb(self, input_state, i = None):
        feed_dict = {self.actor_input["Online"]: input_state}

        action_prob, summary = self.sess.run([self.action_prob["Online"], self.action_summary["Online"]], feed_dict=feed_dict)
        if i:
            self.writer.add_summary(summary, i)
        
        return action_prob
        
    def getTargetActionProb(self, input_state, i = None):
        feed_dict = {self.actor_input["Target"]: input_state}

        action_prob, action_summary, image_summary = self.sess.run([self.action_prob["Target"], self.action_summary["Target"], self.image_summary], feed_dict=feed_dict)
        if i:
            self.writer.add_summary(action_summary, i)
            self.writer.add_summary(image_summary, i)

        return action_prob

    def trainActor(self, inputs, gradients):
        feed_dict = {self.actor_input["Online"]: inputs,
                    self.action_gradient: gradients}

        self.sess.run(self.actor_optimize, feed_dict = feed_dict)


    ###CRITIC###
    def getOnlineStateValues(self, input_state, input_action, i = None):
        feed_dict = {self.critic_state_input["Online"]: input_state,
                     self.critic_action_input["Online"]: input_action}

        state_values, summary = self.sess.run([self.state_values["Online"], self.value_summary["Online"]], feed_dict=feed_dict)
        if i:
            self.writer.add_summary(summary, i)
    
        return state_values

    def getTargetStateValues(self, input_state, input_action, i = None):
        feed_dict = {self.critic_state_input["Target"]: input_state,
                     self.critic_action_input["Target"]: input_action}

        state_values, summary = self.sess.run([self.state_values["Target"], self.value_summary["Target"]], feed_dict=feed_dict)
        if i:
            self.writer.add_summary(summary, i)

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
