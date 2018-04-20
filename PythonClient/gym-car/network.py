import tensorflow as tf
import numpy as np

class DDPG():
    def __init__(self,
                 actor_layers,
                 critic_a_layers,
                 critic_s_layers,
                 critic_layers,
                 actor_lr,
                 critic_lr,
                 obs_space,
                 action_space,
                 tau,
                 batch_size):
        self.actor_layers = actor_layers
        self.critic_a_layers = critic_a_layers
        self.critic_s_layers = critic_s_layers
        self.critic_layers = critic_layers
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
            self.buildActorNetwork()
            self.buildCriticNetwork()

            online = tf.trainable_variables('Online')
            target = tf.trainable_variables('Target')

            self.updateOp = [target[i].assign(self.tau * online[i] + (1 - self.tau) * target[i]) for i in range(len(target))]

            init = tf.initialize_all_variables()
            self.sess = tf.InteractiveSession()
            self.saver = tf.train.Saver()
            self.sess.run(init)


    def buildActorNetwork(self):
        self.actor_input = {}; self.action_prob = {}
        with tf.variable_scope("Online"):
            with tf.variable_scope("Actor"):
                self.actor_input["Online"] = tf.placeholder(tf.float32, [None] + self.obs_space,
                                                  name='input_state')

                conv1 = tf.layers.conv2d(self.actor_input["Online"], 16, 8, [4, 4], activation=tf.nn.relu, name="conv_layer_1")
                conv2 = tf.layers.conv2d(conv1, 32, 4, [2, 2], activation=tf.nn.relu, name="conv_layer_2")
                conv_flat = tf.layers.flatten(conv2, name="layer_flatten")

                layer = conv_flat
                for i in range(len(self.actor_layers)):
                    layer = tf.layers.dense(layer, self.actor_layers[i], tf.nn.relu, name = 'FC_layer_' + str(i))

                #(throttle1, steering1, brake1)
                self.action_prob["Online"] = tf.concat([tf.layers.dense(layer, 1, tf.nn.sigmoid, name='output_throttle'),
                                     tf.layers.dense(layer, 1, tf.nn.tanh, name='output_steering'),
                                     tf.layers.dense(layer, 1, tf.nn.sigmoid, name='output_brake')], 1)

                self.action_gradient = tf.placeholder(tf.float32, [None] + self.action_space)
                self.unnormalized_actor_gradients = tf.gradients(self.action_prob["Online"],
                                                                 tf.trainable_variables('Online/Actor'),
                                                                 -self.action_gradient)
                self.actor_gradients = list(
                    map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))
                self.actor_optimize = tf.train.AdamOptimizer(self.actor_lr). \
                    apply_gradients(zip(self.actor_gradients, tf.trainable_variables('Online/Actor')))

        with tf.variable_scope("Target"):
            with tf.variable_scope("Actor"):
                self.actor_input["Target"] = tf.placeholder(tf.float32, [None] + self.obs_space,
                                                       name='input_state')

                conv1 = tf.layers.conv2d(self.actor_input["Target"], 16, 8, [4, 4], activation=tf.nn.relu, name="conv_layer_1")
                conv2 = tf.layers.conv2d(conv1, 32, 4, [2, 2], activation=tf.nn.relu, name="conv_layer_2")
                conv_flat = tf.layers.flatten(conv2, name="layer_flatten")

                layer = conv_flat
                for i in range(len(self.actor_layers)):
                    layer = tf.layers.dense(layer, self.actor_layers[i], tf.nn.relu, name='FC_layer_' + str(i))

                # (throttle1, steering1, brake1)
                self.action_prob["Target"] = tf.concat([tf.layers.dense(layer, 1, tf.nn.sigmoid, name='output_throttle'),
                                                   tf.layers.dense(layer, 1, tf.nn.tanh, name='output_steering'),
                                                   tf.layers.dense(layer, 1, tf.nn.sigmoid, name='output_brake')],
                                                  1)

            # self.action_summary = tf.summary.histogram('Action Probabilities', self.action_prob)

    def buildCriticNetwork(self):
        self.critic_state_input = {}; self.critic_action_input  = {}; self.state_values = {}
        with tf.variable_scope("Online"):
            with tf.variable_scope("Critic"):
                self.critic_state_input["Online"] = tf.placeholder(tf.float32, [None] + self.obs_space,
                                                  name='input_state')
                self.critic_action_input["Online"] = tf.placeholder(tf.float32, [None] + self.action_space,
                                                  name='input_action')

                conv1 = tf.layers.conv2d(self.critic_state_input["Online"], 16, 8, [4, 4], activation=tf.nn.relu, name="conv_layer_1")
                conv2 = tf.layers.conv2d(conv1, 32, 4, [2, 2], activation=tf.nn.relu, name="conv_layer_2")
                conv_flat = tf.layers.flatten(conv2, name="layer_flatten")

                s_layer = conv_flat
                for i in range(len(self.critic_s_layers)):
                    s_layer = tf.layers.dense(s_layer, self.critic_s_layers[i], tf.nn.relu, name = 'state_layer_' + str(i))

                a_layer = self.critic_action_input["Online"]
                for i in range(len(self.critic_a_layers)):
                    a_layer = tf.layers.dense(a_layer, self.critic_a_layers[i], tf.nn.relu, name = 'action_layer_' + str(i))

                layer = s_layer + a_layer

                for i in range(1, len(self.critic_layers)):
                    layer = tf.layers.dense(layer, self.critic_layers[i], tf.nn.relu, name = 'FC_Layer_' + str(i))

                self.state_values["Online"] = tf.layers.dense(layer, self.action_space[0], name='output_layer')

                self.predicted_q_value = tf.placeholder(tf.float32, [None] + self.action_space)
                self.critic_loss = tf.losses.mean_squared_error(self.predicted_q_value, self.state_values["Online"])
                self.critic_optimize = tf.train.AdamOptimizer(self.critic_lr).minimize(self.critic_loss)
                self.action_grads = tf.gradients(self.state_values["Online"], self.critic_action_input["Online"])


        with tf.variable_scope("Target"):
            with tf.variable_scope("Critic"):
                self.critic_state_input["Target"] = tf.placeholder(tf.float32, [None] + self.obs_space,
                                                                   name='input_state')
                self.critic_action_input["Target"] = tf.placeholder(tf.float32, [None] + self.action_space,
                                                                    name='input_action')

                conv1 = tf.layers.conv2d(self.critic_state_input["Target"], 16, 8, [4, 4], activation=tf.nn.relu, name="conv_layer_1")
                conv2 = tf.layers.conv2d(conv1, 32, 4, [2, 2], activation=tf.nn.relu, name="conv_layer_2")
                conv_flat = tf.layers.flatten(conv2, name="layer_flatten")

                s_layer = conv_flat

                for i in range(len(self.critic_s_layers)):
                    s_layer = tf.layers.dense(s_layer, self.critic_s_layers[i], tf.nn.relu,
                                              name='state_layer_' + str(i))

                a_layer = self.critic_action_input["Target"]
                for i in range(len(self.critic_a_layers)):
                    a_layer = tf.layers.dense(a_layer, self.critic_a_layers[i], tf.nn.relu,
                                              name='action_layer_' + str(i))

                layer = s_layer + a_layer

                for i in range(1, len(self.critic_layers)):
                    layer = tf.layers.dense(layer, self.critic_layers[i], tf.nn.relu, name='FC_Layer_' + str(i))

                self.state_values["Target"] = tf.layers.dense(layer, self.action_space[0], name='output_layer')

            # self.state_summary = tf.summary.histogram('State Values', self.state_values)

    def updateTargetNetwork(self):
        self.sess.run(self.updateOp)

    ###ACTOR###
    def getOnlineActionProb(self, input_state):
        feed_dict = {self.actor_input["Online"]: input_state}

        return self.sess.run(self.action_prob["Online"], feed_dict=feed_dict)

    def getTargetActionProb(self, input_state):
        feed_dict = {self.actor_input["Target"]: input_state}

        return self.sess.run(self.action_prob["Target"], feed_dict=feed_dict)

    def trainActor(self, inputs, gradients):
        feed_dict = {self.actor_input["Online"]: inputs,
                    self.action_gradient: gradients}

        self.sess.run(self.actor_optimize, feed_dict = feed_dict)


    ###CRITIC###
    def getOnlineStateValues(self, input_state, input_action):
        feed_dict = {self.critic_state_input["Online"]: input_state,
                     self.critic_action_input["Online"]: input_action}

        return self.sess.run(self.state_values["Online"], feed_dict=feed_dict)

    def getTargetStateValues(self, input_state, input_action):
        feed_dict = {self.critic_state_input["Target"]: input_state,
                     self.critic_action_input["Target"]: input_action}

        return self.sess.run(self.state_values["Target"], feed_dict=feed_dict)

    def trainCritic(self, input_state, input_action, predicted_q_value):
        feed_dict = {self.critic_state_input["Online"]: input_state,
                     self.critic_action_input["Online"]: input_action,
                     self.predicted_q_value: predicted_q_value}

        self.sess.run(self.critic_optimize, feed_dict=feed_dict)

    def getActionGradients(self, input_state, input_action): #TODO merge into getOnlineStateValues
        feed_dict = {self.critic_state_input["Online"]: input_state,
                     self.critic_action_input["Online"]: input_action}

        return self.sess.run(self.action_grads, feed_dict=feed_dict)