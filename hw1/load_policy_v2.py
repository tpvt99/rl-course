import pickle, numpy as np
import tensorflow as tf

class CustomDenseLayer(tf.keras.layers.Layer):
    def __init__(self, W, b, nonlinear_type):
        super(CustomDenseLayer, self).__init__()
        self.kernel = tf.Variable(initial_value=W, dtype=np.float32, trainable=False)
        self.bias = tf.Variable(initial_value=b, dtype=np.float32, trainable=False)
        self.nonlinear_type = nonlinear_type

    def lrelu(x, leak=0.2):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

    @tf.function
    def call(self, inputs):
        x = tf.matmul(inputs, self.kernel) + self.bias
        if self.nonlinear_type == 'lrelu':
            return tf_util.lrelu(x, leak=.01)  # openai/imitation nn.py:233
        elif self.nonlinear_type == 'tanh':
            return tf.tanh(x)

class ExpertPolicy(tf.keras.Model):
    def __init__(self):
        super(ExpertPolicy, self).__init__()
        self.policy_params = None
        self.layer_lists = []
        self.initialization("experts/Humanoid-v2.pkl")

    def initialization(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.loads(f.read())

        nonlin_type = data['nonlin_type']
        policy_type = [k for k in data.keys() if k != 'nonlin_type'][0]

        assert policy_type == 'GaussianPolicy', 'Policy type {} not supported'.format(policy_type)
        self.policy_params = data[policy_type]
        assert set(self.policy_params.keys()) == {'logstdevs_1_Da', 'hidden', 'obsnorm', 'out'}

        def read_layer(l):
            assert list(l.keys()) == ['AffineLayer']
            assert sorted(l['AffineLayer'].keys()) == ['W', 'b']
            return l['AffineLayer']['W'].astype(np.float32), l['AffineLayer']['b'].astype(np.float32)

        # Hidden layers next
        assert list(self.policy_params['hidden'].keys()) == ['FeedforwardNet']
        layer_params = self.policy_params['hidden']['FeedforwardNet']
        for layer_name in sorted(layer_params.keys()):
            l = layer_params[layer_name]
            W, b = read_layer(l)
            self.layer_lists.append(CustomDenseLayer(W, b, nonlin_type))
        # Output layer
        W, b = read_layer(self.policy_params['out'])
        self.layer_lists.append(CustomDenseLayer(W, b, nonlin_type))

    def normalization(self, inputs, policy_params):

        # Build the policy. First, observation normalization.
        assert list(policy_params['obsnorm'].keys()) == ['Standardizer']
        obsnorm_mean = policy_params['obsnorm']['Standardizer']['mean_1_D']
        obsnorm_meansq = policy_params['obsnorm']['Standardizer']['meansq_1_D']
        obsnorm_stdev = np.sqrt(np.maximum(0, obsnorm_meansq - np.square(obsnorm_mean)))
        print('obs', obsnorm_mean.shape, obsnorm_stdev.shape)
        normedobs_bo = (inputs - obsnorm_mean) / (
                obsnorm_stdev + 1e-6)  # 1e-6 constant from Standardizer class in nn.py:409 in openai/imitation

        return normedobs_bo

    @tf.function
    def call(self, inputs, training = True):
        x = self.normalization(inputs, self.policy_params)
        for layer in self.layer_lists:
            x = layer(x)
        return x