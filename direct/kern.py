import numpy as np
import tensorflow as tf
from pdb import set_trace
from logging import getLogger
logger = getLogger(__name__)


class RBF_RFFs(object):
    def __init__(self, d, log_lengthscale=0, n_rffs=1000, dtype=np.float64, tune_len=True):
        """
        random fourier features (RFFs) of a squared exponential (RBF) kernel

        Input:
            d : number of input dims
            log_lengthscale : log of the lengthscale of the squared exponential kernel
            n_rffs : number of random features (number of basis functions will actually be twice this value)
            dtype : numpy or tensorflow data type to use for tensors
            tune_len : boolean variable specifying whether or not the lengthscale should be trainable
        """
        # TODO: add ability to be non ARD
        logger.info("initializing RBF kernel")
        self.d = int(d)
        self.n_rffs = int(n_rffs)
        self.n_features = 2*n_rffs # each random feature is broken into two
        self.dtype = dtype
        self.freq_weights = np.asarray(np.random.normal(size=(self.d, self.n_rffs), loc=0, scale=1.), dtype=self.dtype)
        self.bf_scale = 1./np.sqrt(self.n_rffs)

        # Set the lengthscale variable
        if np.size(log_lengthscale)==1 and log_lengthscale == 0:
            log_lengthscale = np.zeros((d,1), dtype=self.dtype)
        else:
            log_lengthscale = np.asarray(log_lengthscale, dtype=self.dtype).reshape((d,1))
        self.log_ell = tf.Variable(initial_value=log_lengthscale, trainable=tune_len, name='log_lengthscale')


    def Phi(self, x):
        """
        Get the basis function matrix

        Inputs:
            x : (n, d) input postions

        Outputs:
            Phi : (n, n_features)
        """
        Xfreq = tf.matmul(x, self.freq_weights/tf.exp(self.log_ell)) # scale the frequencies by the lengthscale and multiply with the inputs
        return self.bf_scale * tf.concat([tf.cos(Xfreq), tf.sin(Xfreq)], axis=1)


