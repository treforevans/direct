import numpy as np
import tensorflow as tf
from itertools import combinations
from pdb import set_trace
from logging import getLogger
from time import time
from sys import stdout
logger = getLogger(__name__)


class BayesGLM(object):
    def __init__(self, Phi, y, sig2_grid, Wbar, logP, logpsig, logQ=None, logqsig=None, n_mixtures=1, reinforce_entropy=False):
        """
        Bayesian Generalized Linear Model

        Inputs:
            Phi : coefficient matrix (n,b)
            y : outputs, (n,1)
            sig2_grid : gaussian noise grid, 1d numpy array of floats
            Wbar : grid of weights (d, mbar)
            logP : log-prior over weights. Must be of shape (b,mbar)
            logpsig : log-prior over sig2_grid
            logQ : log-variational over weights. Must be of shape (b,mbar) (even if n_mixtures > 1)
            logqsig : log-variational over sig2_grid
            n_mixtures : int, number of mixture terms in the variational distribution. If == 1, then mean-field
            reinforce_entropy : int, if > 0 then will use an unbaised stocastic gradient estimate of the variational entropy
                with this many samples.
                If <=0 then will use a deterministic, baised lower bound for the variational entropy.
                Can only be used when n_mixtures > 1, if n_mixtures == 1 then entropy can be computed exactly.
        """
        logger.info("Inititializing model")
        # get counts and check sizes
        self.n, self.b = Phi.shape
        assert Wbar.shape[0] == self.b
        self.mbar = Wbar.shape[1]
        assert y.shape == (self.n,1)
        self.Wbar = tf.constant(Wbar)
        assert np.all(sig2_grid) > 0
        assert sig2_grid.ndim == 1
        self.sig2_grid = tf.reshape(sig2_grid, (-1,1))
        self.log_sig2_grid = tf.log(self.sig2_grid)

        # deal with the prior. Note: to get the log prior, do self.prior.log()
        assert logP.shape == (self.b, self.mbar)
        self.logP = tf.nn.log_softmax(logP, axis=1) # ensure that the rows are normalized for a valid distribution
        assert logpsig.shape == sig2_grid.shape
        self.logpsig = tf.reshape(tf.nn.log_softmax(logpsig), (-1,1))

        # initialize the variational distribution
        self.reinforce_entropy = (reinforce_entropy > 0)
        self.n_samples = int(reinforce_entropy)
        assert n_mixtures >= 1
        self.n_mixtures = int(n_mixtures)
        # get the noise var VI dist
        if logqsig is None:
            logqsig = logpsig
        else:
            assert logqsig.shape == logpsig.shape
        self.logqsig = tf.nn.log_softmax(tf.Variable(initial_value=logqsig.reshape((-1,1)), trainable=True), axis=0)
        # now get the latent variable VI dist
        if logQ is None: # in both the mean field and mixture case, logQ must always be mean field if specified!
            logQ = logP
        else:
            assert logQ.shape == logP.shape
        if self.n_mixtures == 1: # then just mean-field VI
            self.logQ = tf.nn.log_softmax(tf.Variable(initial_value=logQ, trainable=True), axis=1)
            assert not self.reinforce_entropy, "do not use reinforce entropy with a factorized variational distribution"
        else: # a mixture distribution
            # initialize the mixture weights to be equal and the variational dist to be the prior with some small pertubations applied to each
            self.log_mix = tf.nn.log_softmax(tf.Variable(initial_value=np.zeros(self.n_mixtures), trainable=True))
            self.logQ = tf.nn.log_softmax(tf.Variable(
                initial_value=np.tile(logQ,(self.n_mixtures,1,1)) + 1e-2*np.random.randn(self.n_mixtures,self.b,self.mbar),
                trainable=True), axis=2)
            if self.reinforce_entropy:
                pass
            else: # compute nessessary terms for the ELBO lower bound
                # first Initialize A which indicates the point that the taylor series approx is taken about.
                #     initialize this value to the prior, ensuring that it is normalized to start with #     the prior value is about where the variational dist will be so we want it close by
                self.logA = tf.Variable(initial_value=tf.nn.log_softmax(logQ, axis=1), trainable=True) # no constraints except postive, seems to work fine
                #self.logA = tf.nn.log_softmax(tf.Variable(initial_value=logP+1e-2*np.random.randn(self.b,self.mbar), trainable=True), axis=1) # constrain to be valid prob dist

                # now get the terms (with coefficient 2) for the Taylor series approx of the inner product between q and logq
                self.q_terms = np.array(list(combinations(range(self.n_mixtures), 2)), dtype=int).T

        # now pre-compute the stuff required for the log-likelihood
        self.yTy = tf.constant(y.T.dot(y))
        self.PhiTy = tf.constant(Phi.T.dot(y))
        self.PhiTPhi = Phi.T.dot(Phi)
        self.diag_PhiTPhi = np.diag(self.PhiTPhi).reshape((-1,1))
        self.H = tf.constant(self.diag_PhiTPhi * np.power(Wbar,2)) # b x mbar
        if self.n_mixtures > 1: # TODO: I need to merge the mixtures computations to work with the more scalable method
            self.terms = np.array(list(combinations(range(self.b), 2)), dtype=int).T
            self.g = tf.constant(2.*(self.PhiTPhi[self.terms[0,:],self.terms[1,:]]).reshape((-1,1)))
        else:
            self.diag_PhiTPhi = tf.constant(self.diag_PhiTPhi)
            self.PhiTPhi = tf.constant(self.PhiTPhi)

        # set some flags
        self.is_trained = False

        if self.reinforce_entropy: # then initialize adam optimizer
            self.sgd = tf.train.AdamOptimizer()
            self.sgd_iter = self.sgd.minimize(-self.ELBO)

    @property
    def ELBO(self):
        """
        Evidence lower bound
        """
        Q = tf.exp(self.logQ) # (n_mix x) b x mbar
        qsig = tf.exp(self.logqsig) # mbar * 1
        # initialize ELBO with the contribution from the Gaussian noise to the logp and logq terms
        ELBO = tf.matmul(a=qsig, transpose_a=True, b=self.logpsig-self.logqsig)
        if self.n_mixtures == 1: # then just mean-field VI
            s = tf.reduce_sum(Q * self.Wbar, axis=1, keepdims=True)
            # compute inner products with log-prior and log-variational distributions
            ELBO += tf.reduce_sum(Q * (self.logP - self.logQ))
            # now add the inner product of the variational dist with the log-likelihood
            # first, just focus on the part that w contributes to
            logl_w = self.yTy - 2*tf.matmul(a=s, transpose_a=True, b=self.PhiTy) + \
                    tf.matmul(a=s, transpose_a=True, b=tf.matmul(a=self.PhiTPhi, b=s)) - tf.matmul(a=self.diag_PhiTPhi, transpose_a=True, b=tf.square(s)) + tf.reduce_sum(Q * self.H)
                    #tf.matmul(a=self.g, transpose_a=True, b=tf.gather(s, self.terms[0]) * tf.gather(s, self.terms[1])) + tf.reduce_sum(Q * self.H)
            # then put it all together with the stuff that includes the Gaussian noise
            ELBO += -(0.5*self.n)*tf.matmul(a=qsig, transpose_a=True, b=self.log_sig2_grid) - \
                    0.5*tf.matmul(a=qsig, transpose_a=True, b=1./self.sig2_grid)*logl_w
        else: # mixture distribution
            # TODO: I need to implement the mixtures logl computations to work with the more scalable method above (using PhiTPhi and diag_PhiTPhi)
            mix = tf.exp(self.log_mix)
            s = tf.reduce_sum(Q * tf.expand_dims(self.Wbar,axis=0), axis=2, keepdims=True) # n_mix x b x 1
            # compute the inner product with the log-prior (see notes Apr 9, 18)
            ELBO += tf.tensordot(mix, tf.reduce_sum(Q * tf.expand_dims(self.logP,axis=0), axis=(1,2)), axes=[[0],[0]])
            # compute the inner product with the log-likelihood
            # first, just focus on the part that w contributes to
            logl_w = self.yTy - \
                    2*tf.tensordot(mix, tf.tensordot(s, self.PhiTy, axes=[[1],[0]]), axes=[[0],[0]]) + \
                    tf.tensordot(mix,
                                 tf.tensordot(self.g, tf.gather(s, self.terms[0], axis=1) * tf.gather(s, self.terms[1], axis=1), axes=[[0],[1]]), # 1 x n_mix x 1
                                 axes=[[0],[1]]) + \
                    tf.tensordot(mix, tf.reduce_sum(Q * tf.expand_dims(self.H,axis=0), axis=(1,2)), axes=[[0],[0]])
            # then put it all together with the stuff that includes the Gaussian noise
            ELBO += -(0.5*self.n)*tf.matmul(a=qsig, transpose_a=True, b=self.log_sig2_grid) - \
                    0.5*tf.matmul(a=qsig, transpose_a=True, b=1./self.sig2_grid)*logl_w
            # now compute the w contribution to the entropy (logq term)
            if self.reinforce_entropy: # then mixture distribution with REINFORCE surrogate for entropy of the weight distribution
                w_samples = tf.stop_gradient(self.sample_variational(n_samples=self.n_samples, sample_sig2=False)) # ensure gradient stops here
                ELBO -= 0.5*tf.reduce_mean(tf.square(self.log_variational(w=w_samples) + 1.))
            else: # mixture distribution with deterministic, biased gradients
                # take the inner product between q and the lower bound of logq (see notes Apr 9, 18)
                QoA = tf.exp(self.logQ-tf.expand_dims(self.logA,axis=0)) # = Q/A of shape n_mix x b x mbar
                ELBO += 1. -\
                        tf.tensordot(mix, tf.reduce_sum(Q * tf.expand_dims(self.logA,axis=0), axis=(1,2)), axes=[[0],[0]]) - \
                        tf.tensordot(mix, tf.exp(self.log_mix+tf.reduce_sum(tf.log(tf.reduce_sum(Q*QoA,axis=2)),axis=1)), axes=[[0],[0]]) - \
                        2.*tf.reduce_sum(tf.exp(tf.gather(self.log_mix, self.q_terms[0]) + tf.gather(self.log_mix, self.q_terms[1]) + \
                                                tf.reduce_sum(tf.log(tf.reduce_sum(tf.gather(Q, self.q_terms[0])*tf.gather(QoA, self.q_terms[1]),axis=2)),axis=1)))
        return ELBO


    def train(self, sess, n_epochs, display_step=50):
        """
        Tune the variational distribution to maximize the ELBO

        Must run: `sess.run(tf.global_variables_initializer())` at some point before training.
        """
        logger.info("Beginnning training iterations. n_epochs=%d"%n_epochs)

        # determine optimization scheme to use
        feed_dict = {}
        if not self.reinforce_entropy: # then use quasi-newton optimizer since deterministic loss
            logger.info("using L-BFGS-B optimizer")
            # setup optimizer
            loss = -self.ELBO
            optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, method="L-BFGS-B", options={'maxiter':n_epochs, 'disp':-1})

            # define function and variables to maintain trace
            # https://stackoverflow.com/questions/44685228/how-to-get-loss-function-history-using-tf-contrib-opt-scipyoptimizerinterface
            def append_trace(neg_ELBO, display_step, t0):
                global epoch, trace
                if epoch==0 or (epoch+1) % display_step == 0:
                    trace['epoch'].append(epoch+1)
                    trace['t_elapsed'].append(time()-t0)
                    trace['ELBO'].append(-neg_ELBO.squeeze())
                    print("Epoch: %04d, ELBO: %.8g, Time elapsed: %.4g seconds." % (epoch+1, trace['ELBO'][-1], trace['t_elapsed'][-1]))
                    stdout.flush()
                epoch += 1
            global epoch, trace
            trace = dict(epoch=[], t_elapsed=[], ELBO=[])
            epoch = 0
            t0 = time()

            # run optimizations
            optimizer.minimize(sess, loss_callback=lambda neg_ELBO: append_trace(neg_ELBO, display_step, t0), fetches=[loss,])
        else: # use SGD
            logger.info("using Adam optimizer")
            trace = dict(epoch=[], t_elapsed=[], ELBO=[])
            t0 = time()
            for epoch in range(n_epochs):
                # run optimizer
                sess.run(self.sgd_iter, feed_dict)

                # Display logs per epoch step
                if epoch==0 or (epoch+1) % display_step == 0:
                    trace['t_elapsed'].append(time()-t0)
                    trace['epoch'].append(epoch+1)
                    trace['ELBO'].append(sess.run(self.ELBO, feed_dict).squeeze())
                    print("Epoch: %04d, loss: %.8g, Time elapsed: %.4g seconds." % (epoch+1, -trace['ELBO'][-1], trace['t_elapsed'][-1]))
                    stdout.flush()
        print("Optimization Finished!")
        trace['t_elapsed'].append(time()-t0)
        trace['epoch'].append(epoch+1)
        trace['ELBO'].append(sess.run(self.ELBO, feed_dict).squeeze())
        print("Epoch: %04d, ELBO: %.8g" % (epoch+1, trace['ELBO'][-1]))
        self.post_train(sess) # run post-training stuff
        return trace


    def post_train(self, sess):
        """
        runs nessessary stuff after training completed
        """
        # compute and save s which is nessessary for predictive posterior mean computations
        if self.n_mixtures == 1: # the mean field dist, (see notebook April 5, 18)
            self.s = tf.constant(sess.run(tf.reduce_sum(tf.exp(self.logQ) * self.Wbar, axis=1, keepdims=True)))
        else: # mixture dist, see notebook Apr 9, 18
            Q = tf.exp(self.logQ)
            mix = tf.exp(self.log_mix)
            self.s = tf.constant(sess.run(tf.matmul(tf.expand_dims(mix,axis=0), tf.reduce_sum(Q * tf.expand_dims(self.Wbar,axis=0), axis=2))).T) # b x 1
        self.is_trained = True # set train flag


    def predict_mean(self, Phi_X):
        """
        Compute the predictive posterior mean using the variational distribution.
        We will consider the full support of the variational distribution, i.e. consider every possible set of parameter values.

        Inputs:
            Phi_X : the basis functions evaluated at the n_test points X. This should be of shape (n_test, b)

        Returns:
            y_mean : tensor of shape (n_test, 1)
        """
        assert self.is_trained
        assert isinstance(Phi_X, np.ndarray)
        y_mean = tf.matmul(Phi_X, self.s)
        return y_mean


    def predict_samples(self, Phi_X, n_samples=100):
        """
        return samples of the predictive posterior

        Inputs:
            Phi_X : the basis functions evaluated at the n_test points X. This should be of shape (n_test, b)
            n_samples : number of samples of the predictive posterior

        Returns:
            y_samples : tensor of shape (n_test, n_samples)
        """
        assert self.is_trained
        assert isinstance(Phi_X, np.ndarray)
        iw_samples, isig2_samples = self.sample_variational(n_samples=n_samples, sample_sig2=True) # indicies of the latent variable values of the weights and noise variances
        gather_index = tf.concat((tf.tile(tf.reshape(tf.range(self.b, dtype=tf.int32),(-1,1)),(n_samples,1)),
                                  tf.reshape(tf.transpose(iw_samples), (-1,1))), axis=1)
        w_samples = tf.transpose(tf.reshape(tf.gather_nd(self.Wbar, gather_index), (n_samples, self.b))) # extract the latent variable values from the indicies
        y_samples = tf.matmul(Phi_X, w_samples)
        sig2_samples = tf.gather(tf.squeeze(self.sig2_grid, axis=1), isig2_samples, axis=0)
        return y_samples, sig2_samples


    def sample_variational(self, n_samples=100, sample_sig2=False):
        """
        return samples of the variational distribution as integers corresponding to the index
        of the samples value along each feature dimension

        Returns:
            w_samples : (b, n_samples)
            sig2_samples : (1, n_samples) if sample_sig2 specified
        """
        # sample the weights
        if self.n_mixtures > 1:
            # first sample n_samples from each mixture
            mixture_samples = tf.multinomial(logits=tf.reshape(self.logQ, (-1, self.mbar)), # must be a 2d array so just stack all of the mixtures vertically
                                       num_samples=n_samples, output_dtype=tf.int32)
            mixture_samples = tf.reshape(mixture_samples, (self.n_mixtures, self.b, n_samples)) # now separate samples from the different mixtures along first axis

            # then decide which to keep by sampling the mixture weights
            i_mixture_samples = tf.squeeze(tf.multinomial(logits=tf.reshape(self.log_mix, (1,-1)), num_samples=n_samples, output_dtype=tf.int32))
            w_samples = tf.transpose(tf.gather_nd(tf.transpose(mixture_samples, perm=(0,2,1)), indices=tf.stack([i_mixture_samples, tf.range(n_samples, dtype=tf.int32)], axis=1)))
        else: # mean field model so sampling is easier
            w_samples = tf.multinomial(logits=self.logQ, num_samples=n_samples, output_dtype=tf.int32)

        # sample the noise variances
        if sample_sig2: # now sample the noise variance (factorizes)
            sig2_samples = tf.multinomial(logits=tf.reshape(self.logqsig, (1,-1)), num_samples=n_samples, output_dtype=tf.int32)
            return w_samples, sig2_samples
        else:
            return w_samples


    def log_variational(self, w, sig2=None):
        """
        evaluate the log variational distribution at points {w,sig2} in the hypothesis space.

        Inputs:
            w : (b, n_points) int specifying the index location of the weight vector in hypothesis space
            sig2 : (1, n_points) ints specifying the index location of the noise variance in hypothesis space

        Outputs:
            log_var : (n_points,)
        """
        if self.n_mixtures == 1:
            raise NotImplementedError("")
        n = tf.shape(w)[1] # number of samples

        # get the contribution from the weights
        #     See notes Oct 21/2018 for derivation
        gather_index = tf.concat((tf.tile(tf.reshape(tf.range(self.b, dtype=tf.int32),(1,-1,1)),(n,1,1)),
                                 tf.expand_dims(tf.transpose(w), axis=2)), axis=2)
        logQsummed = tf.reduce_sum(tf.gather_nd(tf.transpose(self.logQ, perm=(1,2,0)), gather_index), axis=1) # (n, r)
        log_var = tf.reduce_logsumexp(tf.expand_dims(self.log_mix, axis=0) + logQsummed, axis=1)

        if sig2 is not None: # get the contribution from the noise variance (which simply factorizes)
            log_var += tf.gather(tf.reshape(self.logqsig, (-1,)), sig2[0])
        return log_var


