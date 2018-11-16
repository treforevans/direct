# This module contains miscellaneous items that are not used in the direct package but are used for testing or tutorials
import numpy as np
import tensorflow as tf
from logging import getLogger
from pdb import set_trace
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
        logger.info("initializing RBF random fourier features")
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


class KronMatrix(object):
    """
    Tensor class which is a Kronecker product of matricies.

    Trefor Evans
    """

    def __init__(self, K):
        """
        Inputs:
            K  : is a list of numpy arrays
        """
        self.K = K # shallow copy
        self.n = len(self.K)
        self.sshape = np.vstack([np.shape(Ki) for Ki in self.K]) # sizes of the sub matrices
        self.shape = np.atleast_1d(np.prod(self.sshape,axis=0)) # shape of the big matrix
        self.ndim = self.shape.size
        assert self.ndim <= 2, "kron matrix cannot be more than 2d"
        self.square = self.ndim==2 and self.shape[0]==self.shape[1]


    def expand(self, log_expansion=False):
        """
        expands the kronecker product matrix explicitly. Expensive!

        Inputs:
            log_expansion : if used then will perform a numerically stable expansion.
                If specified then the output will be the log of the value.
        """
        Kb = 1.
        for Ki in self.K:
            Kb = np.kron(Kb, Ki)
        return Kb.reshape(self.shape)


class BlockMatrix(object):
    """ create Block matrix """

    def __init__(self, A):
        """
        Builds a block matrix with which matrix-vector multiplication can be made.

        Inputs:
            A : numpy object array of blocks of size (h, w)
                i.e. A = np.array([[ A_11, A_12, ... ],
                                   [ A_21, A_22, ... ], ... ]
                Each block in A must have the methods
                * shape
                * __mul__
                * T (transpose property)
                * expand (only nessessary if is to be used)
        """
        assert A.ndim == 2, 'A must be 2d'
        self.A = A # shallow copy

        # get the shapes of the matricies
        self.block_shape = self.A.shape # shape of the block matrix
        self._partition_shape = ([A_i0.shape[0] for A_i0 in self.A[:,0]], [A_0i.shape[1] for A_0i in self.A[0,:]]) # shape of each partition
        self.shape = tuple([np.sum(self._partition_shape[i]) for i in range(2)]) # overall shape of the expanded matrix

        # ensure the shapes are consistent for all partitions
        for i in range(self.block_shape[0]):
            for j in range(self.block_shape[1]):
                assert np.all(A[i,j].shape == self.partition_shape(i,j)), "A[%d,%d].shape should be %s, not %s" % (i,j,repr(self.partition_shape(i,j)),repr(A[i,j].shape))

        # define how a vector passed to it should be split when a matrix vector product is taken
        self.vec_split =  np.cumsum([0,] + self._partition_shape[1], dtype='i')


    def partition_shape(self, i, j):
        """ returns the shape of A[i,j] """
        return (self._partition_shape[0][i],self._partition_shape[1][j])


    def __mul__( self, x ):
        """ matrix vector multiplication """
        assert x.shape == (self.shape[1], 1)

        # first split the vector x so I don't have to make so many slices (which is slow)
        xs = [x[self.vec_split[j]:self.vec_split[j+1],:] for j in range(self.block_shape[1])]

        # loop through each block row and perform the matrix-vector product
        y = np.empty(self.block_shape[0], dtype=object)
        for i in range(self.block_shape[0]):
            y[i] = 0 # initialize
            for j in range(self.block_shape[1]): # loop accross the row
                y[i] += self.A[i,j] * xs[j]

        # concatenate results
        y = np.concatenate(y,axis=0)
        return y


    def transpose(self):
        """ transpose the kronecker product matrix. This currently copies the matricies explicitly """
        A = self.A.copy()

        # first transpose each block individually
        for i in range(self.block_shape[0]):
            for j in range(self.block_shape[1]):
                A[i,j] = A[i,j].T

        # then, transpose globally
        A = A.T

        # then return a new instance of the object
        return self.__class__(A=A)
    T = property(transpose) # calling self.T will do the same thing as transpose

    def expand(self):
        """ expands each block matrix to form a big, full matrix """
        Abig = np.zeros(np.asarray(self.shape, dtype='i'))
        row_split = np.cumsum([0,] + self._partition_shape[0], dtype='i')
        col_split = np.cumsum([0,] + self._partition_shape[1], dtype='i')
        for i in range(int(round(self.block_shape[0]))):
            for j in range(int(round(self.block_shape[1]))):
                Abig[row_split[i]:row_split[i+1], col_split[j]:col_split[j+1]] = self.A[i,j].expand()
        return Abig


class KhatriRaoMatrix(BlockMatrix):
    """ a Khatri-Rao Matrix (block Kronecker Product matrix) """

    def __init__(self, A, partition=None):
        """
        Khatri-Rao Block Matrix.

        Inputs:
            A : list of sub matricies or 2d array of KronMatricies. If the latter then partition is ignored.
            partition : int specifying the direction that the Khatri-Rao Matrix is partitioned:
                0 : row partitioned
                1 : column partitioned
                Note that if A is an array of KronMatricies then this has now effect.
        """
        # determine whether KronMatricies have already been formed from the partitions or not
        if np.ndim(A)==2 and isinstance(A[0,0], KronMatrix): # then all the work is done
            super(KhatriRaoMatrix, self).__init__(A)
            return

        # else I need to create KronMatrices from each partition
        # get the number of blocks that will be needed
        assert partition in range(2)
        if partition == 0:
            block_shape = (A[0].shape[0], 1)
        elif partition == 1:
            block_shape = (1,A[0].shape[1])
        else:
            raise ValueError('unknown partition')

        # form the KronMatricies
        Akron = np.empty(max(block_shape), dtype=object) # make 1d now and reshape later
        for i in range(max(block_shape)):
            if partition == 0:
                Akron[i] = KronMatrix([Aj[(i,),:] for Aj in A])
            elif partition == 1:
                Akron[i] = KronMatrix([Aj[:,(i,)] for Aj in A])
        Akron = Akron.reshape(block_shape)

        # Create a BlockMatrix from this
        super(KhatriRaoMatrix, self).__init__(Akron)


class KhatriRaoMeshgrid(object):
    """ stores a Meshgrid in a Khatri-Rao form. Each submatrix is a `IndexedSlicesPlusOne' to avoid storing all redundant vectors of ones. """

    def __init__(self, G):
        self.d = G.shape[0]
        self.mbar = G.shape[1]
        self.G = G


    def expand(self):
        """ form the Khatri-Rao matrix and expand """
        return self.khatri_rao.expand()

    @property
    def index_sliced(self):
        """ return the index sliced object """
        return [IndexedSlicesPlusOne(values=xi.reshape((1,-1))-1, indicies=i+np.zeros(1, dtype=int), n_rows=self.d) for i,xi in enumerate(self.G)]

    @property
    def khatri_rao(self):
        return KhatriRaoMatrix([IS.expand() for IS in self.index_sliced], partition=0)


class IndexedSlices(object):
    """ A sparse representation of a set of tensor slices at given indices. """

    def __init__(self, values, indicies, n_rows):
        """
        A sparse representation of a set of tensor slices at given indices.
        This is used where many rows of a matrix are sparse.
        This class is a simple wrapper for a pair of tensor objects:

        Inputs:
            values : A `Tensor` of any dtype with shape `[D0, D1, ..., Dn]`.
            indices: A 1-D integer `Tensor` with shape `[D0]`.
            n_rows : total height of the matrix

        An `IndexedSlices` is typically used to represent a subset of a larger
        tensor `dense` of shape `[LARGE0, D1, .. , DN]` where `LARGE0 >> D0`.
        The values in `indices` are the indices in the first dimension of
        the slices that have been extracted from the larger tensor.

        See the expand method for further details.
        """
        assert values.ndim >= 2
        self.values = values
        assert indicies.ndim == 1
        assert indicies.size == values.shape[0]
        self.indicies = indicies
        self.n_rows = n_rows
        self.shape = (self.n_rows,) + self.values.shape[1:]


    def expand(self):
        A = np.zeros(self.shape)
        A[self.indicies] = self.values # fill in the non-zero rows
        return A


class IndexedSlicesPlusOne(IndexedSlices):
    """ `IndexedSlices` wrapper for when many rows are just ones. Initiaized same as parent but we add one to the tensor for all computations. """

    def expand(self):
        return super(IndexedSlicesPlusOne, self).expand() + 1


def log_softmax(x, axis=0):
    """ compute log_softmax in a stable way """
    x_shifted = x - np.max(x, axis=axis, keepdims=True) # shift x by the largest value
    return x_shifted - np.log(np.sum(np.exp(x_shifted), axis=axis, keepdims=True))


