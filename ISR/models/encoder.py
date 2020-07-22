import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import RandomUniform
from ISR.models.imagemodel import ImageModel


def make_model(arch_params, patch_size):
    """ Returns the model.

    Used to select the model.
    """

    return Encoder(arch_params, patch_size)


class Encoder(ImageModel):
    def __init__(
        self, arch_params={}, patch_size=None, beta=0.2, c_dim=3, kernel_size=3, init_val=0.05
    ):
        self.params = arch_params
        self.beta = beta
        self.c_dim = c_dim
        self.C = self.params['C']
        self.D = self.params['D']
        self.G = self.params['G']
        self.G0 = self.params['G0']
        self.T = self.params['T']
        self.scale = self.params['x']
        self.initializer = RandomUniform(
            minval=-init_val, maxval=init_val, seed=None)
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.model = self._build_encoder()
        self.model._name = 'encoder'
        self.name = 'encoder'
        
    def _dense_block(self, input_layer, d, t):
        """
        Implementation of the (Residual) Dense Block as in the paper
        Residual Dense Network for Image Super-Resolution (Zhang et al. 2018).

        Residuals are incorporated in the RRDB.
        d is an integer only used for naming. (d-th block)
        """

        x = input_layer
        for c in range(1, self.C + 1):
            F_dc = Conv2D(
                self.G,
                kernel_size=self.kernel_size,
                padding='same',
                kernel_initializer=self.initializer,
                name='F_%d_%d_%d' % (t, d, c),
            )(x)
            F_dc = Activation('relu', name='F_%d_%d_%d_Relu' % (t, d, c))(F_dc)
            x = concatenate([x, F_dc], axis=3,
                            name='RDB_Concat_%d_%d_%d' % (t, d, c))

        # DIFFERENCE: in RDN a kernel size of 1 instead of 3 is used here
        x = Conv2D(
            self.G0,
            kernel_size=3,
            padding='same',
            kernel_initializer=self.initializer,
            name='LFF_%d_%d' % (t, d),
        )(x)
        return x

    def _RRDB(self, input_layer, t):
        """Residual in Residual Dense Block.

        t is integer, for naming of RRDB.
        beta is scalar.
        """

        # SUGGESTION: MAKE BETA LEARNABLE
        x = input_layer

        for d in range(1, self.D + 1):
            LFF = self._dense_block(x, d, t)
            LFF_beta = Lambda(lambda x: x * self.beta)(LFF)
            x = Add(name='LRL_%d_%d' % (t, d))([x, LFF_beta])
        x = Lambda(lambda x: x * self.beta)(x)
        x = Add(name='RRDB_%d_out' % (t))([input_layer, x])
        return x

    def _build_encoder(self):
        LR_input = Input(
            shape=(self.patch_size, self.patch_size, 3), name='LR_input')
        pre_blocks = Conv2D(
            self.G0,
            kernel_size=self.kernel_size,
            padding='same',
            kernel_initializer=self.initializer,
            name='Pre_blocks_conv',
        )(LR_input)
        # DIFFERENCE: in RDN an extra convolution is present here
        for t in range(1, self.T + 1):
            if t == 1:
                x = self._RRDB(pre_blocks, t)
            else:
                x = self._RRDB(x, t)
        # DIFFERENCE: in RDN a conv with kernel size of 1 after a concat operation is used here
        post_blocks = Conv2D(
            self.G0,
            kernel_size=3,
            padding='same',
            kernel_initializer=self.initializer,
            name='post_blocks_conv',
        )(x)
        # Global Residual Learning
        GRL = Add(name='GRL')([post_blocks, pre_blocks])
        SR = Conv2D(
            self.c_dim,
            kernel_size=self.kernel_size,
            padding='same',
            kernel_initializer=self.initializer,
            name='SR',
        )(GRL)
        return Model(inputs=LR_input, outputs=SR)
