from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.applications import *
from ISR.models.imagemodel import ImageModel

from .nn_blocks import *


def make_model(arch_params, patch_size):
    """ Returns the model.

    Used to select the model.
    """

    return FaceGenerator(arch_params, patch_size)


class FaceGenerator(ImageModel):

    def __init__(self, patch_size=None):
        self.patch_size=patch_size

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder(512, input_size=self.patch_size//16)
        self.model = self.build_model()
        self.model._name = 'generator'
        self.name = 'rrdn'
    
    def build_encoder(self, nc_in=3, input_size=224):
        input_size = self.patch_size if self.patch_size else input_size
        inp = Input(shape=(input_size, input_size, nc_in))
        inp_ref = Input(shape=(input_size, input_size, nc_in))
        inp_segm_mask = Input(shape=(input_size, input_size, nc_in))
        x = concatenate([inp, inp_ref, inp_segm_mask])
        conv1 = conv_block(x, 64, use_norm=True, strides=2)
        conv2 = conv_block(conv1, 128, use_norm=True, strides=2)
        conv3 = conv_block(conv2, 256, use_norm=True, strides=2)
        conv4 = conv_block(conv3, 512, use_norm=True, strides=2)
        return Model(
            [inp, inp_ref, inp_segm_mask], 
            [
                conv4, 
                resize_tensor(inp, [input_size//2, input_size//2]), 
                resize_tensor(inp, [input_size//4, input_size//4]),
                inp_segm_mask
            ]
        )

    def build_decoder(self, nc_conv_in=512, input_size=14, nc_in=3, latent_dim=512):
        conv4 = Input(shape=(input_size, input_size, nc_conv_in))
        inp_segm_mask = Input(shape=(input_size, input_size, nc_in))
        inp_ds2 = Input(shape=(input_size*8, input_size*8, nc_in))
        inp_ds4 = Input(shape=(input_size*4, input_size*4, nc_in))
        inp_emb = Input(shape=(latent_dim,))
        
        emb_mean_var = embddding_fc_block(inp_emb)
        emb_mean_var = Reshape((1, 1, 256))(emb_mean_var)
        emb = Reshape((1, 1, latent_dim))(inp_emb)
        emb = UpSampling2D((input_size,input_size))(emb)
        
        x = adain_resblock(conv4, emb_mean_var, 512) 
        x = adain_resblock(x, emb_mean_var, 512)   
        x = concatenate([x, emb])
        x = upscale_nn(x, 512)
        x = SPADE_res_block(x, resize_tensor(inp_segm_mask, [input_size*2, input_size*2]), 512, block_id='3')
        x = upscale_nn(x, 256)
        x = SPADE_res_block(x, resize_tensor(inp_segm_mask, [input_size*4, input_size*4]), 256, block_id='4')
        x = concatenate([x, inp_ds4])
        x = upscale_nn(x, 128)
        x = concatenate([x, inp_ds2])
        x = upscale_nn(x, 64)
        
        out = Conv2D(3, kernel_size=4, padding='same', activation="tanh")(x)
        return Model([conv4, inp_ds2, inp_ds4, inp_segm_mask, inp_emb], out)

    def build_model(self):
        image_size = (self.patch_size, self.patch_size, 3)
        inp_src = Input(shape=image_size)
        inp_tar = Input(shape=image_size)
        inp_segm = Input(shape=image_size)
        inp_emb = Input((512))
        m=self.decoder(self.encoder([inp_src, inp_tar, inp_segm]) + [inp_emb])
        return m