from ISR.train import Trainer
from ISR.models import RDN
from ISR.models import Discriminator
from ISR.models import Cut_VGG19

lr_train_patch_size = 80
layers_to_extract = [5, 9]
scale = 1
hr_train_patch_size = lr_train_patch_size * scale

rdn = RDN(arch_params={'C': 6, 'D': 20, 'G': 64, 'G0': 64,
                       'x': scale}, patch_size=lr_train_patch_size)
f_ext = Cut_VGG19(patch_size=hr_train_patch_size,
                  layers_to_extract=layers_to_extract)
discr = Discriminator(patch_size=hr_train_patch_size, kernel_size=3)

loss_weights = {
    'generator': 0.0,
    'feature_extractor': 0.0833,
    'discriminator': 0.01
}
losses = {
    'generator': 'mae',
    'feature_extractor': 'mse',
    'discriminator': 'binary_crossentropy'
}

log_dirs = {'logs': './logs', 'weights': './weights'}

learning_rate = {'initial_value': 0.0004,
                 'decay_factor': 0.5, 'decay_frequency': 30}

flatness = {'min': 0.0, 'max': 0.15, 'increase': 0.01, 'increase_frequency': 5}

trainer = Trainer(
    generator=rdn,
    discriminator=discr,
    feature_extractor=f_ext,
    hr_train_dir='galaxy_zoo/individuals_2blend_train/',
    lr_train_dir='galaxy_zoo/merged_2blend_train/',
    hr_valid_dir='galaxy_zoo/individuals_2blend_valid/',
    lr_valid_dir='galaxy_zoo/merged_2blend_valid/',
    loss_weights=loss_weights,
    learning_rate=learning_rate,
    flatness=flatness,
    dataname='div2k',
    log_dirs=log_dirs,
    weights_generator=None,
    weights_discriminator=None,
    n_validation=5,  #to be modified
)

trainer.train(
    epochs=200,
    steps_per_epoch=5000,
    batch_size=16,
    monitored_metrics={'val_generator_PSNR_Y': 'max'}
)
