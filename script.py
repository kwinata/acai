#!/usr/bin/env python

import importlib
importlib.import_module("acai")
from acai import ACAI, data, math

"""
python acai.py --train_dir=TEMP --latent=32 --latent_width=2 --depth=16 --dataset=car_racing

if __name__ == '__main__':
    flags.DEFINE_integer('depth', 64, 'Depth of first for convolution.')
    flags.DEFINE_integer(
        'latent', 16,
        'Latent space depth, the total latent size is the depth multiplied by '
        'latent_width ** 2.')
    flags.DEFINE_integer('latent_width', 4, 'Width of the latent space.')
    flags.DEFINE_float('advweight', 0.5, 'Adversarial weight.')
    flags.DEFINE_integer('advdepth', 0, 'Depth for adversary network.')
    flags.DEFINE_float('reg', 0.2, 'Amount of discriminator regularization.')
    app.run(main)
"""

class FLAGS:
	batch = 64
	dataset = "car_racing"
	latent_width = 2
	train_dir = "TEMP_CP"
	latent = 32
	depth = 16
	advweight = 0.5
	advdepth = 0
	reg = 0.2


batch = FLAGS.batch
dataset = data.get_dataset(FLAGS.dataset, dict(batch_size=batch))
scales = int(round(math.log(dataset.width // FLAGS.latent_width, 2)))
model = ACAI(
    dataset,
    FLAGS.train_dir,
    latent=FLAGS.latent,
    depth=FLAGS.depth,
    scales=scales,
    advweight=FLAGS.advweight,
    advdepth=FLAGS.advdepth or FLAGS.depth,
    reg=FLAGS.reg
)

print model

model.encode(latent=FLAGS.latent, depth=FLAGS.depth, scales = scales)
