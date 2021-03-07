#python 03_generate_rnn_data.py

import argparse
import numpy as np
import os

from lib import data

import importlib
importlib.import_module("acai")
from acai import ACAI, data, math

ROOT_DIR_NAME = "./data/"
ROLLOUT_DIR_NAME = "./data/rollout/"
SERIES_DIR_NAME = "./data/series/"


"""
Initialize ACAI
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


def get_filelist(N):
    filelist = os.listdir(ROLLOUT_DIR_NAME)
    filelist = [x for x in filelist if x != '.DS_Store']
    filelist.sort()
    length_filelist = len(filelist)

    if length_filelist > N:
      filelist = filelist[:N]

    if length_filelist < N:
      N = length_filelist

    return filelist, N

def encode_episode(episode):

    obs = episode['obs']
    action = episode['action']
    reward = episode['reward']
    done = episode['done']

    done = done.astype(int)  
    reward = np.where(reward>0, 1, 0) * np.where(done==0, 1, 0)

    mu = model.encode(obs, latent=FLAGS.latent, depth=FLAGS.depth, scales = scales)
    log_var = np.full(mu.shape, -1000.0)

    initial_mu = mu[0, :]
    initial_log_var = log_var[0, :]

    return (mu, log_var, action, reward, done, initial_mu, initial_log_var)



def main(args):

    N = args.N

    filelist, N = get_filelist(N)

    file_count = 0

    initial_mus = []
    initial_log_vars = []

    for file in filelist:
      try:
      
        rollout_data = np.load(ROLLOUT_DIR_NAME + file)

        mu, log_var, action, reward, done, initial_mu, initial_log_var = encode_episode(rollout_data)

        np.savez_compressed(SERIES_DIR_NAME + file, mu=mu, log_var=log_var, action = action, reward = reward, done = done)
        initial_mus.append(initial_mu)
        initial_log_vars.append(initial_log_var)

        file_count += 1

        if file_count%50==0:
          print('Encoded {} / {} episodes'.format(file_count, N))

      except Exception as e:
        print(e)
        print('Skipped {}...'.format(file))

    print('Encoded {} / {} episodes'.format(file_count, N))

    initial_mus = np.array(initial_mus)
    initial_log_vars = np.array(initial_log_vars)

    print('ONE MU SHAPE = {}'.format(mu.shape))
    print('INITIAL MU SHAPE = {}'.format(initial_mus.shape))

    np.savez_compressed(ROOT_DIR_NAME + 'initial_z.npz', initial_mu=initial_mus, initial_log_var=initial_log_vars)

    
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=('Generate RNN data'))
  parser.add_argument('--N',default = 10000, help='number of episodes to use to train')
  args = parser.parse_args()

  main(args)
