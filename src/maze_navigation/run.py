# Maze Navigation. Originally proposed in
# Backpropamine: differentiable neuromdulated plasticity.
#
# This code implements the "Grid Maze" task. See Section 4.5 in Miconi et al.
# ICML 2018 ( https://arxiv.org/abs/1804.02464 ), or Section 4.2 in
# Miconi et al. ICLR 2019 ( https://openreview.net/pdf?id=r1lrAiA5Ym )
#
# This file is modified to implement the maze itself. The `run.py` file
# contains the WarpGrad Network used in
#
# Meta-Learning With Warped Gradient Descent
# Flennerhag et. al., ICLR (2020), https://openreview.net/forum?id=rkeiQlBFPB

from alstm import aRNN
from maze import Maze, ObsSpec

import argparse
import torch
import torch.nn as nn
from numpy import random
import random
import pickle
import time

import numpy as np


np.set_printoptions(precision=4)

ADDITIONAL_INPUTS = 4
NUM_ACTIONS = 4
REF_SIZE = 3
TOTAL_INPUTS = REF_SIZE * REF_SIZE + ADDITIONAL_INPUTS + NUM_ACTIONS


def get_suffix(config):
    """Get experiment name.

    Args:
        config (dict): dict with experiment configs.

    Returns:
        suffix (str): experiment name.
    """
    return "run_" + "".join(
        [str(x) + "_" if pair[0] is not 'nbsteps' and
                         pair[0] is not 'rngseed' and
                         pair[0] is not 'save_every' and
                         pair[0] is not 'test_every' and
                         pair[0] is not 'pe' else ''
         for pair in sorted(zip(
            config.keys(), config.values()),
            key=lambda x:x[0]) for x in pair])[:-1] + \
           "_rngseed_" + str(config['rngseed'])


class WNet(nn.Module):

    """Warp-RNN.

    Args:
        isize (int): input size.
        hsize (int): hidden size.
        asize (int): size of adaptive hidden state.
    """

    def __init__(self, isize, hsize, asize=None, **kwargs):
        super(WNet, self).__init__()

        if asize is None:
            asize = hsize

        self.arnn = aRNN(isize, hsize, asize, hsize, **kwargs)
        self.pi = nn.Linear(hsize, NUM_ACTIONS)
        self.v = nn.Linear(hsize, 1)

    def forward(self, inputs, hidden):
        """Run model for one step."""
        output, hidden = self.arnn(inputs, hidden)
        output = output.squeeze(0)
        pi = self.pi(output)
        v = self.v(output)
        return pi, v, hidden


    def adapt_params(self):
        """Fully adaptive parameters."""
        return (list(self.arnn.arnn_layers.parameters()) +
                list(self.pi.parameters()) +
                list(self.v.parameters()))

    def meta_params(self):
        """Meta-parameters."""
        return (list(self.arnn.project_layers.parameters()) +
                list(self.arnn.adapt_layers.parameters()))

    def init_hidden(self, bsz):
        """Initialize hidden state."""
        return self.arnn.init_hidden(bsz)


def main(obs_spec, config):
    """Execute experiment.

    Args:
        obs_spec (ObsSpec): observation specs for maze.
        config (dict): experiment configuration.
    """
    batch_size = config["bs"]

    print("Passed config: ", config)
    suffix = get_suffix(config)

    np.random.seed(config['rngseed'])
    random.seed(config['rngseed'])
    torch.manual_seed(config['rngseed'])

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    net = WNet(TOTAL_INPUTS, config['hs']).to(device)
    print(net)

    optimizer = torch.optim.Adam(
        net.adapt_params(), lr=config["learning_rate"])
    meta_optimizer = torch.optim.Adam(
        net.meta_params(), lr=config["meta_learning_rate"])

    all_losses_objective = []
    all_total_rewards = []
    lossbetweensaves = 0
    nowtime = time.time()

    print("Starting training.")
    for numiter in range(config['nbiter']):
        # Calling Maze randomly samples a goal location
        env = Maze(obs_spec,
                   config["msize"],
                   batch_size,
                   config["wp"],
                   config["rew"])

        hidden = net.init_hidden(batch_size)

        loss = 0
        lossv = 0

        vs = []
        logprobs = []
        cumulative_rewards = np.zeros(batch_size)
        episode_rewards = [np.zeros(batch_size)]

        actions = np.zeros(batch_size, dtype=np.int32)
        for num_step in range(config['eplen']):
            inputs = env.obs(actions, episode_rewards[-1], num_step)
            inputs = torch.from_numpy(inputs).to(device)

            y, v, hidden = net(inputs.unsqueeze(0), hidden)
            y = torch.softmax(y, dim=1)

            dist = torch.distributions.Categorical(y)
            actions = dist.sample()
            logprobs.append(dist.log_prob(actions))
            actions = actions.detach().cpu().numpy()

            rewards = env.step(actions)

            loss += config['bent'] * y.pow(2).sum() / batch_size

            vs.append(v)
            cumulative_rewards += rewards
            episode_rewards.append(rewards)
        ###
        gamma = config['gr']
        returns = torch.zeros(batch_size).to(device)
        for numstepb in reversed(range(config['eplen'])) :
            rewards = torch.from_numpy(episode_rewards[numstepb]).to(device)
            returns = gamma * returns + rewards
            advantage = returns - vs[numstepb][0]
            lossv += advantage.pow(2).sum() / batch_size
            loss -= (logprobs[numstepb] * advantage.detach()).sum() / batch_size

        loss += config['blossv'] * lossv
        loss /= config['eplen']

        loss.backward()
        if numiter > 100:
            optimizer.step()
            optimizer.zero_grad()
            if numiter % config["meta_update_ival"] == 0:
                meta_optimizer.step()
                meta_optimizer.zero_grad()

        lossnum = float(loss)
        lossbetweensaves += lossnum
        all_losses_objective.append(lossnum)
        all_total_rewards.append(cumulative_rewards.mean())

        if (numiter+1) % config['pe'] == 0:

            print(numiter, "====")
            print("Mean loss: ", lossbetweensaves / config['pe'])
            lossbetweensaves = 0
            print("Mean reward (across batch and last",
                  config['pe'], "eps.): ",
                  np.sum(all_total_rewards[-config['pe']:])/ config['pe'])
            previoustime = nowtime
            nowtime = time.time()
            print("Time spent on last", config['pe'], "iters: ",
                  nowtime - previoustime)

        if (numiter+1) % config['save_every'] == 0:
            print("Saving files...")
            losslast100 = np.mean(all_losses_objective[-100:])
            print("Average loss over the last 100 episodes:", losslast100)
            print("Saving local files...")
            prefix = 'WNET_'
            with open(prefix+'loss_'+suffix+'.txt', 'w') as thefile:
                for item in all_total_rewards[::10]:
                    thefile.write("%s\n" % item)
            #torch.save(net.state_dict(), 'torchmodel_'+suffix+'.dat')
            #with open(prefix + 'params_'+suffix+'.dat', 'wb') as fo:
            #     pickle.dump(config, fo)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--rngseed", type=int,
                        help="random seed", default=0)
    parser.add_argument("--rew", type=float,
                        help="reward value", default=10.0)
    parser.add_argument("--wp", type=float,
                        help="penalty for hitting walls", default=.0)
    parser.add_argument("--bent", type=float,
                        help="coefficient for entropy loss", default=0.03)
    parser.add_argument("--blossv", type=float,
                        help="coefficient for value loss", default=.1)
    parser.add_argument("--msize", type=int,
                        help="size of the maze; must be odd", default=13)
    parser.add_argument("--gr", type=float,
                        help="discounting factor for rewards", default=.9)
    parser.add_argument("--gc", type=float,
                        help="gradient norm clipping", default=4.0)
    parser.add_argument("--eplen", type=int,
                        help="length of episodes", default=200)
    parser.add_argument("--hs", type=int,
                        help="size of the hidden layers", default=100)
    parser.add_argument("--bs", type=int,
                        help="batch size", default=30)
    parser.add_argument("--l2", type=float,
                        help="coefficient of L2 norm", default=0)
    parser.add_argument("--nbiter", type=int,
                        help="number of learning cycles", default=1000000)
    parser.add_argument("--save_every", type=int,
                        help="num cycles between checkpoints", default=200)
    parser.add_argument("--pe", type=int,
                        help="num cycles between info readouts", default=100)
    # WarpGrad-specific
    parser.add_argument("--learning_rate", type=float,
                        help="learning rate for adaptive layers", default=1e-3)
    parser.add_argument("--meta_learning_rate", type=float,
                        help="learning rate for warp-layers", default=1e-3)
    parser.add_argument("--meta_update_ival", type=int,
                        help="Meta-update interval", default=30)

    args = parser.parse_args()
    argvars = vars(args)

    obs_spec = ObsSpec(NUM_ACTIONS,
                       REF_SIZE,
                       ADDITIONAL_INPUTS,
                       TOTAL_INPUTS,
                       argvars["eplen"])

    main(obs_spec, argvars)
