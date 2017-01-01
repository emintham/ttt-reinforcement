import itertools
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict

from ttt import TTT

spinner = itertools.cycle(['-', '/', '|', '\\'])

def advance_spinner():
    """Advances the spinner to the next"""
    sys.stdout.write(next(spinner))
    sys.stdout.flush()
    sys.stdout.write('\b')


class Learner:
    """
    V table learner.
    """

    # Stuff you probably want to play around with
    # -------------------------------------------------------------------------
    # Learning rates - something close to 0.1 will work well.
    alpha = [0.1]

    # Probability of exploring instead of picking the greedy choice
    # - 0.01 seems like the best so far to maximize wins against a random opp.
    epsilon = [0.01]

    # Total number of training rounds per alpha per epsilon
    episodes = 30000

    # Number of rounds per test
    testing_rounds = 10

    # Number of training episodes between each test
    testing_interval = 100

    # Polynomial degree to fit the win rate plot
    polynomial_deg = 1

    # Rewards for win/loss/draws
    WIN_REWARD = 1
    DRAW_REWARD = 0
    LOSS_REWARD = -1

    # Opponent's strategy: 'random' or 'self_play'
    opponent_strategy = 'random'

    # Initial values for V table: see vtable_factory
    vtable_init = 'default'

    # Whether to print scatter points
    print_scatterpoints = False

    # Whether to print best fit line across win rates instead of raw win rates
    print_bestfit = True

    # Whether to update all the states symmetric to the current state
    learn_from_symmetry = True

    # Stuff you probably don't need to change
    # -------------------------------------------------------------------------
    SELF = 1
    OPPONENT = -1

    # Colours for plots
    line_colours = [
        'b', # blue
        'g', # green
        'r', # red
        'c', # cyan
        'm', # magenta
        'y', # yellow
        'k', # black
    ]

    # Scatter plot points transparency
    scatter_point_alpha = 0.3

    # Factory for initial v table values
    vtable_factory = {
        'random': lambda: random.random(),
        'empty': float,
        'random_pn': lambda: random.random() * random.choice([-1, 1]),
        'default': lambda: Learner.DRAW_REWARD,
    }

    def __init__(self, game_class, debug=False):
        self.v_table = defaultdict(
            lambda: defaultdict(            # for various values of alpha
                lambda: defaultdict(        # for various values of epsilon
                    self.vtable_factory[self.vtable_init])))

        self.win_rates = {alpha: {epsilon: [] for epsilon in self.epsilon}
                          for alpha in self.alpha}
        self.plotting_episodes = []     # X-axis
        self.game_class = game_class
        self.debug = debug

    def play_random_move(self, game):
        game.play(random.choice(game.moves))

    def play_as_opponent(self, game, alpha, epsilon):
        if self.opponent_strategy == 'random':
            self.play_random_move(game)
        elif self.opponent_strategy == 'self_play':
            v_table = self.v_table[alpha][epsilon]
            chosen_move = min(((move, v_table[game.try_play(move)])
                               for move in game.moves),
                              key=lambda x: x[1])[0]
            game.play(chosen_move)

    def play_as_self(self, game, alpha, epsilon):
        v_table = self.v_table[alpha][epsilon]
        chosen_move = max(((move, v_table[game.try_play(move)])
                           for move in game.moves),
                          key=lambda x: x[1])[0]
        game.play(chosen_move)

    def get_random_starting_player(self):
        return random.choice([self.SELF, self.OPPONENT])

    def get_update_states(self, state):
        """
        Returns all states that should be updated given a particular state.
        This returns the same state if Learner does not learn from symmetries.
        Otherwise it returns all symmetries of the given state.
        """
        if not self.learn_from_symmetry:
            return state
        else:
            return self.game_class.all_symmetries(state)

    def _update_vtable(self, prev_state, prev_v, game, v_table, alpha):
        state = game.as_feature_vector()

        if game.has_ended:
            if game.winner == self.SELF:
                reward = self.WIN_REWARD
            elif game.winner == self.OPPONENT:
                reward = self.LOSS_REWARD
            elif game.draw:
                reward = self.DRAW_REWARD

            for symmetry_state in self.get_update_states(state):
                v_table.setdefault(state, reward)

        chosen_score = v_table[state]
        for symmetry_state in self.get_update_states(prev_state):
            v_table[symmetry_state] = prev_v + alpha * (chosen_score - prev_v)

        return (state, chosen_score)

    def train_one_episode(self, v_table, alpha, epsilon):
        t = self.game_class()
        t.current_player = self.get_random_starting_player()

        while (not t.has_ended):
            prev_state = t.as_feature_vector()
            prev_v = v_table[prev_state]

            if t.current_player == self.OPPONENT:
                self.play_as_opponent(t, alpha, epsilon)
            else:
                chosen_move = None
                chosen_score = 0

                if random.random() > epsilon:
                    self.play_as_self(t, alpha, epsilon)
                else:
                    self.play_random_move(t)

            if self.debug:
                print(t)

            state, chosen_score = self._update_vtable(prev_state, prev_v, t, v_table, alpha)

            prev_state = state
            prev_v = chosen_score

    def train(self):
        for ep in range(self.episodes):
            advance_spinner()

            test = ep % self.testing_interval == 0

            if self.debug:
                print('Episode {}'.format(ep))

            if test:
                self.plotting_episodes.append(ep)

            for alpha in self.alpha:
                for epsilon in self.epsilon:
                    v_table = self.v_table[alpha][epsilon]

                    if test:
                        self.test(ep, alpha, epsilon)

                    self.train_one_episode(v_table, alpha, epsilon)

    def test(self, episodes, alpha, epsilon):
        times_won = 0
        times_draw = 0

        for i in range(self.testing_rounds):
            if self.debug:
                print('Testing round {}'.format(i))

            t = self.game_class()

            while (not t.has_ended):
                if t.current_player == self.SELF:
                    self.play_as_self(t, alpha, epsilon)
                else:
                    self.play_as_opponent(t, alpha, epsilon)

                if self.debug:
                    print(t)

            if t.winner == self.SELF:
                times_won += 1
            elif t.draw:
                times_draw += 1

        win_rate = times_won/self.testing_rounds
        draw_rate = times_draw/self.testing_rounds

        self.win_rates[alpha][epsilon].append(win_rate)

    def play(self, rounds=3, alpha=None, epsilon=None):
        alpha = alpha or self.alpha[0]
        epsilon = epsilon or self.epsilon[0]

        for i in range(rounds):
            print('Round {}'.format(i))

            t = self.game_class()
            t.current_player = self.get_random_starting_player()

            while (not t.has_ended):
                if t.current_player == self.SELF:
                    self.play_as_self(t, alpha, epsilon)
                else:
                    print(t)
                    pos = int(input().strip())
                    t.play(pos)

            if t.winner == self.SELF:
                print('You lost!')
            elif t.winner == self.OPPONENT:
                print('You won!')
            else:
                print('Draw!')

    def _plot_win_rate(self, x, y, param_label, param, line_colour):
        label = param_label.format(param)

        if self.print_bestfit:
            polynomial_coeffs = np.polyfit(x, y, self.polynomial_deg)
            bestfit_y = np.poly1d(polynomial_coeffs)(x)

            if self.print_scatterpoints:
                plt.scatter(x, y, c=line_colour, marker='x',
                            alpha=self.scatter_point_alpha)

            plt.plot(x, bestfit_y, line_colour, label=label)
        else:
            plt.plot(x, y, line_colour, label=label)

    def plot(self, plot_by='epsilon'):
        x = self.plotting_episodes

        if plot_by == 'alpha':
            for alpha, line_colour in zip(self.alpha, self.line_colours[:len(self.alpha)]):
                y = self.win_rates[alpha][self.epsilon[0]]
                self._plot_win_rate(x, y, r'$\alpha={}$', alpha, line_colour)
        elif plot_by == 'epsilon':
            for epsilon, line_colour in zip(self.epsilon, self.line_colours[:len(self.epsilon)]):
                y = self.win_rates[self.alpha[0]][epsilon]
                self._plot_win_rate(x, y, r'$\epsilon={}$', epsilon, line_colour)

        title = 'Win rates againt {} strategy with {} V table'
        title += '(sym)' if self.learn_from_symmetry else ''
        plt.title(title.format(self.opponent_strategy, self.vtable_init))

        plt.xlabel('Episodes')
        plt.ylabel('Win rate')
        plt.axis([0, self.episodes, 0, 1.0])
        plt.legend(loc='lower right')
        plt.show()


class Timer:
    """
    Timer class to print elapsed time for an operation.
    Arg(s):
        obj: a string to describe the operation.
    """

    def __init__(self, obj):
        self.obj = obj

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        end = time.time()
        secs = end - self.start
        output_str = '({}) elapsed time: '.format(self.obj)

        if secs >= 60:
            mins = int(self.secs/60)
            secs -= (60 * mins)
            output_str += '{} min(s) {} s'.format(mins, secs)
        else:
            msecs = int(secs * 1000) % 1000
            secs = int(secs)
            output_str += '{} s {} ms'.format(secs, msecs)

        print(output_str)


if __name__ == '__main__':
    l = Learner(game_class=TTT)

    print('Training... this may take a while depending on your parameters...')
    with Timer('Training'):
        l.train()
    l.plot()
