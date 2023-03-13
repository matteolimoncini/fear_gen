from pymdp import utils
import numpy as np


class TrialEnv(object):
    def __init__(self, data):
        """
        Constructor of the environment
        :param data: Dataframe with 4 columns: 'trial_number','morphing_level','condition','shock'

        'trial_number': number of trial in the experiment
        'morphing_level': 1 if cs+, 0 otherwise
        'condition': 1 cs+, 0 otherwise
        'shock': 1 if the shock is given at this trial, 0 otherwise

        """
        self.data = data

    def get_observation(self, param, trial_number):
        '''

        :param param: column of the dataframe that you want to read
        :param trial_number:
        :return: value of the specific column in the trial_number row
        '''
        value = str(self.data.iloc[int(trial_number)][param])

        if param == 'shock':
            if value == '1':
                value = 'shock'
            if value == '0':
                value = 'no_shock'

        return value

    def step(self, action, current_trial):
        '''
        Take a step in the environment given an action

        :param action:
        :param current_trial:
        :return:
        '''
        surprised_obs = None
        if action == 'null':
            return 'null'
        if action == 'guess cs+':
            shock_obs = self.get_observation(param='shock', trial_number=current_trial)

            if shock_obs == 'shock':  # in this case shock and i have predicted cs+ #TODO change this
                # print("reward")
                surprised_obs = self.get_surprised_low()

            else:  # in this case no shock and i have predicted cs+
                # print("penalty")
                surprised_obs = self.get_surprised_high()

        if action == 'guess cs-':
            shock_obs = self.get_observation(param='shock', trial_number=current_trial)

            if shock_obs == 'shock':  # in this case shock and prediction cs- #TODO change this
                print("penalty")
                surprised_obs = self.get_surprised_high()

            else:  # in this case no shock and prediction cs-
                print("reward")
                surprised_obs = self.get_surprised_low()

        if surprised_obs is None:
            print('ERROR: Action not supported!')

        return surprised_obs

    def get_surprised_high(self):
        return 'surprised'

    def get_surprised_low(self):
        return 'not_surprised'
