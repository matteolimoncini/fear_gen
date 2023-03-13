from pymdp import utils
from pymdp.agent import Agent
import numpy as np
import pandas as pd
from env import TrialEnv

trials = 144
steps = 2  # number of timepoints per trial

# STATES
# state if the img is cs+ or cs-
context = ['cs+', 'cs-']
choice = ['guess cs+', 'guess cs-']
# error = ['low', 'high']


# OBSERVATIONS

# morphing level binary. shock only if morph_level=6
morph_level_obs = ['0', '1']
# the subject receive the electric shock?
shock_obs = ['shock', 'no_shock', 'null']
# observation to discriminate if the subject is surprised or prepared. A sort of diff between user prediction and shock observation
surprised_obs = ['not_surprised', 'surprised', 'null']

# ACTIONS
# the subject predict if should be receive the shock or not
context_action = ['do_nothing']
choice_action = ['guess cs+', 'guess cs-']

# Define `num_states` and `num_factors` below
num_states = [len(context), len(choice)]  # [2,2]
num_factors = len(num_states)  # 2

# Define `num_obs` and `num_modalities` below
num_obs = [len(morph_level_obs), len(shock_obs), len(surprised_obs)]  # [2,2,2]
num_modalities = len(num_obs)  # 3


def matrices():
    """
    The A, B, C, D, E matrices are built.
    """
    # This is the likelihood mapping that truly describes the relationship between the
    # environment's hidden state and the observations the agent will get

    # A matrix
    # A dimension?
    A = utils.obj_array(num_modalities)
    A_morph = np.zeros((len(morph_level_obs), len(context), len(choice)))
    A_shock = np.zeros((len(shock_obs), len(context), len(choice)))
    A_surprise = np.zeros((len(surprised_obs), len(context), len(choice)))

    for j in range(len(choice)):
        # mor_lev_binary/states #cs+ #cs-
        A_morph[:, :, j] = np.array([[0, 1],  # img or
                                     [1, 0]])  # img minac
        # shock-noshock/states
        A_shock[:, :, j] = np.array([[0.75, 0],
                                     [0.25, 1],
                                     [0, 0]])
        # surprise/states
        A_surprise[:, j, j] = np.array([1, 0, 0])
        A_surprise[:, j, int(not j)] = np.array([0, 1, 0])

    A[0] = A_morph
    A[1] = A_shock
    A[2] = A_surprise

    # B matrix
    # The transition mapping that truly describes the environment's dynamics

    # B as an identity matrix. no state transitions
    B = utils.obj_array(num_factors)
    B_context = np.zeros((len(context), len(context), len(context_action)))

    B_context[:, :, 0] = np.eye(len(context))
    # B_context[:,:,2] = np.eye(len(context))

    B_choice = np.zeros((len(choice), len(choice), len(choice_action)))
    for choice_i in range(len(choice)):
        B_choice[choice_i, :, choice_i] = 1.0

    B[0] = B_context
    B[1] = B_choice

    # This is the matrix representing the preferences over outcomes
    # prepared-surprised/cs+-
    C = utils.obj_array_zeros(num_obs)  # num modalities
    C[0] = np.zeros(2)
    C[1] = np.zeros(3)
    C[2] = np.array([2, -3, 0])

    return A, B, C


# fake data generation
morphing_levels = [0] * 36 + [1] * 36 + [0] * 36 + [1] * 36
shocks = [0] * 36 + [1] * 36 + [0] * 36 + [1] * 36
fake_data_half = pd.DataFrame({'morphing level': morphing_levels, 'shock': shocks})

# define an agent
# E = utils.obj_array(1)
# E[0] = np.array([.4,.2,.4])
A, B, C = matrices()
my_agent = Agent(A=A, B=B, C=C, gamma=16)
qs0 = my_agent.qs
# define an environment
my_env = TrialEnv(data=fake_data_half)
list_qs = []
list_action = []

initial_action = 'first_action'

# iterate over all trials
for trial in range(trials):
    print("\n-----TRIAL " + str(trial) + " ------")
    agent_stimul_obs, agent_shock_obs, agent_surpr_obs = 'null', 'null', 'null'
    agent_stimul_obs = my_env.get_observation(param='morphing level', trial_number=trial)
    print('observation (t=0): ', agent_stimul_obs, agent_shock_obs, agent_surpr_obs)
    obs = [morph_level_obs.index(agent_stimul_obs), shock_obs.index(agent_shock_obs),
           surprised_obs.index(agent_surpr_obs)]
    qs = my_agent.infer_states(obs)  # agent update beliefs about hidden states given observations
    print('beliefs over states (t=0):', qs[0] * 100)

    policies = my_agent.infer_policies()  # inferring policies and sampling actions from the posterior
    agent_action = my_agent.sample_action()
    action_name = choice_action[int(agent_action[1])]
    print('beliefs over actions: ', policies[0] * 100)
    print('action: ', action_name)

    agent_surpr_obs = my_env.step(action_name, trial)
    agent_shock_obs = my_env.get_observation(param='shock', trial_number=trial)
    print('observation (t=1): ', agent_stimul_obs, agent_shock_obs, agent_surpr_obs)
    obs = [morph_level_obs.index(agent_stimul_obs), shock_obs.index(agent_shock_obs),
           surprised_obs.index(agent_surpr_obs)]
    qs = my_agent.infer_states(obs)  # agent update beliefs about hidden states given observations
    print('beliefs over states (t=1):', qs[0] * 100)
    # my_agent.reset()
