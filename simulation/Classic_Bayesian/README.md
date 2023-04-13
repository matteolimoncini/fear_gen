Development of a complete rational agent using `pyro`. The task is to build a complete rational agent
without any affective component, that takes part in the original experiment and tries to predict the 
shock expectancy rating observing a set of visual stimuli.

In particular, the agent observes the binarized morphing level of the visual stimulus and predict a binary shock expectancy
rating.
We developed three types of models:
* Agent with full memory: the agent learns what to predict observing all trials
* Agent with partial memory: the agent has a partial memory and it "remembers" only the last $k$ trials
* Agent with train/test split: in this last model we want to simulate the three phases of the original experiment by Reutter.
The habituation trials have been removed since the electrical shock was not applied; the agent learns the trials relative to the fear acquisition phase and tries to predict the shock rating relative to the fear acquisition phase.

The models can be seen in [full memory](./output_analysis.ipynb), [partial memory](./category_learning/sliding_window/pyro_agent_sliding_window.ipynb) and in [train test](pyro_traintest.ipynb) respectively