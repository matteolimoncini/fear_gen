class Dataset(object):
    """Class to handle the datasets"""

    def __init__(self, name='', signals=[], subjects=[], annotations=[]):
        # name of the dataset
        self.name = name
        # contains the list of signals to extract from the dataset
        self.signals = signals
        # contains the annotation object to extract from the dataset
        self.annotations = annotations
        # considered subjects from the datasets
        self.subjects = subjects

    def __str__(self):
        return ("\nDataset\n- name: %s\n- signals: %s\n- subjects: %s\n- annotations: %s\n" % (
        self.name, [s.name for s in self.signals], [s for s in self.subjects], [s.name for s in self.annotations]))
