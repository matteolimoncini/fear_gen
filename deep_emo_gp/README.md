# Deep Emo GP

Deep construction of an affective latent space via multimodal enactment.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing
purposes.

### Prerequisites

The project requires the following packages:

- cvxopt>=1.1.9
- PyWavelets>=0.5.2
- biosppy>=0.5.1
- deepgp

They can be easily installed with:

```
pip install -r requirements.txt
```

### Installing

Once installed all the prerequisites, simply type:

```
pip install deepemogp
```

to install the deepemogp package.

A simple demo can be found in `dEmo.py` where the extraction of EDA and ECG signals from RECOLA dataset is performed,
followed by the learning of a Deep Gaussian Process model.

```python
import deepemogp.feature_extractor as feature_extractor
import deepemogp.signal.physio as physio
import deepemogp.signal.annotation as annotation
import deepemogp.datasets as datasets

show = True

# definition of the feature extractors to be used later
f1 = feature_extractor.FE('wavelet', skip=2)
f2 = feature_extractor.FE('wavelet', window=(8,6), params={'w_mother':'db4'})
f3 = feature_extractor.FE('mean', skip=2)

# definition of the physiological signals to be extracted
eda = physio.EDA(f1)
ecg = physio.ECG(f2)

# definition of the emotional annotation to be extracted
va = annotation.VA('valence', f3)
ar = annotation.VA('arousal', f3)

# extraction of the desired data from the dataset
d = datasets.RECOLA(signals={eda, ecg}, subjects={'P21'}, annotations={va, ar})
print d

for s in d.signals:
    # preprocess ...
    s.preprocess(show=show)

    # ... and extract features from each signal type   
    s.feature_ext.extract_feat(s, show=show)
    print s

for a in d.annotations:
    # preprocess ...
    a.preprocess(show=show)

    # ... and extract features for each annotation type
    a.feature_ext.extract_feat(a, show=show)
    print a

# TO COMPLETE ...

```

## Modules

Explain how the components are modelled.

### FeatureExtractor

`FeatureExtractor` is the class that handle all the feature extraction methods.  
It consists of the following self-explaining class instance attributes:

```python
# name of the feature extractor method
self.name

# processing window / overlapping length (in seconds)
# e.g. (4, 2) : 4 seconds window with 2 seconds overlap
self.window

# number of feature samples to skip from start. useful to align signals
# when different processing windows are selected
self.skip

# additional parameters specific for the extractor
self.params
```

It currently supports:

- `'raw'` - when no feature extraction is needed
- `'mean'` - considers the mean value
- `'wavelet'` - implements the Discrete Wavelet Transform data decomposition

In case of `'wavelet'` the accepted additional parameters are:

- `'w_mother'` - Short name of the desired wavelet family to use (default: `'db3'`, for a complete list,
  see: http://pywavelets.readthedocs.io/en/latest/ref/wavelets.html)
- `'w_maxlev'` - Decomposition level (must be >= 0) (default: it will be calculated the maximum useful level of
  decomposition.)

Example:

```python
f = emo.physio.utils.FeatureExtractor('wavelet', window=(8,6), params={'w_mother':'db4'})
```

creates a wavelet feature extractor with a moving window of 8 seconds, overlap of 6 seconds and wavelet family
Daubechies 4

**only** reversible transformations are taken in consideration, since is required the regeneration of the original
signal.
When adding a new FeatureExtractor technique is needed to provide the following two methods:

- `apply_xxx()` - performs the actual feature extraction starting from a signal
- `reconstruct_xxx()` - performs the generation of the signal starting from a feature

### Signal

`Signal` is a superclass that handles all the signals.
It consists of the following class instance attributes:

```python
# name of the considered signal
self.name
# contains the raw data sampling rate
self.raw_fps
# contains the raw data
self.raw
# contains the processed data sampling rate
self.processed_fps
# contains the processed data
self.processed
# contains the extracted features
self.features
# instance of the feature extraction method
self.feature_ext
```

Currently it is inherited by the following classes:

- `EDA` - class to handle the Electro Dermal Activity signal
- `ECG` - class to handle the Electro CardioGram signal
- `SKT` - class to handle the SKin Temperature signal
- `BVP` - class to handle the Blood Volume Pulse signal
- `FP_OF` - class to handle the facial fiducial points extracted via Openface framework

- `VA` - class to handle the Valence/Arousal dimensional annotation signal

both provide a `preprocess()` method responsible of the preprocessing of the specific signal (e.g.: denoise, detrend...)

### Dataset

`Dataset` is a superclass that handles all the datasets.
It consists of the following class instance attributes:

```python
# name of the dataset
self.name = name
# contains the list of signals to extract from the dataset
self.signals = signals
# considered subjects from the datasets
self.subjects = subjects
# load emotional annotations or not
self.loadAnno = loadAnno
```

Currently it is inherited by the following classes:

- `RECOLA` - class to handle the RECOLA dataset

  from https://diuf.unifr.ch/diva/recola/

  F. Ringeval, A. Sonderegger, J. Sauer and D. Lalanne,
  "Introducing the RECOLA Multimodal Corpus of Remote Collaborative and Affective Interactions",
  2nd International Workshop on Emotion Representation, Analysis and Synthesis in Continuous Time and Space (EmoSPACE),
  in Proc. of IEEE Face & Gestures 2013, Shanghai (China), April 22-26 2013.

- `AMHUSE` - class to handle the AMHUSE dataset

  from http://amhuse.phuselab.di.unimi.it/

  G. Boccignone, D. Conte, V. Cuculo and R. Lanzarotti.
  "AMHUSE: A Multimodal dataset for HUmour SEnsing",
  in Proc. of 19th ACM International Conference on Multimodal Interaction.
  (ICMI'17), ACM. Glasgow (UK).
  DOI: 10.1145/3136755.3136806

## Authors

**DeepEmoGP** is a [PHuSe Lab](http://phuselab.di.unimi.it/) project.

* **Vittorio Cuculo** - *Initial work* - [vcuculo](https://github.com/vcuculo)

See also the list of [contributors](AUTHORS.txt) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

