import numpy as np


def mapToCandide(fiducial, landmarkmode):
    meanModel, tri, mapping, actionV = loadCandideData(landmarkmode)


mean3DShape = x.T
shape2D = y.T

shape3DCentered = mean3DShape - np.mean(mean3DShape, axis=0)
shape2DCentered = shape2D - np.mean(shape2D, axis=0)

scale = np.linalg.norm(shape2DCentered) / np.linalg.norm(shape3DCentered[:, :2])
t = np.mean(shape2D, axis=0) - np.mean(mean3DShape[:, :2], axis=0)

params = np.zeros(self.nParams)
params[0] = scale
params[4] = t[0]
params[5] = t[1]


def loadCandideData(landmarkmode):
    print
    ">> Loading Candide-3 data ..."

    # mean candide-3 model
    meanModel = np.fromfile('data/candide3.dat', dtype=float, sep=" ")
    meanModel = np.reshape(meanModel, [-1, 3])

    # original mesh as face list
    tri = np.fromfile('data/triangles.dat', dtype=int, sep=" ")
    tri = np.reshape(tri, [-1, 3])

    # mean candide-3 model
    mapping = np.fromfile('data/mapping_' + landmarkmode + '.dat', dtype=int, sep=" ")
    mapping = np.reshape(meanModel, [-1, 3])
