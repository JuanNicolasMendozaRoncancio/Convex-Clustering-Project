# test_boosting.py
import numpy as np
from boosting import Boosting

# Generar datos de ejemplo
r = np.array([3, -0.5, 2, 7])
print(r.shape)
print(r[:,None].shape)