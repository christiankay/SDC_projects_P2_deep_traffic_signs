# -*- coding: utf-8 -*-


import random
import numpy as np
import matplotlib.pyplot as plt

# matplotlib inline

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

plt.figure(figsize=(1,1))
plt.imshow(image, cmap="gray")
print(y_train[index])