# TensorFlow is much easier to install using Anaconda, especially
# on Windows or when using a GPU. Please see the installation
# instructions in INSTALL.md

##### Core scientific packages
jupyter==1.0.0
matplotlib==3.3.4
numpy==1.22.0
pandas==1.2.2
scipy==1.6.0

##### Machine Learning packages
scikit-learn==0.24.1

# Optional: the XGBoost library is only used in chapter 7
xgboost==1.3.3

##### TensorFlow-related packages

# If you want to use a GPU, it must have CUDA Compute Capability 3.5 or
# higher support, and you must install CUDA, cuDNN and more: see
# tensorflow.org for the detailed installation instructions.

tensorflow==1.15.5 # or tensorflow-gpu==1.15.5 for GPU support

tensorboard==1.15.0

##### Reinforcement Learning library (chapter 16)

# There are a few dependencies you need to install first, check out:
# https://github.com/openai/gym#installing-everything
gym[atari,Box2D]==0.18.0
# On Windows, install atari_py using:
# pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py

##### Image manipulation
Pillow==9.0.1
graphviz==0.16
pyglet==1.5.0
scikit-image==0.18.1

#pyvirtualdisplay # needed in chapter 16, if on a headless server
                  # (i.e., without screen, e.g., Colab or VM)


##### Additional utilities

# Efficient jobs (caching, parallelism, persistence)
joblib==0.14.1

# Nice utility to diff Jupyter Notebooks.
nbdime==2.1.1

# May be useful with Pandas for complex "where" clauses (e.g., Pandas
# tutorial).
numexpr==2.7.2

# Optional: these libraries can be useful in the classification chapter,
# exercise 4.
nltk==3.6.6
urlextract==1.2.0

