
# coding: utf-8

# In[2]:

import os
from keras.utils.vis_utils import plot_model
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'


# In[3]:

def model_to_png(model, name):
    plot_model(model, to_file= name + '.png', show_shapes=True, show_layer_names=True)


# In[ ]:



