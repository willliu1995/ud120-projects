#%%
from nltk.corpus import stopwords
import numpy as np
import pandas as pd

#%%
sw = stopwords.words("english")
len(sw)

#%%
sw_df = pd.DataFrame(sw)

#%%
sw_df.info()

#%%
sw_df