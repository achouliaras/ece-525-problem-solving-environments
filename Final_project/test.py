import lightgbm as lgb
import pandas as pd
from gensim.models import FastText
from sklearn.metrics import classification_report

from utils.word_embeddings import transform_text_to_embedding
from utils.config import *

fm = FastText.load(str(MODELS_PATH / MODEL_NAME_FAST_TEXT))
data = pd.read_csv(PATH / "initial_data.csv")
data = data.sample(frac=1)


y = data["label"]
url_list = data["url"]
X = transform_text_to_embedding(url_list, fm)


print("after training")
model1 = lgb.Booster(params, model_file=str(MODELS_PATH / MODEL_NAME_LGB))

y_pred1 = model1.predict(X)



print(classification_report(y,y_pred1.round()))

print("initial model")
model2 = lgb.Booster(params, model_file=str(MODELS_PATH / ('initial'+MODEL_NAME_LGB)))

y_pred2 = model1.predict(X)



print(classification_report(y,y_pred2.round()))