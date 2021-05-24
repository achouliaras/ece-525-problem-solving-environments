from pathlib import Path

# Kafka configuration
KAFKA_HOST = 'localhost:9092'
RETRAIN_TOPIC = 'retrain_topic'
TOPICS = ['app_messages', 'retrain_topic']

# Paths and files
PATH = Path('data/')
MODELS_PATH = PATH / 'models'
MESSAGES_PATH = PATH / 'messages'
TRAIN_PATH = PATH / 'train'

TRAIN_DATA = PATH / 'train/train.csv'

MODEL_NAME_LGB = 'model.lgb'
MODEL_NAME_FAST_TEXT = 'model.ftxt'

# Model configuration
params = {
	'boosting_type': 'gbdt',
	'objective': 'binary',
	'metric': ['binary_logloss', 'auc'],
	'learning_rate': 0.1,
	'is_unbalance': True,
}

num_iterations = 100
RETRAIN_EVERY = 2000

# Fast text
vector_size = 100
epochs = 30
