import json

from kafka import KafkaProducer

producer = KafkaProducer(bootstrap_servers='localhost:9092')


def publish_prediction(pred, url, request_id):
    producer.send('app_messages', json.dumps({'request_id': request_id, 'prediction': pred, 'url': url}).encode('utf-8'))
    producer.flush()


def publish_training_completed():
    producer.send('retrain_topic', json.dumps({'training_completed': True}).encode('utf-8'))
    producer.flush()


def read_messages_count(path, repeat_every):
    file_list = list(path.iterdir())
    nfiles = len(file_list)
    if nfiles == 0:
        return 0
    else:
        return ((nfiles - 1) * repeat_every) + len(file_list[-1].open().readlines())


def append_message(message, path, batch_id):
    message_fname = 'messages_{}_.txt'.format(batch_id)
    f = open(path / message_fname, "a")
    f.write("%s\n" % (json.dumps(message)))
    f.close()


def send_retrain_message(batch_id):
    producer.send('retrain_topic', json.dumps({'retrain': True, 'batch_id': batch_id}).encode('utf-8'))
    producer.flush()


def is_retraining_message(msg):
    message = json.loads(msg.value)
    return msg.topic == 'retrain_topic' and 'training_completed' in message and message['training_completed']


def is_application_message(msg):
    message = json.loads(msg.value)
    return msg.topic == 'app_messages' and 'prediction' not in message
