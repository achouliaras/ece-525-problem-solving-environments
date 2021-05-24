import pickle


def load_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_object(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
