def shakespeare():
    import datasets
    return datasets.load_dataset('tiny_shakespeare')["train"][0]["text"]


def simplebooks2():
    return file('../../Downloads/simplebooks/simplebooks-2-raw/train.txt')


def file(path):
    with open(path) as f:
        return f.read()
