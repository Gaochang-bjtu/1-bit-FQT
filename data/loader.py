from data.flowers import load_flowers102
from data.food import load_food101
from data.pets import load_pets37
from data.cars import load_cars196
from data.CUB import load_cub200
def load(name, train_batch_size, test_batch_size):
    if name == 'flower':
        return load_flowers102(train_batch_size, test_batch_size)
    elif name == 'food':
        return load_food101(train_batch_size, test_batch_size)
    elif name == 'pet':
        return load_pets37(train_batch_size, test_batch_size)
    elif name == 'car':
        return load_cars196(train_batch_size, test_batch_size)
    elif name == 'cub':
        return load_cub200(train_batch_size, test_batch_size)
