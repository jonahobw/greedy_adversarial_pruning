import utils
import torchvision
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class Experiment:

    def __init__(self, experiment_number, dataset, model_type, quantize=None, prune=None, attack=None, kwargs=None):
        paths = utils.check_folder_structure(experiment_number, dataset, model_type, quantize, prune, attack)
        cf = torchvision.datasets.CIFAR10(paths['dataset'], download=True)


if __name__ == '__main__':
    for model_type in ['VGG', 'GoogLeNet', 'MobileNetV3', 'ResNet18']:
        a = Experiment(0, 'cifar10', model_type, prune='full')