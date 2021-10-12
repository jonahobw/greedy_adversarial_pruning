import os
import logging

logging.getLogger('utils')

def check_folder_structure(experiment_number, dataset, model_type, quantize, prune, attack):
    root = os.getcwd()

    path_dict = {
        'root': root,
        'datasets': os.path.join(root, 'datasets'),
        'experiments': os.path.join(root, 'experiments'),
        'experiment': os.path.join(root, 'experiments', f"experiment_{experiment_number}")
    }
    path_dict['dataset'] = os.path.join(path_dict['datasets'], dataset)
    path_dict['model_type'] = os.path.join(path_dict['experiment'], model_type)
    path_dict['model_dataset'] = os.path.join(path_dict['model_type'], dataset)
    if quantize:
        path_dict['model'] = os.path.join(path_dict['model_dataset'], f"{model_type.lower()}_{quantize}_quantization")
    elif prune:
        path_dict['model'] = os.path.join(path_dict['model_dataset'],
                                          f"{model_type.lower()}_{prune}_pruning")
    else:
        path_dict['model'] = os.path.join(path_dict['model_dataset'], model_type.lower())

    if attack and 'model' in path_dict:
        path_dict['attacks'] = os.path.join(path_dict['model'], 'attacks')
        path_dict['attack'] = os.path.join(path_dict['attacks'], attack)

    for folder_name in path_dict:
        if not os.path.exists(path_dict[folder_name]):
            os.mkdir(path_dict[folder_name])

    return path_dict
