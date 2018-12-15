import argparse

arg_lists = []
parser = argparse.ArgumentParser(description='Siamese Network')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# data params
data_arg = add_argument_group('Data Params')
data_arg.add_argument('--use_gpu', action='store_true', default=False, help='Use GPU')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

test_parser = argparse.ArgumentParser(description='Model Tester')
test_parser.add_argument('--id_path', type=str, required=True, help='ID Images of users to be used for comparison')
test_parser.add_argument('--data_path', type=str, required=True, help='Dataset to be tested on')

def get_dir_test_config():
    config, unparsed = test_parser.parse_known_args()
    return config, unparsed