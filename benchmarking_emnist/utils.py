import pickle

def split_and_insert_newline(input_string):
    return '_'.join([p + ('\n' if (i + 1) % 2 == 0 else '').strip("_") for i, p in enumerate(input_string.split('_'))])

def split_and_insert_newline_list(input_strings):
    if type(input_strings) == (str):
        return split_and_insert_newline(input_strings)
    output_strings = []
    for input_string in input_strings:
        print(input_string)
        output_strings.append(split_and_insert_newline(input_string))
    return output_strings

def get_dict_from_pkl(file_name):
    with open(f'{file_name}.pkl', 'rb') as f:
        metrics_global = pickle.load(f)
    return metrics_global

def save_dict_to_pkl(dict_obj, file_name):
    with open(f'{file_name}.pkl', 'wb') as f:
        pickle.dump(dict_obj, f, pickle.HIGHEST_PROTOCOL)