def split_and_insert_newline(input_string):
    return '_'.join([p + ('\n' if (i + 1) % 2 == 0 else '').strip("_") for i, p in enumerate(input_string.split('_'))])

def split_and_insert_newline_list(input_strings):
    new_input_strings = []
    for i, input_string in enumerate(input_strings):
        new_input_strings[i] = split_and_insert_newline(input_string)
    return new_input_strings