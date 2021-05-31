import os

file_name = 'model_param.txt'

model_dict = {}

module_list = ['audio_transform', 'audio_encoder', 'audio_decoder', 'text_encoder', 'text_decoder', 'prosody_predictor',
               'asr', 'se', 'sc', 'tts', 'vcb',
               'log_sigma_asr', 'log_sigma_se', 'log_sigma_sc', 'log_sigma_tts', 'log_sigma_vcb']


with open(file_name, 'r') as f:
    for line in f:
        param_name = line.strip()
        model_dict[param_name] = []
        not_in = True
        for module in module_list:
            if module == param_name[:len(module)]:
                model_dict[param_name].append(module)
                not_in = False
        if not_in:
            print(f'!!!!!!!!!!!!!!NOT IN: {param_name}')

for param_name in model_dict:
    if len(model_dict[param_name]) > 1:
        print(f'???????????{param_name}:                          {model_dict[param_name]}')
    else:
        print(f'{param_name}:                          {model_dict[param_name]}')
