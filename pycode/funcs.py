import os
import pickle
import shutil

from datetime import datetime

colors = {'w': '\033[38m',
          'r': '\033[31m',
          'g': '\033[33m',
          'y': '\033[33m',
          'b': '\033[34m',
          'p': '\033[35m',
          'l': '\033[36m',
          'gray': '\033[37m'}

def save_loc():
    uname = datetime.today().strftime('%y-%m-%d') + '_1'
    model_day_number = 2

    if not(os.path.exists(f'/home/iwamelink/projects/GLIOCARE/Synthetic_T1/models/logs_{uname}/')):
        os.mkdir(f'/home/iwamelink/projects/GLIOCARE/Synthetic_T1/models/logs_{uname}/')
    else:
        override = input('{}Do you want to override the file? (y = yes, n = no){}'.format(colors['p'], colors['w']))
        if override == 'n':
            while os.path.exists(f'/home/iwamelink/projects/GLIOCARE/Synthetic_T1/models/logs_{uname}/'):
                uname = uname[:-1] + str(model_day_number)
                model_day_number += 1
            os.mkdir(f'/home/iwamelink/projects/GLIOCARE/Synthetic_T1/models/logs_{uname}/')
        elif override == 'y':
            print('Overriding')
            shutil.rmtree(f'/home/iwamelink/projects/GLIOCARE/Synthetic_T1/models/logs_{uname}/')
            os.mkdir(f'/home/iwamelink/projects/GLIOCARE/Synthetic_T1/models/logs_{uname}/')

    return uname

def save_model_info(uname, unet_params, datagen_params):
    with open(f'/home/iwamelink/projects/GLIOCARE/Synthetic_T1/models/logs_{uname}/model_datagen_params.pickle', 'wb') as handle:
        pickle.dump([unet_params, datagen_params], handle, protocol=pickle.HIGHEST_PROTOCOL)
    del handle

    # Copy entire code folder from current training to the save location. Symlinks is set to True (save metadate such as last time modified)
    shutil.copytree('/home/iwamelink/projects/GLIOCARE/Synthetic_T1/pycode', f'/home/iwamelink/projects/GLIOCARE/Synthetic_T1/models/logs_{uname}/pycode', symlinks=True)