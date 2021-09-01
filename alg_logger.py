from alg_constrants_amd_packages import *

# Neptune.ai Logger
PARAMS = {
    'GAMMA': GAMMA,
    # 'LR': LR,
    # 'CLIP_GRAD': CLIP_GRAD,
    'ENV': ENV,
    'MAX_EPOCHS': MAX_EPOCHS,
}

if NEPTUNE:
    run = neptune.init(project='1919ars/PL-implementations',
                       tags=['DDPG', ENV, f'{MAX_EPOCHS} epochs'],
                       name=f'DDPG_{time.asctime()}',
                       source_files=['alg_constrants_amd_packages.py'])
else:
    run = {}


run['parameters'] = PARAMS