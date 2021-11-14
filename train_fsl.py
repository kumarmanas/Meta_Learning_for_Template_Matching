import numpy as np
import torch
from model.trainer.fsl_trainer import FSLTrainer
from model.utils import (
    pprint, set_gpu,
    get_command_line_parser, 
    postprocess_args,
)
# from ipdb import launch_ipdb_on_exception
#placed csv and dataset path here to make easy to run model on inference
#change below path based on location of image and csv file of split
image_path =  '/dataset/images'
split_path =  '/data/scanimage/split'
test_path=    '/data/scanimage/split/test_seen_with_unseen'
if __name__ == '__main__':

    parser = get_command_line_parser()
    args = postprocess_args(parser.parse_args())
    # with launch_ipdb_on_exception():
    pprint(vars(args))

    set_gpu(args.gpu)
    trainer = FSLTrainer(args)
    #trainer.train()
    print('training done!')
    trainer.evaluate_test()
    print('testing done!')
    #trainer.evaluate_test_demonstration() # use this only for demonstation purpose for showing live prediction of image
    trainer.final_record()
    print('record generated')
    print('Tested CSV file used for testing:',test_path)
    print(args.save_path)



