# CNN Based Meta-Learning for Noisy Image Classification and Template Matching

## Introduction

This master thesis used a few-shot meta learning approach to solve the problem of open-set template matching. In this thesis, template matching is treated as a classification problem, but having availability of just template as class representative. Work is based on non-parametric approach of meta-learning [Prototypical Network](https://arxiv.org/abs/1703.05175) and [FEAT](https://github.com/Sha-Lab/FEAT). Thesis report is available [here](https://drive.google.com/drive/folders/1Utnm5lZp2NL4AhwqZJKfRbIa-7vcI45L?usp=sharing).

## Installation

Running this code requires:

1. [PyTorch and TorchVision](https://pytorch.org/). Tested on version 1.8
2. Numpy
3. [TensorboardX](https://github.com/lanpa/tensorboardX) for visualization of results
4. Initial weights to get better accuracy is stored in [Google-drive](https://drive.google.com/drive/folders/16QzI9kJZpIIQ079eYzUy55pAjIRMt-VK?usp=sharing). These weights will allow faster convergence of training. Weights are obtained using pre-training on mini-Imagenet dataset.
5. Dataset: Dataset is private in this thesis. But can be replaced with own custom dataset or [mini-Imagenet](https://drive.google.com/file/d/16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY/view) or [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).

## Dataset structure

Dataset structure will follow the other few-shot learning(FSL) benchmark as used in [Prototypical Network](https://arxiv.org/abs/1703.05175) or [FEAT](https://github.com/Sha-Lab/FEAT). For this thesis, custom dataset is used. In this dataset, a clean template image is used as a template and using this template a single shot learning model learn the class representation. Then we have other images which belongs to same template and they are classified as same class as in FSL. In dataset which is split in train, val and test, the first row of each class in CSV file should be a clean template and rest can be noisy images. The job of model is to pick one noisy image and classify them into a specific template/class, where model learned the class representation from one clean template. In original FSL model, they don't fix templates as first row in each class in CSV, as they do classification not template matching. 
If you want to test this model for template matching, you can replace dataset with public dataset [mini-Imagenet](https://drive.google.com/file/d/16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY/view) or [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html). But in this case first image of each class will be treated as template, but nevertheless it can give you idea how FSL model work in template matching domain. 

## Code Structures
This model used [Prototypical Network](https://arxiv.org/abs/1703.05175) and [FEAT](https://github.com/Sha-Lab/FEAT) model as base structure. Then these modes are modified for template matching and this is documented along the code structure for changes. Additionally, novel distance function is used which differs from above two SOTA models and codes are modified to incorporate these new distance function. 
To reproduce the result run **train_fsl.py**. By default, **train_fsl.py** commented the training part of code, so you can uncomment it to train them on custom dataset. There are four parts in the code.
 - `model`: It contains the main files of the code, including the few-shot learning trainer, the dataloader, the network architectures, and baseline and comparison models.
 - `data`:  Can be used with public dataset or custom one. Splits can be taken as per [Prototypical Network](https://arxiv.org/abs/1703.05175) or based on new use case.
 - `saves`: The pre-trained initialized weights of ConvNet, Res-12,18 and 50.

## Model Training and Testing
Use file name `train_fsl.py` to start the training, make sure command "trainer.train()" is not commented. 
Training parameters can be either changed in `model/utils.py` file or these parameters can be passed as command line argument. 

Use file name `train_fsl.py` to start the testing, but this time comment the command "trainer.train()".

Note: in file ``train_fsl.py`` three variable contains the path of dataset and CSV file-
* image_path: This is the path of the folder where images are kept.
* split_path: Path where training and validation CSV is stored.
* test_path: Complete path of testing CSV file without .csv extension.

**Task Related Arguments (taken and modified from [FEAT](https://arxiv.org/abs/1812.03664) model)**
- `dataset`: default ScanImage used in this project. Other option can be selected based on your own dataset name.

- `way`: The number of templates/classes in a few-shot task during meta-training, default to `5`. N Templates can be treated as N class. 

- `eval_way`: The number of templates/classes in a few-shot task during meta-test, default to `5`. This indicates that no. of possible templates/classes in which a scanned image can be matched into.

- `shot`: Number of instances in each class in a few-shot task during meta-training, default to `1`. For template matching, shot will be always 1 as we will have only 1 template or one image from each class.

- `eval_shot`: Number of instances in each class in a few-shot task during meta-test, default to `1`. For template matching, shot will be always 1 as we will have only 1 template or one image from each class.

- `query`: Number of instances of image at one go in each episode which needs to be matched with template or classified into one of the template. This is to evaluate the performance during meta-training, default to `15`

- `eval_query`: Number of instances of image at one go in each episode which needs to be matched with template or classified into one of the template. This is to evaluate the performance during meta-testing, default to `15`

**Optimization Related Arguments**
- `max_epoch`: The maximum number of training epochs, default to `2`

- `episodes_per_epoch`: The number of tasks sampled in each epoch, default to `100`

- `num_eval_episodes`: The number of tasks sampled from the meta-val set to evaluate the performance of the model (note that we fix sampling 10,000 tasks from the meta-test set during final evaluation), default to `200`

- `lr`: Learning rate for the model, default to `0.0001` with pre-trained weights

- `lr_mul`: This is specially designed for set-to-set functions like FEAT. The learning rate for the top layer will be multiplied by this value (usually with faster learning rate). Default to `10`

- `lr_scheduler`: The scheduler to set the learning rate (`step`, `multistep`, or `cosine`), default to `step`

- `step_size`: The step scheduler to decrease the learning rate. Set it to a single value if choose the `step` scheduler and provide multiple values when choosing the `multistep` scheduler. Default to `20`

- `gamma`: Learning rate ratio for `step` or `multistep` scheduler, default to `0.2`

- `fix_BN`: Set the encoder to the evaluation mode during the meta-training. This parameter is useful when meta-learning with the WRN. Default to `False`

- `augment`: Whether to do data augmentation or not during meta-training, default to `False`

- `mom`: The momentum value for the SGD optimizer, default to `0.9`

- `weight_decay`: The weight_decay value for SGD optimizer, default to `0.0005`

**Model Related Arguments (taken and modified from [FEAT](https://arxiv.org/abs/1812.03664) model)**
- `model_class`: Select if we are going to use Prototypical Network or FEAT network. Default to `FEAT`. Other option is `ProtoNet`

- `use_euclideanWithCosine`: if this is set to true then distance function to compare template embedding and image is used is a weighted combination of euclidean distance + cosine similarity. Default calue is `False`
- `use_euclidean`: Use the euclidean distance. Default to `True`. When set as False then cosine distance is used

- `backbone_class`: Types of the encoder, i.e., the convolution network (`ConvNet`), ResNet-12 (`Res12`), or ResNet-18 (`Res18`) or ResNet-50(`Res50`), default to `Res12`

- `balance`: This is the balance weight for the contrastive regularizer. Default to `0`

- `temperature`: Temperature over the logits, we #divide# logits with this value. It is useful when meta-learning with pre-trained weights. Default to `64`. Lower `temperature` faster convergence but less accurate

- `temperature2`: Temperature over the logits in the regularizer, we divide logits with this value. This is specially designed for the contrastive regularizer. Default to `64`. Lower `temperature` faster convergence but less accurate

**Other Arguments** 
- `orig_imsize`: Whether to resize the images before loading the data into the memory. `-1` means we do not resize the images and do not read all images into the memory. Default to `-1`

- `multi_gpu`: Whether to use multiple gpus during meta-training, default to `False`

- `gpu`: The index of GPU to use. Please provide multiple indexes if choose `multi_gpu`. Default to `0`

- `log_interval`: How often to log the meta-training information, default to every `50` tasks

- `eval_interval`: How frequently to validate the model over the meta-val set, default to every `1` epoch

- `save_dir`: The path to save the learned models, default to `./checkpoints` 

- `iterations`: How many times model is evaluated in test time. Higher the better, due to less bias in results. Default to `100`

## Training scripts for FEAT

For example, to train the 1-shot 39-way FEAT model with ResNet-12 backbone on our custom dataset scanImage with euclidean distance as distance measure:

    $ python train_fsl.py  --max_epoch 220 --model_class FEAT  --backbone_class Res12 --dataset ScanImage --way 38 --eval_way 39 --shot 1 --eval_shot 1 --query 15 --eval_query 1 --balance 1 --temperature 64 --temperature2 64 --lr 0.0002 --lr_mul 10 --lr_scheduler step --step_size 40 --gamma 0.5 --init_weights ./saves/initialization/scanimage/Res12-pre.pth --eval_interval 1 --use_euclidean --save_dir './saves' --multi_gpu --gpu 0 --iterations 3000 --num_workers 12
This command can be also be used to test the template matching model just change the eval_way as per number of target template at inference time. Then model will automaticaaly parse the final weight after training. As weight file name and folder is based on train time parameter name.

## Note:
Since the dataset right now is private, in future if things changes we can release the datset as well.  However, our final training weights are stored  with file name `ScanImage-FEAT-Res12-38w01s15q-Pre-DIS` in [Google drive](https://drive.google.com/drive/folders/16QzI9kJZpIIQ079eYzUy55pAjIRMt-VK?usp=sharing).
## Acknowledgment
Following repo codes, functions and research work were leveraged to develop this work package.
- [ProtoNet](https://github.com/cyvius96/prototypical-network-pytorch)

- [FEAT](https://github.com/Sha-Lab/FEAT)

- [Transformer](https://github.com/jadore801120/attention-is-all-you-need-pytorch)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss the changes.

## License
[MIT](https://choosealicense.com/licenses/mit/)
