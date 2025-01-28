# Debugging XAI - Can XAI methods detect confounding variables?

## 1. Setup
Install packages from the ``environment.yml`` file - we use conda, but pipenv or venv will work too.

``
conda env create --name envname --file=environment.yml
``

## 2. Data generation
Currently the data can be downloaded internally in the correct structure on MS Teams, or also from [Kaggle](https://www.kaggle.com/datasets/tongpython/cat-and-dog), albeit in a different file structure (we combine all of the training and testing images from the original source into the structure ./images/dog/ or ./images/cat/ and do the train/val/test splits afterwards). 

To generate the data for experiments, use the following scripts:

### Watermark experiments
``
python generate_watermarks.py {split} {rescaled} 
``

where ``split`` is an integer [0..10] describing which 'shuffling' of the data should be used for generation (via random seeds). ``rescaled`` is a string, either ``'rescaled'`` to rescale the data to the range ``[-1,1]`` or blank/anything else to use the default min-max scaling to the range [0,1].

For variable-position watermarks, use:

``
python generate_watermarks_variable.py {split} {rescaled} 
``

where the same parameterisation applies as above.

### COCO Lightness Experiments
Requires the [COCO 2017 train and val data as well as annotations](https://cocodataset.org/#download) placed in the directory structure as 

```
project  
│
└───coco
│   │
│   └─── annotations
│       │   captions_train2017.json
│       │   ...
│   └─── train2017
│       │   000000000009.jpg
│       │   ...
│   └─── val2017
│       │   000000000139.jpg
│       │   ...

```

Then, run the generation script as:

``
python generate_coco_splits.py {split} {rescaled} 
``

Where the same parameterisation for split (shuffling of the data via random seeds) and rescaling apply. Here, the rescaling to [-1,1] (if specified) is done to the HLS lightness channel only.

## 3. Model Training

### Watermark experiments 
Currently this is only pushed to git for the non-rescaled variants, but this will be updated in due course.

``
python train_server_splits.py {split} {all|confounder|suppressor|no_watermark} 
``

or 

``
python train_server_variable.py {split} {all|confounder|suppressor|no_watermark} 
``

where the first argument is, as before, the split of data to train over, and the second argument refers to whether to train ``all`` three model variants, or just one of ``confounder``, ``suppressor``, or ``no_watermark``. ``train_server_variable.py`` is for the variable-position watermark, though in the future this should be unified to one script.

### COCO lightness experiments

``
python coco_train_server_splits.py {split} {conf|sup|norm} 
``

where ``conf`` is for training the confounder model, ``sup`` for the suppressor, or ``norm`` for the normalised brightness model.

## Experiments and results
All XAI method execution and resulting energy metric calculations are run here.

### Watermark experiments
Currently this is only pushed to git for the non-rescaled variants, but this will be updated in due course.

``
python calculate_watermarks_energy.py {split} {model_ind} {variable}  
``
Where split is defined as before, the ``model_ind`` is also an integer [0..4] for which trained model to use, and ``variable`` makes use of the variable-position watermark data if set.

### COCO lightness experiments

``
python calculate_coco_energy.py {split} {model_ind}
``
