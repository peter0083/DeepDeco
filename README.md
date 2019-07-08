# DeepDeco

Generate customizable interior design images with sketches and text description.

## Setup

Clone repository and update python path

```
git clone https://github.com/peter0083/DeepDeco.git
cd DeepDeco
```

## Dependencies

*Deep Photo Style Transfer*

* [Tensorflow](https://www.tensorflow.org/)
* [Numpy](www.numpy.org/)
* [Pillow](https://pypi.python.org/pypi/Pillow/)
* [Scipy](https://www.scipy.org/)
* [PyCUDA](https://pypi.python.org/pypi/pycuda)

*Style Search Engine*

* [Sklearn](https://scikit-learn.org/stable/)
* [Pandas](https://pandas.pydata.org/)

*Installation*

To install the packages above, please run:

```
pip install -r requirements
```

### Download the VGG-19 model weights for style transfer
The VGG-19 model of tensorflow is adopted from [VGG Tensorflow](https://github.com/machrisaa/tensorflow-vgg) with few 
modifications on the class interface. The VGG-19 model weights is stored as .npy file 
and could be download from [Google Drive](https://drive.google.com/file/d/0BxvKyd83BJjYY01PYi1XQjB5R0E/view?usp=sharing) or [BaiduYun Pan](https://pan.baidu.com/s/1o9weflK). 
After downloading, copy the weight file to the **./project/vgg19** directory

### Download the Deep Lab V3 weights
Download DeepLab V3 weights from [DeepLabV3 Tensorflow](http://download.tensorflow.org/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz).
After downloading the file, copy the weight file to **src/ftdeepphoto/deeplab/models/** directory

### Download the GloVe 6B 300d weights
Download GloVe 6B 300d weights from [Stanford NLP group](http://nlp.stanford.edu/data/glove.6B.zip). 
Unzip the zip file and move `glove.6B.300d.txt` to the **src/stylesearch/pickles/glove.6B/** directory.

## Build Environment


## Train

1. to load different Word2Vec weights to the style search engine, run the following script:

```bash
python src/stylesearch/train.py --weight_path /path/to/your_word2vec_weight.txt
```

2. to train style transfer model with your own images, run the following script:

```bash
python src/ftdeepphoto/style_fpst.py \
        --style /path/to/image_style_to_transfer.jpg \
        --style-seg /path/to/style_image_segmentation_map.jpg \
        --checkpoint-dir directory_to_checkpoint/ \
        --train-path dir_to_training_images/ \
        --resized-dir dir_to_resized_training_images/ \
        --seg-dir dir_to_training_segmaps/ \
        --vgg-path vgg/imagenet-vgg-verydeep-19.mat \
        --content-weight 1.5e1 \
        --photo-weight 0.005 \
        --checkpoint-iterations 10  \
        --batch-size 1 \
        --epochs 20000 \
        --deeplab-path ../deeplab/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz \
        --matting-dir matting/
```

## Run Inference

### via API

In a terminal, execute the following command:

```bash
curl -F "file=@/path/to/designer_sketch.png" \
    -F "text='ice cold patterned glass'" \
    -X POST http://deepdeco.club:5000/image \
    --output flask_output.gif
```

### Locally

**It is recommended to run this inference script on AWS EC2 with GPU for optimal results.**

1. setup your `awscli` credentials (required to download the dataset)
2. run 'inference_aws.py' 

```bash
python src/ftdeepphoto/run_fpst.py --in-path \
                /path/to/your_designer_sketch.jpeg \
                --style-path  \
                /path/to/your_style_image.jpg \
                --checkpoint-path checkpoints/ \
                --out-path \
                /path/to/your_output.jpg \
                --deeplab-path \
                src/ftdeepphoto/deeplab/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz
```

## Reference


### Project Structure
The directory structure of your new project looks like this: 

```
├── LICENSE
├── Dockerfile            <- New project Dockerfile that sources from base ML dev image
├── docker-compose.yml    <- Docker Compose configuration file
├── docker_clean_all.sh   <- Helper script to remove all containers and images from your system
├── start.sh              <- Script to run docker compose and any other project specific initialization steps 
├── Makefile              <- Makefile with commands like `make data` or `make train`
├── README.md             <- The top-level README for developers using this project.
├── data
│   ├── external          <- Data from third party sources.
│   ├── interim           <- Intermediate data that has been transformed.
│   ├── processed         <- The final, canonical data sets for modeling.
│   └── raw               <- The original, immutable data dump.
│
├── docs                  <- A default Sphinx project; see sphinx-doc.org for details
│
├── models                <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks             <- Jupyter notebooks. Naming convention is a number (for ordering),
│                            the creator's initials, and a short `-` delimited description, e.g.
│                            `1.0-jqp-initial-data-exploration`.
│
├── references            <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports               <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures           <- Generated graphics and figures to be used in reporting
│
├── requirements.txt      <- The requirements file for reproducing the analysis environment, e.g.
│                            generated with `pip freeze > requirements.txt`
│
├── src                   <- Source code for use in this project.
│   ├── __init__.py       <- Makes src a Python module
│   │
│   ├── data              <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features          <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models            <- Scripts to train models and then use trained models to make
│   │   │                    predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
│
└── tox.ini            <- tox file with settings for running tox; see tox.testrun.org
```

