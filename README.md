# Simple yet Effective methods to Improve the Drug-Target Interaction Prediction

This repository contains the code and data link for Simple yet Effective methods to Improve the Drug-Target Interaction Prediction. Our model achieves significant results compared to traditional and recent baselines. We implement our method based on the codebase of [fairseq](https://github.com/pytorch/fairseq). If you have questions, don't hesitate to open an issue or ask me via <pqz1811@gmail.com> or Lijun Wu via <lijuwu@microsoft.com>. We are happy to hear from you!

## Model Architecture

![](./img/arch.png)

## Requirements and Installation
* Python version == 3.7
* PyTorch version == 1.10.2
* Fairseq version == 0.10.2
* RDKit version == 2020.09.5
* numpy

We will set up the environment using conda. Clone the current repo and fairseq official repo, then merge them:

```shell
# Need to modify url when public
git clone git@github.com:QizhiPei/DTI.git
cd DTI
pwd=$PWD

git clone git@github.com:pytorch/fairseq.git /tmp/fairseq
cd /tmp/fairseq
git checkout v0.10.2

cd $pwd
cp -r -n /tmp/fairseq/* ./
```
Create a new environment: 

```shell
conda create -n fairseq-dti python=3.7
```

Activate the environment:

```shell
conda activate fairseq-dti
```

Install required packages for evaluation:

```shell
conda install -c conda-forge rdkit
pip install future scipy sklearn lifelines requests
```

Install the code from source:

```shell
pip install -e . 
```

## Dataset

* Unlabeled Molecule and Protein
  * Pfam: http://pfam.xfam.org/
  * PubChem: https://pubchem.ncbi.nlm.nih.gov/
* BindingDB Dataset: https://github.com/Shen-Lab/DeepAffinity/tree/master/data/dataset

## Getting Started

### Data Preprocessing

#### Unlabeled Molecule and Protein

```shell
DATADIR=/yourUnlabeledDataDir
DATA_BIN=/yourDataBinDir

# Canonicalize all SMILES
python preprocess/canonicalize.py $DATADIR/train.mol --workers 40 \
  --output-fn $DATADIR/train.mol.can

# Tokenize all SMILES
python preprocess/tokenize_re.py $DATADIR/train.mol.can --workers 40 \
  --output-fn $DATADIR/train.mol.can.re 

# Tokenize all protein sequence
python preprocess/add_space.py $DATADIR/train.pro --workers 40 \
  --output-fn $DATADIR/train.pro.addspace

# You should also canonicalize and tokenize the valid set in the same way.

# Binarize the data
fairseq-preprocess \
    --only-source \
    --trainpref $DATADIR/train.mol.can.re \
    --validpref $DATADIR/valid.mol.can.re \
    --destdir $DATA_BIN/molecule \
    --workers 40 \
    --srcdict preprocess/dict.mol.txt
    
fairseq-preprocess \
    --only-source \
    --trainpref $DATADIR/train.pro.addspace \
    --validpref $DATADIR/valid.pro.addspace \
    --destdir $DATA_BIN/protein \
    --workers 40 \
    --srcdict preprocess/dict.pro.txt
```
#### Paired DTI data

```shell
DATADIR=/yourPairedDataDir
DATA_BIN=/yourDataBinDir/bindingdb(davis or kiba)

# Canonicalize all SMILES
python preprocess/canonicalize.py $DATADIR/train.mol --workers 40 \
  --output-fn $DATADIR/train.mol.can

# Tokenize all SMILES
python preprocess/tokenize_re.py $DATADIR/train.mol.can --workers 40 \
  --output-fn $DATADIR/train.mol.can.re 

# Tokenize all protein sequence
python preprocess/add_space.py $DATADIR/train.pro --workers 40 \
  --output-fn $DATADIR/train.pro.addspace

# You should also process the valid set and test set in the same way.

# Binarize the data
fairseq-preprocess \
    --only-source \
    --trainpref $DATADIR/train.mol.can.re \
    --validpref $DATADIR/valid.mol.can.re \
    --destdir $DATA_BIN/input0 \
    --workers 40 \
    --srcdict preprocess/dict.mol.txt

fairseq-preprocess \
    --only-source \
    --trainpref $DATADIR/train.pro.addspace \
    --validpref $DATADIR/valid.pro.addspace \
    --destdir $DATA_BIN/input1 \
    --workers 40 \
    --srcdict preprocess/dict.pro.txt

mkdir -p $DATA_BIN/label

cp $DATADIR/train.label $DATA_BIN/label/train.label
cp $DATADIR/valid.label $DATA_BIN/label/valid.label
```

### Train Baseline

```shell
DATA_BIN=/yourDataBinDir/bindingdb(davis or kiba)  
FAIRSEQ=$pwd # The path to 
SAVE_PATH=/yourCkptDir
TOTAL_UPDATES=200000 # Total number of training steps
WARMUP_UPDATES=10000 # Warmup the learning rate over this many updates
PEAK_LR=0.00005       # Peak learning rate, adjust as needed
BATCH_SIZE=4		# Batch size
UPDATE_FREQ=4       # Increase the batch size 4x

mkdir -p $SAVE_PATH

python $FAIRSEQ/train.py --task dti_separate $DATA_BIN \
    --num-classes 1 --init-token 0 \
    --max-positions-molecule 512 --max-positions-protein 1024 \
    --save-dir $SAVE_PATH \
    --encoder-layers 12 \
    --criterion dti_separate --regression-target \
    --batch-size $BATCH_SIZE --update-freq $UPDATE_FREQ --required-batch-size-multiple 1 \
    --optimizer adam --weight-decay 0.1 --adam-betas '(0.9,0.98)' --adam-eps 1e-06 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --clip-norm 1.0  --max-update $TOTAL_UPDATES \
    --arch roberta_dti_separate_encoder --dropout 0.1 --attention-dropout 0.1 \
    --skip-invalid-size-inputs-valid-test \
    --fp16 \
    --shorten-method truncate \
    --find-unused-parameters | tee -a ${SAVE_PATH}/training.log
```

* `DATA_BIN` is where you save the preprocessed data
* `FAIRSEQ` is the path to fairseq code base
* `SAVE_PATH` is where you save the checkpoint file and training log

### Train Ours

```shell
DATA_BIN=/yourDataBinDir
DTI_DATASET=bindingdb(davis or kiba)
FAIRSEQ=$pwd
SAVE_PATH=/yourCkptDir
TOTAL_UPDATES=200000 # Total number of training steps
INTERVAL_UPDATES=1000 # Validate and save checkpoint every N updates
WARMUP_UPDATES=10000 # Warmup the learning rate over this many updates
PEAK_LR=0.0001       # Peak learning rate, adjust as needed
BATCH_SIZE=4		# Batch size
UPDATE_FREQ=8       # Increase the batch size 8x
MLMW=2 				# MLM loss weight

# The final real batch size is BATCH_SIZE x GPU_NUM x UPDATE_FREQ

mkdir -p $SAVE_PATH

python $FAIRSEQ/train.py --task dti_mlm_regress_pretrain $DATA_BIN \
	--dti-dataset $DTI_DATASET \
    --num-classes 1 --init-token 0 \
    --max-positions-molecule 512 --max-positions-protein 1024 \
    --save-dir $SAVE_PATH \
    --encoder-layers 12 \
    --criterion dti_mlm_regress_pretrain --regression-target \
    --batch-size $BATCH_SIZE --update-freq $UPDATE_FREQ --required-batch-size-multiple 1 \
    --optimizer adam --weight-decay 0.01 --adam-betas '(0.9,0.98)' --adam-eps 1e-06 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --clip-norm 1.0  --max-update $TOTAL_UPDATES \
    --arch roberta_dti_mlm_regress --dropout 0.1 --attention-dropout 0.1 \
    --skip-invalid-size-inputs-valid-test \
    --fp16 \
    --shorten-method truncate \
    --use-2-attention --find-unused-parameters --ddp-backend no_c10d \
    --validate-interval-updates $INTERVAL_UPDATES \
    --save-interval-updates $INTERVAL_UPDATES \
    --best-checkpoint-metric loss_regress_mse \
    --mlm-weight-0 $MLMW --mlm-weight-1 $MLMW --mlm-weight-paired-0 $MLMW --mlm-weight-paired-1 $MLMW | tee -a ${SAVE_PATH}/training.log
```
* `DATA_BIN` is where you save the preprocessed data
* `DTI_DATASET` is the paired dataset you want to use for training
* `FAIRSEQ` is the path to fairseq code base
* `SAVE_PATH` is where you save the checkpoint file and training log

You can also use fairseq argument `--tensorboard-logdir TENSORBOARD_LOGDIR`  to save logs for tensorboard.

## Evaluation

```shell
# Copy dict to DATA_BIN for evaluation
cp preprocess/dict.mol.txt $DATA_BIN
cp preprocess/dict.pro.txt $DATA_BIN

python $FAIRSEQ/test.py \
	--checkpoint yourCheckpointFilePath \
	--data-bin $DATA_BIN \
	--test-data yourTestSetDirPath \
	--output-fn yourResultFilePath
```

* `DATA_BIN` is where you save the preprocessed data. We need to use the dictionary saved in this path.
* `yourCheckpointFilePath` is the path to `.pt` file
* `yourTestSetDirPath` is the path to your raw test data after tokenized (and canonicalized)
* `yourResultFilePath` is where you want to save the model prediction and ground truth

## Author

Qizhi Pei

Lijun Wu

## Citation

