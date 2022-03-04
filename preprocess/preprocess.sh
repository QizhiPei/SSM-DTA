DATA=/protein/users/v-qizhipei/data/DAVIS
DATA_BIN=/protein/users/v-qizhipei/data-bin/DAVIS

fairseq-preprocess \
    --only-source \
    --trainpref $DATA/train.mol.can.re \
    --validpref $DATA/valid.mol.can.re \
    --destdir $DATA_BIN/input0 \
    --workers 40 \
    --srcdict $DATA/dict.mol.txt

fairseq-preprocess \
    --only-source \
    --trainpref $DATA/train.pro.addspace \
    --validpref $DATA/valid.pro.addspace \
    --destdir $DATA_BIN/input1 \
    --workers 40 \
    --srcdict $DATA/dict.pro.txt

mkdir -p $DATA_BIN/label

cp $DATA/train.label $DATA_BIN/label/train.label
cp $DATA/valid.label $DATA_BIN/label/valid.label

cp $DATA/dict.mol.txt $DATA_BIN
cp $DATA/dict.pro.txt $DATA_BIN

# cp -r $DATA_BIN/pfam_pubchem_bindingdb/molecule $DATA_BIN/pfam_pubchem_dude_$af_type/molecule
# cp -r $DATA_BIN/pfam_pubchem_bindingdb/protein $DATA_BIN/pfam_pubchem_dude_$af_type/protein