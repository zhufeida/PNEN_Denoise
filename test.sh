

## Test on Set12, BSD68, Urban100
## ../dataset/Denoise1/test/Set12
## ../dataset/Denoise1/test/BSD68
## ../dataset/Denoise1/test/Urban100

export TF_CPP_MIN_LOG_LEVEL="2"  
export CUDA_DEVICE_ORDER="PCI_BUS_ID" 

###########################################################################################################################
###########################################################################################################################
export CUDA_VISIBLE_DEVICES="2"
MODEL_DEF="RDN"
SIGMA=50
TRAIN_PATH="checkpoints/${MODEL_DEF}/${SIGMA}/model-480002"
RESULTS_DIR="results/${MODEL_DEF}/${SIGMA}"
NOISE_DIR="results/${MODEL_DEF}/${SIGMA}_noise"
if [ ! -d $RESULTS_DIR ]; then
    mkdir -p $RESULTS_DIR
fi
if [ ! -d $NOISE_DIR ]; then
    mkdir -p $NOISE_DIR
fi
ARGS="--checkpoint_path=$TRAIN_PATH --model_def=$MODEL_DEF --sigma=$SIGMA --results_dir=$RESULTS_DIR --noise_dir=$NOISE_DIR"
python test.py $ARGS
# nohup python -u test.py $ARGS > log/test_RDN_50.log &

export CUDA_VISIBLE_DEVICES="3"
MODEL_DEF="RDN_PNB"
SIGMA=50
TRAIN_PATH="checkpoints/${MODEL_DEF}/${SIGMA}/model-500002"
RESULTS_DIR="results/${MODEL_DEF}/${SIGMA}"
NOISE_DIR="results/${MODEL_DEF}/${SIGMA}_noise"
if [ ! -d $RESULTS_DIR ]; then
    mkdir -p $RESULTS_DIR
fi
if [ ! -d $NOISE_DIR ]; then
    mkdir -p $NOISE_DIR
fi
ARGS="--checkpoint_path=$TRAIN_PATH --model_def=$MODEL_DEF --sigma=$SIGMA --results_dir=$RESULTS_DIR --noise_dir=$NOISE_DIR"
python test.py $ARGS
# nohup python -u test.py $ARGS > log/test_RDN_PNB_50.log &
















