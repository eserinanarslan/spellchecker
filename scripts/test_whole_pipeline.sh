#!/usr/bin/env bash
set -euxo pipefail

# Testing the whole suggestion pipeline 
# from training to the testing of words correction

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 username arc_version nepochs products_index_date [nsamples]"
  echo " e.g: $0 $USER BI-LSTM 4 202204 30"
  exit 1;
fi
USERNAME=$1
ARC_VERSION=$2
NEPOCHS=$3
PRD_VERSION=$4

if [ -z ${5+x} ]; then
    echo "Note: Performing training with the whole training dataset"
else
    NSAMPLES=$5
fi

LOGS_FOLDER=$HOME/conrad_ml_spellchecker/logs
SRC_FOLDER="../src"
cd $SRC_FOLDER

start=`date +%s`
python 00.gather_training_data.py $PRD_VERSION

python 01.prepare_training_data.py $USERNAME $PRD_VERSION&
PREP_PID=`echo $!`
wait $PREP_PID
TRAINING_DATA_DATE=`head -n 1 $LOGS_FOLDER/data_preparation_"$PREP_PID".txt | cut -d '|' -f 2`

python 02.preprocess_data.py $USERNAME $PRD_VERSION &
PRE_PROC_PID=`echo $!`
wait $PRE_PROC_PID
PREPROC_TIMESTAMP=`head -n 1 $LOGS_FOLDER/preprocessing_"$PRE_PROC_PID".txt | cut -d '|' -f 2`

if [ -z ${NSAMPLES+x} ]; then
    # Train with all data samples
    python 03.train_model.py --username $USERNAME --data_date $TRAINING_DATA_DATE \
        --arc_version $ARC_VERSION --n_epochs $NEPOCHS &
else
    python 03.train_model.py --username $USERNAME --data_date $TRAINING_DATA_DATE \
        --arc_version $ARC_VERSION --n_samples $NSAMPLES --n_epochs $NEPOCHS &
fi

TRAINING_PID=`echo $!`
wait $TRAINING_PID
# Extract the training timestamp
TRAING_TIME=`head -n 1 $LOGS_FOLDER/training_"$TRAINING_PID".txt | cut -d '|' -f 2`

python 04.vectorize_dictionary.py --prod_index_date $PREPROC_TIMESTAMP \
    --username $USERNAME --arc_version $ARC_VERSION \
    --train_timestamp $TRAING_TIME

cd -
python test_suggest_pipeline.py $USERNAME $TRAING_TIME $ARC_VERSION \
    "kipschlter Microsooft Schrauebendeher"

echo "The whole pipeline was successfully executed"
end=`date +%s`

secs_to_human() {
    if [[ -z ${1} || ${1} -lt 60 ]] ;then
        min=0 ; secs="${1}"
    else
        time_mins=$(echo "scale=2; ${1}/60" | bc)
        min=`echo ${time_mins} | cut -d'.' -f1`
        secs="0.$(echo ${time_mins} | cut -d'.' -f2)"
        secs=`echo "${secs} * 60"|bc |awk '{print int($1+0.5)}'`
    fi
    echo "Time Elapsed : ${min} minutes and ${secs} seconds."
}

secs_to_human "$(($(date +%s) - ${start}))"