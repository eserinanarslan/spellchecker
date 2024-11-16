#!/bin/bash

# Run all the notebook in an asceding order according to the names of the "ipynb" files
# usage: ./run_notebooks.sh <notebooks_folder_path> <logs_folder_path>

CMD='echo "Running all the notebooks" && cd ../notebooks' ;


if [ -z ${1+x} ]; then
  NOTEBOOKS_FOLDER=../notebooks
else
  NOTEBOOKS_FOLDER=$1
fi
if [ -z ${2+x} ]; then
  LOGS_FOLDER=${NOTEBOOKS_FOLDER}"/logs"
else
  LOGS_FOLDER=$2
fi

for file_ in \
	`find $NOTEBOOKS_FOLDER -type f -iname "*.ipynb" -maxdepth 1 | awk -F/ '{ print $NF }' | sort`
do
	CMD=$CMD" && papermill $file_ ${LOGS_FOLDER}/$file_"
done

CMD=$CMD" &&  cd -"
echo "==============================================================="
echo $CMD
eval $CMD
