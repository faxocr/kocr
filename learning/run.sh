cd ../src
make preprocess
cd ../learning

python make_data.py $1
if [ $? -ne 0 ]
then
    exit
fi

python train_cnn.py