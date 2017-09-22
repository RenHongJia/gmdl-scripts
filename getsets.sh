#!/bin/bash

wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data -O ./data/adult;

wget https://archive.ics.uci.edu/ml/machine-learning-databases/cmc/cmc.data -O ./data/cmc;

wget https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-testing.data -O ./data/poker;

wget http://archive.ics.uci.edu/ml/machine-learning-databases/00362/HT_Sensor_UCIsubmission.zip -O /tmp/ht-sensor;

unzip /tmp/ht-sensor -d /tmp/ht-sensor-tmp;
unzip /tmp/ht-sensor-tmp/HT_Sensor_dataset.zip  -d ./data;
mv -f ./data/HT_Sensor_dataset.dat ./data/ht-sensor;
rm -rf /tmp/ht-sensor /tmp/ht-sensor-tmp;

wget https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census.tar.gz -O /tmp/census;

mkdir /tmp/census-data;
tar -xf /tmp/census -C /tmp/census-data;

cat /tmp/census-data/census-income.data /tmp/census-data/census-income.test > ./data/census;

rm -rf /tmp/census /tmp/census-data;

wget http://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data -O ./data/arrhythmia;

wget https://archive.ics.uci.edu/ml/machine-learning-databases/spectrometer/lrs.data -O ./data/lrs;

wget http://archive.ics.uci.edu/ml/machine-learning-databases/hill-valley/Hill_Valley_without_noise_Testing.data -O /tmp/hill-valley1;
wget http://archive.ics.uci.edu/ml/machine-learning-databases/hill-valley/Hill_Valley_without_noise_Training.data -O /tmp/hill-valley2;

cat /tmp/hill-valley1 /tmp/hill-valley2 > ./data/hill-valley;
rm -f /tmp/hill-valley*;

wget http://archive.ics.uci.edu/ml/machine-learning-databases/dorothea/DOROTHEA/dorothea_train.data -O /tmp/dorothea1;
wget http://archive.ics.uci.edu/ml/machine-learning-databases/dorothea/DOROTHEA/dorothea_test.data -O /tmp/dorothea2;

cat /tmp/dorothea1 /tmp/dorothea2 > ./data/dorothea;
rm -f /tmp/dorothea*;

wget https://archive.ics.uci.edu/ml/machine-learning-databases/00244/fertility_Diagnosis.txt -O ./data/fertility;

wget https://archive.ics.uci.edu/ml/machine-learning-databases/connect-4/connect-4.data.Z -O ./data/;
uncompress ./data/connect-4.data.Z;
mv -f ./data/connect-4.data ./data/connect-4;

wget https://archive.ics.uci.edu/ml/machine-learning-databases/00199/MiniBooNE_PID.txt -O ./data/miniboone;

wget https://archive.ics.uci.edu/ml/machine-learning-databases/balloons/yellow-small+adult-stretch.data -O ./data/balloons;

wget https://archive.ics.uci.edu/ml/machine-learning-databases/libras/movement_libras.data -O ./data/libras;
