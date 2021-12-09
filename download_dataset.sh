#!/bin/sh

echo "*************** Starting Dataset Download*******************"

echo "Downloading Annotations"
wget http://kaldir.vc.in.tum.de/cosmos_anns.zip -q

echo "Downloading Training Images"
wget http://kaldir.vc.in.tum.de/images_train.zip -q

echo "Downloading Validation Images"
wget http://kaldir.vc.in.tum.de/images_val.zip -q

echo "Downloading Test Images"
wget http://kaldir.vc.in.tum.de/images_test.zip -q

echo "*************** Download Complete*******************"
