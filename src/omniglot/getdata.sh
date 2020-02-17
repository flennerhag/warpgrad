set -e

python getdata.py

cd ./data/omniglot-py 

mkdir ./images_resized

cp -r ./images_background/* ./images_resized/
cp -r ./images_evaluation/* ./images_resized/

cd ./images_resized

cp ../../../resize.py .

python resize.py -f '*/*/' -H 28 -W 28

cd ..
cd ..
cd ..
