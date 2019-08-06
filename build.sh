mkdir dependencies

wget https://repo.continuum.io/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p dependencies/miniconda
rm miniconda.sh

source dependencies/miniconda/bin/activate

pip install click
pip install deap
pip install pandas
pip install sklearn
pip install statsmodels
pip install tensorflow
pip install tqdm
pip install xgboost

pip install -e ../automaxout/.

git clone https://github.com/jordangumm/cbbstats.git dependencies/cbbstats

python predict_march_madness/build_examples.py
