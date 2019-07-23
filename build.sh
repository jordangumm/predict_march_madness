wget https://repo.continuum.io/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p miniconda
rm miniconda.sh

bash download.sh data

source miniconda/bin/activate

pip install click
pip install deap
pip install pandas
pip install sklearn
pip install statsmodels
pip install tensorflow
pip install tqdm
pip install xgboost

pip install -e ../automaxout/.
