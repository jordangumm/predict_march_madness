wget https://repo.continuum.io/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p miniconda
rm miniconda.sh

source miniconda/bin/activate

pip install click
pip install deap
pip install pandas
pip install sklearn
pip install tqdm

pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
pip install mkl-service
pip install -e ../automaxout/.
pip install -e ../cbbstats/.
