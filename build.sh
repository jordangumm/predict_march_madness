#wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
wget https://repo.continuum.io/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p miniconda
rm miniconda.sh

source miniconda/bin/activate

pip install pipenv
pipenv install

LD_LIBRARY_PATH=$PWD/miniconda/lib pipenv --three
cp $PWD/miniconda/lib/libpython3.7m.so.1.0 $(pipenv --venv)/lib
