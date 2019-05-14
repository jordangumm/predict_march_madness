wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda

conda activate

pip install pipenv

pipenv install --python=~/miniconda/bin/python --site-packages
