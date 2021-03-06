#!/usr/bin/env bash

if [[ ! -z "$VIRTUAL_ENV" ]]; then
  echo "'deactivate' before running this script."
  exit 1
fi

rm -rf .venv
python3.10 -m venv .venv

source ./.venv/bin/activate

python -m pip install --upgrade pip setuptools wheel


# ##############################################################################
# I want to use the GPU when possible.

 pip3 install -U torch --extra-index-url https://download.pytorch.org/whl/cu113

# ##############################################################################
# Install normal requirements

if [ -f requirements.txt ]; then python -m pip install -r requirements.txt; fi


# ##############################################################################
# Dev only pip installs (not required because they're personal preference)

python -m pip install -U tensorboard
python -m pip install -U pynvim
python -m pip install -U 'python-lsp-server[all]'
python -m pip install -U pre-commit pre-commit-hooks
python -m pip install -U autopep8 flake8 isort pylint yapf pydocstyle black
python -m pip install -U jupyter jupyter_nbextensions_configurator ipyparallel
python -m pip install -U jupyter_nbextensions_configurator jupyterlab_code_formatter

python -m pip install -U jupyterlab
python -m pip install -U jupyterlab_code_formatter
python -m pip install -U jupyterlab-drawio
python -m pip install -U jupyterlab-lsp
python -m pip install -U jupyterlab-spellchecker
python -m pip install -U jupyterlab-git
python -m pip install -U aquirdturtle-collapsible-headings
python -m pip install -U nbdime

python -m pip install -U jupyterlab-git==0.36.0

jupyter labextension install jupyterlab_onedarkpro
jupyter server extension enable --py jupyterlab_git
jupyter serverextension enable --py jupyterlab_code_formatter


# ##############################################################################
# I Run pre-commit hooks (optional)

pre-commit install
