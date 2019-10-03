# aisdb
DSiP: AI for Sleep Disordered Breathing

## Setup

I recommend using [virtualenv](https://docs.python.org/3/library/venv.html) for
Python development. It is optional, but very helpful and a best practice when
working on a team or with different projects. I assume Python 3.7 or newer as
well. Personally, I also manage my Python installs using
[pyenv](https://github.com/pyenv/pyenv) but that isn't necessary.

### Virtual Env setup

You can change the venv directory, but it is handy to just store it inside the
project at `.venv`.

```bash
python3 -m venv ./.venv --prompt aisdb
source ./.venv/bin/activate
```

### Install dependencies

Install pip requirements. For some reason pyEDFlib requires separate
installation of numpy first. Numpy is included in the requirements.txt.

```bash
# Must install numpy first for pyEDFlib
pip install numpy
pip install -r requirements.txt
```

## Start Jupyter lab

```bash
jupyter lab
```

## Download data

I am putting the EDF files into a folder called `local-data` which will not be
in git. We do not want EDF files in git because they are far too large. Anything
you put into the `local-data` directory will be ignored by git.

