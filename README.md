# Copilosh

Your personal assistant directly integrated into your terminal.

## Pre-requisites

Set up the environnements:

Create a conda environnement:
```bash
conda create -n copilosh_env python=3.8 poetry=1.4.0 --y
conda activate copilosh_env
```


```bash
poetry lock --no-update # TO AVOID AUTO-UPDATE
poetry install
```

Create a cache directory, to store your models:
    
```bash
mkdir cache
```

## Installation

Add the copilosh wrapper function to your shell profile file (e.g. `~/.bashrc`, `~/.zshrc`, etc.):

Ubuntu/Unix:
```bash
cat copilosh.sh >> ~/.bashrc
source ~/.bashrc .
```

MacOS for new version :
```bash
cat copilosh.sh >> ~/.zshrc
source ~/.zshrc .
```

MacOS for old version :
```bash
cat copilosh.sh >> ~/.bash_profile
source ~/.bash_profile .
```

Run the assistant server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

Run the frontend:

```bash
cd frontend
npm start
```

## Usage

Use any command in your terminal, if an error occurs, the assistant will suggest a solution.