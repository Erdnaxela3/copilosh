# Copilosh

Your personal assistant directly integrated into your terminal.

## Pre-requisites

Install python dependencies:

```bash
pip install -r requirements.txt
```

Create a cache directory, to store your models:
    
```bash
mkdir cache
```

## Installation

Add the copilosh wrapper function to your shell profile file (e.g. `~/.bashrc`, `~/.zshrc`, etc.):

```bash
cat copilosh.sh >> ~/.bashrc
source ~/.bashrc
```

Run the assistant server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8082
```

## Usage

Use any command in your terminal, if an error occurs, the assistant will suggest a solution.