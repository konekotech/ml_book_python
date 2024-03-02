# Machine Learning Book Codes in Python

This is a repository of machine learning book codes in Python.
This codes are based on the book "機械学習教本" written by 柴原一友.

## Requirements
- Docker (latest)

## Usage

### Build Docker Image

This is a one-time process. You only need to do this once. You do not need to build the image every time you changed Python codes, because you will append the directory which includes python files when you run the container like `-v "./:/app"`.

```sh
docker build -t ml-book ./  
```

### Run Docker Container

This is how you run the code. Please replace `$PYTHON_FILE_NAME` with the name of the Python file you want to run such as `test.py`. Use `test.py` as a sample.

```sh
docker run --rm -v "./:/app" ml-book $PYTHON_FILE_NAME
```

