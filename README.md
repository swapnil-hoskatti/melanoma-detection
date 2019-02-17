# Melanoma Detection 

*"Vision Based Melanoma Detection using Deep Learning"*

## Table of Contents

  - [Installation](#installation)
  - [Usage](#usage)
  - [Dependencies](requirements.txt)
  - [Environment](#Environment) 
  - [License](#license)


## Installation

Clone the git repository:

``` sourceCode console
$ git clone https://github.com/gavindsouza/melanoma-detection.git
$ cd melanoma-detection
```

Install necessary dependencies

``` sourceCode console
$ pip install -r requirements.txt
```

## Usage

### Starting Things Up

To run the application, change the current working directory to
\./melanoma-detection/app
``` sourceCode console
$ cd app
```

run the app by typing the following command in your terminal

``` sourceCode console
$ python -m flask run --without-threads
```

The application can be accessed at _localhost:5000_ in your browser

![](docs/util/screen1.jpg)

## Dependencies

Everything as mentioned in the [requirements file](requirements.txt)

- Flask
- Keras
- Tensorflow
- openCV-python
- scikit-learn
- scikit-image

## Environment

this application was developed under 
#### Python 3.7, Linux 4.20.6

## License

This code has not been licensed yet.
