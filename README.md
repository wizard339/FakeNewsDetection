# FakeNewsDetection
Web application for recognizing fake news

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/downloads/release/python-3916/)
[![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/en/2.3.x/)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/index.html)


## Usage

- Clone my repository.
- Load `model.pkl` from `https://drive.google.com/file/d/1IYza8I4hY0GielDLMl6t9Px2akWf1zDb/view?usp=sharing` and place the file to the working directory.
- Open CMD in working directory.
- Run `pip install -r requirements.txt`
- Open project in any IDE (Pycharm or VSCode)
- Run `main.py`, go to the `http://127.0.0.1:5000/`
- If you want to build your model with the some changes, you can check the `model.ipynb`.
- You can check that the web application is working fine. Sometimes classifications can be wrong.

## Screenshots

<img src="https://github.com/wizard339/FakeNewsDetection/blob/main/screenshot1.png">
<img src="https://github.com/wizard339/FakeNewsDetection/blob/main/screenshot2.png">
<img src="https://github.com/wizard339/FakeNewsDetection/blob/main/screenshot3.png">


## Note
- This project was created for educational purposes.
- The model can show the best results on texts with a length from 7 to 65 words, because it was on such data that it was trained.
- Сlassification results may be inaccurate, so you need to treat them with caution.
