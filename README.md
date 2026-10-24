# LLM Pre-Training Workshop
This repository is for the LLM Pre-Training Workshop hosted by the Google Developers Group.

## Getting Started
To install the required dependencies and set up the environment, run:
```bash
sh setup.sh
```
Then you can start putting your own `.txt` files into the `dataset` folder. The script will automatically read all `.txt` files in that folder and use them to train the tokenizer and the model. To train the tokenizer, run:
```bash
python create_tokenizer.py
```
To process the dataset into tokens and save them as binary, run:
```bash
python process_dataset.py
```
To train the model, run:
```bash
python train_model.py
```

If you want to change the model hyperparameters, you can do so in the `constants.py` file. The hyperparameters are used in both the tokenizer and the model training scripts.