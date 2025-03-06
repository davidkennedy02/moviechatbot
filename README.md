# Movie Chatbot

This repository contains code for a movie chatbot that uses a fine-tuned Seq2Seq model based on the BART architecture to generate responses in a conversation. The purpose of this script is so that I can learn how to utilise pre-trained models within my own work. This is my first time doing so, and I will add comments explaining each line of code in future commits. 

I also love the idea of being able to use extensive text-message conversations in order to create a chatbot that simulates talking to someone you know - I will explore this in due time. 

## Installation

1. Clone the repository:

    ```shell
    git clone https://github.com/davidkennedy02/movie-chatbot.git
    ```

2. Install the required dependencies:

    ```shell
    pip install -r requirements.txt
    ```

3. Download the movie dialogue dataset:

    ```shell
    python dialogue_downloader.py
    ```

## Usage

Prior to running the script, make sure to modify the values used within `fine_tuning.py` file. Additionally, uncomment the `downloadData` function in the `main.py` file if you don't have a copy of the data CSV yet.  

1. Run the `main.py` script:

    ```shell
    python main.py
    ```

2. Enter your input and the chatbot will generate a response based on the trained model.

3. To exit the chatbot, type "exit".

## Fine-tuning the Model

If you want to fine-tune the model with a different dataset or hyperparameters, you can modify the `fineTuneModel` function in the `fine_tuning.py` file.

## Dataset

The movie dialogue dataset used for training the chatbot is available in the `movie_dialogues.csv` file once dowloaded. It is taken from the ["Cornell Movie-Dialogs Corpus¶"](https://convokit.cornell.edu/documentation/movie.html)

## Acknowledgements

The code in this repository is based on the Hugging Face Transformers library and the Convokit library for dialogue analysis.

