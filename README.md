# Comparing the performance of LSTM and GRU for Text Summarization using Pointer Generator Networks

Compare the performance of LSTM and GRU in Text Summarization using Pointer Generator Networks as discussed in https://github.com/abisee/pointer-generator

Based on code at https://github.com/steph1793/Pointer_Generator_Summarizer

## Prerequisites
Python 3.7+    
Tensorflow 2.8+ (Any 2.x should work, but not tested)    
rouge 1.0.1 (pip install rouge)

### Dataset
We use the CNN-DailyMail dataset. The application reads data in the tfrecords format files.

Dataset can be created and processed based on the instructions at https://github.com/abisee/cnn-dailymail

Alternatively, pre-processed dataset can be downloaded from https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail

Easiest way yet, download the [dataset zip file](https://drive.google.com/file/d/1C-pLHLlpL4Ca6S0mxhBpY8cCLCGqOb0f/view?usp=sharing)
and extract it into the dataset directory (by default, this will be "./dataset/")

## Pre-Trained Models (Optional)
In case you want to skip the training and run evaluate the models, you can download the [checkpoint zip file](https://drive.google.com/file/d/1zAWDhxwvc9gZe-bgmkOXVuZqCEed98Oq/view?usp=sharing)
and extract it into the checkpoint directory (by default, this will be "./checkpoint/").

## Usage

Training and Evaluation will perform the actions on all four models. If you wish to train/eval only one model, pass it as --model_name. 

### Training
~~~
python main.py --mode="train" --vocab_path="./dataset/vocab" --data_dir="./dataset/chunked_train"
~~~

### Evaluation
~~~
python main.py --mode="eval" --vocab_path="./dataset/vocab" --data_dir="./dataset/chunked_val"
~~~

### Parameters

Most of the parameters have defaults and can be skipped. Here are the parameters that you can tweak along with their defaults.

| Parameter               | Default         | Description                                                                                                             |
| ----------------------- | --------------- | ----------------------------------------------------------------------------------------------------------------------- |
| max_enc_len             | 512             | Encoder input max sequence length                                                                                       |
| max_dec_len             | 128             | Decoder input max sequence length                                                                                       |
| max_dec_steps           | 128             | maximum number of words of the predicted abstract                                                                       |
| min_dec_steps           | 32              | Minimum number of words of the predicted abstract                                                                       |
| batch_size              | 4               | batch size                                                                                                              |
| beam_size               | 4               | beam size for beam search decoding (must be equal to batch size in decode mode)                                         |
| vocab_size              | 50000           | Vocabulary size                                                                                                         |
| embed_size              | 128             | Words embeddings dimension                                                                                              |
| enc_units               | 256             | Encoder LSTM/GRU cell units number                                                                                      |
| dec_units               | 256             | Decoder LSTM/GRU cell units number                                                                                      |
| attn_units              | 512             | [context vector, decoder state, decoder input] feedforward result dimension - used to compute the attention weights     |
| learning_rate           | 0.15            | Learning rate                                                                                                           |
| adagrad_init_acc        | 0.1             | Adagrad optimizer initial accumulator value. Please refer to the Adagrad optimizer API documentation on tensorflow site |
| max_grad_norm           | 0.8             | Gradient norm above which gradients must be clipped                                                                     |
| checkpoints_save_steps  | 1000            | Save checkpoints every N steps                                                                                          |
| max_checkpoints         | 10              | Maximum number of checkpoints to keep. Older ones are deleted                                                           |
| max_steps               | 50000           | Max number of iterations                                                                                                |
| max_num_to_eval         | 100             | Max number of examples to evaluate                                                                                      |
| checkpoint_dir          | "./checkpoint"  | Checkpoint directory                                                                                                    |
| log_dir                 | "./log"         | Directory in which to write logs                                                                                        |
| results_dir             | None            | Directory in which we write the intermediate results (Article, Actual Summary and Predicted Summary) during evaluation  |
| data_dir                | None            | Data Folder                                                                                                             |
| vocab_path              | None            | Path to vocab file                                                                                                      |
| mode                    | None            | Should be "train" or "eval"                                                                                             |
| model_name              | None            | Name of a specific model. If empty, all models are used                                                                 |
