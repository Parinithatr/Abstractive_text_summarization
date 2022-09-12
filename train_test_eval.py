import tensorflow as tf
from model import PGN
from training_helper import train_model
from test_helper import beam_decode
from batcher import batcher, Vocab, Data_Helper
from tqdm import tqdm
from rouge import Rouge
import pprint
from os import path
from nltk.translate.meteor_score import single_meteor_score
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
from nltk import word_tokenize
import numpy as np

def train(params):
    assert params["mode"].lower() == "train", "change training mode to 'train'"
    
    cellTypes = ["LSTM", "GRU"]
    for e in cellTypes:
        for d in cellTypes:
            model_name = e + "_" + d
            if (not params["model_name"]) or (model_name.upper() == params["model_name"].upper()):
                print("Training", model_name)
                model = PGN(e, d, params)

                print("Creating the vocab ...")
                vocab = Vocab(params["vocab_path"], params["vocab_size"])

                print("Creating the batcher ...")
                b = batcher(params["data_dir"], vocab, params)

                print("Creating the checkpoint manager")
                checkpoint_dir = "{0}/{1}".format(params["checkpoint_dir"], model_name)
                ckpt = tf.train.Checkpoint(step=tf.Variable(0), PGN=model)
                ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=params["max_checkpoints"])

                ckpt.restore(ckpt_manager.latest_checkpoint)
                if ckpt_manager.latest_checkpoint:
                    print("Restored from {}".format(ckpt_manager.latest_checkpoint))
                else:
                    print("Initializing from scratch.")

                print("Starting the training ...")
                train_model(model, b, params, ckpt, ckpt_manager, "{0}/{1}.txt".format(params["log_dir"], model_name))
                model = None


def predict(encoder, decoder, params):
    model_name = encoder + "_" + decoder
    
    print("Evaluating", model_name)
    model = PGN(encoder, decoder, params)

    print("Creating the vocab ...")
    vocab = Vocab(params["vocab_path"], params["vocab_size"])

    print("Creating the batcher ...")
    b = batcher(params["data_dir"], vocab, params)

    print("Creating the checkpoint manager")
    checkpoint_dir = "{0}/{1}".format(params["checkpoint_dir"], model_name)
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), PGN=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=params["max_checkpoints"])
    
    ckpt_path = ckpt_manager.latest_checkpoint
    ckpt.restore(ckpt_path)
    print("Model restored from", ckpt_path)

    for batch in b:
        yield beam_decode(model, batch, vocab, params)


def evaluate(params):
    assert params["mode"].lower() == "eval", "change training mode to 'eval'"
    assert params["beam_size"] == params["batch_size"], "Beam size must be equal to batch_size, change the params"
    
    cellTypes = ["LSTM", "GRU"]
    for e in cellTypes:
        for d in cellTypes:
            model_name = e + "_" + d
            if not params["model_name"] or model_name.upper() == params["model_name"].upper():
                gen = predict(e, d, params)
                reals = []
                preds = []
                with tqdm(total=params["max_num_to_eval"],position=0, leave=True) as pbar:
                    for i in range(params["max_num_to_eval"]):
                        trial = next(gen)
                        reals.append(trial.real_abstract)
                        preds.append(trial.abstract)
                        if params["results_dir"] != None and params["results_dir"] != "":
                            with open(params["results_dir"]+"/" + model_name + "_" + str(i) + ".txt", "w") as f:
                                f.write("Article:\n")
                                f.write(trial.text)
                                f.write("\n\nReal Abstract:\n")
                                f.write(trial.real_abstract)
                                f.write("\n\nPredicted Abstract:\n")
                                f.write(trial.abstract)
                        pbar.update(1)
                #BLeU Scores
                b_scores_1 = []
                b_scores_2 = []
                b_scores_3 = []
                b_scores_4 = []
                for i in range(0, len(reals)):
                    b_scores_1.append(sentence_bleu([word_tokenize(reals[i])], word_tokenize(preds[i]), weights=(1, 0, 0, 0)))
                    b_scores_2.append(sentence_bleu([word_tokenize(reals[i])], word_tokenize(preds[i]), weights=(1./2, 1./2, 0, 0)))
                    b_scores_3.append(sentence_bleu([word_tokenize(reals[i])], word_tokenize(preds[i]), weights=(1./3, 1./3, 1./3, 0)))
                    b_scores_4.append(sentence_bleu([word_tokenize(reals[i])], word_tokenize(preds[i]), weights=(1./4, 1./4, 1./4, 1./4)))
                #METEOR Scores
                m_scores = []
                for i in range(0, len(reals)):
                    m_scores.append(single_meteor_score(word_tokenize(reals[i]), word_tokenize(preds[i])))
                #ROUGE Scores
                r=Rouge()
                r_scores = r.get_scores(preds, reals, avg=True)
                #pprint.pprint(r_scores)
                print("ROGUE-1:", r_scores["rouge-1"]["f"])
                print("ROGUE-2:", r_scores["rouge-2"]["f"])
                print("ROGUE-l:", r_scores["rouge-l"]["f"])
                print("BLeU-1:", np.mean(b_scores_1))
                print("BLeU-2:", np.mean(b_scores_2))
                print("BLeU-3:", np.mean(b_scores_3))
                print("BLeU-4:", np.mean(b_scores_4))
                print("METEOR:", np.mean(m_scores))
