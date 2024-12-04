# ID2223 - Lab 2
by Annysia Dupaya and Lennart Roth

---

The aim of this project is to perform parameter-efficient fine-tuning (PEFT) of a LLM based on the FineTome dataset using Unsloth.
The fine-tuned model is then used in a serverless inference hosted in a Hugging Face space.
We chose to implement a speech-to-text input system for our model.

---

# Implementation
## Task 1: Fine-Tune a language model
We used the jupyter notebook in our repo [Github_repo](https://github.com/chinadupaya/FinetuneFineTome) (the same as the Unsloth example with minimal changes).
We used checkpointing to save the model every 10 steps. It is saved on Google Drive to allow resuming training even after the Colab notebook instance is destroyed.

## Model selection
### First training run
Model: "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"\
Batchsize: 2\
Learning rate: 2e-4\
num train epochs: 1\
fp16: True\
optim: "adamw_8bit"\
weight decay: 0.01\
lr scheduler type: "linear"\

### Second training run
Model: "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"\
Batchsize: 4\
Learning rate: 2e-4\
num train epochs: 1\
fp16: True\
optim: "adamw_8bit"\
weight decay: 0.01\
lr scheduler type: "linear"\

## Model-Centric Approaches
### Hyperparameter Optimization:
As one can see from the model hyperparameters, we changed the batch size as a hyperparameter.

### Model Architecture:
We experimented with changing the model size from 1B to 3B. Changing the model to a different LLM that performs better, even with fewer parameters, would also improve the model's performance. Generally, larger models tend to be slower for inference (especially in a CPU-only environment) but perform better overall.

## Data-Centric Approaches
### Data Augmentation
Augmenting the dataset can be helpful to increase its size without having additional data at hand. This can help the model generalize better.
Ways of doing augmentation for the FineTome dataset could include:
- Paraphrasing
- Back translation
- Synonym replacement

### Dataset extension
Extending the dataset is a good way to improve fine-tuning. If no high-quality data is available, augmentation can be used as an alternative.

### Data Preprocessing
Preprocessing can improve the quality of the dataset and, therefore, the quality of the model. This can include:
- Removal of duplicates
- Correction of erroneous statements
- Filtering for low-quality samples (using heuristics like sentence length)

Data balancing can also improve the quality of the dataset.

### Data annotation
Using a human-in-the-loop as an evaluator of the answers or to refine existing samples can be a good method for a data-centric approach to optimization.

# Results
While the model from the first training run is faster due to its smaller size, it didn't perform as well as the second model.
This can be seen in the loss values after the training of one epoch:
- Training run 1:
- Training run 2:

This can also be evaluated by testing prompts of both models.
The following are some examples of this.

# UI
For the UI, we used Gradio hosted on a Hugging Face space. This displays a record button where one can record audio messages. The LLM then uses this (after transcription by OpenAI Whisper) as its input.

# Deliverables
1. Source Code: [Github repo](https://github.com/chinadupaya/FinetuneFineTome)
2. README.md: [README.md](https://github.com/chinadupaya/FinetuneFineTome/blob/main/README.md)
3. Gradio UI Link: [Hugginspace](https://huggingface.co/spaces/lennart-rth/iris-inside)
