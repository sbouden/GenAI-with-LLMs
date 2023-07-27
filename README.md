# [Generative AI with LLMs](https://www.deeplearning.ai/courses/generative-ai-with-llms/)
In Generative AI with Large Language Models (LLMs), we learn the fundamentals of how generative AI works, and how to deploy it in real-world applications.

# Course summary
## Transformer Architecture Overview

The complete transformer architecture consists of an encoder and a decoder components. The encoder encodes input sequences into a deep representation of the structure and meaning of the input. The decoder, working from input token triggers, uses the encoder's contextual understanding to generate new tokens. It does this in a loop until some stop condition has been reached. While the translation example you explored here used both the encoder and decoder parts of the transformer, you can split these components apart for variations of the architecture. 

- Encoder-only models: Work as sequence-to-sequence models, suitable for classification tasks like sentiment analysis by adding additional layers (e.g., BERT).
- Encoder-decoder models: Perform well in sequence-to-sequence tasks like translation, handling different input and output sequence lengths (e.g., BART, T5).
- Decoder-only models: Commonly used and capable of generalizing to most tasks (e.g., GPT, BLOOM, Jurassic, LLaMA).

## Prompt Engineering

Prompt engineering is essential for enhancing model performance. It involves refining the prompt language to guide the model's behavior. In-context learning is a powerful strategy where examples or additional data are included in the prompt.

## Inference Techniques

- Zero-Shot Inference: Larger models can perform well without specific examples, grasping tasks even without explicit training on them.
- One-Shot Inference: Smaller models benefit from providing a single example within the prompt.
- Few-Shot Inference: Extending one-shot inference with multiple examples can further improve the model's performance.

## Configuration Parameters

Adjusting model parameters during inference can influence output:

- max_new_tokens: Limits the number of tokens the model generates.
- temperature: Controls output randomness (higher for more creativity, lower for focus).
- Random Sampling: Introduces variability by randomly selecting words.
- Top-k and Top-p Sampling: Limits random sampling options for more sensible output.

Remember to experiment with different techniques and parameters to achieve desired model behavior.

## Training and Deployment

Start with an existing model and consider fine-tuning for specific tasks. Ensure models behave well and align with human preferences using reinforcement learning with human feedback if needed.

Evaluate model performance using metrics and benchmarks. Optimize models for deployment, and consider additional infrastructure requirements.

Note that while LLMs have grown in capabilities, they still have limitations, such as inventing information when unsure or limited complex reasoning abilities.

### Learning Objectives of each lab
#### Lab 1

[Lab 1 - Generative AI Use Case: Summarize Dialogue](https://github.com/sbouden/GenAI-with-LLMs/blob/main/summarize_dialogue_with_flanT5.ipynb)
# Generative AI Use Case: Summarize Dialogue with GenAI

- Explore how the input text affects the output of the model
- Perform prompt engineering to direct it towards the task you need
- Compare zero shot, one shot, and few shot inferences to enhance the generative output of Large Language Models


#### Lab 2
[Lab 2 - Fine-tune a generative AI model for dialogue summarization](https://github.com/sbouden/GenAI-with-LLMs/blob/main/fine_tune_gen_ai_model.ipynb)
Fine-tuning and evaluating large language models

- Describe how fine-tuning with instructions using prompt datasets can improve performance on one or more tasks
- Define catastrophic forgetting and explain techniques that can be used to overcome it
- Define the term Parameter-efficient Fine Tuning (PEFT)
- Explain how PEFT decreases computational cost and overcomes catastrophic forgetting
- Explain how fine-tuning with instructions using prompt datasets can increase LLM performance on one or more 


#### Lab 3
[Lab 3 - Fine-tune FLAN-T5 with reinforcement learning to generate more-positive summaries]()
Reinforcement learning and LLM-powered applications

- Describe how RLHF uses human feedback to improve the performance and alignment of large language models
- Explain how data gathered from human labelers is used to train a reward model for RLHF
- Define chain-of-thought prompting and describe how it can be used to improve LLMs reasoning and planning abilities
- Discuss the challenges that LLMs face with knowledge cut-offs, and explain how information retrieval and augmentation techniques can overcome these challenges