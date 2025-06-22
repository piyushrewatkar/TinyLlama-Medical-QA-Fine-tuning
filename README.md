# TinyLlama-Medical-QA-Fine-tuning

This repository contains resources for fine-tuning the TinyLlama-1.1B-Chat-v1.0 model on medical QA datasets using LoRA (Low-Rank Adaptation), specifically optimized for Apple Silicon Macs (M1/M2/M3) leveraging the MPS (Metal Performance Shaders) backend. It also includes a simple terminal-based chatbot to interact with the fine-tuned model.

## Project Structure

- `README.md`: This file.
- `setup_macos_env.sh`: A shell script to set up the Python environment, including PyTorch with MPS support, and install necessary libraries for fine-tuning and running the chatbot.
- `tinyllama_medqa_finetuning_macos_fixed.ipynb`: A Jupyter Notebook demonstrating the step-by-step process of fine-tuning the TinyLlama model on medical QA datasets.
- `tinyllama_chatbot.py`: A Python script for a terminal-based chatbot that uses the fine-tuned TinyLlama model to answer medical questions.

## Features

- **LoRA Fine-tuning**: Efficiently fine-tune TinyLlama on custom medical QA datasets.
- **Apple Silicon Optimization**: Leverages MPS for accelerated training and inference on M1/M2/M3 Macs.
- **Terminal Chatbot**: Interact with the fine-tuned model directly from your terminal for medical Q&A.
- **Environment Setup Script**: Automates the setup of the Python environment with all dependencies.

## Getting Started

Follow these steps to set up your environment, fine-tune the model, and run the chatbot.

### 1. Environment Setup

The `setup_macos_env.sh` script automates the process of setting up a Python virtual environment and installing all required dependencies, including PyTorch with MPS support.

```bash
chmod +x setup_macos_env.sh
./setup_macos_env.sh
```

This script will:
- Check for macOS and Python 3.9+.
- Create a virtual environment named `tinyllama_env`.
- Activate the environment.
- Install PyTorch with MPS support.
- Install `transformers`, `accelerate`, `datasets`, `peft`, `trl`, `jupyter`, `ipykernel`, `numpy`, `pandas`, `matplotlib`, `seaborn`, and `tqdm`.
- Attempt to install `bitsandbytes` (note: it might not be fully compatible with Apple Silicon, but the notebook should still work).
- Register the virtual environment as a Jupyter kernel.

### 2. Activate the Virtual Environment

After running the setup script, activate the virtual environment:

```bash
source tinyllama_env/bin/activate
```

### 3. Fine-tuning the Model

The fine-tuning process is detailed in the Jupyter Notebook.

- Start Jupyter Notebook:
  ```bash
  jupyter notebook
  ```
- Open `tinyllama_medqa_finetuning_macos_fixed.ipynb` in your browser.
- In Jupyter, go to `Kernel > Change kernel` and select `TinyLlama Fine-tuning (Python X.X.X)`.
- Run through the cells in the notebook to fine-tune the TinyLlama model. The fine-tuned LoRA adapters will be saved to a directory (default: `tinyllama-medical-qa-lora-adapters`).

### 4. Running the Chatbot

Once you have fine-tuned the model and the LoRA adapters are saved, you can run the terminal-based chatbot.

- Ensure your virtual environment is activated:
  ```bash
  source tinyllama_env/bin/activate
  ```
- Run the chatbot script:
  ```bash
  python tinyllama_chatbot.py
  ```

The chatbot will load the base TinyLlama model and apply your fine-tuned LoRA adapters. You can then type your medical questions and get responses.

**Chatbot Commands:**
- Type your question and press Enter.
- Type `clear` to clear conversation history.
- Type `history` to view conversation history.
- Type `quit` or `exit` to end the chat.
- Press `Ctrl+C` at any time to exit.

## Requirements

- macOS (Apple Silicon recommended for MPS acceleration)
- Python 3.9+
- `pip`

## Acknowledgements

- **TinyLlama**: For the efficient and powerful base model.
- **Hugging Face Transformers, PEFT, TRL**: For the libraries that enable easy fine-tuning and model interaction.
- **PyTorch**: For the deep learning framework with MPS support.

## License

[Specify your project's license here, e.g., MIT, Apache 2.0, etc.]
