#!/usr/bin/env python3
"""
TinyLlama Medical QA Chatbot
A terminal-based chatbot using the fine-tuned TinyLlama model for medical Q&A.
"""

import os
import sys
import signal
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings("ignore")


class TinyLlamaChatbot:
    """Terminal-based chatbot using fine-tuned TinyLlama model."""
    
    def __init__(self, adapter_path: str = "tinyllama-medical-qa-lora-adapters"):
        """
        Initialize the chatbot with the fine-tuned model.
        
        Args:
            adapter_path: Path to the LoRA adapter directory
        """
        self.adapter_path = adapter_path
        self.base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history_length = 10  # Keep last 10 exchanges
        
    def load_model(self):
        """Load the base model and apply LoRA adapters."""
        print(f"\n🤖 Loading TinyLlama model...")
        print(f"   Device: {self.device}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True
            )
            
            # Set padding token if not already set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            print("   Loading base model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
                trust_remote_code=True
            )
            
            # Load LoRA adapters
            print(f"   Loading LoRA adapters from {self.adapter_path}...")
            self.model = PeftModel.from_pretrained(
                base_model,
                self.adapter_path,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            )
            
            # Move to device if needed
            if self.device.type != "cuda":
                self.model = self.model.to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            print("✅ Model loaded successfully!\n")
            
        except Exception as e:
            print(f"\n❌ Error loading model: {str(e)}")
            print("\nPlease ensure:")
            print("1. The adapter directory exists and contains the necessary files")
            print("2. You have the required packages installed")
            print("3. You have sufficient memory available")
            sys.exit(1)
    
    def format_prompt(self, user_input: str, include_history: bool = True) -> str:
        """
        Format the prompt with conversation history for the TinyLlama chat format.
        
        Args:
            user_input: The user's current input
            include_history: Whether to include conversation history
            
        Returns:
            Formatted prompt string
        """
        messages = []
        
        # Add system message
        system_msg = "You are a helpful medical AI assistant. Provide accurate, informative responses about medical topics while being clear that you cannot replace professional medical advice."
        messages.append(f"<|system|>\n{system_msg}</s>")
        
        # Add conversation history if requested
        if include_history and self.conversation_history:
            # Use only recent history to avoid context length issues
            recent_history = self.conversation_history[-self.max_history_length:]
            for exchange in recent_history:
                messages.append(f"<|user|>\n{exchange['user']}</s>")
                messages.append(f"<|assistant|>\n{exchange['assistant']}</s>")
        
        # Add current user input
        messages.append(f"<|user|>\n{user_input}</s>")
        messages.append("<|assistant|>\n")
        
        return "\n".join(messages)
    
    def generate_response(self, user_input: str, max_length: int = 512) -> str:
        """
        Generate a response from the model.
        
        Args:
            user_input: The user's input text
            max_length: Maximum length of the generated response
            
        Returns:
            Generated response text
        """
        try:
            # Format prompt with history
            prompt = self.format_prompt(user_input)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            if "<|assistant|>" in response:
                response = response.split("<|assistant|>")[-1].strip()
            
            # Clean up any remaining tokens
            response = response.replace("</s>", "").strip()
            
            return response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def add_to_history(self, user_input: str, assistant_response: str):
        """Add an exchange to the conversation history."""
        self.conversation_history.append({
            "user": user_input,
            "assistant": assistant_response
        })
        
        # Keep history size manageable
        if len(self.conversation_history) > self.max_history_length * 2:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
        print("\n🗑️  Conversation history cleared.\n")
    
    def print_welcome(self):
        """Print welcome message and instructions."""
        print("\n" + "="*60)
        print("🏥 TinyLlama Medical QA Chatbot")
        print("="*60)
        print("\nWelcome! I'm a medical AI assistant powered by fine-tuned TinyLlama.")
        print("I can help answer medical questions, but remember:")
        print("⚠️  I cannot replace professional medical advice.")
        print("\nCommands:")
        print("  • Type your question and press Enter")
        print("  • Type 'clear' to clear conversation history")
        print("  • Type 'history' to view conversation history")
        print("  • Type 'quit' or 'exit' to end the chat")
        print("  • Press Ctrl+C at any time to exit")
        print("\n" + "-"*60 + "\n")
    
    def print_history(self):
        """Print the conversation history."""
        if not self.conversation_history:
            print("\n📝 No conversation history yet.\n")
            return
        
        print("\n" + "="*60)
        print("📝 Conversation History")
        print("="*60)
        
        for i, exchange in enumerate(self.conversation_history, 1):
            print(f"\n[Exchange {i}]")
            print(f"User: {exchange['user']}")
            print(f"Assistant: {exchange['assistant']}")
        
        print("\n" + "-"*60 + "\n")
    
    def run(self):
        """Run the interactive chatbot loop."""
        # Set up signal handler for graceful exit
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Load model
        self.load_model()
        
        # Print welcome message
        self.print_welcome()
        
        # Main chat loop
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                # Check for commands
                if user_input.lower() in ['quit', 'exit']:
                    print("\n👋 Thank you for using TinyLlama Medical QA Chatbot. Goodbye!\n")
                    break
                
                elif user_input.lower() == 'clear':
                    self.clear_history()
                    continue
                
                elif user_input.lower() == 'history':
                    self.print_history()
                    continue
                
                elif not user_input:
                    continue
                
                # Generate and display response
                print("\n🤔 Thinking...", end='', flush=True)
                response = self.generate_response(user_input)
                print("\r" + " "*20 + "\r", end='')  # Clear "Thinking..." message
                
                print(f"Assistant: {response}\n")
                
                # Add to history
                self.add_to_history(user_input, response)
                
            except KeyboardInterrupt:
                self._signal_handler(None, None)
            except Exception as e:
                print(f"\n❌ Error: {str(e)}\n")
                continue
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print("\n\n👋 Interrupted. Thank you for using TinyLlama Medical QA Chatbot. Goodbye!\n")
        sys.exit(0)


def main():
    """Main entry point."""
    # Check if adapter directory exists
    adapter_path = "tinyllama-medical-qa-lora-adapters"
    if not os.path.exists(adapter_path):
        print(f"\n❌ Error: Adapter directory '{adapter_path}' not found!")
        print("Please ensure you have run the fine-tuning notebook first.\n")
        sys.exit(1)
    
    # Create and run chatbot
    chatbot = TinyLlamaChatbot(adapter_path=adapter_path)
    chatbot.run()


if __name__ == "__main__":
    main()