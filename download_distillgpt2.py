#!/usr/bin/env python3
"""
Script to download DistilGPT-2 model from Hugging Face
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

def download_distillgpt2():
    """Download DistilGPT-2 model and tokenizer from Hugging Face"""
    
    model_name = "distilgpt2"
    cache_dir = "./models"
    
    print(f"Downloading {model_name} model and tokenizer...")
    print(f"Cache directory: {cache_dir}")
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # Download tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        # Download model
        print("Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float32  # You can use torch.float16 for smaller memory usage
        )
        
        print(f"\nSuccessfully downloaded {model_name}!")
        print(f"Model saved in: {cache_dir}")
        print(f"Model parameters: {model.num_parameters():,}")
        
        # Test the model with a simple generation
        print("\nTesting model with sample text generation...")
        input_text = "The future of artificial intelligence is"
        
        # Tokenize input
        inputs = tokenizer.encode(input_text, return_tensors="pt")
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode and print generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated text: {generated_text}")
        
        return True
        
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return False

def main():
    """Main function to download and test the model"""
    success = download_distillgpt2()
    if success:
        print("\nSetup complete! You can now use the DistilGPT-2 model.")
    else:
        print("\nSetup failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 