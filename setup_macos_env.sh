#!/bin/bash

# ============================================================================
# setup_macos_env.sh - Setup Python Environment for TinyLlama Fine-tuning on macOS
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Environment name
ENV_NAME="tinyllama_env"

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if running on macOS
check_macos() {
    if [[ "$OSTYPE" != "darwin"* ]]; then
        print_error "This script is designed for macOS. Detected OS: $OSTYPE"
        print_warning "The script may not work correctly on non-macOS systems."
        read -p "Do you want to continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        print_success "Running on macOS"
        
        # Check if Apple Silicon
        if [[ $(uname -m) == "arm64" ]]; then
            print_success "Detected Apple Silicon Mac"
        else
            print_info "Detected Intel Mac"
        fi
    fi
}

# Function to check Python version
check_python() {
    print_info "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [[ $PYTHON_MAJOR -ge 3 ]] && [[ $PYTHON_MINOR -ge 9 ]]; then
            print_success "Python $PYTHON_VERSION found (meets requirement >= 3.9)"
            PYTHON_CMD="python3"
        else
            print_error "Python $PYTHON_VERSION found, but Python 3.9+ is required"
            print_info "Please install Python 3.9 or higher using: brew install python@3.9"
            exit 1
        fi
    else
        print_error "Python 3 not found"
        print_info "Please install Python using: brew install python@3.9"
        exit 1
    fi
}

# Function to create virtual environment
create_venv() {
    print_info "Creating virtual environment: $ENV_NAME"
    
    if [ -d "$ENV_NAME" ]; then
        print_warning "Virtual environment '$ENV_NAME' already exists"
        read -p "Do you want to remove it and create a new one? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$ENV_NAME"
        else
            print_info "Using existing virtual environment"
            return
        fi
    fi
    
    $PYTHON_CMD -m venv "$ENV_NAME"
    print_success "Virtual environment created successfully"
}

# Function to activate virtual environment
activate_venv() {
    print_info "Activating virtual environment..."
    source "$ENV_NAME/bin/activate"
    print_success "Virtual environment activated"
}

# Function to upgrade pip
upgrade_pip() {
    print_info "Upgrading pip..."
    pip install --upgrade pip
    print_success "pip upgraded successfully"
}

# Function to install PyTorch with MPS support
install_pytorch() {
    print_info "Installing PyTorch with MPS (Metal Performance Shaders) support..."
    
    # Install PyTorch with MPS support
    pip install torch torchvision torchaudio
    
    # Verify MPS availability
    python -c "import torch; print('MPS available:', torch.backends.mps.is_available())" || true
    
    print_success "PyTorch installed successfully"
}

# Function to install required packages
install_packages() {
    print_info "Installing required packages..."
    
    # Core packages
    print_info "Installing transformers..."
    pip install transformers==4.36.2
    
    print_info "Installing accelerate..."
    pip install accelerate==0.22.0
    
    print_info "Installing datasets..."
    pip install datasets
    
    print_info "Installing PEFT (Parameter-Efficient Fine-Tuning)..."
    pip install peft==0.7.1
    
    print_info "Installing TRL..."
    pip install trl==0.7.10  # Use older version compatible with transformers 4.36.2
    
    print_info "Installing Jupyter and IPython kernel..."
    pip install jupyter ipykernel
    
    # Additional useful packages
    print_info "Installing additional packages..."
    pip install "numpy<2.0"
    pip install pandas matplotlib seaborn tqdm
    
    # Handle bitsandbytes
    print_info "Attempting to install bitsandbytes..."
    print_warning "Note: bitsandbytes may not be fully compatible with Apple Silicon"
    
    # Try to install bitsandbytes
    if pip install bitsandbytes 2>/dev/null; then
        print_success "bitsandbytes installed successfully"
    else
        print_warning "bitsandbytes installation failed (this is expected on Apple Silicon)"
        print_info "The notebook should still work without bitsandbytes"
        print_info "Quantization features may be limited"
    fi
    
    print_success "All compatible packages installed successfully"
}

# Function to register Jupyter kernel
register_kernel() {
    print_info "Registering virtual environment as Jupyter kernel..."
    
    python -m ipykernel install --user --name="$ENV_NAME" --display-name="TinyLlama Fine-tuning (Python $(python --version 2>&1 | cut -d' ' -f2))"
    
    print_success "Jupyter kernel registered successfully"
}

# Function to verify installation
verify_installation() {
    print_info "Verifying installation..."
    
    echo -e "\n${BLUE}Installed packages:${NC}"
    pip list | grep -E "torch|transformers|accelerate|datasets|peft|jupyter|ipykernel" || true
    
    echo -e "\n${BLUE}Python version:${NC}"
    python --version
    
    echo -e "\n${BLUE}PyTorch MPS availability:${NC}"
    python -c "import torch; print('MPS available:', torch.backends.mps.is_available())" || print_error "Failed to check MPS availability"
}

# Function to print final instructions
print_instructions() {
    echo -e "\n${GREEN}============================================${NC}"
    echo -e "${GREEN}Environment setup completed successfully!${NC}"
    echo -e "${GREEN}============================================${NC}"
    
    echo -e "\n${BLUE}To use this environment:${NC}"
    echo -e "1. Activate the virtual environment:"
    echo -e "   ${YELLOW}source $ENV_NAME/bin/activate${NC}"
    
    echo -e "\n2. Start Jupyter Notebook:"
    echo -e "   ${YELLOW}jupyter notebook${NC}"
    
    echo -e "\n3. Open the notebook:"
    echo -e "   ${YELLOW}tinyllama_medqa_finetuning_macos.ipynb${NC}"
    
    echo -e "\n4. Select the kernel:"
    echo -e "   In Jupyter, go to Kernel > Change kernel > ${YELLOW}TinyLlama Fine-tuning${NC}"
    
    echo -e "\n${BLUE}To deactivate the environment when done:${NC}"
    echo -e "   ${YELLOW}deactivate${NC}"
    
    echo -e "\n${BLUE}Notes:${NC}"
    echo -e "- PyTorch is installed with MPS support for GPU acceleration on Apple Silicon"
    echo -e "- If bitsandbytes failed to install, quantization features may be limited"
    echo -e "- The environment is optimized for macOS and Apple Silicon"
    
    echo -e "\n${GREEN}Happy fine-tuning!${NC}\n"
}

# Main execution
main() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}TinyLlama Fine-tuning Environment Setup${NC}"
    echo -e "${BLUE}============================================${NC}\n"
    
    # Check system
    check_macos
    
    # Check Python
    check_python
    
    # Create virtual environment
    create_venv
    
    # Activate virtual environment
    activate_venv
    
    # Upgrade pip
    upgrade_pip
    
    # Install PyTorch
    install_pytorch
    
    # Install other packages
    install_packages
    
    # Register Jupyter kernel
    register_kernel
    
    # Verify installation
    verify_installation
    
    # Print instructions
    print_instructions
}

# Run main function
main