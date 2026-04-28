# Usage:
#   chmod +x install.sh
#   ./install.sh

set -e

if command -v python3 >/dev/null 2>&1; then
    PIP="python3 -m pip"
elif command -v pip3 >/dev/null 2>&1; then
    PIP="pip3"
elif command -v pip >/dev/null 2>&1; then
    PIP="pip"
else
    echo "No python3/pip found. Installing python3 + pip via apt..."
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip
    PIP="python3 -m pip"
fi

echo "Upgrading pip..."
$PIP install --upgrade pip

echo "Installing Python dependencies..."

$PIP install numpy
$PIP install pandas
$PIP install matplotlib
$PIP install Pillow
$PIP install requests
$PIP install beautifulsoup4
$PIP install mwparserfromhell
$PIP install torch
$PIP install torchvision
$PIP install tensorflow
$PIP install transformers
$PIP install datasets
$PIP install peft
$PIP install trl
$PIP install sentencepiece
$PIP install tiktoken
$PIP install git+https://github.com/openai/CLIP.git

echo ""
echo "All dependencies installed."
