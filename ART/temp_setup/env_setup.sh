# Update packages
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip

# Install dependencies
sudo apt install -y software-properties-common

# Add deadsnakes PPA
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update

# Install Python 3.10
sudo apt install -y python3.10 python3.10-venv python3.10-dev

# Check version
python3.10 --version

# System deps (Java 17 + Python 3.10 venv + pip)
sudo apt-get update
DEBIAN_FRONTEND=noninteractive sudo apt-get install -y openjdk-17-jdk

# JAVA_HOME for current shell
export JAVA_HOME="/usr/lib/jvm/java-17-openjdk-amd64"
export PATH="$JAVA_HOME/bin:$PATH"